# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype translation of ANTs' workflows."""
# general purpose
from pkg_resources import resource_filename as pkgr_fn

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import (
    AI,
    ImageMath,
    N4BiasFieldCorrection,
)

# niworkflows
from niworkflows.interfaces.bids import DerivativesDataSink as _DDS
from niworkflows.interfaces.images import RegridToZooms
from niworkflows.interfaces.nibabel import ApplyMask, Binarize
from niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from niworkflows.interfaces.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)

from templateflow.api import get as get_template
from ..utils.filtering import truncation as _trunc
from ..interfaces import DenoiseImage

from .. import __version__


class DerivativesDataSink(_DDS):
    """Generate a BIDS-Derivatives-compatible output folder."""

    out_path_base = f"nirodents-{__version__}"


LOWRES_ZOOMS = (0.4, 0.4, 0.4)
HIRES_ZOOMS = (0.1, 0.1, 0.1)


def init_rodent_brain_extraction_wf(
    ants_affine_init=False,
    factor=20,
    arc=0.12,
    step=4,
    grid=(0, 4, 4),
    slice_direction=1,
    debug=False,
    interim_checkpoints=True,
    mem_gb=3.0,
    mri_scheme="T2w",
    name="rodent_brain_extraction_wf",
    omp_nthreads=None,
    output_dir=None,
    template_id="Fischer344",
    template_specs=None,
    use_float=True,
):
    """
    Build an atlas-based brain extraction pipeline for rodent T1w and T2w MRI data.

    Parameters
    ----------
    ants_affine_init : :obj:`bool`, optional
        Set-up a pre-initialization step with ``antsAI`` to account for mis-oriented images.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_files", "in_mask"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_corrected", "out_brain", "out_mask"]),
        name="outputnode",
    )

    template_specs = template_specs or {}
    if template_id == "WHS" and "resolution" not in template_specs:
        template_specs["resolution"] = 2

    # Find a suitable target template in TemplateFlow
    tpl_target_path = get_template(template_id, suffix=mri_scheme, **template_specs,)
    if not tpl_target_path:
        raise RuntimeError(
            f"An instance of template <tpl-{template_id}> with MR scheme '{mri_scheme}'"
            " could not be found."
        )

    tpl_brainmask_path = get_template(
        template_id,
        atlas=None,
        hemi=None,
        desc="brain",
        suffix="probseg",
        **template_specs,
    ) or get_template(
        template_id,
        atlas=None,
        hemi=None,
        desc="brain",
        suffix="mask",
        **template_specs,
    )

    tpl_regmask_path = get_template(
        template_id,
        atlas=None,
        desc="BrainCerebellumExtraction",
        suffix="mask",
        **template_specs,
    )

    denoise = pe.Node(DenoiseImage(dimension=3, copy_header=True),
                      name="denoise", n_procs=omp_nthreads)

    # Resample template to a controlled, isotropic resolution
    res_tmpl = pe.Node(RegridToZooms(zooms=HIRES_ZOOMS, smooth=True), name="res_tmpl")

    # Create Laplacian images
    tmpl_sigma = pe.Node(niu.Function(function=_lap_sigma),
                         name="tmpl_sigma", run_without_submitting=True)
    lap_tmpl = pe.Node(
        ImageMath(operation="Laplacian", copy_header=True), name="lap_tmpl"
    )
    norm_lap_tmpl = pe.Node(niu.Function(function=_trunc), name="norm_lap_tmpl")
    norm_lap_tmpl.inputs.out_max = 1.0
    norm_lap_tmpl.inputs.percentiles = (1, 99.99)
    norm_lap_tmpl.inputs.clip_max = None
    target_sigma = pe.Node(niu.Function(function=_lap_sigma),
                           name="target_sigma", run_without_submitting=True)
    lap_target = pe.Node(
        ImageMath(operation="Laplacian", copy_header=True), name="lap_target"
    )
    norm_lap_target = pe.Node(niu.Function(function=_trunc), name="norm_lap_target")
    norm_lap_target.inputs.out_max = 1.0
    norm_lap_target.inputs.percentiles = (1, 99.99)
    norm_lap_target.inputs.clip_max = None

    # Set up initial spatial normalization
    ants_params = "testing" if debug else "precise"
    norm = pe.Node(
        Registration(
            from_file=pkgr_fn(
                "nirodents", f"data/artsBrainExtraction_{ants_params}_{mri_scheme}.json"
            )
        ),
        name="norm",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )
    norm.inputs.float = use_float

    # main workflow
    wf = pe.Workflow(name)

    # truncate target intensity for N4 correction
    clip_target = pe.Node(niu.Function(function=_trunc), name="clip_target")
    clip_target.inputs.percentiles = (None, 99.9)
    clip_target.inputs.clip_max = None

    # truncate template intensity to match target
    clip_tmpl = pe.Node(niu.Function(function=_trunc), name="clip_tmpl")
    clip_tmpl.inputs.in_file = _pop(tpl_target_path)
    clip_tmpl.inputs.percentiles = (35.0, 90.0)

    # set INU bspline grid based on voxel size
    init_bspline_grid = pe.Node(niu.Function(function=_bspline_distance), name="init_bspline_grid")
    init_bspline_grid.inputs.slice_dir = slice_direction

    # INU correction of the target image
    init_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=False,
            copy_header=True,
            n_iterations=[50] * (4 - debug),
            convergence_threshold=1e-7,
            shrink_factor=4,
            rescale_intensities=True,
        ),
        n_procs=omp_nthreads,
        name="init_n4",
    )
    clip_inu = pe.Node(niu.Function(function=_trunc), name="clip_inu")
    clip_inu.inputs.percentiles = (1., 99.8)

    # Create a buffer interface as a cache for the actual inputs to registration
    buffernode = pe.Node(
        niu.IdentityInterface(fields=["hires_target"]), name="buffernode"
    )

    # Merge image nodes
    mrg_target = pe.Node(niu.Merge(2), name="mrg_target")
    mrg_tmpl = pe.Node(niu.Merge(2), name="mrg_tmpl")

    # fmt: off
    wf.connect([
        # Target image massaging
        (inputnode, clip_target, [(("in_files", _pop), "in_file")]),
        (inputnode, init_bspline_grid, [(("in_files", _pop), "in_file")]),
        (init_bspline_grid, init_n4, [("out", "args")]),
        (clip_target, denoise, [("out", "input_image")]),
        (denoise, init_n4, [("output_image", "input_image")]),
        (init_n4, clip_inu, [("output_image", "in_file")]),
        (clip_inu, target_sigma, [("out", "in_file")]),
        (clip_inu, buffernode, [("out", "hires_target")]),
        (buffernode, lap_target, [("hires_target", "op1")]),
        (target_sigma, lap_target, [("out", "op2")]),
        (lap_target, norm_lap_target, [("output_image", "in_file")]),
        (buffernode, mrg_target, [("hires_target", "in1")]),
        (norm_lap_target, mrg_target, [("out", "in2")]),
        # Template massaging
        (clip_tmpl, res_tmpl, [("out", "in_file")]),
        (res_tmpl, tmpl_sigma, [("out_file", "in_file")]),
        (res_tmpl, lap_tmpl, [("out_file", "op1")]),
        (tmpl_sigma, lap_tmpl, [("out", "op2")]),
        (lap_tmpl, norm_lap_tmpl, [("output_image", "in_file")]),
        (res_tmpl, mrg_tmpl, [("out_file", "in1")]),
        (norm_lap_tmpl, mrg_tmpl, [("out", "in2")]),
        # Setup inputs to spatial normalization
        (mrg_target, norm, [("out", "moving_image")]),
        (mrg_tmpl, norm, [("out", "fixed_image")]),
    ])
    # fmt: on

    # Graft a template registration-mask if present
    if tpl_regmask_path:
        hires_mask = pe.Node(
            ApplyTransforms(
                input_image=_pop(tpl_regmask_path),
                transforms="identity",
                interpolation="Gaussian",
                float=True,
            ),
            name="hires_mask",
            mem_gb=1,
        )

        # fmt: off
        wf.connect([
            (res_tmpl, hires_mask, [("out_file", "reference_image")]),
            (hires_mask, norm, [("output_image", "fixed_image_masks")]),
        ])
        # fmt: on

    # Finally project brain mask and refine INU correction
    map_brainmask = pe.Node(
        ApplyTransforms(interpolation="Gaussian", float=True),
        name="map_brainmask",
        mem_gb=1,
    )
    map_brainmask.inputs.input_image = str(tpl_brainmask_path)

    thr_brainmask = pe.Node(Binarize(thresh_low=0.50), name="thr_brainmask")

    final_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 4,
            convergence_threshold=1e-7,
            rescale_intensities=True,
            shrink_factor=4,
        ),
        n_procs=omp_nthreads,
        name="final_n4",
    )
    final_mask = pe.Node(ApplyMask(), name="final_mask")

    # fmt: off
    wf.connect([
        (inputnode, map_brainmask, [(("in_files", _pop), "reference_image")]),
        (denoise, final_n4, [("output_image", "input_image")]),
        # Project template's brainmask into subject space
        (norm, map_brainmask, [("reverse_transforms", "transforms"),
                               ("reverse_invert_flags", "invert_transform_flags")]),
        (map_brainmask, thr_brainmask, [("output_image", "in_file")]),
        # take a second pass of N4
        (map_brainmask, final_n4, [("output_image", "mask_image")]),
        (final_n4, final_mask, [("output_image", "in_file")]),
        (thr_brainmask, final_mask, [("out_mask", "in_mask")]),
        (final_n4, outputnode, [("output_image", "out_corrected")]),
        (thr_brainmask, outputnode, [("out_mask", "out_mask")]),
        (final_mask, outputnode, [("out_file", "out_brain")]),
    ])
    # fmt: on

    if interim_checkpoints:
        final_apply = pe.Node(
            ApplyTransforms(interpolation="BSpline", float=True),
            name="final_apply",
            mem_gb=1,
        )
        final_report = pe.Node(
            SimpleBeforeAfter(after_label="target", before_label=f"tpl-{template_id}"),
            name="final_report",
        )
        # fmt: off
        wf.connect([
            (inputnode, final_apply, [(("in_files", _pop), "reference_image")]),
            (res_tmpl, final_apply, [("out_file", "input_image")]),
            (norm, final_apply, [("reverse_transforms", "transforms"),
                                 ("reverse_invert_flags", "invert_transform_flags")]),
            (final_apply, final_report, [("output_image", "before")]),
            (outputnode, final_report, [("out_corrected", "after"), ("out_mask", "wm_seg")]),
        ])
        # fmt: on

    if ants_affine_init:
        # Initialize transforms with antsAI
        lowres_tmpl = pe.Node(
            RegridToZooms(zooms=LOWRES_ZOOMS, smooth=True), name="lowres_tmpl"
        )
        lowres_trgt = pe.Node(
            RegridToZooms(zooms=LOWRES_ZOOMS, smooth=True), name="lowres_trgt"
        )

        init_aff = pe.Node(
            AI(
                convergence=(100, 1e-6, 10),
                metric=("Mattes", 32, "Random", 0.25),
                principal_axes=False,
                search_factor=(factor, arc),
                search_grid=(step, grid),
                transform=("Affine", 0.1),
                verbose=True,
            ),
            name="init_aff",
            n_procs=omp_nthreads,
        )
        # fmt: off
        wf.connect([
            (clip_inu, lowres_trgt, [("out", "in_file")]),
            (lowres_trgt, init_aff, [("out_file", "moving_image")]),
            (clip_tmpl, lowres_tmpl, [("out", "in_file")]),
            (lowres_tmpl, init_aff, [("out_file", "fixed_image")]),
            (init_aff, norm, [("output_transform", "initial_moving_transform")]),
        ])
        # fmt: on

        if tpl_regmask_path:
            lowres_mask = pe.Node(
                ApplyTransforms(
                    input_image=_pop(tpl_regmask_path),
                    transforms="identity",
                    interpolation="MultiLabel",
                ),
                name="lowres_mask",
                mem_gb=1,
            )
            # fmt: off
            wf.connect([
                (lowres_tmpl, lowres_mask, [("out_file", "reference_image")]),
                (lowres_mask, init_aff, [("output_image", "fixed_image_mask")]),
            ])
            # fmt: on

        if interim_checkpoints:
            init_apply = pe.Node(
                ApplyTransforms(interpolation="BSpline", invert_transform_flags=[True]),
                name="init_apply",
                mem_gb=1,
            )
            init_mask = pe.Node(
                ApplyTransforms(interpolation="Gaussian", invert_transform_flags=[True]),
                name="init_mask",
                mem_gb=1,
            )
            init_mask.inputs.input_image = str(tpl_brainmask_path)
            init_report = pe.Node(
                SimpleBeforeAfter(
                    out_report="init_report.svg",
                    before_label="target",
                    after_label="template",
                ),
                name="init_report",
            )
            # fmt: off
            wf.connect([
                (lowres_trgt, init_apply, [("out_file", "reference_image")]),
                (lowres_tmpl, init_apply, [("out_file", "input_image")]),
                (init_aff, init_apply, [("output_transform", "transforms")]),
                (lowres_trgt, init_report, [("out_file", "before")]),
                (init_apply, init_report, [("output_image", "after")]),

                (lowres_trgt, init_mask, [("out_file", "reference_image")]),
                (init_aff, init_mask, [("output_transform", "transforms")]),
                (init_mask, init_report, [("output_image", "wm_seg")]),
            ])
            # fmt: on
    else:
        norm.inputs.initial_moving_transform_com = 1

    if output_dir:
        ds_final_inu = pe.Node(
            DerivativesDataSink(
                base_directory=str(output_dir), desc="preproc", compress=True,
            ), name="ds_final_inu", run_without_submitting=True
        )
        ds_final_msk = pe.Node(
            DerivativesDataSink(
                base_directory=str(output_dir), desc="brain", suffix="mask", compress=True,
            ), name="ds_final_msk", run_without_submitting=True
        )

        # fmt: off
        wf.connect([
            (inputnode, ds_final_inu, [("in_files", "source_file")]),
            (inputnode, ds_final_msk, [("in_files", "source_file")]),
            (outputnode, ds_final_inu, [("out_corrected", "in_file")]),
            (outputnode, ds_final_msk, [("out_mask", "in_file")]),
        ])
        # fmt: on

        if interim_checkpoints:
            ds_report = pe.Node(
                DerivativesDataSink(
                    base_directory=str(output_dir), desc="brain",
                    suffix="mask", datatype="figures"
                ), name="ds_report", run_without_submitting=True
            )
            # fmt: off
            wf.connect([
                (inputnode, ds_report, [("in_files", "source_file")]),
                (final_report, ds_report, [("out_report", "in_file")]),
            ])
            # fmt: on

        if ants_affine_init and interim_checkpoints:
            ds_report_init = pe.Node(
                DerivativesDataSink(
                    base_directory=str(output_dir), desc="init",
                    suffix="mask", datatype="figures"
                ), name="ds_report_init", run_without_submitting=True
            )
            # fmt: off
            wf.connect([
                (inputnode, ds_report_init, [("in_files", "source_file")]),
                (init_report, ds_report_init, [("out_report", "in_file")]),
            ])
            # fmt: on

    return wf


def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files


def _bspline_distance(in_file, spacings=(8, 10, 8), slice_dir=1):
    import numpy as np
    import nibabel as nb

    img = nb.load(in_file)
    zooms = img.header.get_zooms()[:3]
    zooms_round = [round(x, 3) for x in zooms]
    extent = (np.array(img.shape[:3]) - 1) * zooms
    if zooms_round.count(zooms_round[0]) != 3 and np.argmax(zooms) != slice_dir:
        extent[np.argmax(zooms)], extent[slice_dir] = extent[slice_dir], extent[np.argmax(zooms)]
    retval = [f"{v}" for v in np.ceil(extent / np.array(spacings)).astype(int)]
    return f"-b [{'x'.join(retval)}]"


def _lap_sigma(in_file):
    import numpy as np
    import nibabel as nb
    import math

    img = nb.load(in_file)
    min_vox = np.amin(img.header.get_zooms())
    return str(0.3508 * math.exp(1.4652 * min_vox))
