# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype translation of ANTs' workflows."""
import numpy as np
# general purpose
from pkg_resources import resource_filename as pkgr_fn

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants.utils import AI

# niworkflows
from niworkflows.interfaces.ants import ImageMath
from niworkflows.interfaces.images import RegridToZooms
from niworkflows.interfaces.nibabel import ApplyMask, Binarize
from niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)
from niworkflows.interfaces.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter
)

from templateflow.api import get as get_template
from ..utils.filtering import (
    gaussian_filter as _gauss_filter,
    truncation as _trunc
)

LOWRES_ZOOMS = (0.42, 0.42, 0.42)
HIRES_ZOOMS = (0.1, 0.1, 0.1)


def init_rodent_brain_extraction_wf(
    atropos_model=None,
    atropos_refine=True,
    atropos_use_random_seed=True,
    bids_suffix="T2w",
    bspline_fitting_distance=8,
    debug=False,
    final_normalization_quality="precise",
    in_template="WHS",
    init_normalization_quality="3stage",
    mem_gb=3.0,
    name="rodent_brain_extraction_wf",
    omp_nthreads=None,
    template_spec=None,
    use_float=True,
    interim_checkpoints=True,
):
    """
    Build an atlas-based brain extraction pipeline for rodent T1w and T2w MRI data.

    Parameters
    ----------
    atropos_refine : :obj:`bool`, optional
        Run an extra step to refine the brain mask using a brain-tissue segmentation with Atropos.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_files", "in_mask"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_corrected", "out_brain", "out_mask"]),
        name="outputnode"
    )

    # Find a suitable target template in TemplateFlow
    tpl_target_path = get_template(
        in_template,
        resolution=2,
        suffix="T2star" if bids_suffix == "T2w" else "T1w",
    )
    if not tpl_target_path:
        raise RuntimeError(
            f"An instance of template <tpl-{in_template}> with MR scheme '{bids_suffix}'"
            " could not be found.")

    tpl_brainmask_path = get_template(
        in_template, resolution=2, atlas=None, desc="brain", suffix="probseg",
    )
    if not tpl_brainmask_path:
        tpl_brainmask_path = get_template(
            in_template, resolution=2, atlas=None, desc="brain", suffix="mask",
        )

    # Resample both target and template to a controlled, isotropic resolution
    res_tmpl = pe.Node(RegridToZooms(zooms=HIRES_ZOOMS), name="res_tmpl")
    res_target = pe.Node(RegridToZooms(zooms=HIRES_ZOOMS), name="res_target")
    lowres_tmpl = pe.Node(RegridToZooms(zooms=LOWRES_ZOOMS), name="lowres_tmpl")
    lowres_target = pe.Node(RegridToZooms(zooms=LOWRES_ZOOMS), name="lowres_target")
    gauss_tmpl = pe.Node(niu.Function(function=_gauss_filter), name="gauss_tmpl")
    gauss_tmpl.inputs.sigma = tuple(np.array(LOWRES_ZOOMS) * 8.0)

    # Initialize transforms with antsAI
    init_aff = pe.Node(
        AI(
            metric=("Mattes", 32, "Regular", 1.0),
            transform=("Affine", 0.1),
            search_factor=(10, 0.08),
            principal_axes=False,
            convergence=(40, 1e-6, 10),
            search_grid=(25, (0, 0, 0)),
            verbose=True,
        ),
        name="init_aff",
        n_procs=omp_nthreads,
    )

    # main workflow
    wf = pe.Workflow(name)
    # Create a buffer interface as a cache for the actual inputs to registration
    buffernode = pe.Node(niu.IdentityInterface(
        fields=["lowres_target", "hires_target"]), name="buffernode")

    if bids_suffix.lower() == "t2w":
        mask_tmpl = pe.Node(
            ApplyMask(
                in_file=_pop(tpl_target_path),
                in_mask=_pop(tpl_brainmask_path),
            ),
            name="mask_tmpl",
        )
        # truncate target intensity for N4 correction
        clip_target = pe.Node(
            niu.Function(function=_trunc),
            name="clip_target",
        )
        clip_tmpl = pe.Node(
            niu.Function(function=_trunc),
            name="clip_tmpl",
        )
        # INU correction of the target image
        init_n4 = pe.Node(
            N4BiasFieldCorrection(
                dimension=3,
                save_bias=False,
                copy_header=True,
                n_iterations=[50] * (4 - debug),
                convergence_threshold=1e-7,
                shrink_factor=4,
                bspline_fitting_distance=bspline_fitting_distance,
            ),
            n_procs=omp_nthreads,
            name="init_n4",
        )
        clip_inu = pe.Node(
            niu.Function(function=_trunc),
            name="clip_inu",
        )
        gauss_target = pe.Node(niu.Function(function=_gauss_filter), name="gauss_target")
        gauss_target.inputs.sigma = tuple(np.array(LOWRES_ZOOMS) * 2.0)
        wf.connect([
            # truncation, resampling, and initial N4
            (inputnode, res_target, [(("in_files", _pop), "in_file")]),
            (mask_tmpl, clip_tmpl, [("out_file", "in_file")]),
            (res_target, clip_target, [("out_file", "in_file")]),
            (clip_tmpl, res_tmpl, [("out", "in_file")]),
            (clip_target, init_n4, [("out", "input_image")]),
            (init_n4, clip_inu, [("output_image", "in_file")]),
            (clip_inu, gauss_target, [("out", "in_file")]),
            (gauss_target, lowres_target, [("out", "in_file")]),
            (lowres_target, buffernode, [("out_file", "lowres_target")]),
            (clip_inu, buffernode, [("out", "hires_target")]),
        ])
    elif bids_suffix == "t1w":
        wf.connect([
            # resampling and laplacian; no truncation or N4
            (inputnode, res_target, [("in_files", "in_file")]),
            (res_target, buffernode, [("out_file", "lowres_target")]),
        ])

    wf.connect([
        # ants AI inputs
        (buffernode, init_aff, [("lowres_target", "moving_image")]),
        (res_tmpl, gauss_tmpl, [("out_file", "in_file")]),
        (gauss_tmpl, lowres_tmpl, [("out", "in_file")]),
        (lowres_tmpl, init_aff, [("out_file", "fixed_image")]),
    ])

    # Graft a template registration-mask if present
    tpl_regmask_path = get_template(
        in_template, resolution=2, atlas=None, desc="BrainCerebellumExtraction", suffix="mask",
    )
    if tpl_regmask_path:
        lowres_mask = pe.Node(
            ApplyTransforms(
                input_image=_pop(tpl_regmask_path),
                transforms="identity",
                interpolation="MultiLabel",
                float=True),
            name="lowres_mask",
            mem_gb=1
        )

        hires_mask = pe.Node(
            ApplyTransforms(
                input_image=_pop(tpl_regmask_path),
                transforms="identity",
                interpolation="NearestNeighbor",
                float=True),
            name="hires_mask",
            mem_gb=1
        )
        wf.connect([
            (lowres_tmpl, lowres_mask, [("out_file", "reference_image")]),
            (lowres_mask, init_aff, [("output_image", "fixed_image_mask")]),
            (res_tmpl, hires_mask, [("out_file", "reference_image")]),
        ])

    # Spatial normalization step
    lap_tmpl = pe.Node(ImageMath(operation="Laplacian", op2="0.4 1"), name="lap_tmpl")
    lap_target = pe.Node(ImageMath(operation="Laplacian", op2="0.4 1"), name="lap_target")

    # Merge image nodes
    mrg_target = pe.Node(niu.Merge(2), name="mrg_target")
    mrg_tmpl = pe.Node(niu.Merge(2), name="mrg_tmpl")

    # norm_lap_tmpl = pe.Node(ImageMath(operation="Normalize"), name="norm_lap_tmpl")
    # norm_lap_target = pe.Node(ImageMath(operation="Normalize"), name="norm_lap_target")

    # Set up initial spatial normalization
    ants_params = "testing" if debug else "precise"
    norm = pe.Node(
        Registration(from_file=pkgr_fn(
            "nirodents",
            f"data/artsBrainExtraction_{ants_params}_{bids_suffix}.json")
        ),
        name="norm",
        n_procs=omp_nthreads,
        mem_gb=mem_gb,
    )
    norm.inputs.float = use_float

    map_brainmask = pe.Node(
        ApplyTransforms(interpolation="Gaussian", float=True),
        name="map_brainmask",
        mem_gb=1
    )
    map_brainmask.inputs.input_image = str(tpl_brainmask_path)

    thr_brainmask = pe.Node(Binarize(thresh_low=0.5),
                            name="thr_brainmask")

    # Refine INU correction
    final_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            bspline_fitting_distance=bspline_fitting_distance,
            rescale_intensities=True,
            shrink_factor=4,
        ),
        n_procs=omp_nthreads,
        name="final_n4",
    )
    final_mask = pe.Node(ApplyMask(), name="final_mask")

    wf.connect([
        (inputnode, map_brainmask, [(("in_files", _pop), "reference_image")]),
        (inputnode, final_n4, [(("in_files", _pop), "input_image")]),
        # merge laplacian and original images
        (buffernode, lap_target, [("hires_target", "op1")]),
        (buffernode, mrg_target, [("hires_target", "in1")]),
        (lap_target, mrg_target, [("output_image", "in2")]),
        # Template massaging
        (res_tmpl, lap_tmpl, [("out_file", "op1")]),
        (res_tmpl, mrg_tmpl, [("out_file", "in1")]),
        (lap_tmpl, mrg_tmpl, [("output_image", "in2")]),
        # spatial normalization
        (init_aff, norm, [("output_transform", "initial_moving_transform")]),
        (mrg_target, norm, [("out", "moving_image")]),
        (mrg_tmpl, norm, [("out", "fixed_image")]),
        (norm, map_brainmask, [
            ("reverse_transforms", "transforms"),
            ("reverse_invert_flags", "invert_transform_flags")]),
        (map_brainmask, thr_brainmask, [("output_image", "in_file")]),
        # take a second pass of N4
        (map_brainmask, final_n4, [("output_image", "weight_image")]),
        (final_n4, final_mask, [("output_image", "in_file")]),
        (thr_brainmask, final_mask, [("out_file", "in_mask")]),
        (final_n4, outputnode, [("output_image", "out_corrected")]),
        (thr_brainmask, outputnode, [("out_file", "out_mask")]),
        (final_mask, outputnode, [("out_file", "out_brain")]),
    ])

    if tpl_regmask_path:
        wf.connect([
            (hires_mask, norm, [
                ("output_image", "fixed_image_masks")]),
        ])

    if interim_checkpoints:
        init_apply = pe.Node(
            ApplyTransforms(
                interpolation="BSpline",
                float=True),
            name="init_apply",
            mem_gb=1
        )
        init_report = pe.Node(SimpleBeforeAfter(
            before_label="tpl-WHS",
            after_label="target"),
            name="init_report"
        )
        final_apply = pe.Node(
            ApplyTransforms(
                interpolation="BSpline",
                float=True),
            name="final_apply",
            mem_gb=1
        )
        final_report = pe.Node(SimpleBeforeAfter(
            before_label="tpl-WHS",
            after_label="target"),
            name="final_report"
        )
        wf.connect([
            (buffernode, init_apply, [("lowres_target", "input_image")]),
            (res_tmpl, init_apply, [("out_file", "reference_image")]),
            (init_aff, init_apply, [("output_transform", "transforms")]),
            (init_apply, init_report, [("output_image", "after")]),
            (res_tmpl, init_report, [("out_file", "before")]),

            (inputnode, final_apply, [(("in_files", _pop), "reference_image")]),
            (res_tmpl, final_apply, [("out_file", "input_image")]),
            (norm, final_apply, [
                ("reverse_transforms", "transforms"),
                ("reverse_invert_flags", "invert_transform_flags")]),
            (final_apply, final_report, [("output_image", "before")]),
            (outputnode, final_report, [("out_corrected", "after"),
                                        ("out_mask", "wm_seg")]),
        ])

    return wf


def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
