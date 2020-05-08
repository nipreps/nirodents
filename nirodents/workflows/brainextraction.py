# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype translation of ANTs' workflows."""

# general purpose
from multiprocessing import cpu_count
from pkg_resources import resource_filename as pkgr_fn

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection, Atropos
from nipype.interfaces.ants.utils import AI
from nipype.interfaces.afni import MaskTool
from nipype.interfaces.io import DataSink

# niworkflows
from niworkflows.interfaces.ants import ImageMath
from niworkflows.interfaces.images import RegridToZooms
from niworkflows.interfaces.nibabel import ApplyMask
from niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms)

from templateflow.api import get as get_template

LOWRES_ZOOMS = (0.4, 0.4, 0.4)


def init_rodent_brain_extraction_wf(
    atropos_model=None,
    atropos_refine=True,
    atropos_use_random_seed=True,
    bids_suffix='T2w',
    bspline_fitting_distance=8,  # 4
    debug=False,
    final_normalization_quality='precise',
    in_template='WHS',
    init_normalization_quality='3stage',
    mem_gb=3.0,
    name='rodent_brain_extraction_wf',
    omp_nthreads=None,
    template_spec=None,
    use_float=True,
):
    """
    Build an atlas-based brain extraction pipeline for rodent T1w and T2w MRI data.

    Parameters
    ----------
    atropos_refine : :obj:`bool`, optional
        Run an extra step to refine the brain mask using a brain-tissue segmentation with Atropos.

    """


    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files', 'in_mask']),
                        name='inputnode')

    # Find images in templateFlow
    tpl_target_path = get_template(in_template, resolution=debug + 1, suffix="T2star" if bids_suffix == "T2w" else "T1w")
    tpl_regmask_path = get_template(in_template, resolution=debug + 1, atlas='v3', desc='brain', suffix='mask')
    if tpl_regmask_path:
        inputnode.inputs.in_mask = str(tpl_regmask_path)
    tpl_tissue_labels = get_template(in_template,resolution=debug + 1, desc='cerebrum', suffix='dseg')
    tpl_brain_mask = get_template(in_template, resolution=debug + 1, desc='cerebrum', suffix='mask')

    # resample template and target
    res_tmpl = pe.Node(RegridToZooms(
        in_file=tpl_target_path, zooms=LOWRES_ZOOMS), name="res_tmpl")

    res_target = pe.Node(RegridToZooms(zooms=LOWRES_ZOOMS), name="res_target")
    res_target2 = pe.Node(RegridToZooms(zooms=LOWRES_ZOOMS), name="res_target2")

    dil_mask = pe.Node(MaskTool(
        outputtype='NIFTI_GZ', dilate_inputs='2', fill_holes=True),
        name='dil_mask')

    # truncate target intensity for N4 correction
    trunc_opts = {"T1w": "0.005 0.999 256", "T2w": "0.01 0.999 256"}
    trunc = pe.MapNode(ImageMath(operation='TruncateImageIntensity', op2=trunc_opts[bids_suffix]),
                       name='truncate_images', iterfield=['op1'])

    # Initial N4 correction
    inu_n4 = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=False, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=bspline_fitting_distance),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])


    lap_tmpl = pe.Node(ImageMath(operation='Laplacian', op2='0.4'), name='lap_tmpl')
    if tpl_target_path:
        lap_tmpl.inputs.op1 = tpl_target_path
    lap_target = pe.Node(ImageMath(operation='Laplacian', op2='0.4'), name='lap_target')

    norm_lap_tmpl = pe.Node(ImageMath(operation='Normalize'), name='norm_lap_tmpl')
    norm_lap_target = pe.Node(ImageMath(operation='Normalize'), name='norm_lap_target')

    # Merge image nodes
    mrg_target = pe.Node(niu.Merge(2), name='mrg_target')
    mrg_tmpl = pe.Node(niu.Merge(2), name='mrg_tmpl')
    # mrg_tmpl.inputs.in1 = tpl_target_path

    # Create integration nodes to allow compatibility between pipelines
    integrate_1 = pe.Node(niu.IdentityInterface(fields=["in_file"]), name='integrate_1')
    integrate_2 = pe.Node(niu.IdentityInterface(fields=["in_file"]), name='integrate_2')

    # Initialize transforms with antsAI
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.5), #0.25
        transform=('Rigid', 0.1),
        search_factor=(2, 0.015),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        search_grid=(1, (1, 2, 2)),
        verbose=True),
        name='init_aff',
        n_procs=omp_nthreads)

    # Initial warping of template mask to subject space
    warp_mask_1 = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=True), name='warp_mask_1')

    # Set up initial spatial normalization
    init_settings_file = f'data/brainextraction_{init_normalization_quality}_{bids_suffix}.json'
    init_norm = pe.Node(Registration(from_file=pkgr_fn(
        'nirodents', init_settings_file)),
        name='init_norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    init_norm.inputs.float = use_float

    # Refine INU correction
    inu_n4_final = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 5, convergence_threshold=1e-7,
            bspline_fitting_distance=bspline_fitting_distance,
            rescale_intensities=True, shrink_factor=4),
        n_procs=omp_nthreads, name='inu_n4_final', iterfield=['input_image'])

    split_init_transforms = pe.Node(niu.Split(splits=[1, 1]), name='split_init_transforms')
    mrg_init_transforms = pe.Node(niu.Merge(2), name='mrg_init_transforms')

    # Use more precise transforms to warp mask to subject space
    warp_mask_2 = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=[False, True]),
        name='warp_mask_2')

    # morphological closing of warped mask
    close_mask = pe.Node(MaskTool(outputtype='NIFTI_GZ', dilate_inputs='5 -5', fill_holes=True),
        name='close_mask')

    # Use subject-space mask to skull-strip subject
    skullstrip_tar = pe.Node(ApplyMask(), name='skullstrip_tar')
    skullstrip_tpl = pe.Node(ApplyMask(), name='skullstrip_tpl')
    if tpl_target_path:
        skullstrip_tpl.inputs.in_file = tpl_target_path

    # Normalise skull-stripped image to brain template
    final_settings_file = f'data/brainextraction_{final_normalization_quality}_{bids_suffix}.json'
    refine_norm = pe.Node(Registration(from_file=pkgr_fn(
        'nirodents', final_settings_file)),
        name='refine_norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    refine_norm.inputs.float = use_float

    split_final_transforms = pe.Node(niu.Split(splits=[1, 1]), name='split_final_transforms')
    mrg_final_transforms = pe.Node(niu.Merge(2), name='mrg_final_transforms')

    warp_mask_out = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=[False, True]),
        name='warp_mask_out')
    if tpl_brain_mask:
        warp_mask_out.inputs.input_image = tpl_brain_mask
    else:
        warp_mask_out.inputs.input_image = tpl_regmask_path

    warp_seg_labels = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=[False, True]),
        name='warp_seg_labels')
    if tpl_tissue_labels:
        warp_seg_labels.inputs.input_image = tpl_tissue_labels

    segment = pe.Node(Atropos(
        dimension=3, initialization='PriorLabelImage', number_of_tissue_classes=2,
        prior_weighting=0.03, posterior_formulation='Aristotle',
        n_iterations=50, convergence_threshold=0.0001,
        mrf_smoothing_factor=0.015, mrf_radius=[1, 1, 1]), name='segment')

    sinker = pe.Node(DataSink(), name='sinker')

    #workflow definitions
    #target image specific workflows
    tar_prep = pe.Workflow('tar_prep')
    if bids_suffix.lower() == 't2w':
        tar_prep.connect([
            # truncation, resampling, and initial N4
            (inputnode, trunc, [('in_files', 'op1')]),
            (trunc, res_target, [(('output_image', _pop), 'in_file')]),
            (res_target, inu_n4, [('out_file', 'input_image')]),
            (inu_n4, integrate_1, [(('output_image', _pop), 'in_file')]),

            # masked N4 correction
            (trunc, inu_n4_final, [(('output_image', _pop), 'input_image')]),
            (inu_n4_final, integrate_2, [(('output_image', _pop), 'in_file')]),

            # merge laplacian and original images
            (inu_n4_final, lap_target, [(('output_image', _pop), 'op1')]),
            (lap_target, norm_lap_target, [('output_image', 'op1')]),
            (norm_lap_target, mrg_target, [('output_image', 'in2')]),
            (inu_n4_final, res_target2, [(('output_image', _pop), 'in_file')]),
            (res_target2, mrg_target, [('out_file', 'in1')]),
        ])
    elif bids_suffix == 't1w':
        tar_prep.connect([
            # resampling and laplacian; no truncation or N4
            (inputnode, res_target, [('in_files', 'in_file')]),
            (inputnode, lap_target, [('in_files', 'op1')]),
            (lap_target, norm_lap_target, [('output_image', 'op1')]),
            (norm_lap_target, mrg_target, [('output_image', 'in2')]),
            (res_target, mrg_target, [('out_file', 'in1')]),
            (res_target, integrate_1, [('out_file', 'in_file')]),
            (inputnode, integrate_2, [('in_files', 'in_file')])
        ])

    #main workflow
    wf = pe.Workflow(name)
    wf.connect([
        # template prep: dilation of input mask, resampling template, laplacian creation
        (inputnode, dil_mask, [('in_mask', 'in_file')]),
        (res_tmpl, mrg_tmpl, [('out_file', 'in1')]),
        (lap_tmpl, norm_lap_tmpl, [('output_image', 'op1')]),
        (norm_lap_tmpl, mrg_tmpl, [('output_image', 'in2')]),

        # ants AI inputs
        (tar_prep, init_aff, [('integrate_1.out_file', 'moving_image')]),
        (dil_mask, init_aff, [('out_file', 'fixed_image_mask')]),
        (res_tmpl, init_aff, [('out_file', 'fixed_image')]),

        # warp mask to individual space
        (dil_mask, warp_mask_1, [('out_file', 'input_image')]),
        (init_aff, warp_mask_1, [('output_transform', 'transforms')]),
        (inputnode, warp_mask_1, [('in_files', 'reference_image')]),

        # normalisation inputs
        (init_aff, init_norm, [('output_transform', 'initial_moving_transform')]),
        (warp_mask_1, init_norm, [('output_image', 'moving_image_masks')]),
        (dil_mask, init_norm, [('out_file', 'fixed_image_masks')]),
        (mrg_tmpl, init_norm, [('out', 'fixed_image')]),
        (tar_prep, init_norm, [('mrg_target.out', 'moving_image')]),

        #organise initial normalisation transforms for warps
        (init_norm, split_init_transforms, [('reverse_transforms', 'inlist')]),
        (split_init_transforms, mrg_init_transforms, [('out2', 'in1')]),
        (split_init_transforms, mrg_init_transforms, [('out1', 'in2')]),

        # warp mask with initial normalisation transforms
        (tar_prep, warp_mask_2, [('integrate_2.out_file', 'reference_image')]),
        (dil_mask, warp_mask_2, [('out_file', 'input_image')]),
        (mrg_init_transforms, warp_mask_2, [('out', 'transforms')]),
        (warp_mask_2, close_mask, [('output_image', 'in_file')]),

        # mask brains for refined normalisation
        (tar_prep, skullstrip_tar, [('integrate_2.out_file', 'in_file')]),
        (close_mask, skullstrip_tar, [('out_file', 'in_mask')]),
        (inputnode, skullstrip_tpl, [('in_mask', 'in_mask')]),

        # refined normalisation
        (skullstrip_tpl, refine_norm, [('out_file', 'fixed_image')]),
        (skullstrip_tar, refine_norm, [('out_file', 'moving_image')]),

        #organise refined normalisation transforms for warps
        (refine_norm, split_final_transforms, [('reverse_transforms', 'inlist')]),
        (split_final_transforms, mrg_final_transforms, [('out2', 'in1')]),
        (split_final_transforms, mrg_final_transforms, [('out1', 'in2')]),

        #warp mask to subject space and write out
        (mrg_final_transforms, warp_mask_out, [('out', 'transforms')]),
        (skullstrip_tar, warp_mask_out, [('out_file', 'reference_image')]),
        (warp_mask_out, sinker, [('output_image', 'derivatives.@out_mask')]),
    ])
    # add second target prep stage if necessary
    if bids_suffix.lower() == 't2w':
        wf.connect([(warp_mask_1, tar_prep, [('output_image', 'inu_n4_final.weight_image')])])

    # add segmentation if necessary
    if atropos_model:
        wf.connect([
            # Warp labels to subject-space
            (mrg_final_transforms, warp_seg_labels, [('out', 'transforms')]),
            (skullstrip_tar, warp_seg_labels, [('out_file', 'reference_image')]),

            # Segmentation
            (skullstrip_tar, segment, [('out_file', 'intensity_images')]),
            (warp_seg_labels, segment, [('output_image', 'prior_image')]),
            (warp_mask_out, segment, [('output_image', 'mask_image')])
        ])

    return wf

def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
