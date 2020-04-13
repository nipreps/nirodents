# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Nipype translation of ANTs' workflows."""

# general purpose
import os
from multiprocessing import cpu_count
from pkg_resources import resource_filename as pkgr_fn
from packaging.version import parse as parseversion, Version
from warnings import warn

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection, Atropos
from nipype.interfaces.ants.utils import ImageMath, ResampleImageBySpacing, AI
from nipype.interfaces.afni import MaskTool
from nipype.interfaces.fsl import ApplyMask
from nipype.interfaces.io import DataSink

# niworkflows
from niworkflows.interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms)

from templateflow.api import get as get_template

def init_brain_extraction_wf(
    atropos_model=None,
    atropos_refine=True,
    atropos_use_random_seed=True,
    bids_suffix='T1w',
    bspline_fitting_distance=8,  # 4
    debug=False,
    final_normalization_quality='precise',
    in_template='WHS',
    init_normalization_quality='3stage',
    modality='T2w'
    mem_gb=3.0,
    name='brain_extraction_wf',
    omp_nthreads=None,
    tpl_suffix='T2star',
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
    wf = pe.Workflow(name)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files', 'in_mask']),
                        name='inputnode')

    # Find images in templateFlow
    tpl_target_path = get_template(in_template, resolution=debug + 1, suffix=tpl_suffix)
    tpl_regmask_path = get_template(in_template, resolution=debug + 1, atlas='v3', desc='brain', suffix='mask')
    if tpl_regmask_path:
        inputnode.inputs.in_mask = str(tpl_regmask_path)
    tpl_tissue_labels = get_template(in_template,resolution=debug + 1, desc='cerebrum', suffix='dseg')
    tpl_brain_mask = get_template(in_template, resolution=debug + 1, desc='cerebrum', suffix='mask')

    dil_mask = pe.Node(MaskTool(), name = 'dil_mask')
    dil_mask.inputs.outputtype = 'NIFTI_GZ'
    dil_mask.inputs.dilate_inputs = '2'
    dil_mask.inputs.fill_holes = True

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

    # resample template and target
    res_tmpl = pe.Node(ResampleImageBySpacing(
        out_spacing=(0.4, 0.4, 0.4), apply_smoothing=False), name='res_tmpl')
    if tpl_target_path:
        res_tmpl.inputs.input_image = str(tpl_target_path)
    res_target = pe.Node(ResampleImageBySpacing(
        out_spacing=(0.4, 0.4, 0.4), apply_smoothing=False), name='res_target')
    res_target2 = pe.Node(ResampleImageBySpacing(
        out_spacing=(0.4, 0.4, 0.4), apply_smoothing=False), name='res_target2')

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

    # Initialize transforms with antsAI
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.5), #0.25
        transform=('Rigid', 0.1),
        search_factor=(2, 0.015),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True),
        name='init_aff',
        n_procs=omp_nthreads)

    # Initial warping of template mask to subject space
    warp_mask = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=True), name='warp_mask')

    # Tolerate missing ANTs at construction time
    _ants_version = Registration().version
    if _ants_version and parseversion(_ants_version) >= Version('2.3.0'):
        init_aff.inputs.search_grid = (1, (1, 2, 2))

    fixed_mask_trait = 'fixed_image_mask'
    moving_mask_trait = 'moving_image_mask'
    if _ants_version and parseversion(_ants_version) >= Version('2.2.0'):
        fixed_mask_trait += 's'
        moving_mask_trait += 's'

    # Set up initial spatial normalization
    init_settings_file = 'brainextraction_%s_%s.json'
    init_norm = pe.Node(Registration(from_file=pkgr_fn(
        'nirodents', init_settings_file % init_normalization_quality, modality)),
        name='init_norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    init_norm.inputs.float = use_float

    # Refine INU correction
    inu_n4_final = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 5, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=bspline_fitting_distance),
        n_procs=omp_nthreads, name='inu_n4_final', iterfield=['input_image'])
    if _ants_version and parseversion(_ants_version) >= Version('2.1.0'):
        inu_n4_final.inputs.rescale_intensities = True
    else:
        warn("""\
Found ANTs version %s, which is too old. Please consider upgrading to 2.1.0 or \
greater so that the --rescale-intensities option is available with \
N4BiasFieldCorrection.""" % _ants_version, DeprecationWarning)

    split_init_transforms = pe.Node(niu.Split(splits=[1,1]), name='split_init_transforms')
    mrg_init_transforms = pe.Node(niu.Merge(2), name='mrg_init_transforms')

    # Use more precise transforms to warp mask to subject space
    warp_mask_final = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=[False, True]),
        name='warp_mask_final')

    # morphological closing of warped mask
    close_mask = pe.Node(MaskTool(), name = 'close_mask')
    close_mask.inputs.outputtype = 'NIFTI_GZ' 
    close_mask.inputs.dilate_inputs = '5 -5'
    close_mask.inputs.fill_holes = True

    # Use subject-space mask to skull-strip subject
    skullstrip_tar = pe.Node(ApplyMask(), name = 'skullstrip_tar')
    skullstrip_tpl = pe.Node(ApplyMask(), name = 'skullstrip_tpl')
    if tpl_target_path:
        skullstrip_tpl.inputs.in_file = tpl_target_path

    # Normalise skull-stripped image to brain template
    final_settings_file = 'brainextraction_%s_%s.json'
    final_norm = pe.Node(Registration(from_file=pkgr_fn(
        'nirodents', final_settings_file % final_normalization_quality, modality)),
        name='final_norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    final_norm.inputs.float = use_float

    split_final_transforms = pe.Node(niu.Split(splits=[1,1]), name='split_final_transforms')
    mrg_final_transforms = pe.Node(niu.Merge(2), name='mrg_final_transforms')

    warp_seg_mask = pe.Node(ApplyTransforms(
        interpolation='Linear', invert_transform_flags=[False, True]),
        name='warp_seg_mask')
    if tpl_brain_mask:
        warp_seg_mask.inputs.input_image = tpl_brain_mask

    warp_seg_labels =pe.Node(ApplyTransforms(
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

    if modality.lower() == 't2w':
        wf.connect([
            # resampling, truncation, initial N4, and creation of laplacian
            (inputnode, trunc, [('in_files', 'op1')]),
            (trunc, res_target, [(('output_image', _pop), 'input_image')]),
            (res_target, inu_n4, [('output_image', 'input_image')]),

            # dilation of input mask
            (inputnode, dil_mask, [('in_mask', 'in_file')]),

            # ants AI inputs
            (inu_n4, init_aff, [(('output_image', _pop), 'moving_image')]),
            (dil_mask, init_aff, [('out_file', 'fixed_image_mask')]),
            (res_tmpl, init_aff, [('output_image', 'fixed_image')]),

            # warp mask to individual space
            (dil_mask, warp_mask, [('out_file', 'input_image')]),
            (trunc, warp_mask, [(('output_image', _pop), 'reference_image')]),
            (init_aff, warp_mask, [('output_transform', 'transforms')]),

            # masked N4 correction
            (trunc, inu_n4_final, [(('output_image', _pop), 'input_image')]),
            (warp_mask, inu_n4_final, [('output_image', 'weight_image')]),

            # merge laplacian and original images
            (inu_n4_final, lap_target, [(('output_image', _pop), 'op1')]),
            (lap_target, norm_lap_target, [('output_image', 'op1')]),
            (norm_lap_target, mrg_target, [('output_image', 'in2')]),
            (inu_n4_final, res_target2, [(('output_image', _pop), 'input_image')]),
            (res_target2, mrg_target, [('output_image', 'in1')]),

            (res_tmpl, mrg_tmpl, [('output_image', 'in1')]),
            (lap_tmpl, norm_lap_tmpl, [('output_image', 'op1')]),
            (norm_lap_tmpl, mrg_tmpl, [('output_image', 'in2')]),

            # normalisation inputs
            (init_aff, init_norm, [('output_transform', 'initial_moving_transform')]),
            (warp_mask, init_norm, [('output_image', 'moving_image_masks')]),
            (dil_mask, init_norm, [('out_file', 'fixed_image_masks')]),
            (mrg_tmpl, init_norm, [('out', 'fixed_image')]),
            (mrg_target, init_norm, [('out', 'moving_image')]),

            # organise normalisation outputs to warp mask
            (init_norm, split_init_transforms, [('reverse_transforms', 'inlist')]),
            (split_init_transforms, mrg_init_transforms, [('out2', 'in1')]),
            (split_init_transforms, mrg_init_transforms, [('out1', 'in2')]),

            (mrg_init_transforms, warp_mask_final, [('out', 'transforms')]),
            (inu_n4_final, warp_mask_final, [(('output_image', _pop), 'reference_image')]),
            (dil_mask, warp_mask_final, [('out_file', 'input_image')]),
            (warp_mask_final, close_mask, [('output_image', 'in_file')]),
            (close_mask, sinker, [('out_file', 'derivatives.@out_mask')]),

            # mask brains
            (inu_n4_final, skullstrip_tar, [(('output_image', _pop), 'in_file')]),
            (close_mask, skullstrip_tar, [('out_file', 'mask_file')]),
            (inputnode, skullstrip_tpl, [('in_mask', 'mask_file')]),

            # final_normalisation
            (skullstrip_tpl, final_norm, [('out_file', 'fixed_image')]),
            (skullstrip_tar, final_norm, [('out_file', 'moving_image')]),

            # Warp mask and labels to subject-space
            (final_norm, split_final_transforms, [('reverse_transforms', 'inlist')]),
            (split_final_transforms, mrg_final_transforms, [('out2', 'in1')]),
            (split_final_transforms, mrg_final_transforms, [('out1', 'in2')]),

            (mrg_final_transforms, warp_seg_mask, [('out', 'transforms')]),
            (skullstrip_tar, warp_seg_mask, [('out_file', 'reference_image')]),
            (mrg_final_transforms, warp_seg_labels, [('out', 'transforms')]),
            (skullstrip_tar, warp_seg_labels, [('out_file', 'reference_image')]),

            # Segmentation
            (skullstrip_tar, segment, [('out_file', 'intensity_images')]),
            (warp_seg_labels, segment, [('output_image', 'prior_image')]),
            (warp_seg_mask, segment, [('output_image', 'mask_image')])
        ])
        return wf

    elif modality == 'mp2rage':
        wf.connect([
            # resampling and creation of laplacians
            (inputnode, res_target, [('in_files', 'input_image')]),
            (inputnode, lap_target, [('in_files', 'op1')]),
            (lap_target, norm_lap_target, [('output_image', 'op1')]),
            (norm_lap_target, mrg_target, [('output_image', 'in2')]), 
            (res_target, mrg_target, [('output_image', 'in1')]),

            (res_tmpl, mrg_tmpl, [('output_image', 'in1')]),
            (lap_tmpl, norm_lap_tmpl, [('output_image', 'op1')]),
            (norm_lap_tmpl, mrg_tmpl, [('output_image', 'in2')]),

            #dilation of input mask
            (inputnode, dil_mask, [('in_mask', 'in_file')]),

            # ants AI inputs
            (res_tmpl, init_aff, [('output_image', 'fixed_image')]),
            (res_target, init_aff, [('output_image', 'moving_image')]),
            (dil_mask, init_aff, [('out_file', 'fixed_image_mask')]),

            # warp mask to individual space
            (dil_mask, warp_mask, [('out_file', 'input_image')]),
            (inputnode, warp_mask, [('in_files', 'reference_image')]),
            (init_aff, warp_mask, [('output_transform', 'transforms')]),

            # normalisation inputs
            (mrg_tmpl, init_norm, [('out', 'fixed_image')]),
            (mrg_target, init_norm, [('out', 'moving_image')]),    
            (dil_mask, init_norm, [('out_file', 'fixed_image_masks')]),
            (warp_mask, init_norm, [('output_image', 'moving_image_masks')]),
            (init_aff, init_norm, [('output_transform', 'initial_moving_transform')]),

            #organise normalisation outputs to warp mask
            (init_norm, split_init_transforms, [('reverse_transforms', 'inlist')]),
            (split_init_transforms, mrg_init_transforms, [('out2', 'in1')]),
            (split_init_transforms, mrg_init_transforms, [('out1', 'in2')]),

            (mrg_init_transforms, warp_mask_final, [('out', 'transforms')]),
            (inputnode, warp_mask_final, [('in_files', 'reference_image')]),
            (dil_mask, warp_mask_final, [('out_file', 'input_image')]),
            (warp_mask_final, close_mask, [('output_image', 'in_file')]),

            #mask brains
            (inputnode, skullstrip_tar, [('in_files', 'in_file')]),
            (close_mask, skullstrip_tar, [('out_file', 'mask_file')]),
            (inputnode, skullstrip_tpl, [('in_mask', 'mask_file')]),

            #final_normalisation
            (skullstrip_tpl, final_norm, [('out_file', 'fixed_image')]),
            (skullstrip_tar, final_norm, [('out_file', 'moving_image')]),

            # Warp mask and labels to subject-space
            (final_norm, split_final_transforms, [('reverse_transforms', 'inlist')]),
            (split_final_transforms, mrg_final_transforms, [('out2', 'in1')]),
            (split_final_transforms, mrg_final_transforms, [('out1', 'in2')]),

            (mrg_final_transforms, warp_seg_mask, [('out', 'transforms')]),
            (skullstrip_tar, warp_seg_mask, [('out_file', 'reference_image')]),
            (mrg_final_transforms, warp_seg_labels, [('out', 'transforms')]),
            (skullstrip_tar, warp_seg_labels, [('out_file', 'reference_image')]),

            # Segmentation
            (skullstrip_tar, segment, [('out_file', 'intensity_images')]),
            (warp_seg_labels, segment, [('output_image', 'prior_image')]),
            (warp_seg_mask, segment, [('output_image', 'mask_image')]),
            ])
        return wf

def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
