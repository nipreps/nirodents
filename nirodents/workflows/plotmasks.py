# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Run masking and print overlay mosaic plots"""

from nirodents.workflows import brainextraction, createplot
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.io import DataGrabber
import nibabel as nb
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from pathlib import Path

in_dir = os.curdir()

#Grab files for masking
dg = DataGrabber(sort_filelist = True)
dg.inputs.base_directory = in_dir
dg.inputs.template = 'rawfiles/sub-??/anat/*T2w.nii.gz'
results = dg.run()

for file in results.outputs.outfiles:
    #File-specific definitions
    fileparts = file.split('/')
    sub_id = fileparts[-3]
    sub_dir = os.path.join(in_dir, 'derivatives', sub_id)
    Path(sub_dir).mkdir(parents = True, exist_ok = True)

    #brain masking workflow
    brainwf = brainextraction.init_brain_extraction_wf(debug = True)
    brainwf.inputs.inputnode.in_files = file
    brainwf.inputs.sink_mask.base_directory = sub_dir
    brainwf.run()

    #plot mask
    grab_mask = DataGrabber(sort_filelist = False, base_directory = sub_dir, template = '*mask*.nii.gz')
    mask = grab_mask.run()
    plot_name = sub_id + '-mask_plot.svg'
    
    plot_masking.plot_mosaic(img = file, plot_sagittal=False, ncols= 7, 
        overlay_mask = mask.results.outfiles, out_file = os.path.join(sub_dir, plot_name))

if __name__ == '__main__':
    from sys import argv
    main(args=argv[1:])