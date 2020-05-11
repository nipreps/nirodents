# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Create Mosaic Plot to showcase masking"""

import math
import os.path as op
import numpy as np
import nibabel as nb

import matplotlib.pyplot as plt

DEFAULT_DPI = 300
DINA4_LANDSCAPE = (11.69, 8.27)
DINA4_PORTRAIT = (8.27, 11.69)


def plot_slice(
    dslice,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
    annotate=False,
):
    from matplotlib.cm import get_cmap

    if isinstance(cmap, (str, bytes)):
        cmap = get_cmap(cmap)

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]

    phys_sp = np.array(spacing) * dslice.shape
    ax.imshow(
        np.swapaxes(dslice, 0, 1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[0], 0, phys_sp[1]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis("off")

    bgcolor = cmap(min(vmin, 0.0))
    fgcolor = cmap(vmax)

    if annotate:
        ax.text(
            0.95,
            0.95,
            "R",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )
        ax.text(
            0.05,
            0.95,
            "L",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    if label is not None:
        ax.text(
            0.98,
            0.01,
            label,
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    return ax


def plot_mosaic(
    img,
    out_file=None,
    ncols=8,
    title=None,
    overlay_mask=None,
    bbox_mask_file=None,
    only_plot_noise=False,
    annotate=True,
    vmin=None,
    vmax=None,
    cmap="Greys_r",
    plot_sagittal=True,
    fig=None,
    zmax=128,
):

    if isinstance(img, (str, bytes)):
        nii = nb.as_closest_canonical(nb.load(img))
        img_data = nii.get_data()
        zooms = nii.header.get_zooms()
    else:
        img_data = img
        zooms = [1.0, 1.0, 1.0]
        out_file = "mosaic.svg"

    # Remove extra dimensions
    img_data = np.squeeze(img_data)

    if img_data.shape[2] > zmax and bbox_mask_file is None:
        lowthres = np.percentile(img_data, 5)
        mask_file = np.ones_like(img_data)
        mask_file[img_data <= lowthres] = 0
        img_data = _bbox(img_data, mask_file)

    if bbox_mask_file is not None:
        bbox_data = nb.as_closest_canonical(nb.load(bbox_mask_file)).get_data()
        img_data = _bbox(img_data, bbox_data)

    z_vals = np.array(list(range(0, img_data.shape[2])))

    # Reduce the number of slices shown
    if len(z_vals) > zmax:
        rem = 15
        # Crop inferior and posterior
        if not bbox_mask_file:
            # img_data = img_data[..., rem:-rem]
            z_vals = z_vals[rem:-rem]
        else:
            # img_data = img_data[..., 2 * rem:]
            z_vals = z_vals[2 * rem :]

    while len(z_vals) > zmax:
        # Discard one every two slices
        # img_data = img_data[..., ::2]
        z_vals = z_vals[::2]

    n_images = len(z_vals) * 0.7
    nrows = math.ceil(n_images / ncols)
    if plot_sagittal:
        nrows += 1

    if overlay_mask:
        overlay_data = nb.as_closest_canonical(nb.load(overlay_mask)).get_data()

    # create figures
    if fig is None:
        fig = plt.figure(figsize=(22, nrows * 3))

    est_vmin, est_vmax = _get_limits(img_data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    naxis = 1
    new_lims = int(len(z_vals) * 0.15)
    for z_val in z_vals[new_lims:-new_lims]:
        ax = fig.add_subplot(nrows, ncols, naxis)

        if overlay_mask:
            ax.set_rasterized(True)
        plot_slice(
            img_data[:, :, z_val],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[:2],
            label="%d" % z_val,
            annotate=annotate,
        )

        if overlay_mask:
            from matplotlib import cm

            msk_cmap = cm.Reds  # @UndefinedVariable
            msk_cmap._init()
            alphas = np.linspace(0, 0.75, msk_cmap.N + 3)
            msk_cmap._lut[:, -1] = alphas
            plot_slice(
                overlay_data[:, :, z_val],
                vmin=0,
                vmax=1,
                cmap=msk_cmap,
                ax=ax,
                spacing=zooms[:2],
            )
        naxis += 1

    if plot_sagittal:
        naxis = ncols * (nrows - 1) + 1

        step = int(img_data.shape[0] / (ncols + 1))
        start = int(step * 3.5)
        stop = int(img_data.shape[0] - (step * 3.5))

        if step == 0:
            step = 1

        for x_val in list(range(start, stop, step))[:ncols]:
            ax = fig.add_subplot(nrows - 1, ncols, naxis)

            plot_slice(
                img_data[x_val, ...],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label="%d" % x_val,
                spacing=[zooms[0], zooms[2]],
            )
            naxis += 1

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
    )

    if title:
        fig.suptitle(title, fontsize="10")
    fig.subplots_adjust(wspace=0.002, hspace=0.002)

    if out_file is None:
        fname, ext = op.splitext(op.basename(img))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_mosaic.svg")

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
    return out_file


def _bbox(img_data, bbox_data):
    B = np.argwhere(bbox_data)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
    return img_data[ystart:ystop, xstart:xstop, zstart:zstop]


def _get_limits(nifti_file, only_plot_noise=False):
    if isinstance(nifti_file, str):
        nii = nb.as_closest_canonical(nb.load(nifti_file))
        data = nii.get_data()
    else:
        data = nifti_file

    data_mask = np.logical_not(np.isnan(data))

    if only_plot_noise:
        data_mask = np.logical_and(data_mask, data != 0)
        vmin = np.percentile(data[data_mask], 0)
        vmax = np.percentile(data[data_mask], 61)
    else:
        vmin = np.percentile(data[data_mask], 0.5)
        vmax = np.percentile(data[data_mask], 99.5)
    return vmin, vmax
