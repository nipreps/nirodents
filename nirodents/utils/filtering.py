"""Signal processing filters."""


def truncation(
    in_file,
    clip_max=99.9,
    out_file=None,
    out_max=1000,
    out_min=0,
    percentiles=(0.1, 95),
):
    """Truncate and clip the input image intensities."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    img = nb.load(in_file)
    data = img.get_fdata()

    if percentiles[0] is not None:
        a_min = np.percentile(data.reshape(-1), percentiles[0])
    else:
        hist, edges = np.histogram(data.reshape(-1), bins="auto")
        a_min = edges[np.argmax(hist)]

    data -= a_min
    data[data < out_min] = out_min
    a_max = np.percentile(data.reshape(-1), percentiles[1])
    data *= out_max / a_max

    if clip_max is not None:
        data = np.clip(data, out_min, np.percentile(data.reshape(-1), clip_max))

    if out_file is None:
        out_file = fname_presuffix(Path(in_file).name, suffix="_trunc")

    out_file = str(Path(out_file).absolute())
    img.__class__(data, img.affine, img.header).to_filename(out_file)
    return out_file


def gaussian_filter(in_file, sigma, out_file=None):
    """Filter input image by convolving with a Gaussian kernel."""
    from pathlib import Path
    import nibabel as nb
    from scipy.ndimage import gaussian_filter
    from nipype.utils.filemanip import fname_presuffix

    if out_file is None:
        out_file = fname_presuffix(Path(in_file).name, suffix="_gauss")
    out_file = str(Path(out_file).absolute())

    img = nb.load(in_file)
    img.__class__(
        gaussian_filter(img.dataobj, sigma), img.affine, img.header
    ).to_filename(out_file)
    return out_file
