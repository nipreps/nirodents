"""Interfaces."""
from nipype.interfaces.base import traits
from nipype.interfaces.mixins import CopyHeaderInterface
from nipype.interfaces.ants.segmentation import (
    DenoiseImageInputSpec, DenoiseImage as _DI
)


class _DenoiseImageInputSpec(DenoiseImageInputSpec):
    copy_header = traits.Bool(
        True,
        usedefault=True,
        desc="copy headers of the original image into the output (corrected) file",
    )


class DenoiseImage(_DI, CopyHeaderInterface):
    """Extends DenoiseImage to auto copy the header."""

    input_spec = _DenoiseImageInputSpec
    _copy_header_map = {"output_image": "input_image"}
