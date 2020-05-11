"""Command Line Interface to print mask overlay mosaic plots"""

import os
from pathlib import Path


def get_parser():
    """Build parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        prog="plot_mask",
        description="""plot_mask -- Create mosaic plot of mask on a base.""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--base-image",
        type=Path,
        required=True,
        help="The base image for overlay.",
    )
    parser.add_argument(
        "-m",
        "--mask-image",
        type=Path,
        required=True,
        help="The mask image for overlay.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="The location of output file."
    )
    parser.add_argument(
        "-s",
        "--plot-sagittal",
        type=int,
        default=0,
        choices=[0, 1],
        help="Boolean to also print sagittal plane.",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        default=7,
        help="Integer describing number of columns in plot.",
    )

    return parser


def main():
    """Entry point."""
    from nirodents import viz

    # set output if one is not defined
    opts = get_parser().parse_args()
    if opts.output is None:
        mask_dir = os.path.dirname(opts.mask_image)
        output_path = os.path.join(mask_dir, "mask_plot.svg")
    else:
        if os.path.basename(opts.output)[-4:] != ".svg":
            raise ValueError("Output must be .svg file")
        else:
            output_path = opts.output

    # call function
    viz.plot_mosaic(
        img=opts.base_image,
        overlay_mask=opts.mask_image,
        plot_sagittal=bool(opts.plot_sagittal),
        ncols=opts.columns,
        out_file=output_path,
    )


if __name__ == "__main__":
    raise RuntimeError(
        """\
nirodents/cli/plotmask.py should not be run directly;
Please `pip install` nirodents and use the `plot_mask` command."""
    )
