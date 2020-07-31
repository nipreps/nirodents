"""Command Line Interface."""
from pathlib import Path


def get_parser():
    """Build parser object."""
    from os import cpu_count
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from ..__about__ import __version__

    parser = ArgumentParser(
        description="""\
artsBrainExtraction -- Atlas-based brain extraction tool of the \
ANTs-based Rodents ToolS (ARTs) package.\
""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_image",
        action="store",
        type=Path,
        help="The target image for brain extraction.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="artsBrainExtraction v{}".format(__version__),
    )
    parser.add_argument(
        "--template",
        action="store",
        default="Fischer344",
        help="The TemplateFlow ID of the reference template.",
    )
    parser.add_argument(
        "--omp-nthreads",
        action="store",
        type=int,
        default=cpu_count(),
        help="Number of CPUs available for multithreading processes.",
    )
    parser.add_argument(
        "--nprocs",
        action="store",
        type=int,
        default=cpu_count(),
        help="Number of processes that can be run in parallel.",
    )
    parser.add_argument(
        "-m",
        "--mri-scheme",
        action="store",
        type=str,
        default="T2w",
        help="select a particular MRI scheme",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        action="store",
        type=Path,
        default=Path("results").absolute(),
        help="path where intermediate results should be stored",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="path where intermediate results should be stored",
    )
    parser.add_argument(
        "--sloppy",
        dest="debug",
        action="store_true",
        default=False,
        help="Use low-quality tools for speed - TESTING ONLY",
    )
    parser.add_argument(
        "--no-antsAI",
        dest="antsai_init",
        action="store_false",
        default=True,
        help="Skip antsAI initialization step")
    return parser


def main():
    """Entry point."""
    from templateflow import update as update_templateflow
    from ..workflows.brainextraction import init_rodent_brain_extraction_wf

    opts = get_parser().parse_args()
    update_templateflow(overwrite=False)
    be = init_rodent_brain_extraction_wf(
        ants_affine_init=opts.antsai_init,
        debug=opts.debug,
        mri_scheme=opts.mri_scheme,
        omp_nthreads=opts.omp_nthreads,
        output_dir=opts.output_dir,
        template_id=opts.template,
    )
    be.base_dir = opts.work_dir
    be.inputs.inputnode.in_files = opts.input_image

    nipype_plugin = {"plugin": "Linear"}
    if opts.nprocs > 1:
        nipype_plugin["plugin"] = "MultiProc"
        nipype_plugin["plugin_args"] = {
            "nproc": opts.nprocs,
            "raise_insufficient": False,
            "maxtasksperchild": 1,
        }
    be.run(**nipype_plugin)


if __name__ == "__main__":
    raise RuntimeError(
        """\
nirodents/cli/brainextraction.py should not be run directly;
Please `pip install` nirodents and use the `artsBrainExtraction` command."""
    )
