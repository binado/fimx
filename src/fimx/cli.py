"""Command-line interface for plotting Fisher forecasts."""

from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Sequence
from json import loads
from pathlib import Path
from typing import get_args

from .inversion import InversionMethod
from .io import load_dataset
from .plotting import plot


def _parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser(
        description="Overlay Fisher forecast Datasets in a corner plot."
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="NetCDF Dataset to plot; repeat to overlay multiple forecasts.",
    )
    parser.add_argument(
        "--figure-file",
        type=Path,
        default=Path("plot.png"),
        metavar="PATH",
        help="Output figure path (default: plot.png).",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="Output resolution in dots per inch (default: 150).",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        metavar="NAME",
        help="Parameters to include in the plot, in the requested order.",
    )
    parser.add_argument(
        "--filled",
        action=BooleanOptionalAction,
        default=True,
        help="Fill two-dimensional contours (default; use --no-filled for lines).",
    )
    parser.add_argument(
        "--inversion-method",
        dest="method",
        choices=get_args(InversionMethod),
        default="cholesky",
        help="Fisher matrix inversion method (default: cholesky).",
    )
    parser.add_argument(
        "--backend",
        default="getdist",
        metavar="NAME",
        help="Plotting backend (default: getdist).",
    )
    parser.add_argument(
        "--backend-kwargs",
        type=loads,
        metavar="JSON",
        help="JSON object of backend-specific plotting options.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load Fisher forecast files, plot them, and save the figure.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to the process command line.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    datasets = {}
    for path in args.files:
        label = path.stem
        if label in datasets:
            parser.error(
                f"Input filenames must have unique stems; duplicate {label!r}."
            )
        datasets[label] = load_dataset(path)

    figure = plot(
        datasets,
        parameters=args.parameters,
        filled=args.filled,
        method=args.method,
        backend=args.backend,
        backend_kwargs=args.backend_kwargs,
    )
    figure.savefig(args.figure_file, dpi=args.figure_dpi)


if __name__ == "__main__":
    main()
