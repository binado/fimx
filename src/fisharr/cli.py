"""Command-line interface for plotting Fisher forecasts."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

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

    figure = plot(datasets)
    figure.savefig(args.figure_file, dpi=args.figure_dpi)


if __name__ == "__main__":
    main()
