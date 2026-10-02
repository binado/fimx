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
        "--plot-label-var",
        metavar="NAME",
        help="Dataset variable containing string plot labels on the row dimension.",
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

    parameter_labels = None
    if args.plot_label_var is not None:
        labels_by_file = {}
        for file_label, forecast in datasets.items():
            if args.plot_label_var not in forecast.data_vars:
                parser.error(
                    f"Dataset {file_label!r} has no variable {args.plot_label_var!r}."
                )
            variable = forecast[args.plot_label_var]
            if variable.dims != ("row",):
                parser.error(
                    f"Plot label variable {args.plot_label_var!r} in "
                    f"{file_label!r} must have dimensions ('row',)."
                )
            values = variable.values.tolist()
            if any(not isinstance(value, str) for value in values):
                parser.error(
                    f"Plot label variable {args.plot_label_var!r} in "
                    f"{file_label!r} must contain only strings."
                )
            labels_by_file[file_label] = dict(
                zip(forecast.row.values.tolist(), values, strict=True)
            )

        if args.parameters is None:
            first = next(iter(datasets.values()))
            selected_parameters = [
                name
                for name in first.row.values.tolist()
                if all(name in forecast.row.values for forecast in datasets.values())
            ]
        else:
            selected_parameters = args.parameters

        parameter_labels = {}
        label_sources = {}
        for file_label, file_labels in labels_by_file.items():
            for name in selected_parameters:
                if name not in file_labels:
                    continue
                display_label = file_labels[name]
                if name in parameter_labels and parameter_labels[name] != display_label:
                    parser.error(
                        f"Conflicting plot labels for parameter {name!r}: "
                        f"{label_sources[name]!r} uses "
                        f"{parameter_labels[name]!r}, but {file_label!r} uses "
                        f"{display_label!r}."
                    )
                parameter_labels[name] = display_label
                label_sources[name] = file_label

    figure = plot(
        datasets,
        parameters=args.parameters,
        parameter_labels=parameter_labels,
        filled=args.filled,
        method=args.method,
        backend=args.backend,
        backend_kwargs=args.backend_kwargs,
    )
    figure.savefig(args.figure_file, dpi=args.figure_dpi)


if __name__ == "__main__":
    main()
