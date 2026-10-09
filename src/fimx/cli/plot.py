"""Plot subcommand."""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer
import xarray as xr

from ..inversion import InversionMethod
from ..io import load_dataset
from ..plotting import plot as plot_forecasts
from ..plotting.base import MissingParams


def _json_object(value: str) -> dict[str, Any]:
    """Parse a JSON object from a command-line option."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"Expected a JSON object: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter("Expected a JSON object.")
    return parsed


def plot(
    files: Annotated[
        list[Path],
        typer.Option(
            "--file",
            metavar="PATH",
            help="NetCDF Dataset to plot; repeat to overlay multiple forecasts.",
        ),
    ],
    figure_file: Annotated[
        Path,
        typer.Option(
            "--figure-file",
            metavar="PATH",
            help="Output figure path.",
        ),
    ] = Path("plot.png"),
    figure_dpi: Annotated[
        int,
        typer.Option("--figure-dpi", metavar="DPI", help="Output resolution."),
    ] = 150,
    parameters: Annotated[
        list[str] | None,
        typer.Option(
            "--parameters",
            metavar="NAME",
            help="Parameters to include in the plot, in the requested order.",
        ),
    ] = None,
    filled: Annotated[
        bool,
        typer.Option(
            "--filled/--no-filled",
            help="Fill two-dimensional contours.",
        ),
    ] = True,
    method: Annotated[
        InversionMethod,
        typer.Option(
            "--inversion-method",
            help="Fisher matrix inversion method.",
        ),
    ] = "cholesky",
    missing_params: Annotated[
        MissingParams,
        typer.Option(
            "--missing-params",
            help="Handle parameters that are not plotted in a file "
            "(e.g. not shared by all files): marginalize over them or fix them.",
        ),
    ] = "marginalize",
    backend: Annotated[
        str,
        typer.Option("--backend", metavar="NAME", help="Plotting backend."),
    ] = "getdist",
    backend_kwargs: Annotated[
        dict[str, Any] | None,
        typer.Option(
            "--backend-kwargs",
            metavar="JSON",
            parser=_json_object,
            help="JSON object of backend-specific plotting options.",
        ),
    ] = None,
) -> None:
    """Overlay Fisher forecast Datasets in a corner plot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets: dict[str, xr.Dataset] = {}
    for path in files:
        label = path.stem
        if label in datasets:
            typer.echo(
                f"Error: Input filenames must have unique stems; duplicate {label!r}.",
                err=True,
            )
            raise typer.Exit(2)
        datasets[label] = load_dataset(path)

    figure = plot_forecasts(
        datasets,
        parameters=parameters,
        filled=filled,
        method=method,
        missing_params=missing_params,
        backend=backend,
        backend_kwargs=backend_kwargs,
    )
    figure.savefig(figure_file, dpi=figure_dpi)
