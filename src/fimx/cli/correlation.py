"""Command-line correlation heatmaps."""

from pathlib import Path
from typing import Annotated

import typer

from ..io import load_dataset
from ..plotting import plot_correlation as plot_forecast_correlation


def plot_correlation(
    file: Annotated[Path, typer.Argument(help="HDF5/NetCDF4 Dataset with covariance.")],
    output_file: Annotated[
        Path, typer.Option("--output-file", "-o", help="Output figure path.")
    ] = Path("correlation.pdf"),
    figure_dpi: Annotated[
        int, typer.Option("--figure-dpi", help="Output resolution.")
    ] = 150,
    parameters: Annotated[
        list[str] | None,
        typer.Option("--parameters", help="Parameters to display, in order; repeat."),
    ] = None,
) -> None:
    """Plot one stored covariance as a parameter correlation heatmap."""
    figure = plot_forecast_correlation(load_dataset(file), parameters=parameters)
    import matplotlib.pyplot as plt

    try:
        figure.savefig(output_file, dpi=figure_dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
