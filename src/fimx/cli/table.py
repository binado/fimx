"""Command-line LaTeX tables of marginalized errors."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from ..io import load_dataset
from ..tables import latex_table


def table(
    file: Annotated[Path, typer.Argument(help="HDF5/NetCDF4 Dataset with covariance.")],
    output_file: Annotated[
        Path | None,
        typer.Option("--output-file", "-o", help="Write the table here, not stdout."),
    ] = None,
    significant_digits: Annotated[
        int, typer.Option("--significant-digits", min=1, help="Significant digits.")
    ] = 2,
    parameters: Annotated[
        list[str] | None,
        typer.Option("--parameters", help="Parameters to tabulate, in order; repeat."),
    ] = None,
    fiducials: Annotated[
        bool, typer.Option("--fiducials/--no-fiducials", help="Add a fiducial column.")
    ] = False,
) -> None:
    """Print marginalized errors from a stored covariance as a LaTeX table."""
    try:
        text = latex_table(
            load_dataset(file),
            parameters=parameters,
            significant_digits=significant_digits,
            fiducials=fiducials,
        )
        if output_file is not None:
            output_file.write_text(text)
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError, ImportError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error
    if output_file is None:
        typer.echo(text)
    else:
        typer.echo(f"Saved table to {output_file}", err=True)
