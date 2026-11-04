"""Shared dataset handling for parameter removal commands."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import typer
import xarray as xr

from ..io import load_dataset


def _reduce_dataset(
    file: Path,
    parameters: list[str],
    output_file: Path,
    operation: Callable[[xr.DataArray, list[str]], xr.DataArray],
) -> None:
    """Remove parameters and save the reduced forecast to a new file."""
    try:
        if output_file.exists():
            raise ValueError(f"Output file already exists: {output_file}")
        dataset = load_dataset(file)
        if "fisher" not in dataset.data_vars:
            raise ValueError("Dataset must contain a 'fisher' variable.")
        fisher = operation(dataset["fisher"], parameters)
        retained = fisher.row.values.tolist()
        reduced = dataset.drop_vars("covariance", errors="ignore").sel(
            row=retained, col=retained
        )
        fisher.attrs = dataset["fisher"].attrs.copy()
        reduced["fisher"] = fisher
        reduced.to_netcdf(output_file, engine="h5netcdf")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"Saved reduced forecast to {output_file}", err=True)
