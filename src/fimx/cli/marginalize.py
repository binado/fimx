"""Marginalize selected forecast parameters."""

from pathlib import Path
from typing import Annotated

import typer

from ..ops import marginalize as marginalize_parameters
from ._reduce import _reduce_dataset


def marginalize(
    file: Annotated[
        Path, typer.Argument(help="NetCDF Dataset containing a Fisher matrix.")
    ],
    parameters: Annotated[
        list[str],
        typer.Option(
            "--parameters", help="Parameter to marginalize and remove; repeat."
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output-file", "-o", help="New output NetCDF file.")
    ],
) -> None:
    """Remove selected parameters through the Schur complement.

    Parameters
    ----------
    file : pathlib.Path
        Input forecast Dataset.
    parameters : list of str
        Names to marginalize, leaving the others in input order.
    output_file : pathlib.Path
        New output file. Existing files cannot be overwritten.
    """
    _reduce_dataset(file, parameters, output_file, marginalize_parameters)
