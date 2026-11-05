"""Fix selected forecast parameters."""

from pathlib import Path
from typing import Annotated

import typer

from ..ops import fix as fix_parameters
from ._reduce import _reduce_dataset


def fix(
    file: Annotated[
        Path, typer.Argument(help="HDF5/NetCDF4 Dataset containing a Fisher matrix.")
    ],
    parameters: Annotated[
        list[str],
        typer.Option("--parameters", help="Parameter to fix and remove; repeat."),
    ],
    output_file: Annotated[
        Path, typer.Option("--output-file", "-o", help="New output HDF5/NetCDF4 file.")
    ],
) -> None:
    """Remove selected parameters while holding them fixed.

    Parameters
    ----------
    file : pathlib.Path
        Input forecast Dataset.
    parameters : list of str
        Names to fix and remove, leaving the others in input order.
    output_file : pathlib.Path
        New output file. Existing files cannot be overwritten.
    """
    _reduce_dataset(file, parameters, output_file, fix_parameters)
