"""Invert subcommand."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated

import typer
import xarray as xr

from ..inversion import InversionMethod, diagnose, inv
from ..io import load_dataset, save_dataset


def _load_matrix(dataset: xr.Dataset) -> xr.DataArray:
    """Return the Fisher variable of a Dataset, or its only variable."""
    if "fisher" in dataset.data_vars:
        return dataset["fisher"]
    if len(dataset.data_vars) == 1:
        return dataset[next(iter(dataset.data_vars))]
    raise ValueError("Dataset must contain a 'fisher' variable.")


def _save_covariance(
    path: Path,
    dataset: xr.Dataset,
    fisher: xr.DataArray,
    method: InversionMethod,
) -> InversionMethod:
    """Save the Dataset with a covariance from the selected method.

    Any covariance already in the Dataset is ignored and replaced.
    """
    labels = dataset["labels"].values.tolist() if "labels" in dataset else None
    fiducials = dataset.get("fiducials")
    save_dataset(
        path,
        fisher,
        covariance=inv(fisher, method=method, metadata=True),
        fiducials=fiducials,
        labels=labels,
    )
    return method


def _json_number(value: float) -> float | None:
    """Return finite numeric values, representing non-finite values as null."""
    return float(value) if isfinite(float(value)) else None


def _format_number(value: float | None) -> str:
    """Format a diagnostic number for the human-readable report."""
    return "∞/undefined" if value is None else f"{value:.6g}"


@dataclass(frozen=True)
class InversionReport:
    """Diagnostics and inversion result for a Fisher matrix."""

    file: str
    matrix_size: int
    parameters: tuple[str, ...]
    condition_number: float | None
    rank: int
    min_eigenvalue: float | None
    max_eigenvalue: float | None
    positive_definite: bool
    method: InversionMethod
    success: bool
    max_abs_residual: float | None = None
    error: str | None = None

    def __str__(self) -> str:
        """Return a readable, multi-line report for ``print()``."""
        lines = [
            f"Fisher matrix: {self.file}",
            f"Size: {self.matrix_size}",
            f"Parameters: {', '.join(self.parameters)}",
            f"Condition number: {_format_number(self.condition_number)}",
            f"Numerical rank: {self.rank}",
            (
                "Eigenvalue range: "
                f"[{_format_number(self.min_eigenvalue)}, "
                f"{_format_number(self.max_eigenvalue)}]"
            ),
            f"Positive definite: {self.positive_definite}",
        ]
        if self.success:
            residual = _format_number(self.max_abs_residual)
            lines.append(f"{self.method}: max |F @ F_inv - I| = {residual}")
        else:
            lines.append(f"{self.method}: failed ({self.error})")
        return "\n".join(lines)

    def asdict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary of report values."""
        return {
            "file": self.file,
            "matrix_size": self.matrix_size,
            "parameters": list(self.parameters),
            "condition_number": self.condition_number,
            "rank": self.rank,
            "eigenvalues": {
                "min": self.min_eigenvalue,
                "max": self.max_eigenvalue,
            },
            "positive_definite": self.positive_definite,
            "method": self.method,
            "success": self.success,
            "max_abs_residual": self.max_abs_residual if self.success else None,
            "error": None if self.success else self.error,
        }


def _inversion_report(
    path: Path,
    fisher: xr.DataArray,
    method: InversionMethod,
) -> InversionReport:
    """Calculate matrix diagnostics and an inversion residual summary."""
    diagnosis = diagnose(fisher, method=method)
    eigenvalues = diagnosis["eigenvalues"].values
    parameters = tuple(str(name) for name in diagnosis.parameter.values.tolist())
    message = str(diagnosis["error"].item())
    success = bool(diagnosis["success"].item())
    return InversionReport(
        file=str(path),
        matrix_size=len(parameters),
        parameters=parameters,
        condition_number=_json_number(float(diagnosis["condition_number"].item())),
        rank=int(diagnosis["rank"].item()),
        min_eigenvalue=_json_number(float(eigenvalues[0])),
        max_eigenvalue=_json_number(float(eigenvalues[-1])),
        positive_definite=bool(diagnosis["positive_definite"].item()),
        method=method,
        success=success,
        max_abs_residual=_json_number(float(diagnosis["residual"].item())),
        error=None if success else message,
    )


def invert(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            metavar="PATH",
            help="NetCDF file containing a Fisher matrix.",
        ),
    ],
    inversion_method: Annotated[
        InversionMethod,
        typer.Option(
            "--inversion-method",
            help="Method used for the inversion report and --save.",
        ),
    ] = "cholesky",
    save: Annotated[
        Path | None,
        typer.Option(
            "--save",
            metavar="PATH",
            help=(
                "Write the Dataset with a new covariance to PATH, from the "
                "selected method."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the report as JSON."),
    ] = False,
) -> None:
    """Diagnose Fisher matrix inversions."""
    try:
        dataset = load_dataset(file)
        fisher = _load_matrix(dataset)
        inversion_report = _inversion_report(file, fisher, inversion_method)
        successful = inversion_report.success
        saved = (
            _save_covariance(save, dataset, fisher, inversion_method)
            if save is not None and successful
            else None
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error

    if as_json:
        typer.echo(json.dumps(inversion_report.asdict(), allow_nan=False))
    else:
        typer.echo(inversion_report)
    if saved is not None:
        typer.echo(f"Saved {saved} covariance to {save}", err=True)
    if not successful:
        raise typer.Exit(1)
