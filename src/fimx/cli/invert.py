"""Invert subcommand."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, get_args

import numpy as np
import typer
import xarray as xr

from ..construction import _validate_matrix
from ..inversion import InversionMethod, inv
from ..io import load_dataset, save_dataset
from .options import InversionMethodOption


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
    successful: Sequence[InversionMethod],
) -> InversionMethod:
    """Save the Dataset with a covariance from the first successful method.

    Any covariance already in the Dataset is ignored and replaced.
    """
    method = next(
        method for method in get_args(InversionMethod) if method in successful
    )
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
class MethodInversionReport:
    """Result for one inversion method."""

    method: InversionMethod
    success: bool
    max_abs_residual: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class InversionReport:
    """Diagnostics and inversion results for a Fisher matrix."""

    file: str
    matrix_size: int
    parameters: tuple[str, ...]
    condition_number: float | None
    rank: int
    min_eigenvalue: float | None
    max_eigenvalue: float | None
    positive_definite: bool
    methods: tuple[MethodInversionReport, ...]

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
            "Inversion methods:",
        ]
        for result in self.methods:
            if result.success:
                residual = _format_number(result.max_abs_residual)
                lines.append(f"  {result.method}: max |F @ F_inv - I| = {residual}")
            else:
                lines.append(f"  {result.method}: failed ({result.error})")
        return "\n".join(lines)

    def asdict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary of report values."""
        methods: dict[str, dict[str, bool | str | float | None]] = {}
        for result in self.methods:
            if result.success:
                methods[result.method] = {
                    "success": True,
                    "max_abs_residual": result.max_abs_residual,
                }
            else:
                methods[result.method] = {
                    "success": False,
                    "error": result.error,
                }
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
            "methods": methods,
        }


def _inversion_report(
    path: Path,
    fisher: xr.DataArray,
    methods: Sequence[InversionMethod],
) -> InversionReport:
    """Calculate matrix diagnostics and inversion residual summaries."""
    matrix, parameters = _validate_matrix(fisher)
    values = matrix.values
    eigenvalues = np.linalg.eigvalsh(values)
    results: list[MethodInversionReport] = []
    identity = np.eye(len(parameters))
    for method in methods:
        try:
            inverse = inv(matrix, method=method)
        except np.linalg.LinAlgError as error:
            results.append(MethodInversionReport(method, False, error=str(error)))
            continue
        residual = values @ inverse.values - identity
        results.append(
            MethodInversionReport(
                method,
                True,
                max_abs_residual=_json_number(float(np.max(np.abs(residual)))),
            )
        )
    return InversionReport(
        file=str(path),
        matrix_size=len(parameters),
        parameters=tuple(parameters),
        condition_number=_json_number(float(np.linalg.cond(values))),
        rank=int(np.linalg.matrix_rank(values)),
        min_eigenvalue=_json_number(float(eigenvalues[0])),
        max_eigenvalue=_json_number(float(eigenvalues[-1])),
        positive_definite=bool(eigenvalues[0] > 0),
        methods=tuple(results),
    )


def _methods(selected: Sequence[InversionMethodOption] | None) -> list[InversionMethod]:
    """Return the requested methods, or every method in canonical order."""
    options = selected if selected is not None else tuple(InversionMethodOption)
    return [option.value for option in options]


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
        list[InversionMethodOption] | None,
        typer.Option(
            "--inversion-method",
            show_default=False,
            help=(
                "Methods to evaluate (default: all methods). "
                "List several methods or repeat the option."
            ),
        ),
    ] = None,
    save: Annotated[
        Path | None,
        typer.Option(
            "--save",
            metavar="PATH",
            help=(
                "Write the Dataset with a new covariance to PATH, from the first "
                "successful selected method in the order cholesky, inv, pinv."
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
        methods = _methods(inversion_method)
        inversion_report = _inversion_report(file, fisher, methods)
        successful = [
            result.method for result in inversion_report.methods if result.success
        ]
        saved = (
            _save_covariance(save, dataset, fisher, successful)
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
