"""Command-line interface for diagnosing Fisher matrix inversions."""

import json
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import get_args

import numpy as np
import xarray as xr

from .construction import _validate_matrix
from .inversion import InversionMethod, inv
from .io import load_dataset


def _parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser(description="Diagnose Fisher matrix inversions.")
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        metavar="PATH",
        help="NetCDF file containing a Fisher matrix.",
    )
    parser.add_argument(
        "--inversion-method",
        nargs="+",
        choices=get_args(InversionMethod),
        default=list(get_args(InversionMethod)),
        metavar="METHOD",
        help="Methods to evaluate (default: all methods).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    return parser


def _load_matrix(path: Path) -> xr.DataArray:
    """Load a standalone DataArray or the Fisher variable from a Dataset."""
    dataset = load_dataset(path)
    if "fisher" in dataset.data_vars:
        return dataset["fisher"]
    if len(dataset.data_vars) == 1:
        return dataset[next(iter(dataset.data_vars))]
    raise ValueError("Dataset must contain a 'fisher' variable.")


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
    path: Path, F: xr.DataArray, methods: Sequence[InversionMethod]
) -> InversionReport:
    """Calculate matrix diagnostics and inversion residual summaries."""
    matrix, parameters = _validate_matrix(F)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Load a Fisher matrix and print inversion diagnostics.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to the process command line.

    Returns
    -------
    int
        Zero if at least one selected inversion succeeds, otherwise one.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        fisher = _load_matrix(args.file)
        inversion_report = _inversion_report(args.file, fisher, args.inversion_method)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(inversion_report.asdict(), allow_nan=False))
    else:
        print(inversion_report)
    return int(not any(result.success for result in inversion_report.methods))


if __name__ == "__main__":
    raise SystemExit(main())
