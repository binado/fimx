"""Command-line interface for diagnosing Fisher matrix inversions."""

import json
from argparse import ArgumentParser
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any, get_args

import numpy as np
import xarray as xr

from .construction import _validate_matrix
from .inversion import InversionMethod, inv


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
    dataset = xr.load_dataset(path, engine="h5netcdf")
    if "fisher" in dataset.data_vars:
        return dataset["fisher"]
    if len(dataset.data_vars) == 1:
        return dataset[next(iter(dataset.data_vars))]
    raise ValueError("Dataset must contain a 'fisher' variable.")


def _json_number(value: float) -> float | None:
    """Return finite numeric values, representing non-finite values as null."""
    return float(value) if isfinite(float(value)) else None


def _report(F: xr.DataArray, methods: Sequence[InversionMethod]) -> dict[str, Any]:
    """Calculate matrix diagnostics and inversion residual summaries."""
    matrix, parameters = _validate_matrix(F)
    values = matrix.values
    eigenvalues = np.linalg.eigvalsh(values)
    condition_number = float(np.linalg.cond(values))
    summary: dict[str, Any] = {
        "matrix_size": len(parameters),
        "parameters": parameters,
        "condition_number": _json_number(condition_number),
        "rank": int(np.linalg.matrix_rank(values)),
        "eigenvalues": {
            "min": _json_number(float(eigenvalues[0])),
            "max": _json_number(float(eigenvalues[-1])),
        },
        "positive_definite": bool(eigenvalues[0] > 0),
        "methods": {},
    }
    results: dict[str, Any] = summary["methods"]
    identity = np.eye(len(parameters))
    for method in methods:
        try:
            inverse = inv(matrix, method=method)
        except np.linalg.LinAlgError as error:
            results[method] = {"success": False, "error": str(error)}
            continue
        residual = values @ inverse.values - identity
        results[method] = {
            "success": True,
            "max_abs_residual": _json_number(float(np.max(np.abs(residual)))),
        }
    return summary


def _format_number(value: float | None) -> str:
    """Format a diagnostic number for the human-readable report."""
    return "∞/undefined" if value is None else f"{value:.6g}"


def _print_text(path: Path, report: dict[str, Any]) -> None:
    """Print diagnostics in a readable text format."""
    print(f"Fisher matrix: {path}")
    print(f"Size: {report['matrix_size']}")
    print(f"Parameters: {', '.join(report['parameters'])}")
    print(f"Condition number: {_format_number(report['condition_number'])}")
    print(f"Numerical rank: {report['rank']}")
    eigenvalues = report["eigenvalues"]
    print(
        "Eigenvalue range: "
        f"[{_format_number(eigenvalues['min'])}, "
        f"{_format_number(eigenvalues['max'])}]"
    )
    print(f"Positive definite: {report['positive_definite']}")
    print("Inversion methods:")
    for method, result in report["methods"].items():
        if result["success"]:
            residual = _format_number(result["max_abs_residual"])
            print(f"  {method}: max |F @ F_inv - I| = {residual}")
        else:
            print(f"  {method}: failed ({result['error']})")


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
        report = _report(fisher, args.inversion_method)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps({"file": str(args.file), **report}, allow_nan=False))
    else:
        _print_text(args.file, report)
    return int(not any(result["success"] for result in report["methods"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
