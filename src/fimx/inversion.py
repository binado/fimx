"""Matrix inversion, diagnosis, and marginalized constraints."""

from typing import Literal, get_args, overload

import numpy as np
import xarray as xr
from xarray_einstats import linalg

from .arrays import _symmetrize, _validate_matrix

InversionMethod = Literal["cholesky", "inv", "pinv"]


def _solve(values: xr.DataArray, rhs: xr.DataArray) -> xr.DataArray:
    """Solve a positive definite system using labeled Cholesky operations."""
    chol = linalg.cholesky(values, dims=("row", "col"))
    intermediate = linalg.solve(chol, rhs, dims=("row", "col", "rhs_col"))
    return linalg.solve(
        chol.transpose("col", "row"),
        intermediate,
        dims=("col", "row", "rhs_col"),
    ).rename(rhs_col="col")


def _require_method(method: str) -> None:
    """Raise ValueError when an inversion method is unknown."""
    if method not in get_args(InversionMethod):
        raise ValueError(
            f"Unknown inversion method {method!r}. "
            f"Available: {', '.join(get_args(InversionMethod))}."
        )


def _invert(
    values: xr.DataArray, parameters: list[str], method: InversionMethod
) -> xr.DataArray:
    """Return a fresh symmetric inverse for an already validated matrix."""
    if method == "cholesky":
        rhs = xr.DataArray(
            np.eye(len(parameters)),
            dims=("row", "rhs_col"),
            coords={"row": parameters, "rhs_col": parameters},
        )
        result = _solve(values, rhs)
    elif method == "inv":
        result = linalg.inv(values, dims=("row", "col"))
    else:
        result = linalg.pinv(values, dims=("row", "col"), hermitian=True)
    return _symmetrize(result)


def _eigenvalues(values: xr.DataArray) -> xr.DataArray:
    """Return ordered spectral values without parameter coordinates."""
    return (
        linalg.eigvalsh(values, dims=("row", "col"))
        .drop_vars("col")
        .rename(col="index")
    )


def _positive_semidefinite(eigenvalues: xr.DataArray) -> bool:
    """Classify a spectrum with a scale-relative roundoff allowance."""
    tolerance = (
        eigenvalues.sizes["index"] * np.finfo(np.float64).eps * abs(eigenvalues).max()
    )
    return bool((eigenvalues.min() >= -tolerance).item())


def _spectrum(values: xr.DataArray) -> tuple[xr.DataArray, float, int, bool, bool]:
    """Return the labeled spectrum, condition, rank, and definiteness flags."""
    eigenvalues = _eigenvalues(values)
    minimum = float(eigenvalues.min().item())
    condition = float(eigenvalues.max().item() / minimum) if minimum > 0 else np.inf
    rank = int(linalg.matrix_rank(values, dims=("row", "col")).item())
    return (
        eigenvalues,
        condition,
        rank,
        minimum > 0,
        _positive_semidefinite(eigenvalues),
    )


def _residual(values: np.ndarray, inverse: np.ndarray) -> float:
    """Return the maximum absolute residual of a computed inverse."""
    identity = np.eye(values.shape[0])
    return float(np.max(np.abs(values @ inverse - identity)))


def _diagnostics(
    values: xr.DataArray,
    parameters: list[str],
    method: InversionMethod,
    inverse: xr.DataArray | None,
    error: str = "",
) -> xr.Dataset:
    """Describe a validated matrix and its already computed inverse."""
    eigenvalues, condition, rank, positive_definite, positive_semidefinite = _spectrum(
        values
    )
    residual = (
        _residual(values.values, inverse.values) if inverse is not None else np.nan
    )
    return xr.Dataset(
        data_vars={
            "eigenvalues": eigenvalues.copy(deep=True),
            "condition_number": condition,
            "rank": np.int64(rank),
            "positive_definite": bool(positive_definite),
            "positive_semidefinite": bool(positive_semidefinite),
            "residual": np.float64(residual),
            "success": inverse is not None,
            "method": method,
            "error": error,
        },
        coords={"parameter": ("parameter", list(parameters))},
    )


@overload
def inv(
    F: xr.DataArray,
    *,
    method: InversionMethod = "cholesky",
    return_diagnostics: Literal[False] = False,
) -> xr.DataArray: ...


@overload
def inv(
    F: xr.DataArray,
    *,
    method: InversionMethod = "cholesky",
    return_diagnostics: Literal[True],
) -> tuple[xr.DataArray, xr.Dataset]: ...


@overload
def inv(
    F: xr.DataArray,
    *,
    method: InversionMethod = "cholesky",
    return_diagnostics: bool,
) -> xr.DataArray | tuple[xr.DataArray, xr.Dataset]: ...


def inv(
    F: xr.DataArray,
    *,
    method: InversionMethod = "cholesky",
    return_diagnostics: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.Dataset]:
    """Return the labeled inverse, or covariance of a Fisher matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    method : {'cholesky', 'inv', 'pinv'}
        Inversion algorithm. ``'cholesky'`` (default) requires a positive
        definite matrix. ``'inv'`` uses ``numpy.linalg.inv`` and only fails on
        exactly singular matrices. ``'pinv'`` uses the Moore-Penrose
        pseudoinverse, which always succeeds but assigns zero variance to
        unconstrained (null-space) directions, so degenerate parameters appear
        perfectly constrained rather than unconstrained.
    return_diagnostics : bool
        If true, return ``(covariance, diagnostics)`` from this inversion.

    Returns
    -------
    xarray.DataArray
        Fresh symmetric inverse with the input parameter order, when
        ``return_diagnostics=False`` (default).
    tuple of (xarray.DataArray, xarray.Dataset)
        Inverse and diagnostics when ``return_diagnostics=True``. The Dataset
        contains ascending ``eigenvalues`` on dimension ``index`` and scalar
        ``condition_number``, ``rank``, ``positive_definite``,
        ``positive_semidefinite``, ``residual``, ``success``, ``method``, and
        ``error``. Coordinate ``parameter`` carries the input labels.
        ``success`` is true and ``error`` is empty; inversion failures raise
        in either mode.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular (``'inv'``) or not positive definite
        (``'cholesky'``).
    ValueError
        If the matrix contract is violated or ``method`` is unknown.

    Notes
    -----
    Diagnostics describe the input matrix and the computed inverse.
    ``positive_definite`` requires strictly positive eigenvalues;
    ``positive_semidefinite`` permits negative eigenvalues within
    ``n * eps * max(abs(eigenvalues))``, using float64 machine precision.
    The residual is ``max|F @ C - I|``, including for pseudoinverses.

    """
    _require_method(method)
    values, parameters = _validate_matrix(F)
    covariance = _invert(values, parameters, method)
    if return_diagnostics:
        diagnostics = _diagnostics(values, parameters, method, covariance)
        return covariance, diagnostics
    return covariance


def errors(F: xr.DataArray, *, method: InversionMethod = "cholesky") -> xr.DataArray:
    """Return marginalized standard deviations from a Fisher matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    method : {'cholesky', 'inv', 'pinv'}
        Inversion algorithm; see :func:`inv`.

    Returns
    -------
    xarray.DataArray
        Square roots of the covariance diagonal on dimension ``parameter``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular (``'inv'``) or not positive definite
        (``'cholesky'``).
    ValueError
        If the matrix contract is violated or ``method`` is unknown.
    """
    covariance = inv(F, method=method)
    diagonal = linalg.diagonal(covariance, dims=("row", "col"))
    return xr.DataArray(
        np.sqrt(diagonal.values),
        dims=("parameter",),
        coords={"parameter": covariance.coords["row"].values.copy()},
    )
