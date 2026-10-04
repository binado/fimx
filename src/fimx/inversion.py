"""Matrix inversion, diagnosis, and marginalized constraints."""

from typing import Literal, get_args

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


def diagnose(F: xr.DataArray, *, method: InversionMethod = "cholesky") -> xr.Dataset:
    """Return numerical diagnostics for a Fisher matrix and one inversion.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    method : {'cholesky', 'inv', 'pinv'}
        Inversion algorithm; see :func:`inv`.

    Returns
    -------
    xarray.Dataset
        Fresh diagnostics. ``eigenvalues`` lies on dimension ``index``, in
        ascending order. Scalar variables are ``condition_number``, ``rank``,
        ``positive_definite``, ``positive_semidefinite``, ``residual``,
        ``success``, ``method``, and ``error``. The coordinate ``parameter`` carries the matrix labels and
        is not a dimension of any variable.

    Raises
    ------
    ValueError
        If the matrix contract is violated or ``method`` is unknown.

    Notes
    -----
    A failed inversion does not raise. ``success`` is false, ``error`` holds
    the ``LinAlgError`` message, and ``residual`` is NaN; the spectrum is
    still returned. ``condition_number`` is the ratio of the largest
    eigenvalue to the smallest, or infinity unless the matrix is positive
    definite. ``positive_semidefinite`` permits negative eigenvalues within
    ``n * eps * max(abs(eigenvalues))``, where ``eps`` is float64 machine
    precision. ``positive_definite`` requires strictly positive eigenvalues;
    ``success`` describes inversion independently of these flags.
    """
    _require_method(method)
    values, parameters = _validate_matrix(F)
    eigenvalues, condition, rank, positive_definite, positive_semidefinite = _spectrum(
        values
    )
    try:
        inverse = _invert(values, parameters, method)
    except np.linalg.LinAlgError as error:
        residual = np.nan
        success = False
        message = str(error)
    else:
        residual = _residual(values.values, inverse.values)
        success = True
        message = ""
    return xr.Dataset(
        data_vars={
            "eigenvalues": eigenvalues.copy(deep=True),
            "condition_number": condition,
            "rank": np.int64(rank),
            "positive_definite": bool(positive_definite),
            "positive_semidefinite": bool(positive_semidefinite),
            "residual": np.float64(residual),
            "success": bool(success),
            "method": method,
            "error": message,
        },
        coords={"parameter": ("parameter", list(parameters))},
    )


def inv(
    F: xr.DataArray,
    *,
    method: InversionMethod = "cholesky",
    metadata: bool = False,
) -> xr.DataArray:
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
    metadata : bool
        If true, record inversion diagnostics in the result's ``attrs``; see
        Notes.

    Returns
    -------
    xarray.DataArray
        Fresh symmetric inverse with the input parameter order.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular (``'inv'``) or not positive definite
        (``'cholesky'``).
    ValueError
        If the matrix contract is violated or ``method`` is unknown.

    Notes
    -----
    With ``metadata=True`` the result's ``attrs`` hold ``method``,
    ``condition_number`` (largest over smallest eigenvalue of ``F``, ``inf``
    unless ``F`` is positive definite) and ``residual`` (``max|F @ C - I|``).
    Like any xarray attributes, they describe this inversion only and are
    dropped by most subsequent operations.
    """
    _require_method(method)
    values, parameters = _validate_matrix(F)
    covariance = _invert(values, parameters, method)
    if metadata:
        _, condition, _, _, _ = _spectrum(values)
        covariance.attrs = {
            "method": method,
            "condition_number": condition,
            "residual": _residual(values.values, covariance.values),
        }
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
