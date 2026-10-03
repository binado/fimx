"""Matrix inversion and marginalized constraints."""

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


def _diagnostics(values: xr.DataArray, inverse: xr.DataArray) -> dict[str, float]:
    """Return the condition number of a matrix and the residual of its inverse."""
    eigenvalues = np.linalg.eigvalsh(values.values)
    condition = (
        float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else np.inf
    )
    identity = np.eye(values.shape[0])
    residual = float(np.max(np.abs(values.values @ inverse.values - identity)))
    return {"condition_number": condition, "residual": residual}


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
    if method not in get_args(InversionMethod):
        raise ValueError(
            f"Unknown inversion method {method!r}. "
            f"Available: {', '.join(get_args(InversionMethod))}."
        )
    values, parameters = _validate_matrix(F)
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
    covariance = _symmetrize(result)
    if metadata:
        covariance.attrs = {"method": method, **_diagnostics(values, covariance)}
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
