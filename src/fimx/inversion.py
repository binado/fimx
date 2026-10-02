"""Matrix inversion and marginalized constraints."""

from typing import Literal, get_args

import numpy as np
import xarray as xr
from xarray_einstats import linalg

from .construction import _symmetrize, _validate_matrix

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


def inv(F: xr.DataArray, *, method: InversionMethod = "cholesky") -> xr.DataArray:
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
    return _symmetrize(result)


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
