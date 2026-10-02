"""Cholesky inversion and marginalized constraints."""

import numpy as np
import xarray as xr
from xarray_einstats import linalg

from .construction import _new_matrix, _validate_matrix


def _solve(values: xr.DataArray, rhs: xr.DataArray) -> xr.DataArray:
    """Solve a positive definite system using labeled Cholesky operations."""
    chol = linalg.cholesky(values, dims=("row", "col"))
    intermediate = linalg.solve(chol, rhs, dims=("row", "col", "rhs_col"))
    return linalg.solve(
        chol.transpose("col", "row"),
        intermediate,
        dims=("col", "row", "rhs_col"),
    ).rename(rhs_col="col")


def inv(F: xr.DataArray) -> xr.DataArray:
    """Return the labeled inverse, or covariance of a Fisher matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.

    Returns
    -------
    xarray.DataArray
        Fresh symmetric inverse with the input parameter order.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular or not positive definite.
    ValueError
        If the matrix contract is violated.
    """
    values, parameters = _validate_matrix(F)
    rhs = xr.DataArray(
        np.eye(len(parameters)),
        dims=("row", "rhs_col"),
        coords={"row": parameters, "rhs_col": parameters},
    )
    covariance = _solve(values, rhs)
    raw = covariance.values
    return _new_matrix(raw / 2 + raw.T / 2, parameters)


def errors(F: xr.DataArray) -> xr.DataArray:
    """Return marginalized standard deviations from a Fisher matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.

    Returns
    -------
    xarray.DataArray
        Square roots of the covariance diagonal on dimension ``parameter``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular or not positive definite.
    ValueError
        If the matrix contract is violated.
    """
    covariance = inv(F)
    diagonal = linalg.diagonal(covariance, dims=("row", "col"))
    return xr.DataArray(
        np.sqrt(diagonal.values),
        dims=("parameter",),
        coords={"parameter": covariance.coords["row"].values.copy()},
    )
