"""Cholesky inversion and marginalized constraints."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .construction import _new_matrix, _validate_matrix


def _solve(
    values: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve a positive definite system using Cholesky, without fallback."""
    chol = np.linalg.cholesky(values)
    return np.asarray(
        np.linalg.solve(chol.T, np.linalg.solve(chol, rhs)), dtype=np.float64
    )


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
    covariance = _solve(values, np.eye(len(parameters)))
    return _new_matrix(covariance / 2 + covariance.T / 2, parameters)


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
    return xr.DataArray(
        np.sqrt(np.diag(covariance.values)),
        dims=("parameter",),
        coords={"parameter": covariance.coords["row"].values.copy()},
    )
