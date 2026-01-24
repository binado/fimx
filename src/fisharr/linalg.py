from typing import Hashable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .accessors import get_matrix_dims
from .validation import ensure_dim_not_in_dataarray, ensure_dims, is_square_matrix


def diagonal(da: xr.DataArray, offset: int = 0, out_dim: Hashable = "diagonal"):
    ensure_dim_not_in_dataarray(da, out_dim)
    matrix_dims = get_matrix_dims(da)
    kwargs = {"axis1": -2, "axis2": -1, "offset": offset}
    return xr.apply_ufunc(
        np.diagonal,
        da,
        input_core_dims=[
            matrix_dims,
        ],
        output_core_dims=[[out_dim]],
        vectorize=False,
        output_dtypes=[da.dtype],
        kwargs=kwargs,
    )


def det(da: xr.DataArray) -> xr.DataArray:
    matrix_dims = get_matrix_dims(da)
    return xr.apply_ufunc(
        np.linalg.det,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[[]],
    )


def inv(da: xr.DataArray) -> xr.DataArray:
    ensure_dims(da)
    is_square_matrix(da, raise_exception=True)
    matrix_dims = get_matrix_dims(da)
    return xr.apply_ufunc(
        np.linalg.inv,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[matrix_dims],
    )


def pinv(da: xr.DataArray, rcond: float = 1e-15) -> xr.DataArray:
    matrix_dims = get_matrix_dims(da)
    row_dim, col_dim = matrix_dims
    return xr.apply_ufunc(
        np.linalg.pinv,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[[col_dim, row_dim]],
        kwargs={"rcond": rcond},
    )


def _matrix_transpose(a: NDArray) -> NDArray:
    return np.swapaxes(a, -2, -1)


def _symmetrize(a: NDArray) -> NDArray:
    at = _matrix_transpose(a)
    return (a + at) / 2


def symmetrize(da: xr.DataArray) -> xr.DataArray:
    ensure_dims(da)
    is_square_matrix(da, raise_exception=True)
    matrix_dims = get_matrix_dims(da)
    return xr.apply_ufunc(
        _symmetrize,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[matrix_dims],
    )
