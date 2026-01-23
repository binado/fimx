from typing import Any, Hashable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from fishy.exceptions import InsufficientDimsError


def ensure_dims(da: xr.DataArray, expected: int = 2) -> None:
    if da.ndim < expected:
        raise InsufficientDimsError(da, expected=expected)


def _ensure_no_dim_collision(da: xr.DataArray, dim: Hashable) -> None:
    if dim in da.dims:
        raise ValueError(f"Dimension '{dim}' already exists in DataArray")


def canonicalize_dims(da: xr.DataArray, row: Hashable, col: Hashable) -> xr.DataArray:
    if row not in da.dims:
        raise ValueError(f"Dimension '{row}' not found in DataArray")
    if col not in da.dims:
        raise ValueError(f"Dimension '{col}' not found in DataArray")
    batch_dims = tuple(dim for dim in da.dims if dim not in (row, col))
    return da.transpose(*batch_dims, row, col)


def _get_matrix_dims(da: xr.DataArray) -> tuple[Hashable, Hashable]:
    ensure_dims(da)
    dims = da.dims
    row, col = dims[-2], dims[-1]
    return row, col


def _get_batch_dims(da: xr.DataArray) -> tuple[Hashable, ...]:
    matrix_dims = _get_matrix_dims(da)
    return tuple(dim for dim in da.dims if dim not in matrix_dims)


def _get_matrix_shape(da: xr.DataArray) -> tuple[int, int]:
    row, col = _get_matrix_dims(da)
    return da.sizes[row], da.sizes[col]


def _ensure_square(da: xr.DataArray) -> None:
    ensure_dims(da)
    row_size, col_size = _get_matrix_shape(da)
    if row_size != col_size:
        raise ValueError(
            "The last two dimensions of the DataArray must have equal size"
        )


def at(da: xr.DataArray, row: Any | None = None, col: Any | None = None, **kwargs):
    ensure_dims(da)
    row_dim, col_dim = _get_matrix_dims(da)
    if row is None and col is None:
        raise ValueError("Either row or col must be specified")

    indexers = kwargs.pop("indexers", {})
    if row is not None:
        indexers[row_dim] = row
    if col is not None:
        indexers[col_dim] = col
    return da.isel(indexers=indexers, **kwargs)


def diagonal(da: xr.DataArray, offset: int = 0, out_dim: Hashable = "diagonal"):
    _ensure_no_dim_collision(da, out_dim)
    matrix_dims = _get_matrix_dims(da)
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
    matrix_dims = _get_matrix_dims(da)
    return xr.apply_ufunc(
        np.linalg.det,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[[]],
    )


def inv(da: xr.DataArray) -> xr.DataArray:
    _ensure_square(da)
    matrix_dims = _get_matrix_dims(da)
    return xr.apply_ufunc(
        np.linalg.inv,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[matrix_dims],
    )


def pinv(da: xr.DataArray, rcond: float = 1e-15) -> xr.DataArray:
    matrix_dims = _get_matrix_dims(da)
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
    _ensure_square(da)
    matrix_dims = _get_matrix_dims(da)
    return xr.apply_ufunc(
        _symmetrize,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[matrix_dims],
    )
