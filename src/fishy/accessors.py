from __future__ import annotations

from typing import Hashable

import xarray as xr

from .validation import ensure_dims_in_dataarray


def get_matrix_dims(da: xr.DataArray) -> tuple[Hashable, Hashable]:
    dims = da.dims
    row, col = dims[-2], dims[-1]
    return row, col


def get_matrix_shape(da: xr.DataArray) -> tuple[int, int]:
    row, col = get_matrix_dims(da)
    return da.sizes[row], da.sizes[col]


def get_batch_dims(da: xr.DataArray) -> tuple[Hashable, ...]:
    matrix_dims = get_matrix_dims(da)
    return tuple(dim for dim in da.dims if dim not in matrix_dims)


def get_matrix_coords(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    dims = da.dims
    row, col = dims[-2], dims[-1]
    return da.coords[row], da.coords[col]


def canonicalize_dims(da: xr.DataArray, row: Hashable, col: Hashable) -> xr.DataArray:
    ensure_dims_in_dataarray(da, row, col)
    batch_dims = tuple(dim for dim in da.dims if dim not in (row, col))
    return da.transpose(*batch_dims, row, col)
