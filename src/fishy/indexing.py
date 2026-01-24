from typing import Any, Hashable

import xarray as xr

from .accessors import get_matrix_dims
from .validation import ensure_dims, is_square_matrix


def get_indexers(
    da: xr.DataArray, row: Any | None = None, col: Any | None = None
) -> dict[Hashable, Any]:
    if row is None and col is None:
        raise ValueError("Either row or col must be specified")
    row_dim, col_dim = get_matrix_dims(da)
    indexers = {}
    if row is not None:
        indexers[row_dim] = row
    if col is not None:
        indexers[col_dim] = col
    return indexers


def isel_matrix(
    da: xr.DataArray, row: Any | None = None, col: Any | None = None, **kwargs
) -> xr.DataArray:
    indexers = get_indexers(da, row, col)
    return da.isel(indexers, **kwargs)


apply_matrix_indexers = isel_matrix


def isel_matrix_dataset(
    ds: xr.Dataset,
    da: xr.DataArray,
    row: Any | None = None,
    col: Any | None = None,
    **kwargs,
) -> xr.Dataset:
    indexers = get_indexers(da, row, col)
    return ds.isel(indexers, **kwargs)


apply_matrix_indexers_to_dataset = isel_matrix_dataset


def sel_matrix(
    da: xr.DataArray, row: Any | None = None, col: Any | None = None, **kwargs
) -> xr.DataArray:
    indexers = get_indexers(da, row, col)
    return da.sel(indexers, **kwargs)


def sel_matrix_dataset(
    ds: xr.Dataset,
    da: xr.DataArray,
    row: Any | None = None,
    col: Any | None = None,
    **kwargs,
) -> xr.Dataset:
    indexers = get_indexers(da, row, col)
    return ds.sel(indexers, **kwargs)


def submatrix(da: xr.DataArray, coords: Any) -> xr.DataArray:
    ensure_dims(da)
    is_square_matrix(da, raise_exception=True)
    return isel_matrix(da, row=coords, col=coords)
