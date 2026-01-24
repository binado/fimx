from typing import Any, Hashable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .utils import ensure_dim_not_in_dataarray, ensure_dims, get_matrix_dims


def _get_matrix_shape(da: xr.DataArray) -> tuple[int, int]:
    row, col = get_matrix_dims(da)
    return da.sizes[row], da.sizes[col]


def _ensure_square(da: xr.DataArray) -> None:
    ensure_dims(da)
    row_size, col_size = _get_matrix_shape(da)
    if row_size != col_size:
        raise ValueError(
            "The last two dimensions of the DataArray must have equal size"
        )


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


def apply_matrix_indexers(
    da: xr.DataArray, row: Any | None = None, col: Any | None = None, **kwargs
) -> xr.DataArray:
    indexers = get_indexers(da, row, col)
    return da.isel(indexers=indexers, **kwargs)


def apply_matrix_indexers_to_dataset(
    ds: xr.Dataset,
    da: xr.DataArray,
    row: Any | None = None,
    col: Any | None = None,
    **kwargs,
):
    indexers = get_indexers(da, row, col)
    return ds.isel(indexers=indexers, **kwargs)


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
    _ensure_square(da)
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
    _ensure_square(da)
    matrix_dims = get_matrix_dims(da)
    return xr.apply_ufunc(
        _symmetrize,
        da,
        input_core_dims=[matrix_dims],
        output_core_dims=[matrix_dims],
    )


def submatrix(da: xr.DataArray, coords: Any) -> xr.DataArray:
    _ensure_square(da)
    return apply_matrix_indexers(da, row=coords, col=coords)
