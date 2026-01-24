from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Mapping, TypeAlias

import numpy as np
import xarray as xr

from .utils import ensure_dims, get_matrix_dims, is_square_matrix

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

MatrixDims: TypeAlias = tuple[Hashable, Hashable]


def normalize_dataarray(da: xr.DataArray) -> xr.DataArray:
    ensure_dims(da, expected=2)
    is_square_matrix(da, raise_exception=True)
    row, col = get_matrix_dims(da)
    coords = da.coords.get(row, da.coords.get(col, None))
    if coords is None:
        raise ValueError("Matrix dimensions must be present as coordinates.")

    return da.assign_coords({row: coords, col: coords})


def normalize_dataset(ds: xr.Dataset, key: Hashable) -> xr.Dataset:
    da = ds[key]
    da_normalized = normalize_dataarray(da)
    return ds.assign({key: da_normalized})


def build_dataarray_from_array(
    array: ArrayLike,
    parameters: Sequence[Hashable],
    matrix_dims: tuple[Hashable, Hashable],
    batch_dims: Sequence[Hashable] | None = None,
    batch_coords: Mapping[Hashable, ArrayLike | xr.Coordinates] | None = None,
    **kwargs,
) -> xr.DataArray:
    array = np.asarray(array)
    ensure_dims(array, expected=2)
    is_square_matrix(array, raise_exception=True)
    if array.shape[-1] != len(parameters):
        raise ValueError("parameters length must match matrix size.")

    expected_batch_dims = array.ndim - 2
    if batch_dims is None and expected_batch_dims > 0:
        raise ValueError(
            "batch_dims must be provided for arrays with more than 2 dimensions."
        )
    batch_dims_as_tuple = tuple(batch_dims or ())
    if len(batch_dims_as_tuple) != expected_batch_dims:
        raise ValueError("batch_dims length must match array batch dimensions.")

    dims = batch_dims_as_tuple + matrix_dims
    coords: dict[Hashable, Sequence[Hashable] | ArrayLike | xr.Coordinates] = {
        dim: parameters for dim in matrix_dims
    }
    if batch_coords is not None:
        coords.update({dim: batch_coords[dim] for dim in batch_dims_as_tuple})

    return xr.DataArray(array, dims=dims, coords=coords, **kwargs)


def partition_matrices(
    values: np.ndarray, idx_keep: list[int], idx_drop: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_kk = submatrix(values, idx_keep, idx_keep)
    f_kd = submatrix(values, idx_keep, idx_drop)
    f_dd = submatrix(values, idx_drop, idx_drop)
    f_dk = submatrix(values, idx_drop, idx_keep)
    return f_kk, f_kd, f_dd, f_dk


def submatrix(values: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    if values.ndim == 2:
        return values[np.ix_(rows, cols)]
    return np.take(np.take(values, rows, axis=1), cols, axis=2)
