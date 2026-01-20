from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike
import xarray as xr

from .dimensions import ParameterDims


def stack_batches(
    da: xr.DataArray,
    *,
    parameter_dims: ParameterDims,
    batch_dim: Hashable,
) -> xr.DataArray:
    batch_dims = [dim for dim in da.dims if dim not in parameter_dims]
    if not batch_dims:
        return da
    if len(batch_dims) == 1:
        if batch_dims[0] != batch_dim:
            return da.rename({batch_dims[0]: batch_dim})
        return da
    if batch_dim in da.dims and batch_dim not in batch_dims:
        raise ValueError("batch_dim conflicts with existing dimension names.")
    return da.stack({batch_dim: batch_dims})


def build_dataarray_from_array(
    array: np.ndarray,
    parameters: Sequence[str],
    *,
    parameter_dims: ParameterDims,
    batch_dim: Hashable,
) -> xr.DataArray:
    array = _reshape_batch_dims(np.asarray(array))
    if array.shape[-1] != array.shape[-2]:
        raise ValueError("Fisher matrix must be square.")
    if array.shape[-1] != len(parameters):
        raise ValueError("parameters length must match matrix size.")

    if array.ndim == 2:
        dims = parameter_dims
        coords = {parameter_dims[0]: parameters, parameter_dims[1]: parameters}
    else:
        dims = (batch_dim,) + parameter_dims
        coords = {
            batch_dim: np.arange(array.shape[0]),
            parameter_dims[0]: parameters,
            parameter_dims[1]: parameters,
        }
    return xr.DataArray(array, dims=dims, coords=coords)


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


def _reshape_batch_dims(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError("Data must be at least 2D.")
    if array.ndim == 2:
        return array
    batch_size = int(np.prod(array.shape[:-2]))
    return array.reshape(batch_size, array.shape[-2], array.shape[-1])


def _ensure_batch_coords(batch_coords: ArrayLike | None, batch_size: int) -> np.ndarray:
    """Ensure batch coordinates are provided, using integer indices as fallback.

    Parameters
    ----------
    batch_coords : ArrayLike | None
        User-provided batch coordinates, or None
    batch_size : int
        Size of the batch dimension

    Returns
    -------
    np.ndarray
        Batch coordinates array
    """
    if batch_coords is None:
        return np.arange(batch_size)
    return np.asarray(batch_coords)


def build_matrix_dataarray(
    values: np.ndarray,
    parameters: Sequence[str],
    *,
    parameter_dims: ParameterDims,
    batch_dim: Hashable,
    batch_coords: ArrayLike | None = None,
) -> xr.DataArray:
    """Build a DataArray for matrix-shaped values with automatic batch detection.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (n, n) or (batch, n, n)
    parameters : Sequence[str]
        Parameter names for the matrix dimensions
    parameter_dims : ParameterDims
        Names for the parameter dimensions
    batch_dim : Hashable
        Name for the batch dimension (used only if values has batch dims)
    batch_coords : Sequence | None
        Coordinates for the batch dimension (if None, uses integer indices)

    Returns
    -------
    xr.DataArray
        DataArray with appropriate dimensions and coordinates
    """
    if values.ndim == 2:
        dims = parameter_dims
        coords = {parameter_dims[0]: parameters, parameter_dims[1]: parameters}
    elif values.ndim == 3:
        dims = (batch_dim,) + parameter_dims
        coords = {
            batch_dim: _ensure_batch_coords(batch_coords, values.shape[0]),
            parameter_dims[0]: parameters,
            parameter_dims[1]: parameters,
        }
    else:
        raise ValueError(f"Expected 2D or 3D array, got {values.ndim}D")
    return xr.DataArray(values, dims=dims, coords=coords)


def build_vector_dataarray(
    values: np.ndarray,
    parameters: Sequence[str],
    *,
    parameter_dim: Hashable,
    batch_dim: Hashable,
    batch_coords: ArrayLike | None = None,
) -> xr.DataArray:
    """Build a DataArray for vector-shaped values with automatic batch detection.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (n,) or (batch, n)
    parameters : Sequence[str]
        Parameter names
    parameter_dim : Hashable
        Name for the parameter dimension
    batch_dim : Hashable
        Name for the batch dimension (used only if values has batch dims)
    batch_coords : Sequence | None
        Coordinates for the batch dimension (if None, uses integer indices)

    Returns
    -------
    xr.DataArray
        DataArray with appropriate dimensions and coordinates
    """
    if values.ndim == 1:
        dims = (parameter_dim,)
        coords = {parameter_dim: parameters}
    elif values.ndim == 2:
        dims = (batch_dim, parameter_dim)
        coords = {
            batch_dim: _ensure_batch_coords(batch_coords, values.shape[0]),
            parameter_dim: parameters,
        }
    else:
        raise ValueError(f"Expected 1D or 2D array, got {values.ndim}D")
    return xr.DataArray(values, dims=dims, coords=coords)


def build_scalar_dataarray(
    values: np.ndarray,
    *,
    batch_dim: Hashable,
    batch_coords: ArrayLike | None = None,
) -> xr.DataArray:
    """Build a DataArray for scalar values with automatic batch detection.

    Parameters
    ----------
    values : np.ndarray
        Array of shape () or (batch,)
    batch_dim : Hashable
        Name for the batch dimension (used only if values has batch dims)
    batch_coords : Sequence | None
        Coordinates for the batch dimension (if None, uses integer indices)

    Returns
    -------
    xr.DataArray
        DataArray with appropriate dimensions and coordinates
    """
    if values.ndim == 0 or (values.ndim == 1 and values.shape[0] == 1):
        return xr.DataArray(values)
    elif values.ndim == 1:
        coords = {batch_dim: _ensure_batch_coords(batch_coords, values.shape[0])}
        return xr.DataArray(values, dims=(batch_dim,), coords=coords)
    else:
        raise ValueError(f"Expected 0D or 1D array, got {values.ndim}D")
