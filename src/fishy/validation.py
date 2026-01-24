from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

import numpy as np
import xarray as xr

from fishy.exceptions import InsufficientDimsError, MatrixNotSquareError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def ensure_dims(a: xr.DataArray | ArrayLike, expected: int = 2) -> None:
    ndim = a.ndim if isinstance(a, xr.DataArray) else np.asarray(a).ndim
    if ndim < expected:
        raise InsufficientDimsError(a, expected=expected)


def is_square_matrix(a: xr.DataArray | ArrayLike, raise_exception: bool = True) -> bool:
    shape = a.shape if isinstance(a, xr.DataArray) else np.asarray(a).shape
    is_square = shape[-2] == shape[-1]
    if not is_square and raise_exception:
        raise MatrixNotSquareError(a)

    return is_square


def ensure_dim_not_in_dataarray(da: xr.DataArray, dim: Hashable) -> None:
    if dim in da.dims:
        raise ValueError(f"Dimension '{dim}' already exists in DataArray")


def ensure_dims_in_dataarray(da: xr.DataArray, *dim: Hashable) -> None:
    missing_dims = [str(d) for d in dim if d not in da.dims]
    if missing_dims:
        missing_dims_to_str = ", ".join(missing_dims)
        raise ValueError(f"Dimensions {missing_dims_to_str} not found in DataArray")
