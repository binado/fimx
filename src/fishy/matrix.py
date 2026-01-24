from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np
import xarray as xr

import fishy.linalg


@dataclass
class Matrix:
    data: xr.DataArray

    def __post_init__(self):
        if self.data.ndim < 2:
            raise ValueError("data array must have at least 2 dimensions")

    @property
    def row_dim(self) -> Hashable:
        return self.data.dims[-2]

    @property
    def col_dim(self) -> Hashable:
        return self.data.dims[-1]

    @property
    def matrix_dims(self) -> tuple[Hashable, Hashable]:
        return self.row_dim, self.col_dim

    @property
    def matrix_shape(self) -> tuple[int, int]:
        sizes = self.data.sizes
        return (sizes[self.row_dim], sizes[self.col_dim])

    @property
    def is_square(self) -> bool:
        return self.matrix_shape[0] == self.matrix_shape[1]

    def to_numpy(self) -> np.ndarray:
        return self.data.to_numpy()

    def at(self, row: Any | None = None, col: Any | None = None) -> xr.DataArray:
        if row is None and col is None:
            raise ValueError("row and column cannot both be None")
        indexer = {}
        if row is not None:
            indexer[self.row_dim] = row
        if col is not None:
            indexer[self.col_dim] = col
        return self.data.sel(indexer) if indexer else self.data

    def diagonal(self, offset: int = 0, out_dim: Hashable = "diagonal") -> xr.DataArray:
        return fishy.linalg.diagonal(self.data, offset=offset, out_dim=out_dim)

    def det(self) -> xr.DataArray:
        return fishy.linalg.det(self.data)

    def inv(self) -> "Matrix":
        return Matrix(fishy.linalg.inv(self.data))

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if self.col_dim != other.row_dim:
            raise ValueError(
                f"Incompatible dimensions: {self.col_dim} != {other.row_dim}"
            )
        if self.matrix_shape[1] != other.matrix_shape[0]:
            raise ValueError(
                f"Incompatible shapes: {self.matrix_shape} @ {other.matrix_shape}"
            )
        result_data = xr.apply_ufunc(
            np.matmul,
            self.data,
            other.data,
            input_core_dims=[self.matrix_dims, other.matrix_dims],
            output_core_dims=[[self.row_dim, other.col_dim]],
        )
        return Matrix(result_data)
