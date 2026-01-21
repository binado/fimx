from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeAlias

MatrixDims: TypeAlias = tuple[Hashable, Hashable]
ParameterDims: TypeAlias = MatrixDims


@dataclass(frozen=True)
class DatasetDims:
    """Dimension names for Fisher matrix xarray DataArrays.

    Attributes
    ----------
    row : Hashable
        Row dimension name for Fisher matrices
    col : Hashable
        Column dimension name for Fisher matrices
    parameter : Hashable
        Dimension name for parameter metadata vectors
    batch : Hashable
        Batch dimension name
    """

    row: Hashable = "row"
    col: Hashable = "col"
    parameter: Hashable = "parameter"
    batch: Hashable = "batch"

    @property
    def matrix_dims(self) -> MatrixDims:
        return (self.row, self.col)

    @property
    def parameter_dims(self) -> ParameterDims:
        return self.matrix_dims
