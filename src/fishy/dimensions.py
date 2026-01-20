from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeAlias

ParameterDims: TypeAlias = tuple[Hashable, Hashable]


@dataclass(frozen=True)
class DatasetDims:
    """Dimension names for Fisher matrix xarray DataArrays.

    Attributes
    ----------
    parameter_i : Hashable
        First parameter dimension name (rows)
    parameter_j : Hashable
        Second parameter dimension name (columns)
    batch : Hashable
        Batch dimension name
    """

    parameter_i: Hashable = "parameter_i"
    parameter_j: Hashable = "parameter_j"
    batch: Hashable = "batch"

    @property
    def parameter_dims(self) -> ParameterDims:
        return (self.parameter_i, self.parameter_j)
