from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import xarray as xr

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

    @classmethod
    def infer_from_data(
        cls,
        data: xr.DataArray,
        parameter_coord: Hashable,
        *,
        parameter_dim: Hashable | None = None,
        batch_dim: Hashable | None = None,
    ) -> "DatasetDims":
        """Infer matrix dimension names from a parameter coordinate.

        Parameters
        ----------
        data : xr.DataArray
            DataArray containing the Fisher matrix values.
        parameter_coord : Hashable
            Coordinate name that stores the parameter labels.
        parameter_dim : Hashable | None
            Dimension name for parameter metadata. If None, uses the default.
        batch_dim : Hashable | None
            Dimension name for the stacked batch axis. If None, uses the default.

        Returns
        -------
        DatasetDims
            Inferred dimension names.
        """
        if parameter_coord not in data.coords:
            raise ValueError(f"Missing coordinate '{parameter_coord}'.")
        coord = data.coords[parameter_coord]
        candidate_dims = [dim for dim in coord.dims if dim in data.dims]
        coord_values = np.asarray(coord.values)
        for dim in data.dims:
            if dim not in data.coords:
                continue
            values = np.asarray(data.coords[dim].values)
            if values.shape == coord_values.shape and np.array_equal(
                values, coord_values
            ):
                candidate_dims.append(dim)
        seen: set[Hashable] = set()
        ordered = []
        for dim in data.dims:
            if dim in candidate_dims and dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        if len(ordered) < 2:
            raise ValueError(
                "Unable to infer row/col dims from "
                f"coordinate '{parameter_coord}'. Provide dims explicitly."
            )
        defaults = cls()
        return cls(
            row=ordered[0],
            col=ordered[1],
            parameter=parameter_dim
            if parameter_dim is not None
            else defaults.parameter,
            batch=batch_dim if batch_dim is not None else defaults.batch,
        )
