"""Construction of labeled Fisher matrix Datasets with parameter metadata."""

from collections.abc import Hashable, Mapping

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .arrays import _validate_matrix, matrix, vector


def dataset(
    F: xr.DataArray, arrays: Mapping[Hashable, ArrayLike | xr.DataArray]
) -> xr.Dataset:
    """Combine a Fisher matrix and arrays on shared parameter coordinates.

    Parameters
    ----------
    F : xarray.DataArray
        Canonical matrix with dimensions ``('row', 'col')``.
    arrays : mapping of hashable to array_like or xarray.DataArray
        Additional variables. One-dimensional arrays use ``row``; two-
        dimensional arrays use ``('row', 'col')``. Plain arrays follow matrix
        order. DataArray dimensions are renamed by position and indexed axes
        are aligned by label. Unindexed axes follow matrix order. Names
        ``fisher``, ``row``, and ``col`` are reserved.

    Returns
    -------
    xarray.Dataset
        Independent variables with shared matrix coordinates. Additional
        arrays retain their dtypes; positive definiteness is not required.

    Raises
    ------
    TypeError
        If F is not a DataArray or arrays is not a mapping.
    ValueError
        If the matrix is invalid, names are reserved, arrays are not one- or
        two-dimensional, or xarray rejects incompatible sizes or coordinates.

    Notes
    -----
    Shape, coordinate, and merge compatibility are checked by xarray.
    Metadata need not be numeric or finite. Plotting validates fiducials.
    """
    validated_fisher, labels = _validate_matrix(F)
    fisher = matrix(validated_fisher.values, labels)
    if not isinstance(arrays, Mapping):
        raise TypeError("Arrays must be a mapping of names to arrays.")
    variables: dict[Hashable, xr.DataArray] = {"fisher": fisher}
    dimensions = {1: ("row",), 2: ("row", "col")}
    ordered = fisher.sortby(["row", "col"])
    for name, array in arrays.items():
        if name in {"fisher", "row", "col"}:
            raise ValueError(f"Variable name {name!r} is reserved.")
        variable = (
            array
            if isinstance(array, xr.DataArray)
            else xr.DataArray(np.asarray(array))
        )
        if variable.ndim not in dimensions:
            raise ValueError("Additional arrays must be one- or two-dimensional.")
        dims = dimensions[variable.ndim]
        variable = variable.rename(dict(zip(variable.dims, dims, strict=True)))
        variable = variable.assign_coords(
            {dim: fisher.coords[dim] for dim in dims if dim not in variable.indexes}
        )
        # Exact alignment on sorted coordinates permits reordered labels while
        # letting xarray reject missing, extra, or duplicate labels and sizes.
        xr.align(ordered, variable.sortby(list(dims)), join="exact")
        variable = variable.sel({dim: fisher.coords[dim] for dim in dims})
        if variable.ndim == 1:
            canonical = vector(variable.values, labels)
            canonical.attrs = variable.attrs.copy()
            variable = canonical
        variables[name] = variable
    return xr.Dataset(variables).copy(deep=True)
