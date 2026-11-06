"""Construction of labeled Fisher matrix Datasets with parameter metadata."""

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .arrays import _labels, _plot_labels, _validate_matrix, matrix, vector
from .inversion import _eigenvalues, _positive_semidefinite


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


def _stored_covariance(
    forecast: xr.Dataset, parameters: Sequence[str] | None
) -> tuple[xr.DataArray, list[str], list[str]]:
    """Validate a Dataset's stored covariance and select parameters.

    Returns the validated covariance, its parameter names, and the selected
    names in display order. Raises ``TypeError``, ``ValueError``, ``KeyError``
    or ``numpy.linalg.LinAlgError`` for invalid input.
    """
    if not isinstance(forecast, xr.Dataset):
        raise TypeError("Plot inputs must be xarray.Dataset objects.")
    if "covariance" not in forecast.data_vars:
        raise ValueError("Dataset must contain a 'covariance' variable.")
    covariance, names = _validate_matrix(forecast["covariance"])
    if not _positive_semidefinite(_eigenvalues(covariance)):
        raise np.linalg.LinAlgError(
            "Correlation requires a positive semidefinite covariance."
        )
    if not np.all(np.diag(covariance.values) > 0):
        raise np.linalg.LinAlgError("Correlation requires positive variances.")
    selected = names if parameters is None else _labels(parameters, name="parameters")
    for name in selected:
        if name not in names:
            raise KeyError(name)
    return covariance, names, selected


def _display_labels(
    forecast: xr.Dataset,
    names: Sequence[str],
    selected: Sequence[str],
    parameter_labels: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve LaTeX display labels from Dataset ``labels`` and overrides.

    Parameters without a label are absent from the result.
    """
    display: dict[str, str] = {}
    if "labels" in forecast.data_vars:
        labels = forecast["labels"]
        if labels.dims != ("row",):
            raise ValueError("Plot label dimensions must be exactly ('row',).")
        display.update(zip(names, _plot_labels(labels.values.tolist()), strict=True))
    if parameter_labels is not None:
        if not isinstance(parameter_labels, Mapping):
            raise TypeError("Parameter labels must be a mapping of names to strings.")
        if any(not isinstance(name, str) for name in parameter_labels):
            raise ValueError("Parameter label keys must be strings.")
        _plot_labels(list(parameter_labels.values()))
        display.update(parameter_labels)
    _plot_labels([display.get(name, name) for name in selected])
    return display
