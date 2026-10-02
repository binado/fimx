"""Construction and shared validation of labeled dense matrices."""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


def _labels(parameters: Sequence[str], *, name: str) -> list[str]:
    """Validate a nonempty sequence of unique string labels."""
    if isinstance(parameters, str):
        # A bare string is an invalid label sequence, even for one parameter.
        raise ValueError(f"{name} must be a sequence of parameter names.")  # noqa: TRY004
    labels = list(parameters)
    if not labels or any(not isinstance(label, str) for label in labels):
        raise ValueError(f"{name} must be a nonempty sequence of unique string labels.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique string labels.")
    return labels


def _real_values(values: ArrayLike) -> NDArray[np.float64]:
    """Convert finite, real numeric data to float64."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError("Values must be real numeric data.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Values must be finite.")
    return result


def _coordinate_labels(array: xr.DataArray, dimension: str) -> list[str]:
    """Read an explicit one-dimensional parameter coordinate."""
    if dimension not in array.coords or array.coords[dimension].dims != (dimension,):
        raise ValueError(f"An explicit coordinate for {dimension!r} is required.")
    return _labels(array.coords[dimension].values.tolist(), name=dimension)


def _validate_matrix(F: xr.DataArray) -> tuple[NDArray[np.float64], list[str]]:
    """Validate a canonical matrix without checking positive definiteness."""
    if not isinstance(F, xr.DataArray):
        raise TypeError("Matrix inputs must be xarray.DataArray objects.")
    if F.dims != ("row", "col"):
        raise ValueError("Matrix dimensions must be exactly ('row', 'col').")
    if F.shape[0] == 0 or F.shape[0] != F.shape[1]:
        raise ValueError("Matrices must be nonempty and square.")
    parameters = _coordinate_labels(F, "row")
    if parameters != _coordinate_labels(F, "col"):
        raise ValueError("Row and column parameter coordinates must match in order.")
    values = _real_values(F.values)
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Matrices must be symmetric.")
    return values, parameters


def _new_matrix(values: ArrayLike, parameters: Sequence[str]) -> xr.DataArray:
    """Create a fresh matrix with only canonical coordinates."""
    labels = list(parameters)
    return xr.DataArray(
        _real_values(values).copy(),
        dims=("row", "col"),
        coords={"row": labels, "col": labels.copy()},
    )


def matrix(values: ArrayLike, parameters: Sequence[str]) -> xr.DataArray:
    """Construct a labeled, symmetric dense matrix.

    Parameters
    ----------
    values : array_like
        Nonempty square array of finite, real numeric values.
    parameters : sequence of str
        Unique parameter names, in matrix order.

    Returns
    -------
    xarray.DataArray
        Fresh float64 matrix with dimensions ``('row', 'col')``.

    Raises
    ------
    ValueError
        If the values or parameter names violate the matrix contract.

    Notes
    -----
    Symmetry uses ``rtol=1e-10`` and ``atol=1e-12``. Positive definiteness
    is checked only by operations requiring a Cholesky solve.
    """
    labels = _labels(parameters, name="parameters")
    array = _real_values(values)
    if array.ndim != 2 or array.shape != (len(labels), len(labels)):
        raise ValueError("Values must be square and match the parameter count.")
    result = _new_matrix(array, labels)
    _validate_matrix(result)
    return result
