"""Construction of labeled dense arrays and shared validation."""

from collections.abc import Mapping, Sequence

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


def _plot_labels(labels: Sequence[str]) -> list[str]:
    """Validate unique plot labels given as LaTeX math without enclosing ``$``."""
    validated = _labels(labels, name="labels")
    if any("$" in label for label in validated):
        raise ValueError("labels must not include '$'; they are rendered as math.")
    return validated


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


def _validate_shape(F: xr.DataArray) -> list[str]:
    """Validate canonical dimensions and coordinates, ignoring symmetry."""
    if not isinstance(F, xr.DataArray):
        raise TypeError("Matrix inputs must be xarray.DataArray objects.")
    if F.dims != ("row", "col"):
        raise ValueError("Matrix dimensions must be exactly ('row', 'col').")
    if F.shape[0] == 0 or F.shape[0] != F.shape[1]:
        raise ValueError("Matrices must be nonempty and square.")
    parameters = _coordinate_labels(F, "row")
    if parameters != _coordinate_labels(F, "col"):
        raise ValueError("Row and column parameter coordinates must match in order.")
    return parameters


def _validate_matrix(F: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Validate a canonical matrix without checking positive definiteness."""
    parameters = _validate_shape(F)
    values = _real_values(F.values)
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Matrices must be symmetric.")
    return _new_matrix(values, parameters), parameters


def _new_matrix(values: ArrayLike, parameters: Sequence[str]) -> xr.DataArray:
    """Create a fresh matrix with only canonical coordinates."""
    labels = list(parameters)
    return xr.DataArray(
        _real_values(values).copy(),
        dims=("row", "col"),
        coords={"row": labels, "col": labels.copy()},
    )


def _new_vector(values: ArrayLike, parameters: Sequence[str]) -> xr.DataArray:
    """Create a fresh vector with only canonical coordinates."""
    labels = list(parameters)
    return xr.DataArray(
        np.asarray(values).copy(),
        dims=("row",),
        coords={"row": labels},
    )


def _symmetrize(A: xr.DataArray) -> xr.DataArray:
    """Return the fresh canonical matrix ``(A + A.T) / 2`` with swapped labels."""
    swapped = A.rename({"row": "col", "col": "row"})
    symmetric = ((A + swapped) / 2).transpose("row", "col")
    return _new_matrix(symmetric.values, symmetric.row.values.tolist())


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


def vector(values: ArrayLike, parameters: Sequence[str]) -> xr.DataArray:
    """Construct a labeled, dense one-dimensional array.

    Parameters
    ----------
    values : array_like
        Nonempty one-dimensional array with one value per parameter.
        Values retain their input dtype.
    parameters : sequence of str
        Unique parameter names, in vector order.

    Returns
    -------
    xarray.DataArray
        Fresh array with dimension ``('row',)`` and an explicit ``row``
        coordinate.

    Raises
    ------
    ValueError
        If the values are not one-dimensional and nonempty, or if the
        parameter names are invalid or do not match the value count.
    """
    labels = _labels(parameters, name="parameters")
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != len(labels):
        raise ValueError(
            "Values must be one-dimensional and match the parameter count."
        )
    return _new_vector(array, labels)


def gaussian_prior(sigmas: Mapping[str, float]) -> xr.DataArray:
    """Construct independent Gaussian prior information from standard deviations.

    Parameters
    ----------
    sigmas : mapping of str to float
        Nonempty mapping of unique parameter names to finite, positive
        standard deviations. Mapping order determines matrix order.

    Returns
    -------
    xarray.DataArray
        Fresh diagonal matrix with entries ``1 / sigma**2``.

    Raises
    ------
    ValueError
        If names or standard deviations are invalid, or information cannot
        be represented as finite float64 values.
    """
    labels = _labels(list(sigmas), name="sigmas")
    deviations = _real_values(list(sigmas.values()))
    if np.any(deviations <= 0):
        raise ValueError("Standard deviations must be positive.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        information = np.square(1 / deviations)
    return _new_matrix(np.diag(information), labels)
