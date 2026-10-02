"""Parameter removal and changes of variables."""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .construction import (
    _coordinate_labels,
    _new_matrix,
    _real_values,
    _validate_matrix,
)
from .inversion import _solve


def _selection(
    labels: list[str], parameters: str | Sequence[str]
) -> tuple[list[int], list[int]]:
    """Resolve parameter removal while preserving retained input order."""
    selected = [parameters] if isinstance(parameters, str) else list(parameters)
    if any(not isinstance(name, str) for name in selected):
        raise ValueError("Selected parameters must be strings.")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected parameters must be unique.")
    for name in selected:
        if name not in labels:
            raise KeyError(name)
    if len(selected) == len(labels):
        raise ValueError("Cannot remove all parameters.")
    removed = set(selected)
    return (
        [i for i, name in enumerate(labels) if name not in removed],
        [i for i, name in enumerate(labels) if name in removed],
    )


def fix(F: xr.DataArray, parameters: str | Sequence[str]) -> xr.DataArray:
    """Fix selected parameters by keeping the remaining principal submatrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    parameters : str or sequence of str
        Names to remove. An empty sequence returns a fresh copy.

    Returns
    -------
    xarray.DataArray
        Fresh principal submatrix in retained input order.

    Raises
    ------
    KeyError
        If a selected name is unknown.
    ValueError
        If input is malformed, names are duplicated, or all are removed.
    """
    values, labels = _validate_matrix(F)
    keep, _ = _selection(labels, parameters)
    return _new_matrix(values[np.ix_(keep, keep)], [labels[i] for i in keep])


def marginalize(F: xr.DataArray, parameters: str | Sequence[str]) -> xr.DataArray:
    """Remove nuisance parameters through the Schur complement.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    parameters : str or sequence of str
        Names to remove. An empty sequence returns a fresh copy.

    Returns
    -------
    xarray.DataArray
        Fresh marginalized matrix in retained input order.

    Raises
    ------
    KeyError
        If a selected name is unknown.
    ValueError
        If input is malformed, names are duplicated, or all are removed.
    numpy.linalg.LinAlgError
        If the removed block is singular or not positive definite.

    Notes
    -----
    For retained block A, cross block B and removed block D, compute
    ``A - B @ solve(D, B.T)``. Only D must be positive definite.
    """
    values, labels = _validate_matrix(F)
    keep, drop = _selection(labels, parameters)
    result = values[np.ix_(keep, keep)]
    if drop:
        cross = values[np.ix_(keep, drop)]
        result = result - cross @ _solve(values[np.ix_(drop, drop)], cross.T)
        result = result / 2 + result.T / 2
    return _new_matrix(result, [labels[i] for i in keep])


def transform(F: xr.DataArray, jacobian: xr.DataArray) -> xr.DataArray:
    """Change variables using a labeled Jacobian of old with respect to new.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    jacobian : xarray.DataArray
        Finite real array with dimensions ``('old_parameter', 'new_parameter')``
        and unique string coordinates. Entries are
        ``J[i, j] = d theta_i / d phi_j``. Old labels must match all F labels;
        rows are reordered to F. New labels must be nonempty.

    Returns
    -------
    xarray.DataArray
        Fresh symmetric matrix ``J.T @ F @ J`` in new-parameter order.

    Raises
    ------
    ValueError
        If the matrix or Jacobian contract is violated.
    TypeError
        If the Jacobian is not a DataArray.
    """
    values, labels = _validate_matrix(F)
    if not isinstance(jacobian, xr.DataArray):
        raise TypeError("The Jacobian must be an xarray.DataArray.")
    if jacobian.dims != ("old_parameter", "new_parameter"):
        raise ValueError(
            "Jacobian dimensions must be ('old_parameter', 'new_parameter')."
        )
    old = _coordinate_labels(jacobian, "old_parameter")
    new = _coordinate_labels(jacobian, "new_parameter")
    if set(old) != set(labels):
        raise ValueError(
            "Jacobian old labels must match the complete matrix parameters."
        )
    J = _real_values(jacobian.values)[[old.index(label) for label in labels], :]
    result = J.T @ values @ J
    return _new_matrix(result / 2 + result.T / 2, new)
