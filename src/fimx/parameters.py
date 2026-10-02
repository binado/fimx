"""Parameter addition, removal and changes of variables."""

from collections.abc import Sequence

import xarray as xr
from xarray_einstats import linalg

from .construction import (
    _coordinate_labels,
    _labels,
    _new_matrix,
    _real_values,
    _symmetrize,
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


def expand(F: xr.DataArray, parameters: Sequence[str]) -> xr.DataArray:
    """Embed a matrix in a larger parameter set, filling new entries with zeros.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    parameters : sequence of str
        Unique target names, in the desired output order. Must contain every
        name in ``F``; the order may differ from that of ``F``.

    Returns
    -------
    xarray.DataArray
        Fresh matrix over ``parameters``. Entries involving names absent from
        ``F`` are zero, so matrices expanded to the same ``parameters`` can be
        added directly with ``+``.

    Raises
    ------
    ValueError
        If input is malformed, ``parameters`` is invalid, or a name in ``F``
        is missing from ``parameters``.
    """
    values, labels = _validate_matrix(F)
    target = _labels(parameters, name="parameters")
    missing = [name for name in labels if name not in target]
    if missing:
        raise ValueError(f"parameters is missing names present in F: {missing}.")
    filled = values.reindex(row=target, col=target, fill_value=0)
    return _new_matrix(filled.values, target)


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
    retained = [labels[i] for i in keep]
    selected = values.sel(row=retained, col=retained)
    return _new_matrix(selected.values, retained)


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
    retained = [labels[i] for i in keep]
    removed = [labels[i] for i in drop]
    result = values.sel(row=retained, col=retained)
    if drop:
        cross = values.sel(row=retained, col=removed)
        rhs = cross.transpose("col", "row").rename(row="rhs_col").rename(col="row")
        solved = _solve(values.sel(row=removed, col=removed), rhs).rename(
            row="drop_parameter", col="rhs_col"
        )
        product = linalg.matmul(
            cross,
            solved,
            dims=(("row", "col"), ("drop_parameter", "rhs_col")),
        ).rename(rhs_col="col")
        result = result - product
    return _symmetrize(result)


def transform(F: xr.DataArray, jacobian: xr.DataArray) -> xr.DataArray:
    """Change variables using a labeled Jacobian of old with respect to new.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    jacobian : xarray.DataArray
        Finite real array with dimensions ``('old', 'new')``
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
    if jacobian.dims != ("old", "new"):
        raise ValueError("Jacobian dimensions must be ('old', 'new').")
    old = _coordinate_labels(jacobian, "old")
    new = _coordinate_labels(jacobian, "new")
    if set(old) != set(labels):
        raise ValueError(
            "Jacobian old labels must match the complete matrix parameters."
        )
    J = xr.DataArray(
        _real_values(jacobian.sel(old=labels).values),
        dims=("old", "new"),
        coords={"old": labels, "new": new},
    ).rename(old="col")
    right = linalg.matmul(values, J, dims=("row", "col", "new"))
    left = J.transpose("new", "col")
    right = right.rename(row="old", new="new_right")
    result = linalg.matmul(
        left,
        right,
        dims=(
            ("new", "col"),
            ("old", "new_right"),
        ),
    ).rename(new="row", new_right="col")
    return _symmetrize(result)
