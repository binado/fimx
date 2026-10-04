"""Symmetrization, combination and parameter manipulation of Fisher matrices."""

import operator
from collections.abc import Sequence
from functools import reduce
from typing import cast

import numpy as np
import xarray as xr
from xarray_einstats import linalg

from .arrays import (
    _coordinate_labels,
    _labels,
    _new_matrix,
    _real_values,
    _symmetrize,
    _validate_matrix,
    _validate_shape,
)
from .inversion import (
    InversionMethod,
    _eigenvalues,
    _invert,
    _positive_semidefinite,
    _require_method,
    _solve,
)

__all__ = [
    "combine",
    "correlation",
    "expand",
    "fix",
    "fom",
    "marginalize",
    "symmetrize",
    "transform",
]


def combine(*matrices: xr.DataArray) -> xr.DataArray:
    """Add independent information over the union of parameter labels.

    Parameters
    ----------
    *matrices : xarray.DataArray
        One or more matrices satisfying the canonical matrix contract.

    Returns
    -------
    xarray.DataArray
        Fresh sum. Keep the first input order and append newly encountered
        labels in subsequent input order; absent entries contribute zero.

    Raises
    ------
    ValueError
        If no matrices are supplied or any matrix is malformed.
    """
    if not matrices:
        raise ValueError("At least one matrix is required.")
    labels = [list(_validate_matrix(F)[1]) for F in matrices]
    union = list(dict.fromkeys(label for group in labels for label in group))
    total = reduce(operator.add, (expand(F, union) for F in matrices))
    return _new_matrix(total.values, union)


def symmetrize(F: xr.DataArray) -> xr.DataArray:
    """Return the symmetric part of a labeled matrix as a fresh canonical matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Square labeled matrix with dimensions ``('row', 'col')``, matching
        ordered parameter coordinates and finite real values. Symmetry is
        not required.

    Returns
    -------
    xarray.DataArray
        Fresh symmetric matrix ``(F + F.T) / 2`` over the same parameters.

    Raises
    ------
    TypeError
        If the input is not an xarray.DataArray.
    ValueError
        If dimensions, coordinates or values violate the matrix contract.
    """
    _validate_shape(F)
    return _symmetrize(F)


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


def _kept(labels: list[str], parameters: str | Sequence[str]) -> list[str]:
    """Validate the parameters retained by a figure of merit."""
    selected = [parameters] if isinstance(parameters, str) else list(parameters)
    if any(not isinstance(name, str) for name in selected):
        raise ValueError("Selected parameters must be strings.")
    if not selected:
        raise ValueError("Selected parameters must be nonempty.")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected parameters must be unique.")
    for name in selected:
        if name not in labels:
            raise KeyError(name)
    return selected


def correlation(
    F: xr.DataArray, *, method: InversionMethod = "cholesky"
) -> xr.DataArray:
    """Return the correlation matrix of a Fisher matrix.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    method : {'cholesky', 'inv', 'pinv'}
        Inversion algorithm used to obtain the covariance; see :func:`inv`.
        ``'pinv'`` supports singular positive semidefinite matrices when all
        covariance diagonal entries are positive.

    Returns
    -------
    xarray.DataArray
        Fresh canonical matrix ``C_ij / (sigma_i sigma_j)``, with ones on
        the diagonal.

    Raises
    ------
    numpy.linalg.LinAlgError
        If inversion fails, the input or covariance is not positive
        semidefinite within numerical roundoff, or a covariance diagonal entry
        is not positive. A zero variance can occur with ``method='pinv'``.
    ValueError
        If the matrix contract is violated or ``method`` is unknown.

    Notes
    -----
    Semidefiniteness uses the scale-relative eigenvalue roundoff allowance
    described in :func:`diagnose`, without modifying the spectrum. Singular
    pseudoinverse correlations describe normalized pseudoinverse entries,
    not unconstrained uncertainties.
    """
    _require_method(method)
    values, parameters = _validate_matrix(F)
    if not _positive_semidefinite(_eigenvalues(values)):
        raise np.linalg.LinAlgError("Correlation requires positive semidefinite input.")
    covariance = _invert(values, parameters, method)
    if not _positive_semidefinite(_eigenvalues(covariance)):
        raise np.linalg.LinAlgError(
            "Correlation requires a positive semidefinite covariance."
        )
    variance = linalg.diagonal(covariance, dims=("row", "col"))
    if not bool((variance > 0).all().item()):
        raise np.linalg.LinAlgError("Correlation requires positive variances.")
    sigma = cast(xr.DataArray, np.sqrt(variance))
    normalized = covariance / sigma / sigma.rename(row="col")
    normalized = xr.where(covariance.row == covariance.col, 1.0, normalized)
    return _symmetrize(normalized.transpose("row", "col"))


def fom(F: xr.DataArray, parameters: str | Sequence[str] | None = None) -> float:
    """Return the Dark Energy Task Force figure of merit.

    Parameters
    ----------
    F : xarray.DataArray
        Matrix satisfying the canonical matrix contract.
    parameters : str or sequence of str, optional
        Parameters kept in the figure of merit. Every other parameter is
        marginalized. ``None`` keeps the whole matrix.

    Returns
    -------
    float
        ``sqrt(det F_subset)``, equal to ``1 / sqrt(det C_subset)`` for the
        marginalized covariance of the kept parameters.

    Raises
    ------
    KeyError
        If a selected name is unknown.
    ValueError
        If input is malformed, or the selection is empty, duplicated, or not
        made of strings.
    numpy.linalg.LinAlgError
        If the selected marginalized matrix, or the removed block, is not
        positive definite.

    Notes
    -----
    The figure of merit is computed from the Cholesky diagonal without
    forming the determinant. The determinant does not depend on parameter
    order. Passing every parameter keeps the original matrix, because
    :func:`marginalize` refuses to remove all of them.
    """
    matrix, labels = _validate_matrix(F)
    if parameters is not None:
        selected = _kept(labels, parameters)
        complement = [name for name in labels if name not in set(selected)]
        if complement:
            matrix = marginalize(matrix, complement)
    factor = linalg.cholesky(matrix, dims=("row", "col"))
    diagonal = linalg.diagonal(factor, dims=("row", "col"))
    return float(np.exp(np.log(diagonal).sum()).item())


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
