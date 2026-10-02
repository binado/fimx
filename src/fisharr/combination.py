"""Combination of independent information and Gaussian priors."""

from collections.abc import Mapping

import numpy as np
import xarray as xr

from .construction import _labels, _new_matrix, _real_values, _validate_matrix


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
    validated = [_validate_matrix(F) for F in matrices]
    union = list(dict.fromkeys(label for _, labels in validated for label in labels))
    result = np.zeros((len(union), len(union)), dtype=np.float64)
    for values, labels in validated:
        aligned = _new_matrix(values, labels).reindex(
            row=union, col=union, fill_value=0
        )
        result += aligned.values
    return _new_matrix(result, union)


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
