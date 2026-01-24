from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Diagnostics:
    """Diagnostic information for Fisher matrices.

    Attributes
    ----------
    min_eigenvalue : xr.DataArray
        Minimum eigenvalues
    max_eigenvalue : xr.DataArray
        Maximum eigenvalues
    condition_number : xr.DataArray
        Condition numbers
    is_degenerate : xr.DataArray
        Boolean flags indicating degeneracy
    method : str
        Method used for computation ('eig' or 'svd')
    threshold : float | None
        Threshold used for degeneracy detection
    """

    min_eigenvalue: xr.DataArray
    max_eigenvalue: xr.DataArray
    condition_number: xr.DataArray
    is_degenerate: xr.DataArray
    method: str
    threshold: float | None


def min_eigenvalues(values: np.ndarray) -> np.ndarray:
    """Compute minimum eigenvalues of matrices.

    Works on both single (n, n) and batched (batch, n, n) arrays.

    Parameters
    ----------
    values : np.ndarray
        Matrix or batch of matrices

    Returns
    -------
    np.ndarray
        Minimum eigenvalue for each matrix. Shape () or (batch,).
    """
    eigvals = np.linalg.eigvalsh(values)
    return np.min(eigvals, axis=-1)


def min_max_eigenvalues(
    values: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute minimum and maximum eigenvalues of matrices.

    Parameters
    ----------
    values : np.ndarray
        Matrix or batch of matrices
    method : str
        Method to use: 'eig' for eigenvalues, 'svd' for singular values

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (min_values, max_values) where each is shape () or (batch,)

    Raises
    ------
    ValueError
        If method is not 'eig' or 'svd'
    """
    # Method dispatch dictionary
    _METHODS = {
        "eig": lambda v: _min_max_eigvals(v),
        "svd": lambda v: _min_max_svals(v),
    }

    if method not in _METHODS:
        raise ValueError(f"method must be 'eig' or 'svd', got: {method}")

    return _METHODS[method](values)


def _min_max_eigvals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute min/max eigenvalues."""
    eigvals = np.linalg.eigvalsh(values)
    return np.min(eigvals, axis=-1), np.max(eigvals, axis=-1)


def _min_max_svals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute min/max singular values."""
    svals = np.linalg.svd(values, compute_uv=False)
    return np.min(svals, axis=-1), np.max(svals, axis=-1)


def condition_numbers(values: np.ndarray, method: str) -> np.ndarray:
    """Compute condition numbers of matrices.

    Parameters
    ----------
    values : np.ndarray
        Matrix or batch of matrices
    method : str
        Method to use: 'eig' for eigenvalue-based, 'svd' for singular value-based

    Returns
    -------
    np.ndarray
        Condition number for each matrix. Shape () or (batch,).

    Raises
    ------
    ValueError
        If method is not 'eig' or 'svd'
    """
    # Method dispatch dictionary
    _METHODS = {
        "eig": lambda v: _condition_number_eig(v),
        "svd": lambda v: _condition_number_svd(v),
    }

    if method not in _METHODS:
        raise ValueError(f"method must be 'eig' or 'svd', got: {method}")

    return _METHODS[method](values)


def _condition_number_eig(values: np.ndarray) -> np.ndarray:
    """Compute condition number using eigenvalues."""
    eigvals = np.linalg.eigvalsh(values)
    return np.max(eigvals, axis=-1) / np.min(eigvals, axis=-1)


def _condition_number_svd(values: np.ndarray) -> np.ndarray:
    """Compute condition number using singular values."""
    svals = np.linalg.svd(values, compute_uv=False)
    return np.max(svals, axis=-1) / np.min(svals, axis=-1)
