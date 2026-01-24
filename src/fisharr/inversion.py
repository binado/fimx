from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .exceptions import FailedInversionError


def invert_matrices(
    values: npt.NDArray, *, method: str, rcond: float | None = None
) -> npt.NDArray:
    """Invert Fisher matrices using specified method.

    Supports batched operations - works on both 2D (n, n) and 3D (batch, n, n) arrays.

    Parameters
    ----------
    values : npt.NDArray
        Matrix or batch of matrices to invert. Shape (n, n) or (batch, n, n).
    method : str
        Inversion method: 'inv', 'pinv', 'svd', or 'cholesky'
    rcond : float | None
        Cutoff for small singular values (used by 'pinv' and 'svd' methods)

    Returns
    -------
    npt.NDArray
        Inverted matrix or batch of matrices with same shape as input

    Raises
    ------
    ValueError
        If method is unknown
    FailedInversionError
        If matrix inversion fails (e.g., not positive definite for Cholesky)
    """
    if method == "cholesky":
        # Cholesky needs special handling for error reporting
        return _invert_cholesky_batched(values)

    # Method dispatch dictionary
    _METHODS = {
        "inv": lambda v: np.linalg.inv(v),
        "pinv": lambda v: np.linalg.pinv(v, rcond=rcond),
        "svd": lambda v: _svd_inverse(v, rcond),
    }

    if method not in _METHODS:
        raise ValueError(
            f"Unknown inversion method: {method}. "
            f"Choose from: {list(_METHODS.keys()) + ['cholesky']}"
        )

    return _METHODS[method](values)


def _invert_cholesky_batched(values: npt.NDArray) -> npt.NDArray:
    """Invert using Cholesky decomposition, handles batches.

    Parameters
    ----------
    values : npt.NDArray
        Positive definite matrix or batch of matrices

    Returns
    -------
    npt.NDArray
        Inverted matrix or batch of matrices

    Raises
    ------
    FailedInversionError
        If matrix is not positive definite
    """
    # Handle single matrix case
    if values.ndim == 2:
        return _invert_cholesky(values)

    # For batched case, process each matrix individually to provide
    # better error messages (which specific matrix in the batch failed)
    try:
        return np.stack([_invert_cholesky(mat) for mat in values], axis=0)
    except FailedInversionError:
        # Re-raise with original diagnostics
        raise


def _invert_cholesky(mat: npt.NDArray) -> npt.NDArray:
    """Invert a single positive definite matrix using Cholesky decomposition.

    Parameters
    ----------
    mat : npt.NDArray
        Positive definite matrix of shape (n, n)

    Returns
    -------
    npt.NDArray
        Inverted matrix of shape (n, n)

    Raises
    ------
    FailedInversionError
        If matrix is not positive definite
    """
    try:
        identity = np.eye(mat.shape[0], dtype=mat.dtype)
        chol = np.linalg.cholesky(mat)
        y = np.linalg.solve(chol, identity)
        return np.linalg.solve(chol.T, y)
    except np.linalg.LinAlgError as e:
        raise FailedInversionError.from_matrix(mat, "cholesky", e) from e


def _svd_inverse(values: npt.NDArray, rcond: float | None) -> npt.NDArray:
    """Compute pseudoinverse using SVD.

    Handles both single and batched matrices using numpy's vectorized operations.

    Parameters
    ----------
    values : npt.NDArray
        Matrix or batch of matrices. Shape (n, n) or (batch, n, n).
    rcond : float | None
        Cutoff for small singular values. If None, uses machine precision.

    Returns
    -------
    npt.NDArray
        Pseudoinverse with same shape as input
    """
    # Add batch dimension if needed
    original_ndim = values.ndim
    if original_ndim == 2:
        values = values[np.newaxis, ...]

    # SVD works on batches naturally
    u, s, vh = np.linalg.svd(values)

    # Compute cutoff and apply to singular values
    cutoff = (rcond or 0.0) * s.max(axis=-1, keepdims=True)
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)

    # Compute inverse: vh.T @ diag(s_inv) @ u.T
    # s_inv is 1D per batch, broadcasting multiplies each row of vh.T by corresponding element
    result = np.einsum("...ij,...j,...kj->...ik", vh.swapaxes(-2, -1), s_inv, u)

    # Remove batch dimension if input was 2D
    if original_ndim == 2:
        result = result[0]

    return result
