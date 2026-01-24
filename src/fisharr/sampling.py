from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .exceptions import FailedInversionError


def sample_from_fisher(
    values: npt.NDArray, num_samples: int, rng: np.random.Generator
) -> npt.NDArray:
    """Sample parameter values from Fisher matrix using Cholesky decomposition.

    Works on both single (n, n) and batched (batch, n, n) Fisher matrices.

    Parameters
    ----------
    values : npt.NDArray
        Fisher matrix or batch of Fisher matrices. Must be positive definite.
        Shape (n, n) or (batch, n, n).
    num_samples : int
        Number of samples to generate
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    npt.NDArray
        Samples from multivariate Gaussian with covariance = Fisher^-1.
        Shape (num_samples, n) for single matrix or (batch, num_samples, n) for batched.

    Raises
    ------
    FailedInversionError
        If matrix is not positive definite
    """
    original_ndim = values.ndim

    # Ensure 3D for uniform processing
    if original_ndim == 2:
        values = values[np.newaxis, ...]

    try:
        # Cholesky decomposition works on batches naturally
        chol = np.linalg.cholesky(values)
    except np.linalg.LinAlgError as e:
        # For single matrix, provide detailed diagnostics
        if original_ndim == 2:
            raise FailedInversionError.from_matrix(values[0], "cholesky", e) from e
        # For batches, just indicate which failed (more complex error handling needed)
        raise FailedInversionError(
            f"Cholesky decomposition failed on batch: {e}",
            method="cholesky",
        ) from e

    batch_size, n_params = values.shape[0], values.shape[-1]

    # Generate standard normal samples for all batches
    # Shape: (batch, num_samples, n_params)
    z = rng.standard_normal((batch_size, num_samples, n_params))

    # Solve: chol.T @ samples.T = z.T for each batch
    # Reshape z for batch solving: (batch, n_params, num_samples)
    z_reshaped = np.swapaxes(z, -2, -1)

    # Transpose cholesky factors
    chol_T = np.swapaxes(chol, -2, -1)

    # Solve using vectorized operations
    # np.linalg.solve handles batches when shapes are (..., M, M) @ (..., M, K)
    samples_T = np.linalg.solve(chol_T, z_reshaped)

    # Back to (batch, num_samples, n_params)
    samples = np.swapaxes(samples_T, -2, -1)

    # Remove batch dimension if input was 2D
    if original_ndim == 2:
        samples = samples[0]

    return samples
