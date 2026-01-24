"""Custom exceptions for the fisharr package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import numpy.typing as npt


class MatrixNotSquareError(ValueError):
    """Raised when a matrix is not square."""

    def __init__(self, data: npt.ArrayLike | xr.DataArray) -> None:
        total_shape = (
            data.shape if isinstance(data, xr.DataArray) else np.asarray(data).shape
        )
        self.shape = total_shape[:2]
        super().__init__(f"Expected square matrix, got shape {self.shape}")


class InsufficientDimsError(ValueError):
    """Raised when a matrix has insufficient dimensions for an operation.

    Attributes
    ----------
    shape : tuple[int, ...]
        The shape of the matrix that failed the operation
    """

    def __init__(self, data: npt.ArrayLike | xr.DataArray, expected: int) -> None:
        self.actual_dims = (
            data.ndim if isinstance(data, xr.DataArray) else np.asarray(data).ndim
        )
        self.expected_dims = expected
        super().__init__(
            f"Expected array with at least {self.expected_dims} dimensions, got {self.actual_dims}"
        )


class FailedInversionError(Exception):
    """Raised when matrix inversion fails.

    This exception provides diagnostic information to help identify why
    the inversion failed and suggests possible remedies.

    Attributes
    ----------
    method : str
        The inversion method that failed
    condition_number : float | None
        Condition number of the matrix (if available)
    min_eigenvalue : float | None
        Minimum eigenvalue (if available)
    max_eigenvalue : float | None
        Maximum eigenvalue (if available)
    """

    def __init__(
        self,
        message: str,
        *,
        method: str,
        condition_number: float | None = None,
        min_eigenvalue: float | None = None,
        max_eigenvalue: float | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.condition_number = condition_number
        self.min_eigenvalue = min_eigenvalue
        self.max_eigenvalue = max_eigenvalue

    @classmethod
    def from_matrix(
        cls, mat: np.ndarray, method: str, original_error: Exception
    ) -> "FailedInversionError":
        """Create exception with diagnostics from matrix analysis.

        Parameters
        ----------
        mat : np.ndarray
            The matrix that failed to invert
        method : str
            The inversion method that was attempted
        original_error : Exception
            The original exception that was raised

        Returns
        -------
        FailedInversionError
            Exception with diagnostic information
        """
        try:
            cond = np.linalg.cond(mat)
            eigvals = np.linalg.eigvalsh(mat)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
        except Exception:
            # If diagnostics fail, just use basic message
            return cls(
                f"Matrix inversion failed with method='{method}': {original_error}",
                method=method,
            )

        message = (
            f"Matrix inversion failed with method='{method}'.\n"
            f"Diagnostics:\n"
            f"  Condition number: {cond:.2e}\n"
            f"  Min eigenvalue: {min_eig:.2e}\n"
            f"  Max eigenvalue: {max_eig:.2e}\n"
        )

        if min_eig <= 0:
            message += (
                "  ⚠ Matrix is not positive definite (has non-positive eigenvalues)\n"
            )

        if cond > 1e10:
            message += "  ⚠ Matrix is poorly conditioned\n"
            message += "\nSuggestions:\n"
            message += "  - Try method='svd' for robust pseudoinverse\n"
            message += "  - Check for parameter degeneracies\n"
            message += "  - Consider marginalizing over degenerate parameters\n"
        elif method == "cholesky":
            message += "\nSuggestions:\n"
            message += "  - Try method='svd' if matrix is nearly singular\n"
            message += "  - Check that matrix is positive definite\n"

        return cls(
            message,
            method=method,
            condition_number=cond,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
        )
