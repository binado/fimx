"""Functions for Fisher information on labeled xarray matrices."""

from .arrays import gaussian_prior, matrix, vector
from .datasets import dataset
from .inversion import errors, inv
from .ops import combine, expand, fix, marginalize, symmetrize, transform
from .plotting import plot

__all__ = [
    "combine",
    "dataset",
    "errors",
    "expand",
    "fix",
    "gaussian_prior",
    "inv",
    "marginalize",
    "matrix",
    "plot",
    "symmetrize",
    "transform",
    "vector",
]
