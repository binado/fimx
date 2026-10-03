"""Functions for Fisher information on labeled xarray matrices."""

from .arrays import gaussian_prior, matrix, vector
from .datasets import dataset
from .inversion import diagnose, errors, inv
from .ops import (
    combine,
    correlation,
    expand,
    fix,
    fom,
    marginalize,
    symmetrize,
    transform,
)
from .plotting import plot

__all__ = [
    "combine",
    "correlation",
    "dataset",
    "diagnose",
    "errors",
    "expand",
    "fix",
    "fom",
    "gaussian_prior",
    "inv",
    "marginalize",
    "matrix",
    "plot",
    "symmetrize",
    "transform",
    "vector",
]
