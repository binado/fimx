"""Functions for Fisher information on labeled xarray matrices."""

from .combination import combine, gaussian_prior
from .construction import matrix
from .datasets import dataset
from .inversion import errors, inv
from .parameters import fix, marginalize, transform
from .plotting import plot

__all__ = [
    "combine",
    "dataset",
    "errors",
    "fix",
    "gaussian_prior",
    "inv",
    "marginalize",
    "matrix",
    "plot",
    "transform",
]
