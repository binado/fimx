"""Shared command-line option types."""

from enum import StrEnum
from typing import get_args

from ..inversion import InversionMethod


class InversionMethodOption(StrEnum):
    """Inversion methods accepted on the command line."""

    cholesky = "cholesky"
    inv = "inv"
    pinv = "pinv"


_METHODS = tuple(method.value for method in InversionMethodOption)
if _METHODS != get_args(InversionMethod):
    raise RuntimeError(
        "CLI inversion methods must match fimx.inversion.InversionMethod."
    )
