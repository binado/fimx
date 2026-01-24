from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

T = TypeVar("T")


def normalize_metadata(
    names: Sequence[str],
    values: Sequence[T] | Mapping[str, T] | None,
    *,
    default_factory: Callable[[str], T] | None = None,
    default_value: T | None = None,
) -> dict[str, T | None]:
    if values is None:
        return {name: _fallback(name, default_factory, default_value) for name in names}
    if isinstance(values, Mapping):
        # Note: We use 'if name in values' rather than .get() to avoid
        # eagerly evaluating _fallback(), which may call default_factory
        return {
            name: values[name]
            if name in values
            else _fallback(name, default_factory, default_value)
            for name in names
        }
    values_list = list(values)
    if len(values_list) != len(names):
        raise ValueError("Metadata length must match parameter length.")
    return {name: values_list[idx] for idx, name in enumerate(names)}


def normalize_metadata_array(
    names: Sequence[str],
    values: Sequence[T] | Mapping[str, T] | ArrayLike | None,
    *,
    fill_value: object = np.nan,
    dtype: np.dtype | None = None,
) -> npt.NDArray | None:
    """Normalize metadata values into an array aligned with parameter names.

    Parameters
    ----------
    names : Sequence[str]
        Parameter names defining the output order
    values : Sequence[T] | Mapping[str, T] | ArrayLike | None
        Metadata values provided as a sequence, mapping, or array-like
    fill_value : object
        Fill value used for missing entries (default: np.nan)
    dtype : np.dtype | None
        Optional dtype for the output array

    Returns
    -------
    npt.NDArray | None
        Array of values aligned to names, or None if values is None
    """
    if values is None:
        return None
    if isinstance(values, Mapping):
        normalized = [values[name] if name in values else fill_value for name in names]
    elif isinstance(values, Sequence):
        normalized = list(values)
    else:
        normalized = np.asarray(values).tolist()
        if not isinstance(normalized, list):
            normalized = [normalized]
    if len(normalized) != len(names):
        raise ValueError("Metadata length must match parameter length.")
    normalized = [fill_value if value is None else value for value in normalized]
    return np.asarray(normalized, dtype=dtype)


def _fallback(
    name: str,
    default_factory: Callable[[str], T] | None,
    default_value: T | None,
) -> T | None:
    if default_factory is not None:
        return default_factory(name)
    return default_value
