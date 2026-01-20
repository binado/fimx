from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

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


def _fallback(
    name: str,
    default_factory: Callable[[str], T] | None,
    default_value: T | None,
) -> T | None:
    if default_factory is not None:
        return default_factory(name)
    return default_value
