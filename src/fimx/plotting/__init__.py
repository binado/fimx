"""Backend selection for Fisher forecast plots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import xarray as xr

from .base import PlotBackend

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _getdist() -> PlotBackend:
    """Load the optional GetDist implementation."""
    try:
        from .getdist import plot
    except ModuleNotFoundError as exc:
        if exc.name not in {"getdist", "matplotlib"}:
            raise
        raise ImportError(
            "The GetDist backend requires plotting dependencies. "
            "Install them with uv add 'fimx[plotting]'."
        ) from exc
    return plot


_BACKENDS: dict[str, Callable[[], PlotBackend]] = {"getdist": _getdist}


def plot(
    datasets: Mapping[str, xr.Dataset],
    *,
    parameters: Sequence[str] | None = None,
    filled: bool = True,
    backend: str = "getdist",
    backend_kwargs: Mapping[str, Any] | None = None,
) -> Figure:
    """Plot overlaid marginalized constraints from labeled Fisher forecasts.

    Parameters
    ----------
    datasets : mapping of str to xarray.Dataset
        Nonempty forecast mapping. Keys provide legend labels and overlay
        order. Each Dataset requires ``fisher`` on ``('row', 'col')`` and
        finite ``fiducials`` on ``row`` with matching parameter coordinates.
    parameters : sequence of str, optional
        Unique parameters present in every forecast, in plotting order.
        Defaults to their intersection in the first forecast's order.
        Omitted parameters are marginalized over.
    filled : bool
        Whether to fill two-dimensional contours. Default is True.
    backend : str
        Backend name. Currently only ``'getdist'`` is supported.
    backend_kwargs : mapping, optional
        GetDist ``triangle_plot`` options excluding ``roots``, ``params``,
        ``legend_labels``, and ``filled``. Native styles and confidence
        levels apply unless customized.

    Returns
    -------
    matplotlib.figure.Figure
        Figure for editing or saving; it is not automatically shown or saved.

    Raises
    ------
    TypeError
        If containers have incorrect types or backend options are reserved.
    ValueError
        If data are invalid, no parameters are shared, or backend is unknown.
    KeyError
        If a requested parameter is unavailable in any forecast.
    numpy.linalg.LinAlgError
        If any complete Fisher matrix is singular or not positive definite.
    ImportError
        If the selected backend's optional dependencies are unavailable.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown plotting backend {backend!r}. Available: getdist.")
    implementation = _BACKENDS[backend]()
    return implementation(
        datasets, parameters=parameters, filled=filled, backend_kwargs=backend_kwargs
    )


__all__ = ["plot"]
