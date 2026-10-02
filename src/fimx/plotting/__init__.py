"""Backend selection for Fisher forecast plots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import xarray as xr

from ..inversion import InversionMethod
from .base import PlotBackend, _prepare

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
    parameter_labels: Mapping[str, str] | None = None,
    filled: bool = True,
    method: InversionMethod = "cholesky",
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
    parameter_labels : mapping of str to str, optional
        Display labels keyed by parameter name. Parameters without a supplied
        display label keep their original name.
    filled : bool
        Whether to fill two-dimensional contours. Default is True.
    method : {'cholesky', 'inv', 'pinv'}
        Algorithm used to invert each full Fisher matrix; see
        :func:`fimx.inv`. Default is ``'cholesky'``.
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
        If data or parameter labels are invalid, no parameters are shared, or
        backend or inversion method is unknown.
    KeyError
        If a requested parameter is unavailable in any forecast.
    numpy.linalg.LinAlgError
        If any complete Fisher matrix is singular (``'inv'``) or not positive
        definite (``'cholesky'``). ``'pinv'`` never raises this.
    ImportError
        If the selected backend's optional dependencies are unavailable.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown plotting backend {backend!r}. Available: getdist.")
    implementation = _BACKENDS[backend]()
    names, labels, distributions = _prepare(
        datasets, parameters, method, parameter_labels=parameter_labels
    )
    return implementation(
        names,
        labels,
        distributions,
        filled=filled,
        backend_kwargs=backend_kwargs,
    )


__all__ = ["plot"]
