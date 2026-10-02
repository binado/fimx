"""Analytic Gaussian corner plots using GetDist."""

from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
from matplotlib.figure import Figure

from .base import PlotBackend, _prepare


def plot(
    datasets: Mapping[str, xr.Dataset],
    *,
    parameters: Sequence[str] | None = None,
    filled: bool = True,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> Figure:
    """Overlay marginalized Gaussian forecasts with GetDist.

    Parameters
    ----------
    datasets : mapping of str to xarray.Dataset
        Forecasts containing ``fisher`` and ``fiducials`` variables.
    parameters : sequence of str, optional
        Ordered parameter selection; otherwise use the common parameters.
    filled : bool
        Whether to fill the two-dimensional contours.
    backend_kwargs : mapping, optional
        Additional arguments to GetDist's ``triangle_plot``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid forecasts.

    Raises
    ------
    TypeError
        If backend options override arguments controlled by this function.
    """
    options = dict(backend_kwargs or {})
    reserved = {"roots", "params", "legend_labels", "filled"}.intersection(options)
    if reserved:
        raise TypeError(
            f"Backend options cannot override: {', '.join(sorted(reserved))}."
        )
    names, distributions = _prepare(datasets, parameters)
    roots = [
        GaussianND(item.mean, item.covariance, names=names, label=item.label)
        for item in distributions
    ]
    plotter = plots.get_subplot_plotter()
    plotter.triangle_plot(
        roots,
        params=names,
        legend_labels=[item.label for item in distributions],
        filled=filled,
        **options,
    )
    figure = plotter.fig
    assert figure is not None
    return figure


_backend: PlotBackend = plot
