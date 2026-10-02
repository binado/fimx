"""Analytic Gaussian corner plots using GetDist."""

from collections.abc import Mapping, Sequence
from typing import Any

from getdist import plots
from getdist.gaussian_mixtures import GaussianND
from matplotlib.figure import Figure

from .base import PlotBackend, _Gaussian


def plot(
    names: Sequence[str],
    parameter_labels: Sequence[str],
    distributions: Sequence[_Gaussian],
    *,
    filled: bool = True,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> Figure:
    """Overlay marginalized Gaussian forecasts with GetDist.

    Parameters
    ----------
    names : sequence of str
        Ordered parameter names used to identify the distributions' axes.
    parameter_labels : sequence of str
        Ordered display labels matching ``names``. GetDist renders them as
        LaTeX math, so they must not include enclosing ``$``. A label equal
        to its name is treated as absent and the plain name is drawn.
    distributions : sequence of _Gaussian
        Prepared marginal means and covariances, one per forecast.
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
    names = list(names)
    parameter_labels = [
        "" if label == name else label
        for name, label in zip(names, parameter_labels, strict=True)
    ]
    roots = [
        GaussianND(
            item.mean,
            item.covariance,
            names=names,
            labels=parameter_labels,
            label=item.label,
        )
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
