"""Correlation heatmaps from stored forecast covariances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..arrays import _labels, _plot_labels, _validate_matrix
from ..inversion import _eigenvalues, _positive_semidefinite

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_correlation(
    forecast: xr.Dataset,
    *,
    parameters: Sequence[str] | None = None,
    parameter_labels: Mapping[str, str] | None = None,
) -> Figure:
    """Plot a correlation heatmap from a stored covariance.

    Parameters
    ----------
    forecast : xarray.Dataset
        Dataset containing a finite, symmetric, positive semidefinite
        ``covariance`` on ``('row', 'col')`` with positive variances.
        Fisher matrices and fiducials are not required or used.
    parameters : sequence of str, optional
        Unique parameters to display, in order. Defaults to covariance order.
        Selection does not condition on omitted parameters.
    parameter_labels : mapping of str to str, optional
        Display labels overriding the Dataset's optional ``labels`` on ``row``.
        Labels are unique LaTeX math strings without enclosing ``$``.
        Parameters without a display label use their plain names.

    Returns
    -------
    matplotlib.figure.Figure
        Editable heatmap; it is not automatically shown or saved.

    Raises
    ------
    TypeError
        If forecast or parameter_labels has an incorrect type.
    ValueError
        If covariance, labels, or parameter selection is invalid.
    KeyError
        If a selected parameter is absent.
    numpy.linalg.LinAlgError
        If covariance is not positive semidefinite or has nonpositive variances.
    ImportError
        If Matplotlib is unavailable; install ``fimx[plotting]``.
    """
    if not isinstance(forecast, xr.Dataset):
        raise TypeError("Plot inputs must be xarray.Dataset objects.")
    if "covariance" not in forecast.data_vars:
        raise ValueError("Dataset must contain a 'covariance' variable.")
    covariance, names = _validate_matrix(forecast["covariance"])
    if not _positive_semidefinite(_eigenvalues(covariance)):
        raise np.linalg.LinAlgError(
            "Correlation requires a positive semidefinite covariance."
        )
    if not np.all(np.diag(covariance.values) > 0):
        raise np.linalg.LinAlgError("Correlation requires positive variances.")
    selected = names if parameters is None else _labels(parameters, name="parameters")
    for name in selected:
        if name not in names:
            raise KeyError(name)
    display: dict[str, str] = {}
    if "labels" in forecast.data_vars:
        labels = forecast["labels"]
        if labels.dims != ("row",):
            raise ValueError("Plot label dimensions must be exactly ('row',).")
        display.update(zip(names, _plot_labels(labels.values.tolist()), strict=True))
    if parameter_labels is not None:
        if not isinstance(parameter_labels, Mapping):
            raise TypeError("Parameter labels must be a mapping of names to strings.")
        if any(not isinstance(name, str) for name in parameter_labels):
            raise ValueError("Parameter label keys must be strings.")
        _plot_labels(list(parameter_labels.values()))
        display.update(parameter_labels)
    _plot_labels([display.get(name, name) for name in selected])
    ticks_labels = [
        f"${display[name]}$" if name in display else name for name in selected
    ]
    values = covariance.sel(row=selected, col=selected).values
    sigma = np.sqrt(np.diag(values))
    correlation = values / sigma[:, None] / sigma[None, :]
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise ImportError(
            "Correlation plotting requires Matplotlib. "
            "Install it with uv add 'fimx[plotting]'."
        ) from exc
    n = len(selected)
    figure, ax = plt.subplots(
        figsize=(0.2 * n + 2.0, 0.2 * n + 1.2), constrained_layout=True
    )
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n), ticks_labels, rotation=90)
    ax.set_yticks(np.arange(n), ticks_labels)
    ax.tick_params(length=0, labelsize="x-small")
    figure.colorbar(image, ax=ax, shrink=0.8, label="correlation")
    return figure
