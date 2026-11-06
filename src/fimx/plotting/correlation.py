"""Correlation heatmaps from stored forecast covariances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..datasets import _display_labels, _stored_covariance

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
    covariance, names, selected = _stored_covariance(forecast, parameters)
    display = _display_labels(forecast, names, selected, parameter_labels)
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
