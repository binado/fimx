"""Backend protocol and shared preparation for marginalized Gaussian plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..construction import _labels
from ..datasets import _validate_dataset
from ..inversion import inv

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PlotBackend(Protocol):
    """Callable interface implemented by each plotting backend."""

    def __call__(
        self,
        datasets: Mapping[str, xr.Dataset],
        *,
        parameters: Sequence[str] | None = None,
        filled: bool = True,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Figure:
        """Return a Figure overlaying the supplied forecasts."""
        ...


@dataclass(frozen=True)
class _Gaussian:
    """Prepared marginal distribution for one forecast."""

    label: str
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]


def _prepare(
    datasets: Mapping[str, xr.Dataset], parameters: Sequence[str] | None
) -> tuple[list[str], list[_Gaussian]]:
    """Validate forecasts and select marginalized Gaussian distributions."""
    if not isinstance(datasets, Mapping):
        raise TypeError("Plot inputs must be a mapping of labels to Datasets.")
    if not datasets:
        raise ValueError("At least one Dataset is required.")
    if any(not isinstance(label, str) for label in datasets):
        raise ValueError("Plot labels must be strings.")
    forecasts = {label: _validate_dataset(ds) for label, ds in datasets.items()}
    first = next(iter(forecasts.values()))
    if parameters is None:
        names = [
            name
            for name in first.row.values.tolist()
            if all(name in ds.row.values for ds in forecasts.values())
        ]
        if not names:
            raise ValueError("Datasets have no shared parameters to plot.")
    else:
        names = _labels(parameters, name="parameters")
        for label, ds in forecasts.items():
            for name in names:
                if name not in ds.row.values:
                    raise KeyError(f"Parameter {name!r} is absent from {label!r}.")
    distributions = []
    for label, ds in forecasts.items():
        covariance = inv(ds["fisher"]).sel(row=names, col=names)
        distributions.append(
            _Gaussian(label, ds["fiducials"].sel(row=names).values, covariance.values)
        )
    return names, distributions
