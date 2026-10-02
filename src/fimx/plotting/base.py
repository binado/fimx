"""Backend protocol and shared preparation for marginalized Gaussian plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..construction import _labels, _real_values, _validate_matrix
from ..inversion import InversionMethod, inv

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _Gaussian:
    """Prepared marginal distribution for one forecast."""

    label: str
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]


class PlotBackend(Protocol):
    """Callable interface implemented by each plotting backend.

    Backends receive parameter names, display labels, and already-inverted
    marginal covariances; they never see Fisher matrices.
    """

    def __call__(
        self,
        names: Sequence[str],
        parameter_labels: Sequence[str],
        distributions: Sequence[_Gaussian],
        *,
        filled: bool = True,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Figure:
        """Return a Figure overlaying the supplied Gaussian distributions."""
        ...


def _validate_dataset(forecast: xr.Dataset) -> xr.Dataset:
    """Validate the Fisher matrix and numeric reference values for plotting."""
    if not isinstance(forecast, xr.Dataset):
        raise TypeError("Plot inputs must be xarray.Dataset objects.")
    if not {"fisher", "fiducials"}.issubset(forecast.data_vars):
        raise ValueError("Datasets must contain 'fisher' and 'fiducials' variables.")
    fisher, labels = _validate_matrix(forecast["fisher"])
    reference = forecast["fiducials"]
    if reference.dims != ("row",):
        raise ValueError("Fiducial dimensions must be exactly ('row',).")
    variables = {
        "fisher": fisher,
        "fiducials": xr.DataArray(
            _real_values(reference.values).copy(),
            dims="row",
            coords={"row": labels},
        ),
    }
    if "labels" in forecast.data_vars:
        display = forecast["labels"]
        if display.dims != ("row",):
            raise ValueError("Plot label dimensions must be exactly ('row',).")
        variables["labels"] = xr.DataArray(
            _labels(display.values.tolist(), name="labels"),
            dims="row",
            coords={"row": labels},
        )
    return xr.Dataset(variables)


def _dataset_labels(
    forecasts: Mapping[str, xr.Dataset], names: Sequence[str]
) -> dict[str, str]:
    """Collect the ``labels`` variable of each forecast for the plotted names."""
    collected: dict[str, str] = {}
    sources: dict[str, str] = {}
    for file_label, forecast in forecasts.items():
        if "labels" not in forecast.data_vars:
            continue
        by_name = dict(
            zip(forecast.row.values.tolist(), forecast["labels"].values.tolist())
        )
        for name in names:
            if name not in by_name:
                continue
            if name in collected and collected[name] != by_name[name]:
                raise ValueError(
                    f"Conflicting plot labels for parameter {name!r}: "
                    f"{sources[name]!r} uses {collected[name]!r}, but "
                    f"{file_label!r} uses {by_name[name]!r}."
                )
            collected[name] = by_name[name]
            sources[name] = file_label
    return collected


def _prepare(
    datasets: Mapping[str, xr.Dataset],
    parameters: Sequence[str] | None,
    method: InversionMethod = "cholesky",
    parameter_labels: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str], list[_Gaussian]]:
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
    if parameter_labels is not None:
        if not isinstance(parameter_labels, Mapping):
            raise TypeError("Parameter labels must be a mapping of names to strings.")
        if any(
            not isinstance(name, str) or not isinstance(label, str)
            for name, label in parameter_labels.items()
        ):
            raise ValueError("Parameter label keys and values must be strings.")
    display = {**_dataset_labels(forecasts, names), **(parameter_labels or {})}
    plot_names = [display.get(name, name) for name in names]
    distributions = []
    for label, ds in forecasts.items():
        covariance = inv(ds["fisher"], method=method).sel(row=names, col=names)
        distributions.append(
            _Gaussian(label, ds["fiducials"].sel(row=names).values, covariance.values)
        )
    return names, plot_names, distributions
