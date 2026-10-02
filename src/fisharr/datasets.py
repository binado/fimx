"""Construction and validation of Fisher forecast Datasets."""

import xarray as xr

from .construction import _coordinate_labels, _real_values, _validate_matrix


def dataset(F: xr.DataArray, fiducials: xr.DataArray) -> xr.Dataset:
    """Combine a Fisher matrix and reference values with shared coordinates.

    Parameters
    ----------
    F : xarray.DataArray
        Canonical matrix with dimensions ``('row', 'col')``.
    fiducials : xarray.DataArray
        Finite real values on dimension ``row`` with the same parameter set
        as F. Values are reordered to match F before Dataset construction.

    Returns
    -------
    xarray.Dataset
        Independent float64 variables ``fisher`` and ``fiducials`` with
        canonical coordinates. Positive definiteness is not required.

    Raises
    ------
    TypeError
        If either input is not a DataArray.
    ValueError
        If the matrix or fiducials violate their coordinate or value contract.
    """
    fisher, labels = _validate_matrix(F)
    if not isinstance(fiducials, xr.DataArray):
        raise TypeError("Fiducials must be an xarray.DataArray.")
    if fiducials.dims != ("row",):
        raise ValueError("Fiducial dimensions must be exactly ('row',).")
    names = _coordinate_labels(fiducials, "row")
    if set(names) != set(labels):
        raise ValueError(
            "Fiducial parameters must match the complete matrix parameters."
        )
    values = _real_values(fiducials.sel(row=labels).values).copy()
    reference = xr.DataArray(values, dims="row", coords={"row": labels.copy()})
    return xr.Dataset({"fisher": fisher, "fiducials": reference})


def _validate_dataset(forecast: xr.Dataset) -> xr.Dataset:
    """Validate a forecast and return independent canonical variables."""
    if not isinstance(forecast, xr.Dataset):
        raise TypeError("Plot inputs must be xarray.Dataset objects.")
    if not {"fisher", "fiducials"}.issubset(forecast.data_vars):
        raise ValueError("Datasets must contain 'fisher' and 'fiducials' variables.")
    return dataset(forecast["fisher"], forecast["fiducials"])
