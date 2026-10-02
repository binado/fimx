"""NetCDF loading helpers."""

from os import PathLike

import xarray as xr


def load_dataset(path: str | PathLike[str]) -> xr.Dataset:
    """Load a NetCDF Dataset using the optional h5netcdf engine.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a NetCDF file.

    Returns
    -------
    xarray.Dataset
        Loaded Dataset.
    """
    return xr.load_dataset(path, engine="h5netcdf")


__all__ = ["load_dataset"]
