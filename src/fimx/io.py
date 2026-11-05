"""HDF5 (NetCDF4) loading and saving of forecast Datasets."""

from collections.abc import Hashable, Sequence
from os import PathLike

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .arrays import _plot_labels, _real_values, _validate_matrix
from .datasets import dataset

# Loose bound on |F @ C @ F - F| relative to max|F|. It is meant to catch a
# covariance that belongs to another matrix, not to judge round-off.
_COVARIANCE_RTOL = 1e-3


def load_dataset(path: str | PathLike[str]) -> xr.Dataset:
    """Load an HDF5 (NetCDF4) Dataset using the optional h5netcdf engine.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an HDF5 file in the NetCDF4 convention. The extension is
        not checked, so ``.nc``, ``.h5`` and ``.hdf5`` all work.

    Returns
    -------
    xarray.Dataset
        Loaded Dataset.
    """
    return xr.load_dataset(path, engine="h5netcdf")


def _aligned_covariance(fisher: xr.DataArray, covariance: xr.DataArray) -> xr.DataArray:
    """Return a covariance in the Fisher matrix order if it is consistent with it."""
    validated, parameters = _validate_matrix(covariance)
    expected = fisher.row.values.tolist()
    if set(parameters) != set(expected):
        raise ValueError("Covariance parameters must match the Fisher matrix.")
    ordered = validated.sel(row=expected, col=expected)
    # F @ C @ F == F holds for an inverse and for a pseudoinverse alike.
    product = fisher.values @ ordered.values @ fisher.values
    if not np.allclose(
        product,
        fisher.values,
        rtol=0,
        atol=_COVARIANCE_RTOL * np.max(np.abs(fisher.values)),
    ):
        raise ValueError("Covariance is not an inverse of the Fisher matrix.")
    ordered.attrs = dict(covariance.attrs)
    return ordered


def save_dataset(
    path: str | PathLike[str],
    fisher: xr.DataArray,
    *,
    covariance: xr.DataArray | None = None,
    fiducials: ArrayLike | xr.DataArray | None = None,
    labels: Sequence[str] | None = None,
) -> None:
    """Save a forecast as an HDF5 (NetCDF4) Dataset in the canonical schema.

    Parameters
    ----------
    path : str or os.PathLike
        Output path. The optional h5netcdf engine is used, and the extension
        is free: ``.nc``, ``.h5`` and ``.hdf5`` write the same file.
    fisher : xarray.DataArray
        Canonical Fisher matrix, stored as ``fisher``.
    covariance : xarray.DataArray, optional
        Inverse or pseudoinverse of ``fisher``, stored as ``covariance`` with
        its ``attrs``, if any.
        It is never computed here.
    fiducials : array_like or xarray.DataArray, optional
        Finite, real reference values, one per parameter, stored as
        ``fiducials``.
    labels : sequence of str, optional
        Unique axis labels for plotting, one per parameter in matrix order,
        stored as ``labels``. They are LaTeX math without the enclosing
        ``$``, for example ``r"\\Omega_m"``.

    Raises
    ------
    ValueError
        If any input violates the matrix contract, if ``covariance`` does not
        match ``fisher`` (checked loosely, so that mismatched matrices are
        caught but round-off is not), or if ``fiducials`` or ``labels`` are
        invalid.

    Notes
    -----
    The file can be read with :func:`load_dataset` and passed to
    :func:`fimx.plot`, and is the input of the ``fimx plot corner`` and
    ``fimx invert`` commands.
    """
    validated, _ = _validate_matrix(fisher)
    arrays: dict[Hashable, ArrayLike | xr.DataArray] = {}
    if covariance is not None:
        arrays["covariance"] = _aligned_covariance(validated, covariance)
    if fiducials is not None:
        _real_values(fiducials)
        arrays["fiducials"] = fiducials
    if labels is not None:
        arrays["labels"] = _plot_labels(labels)
    dataset(fisher, arrays).to_netcdf(path, engine="h5netcdf")


__all__ = ["load_dataset", "save_dataset"]
