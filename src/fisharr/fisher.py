from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

import fisharr.linalg

from .accessors import get_batch_dims, get_matrix_coords, get_matrix_dims
from .construction import (
    build_dataarray_from_array,
    normalize_dataarray,
    normalize_dataset,
    partition_matrices,
)
from .diagnostics import (
    Diagnostics,
    condition_numbers,
    min_eigenvalues,
    min_max_eigenvalues,
)
from .dimensions import MatrixDims
from .indexing import submatrix
from .inversion import invert_matrices
from .metadata import normalize_metadata_array
from .sampling import sample_from_fisher

DEFAULT_ROW_DIM = "row"
DEFAULT_COL_DIM = "col"
DEFAULT_PARAMETER_DIM = "parameter"

FISHER_VAR = "fisher_matrix"
COVARIANCE_VAR = "covariance"
FIDUCIALS_VAR = "fiducials"
UNITS_VAR = "units"
LABELS_VAR = "labels"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return False


class FisherMatrix:
    def __init__(
        self,
        data: ArrayLike | xr.DataArray | xr.Dataset,
        parameters: Sequence[str] | str | None = None,
        *,
        labels: Sequence[str] | Mapping[str, str] | None = None,
        units: Sequence[str | None] | Mapping[str, str | None] | None = None,
        fiducials: Sequence[float | None]
        | Mapping[str, float | None]
        | ArrayLike
        | None = None,
        batch_dims: Sequence[str] | None = None,
        parameter_dim: str = DEFAULT_PARAMETER_DIM,
    ) -> None:
        self._cache: xr.Dataset = xr.Dataset()
        self._parameter_dim = parameter_dim

        if isinstance(data, xr.Dataset):
            if FISHER_VAR not in data:
                raise ValueError("Dataset must contain a fisher_matrix variable.")
            ds = normalize_dataset(data, FISHER_VAR)
            da = ds[FISHER_VAR]
        elif isinstance(data, xr.DataArray):
            da = normalize_dataarray(data)
            ds = None
        else:
            if parameters is None or isinstance(parameters, str):
                raise ValueError(
                    "parameters are required when data is not an xarray object."
                )
            matrix_dims = (DEFAULT_ROW_DIM, DEFAULT_COL_DIM)
            da = build_dataarray_from_array(
                data,
                parameters,
                matrix_dims=matrix_dims,
                batch_dims=batch_dims,
            )
            ds = None

        params, _ = get_matrix_coords(da)
        if ds is None:
            ds = xr.Dataset({FISHER_VAR: da})
        self._dataset = ds.assign_coords({parameter_dim: params})

        self._sync_metadata(labels=labels, units=units, fiducials=fiducials)

    @property
    def data(self) -> xr.DataArray:
        return self._dataset[FISHER_VAR]

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    @property
    def parameters(self) -> list[str]:
        row_dim, _ = self.matrix_dims
        return self._dataset[FISHER_VAR].coords[row_dim].values.tolist()

    @property
    def matrix_dims(self) -> MatrixDims:
        return get_matrix_dims(self.data)

    @property
    def batch_dims(self) -> tuple[str, ...]:
        return tuple(str(d) for d in get_batch_dims(self.data))

    @property
    def labels(self) -> xr.DataArray | None:
        return getattr(self._dataset, LABELS_VAR, None)

    @labels.setter
    def labels(self, value: Sequence[str] | Mapping[str, str] | None) -> None:
        self._set_metadata_var(LABELS_VAR, value, dtype=np.dtype(object))

    @property
    def units(self) -> xr.DataArray | None:
        return getattr(self._dataset, UNITS_VAR, None)

    @units.setter
    def units(
        self, value: Sequence[str | None] | Mapping[str, str | None] | None
    ) -> None:
        self._set_metadata_var(UNITS_VAR, value, dtype=np.dtype(object))

    @property
    def fiducials(self) -> xr.DataArray | None:
        return getattr(self._dataset, FIDUCIALS_VAR, None)

    @fiducials.setter
    def fiducials(
        self,
        value: Sequence[float | None] | Mapping[str, float | None] | ArrayLike | None,
    ) -> None:
        self._set_metadata_var(FIDUCIALS_VAR, value, dtype=np.dtype(float))

    def clear_cache(self) -> None:
        self._cache = xr.Dataset()

    def to_dataset(self, *, include_cache: bool = False) -> xr.Dataset:
        """Return the underlying dataset.

        Parameters
        ----------
        include_cache : bool
            If True, merge cached variables into the returned dataset

        Returns
        -------
        xr.Dataset
            Copy of the underlying dataset, optionally with cached data
        """
        if not include_cache or not self._cache.data_vars:
            return self._dataset.copy()
        return xr.merge([self._dataset, self._cache], compat="no_conflicts")

    def covariance(
        self,
        method: str = "cholesky",
        *,
        rcond: float | None = None,
        cache: bool = True,
    ) -> xr.DataArray:
        cached = self._cache.get(COVARIANCE_VAR)
        if cached is not None:
            if (
                cached.attrs.get("method") == method
                and cached.attrs.get("rcond") == rcond
            ):
                return cached

        inv = invert_matrices(self.data.values, method=method, rcond=rcond)
        cov = self._matrix_dataarray(inv, self.parameters)
        cov.attrs["method"] = method
        cov.attrs["rcond"] = rcond
        if cache:
            self._cache[COVARIANCE_VAR] = cov
        return cov

    def marginalized_errors(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        return cov.pipe(fisharr.linalg.diagonal).pipe(np.sqrt)

    def correlation(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        values = np.asarray(cov.values)
        diag = np.diagonal(values, axis1=-2, axis2=-1)
        denom = np.sqrt(diag[..., :, None] * diag[..., None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov.values / denom

        n_params = corr.shape[-1]

        if corr.ndim == 2:
            diag_idx = np.arange(n_params)
            mask = np.isfinite(corr[diag_idx, diag_idx])
            corr[diag_idx[mask], diag_idx[mask]] = 1.0
        else:
            for batch_idx in range(corr.shape[0]):
                diag_idx = np.arange(n_params)
                mask = np.isfinite(corr[batch_idx, diag_idx, diag_idx])
                corr[batch_idx, diag_idx[mask], diag_idx[mask]] = 1.0

        return self._matrix_dataarray(corr, self.parameters)

    def condition_number(self, method: str = "svd") -> xr.DataArray:
        values = condition_numbers(self.data.values, method=method)
        return self._scalar_dataarray(values)

    def is_degenerate(
        self,
        *,
        min_eigenvalue: float | None = None,
        condition_number: float | None = None,
    ) -> xr.DataArray:
        n_provided = sum([min_eigenvalue is not None, condition_number is not None])
        if n_provided != 1:
            raise ValueError(
                "Specify exactly one of min_eigenvalue or condition_number."
            )
        if min_eigenvalue is not None:
            min_vals = min_eigenvalues(self.data.values)
            return self._scalar_dataarray(min_vals <= min_eigenvalue)
        cond = condition_numbers(self.data.values, method="svd")
        return self._scalar_dataarray(cond >= condition_number)

    def diagnostics(
        self, method: str = "svd", threshold: float | None = None
    ) -> Diagnostics:
        min_vals, max_vals = min_max_eigenvalues(self.data.values, method=method)
        cond = max_vals / min_vals
        if threshold is None:
            is_deg = np.zeros_like(min_vals, dtype=bool)
        else:
            is_deg = min_vals <= threshold
        return Diagnostics(
            min_eigenvalue=self._scalar_dataarray(min_vals),
            max_eigenvalue=self._scalar_dataarray(max_vals),
            condition_number=self._scalar_dataarray(cond),
            is_degenerate=self._scalar_dataarray(is_deg),
            method=method,
            threshold=threshold,
        )

    def marginalize(
        self, parameters: str | Sequence[str], *, method: str = "cholesky"
    ) -> "FisherMatrix":
        names = self._normalize_parameter_list(parameters)
        if not names:
            return self
        keep, drop = self._split_parameters(names)
        if not keep:
            raise ValueError("Cannot marginalize all parameters.")

        idx_keep = self._indices_for(keep)
        idx_drop = self._indices_for(drop)
        f_kk, f_kd, f_dd, f_dk = partition_matrices(
            self.data.values, idx_keep, idx_drop
        )

        inv_dd = invert_matrices(f_dd, method=method)
        reduced = f_kk - f_kd @ inv_dd @ f_dk
        return self._new_from_values(reduced, keep)

    def fix(self, parameters: str | Sequence[str]) -> "FisherMatrix":
        names = self._normalize_parameter_list(parameters)
        if not names:
            return self
        keep, _ = self._split_parameters(names)
        if not keep:
            raise ValueError("Cannot fix all parameters.")

        reduced_da = submatrix(self.data, keep)
        return self._new_from_values(reduced_da.values, keep)

    def transform(
        self,
        jacobian: np.ndarray | xr.DataArray,
        *,
        new_parameters: Sequence[str] | None = None,
    ) -> "FisherMatrix":
        jacobian_array, new_params = self._normalize_jacobian(jacobian, new_parameters)
        values = self.data.values
        transformed = np.matmul(
            np.matmul(jacobian_array, values), np.swapaxes(jacobian_array, -2, -1)
        )
        return self._new_from_values(transformed, new_params)

    def add_prior(
        self,
        *,
        diagonal: Mapping[str, float] | None = None,
        fisher: "FisherMatrix | xr.DataArray | xr.Dataset | None" = None,
    ) -> "FisherMatrix":
        n_provided = sum([diagonal is not None, fisher is not None])
        if n_provided != 1:
            raise ValueError("Specify exactly one of diagonal or fisher.")
        other_dataset: xr.Dataset | None = None
        if diagonal is not None:
            prior = self._diagonal_prior(diagonal)
        else:
            if isinstance(fisher, FisherMatrix):
                prior = fisher.data
                other_dataset = fisher.dataset
            elif isinstance(fisher, xr.Dataset):
                prior = fisher[FISHER_VAR]
                other_dataset = fisher
            else:
                prior = fisher
        if prior is None:
            raise ValueError("Prior DataArray is required.")
        return self._add_dataarrays(self.data, prior, other_dataset=other_dataset)

    def __add__(self, other: "FisherMatrix") -> "FisherMatrix":
        if self.matrix_dims != other.matrix_dims:
            raise ValueError("Row/col dims must match to add Fisher matrices.")
        return self._add_dataarrays(self.data, other.data, other_dataset=other.dataset)

    def figure_of_merit(self, parameters: Sequence[str] | None = None) -> xr.DataArray:
        if parameters is None:
            cov = self.covariance()
        else:
            names = self._normalize_parameter_list(parameters)
            sub = self.fix([p for p in self.parameters if p not in names])
            cov = sub.covariance()
        det = np.linalg.det(cov.values)
        fom = 1.0 / np.sqrt(det)
        return self._scalar_dataarray(fom)

    def snr(
        self,
        *,
        parameter: str,
        fiducial: float | None = None,
        deviation: float | None = None,
        method: str = "cholesky",
    ) -> xr.DataArray:
        if deviation is None:
            stored = self._metadata_value(self.fiducials, parameter)
            if fiducial is None or stored is None:
                raise ValueError(
                    "Provide deviation or both fiducial and stored fiducial."
                )
            deviation = fiducial - stored

        errors = self.marginalized_errors(method=method)
        sigma = errors.sel({self._parameter_dim: parameter}).values
        return self._scalar_dataarray(np.abs(deviation) / sigma)

    def sample(self, num_samples: int, *, method: str = "cholesky") -> xr.DataArray:
        if method != "cholesky":
            raise ValueError("Only cholesky sampling is supported.")

        values = self.data.values
        rng = np.random.default_rng()
        samples = sample_from_fisher(values, num_samples, rng)

        dims = self.batch_dims + ("sample", self._parameter_dim)
        coords: dict[str, np.ndarray] = {
            self._parameter_dim: np.asarray(self.parameters),
            "sample": np.arange(num_samples),
        }
        for dim in self.batch_dims:
            coords[dim] = self.data.coords[dim].values
        return xr.DataArray(samples, dims=dims, coords=coords)

    def format_constraint(
        self, parameter: str, *, fiducial: float | None = None, use_latex: bool = True
    ) -> str:
        if fiducial is None:
            fiducial = self._metadata_value(self.fiducials, parameter)
        sigma = self.marginalized_errors().sel({self._parameter_dim: parameter}).values
        label = self._metadata_value(self.labels, parameter) or parameter
        if use_latex:
            return f"${label} = {fiducial} \\pm {sigma}$"
        return f"{label} = {fiducial} +/- {sigma}"

    def format_summary(
        self,
        *,
        fiducials: Mapping[str, float] | None = None,
        units: Mapping[str, str] | None = None,
        use_latex: bool = False,
    ) -> str:
        """Format parameter constraints as a summary string.

        Parameters
        ----------
        fiducials : Mapping[str, float] | None
            Override fiducial values for specific parameters
        units : Mapping[str, str] | None
            Override units for specific parameters
        use_latex : bool
            If True, format output as LaTeX (default: False)

        Returns
        -------
        str
            Formatted summary string with one parameter per line
        """
        errors = self.marginalized_errors()
        lines: list[str] = []
        for name in self.parameters:
            fid = self._metadata_value(self.fiducials, name)
            if fiducials is not None and name in fiducials:
                fid = fiducials[name]
            err = errors.sel({self._parameter_dim: name}).values
            unit = self._metadata_value(self.units, name)
            if units is not None and name in units:
                unit = units[name]
            label = self._metadata_value(self.labels, name) or name
            if use_latex:
                line = f"${label} = {fid} \\pm {err}$"
            else:
                line = f"{label} = {fid} +/- {err}"
            if unit and not use_latex:
                line = f"{line} {unit}"
            lines.append(line)
        return "\n".join(lines)

    def print_summary(
        self,
        *,
        fiducials: Mapping[str, float] | None = None,
        units: Mapping[str, str] | None = None,
        use_latex: bool = False,
    ) -> None:
        """Print parameter constraints summary.

        Parameters
        ----------
        fiducials : Mapping[str, float] | None
            Override fiducial values for specific parameters
        units : Mapping[str, str] | None
            Override units for specific parameters
        use_latex : bool
            If True, format output as LaTeX (default: False)
        """
        summary = self.format_summary(
            fiducials=fiducials, units=units, use_latex=use_latex
        )
        print(summary)

    def save(self, path: str) -> None:
        if path.endswith(".zarr"):
            self._dataset.to_zarr(path, mode="w")
            return
        self._dataset.to_netcdf(path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        parameters: Sequence[str] | str | None = None,
        parameter_dim: str = DEFAULT_PARAMETER_DIM,
    ) -> "FisherMatrix":
        if path.endswith(".zarr"):
            ds = xr.open_dataset(path)
        else:
            ds = xr.open_dataset(path)
            if FISHER_VAR not in ds and len(ds.data_vars) == 1:
                da = xr.open_dataarray(path)
                return cls(
                    da,
                    parameters=parameters,
                    parameter_dim=parameter_dim,
                )
        return cls(ds, parameters=parameters, parameter_dim=parameter_dim)

    def _split_parameters(self, selected: Sequence[str]) -> tuple[list[str], list[str]]:
        selected_set = set(selected)
        keep = [name for name in self.parameters if name not in selected_set]
        drop = [name for name in self.parameters if name in selected_set]
        if len(drop) != len(selected):
            missing = sorted(set(selected) - set(self.parameters))
            raise KeyError(f"Unknown parameters: {missing}")
        return keep, drop

    def _normalize_parameter_list(self, parameters: str | Sequence[str]) -> list[str]:
        if isinstance(parameters, str):
            return [parameters]
        return list(parameters)

    @functools.cached_property
    def _parameter_index_map(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.parameters)}

    def _indices_for(self, names: Sequence[str]) -> list[int]:
        return [self._parameter_index_map[name] for name in names]

    def _batch_coords(self) -> dict[str, np.ndarray]:
        return {dim: self.data.coords[dim].values for dim in self.batch_dims}

    def _matrix_dataarray(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> xr.DataArray:
        row_dim, col_dim = self.matrix_dims
        dims = self.batch_dims + (row_dim, col_dim)
        coords = {
            **self._batch_coords(),
            row_dim: list(parameters),
            col_dim: list(parameters),
        }
        return xr.DataArray(values, dims=dims, coords=coords)

    def _vector_dataarray(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> xr.DataArray:
        dims = self.batch_dims + (self._parameter_dim,)
        coords = {
            **self._batch_coords(),
            self._parameter_dim: list(parameters),
        }
        return xr.DataArray(values, dims=dims, coords=coords)

    def _scalar_dataarray(self, values: np.ndarray) -> xr.DataArray:
        if not self.batch_dims:
            return xr.DataArray(values)
        return xr.DataArray(values, dims=self.batch_dims, coords=self._batch_coords())

    def _build_dataset_with_metadata(
        self,
        matrix: xr.DataArray,
        parameters: Sequence[str],
        *,
        other_dataset: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Build a Dataset with Fisher matrix and merged metadata."""
        params = list(parameters)
        ds = xr.Dataset({FISHER_VAR: matrix}, coords={self._parameter_dim: params})
        merged = self._merge_metadata_arrays(params, other_dataset=other_dataset)
        for key, arr in merged.items():
            ds[key] = xr.DataArray(
                arr,
                dims=(self._parameter_dim,),
                coords={self._parameter_dim: params},
            )
        return ds

    def _new_from_values(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> "FisherMatrix":
        data = self._matrix_dataarray(values, parameters)
        ds = self._build_dataset_with_metadata(data, parameters)
        return FisherMatrix(ds, parameter_dim=self._parameter_dim)

    def _add_dataarrays(
        self,
        left: xr.DataArray,
        right: xr.DataArray,
        *,
        other_dataset: xr.Dataset | None = None,
    ) -> "FisherMatrix":
        aligned_left, aligned_right = xr.align(
            left, right, join="outer", fill_value=0.0
        )
        summed = aligned_left + aligned_right
        row_dim = summed.dims[-2]
        params = list(summed.coords[row_dim].values)
        ds = self._build_dataset_with_metadata(
            summed, params, other_dataset=other_dataset
        )
        return FisherMatrix(ds, parameter_dim=self._parameter_dim)

    def _diagonal_prior(self, diagonal: Mapping[str, float]) -> xr.DataArray:
        params = self.parameters
        size = len(params)

        diag_values = np.zeros((size, size), dtype=float)
        indices = self._indices_for(list(diagonal.keys()))
        diag_values[indices, indices] = list(diagonal.values())

        if self.batch_dims:
            batch_shape = tuple(self.data.sizes[dim] for dim in self.batch_dims)
            diag_values = np.broadcast_to(diag_values, batch_shape + (size, size))
        return self._matrix_dataarray(diag_values, params)

    def _normalize_jacobian(
        self,
        jacobian: np.ndarray | xr.DataArray,
        new_parameters: Sequence[str] | None,
    ) -> tuple[np.ndarray, list[str]]:
        if isinstance(jacobian, xr.DataArray):
            if jacobian.ndim != 2:
                raise ValueError("Jacobian DataArray must be 2D.")
            old_dim = jacobian.dims[-1]
            old_params = list(jacobian.coords[old_dim].values)
            if old_params != self.parameters:
                jacobian = jacobian.sel({old_dim: self.parameters})
            new_dim = jacobian.dims[0]
            new_params = list(jacobian.coords[new_dim].values)
            return jacobian.values, new_params

        jacobian_array = np.asarray(jacobian)
        if jacobian_array.ndim != 2:
            raise ValueError("Jacobian array must be 2D.")
        if new_parameters is None:
            raise ValueError("new_parameters is required for array Jacobians.")
        return jacobian_array, list(new_parameters)

    def _sync_metadata(
        self,
        *,
        labels: Sequence[str] | Mapping[str, str] | None,
        units: Sequence[str | None] | Mapping[str, str | None] | None,
        fiducials: Sequence[float | None]
        | Mapping[str, float | None]
        | ArrayLike
        | None,
    ) -> None:
        if labels is not None:
            self.labels = labels
        elif LABELS_VAR in self._dataset:
            self._dataset[LABELS_VAR] = self._sanitize_metadata_da(
                self._dataset[LABELS_VAR], dtype=np.dtype(object)
            )

        if units is not None:
            self.units = units
        elif UNITS_VAR in self._dataset:
            self._dataset[UNITS_VAR] = self._sanitize_metadata_da(
                self._dataset[UNITS_VAR], dtype=np.dtype(object)
            )

        if fiducials is not None:
            self.fiducials = fiducials
        elif FIDUCIALS_VAR in self._dataset:
            self._dataset[FIDUCIALS_VAR] = self._sanitize_metadata_da(
                self._dataset[FIDUCIALS_VAR], dtype=np.dtype(float)
            )

    def _set_metadata_var(
        self,
        key: str,
        value,
        *,
        dtype: np.dtype,
    ) -> None:
        if value is None:
            if key in self._dataset:
                self._dataset = self._dataset.drop_vars(key)
            return
        values = normalize_metadata_array(
            self.parameters,
            value,
            fill_value=np.nan,
            dtype=dtype,
        )
        if values is None:
            return
        self._dataset[key] = xr.DataArray(
            values,
            dims=(self._parameter_dim,),
            coords={self._parameter_dim: self.parameters},
        )

    def _metadata_value(self, da: xr.DataArray | None, parameter: str) -> object | None:
        if da is None:
            return None
        if self._parameter_dim not in da.dims:
            raise ValueError("Metadata arrays must use the parameter dimension.")
        value = da.sel({self._parameter_dim: parameter}).item()
        if _is_missing(value):
            return None
        return value

    def _metadata_array(
        self,
        key: str,
        parameters: Sequence[str] | None = None,
        dataset: xr.Dataset | None = None,
    ) -> np.ndarray | None:
        ds = dataset or self._dataset
        if key not in ds:
            return None
        da = ds[key]
        if self._parameter_dim not in da.dims:
            raise ValueError(f"Metadata variable {key} must use parameter dim.")
        target = self.parameters if parameters is None else list(parameters)
        values = da.reindex({self._parameter_dim: target}).values
        return np.asarray(values)

    def _merge_metadata_arrays(
        self,
        parameters: Sequence[str],
        *,
        other_dataset: xr.Dataset | None = None,
    ) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}
        for key, dtype in (
            (LABELS_VAR, np.dtype(object)),
            (UNITS_VAR, np.dtype(object)),
            (FIDUCIALS_VAR, np.dtype(float)),
        ):
            left = self._metadata_array(key, parameters=parameters)
            right = self._metadata_array(
                key, parameters=parameters, dataset=other_dataset
            )
            if left is None and right is None:
                continue
            if left is None:
                merged[key] = np.asarray(right, dtype=dtype)
                continue
            if right is None:
                merged[key] = np.asarray(left, dtype=dtype)
                continue
            left = np.asarray(left, dtype=dtype)
            right = np.asarray(right, dtype=dtype)
            mask = np.array([not _is_missing(value) for value in right], dtype=bool)
            combined = left.copy()
            combined[mask] = right[mask]
            merged[key] = combined
        return merged

    def _sanitize_metadata_da(
        self, da: xr.DataArray, *, dtype: np.dtype
    ) -> xr.DataArray:
        if da.ndim != 1:
            raise ValueError("Metadata arrays must be 1D.")
        if self._parameter_dim not in da.dims:
            raise ValueError("Metadata arrays must use the parameter dimension.")
        values = da.reindex({self._parameter_dim: self.parameters}).values.tolist()
        values = [np.nan if value is None else value for value in values]
        return xr.DataArray(
            np.asarray(values, dtype=dtype),
            dims=(self._parameter_dim,),
            coords={self._parameter_dim: self.parameters},
        )
