from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import xarray as xr

from .array_ops import (
    build_dataarray_from_array,
    build_matrix_dataarray,
    build_scalar_dataarray,
    build_vector_dataarray,
    stack_batches,
)
from .diagnostics import (
    Diagnostics,
    condition_numbers,
    min_eigenvalues,
    min_max_eigenvalues,
)
from .dimensions import DatasetDims, ParameterDims
from .inversion import invert_matrices
from .metadata import normalize_metadata
from .sampling import sample_from_fisher

DEFAULT_DIMS = DatasetDims()


class FisherMatrix:
    def __init__(
        self,
        data: np.ndarray | xr.DataArray,
        *,
        parameters: Sequence[str] | None = None,
        labels: Sequence[str] | Mapping[str, str] | None = None,
        units: Sequence[str | None] | Mapping[str, str | None] | None = None,
        fiducials: Sequence[float | None] | Mapping[str, float | None] | None = None,
        parameter_dims: ParameterDims | None = None,
        batch_dim: Hashable | None = None,
        dims: DatasetDims | None = None,
    ) -> None:
        self._batch_dim: Hashable | None
        self._dims: DatasetDims
        # Cache uses tuple keys for type safety and flexibility
        self._cache: dict[tuple, xr.DataArray] = {}
        explicit_parameter_dims = parameter_dims is not None or dims is not None
        if dims is not None:
            if parameter_dims is not None or batch_dim is not None:
                raise ValueError("Use dims or parameter_dims/batch_dim, not both.")
            parameter_dims = dims.parameter_dims
            batch_dim = dims.batch

        if parameter_dims is None:
            parameter_dims = DEFAULT_DIMS.parameter_dims
        if batch_dim is None:
            batch_dim = DEFAULT_DIMS.batch

        if isinstance(data, xr.DataArray):
            da = data
            if da.ndim < 2:
                raise ValueError("DataArray must be at least 2D.")
            if explicit_parameter_dims:
                if not all(dim in da.dims for dim in parameter_dims):
                    raise ValueError("Parameter dims are missing from DataArray.")
            else:
                parameter_dims = (da.dims[-2], da.dims[-1])

            da = stack_batches(da, parameter_dims=parameter_dims, batch_dim=batch_dim)
            self._batch_dim = batch_dim if da.ndim == 3 else None
            self._dims = DatasetDims(
                parameter_i=parameter_dims[0],
                parameter_j=parameter_dims[1],
                batch=batch_dim,
            )
            if parameters is None:
                parameters = self._extract_parameters_from_da(da)
            da = self._ensure_parameter_coords(da, list(parameters))
        else:
            self._batch_dim = batch_dim
            if parameters is None:
                raise ValueError(
                    "parameters are required when data is not an xarray DataArray."
                )
            self._dims = DatasetDims(
                parameter_i=parameter_dims[0],
                parameter_j=parameter_dims[1],
                batch=batch_dim,
            )
            da = build_dataarray_from_array(
                np.asarray(data),
                list(parameters),
                parameter_dims=parameter_dims,
                batch_dim=batch_dim,
            )

        self._data = da
        self._sync_metadata(labels=labels, units=units, fiducials=fiducials)

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def parameters(self) -> list[str]:
        # .values returns numpy array, .tolist() converts to Python list
        return self._data.coords[self._dims.parameter_i].values.tolist()

    @property
    def parameter_dims(self) -> ParameterDims:
        return self._dims.parameter_dims

    @property
    def dims(self) -> DatasetDims:
        return self._dims

    @property
    def labels(self) -> dict[str, str]:
        return self._get_attr_map("labels", default_factory=lambda name: name)

    @labels.setter
    def labels(self, value: Sequence[str] | Mapping[str, str] | None) -> None:
        self._set_attr_map("labels", value, default_factory=lambda name: name)

    @property
    def units(self) -> dict[str, str | None]:
        return self._get_attr_map("units", default_value=None)

    @units.setter
    def units(
        self, value: Sequence[str | None] | Mapping[str, str | None] | None
    ) -> None:
        self._set_attr_map("units", value, default_value=None)

    @property
    def fiducials(self) -> dict[str, float | None]:
        return self._get_attr_map("fiducials", default_value=None)

    @fiducials.setter
    def fiducials(
        self, value: Sequence[float | None] | Mapping[str, float | None] | None
    ) -> None:
        self._set_attr_map("fiducials", value, default_value=None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def covariance(
        self, method: str = "cholesky", *, rcond: float | None = None
    ) -> xr.DataArray:
        # Use tuple for cache key to avoid string formatting issues
        cache_key = ("covariance", method, rcond)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        inv = invert_matrices(self._data.values, method=method, rcond=rcond)
        cov = self._matrix_dataarray(inv, self.parameters)
        self._cache[cache_key] = cov
        return cov

    def marginalized_errors(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        values = np.asarray(cov.values)
        diag = np.diagonal(values, axis1=-2, axis2=-1)
        errors = np.sqrt(diag)
        return self._vector_dataarray(errors, self.parameters)

    def correlation(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        values = np.asarray(cov.values)
        diag = np.diagonal(values, axis1=-2, axis2=-1)
        denom = np.sqrt(diag[..., :, None] * diag[..., None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov.values / denom

        # Set diagonal to 1.0 only where variance is non-zero (finite correlation)
        # Use numpy's fill_diagonal for 2D, or explicit indexing for batched
        n_params = corr.shape[-1]

        if corr.ndim == 2:
            # Simple case: use diagonal indexing
            diag_idx = np.arange(n_params)
            mask = np.isfinite(corr[diag_idx, diag_idx])
            corr[diag_idx[mask], diag_idx[mask]] = 1.0
        else:
            # Batched case: set diagonals for each batch where finite
            for batch_idx in range(corr.shape[0]):
                diag_idx = np.arange(n_params)
                mask = np.isfinite(corr[batch_idx, diag_idx, diag_idx])
                corr[batch_idx, diag_idx[mask], diag_idx[mask]] = 1.0

        return self._matrix_dataarray(corr, self.parameters)

    def condition_number(self, method: str = "svd") -> xr.DataArray:
        values = condition_numbers(self._data.values, method=method)
        return self._scalar_dataarray(values)

    def is_degenerate(
        self,
        *,
        min_eigenvalue: float | None = None,
        condition_number: float | None = None,
    ) -> xr.DataArray:
        # Check that exactly one parameter is provided
        n_provided = sum([min_eigenvalue is not None, condition_number is not None])
        if n_provided != 1:
            raise ValueError(
                "Specify exactly one of min_eigenvalue or condition_number."
            )
        if min_eigenvalue is not None:
            min_vals = min_eigenvalues(self._data.values)
            return self._scalar_dataarray(min_vals <= min_eigenvalue)
        cond = condition_numbers(self._data.values, method="svd")
        return self._scalar_dataarray(cond >= condition_number)

    def diagnostics(
        self, method: str = "svd", threshold: float | None = None
    ) -> Diagnostics:
        min_vals, max_vals = min_max_eigenvalues(self._data.values, method=method)
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

        # Use xarray's coordinate-based selection
        param_i, param_j = self._dims.parameter_i, self._dims.parameter_j
        f_kk = self._data.sel({param_i: keep, param_j: keep})
        f_kd = self._data.sel({param_i: keep, param_j: drop})
        f_dd = self._data.sel({param_i: drop, param_j: drop})
        f_dk = self._data.sel({param_i: drop, param_j: keep})

        # Compute Schur complement: F_kk - F_kd @ F_dd^-1 @ F_dk
        inv_dd = invert_matrices(f_dd.values, method=method)
        reduced = f_kk.values - f_kd.values @ inv_dd @ f_dk.values
        return self._new_from_values(reduced, keep)

    def fix(self, parameters: str | Sequence[str]) -> "FisherMatrix":
        names = self._normalize_parameter_list(parameters)
        if not names:
            return self
        keep, _ = self._split_parameters(names)
        if not keep:
            raise ValueError("Cannot fix all parameters.")

        # Use xarray's coordinate-based selection
        param_i, param_j = self._dims.parameter_i, self._dims.parameter_j
        reduced_da = self._data.sel({param_i: keep, param_j: keep})
        return self._new_from_values(reduced_da.values, keep)

    def transform(
        self,
        jacobian: np.ndarray | xr.DataArray,
        *,
        new_parameters: Sequence[str] | None = None,
    ) -> "FisherMatrix":
        jacobian_array, new_params = self._normalize_jacobian(jacobian, new_parameters)
        values = self._data.values
        transformed = np.matmul(
            np.matmul(jacobian_array, values), np.swapaxes(jacobian_array, -2, -1)
        )
        return self._new_from_values(transformed, new_params)

    def add_prior(
        self,
        *,
        diagonal: Mapping[str, float] | None = None,
        fisher: "FisherMatrix | xr.DataArray | None" = None,
    ) -> "FisherMatrix":
        # Check that exactly one parameter is provided
        n_provided = sum([diagonal is not None, fisher is not None])
        if n_provided != 1:
            raise ValueError("Specify exactly one of diagonal or fisher.")
        if diagonal is not None:
            prior = self._diagonal_prior(diagonal)
        else:
            prior = fisher.data if isinstance(fisher, FisherMatrix) else fisher
        if prior is None:
            raise ValueError("Prior DataArray is required.")
        return self._add_dataarrays(self._data, prior)

    def __add__(self, other: "FisherMatrix") -> "FisherMatrix":
        if self._dims.parameter_dims != other._dims.parameter_dims:
            raise ValueError("Parameter dims must match to add Fisher matrices.")
        return self._add_dataarrays(self._data, other.data)

    def figure_of_merit(self, parameters: Sequence[str]) -> xr.DataArray:
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
            stored = self.fiducials.get(parameter)
            if fiducial is None or stored is None:
                raise ValueError(
                    "Provide deviation or both fiducial and stored fiducial."
                )
            deviation = fiducial - stored

        errors = self.marginalized_errors(method=method)
        sigma = errors.sel({self._dims.parameter_i: parameter}).values
        return self._scalar_dataarray(np.abs(deviation) / sigma)

    def sample(self, num_samples: int, *, method: str = "cholesky") -> xr.DataArray:
        if method != "cholesky":
            raise ValueError("Only cholesky sampling is supported.")

        values = self._data.values
        rng = np.random.default_rng()
        samples = sample_from_fisher(values, num_samples, rng)

        dims = ("sample", self._dims.parameter_i)
        coords = {
            self._dims.parameter_i: self.parameters,
            "sample": np.arange(num_samples),
        }
        if self._batch_dim is not None:
            dims = (self._batch_dim,) + dims
            coords[self._batch_dim] = self._data.coords[self._batch_dim].values
        return xr.DataArray(samples, dims=dims, coords=coords)

    def format_constraint(
        self, parameter: str, *, fiducial: float | None = None, use_latex: bool = True
    ) -> str:
        if fiducial is None:
            fiducial = self.fiducials.get(parameter)
        sigma = (
            self.marginalized_errors().sel({self._dims.parameter_i: parameter}).values
        )
        label = self.labels.get(parameter, parameter)
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
        fiducials_map = dict(self.fiducials)
        if fiducials:
            fiducials_map.update(fiducials)
        units_map = dict(self.units)
        if units:
            units_map.update(units)

        errors = self.marginalized_errors()
        lines: list[str] = []
        for name in self.parameters:
            fid = fiducials_map.get(name)
            err = errors.sel({self._dims.parameter_i: name}).values
            unit = units_map.get(name)
            label = self.labels.get(name, name)
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
            self._data.to_zarr(path, mode="w")
            return
        self._data.to_netcdf(path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        parameter_dims: ParameterDims | None = None,
        batch_dim: Hashable | None = None,
        dims: DatasetDims | None = None,
    ) -> "FisherMatrix":
        if path.endswith(".zarr"):
            da = xr.open_zarr(path)
        else:
            da = xr.open_dataarray(path)
        return cls(da, parameter_dims=parameter_dims, batch_dim=batch_dim, dims=dims)

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

    def _indices_for(self, names: Sequence[str]) -> list[int]:
        index_map = {name: idx for idx, name in enumerate(self.parameters)}
        return [index_map[name] for name in names]

    def _get_batch_coords(self) -> np.ndarray | None:
        """Get batch coordinates if batch dimension exists.

        Returns
        -------
        np.ndarray | None
            Batch coordinates array, or None if no batch dimension
        """
        if self._batch_dim is None:
            return None
        return self._data.coords[self._batch_dim].values

    def _matrix_dataarray(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> xr.DataArray:
        batch_coords = self._get_batch_coords()
        return build_matrix_dataarray(
            values,
            parameters,
            parameter_dims=self._dims.parameter_dims,
            batch_dim=self._batch_dim or "batch",
            batch_coords=batch_coords,
        )

    def _vector_dataarray(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> xr.DataArray:
        batch_coords = self._get_batch_coords()
        return build_vector_dataarray(
            values,
            parameters,
            parameter_dim=self._dims.parameter_i,
            batch_dim=self._batch_dim or "batch",
            batch_coords=batch_coords,
        )

    def _scalar_dataarray(self, values: np.ndarray) -> xr.DataArray:
        batch_coords = self._get_batch_coords()
        return build_scalar_dataarray(
            values,
            batch_dim=self._batch_dim or "batch",
            batch_coords=batch_coords,
        )

    def _new_from_values(
        self, values: np.ndarray, parameters: Sequence[str]
    ) -> "FisherMatrix":
        labels = {name: self.labels.get(name, name) for name in parameters}
        units = {name: self.units.get(name) for name in parameters}
        fiducials = {name: self.fiducials.get(name) for name in parameters}
        return FisherMatrix(
            values,
            parameters=list(parameters),
            labels=labels,
            units=units,
            fiducials=fiducials,
            parameter_dims=self._dims.parameter_dims,
            batch_dim=self._batch_dim or "batch",
        )

    def _add_dataarrays(
        self, left: xr.DataArray, right: xr.DataArray
    ) -> "FisherMatrix":
        aligned_left, aligned_right = xr.align(
            left, right, join="outer", fill_value=0.0
        )
        summed = aligned_left + aligned_right
        params = list(summed.coords[self._dims.parameter_i].values)

        labels = dict(self.labels)
        labels.update(getattr(right, "attrs", {}).get("labels", {}))
        units = dict(self.units)
        units.update(getattr(right, "attrs", {}).get("units", {}))
        fiducials = dict(self.fiducials)
        fiducials.update(getattr(right, "attrs", {}).get("fiducials", {}))
        return FisherMatrix(
            summed,
            parameters=params,
            labels=labels,
            units=units,
            fiducials=fiducials,
            parameter_dims=self._dims.parameter_dims,
            batch_dim=self._batch_dim or "batch",
        )

    def _diagonal_prior(self, diagonal: Mapping[str, float]) -> xr.DataArray:
        params = self.parameters
        size = len(params)

        # Create diagonal matrix using vectorized indexing
        diag_values = np.zeros((size, size), dtype=float)
        indices = [params.index(name) for name in diagonal.keys()]
        values = list(diagonal.values())
        diag_values[indices, indices] = values

        if self._batch_dim is not None:
            diag_values = np.broadcast_to(
                diag_values, (self._data.sizes[self._batch_dim], size, size)
            )
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
        fiducials: Sequence[float | None] | Mapping[str, float | None] | None,
    ) -> None:
        self.labels = labels if labels is not None else self._data.attrs.get("labels")
        self.units = units if units is not None else self._data.attrs.get("units")
        self.fiducials = (
            fiducials if fiducials is not None else self._data.attrs.get("fiducials")
        )

        if labels is None and "labels" not in self._data.attrs:
            self._set_attr_map("labels", None, default_factory=lambda name: name)
        if units is None and "units" not in self._data.attrs:
            self._set_attr_map("units", None, default_value=None)
        if fiducials is None and "fiducials" not in self._data.attrs:
            self._set_attr_map("fiducials", None, default_value=None)

    def _get_attr_map(
        self,
        key: str,
        *,
        default_factory=None,
        default_value=None,
    ) -> dict:
        raw = self._data.attrs.get(key)
        if isinstance(raw, str):
            raw = None
        if raw is not None and not isinstance(raw, (Mapping, Sequence)):
            raw = None
        return normalize_metadata(
            self.parameters,
            raw,
            default_factory=default_factory,
            default_value=default_value,
        )

    def _set_attr_map(
        self,
        key: str,
        value,
        *,
        default_factory=None,
        default_value=None,
    ) -> None:
        mapping = normalize_metadata(
            self.parameters,
            value,
            default_factory=default_factory,
            default_value=default_value,
        )
        self._data.attrs[key] = mapping

    def _extract_parameters_from_da(self, da: xr.DataArray) -> list[str]:
        dim = self._dims.parameter_i
        if dim not in da.coords:
            raise ValueError(f"Missing coordinate for parameter dim {dim}.")
        return list(da.coords[dim].values)

    def _ensure_parameter_coords(
        self, da: xr.DataArray, parameters: list[str]
    ) -> xr.DataArray:
        if da.sizes[self._dims.parameter_i] != len(parameters):
            raise ValueError("parameters length must match matrix size.")
        if da.sizes[self._dims.parameter_j] != len(parameters):
            raise ValueError("parameters length must match matrix size.")
        return da.assign_coords(
            {self._dims.parameter_i: parameters, self._dims.parameter_j: parameters}
        )
