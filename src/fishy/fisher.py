from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Diagnostics:
    min_eigenvalue: xr.DataArray
    max_eigenvalue: xr.DataArray
    condition_number: xr.DataArray
    is_degenerate: xr.DataArray
    method: str
    threshold: float | None


class FisherMatrix:
    def __init__(
        self,
        data: np.ndarray | xr.DataArray,
        *,
        parameters: Sequence[str] | None = None,
        labels: Sequence[str] | Mapping[str, str] | None = None,
        units: Sequence[str | None] | Mapping[str, str | None] | None = None,
        fiducials: Sequence[float | None] | Mapping[str, float | None] | None = None,
        parameter_dims: tuple[str, str] = ("parameter", "parameter2"),
        batch_dim: str = "batch",
    ) -> None:
        self._cache: dict[str, xr.DataArray] = {}
        if isinstance(data, xr.DataArray):
            da = data
            if da.ndim not in (2, 3):
                raise ValueError("DataArray must be 2D or 3D with a single batch dimension.")
            if all(dim in da.dims for dim in parameter_dims):
                self._parameter_dims = tuple(parameter_dims)
            else:
                self._parameter_dims = (da.dims[-2], da.dims[-1])
            self._batch_dim = da.dims[0] if da.ndim == 3 else None
            if parameters is None:
                parameters = self._extract_parameters_from_da(da)
            da = self._ensure_parameter_coords(da, list(parameters))
        else:
            self._parameter_dims = tuple(parameter_dims)
            self._batch_dim = batch_dim
            da = self._build_dataarray_from_array(
                np.asarray(data), list(parameters or []), parameter_dims, batch_dim
            )

        self._data = da
        self._sync_metadata(parameters=list(parameters or self.parameters), labels=labels, units=units, fiducials=fiducials)

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def parameters(self) -> list[str]:
        return list(self._data.coords[self._parameter_dims[0]].values.tolist())

    @property
    def parameter_names(self) -> list[str]:
        return self.parameters

    @property
    def parameter_dims(self) -> tuple[str, str]:
        return self._parameter_dims

    @property
    def labels(self) -> dict[str, str]:
        return self._get_attr_map("labels", default=lambda name: name)

    @labels.setter
    def labels(self, value: Sequence[str] | Mapping[str, str] | None) -> None:
        self._set_attr_map("labels", value, default=lambda name: name)

    @property
    def units(self) -> dict[str, str | None]:
        return self._get_attr_map("units", default=None)

    @units.setter
    def units(self, value: Sequence[str | None] | Mapping[str, str | None] | None) -> None:
        self._set_attr_map("units", value, default=None)

    @property
    def fiducials(self) -> dict[str, float | None]:
        return self._get_attr_map("fiducials", default=None)

    @fiducials.setter
    def fiducials(self, value: Sequence[float | None] | Mapping[str, float | None] | None) -> None:
        self._set_attr_map("fiducials", value, default=None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def covariance(self, method: str = "cholesky", *, rcond: float | None = None) -> xr.DataArray:
        cache_key = f"covariance:{method}:{rcond}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        inv = _invert_matrices(self._data.values, method=method, rcond=rcond)
        cov = self._matrix_dataarray(inv, self.parameters)
        self._cache[cache_key] = cov
        return cov

    def marginalized_errors(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        diag = np.diagonal(cov.values, axis=-2, axis2=-1)
        errors = np.sqrt(diag)
        return self._vector_dataarray(errors, self.parameters)

    def correlation(self, method: str = "cholesky") -> xr.DataArray:
        cov = self.covariance(method=method)
        diag = np.diagonal(cov.values, axis=-2, axis2=-1)
        denom = np.sqrt(diag[..., :, None] * diag[..., None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov.values / denom
        corr = np.where(np.isfinite(corr), corr, 0.0)
        if corr.ndim == 2:
            np.fill_diagonal(corr, 1.0)
        else:
            for idx in range(corr.shape[0]):
                np.fill_diagonal(corr[idx], 1.0)
        return self._matrix_dataarray(corr, self.parameters)

    def condition_number(self, method: str = "svd") -> xr.DataArray:
        values = _condition_numbers(self._data.values, method=method)
        return self._scalar_dataarray(values)

    def is_degenerate(
        self,
        *,
        min_eigenvalue: float | None = None,
        condition_number: float | None = None,
    ) -> xr.DataArray:
        if (min_eigenvalue is None) == (condition_number is None):
            raise ValueError("Specify exactly one of min_eigenvalue or condition_number.")
        if min_eigenvalue is not None:
            min_vals = _min_eigenvalues(self._data.values)
            return self._scalar_dataarray(min_vals <= min_eigenvalue)
        cond = _condition_numbers(self._data.values, method="svd")
        return self._scalar_dataarray(cond >= condition_number)

    def diagnostics(self, method: str = "svd", threshold: float | None = None) -> Diagnostics:
        min_vals, max_vals = _min_max_eigenvalues(self._data.values, method=method)
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

    def marginalize(self, parameters: str | Sequence[str], *, method: str = "cholesky") -> "FisherMatrix":
        names = self._normalize_parameter_list(parameters)
        if not names:
            return self
        keep, drop = self._split_parameters(names)
        if not keep:
            raise ValueError("Cannot marginalize all parameters.")

        values = self._data.values
        idx_keep = self._indices_for(keep)
        idx_drop = self._indices_for(drop)

        f_kk, f_kd, f_dd, f_dk = _partition_matrices(values, idx_keep, idx_drop)
        inv_dd = _invert_matrices(f_dd, method=method)
        reduced = f_kk - np.matmul(np.matmul(f_kd, inv_dd), f_dk)
        return self._new_from_values(reduced, keep)

    def fix(self, parameters: str | Sequence[str]) -> "FisherMatrix":
        names = self._normalize_parameter_list(parameters)
        if not names:
            return self
        keep, drop = self._split_parameters(names)
        if not keep:
            raise ValueError("Cannot fix all parameters.")

        values = self._data.values
        idx_keep = self._indices_for(keep)
        reduced = _submatrix(values, idx_keep, idx_keep)
        return self._new_from_values(reduced, keep)

    def transform(
        self,
        jacobian: np.ndarray | xr.DataArray,
        *,
        new_parameters: Sequence[str] | None = None,
    ) -> "FisherMatrix":
        jacobian_array, new_params = self._normalize_jacobian(jacobian, new_parameters)
        values = self._data.values
        transformed = np.matmul(np.matmul(jacobian_array, values), np.swapaxes(jacobian_array, -2, -1))
        return self._new_from_values(transformed, new_params)

    def add_prior(
        self,
        *,
        diagonal: Mapping[str, float] | None = None,
        fisher: "FisherMatrix | xr.DataArray | None" = None,
    ) -> "FisherMatrix":
        if (diagonal is None) == (fisher is None):
            raise ValueError("Specify exactly one of diagonal or fisher.")
        if diagonal is not None:
            prior = self._diagonal_prior(diagonal)
        else:
            prior = fisher.data if isinstance(fisher, FisherMatrix) else fisher
        return self._add_dataarrays(self._data, prior)

    def __add__(self, other: "FisherMatrix") -> "FisherMatrix":
        if self.parameter_dims != other.parameter_dims:
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
                raise ValueError("Provide deviation or both fiducial and stored fiducial.")
            deviation = fiducial - stored

        errors = self.marginalized_errors(method=method)
        sigma = errors.sel({self._parameter_dims[0]: parameter}).values
        return self._scalar_dataarray(np.abs(deviation) / sigma)

    def sample(self, num_samples: int, *, method: str = "cholesky") -> xr.DataArray:
        if method != "cholesky":
            raise ValueError("Only cholesky sampling is supported.")

        values = self._data.values
        rng = np.random.default_rng()
        samples = _sample_from_fisher(values, num_samples, rng)

        dims = ("sample", self._parameter_dims[0])
        coords = {self._parameter_dims[0]: self.parameters, "sample": np.arange(num_samples)}
        if self._batch_dim is not None:
            dims = (self._batch_dim,) + dims
            coords[self._batch_dim] = self._data.coords[self._batch_dim].values
        return xr.DataArray(samples, dims=dims, coords=coords)

    def format_constraint(self, parameter: str, *, fiducial: float | None = None, use_latex: bool = True) -> str:
        if fiducial is None:
            fiducial = self.fiducials.get(parameter)
        sigma = self.marginalized_errors().sel({self._parameter_dims[0]: parameter}).values
        label = self.labels.get(parameter, parameter)
        if use_latex:
            return f"${label} = {fiducial} \\pm {sigma}$"
        return f"{label} = {fiducial} +/- {sigma}"

    def print_summary(
        self,
        *,
        fiducials: Mapping[str, float] | None = None,
        units: Mapping[str, str] | None = None,
        use_latex: bool = False,
    ) -> str:
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
            err = errors.sel({self._parameter_dims[0]: name}).values
            unit = units_map.get(name)
            label = self.labels.get(name, name)
            if use_latex:
                line = f"${label} = {fid} \\pm {err}$"
            else:
                line = f"{label} = {fid} +/- {err}"
            if unit and not use_latex:
                line = f"{line} {unit}"
            lines.append(line)
        summary = "\n".join(lines)
        print(summary)
        return summary

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
        parameter_dims: tuple[str, str] | None = None,
        batch_dim: str = "batch",
    ) -> "FisherMatrix":
        if path.endswith(".zarr"):
            da = xr.open_zarr(path)
        else:
            da = xr.open_dataarray(path)
        if parameter_dims is None:
            if da.ndim < 2:
                raise ValueError("DataArray must be at least 2D.")
            parameter_dims = (da.dims[-2], da.dims[-1])
        return cls(da, parameter_dims=parameter_dims, batch_dim=batch_dim)

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

    def _matrix_dataarray(self, values: np.ndarray, parameters: Sequence[str]) -> xr.DataArray:
        dims = self._parameter_dims
        coords = {dims[0]: parameters, dims[1]: parameters}
        if self._batch_dim is not None:
            dims = (self._batch_dim,) + dims
            coords[self._batch_dim] = self._data.coords[self._batch_dim].values
        return xr.DataArray(values, dims=dims, coords=coords)

    def _vector_dataarray(self, values: np.ndarray, parameters: Sequence[str]) -> xr.DataArray:
        dims = (self._parameter_dims[0],)
        coords = {self._parameter_dims[0]: parameters}
        if self._batch_dim is not None:
            dims = (self._batch_dim,) + dims
            coords[self._batch_dim] = self._data.coords[self._batch_dim].values
        return xr.DataArray(values, dims=dims, coords=coords)

    def _scalar_dataarray(self, values: np.ndarray) -> xr.DataArray:
        if self._batch_dim is None:
            return xr.DataArray(values)
        return xr.DataArray(values, dims=(self._batch_dim,), coords={self._batch_dim: self._data.coords[self._batch_dim].values})

    def _new_from_values(self, values: np.ndarray, parameters: Sequence[str]) -> "FisherMatrix":
        labels = {name: self.labels.get(name, name) for name in parameters}
        units = {name: self.units.get(name) for name in parameters}
        fiducials = {name: self.fiducials.get(name) for name in parameters}
        return FisherMatrix(
            values,
            parameters=list(parameters),
            labels=labels,
            units=units,
            fiducials=fiducials,
            parameter_dims=self._parameter_dims,
            batch_dim=self._batch_dim or "batch",
        )

    def _add_dataarrays(self, left: xr.DataArray, right: xr.DataArray) -> "FisherMatrix":
        aligned_left, aligned_right = xr.align(left, right, join="outer", fill_value=0.0)
        summed = aligned_left + aligned_right
        params = list(summed.coords[self._parameter_dims[0]].values)

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
            parameter_dims=self._parameter_dims,
            batch_dim=self._batch_dim or "batch",
        )

    def _diagonal_prior(self, diagonal: Mapping[str, float]) -> xr.DataArray:
        params = self.parameters
        size = len(params)
        diag_values = np.zeros((size, size), dtype=float)
        for name, value in diagonal.items():
            idx = params.index(name)
            diag_values[idx, idx] = value
        if self._batch_dim is not None:
            diag_values = np.broadcast_to(diag_values, (self._data.sizes[self._batch_dim], size, size))
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
        parameters: Sequence[str],
        labels: Sequence[str] | Mapping[str, str] | None,
        units: Sequence[str | None] | Mapping[str, str | None] | None,
        fiducials: Sequence[float | None] | Mapping[str, float | None] | None,
    ) -> None:
        self.labels = labels if labels is not None else self._data.attrs.get("labels")
        self.units = units if units is not None else self._data.attrs.get("units")
        self.fiducials = fiducials if fiducials is not None else self._data.attrs.get("fiducials")

        if labels is None and "labels" not in self._data.attrs:
            self._set_attr_map("labels", None, default=lambda name: name)
        if units is None and "units" not in self._data.attrs:
            self._set_attr_map("units", None, default=None)
        if fiducials is None and "fiducials" not in self._data.attrs:
            self._set_attr_map("fiducials", None, default=None)

    def _get_attr_map(self, key: str, default) -> dict:
        raw = self._data.attrs.get(key, {})
        if not isinstance(raw, Mapping):
            raw = {}
        return {name: raw.get(name, default(name) if callable(default) else default) for name in self.parameters}

    def _set_attr_map(self, key: str, value, default) -> None:
        mapping = _normalize_meta(self.parameters, value, default)
        self._data.attrs[key] = mapping

    def _build_dataarray_from_array(
        self,
        array: np.ndarray,
        parameters: list[str],
        parameter_dims: tuple[str, str],
        batch_dim: str,
    ) -> xr.DataArray:
        if array.ndim not in (2, 3):
            raise ValueError("Data must be 2D or 3D with a single batch dimension.")
        if not parameters:
            raise ValueError("parameters are required when data is not an xarray DataArray.")
        if array.shape[-1] != array.shape[-2]:
            raise ValueError("Fisher matrix must be square.")
        if array.shape[-1] != len(parameters):
            raise ValueError("parameters length must match matrix size.")

        if array.ndim == 2:
            dims = parameter_dims
            coords = {parameter_dims[0]: parameters, parameter_dims[1]: parameters}
        else:
            dims = (batch_dim,) + parameter_dims
            coords = {batch_dim: np.arange(array.shape[0]), parameter_dims[0]: parameters, parameter_dims[1]: parameters}
        return xr.DataArray(array, dims=dims, coords=coords)

    def _extract_parameters_from_da(self, da: xr.DataArray) -> list[str]:
        dim = self._parameter_dims[0]
        if dim not in da.coords:
            raise ValueError(f"Missing coordinate for parameter dim {dim}.")
        return list(da.coords[dim].values)

    def _ensure_parameter_coords(self, da: xr.DataArray, parameters: list[str]) -> xr.DataArray:
        if da.sizes[self._parameter_dims[0]] != len(parameters):
            raise ValueError("parameters length must match matrix size.")
        if da.sizes[self._parameter_dims[1]] != len(parameters):
            raise ValueError("parameters length must match matrix size.")
        return da.assign_coords({self._parameter_dims[0]: parameters, self._parameter_dims[1]: parameters})


def _normalize_meta(names: Sequence[str], meta, default) -> dict[str, object]:
    if meta is None:
        return {name: default(name) if callable(default) else default for name in names}
    if isinstance(meta, Mapping):
        return {name: meta.get(name, default(name) if callable(default) else default) for name in names}
    values = list(meta)
    if len(values) != len(names):
        raise ValueError("Metadata length must match parameter length.")
    return {name: values[idx] for idx, name in enumerate(names)}


def _invert_matrices(values: np.ndarray, *, method: str, rcond: float | None = None) -> np.ndarray:
    if method == "inv":
        return np.linalg.inv(values)
    if method == "pinv":
        return np.linalg.pinv(values, rcond=rcond)
    if method == "svd":
        return _svd_inverse(values, rcond)
    if method != "cholesky":
        raise ValueError(f"Unknown inversion method: {method}")

    if values.ndim == 2:
        return _invert_cholesky(values)
    return np.stack([_invert_cholesky(mat) for mat in values], axis=0)


def _invert_cholesky(mat: np.ndarray) -> np.ndarray:
    identity = np.eye(mat.shape[0], dtype=mat.dtype)
    chol = np.linalg.cholesky(mat)
    y = np.linalg.solve(chol, identity)
    return np.linalg.solve(chol.T, y)


def _svd_inverse(values: np.ndarray, rcond: float | None) -> np.ndarray:
    if values.ndim == 2:
        return _svd_inverse_single(values, rcond)
    return np.stack([_svd_inverse_single(mat, rcond) for mat in values], axis=0)


def _svd_inverse_single(mat: np.ndarray, rcond: float | None) -> np.ndarray:
    u, s, vh = np.linalg.svd(mat)
    cutoff = (rcond or 0.0) * s.max()
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
    return (vh.T * s_inv) @ u.T


def _partition_matrices(
    values: np.ndarray, idx_keep: list[int], idx_drop: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_kk = _submatrix(values, idx_keep, idx_keep)
    f_kd = _submatrix(values, idx_keep, idx_drop)
    f_dd = _submatrix(values, idx_drop, idx_drop)
    f_dk = _submatrix(values, idx_drop, idx_keep)
    return f_kk, f_kd, f_dd, f_dk


def _submatrix(values: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    if values.ndim == 2:
        return values[np.ix_(rows, cols)]
    return np.take(np.take(values, rows, axis=1), cols, axis=2)


def _min_eigenvalues(values: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(values)
    return np.min(eigvals, axis=-1)


def _min_max_eigenvalues(values: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "eig":
        eigvals = np.linalg.eigvalsh(values)
        return np.min(eigvals, axis=-1), np.max(eigvals, axis=-1)
    if method != "svd":
        raise ValueError("method must be 'eig' or 'svd'")
    svals = np.linalg.svd(values, compute_uv=False)
    return np.min(svals, axis=-1), np.max(svals, axis=-1)


def _condition_numbers(values: np.ndarray, method: str) -> np.ndarray:
    if method == "svd":
        svals = np.linalg.svd(values, compute_uv=False)
        return np.max(svals, axis=-1) / np.min(svals, axis=-1)
    if method == "eig":
        eigvals = np.linalg.eigvalsh(values)
        return np.max(eigvals, axis=-1) / np.min(eigvals, axis=-1)
    raise ValueError("method must be 'eig' or 'svd'")


def _sample_from_fisher(values: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if values.ndim == 2:
        return _sample_single(values, num_samples, rng)
    return np.stack([_sample_single(mat, num_samples, rng) for mat in values], axis=0)


def _sample_single(mat: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    chol = np.linalg.cholesky(mat)
    z = rng.standard_normal((num_samples, mat.shape[0]))
    return np.linalg.solve(chol.T, z.T).T
