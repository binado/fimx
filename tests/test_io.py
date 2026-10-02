"""Saving and loading forecasts in the canonical Dataset schema."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fimx import inv, matrix
from fimx.io import load_dataset, save_dataset


def test_round_trip_with_all_fields(F: xr.DataArray, tmp_path: Path) -> None:
    path = tmp_path / "forecast.nc"
    covariance = inv(F, metadata=True)
    save_dataset(
        path,
        F,
        covariance=covariance,
        fiducials=[1.0, 2.0, 3.0],
        labels=["a_1", "b_1", "c_1"],
    )
    loaded = load_dataset(path)
    assert set(loaded.data_vars) == {"fisher", "covariance", "fiducials", "labels"}
    np.testing.assert_allclose(loaded["fisher"].values, F.values)
    np.testing.assert_allclose(loaded["covariance"].values, covariance.values)
    np.testing.assert_allclose(loaded["fiducials"].values, [1.0, 2.0, 3.0])
    assert loaded["labels"].values.tolist() == ["a_1", "b_1", "c_1"]
    assert loaded["covariance"].attrs["condition_number"] == pytest.approx(
        covariance.attrs["condition_number"]
    )


def test_fisher_only(F: xr.DataArray, tmp_path: Path) -> None:
    path = tmp_path / "forecast.nc"
    save_dataset(path, F)
    assert set(load_dataset(path).data_vars) == {"fisher"}


def test_covariance_is_not_computed(F: xr.DataArray, tmp_path: Path) -> None:
    path = tmp_path / "forecast.nc"
    save_dataset(path, F)
    assert "covariance" not in load_dataset(path)


def test_reordered_covariance_is_aligned(F: xr.DataArray, tmp_path: Path) -> None:
    path = tmp_path / "forecast.nc"
    covariance = inv(F).sel(row=["c", "a", "b"], col=["c", "a", "b"])
    save_dataset(path, F, covariance=covariance)
    loaded = load_dataset(path)
    assert loaded["covariance"].row.values.tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(loaded["covariance"].values, inv(F).values)


def test_pseudoinverse_of_singular_matrix_is_accepted(tmp_path: Path) -> None:
    F = matrix([[4, 0], [0, 0]], ["a", "b"])
    save_dataset(tmp_path / "forecast.nc", F, covariance=inv(F, method="pinv"))


def test_covariance_of_another_matrix_raises(F: xr.DataArray, tmp_path: Path) -> None:
    other = matrix(2 * F.values, ["a", "b", "c"])
    with pytest.raises(ValueError, match="not an inverse"):
        save_dataset(tmp_path / "forecast.nc", F, covariance=inv(other))


def test_covariance_with_other_parameters_raises(
    F: xr.DataArray, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="must match"):
        save_dataset(
            tmp_path / "forecast.nc", F, covariance=inv(F).sel(row=["a"], col=["a"])
        )


@pytest.mark.parametrize(
    "labels", [["a", "b"], ["a", "b", 3], ["a", "a", "b"], [], "abc"]
)
def test_invalid_labels_raise(F: xr.DataArray, tmp_path: Path, labels: object) -> None:
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "forecast.nc", F, labels=labels)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("fiducials", [[1, 2], [1, 2, np.nan], ["x", "y", "z"]])
def test_invalid_fiducials_raise(
    F: xr.DataArray, tmp_path: Path, fiducials: object
) -> None:
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "forecast.nc", F, fiducials=fiducials)  # ty: ignore[invalid-argument-type]


def test_invalid_fisher_raises(tmp_path: Path) -> None:
    broken = xr.DataArray(np.eye(2), dims=("row", "col"))
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "forecast.nc", broken)


def test_labels_with_dollar_signs_raise(F: xr.DataArray, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not include"):
        save_dataset(tmp_path / "forecast.nc", F, labels=["$a$", "$b$", "$c$"])
