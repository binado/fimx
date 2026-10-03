import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from fimx import inv, matrix
from fimx.cli import app
from fimx.io import load_dataset, save_dataset

runner = CliRunner()


def test_invert_default_cholesky_and_text_output(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "fisher.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(app, ["invert", "--file", str(path)])

    assert result.exit_code == 0
    output = result.stdout
    assert "Condition number:" in output
    assert "Numerical rank: 2" in output
    assert "cholesky:" in output
    assert "inv:" not in output
    assert "pinv:" not in output


def test_invert_dataset_json_and_selected_method(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    xr.Dataset({"fisher": F, "fiducials": ("row", [0.0, 1.0])}).to_netcdf(
        path, engine="h5netcdf"
    )

    result = runner.invoke(
        app, ["invert", "--file", str(path), "--inversion-method", "inv", "--json"]
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["matrix_size"] == 2
    assert report["parameters"] == ["a", "b"]
    assert report["positive_definite"] is True
    assert report["method"] == "inv"
    assert report["success"] is True
    assert report["max_abs_residual"] < 1e-12
    assert report["error"] is None


def test_invert_selected_pinv_succeeds_on_singular_matrix(tmp_path: Path) -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    path = tmp_path / "singular.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(
        app,
        ["invert", "--file", str(path), "--inversion-method", "pinv", "--json"],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["rank"] == 1
    assert report["positive_definite"] is False
    assert report["method"] == "pinv"
    assert report["success"] is True


def test_invert_method_failure_returns_nonzero(tmp_path: Path) -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    path = tmp_path / "indefinite.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(
        app,
        ["invert", "--file", str(path), "--inversion-method", "cholesky", "--json"],
    )

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["method"] == "cholesky"
    assert report["success"] is False
    assert report["error"] is not None


def test_invert_unknown_method_exits(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "fisher.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(
        app, ["invert", "--file", str(path), "--inversion-method", "bogus"]
    )

    assert result.exit_code != 0


def test_invert_nonfinite_condition_number_is_json_null(tmp_path: Path) -> None:
    F = matrix([[0, 0], [0, 0]], ["a", "b"])
    path = tmp_path / "zero.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(app, ["invert", "--file", str(path), "--json"])

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["condition_number"] is None


def _forecast(tmp_path: Path) -> tuple[Path, xr.DataArray]:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    save_dataset(path, F, fiducials=[0.0, 1.0], labels=["a_1", "b_1"])
    return path, F


def test_save_writes_covariance_and_keeps_other_variables(tmp_path: Path) -> None:
    path, F = _forecast(tmp_path)
    out = tmp_path / "with_covariance.nc"

    result = runner.invoke(app, ["invert", "--file", str(path), "--save", str(out)])

    assert result.exit_code == 0
    saved = load_dataset(out)
    np.testing.assert_allclose(saved["covariance"].values, inv(F).values)
    assert saved["covariance"].attrs["method"] == "cholesky"
    np.testing.assert_allclose(saved["fiducials"].values, [0.0, 1.0])
    assert saved["labels"].values.tolist() == ["a_1", "b_1"]
    assert "cholesky" in result.stderr


def test_save_replaces_existing_covariance(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    save_dataset(path, F, covariance=inv(F, method="inv", metadata=True))
    out = tmp_path / "out.nc"

    runner.invoke(app, ["invert", "--file", str(path), "--save", str(out)])

    assert load_dataset(out)["covariance"].attrs["method"] == "cholesky"


def test_save_uses_selected_pinv_on_singular_matrix(tmp_path: Path) -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    path = tmp_path / "singular.nc"
    F.to_netcdf(path, engine="h5netcdf")
    out = tmp_path / "out.nc"

    result = runner.invoke(
        app,
        [
            "invert",
            "--file",
            str(path),
            "--inversion-method",
            "pinv",
            "--save",
            str(out),
        ],
    )

    assert result.exit_code == 0

    assert load_dataset(out)["covariance"].attrs["method"] == "pinv"


def test_save_is_skipped_if_method_fails(tmp_path: Path) -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    path = tmp_path / "indefinite.nc"
    F.to_netcdf(path, engine="h5netcdf")
    out = tmp_path / "out.nc"

    result = runner.invoke(
        app,
        [
            "invert",
            "--file",
            str(path),
            "--inversion-method",
            "cholesky",
            "--save",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert not out.exists()


def test_save_works_with_json_output(tmp_path: Path) -> None:
    path, _ = _forecast(tmp_path)
    out = tmp_path / "out.nc"

    result = runner.invoke(
        app, ["invert", "--file", str(path), "--json", "--save", str(out)]
    )

    assert result.exit_code == 0
    json.loads(result.stdout)
    assert out.exists()


def test_cli_without_typer_explains_the_extra() -> None:
    script = """
import importlib.abc
import sys

class BlockTyper(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "typer" or fullname.startswith("typer."):
            raise ModuleNotFoundError(name="typer")

sys.meta_path.insert(0, BlockTyper())
import fimx
from fimx.__main__ import main
assert "typer" not in sys.modules
try:
    main(["--help"])
except SystemExit as exc:
    assert "uv add 'fimx[cli]'" in str(exc.code)
else:
    raise AssertionError("Expected missing-extra error")
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())
