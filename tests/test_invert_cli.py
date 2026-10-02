import json
from pathlib import Path

import numpy as np
import xarray as xr

from fimx import inv, matrix
from fimx.invert_cli import main
from fimx.io import load_dataset, save_dataset


def test_invert_dataarray_default_methods_and_text_output(
    tmp_path: Path, capsys
) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "fisher.nc"
    F.to_netcdf(path, engine="h5netcdf")

    assert main(["--file", str(path)]) == 0

    output = capsys.readouterr().out
    assert "Condition number:" in output
    assert "Numerical rank: 2" in output
    assert "cholesky:" in output
    assert "inv:" in output
    assert "pinv:" in output


def test_invert_dataset_json_and_selected_method(tmp_path: Path, capsys) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    xr.Dataset({"fisher": F, "fiducials": ("row", [0.0, 1.0])}).to_netcdf(
        path, engine="h5netcdf"
    )

    assert main(["--file", str(path), "--inversion-method", "inv", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["matrix_size"] == 2
    assert report["parameters"] == ["a", "b"]
    assert report["positive_definite"] is True
    assert list(report["methods"]) == ["inv"]
    assert report["methods"]["inv"]["success"] is True
    assert report["methods"]["inv"]["max_abs_residual"] < 1e-12


def test_invert_reports_method_failure_and_continues(tmp_path: Path, capsys) -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    path = tmp_path / "singular.nc"
    F.to_netcdf(path, engine="h5netcdf")

    assert main(["--file", str(path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["rank"] == 1
    assert report["positive_definite"] is False
    assert report["methods"]["cholesky"]["success"] is False
    assert report["methods"]["inv"]["success"] is False
    assert report["methods"]["pinv"]["success"] is True


def test_invert_all_selected_methods_failure_returns_nonzero(
    tmp_path: Path, capsys
) -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    path = tmp_path / "indefinite.nc"
    F.to_netcdf(path, engine="h5netcdf")

    result = main(["--file", str(path), "--inversion-method", "cholesky", "--json"])

    assert result == 1
    report = json.loads(capsys.readouterr().out)
    assert report["methods"]["cholesky"]["success"] is False


def test_invert_nonfinite_condition_number_is_json_null(tmp_path: Path, capsys) -> None:
    F = matrix([[0, 0], [0, 0]], ["a", "b"])
    path = tmp_path / "zero.nc"
    F.to_netcdf(path, engine="h5netcdf")

    assert main(["--file", str(path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["condition_number"] is None


def _forecast(tmp_path: Path) -> tuple[Path, xr.DataArray]:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    save_dataset(path, F, fiducials=[0.0, 1.0], labels=["$a$", "$b$"])
    return path, F


def test_save_writes_covariance_and_keeps_other_variables(
    tmp_path: Path, capsys
) -> None:
    path, F = _forecast(tmp_path)
    out = tmp_path / "with_covariance.nc"

    assert main(["--file", str(path), "--save", str(out)]) == 0

    saved = load_dataset(out)
    np.testing.assert_allclose(saved["covariance"].values, inv(F).values)
    assert saved["covariance"].attrs["method"] == "cholesky"
    np.testing.assert_allclose(saved["fiducials"].values, [0.0, 1.0])
    assert saved["labels"].values.tolist() == ["$a$", "$b$"]
    assert "cholesky" in capsys.readouterr().err


def test_save_replaces_existing_covariance(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    path = tmp_path / "forecast.nc"
    save_dataset(path, F, covariance=inv(F, method="inv", metadata=True))
    out = tmp_path / "out.nc"

    main(["--file", str(path), "--save", str(out)])

    assert load_dataset(out)["covariance"].attrs["method"] == "cholesky"


def test_save_uses_first_successful_method(tmp_path: Path) -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    path = tmp_path / "singular.nc"
    F.to_netcdf(path, engine="h5netcdf")
    out = tmp_path / "out.nc"

    assert main(["--file", str(path), "--save", str(out)]) == 0

    assert load_dataset(out)["covariance"].attrs["method"] == "pinv"


def test_save_respects_selected_methods(tmp_path: Path) -> None:
    path, _ = _forecast(tmp_path)
    out = tmp_path / "out.nc"

    main(["--file", str(path), "--inversion-method", "pinv", "inv", "--save", str(out)])

    assert load_dataset(out)["covariance"].attrs["method"] == "inv"


def test_save_is_skipped_if_every_method_fails(tmp_path: Path) -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    path = tmp_path / "indefinite.nc"
    F.to_netcdf(path, engine="h5netcdf")
    out = tmp_path / "out.nc"

    result = main(
        ["--file", str(path), "--inversion-method", "cholesky", "--save", str(out)]
    )

    assert result == 1
    assert not out.exists()


def test_save_works_with_json_output(tmp_path: Path, capsys) -> None:
    path, _ = _forecast(tmp_path)
    out = tmp_path / "out.nc"

    main(["--file", str(path), "--json", "--save", str(out)])

    json.loads(capsys.readouterr().out)
    assert out.exists()
