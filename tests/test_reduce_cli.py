"""CLI coverage for fixing and marginalizing forecast parameters."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from fimx import fix, inv, marginalize, matrix
from fimx.cli import app
from fimx.io import load_dataset

runner = CliRunner()


@pytest.fixture(params=["fix", "marginalize"])
def command(request: pytest.FixtureRequest) -> str:
    """Return each parameter removal command."""
    return str(request.param)


@pytest.fixture
def forecast(tmp_path: Path, F: xr.DataArray) -> Path:
    """Save a forecast with covariance, metadata, and an unrelated variable."""
    path = tmp_path / "forecast.nc"
    F.attrs["units"] = "example"
    dataset = xr.Dataset(
        {
            "fisher": F,
            "covariance": inv(F),
            "fiducials": ("row", [1.0, 2.0, 3.0]),
            "labels": ("row", ["alpha", "beta", "gamma"]),
            "custom": F * 2,
            "version": 1,
        },
        attrs={"survey": "example"},
    )
    dataset["fiducials"].attrs["description"] = "reference values"
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


@pytest.mark.parametrize("output_option", ["-o", "--output-file"])
@pytest.mark.parametrize("parameters", [["b"], ["c", "a"]])
def test_reduce_dataset(
    command: str, forecast: Path, output_option: str, parameters: list[str]
) -> None:
    original_bytes = forecast.read_bytes()
    original = load_dataset(forecast)
    output = forecast.with_name("reduced.nc")
    arguments = [command, str(forecast), output_option, str(output)]
    for parameter in parameters:
        arguments.extend(["--parameters", parameter])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    saved = load_dataset(output)
    operation = fix if command == "fix" else marginalize
    expected = operation(original["fisher"], parameters)
    np.testing.assert_allclose(saved["fisher"].values, expected.values)
    retained = expected.row.values.tolist()
    assert saved.row.values.tolist() == retained
    assert saved.col.values.tolist() == retained
    for name in ["fiducials", "labels", "custom", "version"]:
        xr.testing.assert_identical(
            saved[name], original.sel(row=retained, col=retained)[name]
        )
    assert saved.attrs == original.attrs
    assert saved["fisher"].attrs == original["fisher"].attrs
    assert "covariance" not in saved
    assert forecast.read_bytes() == original_bytes


@pytest.mark.parametrize("parameters", [["unknown"], ["a", "a"], ["a", "b", "c"]])
def test_invalid_parameters_leave_no_output(
    command: str, forecast: Path, parameters: list[str]
) -> None:
    output = forecast.with_name("reduced.nc")
    arguments = [command, str(forecast), "-o", str(output)]
    for parameter in parameters:
        arguments.extend(["--parameters", parameter])

    result = runner.invoke(app, arguments)

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not output.exists()


@pytest.mark.parametrize("missing", ["file", "parameters", "output"])
def test_required_arguments(command: str, forecast: Path, missing: str) -> None:
    output = forecast.with_name("reduced.nc")
    arguments = [command]
    if missing != "file":
        arguments.append(str(forecast))
    if missing != "parameters":
        arguments.extend(["--parameters", "a"])
    if missing != "output":
        arguments.extend(["-o", str(output)])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert not output.exists()


@pytest.mark.parametrize("same_as_input", [False, True])
def test_existing_output_is_preserved(
    command: str, forecast: Path, same_as_input: bool
) -> None:
    output = forecast if same_as_input else forecast.with_name("existing.nc")
    if not same_as_input:
        output.write_bytes(b"existing output")
    original_bytes = output.read_bytes()

    result = runner.invoke(
        app, [command, str(forecast), "--parameters", "a", "-o", str(output)]
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_bytes() == original_bytes


@pytest.mark.parametrize("missing_file", [False, True])
def test_invalid_input_leaves_no_output(
    command: str, tmp_path: Path, missing_file: bool
) -> None:
    path = tmp_path / "input.nc"
    output = tmp_path / "reduced.nc"
    if not missing_file:
        xr.Dataset({"other": ("row", [1.0])}).to_netcdf(path, engine="h5netcdf")

    result = runner.invoke(
        app, [command, str(path), "--parameters", "a", "-o", str(output)]
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not output.exists()


def test_marginalize_failure_leaves_no_output(tmp_path: Path) -> None:
    path = tmp_path / "singular.nc"
    output = tmp_path / "reduced.nc"
    xr.Dataset({"fisher": matrix([[1, 0], [0, 0]], ["a", "b"])}).to_netcdf(
        path, engine="h5netcdf"
    )

    result = runner.invoke(
        app, ["marginalize", str(path), "--parameters", "b", "-o", str(output)]
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not output.exists()
