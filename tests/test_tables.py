"""LaTeX table of marginalized errors."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from fimx import latex_table, matrix
from fimx.cli import app
from fimx.io import save_dataset

runner = CliRunner()


@pytest.fixture
def forecast() -> xr.Dataset:
    """Provide a covariance with sigma equal to 2 and 3."""
    return xr.Dataset({"covariance": matrix([[4, -3], [-3, 9]], ["a", "b"])})


def test_values_order_and_input_unchanged(forecast: xr.Dataset) -> None:
    original = forecast.copy(deep=True)
    table = latex_table(forecast)
    assert r"\begin{tabular}" in table
    assert table.index("2.0") < table.index("3.0")
    xr.testing.assert_identical(forecast, original)


@pytest.mark.parametrize(
    ("variance", "digits", "expected"),
    [
        (0.012345**2, 1, "0.01"),
        (0.012345**2, 3, "0.0123"),
        (0.01, 2, "0.10"),
        (1.0, 1, "1"),
        (1e-10, 2, r"$1.0\times10^{-5}$"),
        (900.0, 1, r"$3\times10^{1}$"),
    ],
)
def test_significant_digits(variance: float, digits: int, expected: str) -> None:
    ds = xr.Dataset({"covariance": matrix([[variance, 0], [0, 1]], ["a", "b"])})
    assert expected in latex_table(ds, significant_digits=digits)


def test_labels_and_selection(forecast: xr.Dataset) -> None:
    forecast["labels"] = ("row", [r"\alpha", r"\beta"])
    table = latex_table(
        forecast, parameters=["b", "a"], parameter_labels={"a": r"\gamma"}
    )
    assert table.index(r"$\beta$") < table.index(r"$\gamma$")
    assert r"$\alpha$" not in table


def test_plain_names_are_escaped() -> None:
    ds = xr.Dataset({"covariance": matrix([[4, 0], [0, 9]], ["w_0", "w_a"])})
    assert r"w\_0" in latex_table(ds)


def test_fiducials(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="fiducials"):
        latex_table(forecast, fiducials=True)
    forecast["fiducials"] = ("row", [0.5, 1.5])
    table = latex_table(forecast, fiducials=True)
    assert "Fiducial" in table
    assert "0.50" in table
    assert "Fiducial" not in latex_table(forecast)


@pytest.mark.parametrize("digits", [0, -1, 1.5, True])
def test_invalid_digits(forecast: xr.Dataset, digits: Any) -> None:
    with pytest.raises(ValueError, match="significant_digits"):
        latex_table(forecast, significant_digits=digits)


def test_invalid_inputs(forecast: xr.Dataset) -> None:
    with pytest.raises(KeyError):
        latex_table(forecast, parameters=["z"])
    with pytest.raises(ValueError, match="covariance"):
        latex_table(forecast.drop_vars("covariance"))
    bad = xr.Dataset({"covariance": matrix([[1, 2], [2, 1]], ["a", "b"])})
    with pytest.raises(np.linalg.LinAlgError):
        latex_table(bad)


@pytest.fixture
def forecast_file(tmp_path: Path) -> Path:
    path = tmp_path / "forecast.nc"
    fisher = matrix([[0.25, 0], [0, 1 / 9]], ["a", "b"])
    save_dataset(
        path,
        fisher,
        covariance=matrix([[4, 0], [0, 9]], ["a", "b"]),
        fiducials=xr.DataArray([0.5, 1.5], dims="row", coords={"row": ["a", "b"]}),
    )
    return path


def test_cli_stdout_and_options(forecast_file: Path) -> None:
    result = runner.invoke(
        app,
        ["table", str(forecast_file), "--significant-digits", "3", "--fiducials"],
    )
    assert result.exit_code == 0
    assert "2.00" in result.stdout
    assert "Fiducial" in result.stdout


def test_cli_output_file(forecast_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "table.tex"
    result = runner.invoke(
        app, ["table", str(forecast_file), "-o", str(out), "--parameters", "b"]
    )
    assert result.exit_code == 0
    assert r"\begin{tabular}" in out.read_text()
    assert "3.0" in out.read_text()


def test_cli_errors(forecast_file: Path) -> None:
    result = runner.invoke(app, ["table", str(forecast_file), "--parameters", "z"])
    assert result.exit_code == 2
    assert "Error" in result.output
