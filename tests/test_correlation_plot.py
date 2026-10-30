"""Stored-covariance heatmap behavior."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from fimx import matrix, plot_correlation
from fimx.cli import app


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def forecast() -> xr.Dataset:
    """Provide a covariance without Fisher or fiducial variables."""
    return xr.Dataset({"covariance": matrix([[4, -3], [-3, 9]], ["a", "b"])})


def test_values_and_order(forecast: xr.Dataset) -> None:
    original = forecast.copy(deep=True)
    figure = plot_correlation(forecast)
    ax = figure.axes[0]
    np.testing.assert_allclose(
        np.asarray(ax.images[0].get_array()), [[1, -0.5], [-0.5, 1]]
    )
    assert ax.images[0].get_clim() == (-1, 1)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b"]
    xr.testing.assert_identical(forecast, original)
    figure.canvas.draw()


def test_labels_and_selection(forecast: xr.Dataset) -> None:
    forecast["labels"] = ("row", [r"\alpha", r"\beta"])
    figure = plot_correlation(
        forecast, parameters=["b", "a"], parameter_labels={"a": r"\gamma"}
    )
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
        r"$\beta$",
        r"$\gamma$",
    ]
    figure.canvas.draw()
    figure = plot_correlation(forecast, parameters=["b"])
    np.testing.assert_allclose(np.asarray(figure.axes[0].images[0].get_array()), [[1]])


@pytest.mark.parametrize("parameters", [[], ["a", "a"], "a"])
def test_invalid_selection(forecast: xr.Dataset, parameters: list[str] | str) -> None:
    with pytest.raises(ValueError):
        plot_correlation(forecast, parameters=parameters)


@pytest.mark.parametrize("values", [[[0, 0], [0, 1]], [[1, 2], [2, 1]]])
def test_invalid_variances_or_spectrum(values: list[list[int]]) -> None:
    with pytest.raises(np.linalg.LinAlgError):
        plot_correlation(xr.Dataset({"covariance": matrix(values, ["a", "b"])}))


@pytest.mark.parametrize("values", [[[1, np.nan], [np.nan, 1]], [[1, 2], [0, 1]]])
def test_invalid_covariance(forecast: xr.Dataset, values: list[list[float]]) -> None:
    forecast.covariance.values[:] = values
    with pytest.raises(ValueError):
        plot_correlation(forecast)


def test_missing_data_and_labels(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="covariance"):
        plot_correlation(xr.Dataset())
    with pytest.raises(KeyError):
        plot_correlation(forecast, parameters=["missing"])
    with pytest.raises(ValueError):
        plot_correlation(forecast, parameter_labels={"a": "$a$"})
    forecast["labels"] = ("row", ["same", "same"])
    with pytest.raises(ValueError):
        plot_correlation(forecast)


def test_cli(forecast: xr.Dataset, tmp_path: Path) -> None:
    source = tmp_path / "forecast.nc"
    output = tmp_path / "correlation.pdf"
    forecast.to_netcdf(source, engine="h5netcdf")
    result = CliRunner().invoke(
        app,
        [
            "plot-correlation",
            "--file",
            str(source),
            "--figure-file",
            str(output),
            "--parameters",
            "b",
            "--figure-dpi",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"%PDF")
    assert not plt.get_fignums()


@pytest.mark.parametrize("blocked", ["matplotlib", "getdist"])
def test_optional_imports(blocked: str) -> None:
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == sys.argv[1]:
            raise ModuleNotFoundError(name=sys.argv[1])
sys.meta_path.insert(0, BlockOptional())
import fimx
import xarray as xr
assert 'matplotlib' not in sys.modules
assert 'getdist' not in sys.modules
ds = xr.Dataset({'covariance': fimx.matrix([[1]], ['a'])})
if sys.argv[1] == 'matplotlib':
    try:
        fimx.plot_correlation(ds)
    except ImportError as exc:
        assert "uv add 'fimx[plotting]'" in str(exc)
    else:
        raise AssertionError('Expected ImportError')
else:
    figure = fimx.plot_correlation(ds)
    figure.canvas.draw()
"""
    subprocess.run(
        [sys.executable, "-c", script, blocked],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
