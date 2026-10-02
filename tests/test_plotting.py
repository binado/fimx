"""Public plotting behavior and analytic Gaussian constraints."""

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
from matplotlib.figure import Figure
from typer.testing import CliRunner

from fimx import dataset, matrix, plot
from fimx.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Release figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def forecast(F: xr.DataArray) -> xr.Dataset:
    """Return a forecast with distinct reference values."""
    return dataset(
        F,
        {
            "fiducials": xr.DataArray(
                [1, 2, 3], dims="row", coords={"row": F.row.values}
            )
        },
    )


def test_overlay_and_immutability(forecast: xr.Dataset) -> None:
    shifted = forecast.copy(deep=True)
    shifted["fiducials"] += 2
    original = forecast.copy(deep=True)
    figure = plot({"first": forecast, "second": shifted})
    assert isinstance(figure, Figure)
    assert [text.get_text() for text in figure.legends[0].texts] == ["first", "second"]
    xr.testing.assert_identical(forecast, original)
    assert len(figure.axes) == 6


def test_overlay_centers(forecast: xr.Dataset) -> None:
    shifted = forecast.copy(deep=True)
    shifted["fiducials"] += 2
    figure = plot({"first": forecast, "second": shifted}, parameters=["a"])
    means = []
    for line in figure.axes[0].lines:
        x, density = line.get_data()
        quadratic, linear, _ = np.polyfit(
            np.asarray(x, dtype=float), np.log(np.asarray(density, dtype=float)), 2
        )
        means.append(-linear / (2 * quadratic))
    np.testing.assert_allclose(means, [1, 3])


def test_intersection_order_and_direct_dataset(forecast: xr.Dataset) -> None:
    other = dataset(
        matrix([[2, 0], [0, 3]], ["c", "a"]),
        {"fiducials": xr.DataArray([3, 1], dims="row", coords={"row": ["c", "a"]})},
    )
    forecast["metadata"] = xr.DataArray("example")
    figure = plot({"first": forecast, "second": other}, filled=False)
    assert [ax.get_xlabel() for ax in figure.axes if ax.get_xlabel()] == ["a", "c"]


def test_marginalized_width_and_center() -> None:
    forecast = dataset(
        matrix([[2, 1], [1, 2]], ["a", "nuisance"]),
        {
            "fiducials": xr.DataArray(
                [5, 10], dims="row", coords={"row": ["a", "nuisance"]}
            )
        },
    )
    figure = plot({"survey": forecast}, parameters=["a"])
    x, density = figure.axes[0].lines[0].get_data()
    # The analytic Gaussian's log density is quadratic. Recover its mean and
    # variance without relying on the backend's density grid or normalization.
    quadratic, linear, _ = np.polyfit(
        np.asarray(x, dtype=float), np.log(np.asarray(density, dtype=float)), 2
    )
    np.testing.assert_allclose(-linear / (2 * quadratic), 5)
    np.testing.assert_allclose(-1 / (2 * quadratic), 2 / 3)


def test_explicit_order_and_customization(forecast: xr.Dataset) -> None:
    figure = plot(
        {"survey": forecast},
        parameters=["c", "a"],
        backend_kwargs={"contour_colors": ["red"], "line_args": [{"color": "red"}]},
    )
    assert [ax.get_xlabel() for ax in figure.axes if ax.get_xlabel()] == ["c", "a"]
    assert any(line.get_color() == "red" for ax in figure.axes for line in ax.lines)


@pytest.mark.parametrize("option", ["roots", "params", "legend_labels", "filled"])
def test_reserved_options(forecast: xr.Dataset, option: str) -> None:
    with pytest.raises(TypeError, match="cannot override"):
        plot({"survey": forecast}, backend_kwargs={option: None})


@pytest.mark.parametrize("parameters", [[], ["a", "a"], "a"])
def test_invalid_selections(forecast: xr.Dataset, parameters: list[str] | str) -> None:
    with pytest.raises(ValueError):
        plot({"survey": forecast}, parameters=parameters)


def test_plot_failures(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="At least one"):
        plot({})
    with pytest.raises(ValueError, match="Unknown plotting backend"):
        plot({"survey": forecast}, backend="corner")
    with pytest.raises(KeyError):
        plot({"survey": forecast}, parameters=["missing"])
    with pytest.raises(TypeError):
        plot({"survey": forecast.fisher})
    with pytest.raises(TypeError):
        plot([forecast])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must contain"):
        plot({"survey": forecast.drop_vars("fiducials")})
    invalid = forecast.copy(deep=True)
    invalid["fiducials"] = invalid.fiducials.astype(float)
    invalid.fiducials.values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plot({"survey": invalid})
    invalid = forecast.copy(deep=True)
    invalid.fisher.values[:] = 0
    with pytest.raises(np.linalg.LinAlgError):
        plot({"survey": invalid})
    unrelated = forecast.assign_coords(row=["x", "y", "z"], col=["x", "y", "z"])
    with pytest.raises(ValueError, match="no shared"):
        plot({"survey": forecast, "other": unrelated})


@pytest.mark.parametrize(
    "values", [[1, np.inf, 3], [1j, 2j, 3j], ["a", "b", "c"], np.eye(3)]
)
def test_plot_validates_fiducials(F: xr.DataArray, values: object) -> None:
    forecast = dataset(F, {"fiducials": values})  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        plot({"survey": forecast})


@pytest.mark.parametrize("blocked", ["getdist", "matplotlib"])
def test_import_without_optional_dependencies(blocked: str) -> None:
    # A fresh interpreter with an import finder simulates a core-only install.
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
F = fimx.matrix([[1]], ['a'])
ds = fimx.dataset(F, {'fiducials': [0]})
assert 'getdist' not in sys.modules
assert 'matplotlib' not in sys.modules
try:
    fimx.plot({'survey': ds})
except ImportError as exc:
    assert "uv add 'fimx[plotting]'" in str(exc)
else:
    raise AssertionError('Expected missing-dependency error')
"""
    subprocess.run(
        [sys.executable, "-c", script, blocked], check=True, env=os.environ.copy()
    )


@pytest.fixture
def degenerate_forecast() -> xr.Dataset:
    """Return a forecast where ``b`` and ``c`` are perfectly degenerate."""
    return dataset(
        matrix([[4, 0, 0], [0, 1, 1], [0, 1, 1]], ["a", "b", "c"]),
        {
            "fiducials": xr.DataArray(
                [0, 0, 0], dims="row", coords={"row": ["a", "b", "c"]}
            )
        },
    )


def test_plot_pinv_handles_degenerate_fisher(degenerate_forecast: xr.Dataset) -> None:
    figure = plot({"survey": degenerate_forecast}, parameters=["a"], method="pinv")
    assert isinstance(figure, Figure)


def test_plot_default_method_rejects_degenerate_fisher(
    degenerate_forecast: xr.Dataset,
) -> None:
    with pytest.raises(np.linalg.LinAlgError):
        plot({"survey": degenerate_forecast}, parameters=["a"])


def test_plot_unknown_method_raises(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="Unknown inversion method"):
        plot({"survey": forecast}, method="lu")  # ty: ignore[invalid-argument-type]


def test_cli_inversion_method_pinv_saves_degenerate_figure(
    degenerate_forecast: xr.Dataset, tmp_path: Path
) -> None:
    source = tmp_path / "survey.nc"
    degenerate_forecast.to_netcdf(source)
    figure_file = tmp_path / "out.png"
    argv = ["plot", "--file", str(source), "--figure-file", str(figure_file)]
    argv += ["--parameters", "a", "--inversion-method", "pinv"]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0
    assert figure_file.exists()


def test_cli_unknown_inversion_method_exits(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["plot", "--file", str(tmp_path / "x.nc"), "--inversion-method", "lu"]
    )
    assert result.exit_code != 0


def _labeled(forecast: xr.Dataset, labels: list[str]) -> xr.Dataset:
    return forecast.assign(labels=("row", np.array(labels, dtype=object)))


def _xlabels(figure: Figure) -> list[str]:
    return [ax.get_xlabel() for ax in figure.axes if ax.get_xlabel()]


def test_dataset_labels_are_used(forecast: xr.Dataset) -> None:
    figure = plot({"survey": _labeled(forecast, ["alpha", "beta", "gamma"])})
    assert all(
        label in " ".join(_xlabels(figure)) for label in ["alpha", "beta", "gamma"]
    )


def test_parameter_labels_override_dataset_labels(forecast: xr.Dataset) -> None:
    figure = plot(
        {"survey": _labeled(forecast, ["alpha", "beta", "gamma"])},
        parameter_labels={"a": "override"},
    )
    text = " ".join(_xlabels(figure))
    assert "override" in text
    assert "alpha" not in text
    assert "beta" in text


def test_conflicting_dataset_labels_raise(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="Conflicting plot labels"):
        plot(
            {
                "one": _labeled(forecast, ["alpha", "beta", "gamma"]),
                "two": _labeled(forecast, ["other", "beta", "gamma"]),
            }
        )


def test_conflicting_labels_of_unplotted_parameters_are_ignored(
    forecast: xr.Dataset,
) -> None:
    figure = plot(
        {
            "one": _labeled(forecast, ["alpha", "beta", "gamma"]),
            "two": _labeled(forecast, ["other", "beta", "gamma"]),
        },
        parameters=["b", "c"],
    )
    assert isinstance(figure, Figure)


@pytest.mark.parametrize("labels", [["a", "b", 3], ["x", "x", "y"]])
def test_invalid_dataset_labels_raise(forecast: xr.Dataset, labels: list) -> None:
    with pytest.raises(ValueError):
        plot({"survey": _labeled(forecast, labels)})


def test_cli_uses_dataset_labels(forecast: xr.Dataset, tmp_path: Path) -> None:
    source = tmp_path / "survey.nc"
    _labeled(forecast, ["alpha", "beta", "gamma"]).to_netcdf(source)
    figure_file = tmp_path / "out.png"
    result = runner.invoke(
        app, ["plot", "--file", str(source), "--figure-file", str(figure_file)]
    )
    assert result.exit_code == 0
    assert figure_file.exists()


def test_dataset_labels_with_dollar_signs_raise(forecast: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="must not include"):
        plot({"survey": _labeled(forecast, ["$a$", "$b$", "$c$"])})


def test_latex_dataset_labels_render(forecast: xr.Dataset) -> None:
    figure = plot({"survey": _labeled(forecast, [r"\Omega_m", "h", r"\sigma_8"])})
    figure.canvas.draw()
    assert r"$\Omega_m$" in _xlabels(figure)
