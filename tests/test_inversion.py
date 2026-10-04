from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from fimx import diagnose, errors, inv, matrix

METHODS = ["cholesky", "inv", "pinv"]


def test_labeled_inverse_and_errors(F: xr.DataArray) -> None:
    covariance = inv(F)
    expected = np.linalg.inv(F.values)
    np.testing.assert_allclose(covariance.values, expected)
    np.testing.assert_allclose(F.values @ covariance.values, np.eye(3), atol=1e-12)
    assert covariance.dims == ("row", "col")
    assert (
        covariance.row.values.tolist()
        == covariance.col.values.tolist()
        == ["a", "b", "c"]
    )
    np.testing.assert_array_equal(covariance.values, covariance.values.T)
    sigma = errors(F)
    np.testing.assert_allclose(sigma.values, np.sqrt(np.diag(expected)))
    assert sigma.dims == ("parameter",)
    assert set(sigma.coords) == {"parameter"}
    assert sigma.parameter.values.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("operation", [inv, errors])
@pytest.mark.parametrize(
    "values", [[[1, 1], [1, 1]], [[1, 0], [0, 0]], [[1, 2], [2, 1]]]
)
def test_cholesky_failure(
    operation: Callable[[xr.DataArray], xr.DataArray], values: list[list[float]]
) -> None:
    F = matrix(values, ["a", "b"])
    with pytest.raises(np.linalg.LinAlgError):
        operation(F)


def test_one_parameter() -> None:
    F = matrix([[4]], ["a"])
    np.testing.assert_allclose(inv(F).values, [[0.25]])
    np.testing.assert_allclose(errors(F).values, [0.5])


@pytest.mark.parametrize("method", METHODS)
def test_inv_positive_definite_methods_agree(F: xr.DataArray, method: str) -> None:
    covariance = inv(F, method=method)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(F.values @ covariance.values, np.eye(3), atol=1e-8)
    assert covariance.dims == ("row", "col")
    assert covariance.row.values.tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(covariance.values, covariance.values.T)


@pytest.mark.parametrize("method", METHODS)
def test_errors_positive_definite_methods_agree(F: xr.DataArray, method: str) -> None:
    sigma = errors(F, method=method)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(sigma.values, errors(F).values)


def test_pinv_singular_matrix_returns_pseudoinverse() -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    np.testing.assert_allclose(inv(F, method="pinv").values, np.full((2, 2), 0.25))


def test_pinv_unconstrained_direction_has_zero_variance() -> None:
    F = matrix([[4, 0], [0, 0]], ["a", "b"])
    np.testing.assert_allclose(inv(F, method="pinv").values, [[0.25, 0], [0, 0]])


def test_inv_indefinite_matrix_returns_inverse() -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    np.testing.assert_allclose(
        inv(F, method="inv").values, [[-1 / 3, 2 / 3], [2 / 3, -1 / 3]]
    )


def test_inv_singular_matrix_raises() -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    with pytest.raises(np.linalg.LinAlgError):
        inv(F, method="inv")


@pytest.mark.parametrize("operation", [inv, errors])
def test_unknown_method_raises(
    F: xr.DataArray, operation: Callable[..., xr.DataArray]
) -> None:
    with pytest.raises(ValueError, match="Unknown inversion method"):
        operation(F, method="lu")


def test_metadata_off_by_default(F: xr.DataArray) -> None:
    assert inv(F).attrs == {}


@pytest.mark.parametrize("method", METHODS)
def test_metadata_diagnostics(F: xr.DataArray, method: str) -> None:
    covariance = inv(F, method=method, metadata=True)  # ty: ignore[invalid-argument-type]
    eigenvalues = np.linalg.eigvalsh(F.values)
    assert covariance.attrs["method"] == method
    assert covariance.attrs["condition_number"] == pytest.approx(
        eigenvalues[-1] / eigenvalues[0]
    )
    assert covariance.attrs["residual"] < 1e-6


def test_metadata_does_not_change_values(F: xr.DataArray) -> None:
    xr.testing.assert_identical(inv(F, metadata=True).drop_attrs(), inv(F))


def test_metadata_condition_number_of_diagonal_matrix() -> None:
    F = matrix([[4, 0], [0, 1]], ["a", "b"])
    assert inv(F, metadata=True).attrs["condition_number"] == pytest.approx(4)


@pytest.mark.parametrize(
    "values", [[[1, 1], [1, 1]], [[4, 0], [0, 0]], [[1, 2], [2, 1]]]
)
def test_metadata_condition_number_infinite_if_not_positive_definite(
    values: list[list[float]],
) -> None:
    F = matrix(values, ["a", "b"])
    method = "inv" if values == [[1, 2], [2, 1]] else "pinv"
    covariance = inv(F, method=method, metadata=True)
    assert covariance.attrs["condition_number"] == np.inf


def test_diagnose_matches_metadata(F: xr.DataArray) -> None:
    diagnosis = diagnose(F)
    covariance = inv(F, metadata=True)
    assert diagnosis["method"].item() == covariance.attrs["method"]
    assert bool(diagnosis["success"].item())
    assert diagnosis["error"].item() == ""
    assert float(diagnosis["condition_number"]) == pytest.approx(
        covariance.attrs["condition_number"]
    )
    assert float(diagnosis["residual"]) == pytest.approx(covariance.attrs["residual"])
    assert diagnosis.parameter.values.tolist() == ["a", "b", "c"]
    assert diagnosis["eigenvalues"].dims == ("index",)
    assert bool(diagnosis["positive_definite"].item())
    assert int(diagnosis["rank"]) == 3
    np.testing.assert_allclose(
        diagnosis["eigenvalues"].values, np.linalg.eigvalsh(F.values)
    )
    assert set(diagnosis.data_vars) == {
        "eigenvalues",
        "condition_number",
        "rank",
        "positive_definite",
        "positive_semidefinite",
        "residual",
        "success",
        "method",
        "error",
    }


def test_diagnose_keeps_spectrum_when_cholesky_fails() -> None:
    F = matrix([[1, 2], [2, 1]], ["a", "b"])
    diagnosis = diagnose(F)
    assert not bool(diagnosis["success"].item())
    assert str(diagnosis["error"].item())
    assert np.isnan(float(diagnosis["residual"]))
    assert diagnosis["eigenvalues"].sizes["index"] == 2
    assert not bool(diagnosis["positive_definite"].item())
    assert diagnosis["condition_number"].item() == np.inf


def test_diagnose_pinv_of_singular_matrix() -> None:
    F = matrix([[1, 1], [1, 1]], ["a", "b"])
    diagnosis = diagnose(F, method="pinv")
    assert bool(diagnosis["success"].item())
    assert int(diagnosis["rank"]) == 1
    assert diagnosis["error"].item() == ""
    assert np.isfinite(float(diagnosis["residual"]))


def test_diagnose_unknown_method_raises(F: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="Unknown inversion method"):
        diagnose(F, method="lu")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("diagonal", "positive_definite", "positive_semidefinite", "rank"),
    [
        ([1.0, 2.0], True, True, 2),
        ([0.0, 1.0], False, True, 1),
        ([0.0, 0.0], False, True, 0),
        ([-1.0, 2.0], False, False, 2),
        ([-np.finfo(np.float64).eps, 1.0], False, True, 1),
        ([-1e-12, 1.0], False, False, 2),
        ([-1.0, -2.0], False, False, 2),
    ],
)
@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_diagnose_definiteness_is_scale_relative(
    diagonal: list[float],
    positive_definite: bool,
    positive_semidefinite: bool,
    rank: int,
    scale: float,
) -> None:
    F = matrix(np.diag(diagonal) * scale, ["z", "a"])
    original = F.copy(deep=True)
    diagnosis = diagnose(F, method="pinv")
    assert diagnosis.positive_definite.item() is positive_definite
    assert diagnosis.positive_semidefinite.item() is positive_semidefinite
    assert diagnosis.positive_semidefinite.dims == ()
    assert diagnosis.positive_semidefinite.dtype == np.bool_
    assert diagnosis.success.item() is True
    assert diagnosis["rank"].item() == rank
    assert diagnosis.eigenvalues.dims == ("index",)
    assert set(diagnosis.coords) == {"parameter"}
    assert diagnosis.parameter.values.tolist() == ["z", "a"]
    np.testing.assert_allclose(diagnosis.eigenvalues.values / scale, sorted(diagonal))
    if positive_definite:
        assert diagnosis.condition_number.item() == pytest.approx(2)
    else:
        assert diagnosis.condition_number.item() == np.inf
    diagnosis.eigenvalues.values[0] = 99
    xr.testing.assert_identical(F, original)
