from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from fimx import errors, inv, matrix


def test_labeled_inverse_and_errors(F: xr.DataArray) -> None:
    covariance = inv(F)
    expected = np.linalg.inv(F.values)
    np.testing.assert_allclose(covariance.values, expected)
    np.testing.assert_allclose(F.values @ covariance.values, np.eye(3), atol=1e-15)
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
