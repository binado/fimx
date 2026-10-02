from collections.abc import Mapping

import numpy as np
import pytest
import xarray as xr

from fimx import combine, gaussian_prior, matrix


def test_identical_and_reordered_combination(F: xr.DataArray) -> None:
    reordered = F.sel(row=["c", "a", "b"], col=["c", "a", "b"])
    result = combine(F, F, reordered)
    xr.testing.assert_allclose(result, 3 * F)


def test_overlapping_and_disjoint_combination() -> None:
    first = matrix([[4, 1], [1, 2]], ["b", "a"])
    second = matrix([[3, 0.5], [0.5, 6]], ["c", "b"])
    third = matrix([[2, 0.2], [0.2, 5]], ["e", "d"])
    result = combine(first, second, third)
    assert (
        result.row.values.tolist()
        == result.col.values.tolist()
        == ["b", "a", "c", "e", "d"]
    )
    np.testing.assert_allclose(
        result.values,
        [
            [10, 1, 0.5, 0, 0],
            [1, 2, 0, 0, 0],
            [0.5, 0, 3, 0, 0],
            [0, 0, 0, 2, 0.2],
            [0, 0, 0, 0.2, 5],
        ],
    )


def test_zero_inputs() -> None:
    with pytest.raises(ValueError):
        combine()


def test_combination_does_not_fill_existing_invalid_values(F: xr.DataArray) -> None:
    bad = F.copy(deep=True)
    bad.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        combine(matrix([[1]], ["z"]), bad)


def test_gaussian_prior() -> None:
    result = gaussian_prior({"b": 0.5, "a": 2})
    xr.testing.assert_equal(result, matrix([[4, 0], [0, 0.25]], ["b", "a"]))
    F = matrix([[1, 0.5], [0.5, 1]], ["a", "b"])
    np.testing.assert_allclose(combine(F, result).values, [[1.25, 0.5], [0.5, 5]])


@pytest.mark.parametrize(
    "sigmas",
    [
        {},
        {"a": 0},
        {"a": -1},
        {"a": np.nan},
        {"a": np.inf},
        {"a": -np.inf},
        {"a": 1j},
        {"a": "1"},
        {0: 1},
        {"a": 1e-300},
    ],
)
def test_invalid_prior(sigmas: Mapping[str, float]) -> None:
    with pytest.raises(ValueError):
        gaussian_prior(sigmas)
