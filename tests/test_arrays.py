from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest
import xarray as xr
from numpy.typing import ArrayLike

import fimx
from fimx import (
    combine,
    errors,
    expand,
    fix,
    gaussian_prior,
    inv,
    marginalize,
    matrix,
    transform,
    vector,
)


def test_public_api() -> None:
    assert set(fimx.__all__) == {
        "matrix",
        "vector",
        "fix",
        "marginalize",
        "inv",
        "errors",
        "expand",
        "transform",
        "combine",
        "correlation",
        "symmetrize",
        "fom",
        "diagnose",
        "gaussian_prior",
        "dataset",
        "plot",
    }
    assert not hasattr(fimx, "FisherMatrix")


def test_construction() -> None:
    values = np.array([[4, 1], [1, 2]], dtype=np.int32)
    parameters = ["a", "b"]
    result = matrix(values, parameters)
    assert result.dims == ("row", "col")
    assert set(result.coords) == {"row", "col"}
    assert result.dtype == np.float64
    assert result.row.values.tolist() == result.col.values.tolist() == parameters
    np.testing.assert_array_equal(result.values, values)
    assert not np.shares_memory(result.values, values)
    result.values[0, 0] = 10
    result.coords["row"].values[0] = "z"
    assert values[0, 0] == 4
    assert parameters == ["a", "b"]


def test_vector_construction() -> None:
    values = np.array([4, 1], dtype=np.int32)
    parameters = ["a", "b"]
    result = vector(values, parameters)
    assert result.dims == ("row",)
    assert set(result.coords) == {"row"}
    assert result.dtype == values.dtype
    assert result.row.values.tolist() == parameters
    np.testing.assert_array_equal(result.values, values)
    assert not np.shares_memory(result.values, values)
    result.values[0] = 10
    result.coords["row"].values[0] = "z"
    assert values[0] == 4
    assert parameters == ["a", "b"]


@pytest.mark.parametrize(
    ("values", "parameters"),
    [
        (np.ones((2, 2)), ["a", "b"]),
        ([1, 2], ["a"]),
        ([1], ["a", "b"]),
        ([], []),
        ([1, 2], ["a", "a"]),
        ([1, 2], [1, 2]),
        ([1], "a"),
    ],
)
def test_invalid_vector_construction(
    values: ArrayLike, parameters: Sequence[str]
) -> None:
    with pytest.raises(ValueError):
        vector(values, parameters)


@pytest.mark.parametrize(
    ("values", "parameters"),
    [
        (np.ones((2, 2, 2)), ["a", "b"]),
        ([1, 2], ["a", "b"]),
        (np.ones((2, 3)), ["a", "b"]),
        (np.eye(2), ["a"]),
        (np.empty((0, 0)), []),
        (np.eye(2), ["a", "a"]),
        (np.eye(2), [1, 2]),
        ([[1]], "a"),
        (np.eye(2, dtype=complex), ["a", "b"]),
        ([["1", "0"], ["0", "1"]], ["a", "b"]),
        (np.eye(2, dtype=object), ["a", "b"]),
        (np.eye(2).astype("timedelta64[s]"), ["a", "b"]),
        ([[1, np.nan], [np.nan, 1]], ["a", "b"]),
        ([[1, 0], [0, np.inf]], ["a", "b"]),
        ([[1, 2], [0, 1]], ["a", "b"]),
    ],
)
def test_invalid_construction(values: ArrayLike, parameters: Sequence[str]) -> None:
    with pytest.raises(ValueError):
        matrix(values, parameters)


def test_symmetry_tolerance() -> None:
    accepted = matrix([[2, 1], [1 + 5e-11, 2]], ["a", "b"])
    np.testing.assert_array_equal(fix(accepted, []).values, accepted.values)
    with pytest.raises(ValueError, match="symmetric"):
        matrix([[2, 1], [1 + 5e-9, 2]], ["a", "b"])
    matrix([[1, 0], [5e-13, 1]], ["a", "b"])
    with pytest.raises(ValueError, match="symmetric"):
        matrix([[1, 0], [5e-11, 1]], ["a", "b"])


def _invalid_matrices() -> list[xr.DataArray]:
    """Build malformed arrays without using fimx construction."""
    coords = {"row": ["a", "b"], "col": ["a", "b"]}
    return [
        xr.DataArray(np.eye(2), dims=("x", "y")),
        xr.DataArray(np.eye(2), dims=("col", "row"), coords=coords),
        xr.DataArray(np.ones((1, 2, 2)), dims=("batch", "row", "col"), coords=coords),
        xr.DataArray(np.ones((2, 3)), dims=("row", "col")),
        xr.DataArray(np.empty((0, 0)), dims=("row", "col")),
        xr.DataArray(np.eye(2), dims=("row", "col")),
        xr.DataArray(np.eye(2), dims=("row", "col"), coords={"row": ["a", "b"]}),
        xr.DataArray(np.eye(2), dims=("row", "col"), coords={"col": ["a", "b"]}),
        xr.DataArray(
            np.eye(2),
            dims=("row", "col"),
            coords={"row": ["a", "b"], "col": ["b", "a"]},
        ),
        xr.DataArray(
            np.eye(2),
            dims=("row", "col"),
            coords={"row": ["a", "a"], "col": ["a", "a"]},
        ),
        xr.DataArray(
            np.eye(2), dims=("row", "col"), coords={"row": [0, 1], "col": [0, 1]}
        ),
        xr.DataArray(np.eye(2, dtype=complex), dims=("row", "col"), coords=coords),
        xr.DataArray([["1", "0"], ["0", "1"]], dims=("row", "col"), coords=coords),
        xr.DataArray([[1, 0], [0, np.nan]], dims=("row", "col"), coords=coords),
        xr.DataArray([[1, 0], [0, np.inf]], dims=("row", "col"), coords=coords),
        xr.DataArray([[1, 1], [0, 1]], dims=("row", "col"), coords=coords),
    ]


@pytest.mark.parametrize("bad", _invalid_matrices())
@pytest.mark.parametrize(
    "operation", [inv, errors, combine, expand, fix, marginalize, transform]
)
def test_all_matrix_entry_points_validate(
    bad: xr.DataArray, operation: Callable[..., xr.DataArray], J: xr.DataArray
) -> None:
    with pytest.raises(ValueError):
        if operation in (fix, marginalize):
            operation(bad, [])
        elif operation is expand:
            operation(bad, ["a", "b"])
        elif operation is transform:
            operation(bad, J)
        else:
            operation(bad)


@pytest.mark.parametrize(
    "operation", [inv, errors, combine, expand, fix, marginalize, transform]
)
def test_matrix_inputs_must_be_dataarrays(
    operation: Callable[..., xr.DataArray], J: xr.DataArray
) -> None:
    with pytest.raises(TypeError):
        if operation in (fix, marginalize):
            operation(np.eye(2), [])
        elif operation is expand:
            operation(np.eye(2), ["a", "b"])
        elif operation is transform:
            operation(np.eye(2), J)
        else:
            operation(np.eye(2))


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
