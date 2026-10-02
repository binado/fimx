from collections.abc import Callable, Sequence

import numpy as np
import pytest
import xarray as xr

from fimx import fix, inv, marginalize, matrix, transform


def test_fix_vs_marginalization() -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    np.testing.assert_allclose(fix(F, "b").values, [[4]])
    np.testing.assert_allclose(marginalize(F, "b").values, [[3.5]])


@pytest.mark.parametrize("removed", ["b", ["c", "b"]])
def test_retained_covariance(F: xr.DataArray, removed: str | Sequence[str]) -> None:
    reduced = marginalize(F, removed)
    keep = reduced.row.values.tolist()
    expected = inv(F).sel(row=keep, col=keep)
    xr.testing.assert_allclose(inv(reduced), expected)
    np.testing.assert_array_equal(reduced.values, reduced.values.T)


@pytest.mark.parametrize("operation", [fix, marginalize])
@pytest.mark.parametrize("removed", ["b", ["c", "b"], [], ()])
def test_selections(
    F: xr.DataArray,
    operation: Callable[[xr.DataArray, str | Sequence[str]], xr.DataArray],
    removed: str | Sequence[str],
) -> None:
    result = operation(F, removed)
    keep = [label for label in F.row.values.tolist() if label not in removed]
    assert result.row.values.tolist() == result.col.values.tolist() == keep
    assert not np.shares_memory(result.values, F.values)
    if not removed:
        xr.testing.assert_equal(result, F)


@pytest.mark.parametrize("operation", [fix, marginalize])
@pytest.mark.parametrize(
    ("removed", "exception"),
    [
        ("missing", KeyError),
        (["a", "missing"], KeyError),
        (["a", "a"], ValueError),
        (["c", "b", "a"], ValueError),
        ([1], ValueError),
    ],
)
def test_invalid_selections(
    F: xr.DataArray,
    operation: Callable[[xr.DataArray, str | Sequence[str]], xr.DataArray],
    removed: str | Sequence[str],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        operation(F, removed)


def test_marginalization_needs_only_removed_block_positive_definite() -> None:
    singular = matrix([[1, 1], [1, 1]], ["a", "b"])
    np.testing.assert_allclose(marginalize(singular, "b").values, [[0]])
    np.testing.assert_allclose(fix(singular, "b").values, [[1]])
    indefinite = matrix([[-1, 0], [0, 2]], ["a", "b"])
    np.testing.assert_allclose(marginalize(indefinite, "b").values, [[-1]])


@pytest.mark.parametrize("values", [[[1, 0], [0, 0]], [[1, 0], [0, -1]]])
def test_marginalization_fails_for_removed_block(values: list[list[float]]) -> None:
    with pytest.raises(np.linalg.LinAlgError):
        marginalize(matrix(values, ["a", "b"]), "b")


def test_reordered_rectangular_transform(F: xr.DataArray, J: xr.DataArray) -> None:
    result = transform(F, J)
    aligned = J.sel(old=F.row.values).values
    np.testing.assert_allclose(result.values, aligned.T @ F.values @ aligned)
    assert result.row.values.tolist() == result.col.values.tolist() == ["y", "x"]
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.values, result.values.T)


def test_transform_can_expand_parameter_space(F: xr.DataArray) -> None:
    J = xr.DataArray(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        dims=("old", "new"),
        coords={
            "old": ["a", "b", "c"],
            "new": ["w", "x", "y", "z"],
        },
    )
    np.testing.assert_allclose(transform(F, J).values, J.values.T @ F.values @ J.values)


@pytest.mark.parametrize(
    "kind",
    [
        "dimensions",
        "transposed",
        "missing_old",
        "missing_new",
        "incomplete",
        "extra",
        "duplicate_old",
        "duplicate_new",
        "numeric_old",
        "numeric_new",
        "empty_new",
        "empty_old",
        "complex",
        "nonnumeric",
        "nan",
        "inf",
    ],
)
def test_invalid_jacobians(F: xr.DataArray, J: xr.DataArray, kind: str) -> None:
    if kind == "dimensions":
        J = J.rename(old="old_parameter")
    elif kind == "transposed":
        J = J.transpose()
    elif kind.startswith("missing_"):
        J = J.drop_vars(kind.removeprefix("missing_"))
    elif kind == "incomplete":
        J = J.isel(old=slice(0, 2))
    elif kind == "extra":
        J = J.reindex(old=["a", "b", "c", "d"], fill_value=0)
    elif kind == "duplicate_old":
        J = J.assign_coords(old=["a", "a", "b"])
    elif kind == "duplicate_new":
        J = J.assign_coords(new=["x", "x"])
    elif kind == "numeric_old":
        J = J.assign_coords(old=[0, 1, 2])
    elif kind == "numeric_new":
        J = J.assign_coords(new=[0, 1])
    elif kind.startswith("empty_"):
        J = J.isel({kind.removeprefix("empty_"): slice(0, 0)})
    elif kind == "complex":
        J = J.astype(complex)
    elif kind == "nonnumeric":
        J = J.astype(str)
    else:
        J = J.astype(float)
        J.values[0, 0] = np.nan if kind == "nan" else np.inf
    with pytest.raises(ValueError):
        transform(F, J)


def test_jacobian_must_be_dataarray(F: xr.DataArray) -> None:
    with pytest.raises(TypeError):
        transform(F, np.eye(3))  # ty: ignore[invalid-argument-type]
