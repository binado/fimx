from collections.abc import Callable, Sequence

import numpy as np
import pytest
import xarray as xr

from fimx import (
    combine,
    correlation,
    errors,
    expand,
    fix,
    fom,
    inv,
    marginalize,
    matrix,
    symmetrize,
    transform,
)


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


def test_direct_dataarray_and_canonical_output(direct: xr.DataArray) -> None:
    original = direct.copy(deep=True)
    result = combine(direct)
    assert result.dtype == np.float64
    assert set(result.coords) == {"row", "col"}
    assert result.attrs == {}
    assert result.name is None
    np.testing.assert_array_equal(result.values, direct.values)
    result.values[0, 0] = 100
    xr.testing.assert_identical(direct, original)


def test_expand_fills_new_entries_with_zeros() -> None:
    F = matrix([[4, 1], [1, 2]], ["b", "a"])
    result = expand(F, ["a", "c", "b"])
    expected = matrix([[2, 0, 1], [0, 0, 0], [1, 0, 4]], ["a", "c", "b"])
    xr.testing.assert_equal(result, expected)


def test_expand_to_same_parameters_is_a_reorder(F: xr.DataArray) -> None:
    result = expand(F, ["c", "a", "b"])
    xr.testing.assert_equal(result, F.sel(row=["c", "a", "b"], col=["c", "a", "b"]))


def test_expanded_matrices_add_natively() -> None:
    first = matrix([[4, 1], [1, 2]], ["a", "b"])
    second = matrix([[3, 0.5], [0.5, 6]], ["b", "c"])
    union = ["a", "b", "c"]
    summed = expand(first, union) + expand(second, union)
    xr.testing.assert_allclose(summed, combine(first, second))


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (["a", "b"], ValueError),  # drops "c"
        (["a", "a", "b", "c"], ValueError),
        ("abc", ValueError),
        ([], ValueError),
    ],
)
def test_expand_invalid_parameters(
    F: xr.DataArray, parameters: Sequence[str], error: type[Exception]
) -> None:
    with pytest.raises(error):
        expand(F, parameters)


def test_expand_does_not_mutate_input(F: xr.DataArray) -> None:
    original = F.copy(deep=True)
    result = expand(F, ["a", "b", "c"])
    result.values[0, 0] = -1
    xr.testing.assert_identical(F, original)


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


def test_symmetrize_averages_with_transpose() -> None:
    values = np.array([[2.0, 1.0], [1.0 + 5e-9, 2.0]])
    bad = xr.DataArray(
        values,
        dims=("row", "col"),
        coords={"row": ["a", "b"], "col": ["a", "b"]},
    )
    with pytest.raises(ValueError, match="symmetric"):
        matrix(values, ["a", "b"])
    result = symmetrize(bad)
    np.testing.assert_allclose(result.values, (values + values.T) / 2)
    np.testing.assert_array_equal(result.values, result.values.T)
    assert result.dtype == np.float64
    assert result.row.values.tolist() == result.col.values.tolist() == ["a", "b"]
    xr.testing.assert_allclose(
        errors(result), errors(matrix([[2, 1], [1, 2]], ["a", "b"]))
    )


def test_symmetrize_returns_fresh_canonical_matrix(F: xr.DataArray) -> None:
    original = F.copy(deep=True)
    result = symmetrize(F)
    xr.testing.assert_equal(result, F)
    assert set(result.coords) == {"row", "col"}
    assert result.attrs == {}
    assert result.name is None
    assert not np.shares_memory(result.values, F.values)
    result.values[0, 0] = -1
    xr.testing.assert_identical(F, original)


def test_symmetrize_requires_dataarray() -> None:
    with pytest.raises(TypeError):
        symmetrize(np.eye(2))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "bad",
    [
        xr.DataArray(np.eye(2), dims=("x", "y")),
        xr.DataArray(np.ones((2, 3)), dims=("row", "col")),
        xr.DataArray(np.empty((0, 0)), dims=("row", "col")),
        xr.DataArray(np.eye(2), dims=("row", "col")),
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
        xr.DataArray(
            [[1, 0], [0, np.nan]],
            dims=("row", "col"),
            coords={"row": ["a", "b"], "col": ["a", "b"]},
        ),
    ],
)
def test_invalid_symmetrize(bad: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        symmetrize(bad)


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


def test_correlation_of_diagonal_matrix_is_identity() -> None:
    F = matrix([[4, 0], [0, 1]], ["a", "b"])
    original = F.copy(deep=True)
    result = correlation(F)
    expected = matrix(np.eye(2), ["a", "b"])
    xr.testing.assert_identical(result, expected)
    xr.testing.assert_identical(F, original)
    result.values[0, 0] = 0
    xr.testing.assert_identical(F, original)


@pytest.mark.parametrize("method", ["cholesky", "inv", "pinv"])
def test_correlation_matches_inverse(method: str) -> None:
    F = matrix([[4, 1], [1, 2]], ["b", "a"])
    covariance = inv(F)
    sigma = np.sqrt(np.diag(covariance.values))
    expected = covariance.values / np.outer(sigma, sigma)
    original = F.copy(deep=True)
    result = correlation(F, method=method)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(result.values, expected)
    assert result.dims == ("row", "col")
    assert result.row.values.tolist() == result.col.values.tolist() == ["b", "a"]
    np.testing.assert_array_equal(np.diag(result.values), [1, 1])
    np.testing.assert_array_equal(result.values, result.values.T)
    assert not np.shares_memory(result.values, F.values)
    result.values[0, 0] = 99
    xr.testing.assert_identical(F, original)


def test_correlation_rejects_zero_variance() -> None:
    F = matrix([[1, 0], [0, 0]], ["a", "b"])
    with pytest.raises(np.linalg.LinAlgError):
        correlation(F, method="pinv")


def test_fom_of_diagonal_matrix() -> None:
    F = matrix([[4, 0], [0, 1]], ["a", "b"])
    assert fom(F) == pytest.approx(2)


def test_fom_matches_determinant_and_schur_complement() -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    assert fom(F) == pytest.approx(np.sqrt(7))
    assert fom(F, "a") == pytest.approx(np.sqrt(3.5))
    assert fom(F, ["b", "a"]) == pytest.approx(fom(F))


def test_invalid_fom_selection() -> None:
    F = matrix([[4, 1], [1, 2]], ["a", "b"])
    with pytest.raises(KeyError):
        fom(F, "z")
    with pytest.raises(ValueError):
        fom(F, ["a", "a"])
    with pytest.raises(ValueError):
        fom(F, [])


@pytest.mark.parametrize("diagonal", [[0, 1], [-1, -2], [-1, 2], [-1, -2, 3]])
@pytest.mark.parametrize("selection", ["full", "all", "subset", "removed"])
def test_fom_requires_positive_definite_matrix(
    diagonal: list[float], selection: str
) -> None:
    labels = [f"p{i}" for i in range(len(diagonal))]
    parameters = None
    if selection in {"subset", "removed"}:
        diagonal = [*diagonal, 4]
        labels = [*labels, "valid"]
        parameters = labels[:-1] if selection == "subset" else ["valid"]
    elif selection == "all":
        parameters = labels[::-1]
    F = matrix(np.diag(diagonal), labels)
    original = F.copy(deep=True)
    with pytest.raises(np.linalg.LinAlgError):
        fom(F, parameters)
    xr.testing.assert_identical(F, original)


def test_fom_large_representable_value_without_determinant_overflow() -> None:
    F = matrix(np.diag([1e200, 1e200]), ["b", "a"])
    original = F.copy(deep=True)
    assert fom(F) == pytest.approx(1e200)
    xr.testing.assert_identical(F, original)


@pytest.mark.parametrize("parameters", [None, ["a", "b"], ["b", "a"], "a"])
def test_fom_preserves_input_and_parameter_order(
    F: xr.DataArray, parameters: str | list[str] | None
) -> None:
    original = F.copy(deep=True)
    reordered = F.sel(row=["c", "b", "a"], col=["c", "b", "a"])
    assert fom(F, parameters) == pytest.approx(fom(reordered, parameters))
    xr.testing.assert_identical(F, original)


@pytest.mark.parametrize("method", ["inv", "pinv"])
def test_correlation_rejects_indefinite_input_with_positive_variances(
    method: str,
) -> None:
    F = matrix([[-1, 2], [2, -1]], ["b", "a"])
    with pytest.raises(np.linalg.LinAlgError, match="positive semidefinite"):
        correlation(F, method=method)  # ty: ignore[invalid-argument-type]


def test_correlation_pinv_accepts_singular_positive_semidefinite_input() -> None:
    F = matrix([[1, 1], [1, 1]], ["b", "a"])
    original = F.copy(deep=True)
    result = correlation(F, method="pinv")
    xr.testing.assert_identical(result, matrix(np.ones((2, 2)), ["b", "a"]))
    assert not np.shares_memory(result.values, F.values)
    result.values[0, 0] = 99
    xr.testing.assert_identical(F, original)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_correlation_rejects_indefinite_covariance_within_input_roundoff(
    scale: float,
) -> None:
    # The Fisher spectrum passes the roundoff allowance, but its inverse does not.
    F = matrix(np.diag([-np.finfo(np.float64).eps, 1.0]) * scale, ["b", "a"])
    with pytest.raises(np.linalg.LinAlgError, match="positive semidefinite covariance"):
        correlation(F, method="inv")


def test_correlation_validates_method_before_definiteness() -> None:
    F = matrix([[-1, 2], [2, -1]], ["b", "a"])
    with pytest.raises(ValueError, match="Unknown inversion method"):
        correlation(F, method="lu")  # ty: ignore[invalid-argument-type]
