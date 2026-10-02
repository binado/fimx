"""Public Dataset construction with parameter arrays and metadata."""

import numpy as np
import pytest
import xarray as xr

from fisharr import dataset, matrix


def test_dataset_alignment_and_independence(F: xr.DataArray) -> None:
    reference = xr.DataArray(
        [30, 10, 20], dims="parameter", coords={"parameter": ["c", "a", "b"]}
    )
    original = F.copy(deep=True)
    original_reference = reference.copy(deep=True)
    result = dataset(F, {"fiducials": reference})
    assert set(result.data_vars) == {"fisher", "fiducials"}
    assert result.fiducials.dims == ("row",)
    assert result.fiducials.dtype == reference.dtype
    np.testing.assert_array_equal(result.fiducials.values, [10, 20, 30])
    result.fisher.values[0, 0] = 100
    result.fiducials.values[0] = 100
    result.row.values[0] = "changed"
    xr.testing.assert_identical(F, original)
    xr.testing.assert_identical(reference, original_reference)


def test_metadata_dimensions_and_dtypes(F: xr.DataArray) -> None:
    covariance = np.eye(3)
    result = dataset(
        F,
        {
            "units": ["km", "s", "m"],
            "covariance": covariance,
            42: [1, 2, 3],
            ("other", "metadata"): [np.nan, np.inf, 0],
        },
    )
    assert result.units.dims == ("row",)
    assert result.covariance.dims == ("row", "col")
    assert result.units.dtype.kind == "U"
    np.testing.assert_array_equal(result[42].values, [1, 2, 3])
    assert np.isnan(result[("other", "metadata")].values[0])
    result.covariance.values[0, 0] = 100
    np.testing.assert_array_equal(covariance, np.eye(3))


def test_matrix_and_unindexed_array_order() -> None:
    F = matrix([[4, 1], [1, 2]], ["b", "a"])
    result = dataset(
        F,
        {
            "plain": [20, 10],
            "unindexed": xr.DataArray([20, 10], dims="parameter"),
            "complex": [1j, 2j],
        },
    )
    np.testing.assert_array_equal(result.plain.values, [20, 10])
    np.testing.assert_array_equal(result.unindexed.values, [20, 10])
    np.testing.assert_array_equal(result.complex.values, [1j, 2j])
    assert result.row.values.tolist() == ["b", "a"]


def test_matrix_metadata_alignment() -> None:
    F = matrix([[4, 1], [1, 2]], ["b", "a"])
    metadata = xr.DataArray(
        [[1, 2], [3, 4]], dims=("x", "y"), coords={"x": ["a", "b"], "y": ["b", "a"]}
    )
    result = dataset(F, {"metadata": metadata})
    np.testing.assert_array_equal(result.metadata.values, [[3, 4], [1, 2]])
    assert result.metadata.dims == ("row", "col")


@pytest.mark.parametrize("values", [0, np.zeros((1, 1, 1)), [1, 2], np.zeros((3, 2))])
def test_incompatible_array_dimensions_and_sizes(
    F: xr.DataArray, values: object
) -> None:
    with pytest.raises(ValueError):
        dataset(F, {"metadata": values})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "names", [["a", "b", "d"], ["a", "a", "c"], ["a", "b"], ["a", "b", "c", "d"]]
)
def test_incompatible_indexed_coordinates(F: xr.DataArray, names: list[str]) -> None:
    reference = xr.DataArray(np.zeros(len(names)), dims="row", coords={"row": names})
    with pytest.raises(ValueError):
        dataset(F, {"metadata": reference})


@pytest.mark.parametrize("name", ["fisher", "row", "col"])
def test_reserved_names(F: xr.DataArray, name: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        dataset(F, {name: [1, 2, 3]})


def test_dataset_requires_matrix_and_mapping(F: xr.DataArray) -> None:
    with pytest.raises(TypeError):
        dataset(F, [1, 2, 3])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        dataset([[1]], {})  # ty: ignore[invalid-argument-type]


def test_dataset_accepts_singular_information_and_no_metadata() -> None:
    result = dataset(matrix([[0]], ["a"]), {})
    assert result.fisher.item() == 0
    assert set(result.data_vars) == {"fisher"}
