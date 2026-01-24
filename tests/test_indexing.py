import numpy as np
import pytest
import xarray as xr

from fisharr import indexing


@pytest.fixture
def square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(
        data, dims=["row", "col"], coords={"row": ["a", "b"], "col": ["x", "y"]}
    )


@pytest.fixture
def dataset_with_matrix(square_matrix: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({"matrix": square_matrix, "other": xr.DataArray([1, 2])})


class TestGetIndexers:
    def test_row_only(self, square_matrix: xr.DataArray):
        indexers = indexing.get_indexers(square_matrix, row=0)
        assert indexers == {"row": 0}

    def test_col_only(self, square_matrix: xr.DataArray):
        indexers = indexing.get_indexers(square_matrix, col=1)
        assert indexers == {"col": 1}

    def test_row_and_col(self, square_matrix: xr.DataArray):
        indexers = indexing.get_indexers(square_matrix, row=0, col=1)
        assert indexers == {"row": 0, "col": 1}

    def test_raises_when_neither_specified(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="Either row or col"):
            indexing.get_indexers(square_matrix)


class TestApplyMatrixIndexers:
    def test_at_row(self, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers(square_matrix, row=0)
        expected = square_matrix.isel(row=0)
        xr.testing.assert_equal(result, expected)

    def test_at_col(self, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers(square_matrix, col=1)
        expected = square_matrix.isel(col=1)
        xr.testing.assert_equal(result, expected)

    def test_at_row_and_col(self, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers(square_matrix, row=0, col=1)
        expected = square_matrix.isel(row=0, col=1)
        xr.testing.assert_equal(result, expected)

    def test_raises_when_neither_specified(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="Either row or col"):
            indexing.apply_matrix_indexers(square_matrix)

    def test_with_slice(self, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers(square_matrix, row=slice(0, 1))
        expected = square_matrix.isel(row=slice(0, 1))
        xr.testing.assert_equal(result, expected)


class TestApplyMatrixIndexersToDataset:
    def test_at_row(self, dataset_with_matrix: xr.Dataset, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers_to_dataset(
            dataset_with_matrix, square_matrix, row=0
        )
        expected = dataset_with_matrix.isel(row=0)
        xr.testing.assert_equal(result, expected)

    def test_at_col(self, dataset_with_matrix: xr.Dataset, square_matrix: xr.DataArray):
        result = indexing.apply_matrix_indexers_to_dataset(
            dataset_with_matrix, square_matrix, col=1
        )
        expected = dataset_with_matrix.isel(col=1)
        xr.testing.assert_equal(result, expected)

    def test_at_row_and_col(
        self, dataset_with_matrix: xr.Dataset, square_matrix: xr.DataArray
    ):
        result = indexing.apply_matrix_indexers_to_dataset(
            dataset_with_matrix, square_matrix, row=0, col=1
        )
        expected = dataset_with_matrix.isel(row=0, col=1)
        xr.testing.assert_equal(result, expected)


class TestSubmatrix:
    def test_submatrix_single_index(self, square_matrix: xr.DataArray):
        result = indexing.submatrix(square_matrix, coords=0)
        expected = square_matrix.isel(row=0, col=0)
        xr.testing.assert_equal(result, expected)

    def test_submatrix_slice(self, square_matrix: xr.DataArray):
        result = indexing.submatrix(square_matrix, coords=slice(0, 1))
        expected = square_matrix.isel(row=slice(0, 1), col=slice(0, 1))
        xr.testing.assert_equal(result, expected)

    def test_submatrix_list(self, square_matrix: xr.DataArray):
        result = indexing.submatrix(square_matrix, coords=[0, 1])
        expected = square_matrix.isel(row=[0, 1], col=[0, 1])
        xr.testing.assert_equal(result, expected)

    def test_submatrix_raises_on_non_square(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        da = xr.DataArray(data, dims=["row", "col"])
        with pytest.raises(ValueError, match="square"):
            indexing.submatrix(da, coords=0)

    def test_submatrix_raises_on_1d(self):
        da = xr.DataArray([1, 2, 3], dims=["x"])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            indexing.submatrix(da, coords=0)
