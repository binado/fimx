import numpy as np
import pytest
import xarray as xr

from fishy import accessors


@pytest.fixture
def square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(
        data,
        dims=["row", "col"],
        coords={"row": ["a", "b"], "col": ["x", "y"]},
    )


@pytest.fixture
def non_square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return xr.DataArray(data, dims=["row", "col"])


@pytest.fixture
def batched_matrix() -> xr.DataArray:
    data = np.stack([np.eye(2), 2 * np.eye(2)])
    return xr.DataArray(data, dims=["batch", "row", "col"])


class TestGetMatrixDims:
    def test_returns_last_two_dims(self, square_matrix: xr.DataArray):
        row, col = accessors.get_matrix_dims(square_matrix)
        assert row == "row"
        assert col == "col"

    def test_batched_matrix(self, batched_matrix: xr.DataArray):
        row, col = accessors.get_matrix_dims(batched_matrix)
        assert row == "row"
        assert col == "col"


class TestGetMatrixShape:
    def test_square_matrix(self, square_matrix: xr.DataArray):
        shape = accessors.get_matrix_shape(square_matrix)
        assert shape == (2, 2)

    def test_non_square_matrix(self, non_square_matrix: xr.DataArray):
        shape = accessors.get_matrix_shape(non_square_matrix)
        assert shape == (2, 3)


class TestGetBatchDims:
    def test_no_batch_dims(self, square_matrix: xr.DataArray):
        batch_dims = accessors.get_batch_dims(square_matrix)
        assert batch_dims == ()

    def test_single_batch_dim(self, batched_matrix: xr.DataArray):
        batch_dims = accessors.get_batch_dims(batched_matrix)
        assert batch_dims == ("batch",)

    def test_multiple_batch_dims(self):
        data = np.ones((2, 3, 4, 4))
        da = xr.DataArray(data, dims=["batch1", "batch2", "row", "col"])
        batch_dims = accessors.get_batch_dims(da)
        assert batch_dims == ("batch1", "batch2")


class TestGetMatrixCoords:
    def test_returns_coords(self, square_matrix: xr.DataArray):
        row_coords, col_coords = accessors.get_matrix_coords(square_matrix)
        xr.testing.assert_equal(row_coords, square_matrix.coords["row"])
        xr.testing.assert_equal(col_coords, square_matrix.coords["col"])


class TestCanonicalizeDims:
    def test_already_canonical(self, square_matrix: xr.DataArray):
        result = accessors.canonicalize_dims(square_matrix, "row", "col")
        assert result.dims == ("row", "col")

    def test_transpose_to_canonical(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        da = xr.DataArray(data, dims=["col", "row"])
        result = accessors.canonicalize_dims(da, "row", "col")
        assert result.dims == ("row", "col")

    def test_batched_canonical(self, batched_matrix: xr.DataArray):
        result = accessors.canonicalize_dims(batched_matrix, "row", "col")
        assert result.dims == ("batch", "row", "col")

    def test_batched_transpose(self):
        data = np.ones((2, 3, 4))
        da = xr.DataArray(data, dims=["col", "batch", "row"])
        result = accessors.canonicalize_dims(da, "row", "col")
        assert result.dims == ("batch", "row", "col")

    def test_raises_on_missing_dim(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="not found"):
            accessors.canonicalize_dims(square_matrix, "row", "missing")
