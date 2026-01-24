import numpy as np
import pytest
import xarray as xr

from fisharr import validation


@pytest.fixture
def square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(data, dims=["row", "col"])


@pytest.fixture
def non_square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return xr.DataArray(data, dims=["row", "col"])


class TestEnsureDims:
    def test_raises_on_1d(self):
        da = xr.DataArray([1, 2, 3], dims=["x"])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            validation.ensure_dims(da)

    def test_passes_on_2d(self, square_matrix: xr.DataArray):
        validation.ensure_dims(square_matrix)

    def test_raises_on_insufficient_dims(self):
        da = xr.DataArray([[1, 2], [3, 4]], dims=["x", "y"])
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            validation.ensure_dims(da, expected=3)

    def test_passes_with_custom_expected(self):
        da = xr.DataArray([[[1, 2], [3, 4]]], dims=["batch", "x", "y"])
        validation.ensure_dims(da, expected=3)

    def test_works_with_numpy_array(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            validation.ensure_dims(arr)


class TestEnsureDimNotInDataarray:
    def test_raises_when_dim_exists(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="already exists"):
            validation.ensure_dim_not_in_dataarray(square_matrix, "row")

    def test_passes_when_dim_not_exists(self, square_matrix: xr.DataArray):
        validation.ensure_dim_not_in_dataarray(square_matrix, "new_dim")


class TestEnsureDimsInDataarray:
    def test_raises_when_dim_missing(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="not found"):
            validation.ensure_dims_in_dataarray(square_matrix, "missing_dim")

    def test_raises_when_multiple_dims_missing(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="dim1.*dim2|dim2.*dim1"):
            validation.ensure_dims_in_dataarray(square_matrix, "dim1", "dim2")

    def test_passes_when_all_dims_exist(self, square_matrix: xr.DataArray):
        validation.ensure_dims_in_dataarray(square_matrix, "row", "col")

    def test_passes_with_single_dim(self, square_matrix: xr.DataArray):
        validation.ensure_dims_in_dataarray(square_matrix, "row")


class TestIsSquareMatrix:
    def test_raises_on_non_square(self, non_square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="square"):
            validation.is_square_matrix(non_square_matrix)

    def test_passes_on_square(self, square_matrix: xr.DataArray):
        assert validation.is_square_matrix(square_matrix)

    def test_returns_false_without_exception(self, non_square_matrix: xr.DataArray):
        assert not validation.is_square_matrix(non_square_matrix, raise_exception=False)

    def test_works_with_numpy_array(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert not validation.is_square_matrix(arr, raise_exception=False)

    def test_works_with_batched_matrix(self):
        data = np.stack([np.eye(2), 2 * np.eye(2)])
        da = xr.DataArray(data, dims=["batch", "row", "col"])
        assert validation.is_square_matrix(da)
