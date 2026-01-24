import numpy as np
import pytest
import xarray as xr

from fishy import accessors, indexing, linalg, validation


@pytest.fixture
def square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(data, dims=["row", "col"])


@pytest.fixture
def non_square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return xr.DataArray(data, dims=["row", "col"])


@pytest.fixture
def identity_matrix() -> xr.DataArray:
    return xr.DataArray(np.eye(3), dims=["row", "col"])


@pytest.fixture
def batched_matrix() -> xr.DataArray:
    data = np.stack([np.eye(2), 2 * np.eye(2)])
    return xr.DataArray(data, dims=["batch", "row", "col"])


class TestValidation:
    def test_ensure_dims_raises_on_1d(self):
        da = xr.DataArray([1, 2, 3], dims=["x"])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            validation.ensure_dims(da)

    def test_ensure_dims_passes_on_2d(self, square_matrix: xr.DataArray):
        validation.ensure_dims(square_matrix)

    def test_ensure_no_dim_collision_raises(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="already exists"):
            validation.ensure_dim_not_in_dataarray(square_matrix, "row")

    def test_ensure_no_dim_collision_passes(self, square_matrix: xr.DataArray):
        validation.ensure_dim_not_in_dataarray(square_matrix, "new_dim")

    def test_is_square_matrix_raises_on_non_square(
        self, non_square_matrix: xr.DataArray
    ):
        with pytest.raises(ValueError, match="square"):
            validation.is_square_matrix(non_square_matrix)

    def test_is_square_matrix_passes(self, square_matrix: xr.DataArray):
        assert validation.is_square_matrix(square_matrix)


class TestAccessors:
    def test_get_matrix_dims(self, square_matrix: xr.DataArray):
        row, col = accessors.get_matrix_dims(square_matrix)
        assert row == "row"
        assert col == "col"

    def test_get_batch_dims(self, batched_matrix: xr.DataArray):
        batch_dims = accessors.get_batch_dims(batched_matrix)
        assert batch_dims == ("batch",)

    def test_get_matrix_shape(self, non_square_matrix: xr.DataArray):
        shape = accessors.get_matrix_shape(non_square_matrix)
        assert shape == (2, 3)


class TestIndexing:
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

    def test_at_raises_when_neither_specified(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="Either row or col"):
            indexing.apply_matrix_indexers(square_matrix)


class TestDiagonal:
    def test_diagonal_square(self, identity_matrix: xr.DataArray):
        result = linalg.diagonal(identity_matrix)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result.values, expected)
        assert "diagonal" in result.dims

    def test_diagonal_non_square(self, non_square_matrix: xr.DataArray):
        result = linalg.diagonal(non_square_matrix)
        expected = np.array([1.0, 5.0])
        np.testing.assert_array_equal(result.values, expected)

    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_diagonal_with_offset(self, offset: int):
        data = np.arange(9).reshape(3, 3)
        da = xr.DataArray(data, dims=["row", "col"])
        result = linalg.diagonal(da, offset=offset)
        expected = np.diagonal(data, offset=offset)
        np.testing.assert_array_equal(result.values, expected)

    def test_diagonal_custom_out_dim(self, square_matrix: xr.DataArray):
        result = linalg.diagonal(square_matrix, out_dim="diag")
        assert "diag" in result.dims

    def test_diagonal_raises_on_dim_collision(self, square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="already exists"):
            linalg.diagonal(square_matrix, out_dim="row")

    def test_diagonal_batched(self, batched_matrix: xr.DataArray):
        result = linalg.diagonal(batched_matrix)
        assert result.dims == ("batch", "diagonal")
        np.testing.assert_array_equal(result.sel(batch=0).values, [1.0, 1.0])
        np.testing.assert_array_equal(result.sel(batch=1).values, [2.0, 2.0])


class TestDet:
    def test_det_2x2(self, square_matrix: xr.DataArray):
        result = linalg.det(square_matrix)
        expected = np.linalg.det(square_matrix.values)
        np.testing.assert_allclose(result.values, expected)

    def test_det_identity(self, identity_matrix: xr.DataArray):
        result = linalg.det(identity_matrix)
        np.testing.assert_allclose(result.values, 1.0)

    def test_det_batched(self, batched_matrix: xr.DataArray):
        result = linalg.det(batched_matrix)
        assert result.dims == ("batch",)
        np.testing.assert_allclose(result.values, [1.0, 4.0])


class TestInv:
    def test_inv_2x2(self, square_matrix: xr.DataArray):
        result = linalg.inv(square_matrix)
        expected = np.linalg.inv(square_matrix.values)
        np.testing.assert_allclose(result.values, expected)
        assert result.dims == square_matrix.dims

    def test_inv_identity(self, identity_matrix: xr.DataArray):
        result = linalg.inv(identity_matrix)
        np.testing.assert_allclose(result.values, np.eye(3))

    def test_inv_roundtrip(self, square_matrix: xr.DataArray):
        inv_result = linalg.inv(square_matrix)
        roundtrip = linalg.inv(inv_result)
        np.testing.assert_allclose(roundtrip.values, square_matrix.values)

    def test_inv_raises_on_non_square(self, non_square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="square"):
            linalg.inv(non_square_matrix)

    def test_inv_batched(self, batched_matrix: xr.DataArray):
        result = linalg.inv(batched_matrix)
        assert result.dims == batched_matrix.dims
        np.testing.assert_allclose(result.sel(batch=0).values, np.eye(2))
        np.testing.assert_allclose(result.sel(batch=1).values, 0.5 * np.eye(2))


class TestPinv:
    def test_pinv_square(self, square_matrix: xr.DataArray):
        result = linalg.pinv(square_matrix)
        expected = np.linalg.pinv(square_matrix.values)
        np.testing.assert_allclose(result.values, expected)
        assert result.dims == ("col", "row")

    def test_pinv_non_square(self, non_square_matrix: xr.DataArray):
        result = linalg.pinv(non_square_matrix)
        expected = np.linalg.pinv(non_square_matrix.values)
        np.testing.assert_allclose(result.values, expected)
        assert result.dims == ("col", "row")

    def test_pinv_identity(self, identity_matrix: xr.DataArray):
        result = linalg.pinv(identity_matrix)
        np.testing.assert_allclose(result.values, np.eye(3))

    def test_pinv_batched(self, batched_matrix: xr.DataArray):
        result = linalg.pinv(batched_matrix)
        assert result.dims == ("batch", "col", "row")


class TestSymmetrize:
    def test_symmetrize_symmetric_matrix(self, identity_matrix: xr.DataArray):
        result = linalg.symmetrize(identity_matrix)
        np.testing.assert_allclose(result.values, identity_matrix.values)

    def test_symmetrize_asymmetric_matrix(self, square_matrix: xr.DataArray):
        result = linalg.symmetrize(square_matrix)
        expected = (square_matrix.values + square_matrix.values.T) / 2
        np.testing.assert_allclose(result.values, expected)
        assert result.dims == square_matrix.dims

    def test_symmetrize_result_is_symmetric(self, square_matrix: xr.DataArray):
        result = linalg.symmetrize(square_matrix)
        np.testing.assert_allclose(result.values, result.values.T)

    def test_symmetrize_raises_on_non_square(self, non_square_matrix: xr.DataArray):
        with pytest.raises(ValueError, match="square"):
            linalg.symmetrize(non_square_matrix)

    def test_symmetrize_batched(self, batched_matrix: xr.DataArray):
        result = linalg.symmetrize(batched_matrix)
        assert result.dims == batched_matrix.dims
        for i in range(result.sizes["batch"]):
            batch_result = result.isel(batch=i).values
            np.testing.assert_allclose(batch_result, batch_result.T)

    def test_symmetrize_does_not_mutate_input(self, square_matrix: xr.DataArray):
        original = square_matrix.values.copy()
        _ = linalg.symmetrize(square_matrix)
        np.testing.assert_array_equal(square_matrix.values, original)
