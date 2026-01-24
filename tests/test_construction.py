import numpy as np
import pytest
import xarray as xr

from fishy import construction


@pytest.fixture
def square_matrix() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(
        data, dims=["row", "col"], coords={"row": ["a", "b"], "col": ["a", "b"]}
    )


@pytest.fixture
def square_matrix_with_row_coords_only() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(data, dims=["row", "col"], coords={"row": ["a", "b"]})


@pytest.fixture
def square_matrix_with_col_coords_only() -> xr.DataArray:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.DataArray(data, dims=["row", "col"], coords={"col": ["a", "b"]})


class TestNormalizeDataarray:
    def test_assigns_coords_to_both_dims(self, square_matrix: xr.DataArray):
        result = construction.normalize_dataarray(square_matrix)
        assert list(result.coords["row"].values) == ["a", "b"]
        assert list(result.coords["col"].values) == ["a", "b"]

    def test_assigns_col_coords_from_row(
        self, square_matrix_with_row_coords_only: xr.DataArray
    ):
        result = construction.normalize_dataarray(square_matrix_with_row_coords_only)
        assert list(result.coords["row"].values) == ["a", "b"]
        assert list(result.coords["col"].values) == ["a", "b"]

    def test_uses_row_default_index_when_only_col_has_coords(
        self, square_matrix_with_col_coords_only: xr.DataArray
    ):
        result = construction.normalize_dataarray(square_matrix_with_col_coords_only)
        assert list(result.coords["row"].values) == [0, 1]
        assert list(result.coords["col"].values) == [0, 1]

    def test_raises_on_1d(self):
        da = xr.DataArray([1, 2, 3], dims=["x"])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            construction.normalize_dataarray(da)

    def test_raises_on_non_square(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        da = xr.DataArray(data, dims=["row", "col"])
        with pytest.raises(ValueError, match="square"):
            construction.normalize_dataarray(da)


class TestNormalizeDataset:
    def test_normalizes_specified_key(
        self, square_matrix_with_row_coords_only: xr.DataArray
    ):
        ds = xr.Dataset({"matrix": square_matrix_with_row_coords_only})
        result = construction.normalize_dataset(ds, "matrix")
        assert list(result["matrix"].coords["col"].values) == ["a", "b"]

    def test_preserves_other_variables(
        self, square_matrix_with_row_coords_only: xr.DataArray
    ):
        ds = xr.Dataset(
            {
                "matrix": square_matrix_with_row_coords_only,
                "other": xr.DataArray([1, 2, 3]),
            }
        )
        result = construction.normalize_dataset(ds, "matrix")
        xr.testing.assert_equal(result["other"], ds["other"])


class TestBuildDataarrayFromArray:
    def test_basic_2d(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        params = ["a", "b"]
        result = construction.build_dataarray_from_array(
            array, parameters=params, matrix_dims=("row", "col")
        )
        assert result.dims == ("row", "col")
        assert list(result.coords["row"].values) == ["a", "b"]
        assert list(result.coords["col"].values) == ["a", "b"]
        np.testing.assert_array_equal(result.values, array)

    def test_with_batch_dims(self):
        array = np.stack([np.eye(2), 2 * np.eye(2)])
        params = ["a", "b"]
        result = construction.build_dataarray_from_array(
            array,
            parameters=params,
            matrix_dims=("row", "col"),
            batch_dims=["batch"],
        )
        assert result.dims == ("batch", "row", "col")

    def test_with_batch_coords(self):
        array = np.stack([np.eye(2), 2 * np.eye(2)])
        params = ["a", "b"]
        result = construction.build_dataarray_from_array(
            array,
            parameters=params,
            matrix_dims=("row", "col"),
            batch_dims=["batch"],
            batch_coords={"batch": [0.1, 0.2]},
        )
        assert list(result.coords["batch"].values) == [0.1, 0.2]

    def test_raises_on_1d_array(self):
        array = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            construction.build_dataarray_from_array(
                array, parameters=["a", "b", "c"], matrix_dims=("row", "col")
            )

    def test_raises_on_non_square_array(self):
        array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="square"):
            construction.build_dataarray_from_array(
                array, parameters=["a", "b"], matrix_dims=("row", "col")
            )

    def test_raises_on_mismatched_parameters_length(self):
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="parameters length"):
            construction.build_dataarray_from_array(
                array, parameters=["a", "b", "c"], matrix_dims=("row", "col")
            )

    def test_raises_on_missing_batch_dims(self):
        array = np.stack([np.eye(2), 2 * np.eye(2)])
        with pytest.raises(ValueError, match="batch_dims must be provided"):
            construction.build_dataarray_from_array(
                array, parameters=["a", "b"], matrix_dims=("row", "col")
            )

    def test_raises_on_mismatched_batch_dims_length(self):
        array = np.stack([np.eye(2), 2 * np.eye(2)])
        with pytest.raises(ValueError, match="batch_dims length"):
            construction.build_dataarray_from_array(
                array,
                parameters=["a", "b"],
                matrix_dims=("row", "col"),
                batch_dims=["batch1", "batch2"],
            )


class TestPartitionMatrices:
    def test_2d_partition(self):
        values = np.arange(9).reshape(3, 3)
        idx_keep = [0, 2]
        idx_drop = [1]
        f_kk, f_kd, f_dd, f_dk = construction.partition_matrices(
            values, idx_keep, idx_drop
        )
        expected_kk = np.array([[0, 2], [6, 8]])
        expected_kd = np.array([[1], [7]])
        expected_dd = np.array([[4]])
        expected_dk = np.array([[3, 5]])
        np.testing.assert_array_equal(f_kk, expected_kk)
        np.testing.assert_array_equal(f_kd, expected_kd)
        np.testing.assert_array_equal(f_dd, expected_dd)
        np.testing.assert_array_equal(f_dk, expected_dk)

    def test_3d_partition(self):
        values = np.stack([np.arange(9).reshape(3, 3), np.arange(9, 18).reshape(3, 3)])
        idx_keep = [0, 2]
        idx_drop = [1]
        f_kk, f_kd, f_dd, f_dk = construction.partition_matrices(
            values, idx_keep, idx_drop
        )
        assert f_kk.shape == (2, 2, 2)
        assert f_kd.shape == (2, 2, 1)
        assert f_dd.shape == (2, 1, 1)
        assert f_dk.shape == (2, 1, 2)
        np.testing.assert_array_equal(f_kk[0], np.array([[0, 2], [6, 8]]))
        np.testing.assert_array_equal(f_kk[1], np.array([[9, 11], [15, 17]]))

    def test_empty_drop(self):
        values = np.arange(4).reshape(2, 2)
        idx_keep = [0, 1]
        idx_drop: list[int] = []
        f_kk, f_kd, f_dd, f_dk = construction.partition_matrices(
            values, idx_keep, idx_drop
        )
        np.testing.assert_array_equal(f_kk, values)
        assert f_kd.shape == (2, 0)
        assert f_dd.shape == (0, 0)
        assert f_dk.shape == (0, 2)
