import numpy as np
import pytest
import xarray as xr

from fisharr.fisher import (
    COVARIANCE_VAR,
    DEFAULT_COL_DIM,
    DEFAULT_ROW_DIM,
    FISHER_VAR,
    FisherMatrix,
)


@pytest.fixture
def parameters() -> list[str]:
    return ["a", "b", "c"]


@pytest.fixture
def fisher_values() -> np.ndarray:
    return np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])


@pytest.fixture
def fisher_matrix(parameters: list[str], fisher_values: np.ndarray) -> FisherMatrix:
    return FisherMatrix(fisher_values, parameters)


@pytest.fixture
def batched_fisher_values(fisher_values: np.ndarray) -> np.ndarray:
    return np.stack([fisher_values, 2 * fisher_values])


@pytest.fixture
def batched_fisher_matrix(
    parameters: list[str], batched_fisher_values: np.ndarray
) -> FisherMatrix:
    return FisherMatrix(batched_fisher_values, parameters, batch_dims=["batch"])


class TestFisherMatrixInit:
    def test_init_from_numpy_array(
        self, parameters: list[str], fisher_values: np.ndarray
    ) -> None:
        fm = FisherMatrix(fisher_values, parameters)

        assert fm.parameters == parameters
        assert fm.matrix_dims == (DEFAULT_ROW_DIM, DEFAULT_COL_DIM)
        np.testing.assert_array_equal(fm.data.values, fisher_values)

    def test_init_from_numpy_array_requires_parameters(
        self, fisher_values: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="parameters are required"):
            FisherMatrix(fisher_values)

    def test_init_from_dataarray(
        self, parameters: list[str], fisher_values: np.ndarray
    ) -> None:
        da = xr.DataArray(
            fisher_values,
            dims=["row", "col"],
            coords={"row": parameters, "col": parameters},
        )
        fm = FisherMatrix(da)

        assert fm.parameters == parameters
        np.testing.assert_array_equal(fm.data.values, fisher_values)

    def test_init_from_dataarray_ignores_parameters_argument(
        self, fisher_values: np.ndarray
    ) -> None:
        da = xr.DataArray(
            fisher_values,
            dims=["row", "col"],
            coords={"row": ["x", "y", "z"], "col": ["x", "y", "z"]},
        )
        new_params = ["a", "b", "c"]
        fm = FisherMatrix(da, parameters=new_params)

        assert fm.parameters == ["x", "y", "z"]

    def test_init_from_dataarray_extracts_params_from_coord_name(
        self, fisher_values: np.ndarray
    ) -> None:
        params = ["x", "y", "z"]
        da = xr.DataArray(
            fisher_values,
            dims=["row", "col"],
            coords={"row": params, "col": params, "param": ("row", params)},
        )
        fm = FisherMatrix(da, parameters="param")

        assert fm.parameters == params

    def test_init_from_dataset(
        self, parameters: list[str], fisher_values: np.ndarray
    ) -> None:
        da = xr.DataArray(
            fisher_values,
            dims=["row", "col"],
            coords={"row": parameters, "col": parameters},
        )
        ds = xr.Dataset({FISHER_VAR: da})
        fm = FisherMatrix(ds)

        assert fm.parameters == parameters
        np.testing.assert_array_equal(fm.data.values, fisher_values)

    def test_init_from_dataset_missing_fisher_var(self) -> None:
        ds = xr.Dataset({"other_var": xr.DataArray([1, 2, 3])})
        with pytest.raises(ValueError, match="must contain a fisher_matrix"):
            FisherMatrix(ds)

    def test_init_rejects_1d_data(self) -> None:
        da = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            FisherMatrix(da)

    def test_init_with_batch_dimension(
        self, parameters: list[str], batched_fisher_values: np.ndarray
    ) -> None:
        fm = FisherMatrix(batched_fisher_values, parameters, batch_dims=["batch"])

        assert fm.batch_dims == ("batch",)
        assert fm.data.ndim == 3
        assert fm.data.sizes["batch"] == 2

    def test_init_with_batch_dimension_requires_batch_dims(
        self, parameters: list[str], batched_fisher_values: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="batch_dims must be provided"):
            FisherMatrix(batched_fisher_values, parameters)

    def test_init_preserves_multiple_batch_dims(
        self, parameters: list[str], fisher_values: np.ndarray
    ) -> None:
        batch1, batch2 = 2, 3
        n_params = len(parameters)
        multi_batch = np.broadcast_to(
            fisher_values, (batch1, batch2, n_params, n_params)
        ).copy()
        da = xr.DataArray(
            multi_batch,
            dims=["batch1", "batch2", "row", "col"],
            coords={"row": parameters, "col": parameters},
        )
        fm = FisherMatrix(da)

        assert fm.batch_dims == ("batch1", "batch2")
        assert fm.data.sizes["batch1"] == batch1
        assert fm.data.sizes["batch2"] == batch2

    def test_init_with_custom_parameter_dim(
        self, parameters: list[str], fisher_values: np.ndarray
    ) -> None:
        fm = FisherMatrix(
            fisher_values,
            parameters,
            parameter_dim="params",
        )

        assert fm._parameter_dim == "params"
        assert "params" in fm.dataset.coords


class TestFisherMatrixProperties:
    def test_data_property(self, fisher_matrix: FisherMatrix) -> None:
        assert isinstance(fisher_matrix.data, xr.DataArray)
        assert fisher_matrix.data.name is None or FISHER_VAR in str(
            fisher_matrix.dataset
        )

    def test_dataset_property(self, fisher_matrix: FisherMatrix) -> None:
        ds = fisher_matrix.dataset
        assert isinstance(ds, xr.Dataset)
        assert FISHER_VAR in ds

    def test_parameters_property(
        self, fisher_matrix: FisherMatrix, parameters: list[str]
    ) -> None:
        assert fisher_matrix.parameters == parameters

    def test_matrix_dims_property(self, fisher_matrix: FisherMatrix) -> None:
        row_dim, col_dim = fisher_matrix.matrix_dims
        assert row_dim == DEFAULT_ROW_DIM
        assert col_dim == DEFAULT_COL_DIM


class TestCovariance:
    def test_covariance_default_method(self, fisher_matrix: FisherMatrix) -> None:
        cov = fisher_matrix.covariance()

        assert isinstance(cov, xr.DataArray)
        assert cov.shape == fisher_matrix.data.shape
        assert cov.attrs["method"] == "cholesky"

    def test_covariance_is_inverse_of_fisher(
        self, fisher_matrix: FisherMatrix, fisher_values: np.ndarray
    ) -> None:
        cov = fisher_matrix.covariance()

        product = fisher_values @ cov.values
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_covariance_pinv_method(self, fisher_matrix: FisherMatrix) -> None:
        cov = fisher_matrix.covariance(method="pinv")

        assert cov.attrs["method"] == "pinv"
        product = fisher_matrix.data.values @ cov.values
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_covariance_caching(self, fisher_matrix: FisherMatrix) -> None:
        cov1 = fisher_matrix.covariance()
        cov2 = fisher_matrix.covariance()

        xr.testing.assert_equal(cov1, cov2)
        assert COVARIANCE_VAR in fisher_matrix._cache

    def test_covariance_cache_invalidation_on_method_change(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        cov_chol = fisher_matrix.covariance(method="cholesky")
        cov_pinv = fisher_matrix.covariance(method="pinv")

        assert cov_chol.attrs["method"] != cov_pinv.attrs["method"]

    def test_covariance_cache_disabled(self, fisher_matrix: FisherMatrix) -> None:
        cov1 = fisher_matrix.covariance(cache=False)
        cov2 = fisher_matrix.covariance(cache=False)

        assert cov1 is not cov2

    def test_covariance_batched(self, batched_fisher_matrix: FisherMatrix) -> None:
        cov = batched_fisher_matrix.covariance()

        assert cov.ndim == 3
        assert cov.sizes["batch"] == 2

        for i in range(2):
            fisher_i = batched_fisher_matrix.data.isel({"batch": i}).values
            cov_i = cov.isel({"batch": i}).values
            product = fisher_i @ cov_i
            np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_clear_cache(self, fisher_matrix: FisherMatrix) -> None:
        _ = fisher_matrix.covariance()
        fisher_matrix.clear_cache()
        assert fisher_matrix._cache.data_vars == {}


class TestMarginalize:
    def test_marginalize_single_parameter(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.marginalize("c")

        assert result.parameters == ["a", "b"]
        assert result.data.shape == (2, 2)

    def test_marginalize_multiple_parameters(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.marginalize(["b", "c"])

        assert result.parameters == ["a"]
        assert result.data.shape == (1, 1)

    def test_marginalize_preserves_order(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.marginalize("b")

        assert result.parameters == ["a", "c"]

    def test_marginalize_empty_list_returns_self(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        result = fisher_matrix.marginalize([])

        assert result is fisher_matrix

    def test_marginalize_all_parameters_raises(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        with pytest.raises(ValueError, match="Cannot marginalize all"):
            fisher_matrix.marginalize(["a", "b", "c"])

    def test_marginalize_unknown_parameter_raises(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        with pytest.raises(KeyError, match="Unknown parameters"):
            fisher_matrix.marginalize("unknown")

    def test_marginalize_batched(self, batched_fisher_matrix: FisherMatrix) -> None:
        result = batched_fisher_matrix.marginalize("c")

        assert result.parameters == ["a", "b"]
        assert result.data.ndim == 3
        assert result.data.sizes["batch"] == 2

    def test_marginalize_result_has_larger_errors(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        original_cov = fisher_matrix.covariance()
        marginalized = fisher_matrix.marginalize("c")
        marginalized_cov = marginalized.covariance()

        orig_var_a = original_cov.sel(row="a", col="a").values
        marg_var_a = marginalized_cov.sel(row="a", col="a").values
        assert marg_var_a >= orig_var_a


class TestFix:
    def test_fix_single_parameter(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.fix("c")

        assert result.parameters == ["a", "b"]
        assert result.data.shape == (2, 2)

    def test_fix_multiple_parameters(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.fix(["b", "c"])

        assert result.parameters == ["a"]
        assert result.data.shape == (1, 1)

    def test_fix_preserves_submatrix_values(
        self, fisher_matrix: FisherMatrix, fisher_values: np.ndarray
    ) -> None:
        result = fisher_matrix.fix("c")

        expected = fisher_values[:2, :2]
        np.testing.assert_array_equal(result.data.values, expected)

    def test_fix_empty_list_returns_self(self, fisher_matrix: FisherMatrix) -> None:
        result = fisher_matrix.fix([])

        assert result is fisher_matrix

    def test_fix_all_parameters_raises(self, fisher_matrix: FisherMatrix) -> None:
        with pytest.raises(ValueError, match="Cannot fix all"):
            fisher_matrix.fix(["a", "b", "c"])

    def test_fix_unknown_parameter_raises(self, fisher_matrix: FisherMatrix) -> None:
        with pytest.raises(KeyError, match="Unknown parameters"):
            fisher_matrix.fix("unknown")

    def test_fix_batched(self, batched_fisher_matrix: FisherMatrix) -> None:
        result = batched_fisher_matrix.fix("c")

        assert result.parameters == ["a", "b"]
        assert result.data.ndim == 3
        assert result.data.sizes["batch"] == 2

    def test_fix_result_has_smaller_errors(self, fisher_matrix: FisherMatrix) -> None:
        original_cov = fisher_matrix.covariance()
        fixed = fisher_matrix.fix("c")
        fixed_cov = fixed.covariance()

        orig_var_a = original_cov.sel(row="a", col="a").values
        fixed_var_a = fixed_cov.sel(row="a", col="a").values
        assert fixed_var_a <= orig_var_a


class TestFixVsMarginalize:
    def test_fix_gives_tighter_constraints_than_marginalize(
        self, fisher_matrix: FisherMatrix
    ) -> None:
        fixed = fisher_matrix.fix("c")
        marginalized = fisher_matrix.marginalize("c")

        fixed_cov = fixed.covariance()
        marg_cov = marginalized.covariance()

        for param in ["a", "b"]:
            fixed_var = fixed_cov.sel(row=param, col=param).values
            marg_var = marg_cov.sel(row=param, col=param).values
            assert fixed_var <= marg_var
