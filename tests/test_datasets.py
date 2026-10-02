"""Public forecast Dataset construction contracts."""

import numpy as np
import pytest
import xarray as xr

from fisharr import dataset, matrix


def test_dataset_alignment_and_independence(F: xr.DataArray) -> None:
    reference = xr.DataArray([30, 10, 20], dims="row", coords={"row": ["c", "a", "b"]})
    original = F.copy(deep=True)
    original_reference = reference.copy(deep=True)
    result = dataset(F, reference)
    assert set(result.data_vars) == {"fisher", "fiducials"}
    assert result.fiducials.dims == ("row",)
    assert result.fiducials.dtype == np.float64
    np.testing.assert_array_equal(result.fiducials.values, [10, 20, 30])
    result.fisher.values[0, 0] = 100
    result.fiducials.values[0] = 100
    result.row.values[0] = "changed"
    xr.testing.assert_identical(F, original)
    xr.testing.assert_identical(reference, original_reference)


@pytest.mark.parametrize(
    "reference",
    [
        xr.DataArray(
            [1, 2, 3], dims="parameter", coords={"parameter": ["a", "b", "c"]}
        ),
        xr.DataArray([1, 2, 3], dims="row"),
        xr.DataArray([1, 2, 3], dims="row", coords={"row": ["a", "a", "c"]}),
        xr.DataArray([1, 2, 3], dims="row", coords={"row": ["a", "b", "d"]}),
        xr.DataArray([1, 2], dims="row", coords={"row": ["a", "b"]}),
        xr.DataArray([1, np.nan, 3], dims="row", coords={"row": ["a", "b", "c"]}),
        xr.DataArray([1, np.inf, 3], dims="row", coords={"row": ["a", "b", "c"]}),
        xr.DataArray([1j, 2j, 3j], dims="row", coords={"row": ["a", "b", "c"]}),
    ],
)
def test_invalid_fiducials(F: xr.DataArray, reference: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        dataset(F, reference)


def test_dataset_requires_arrays(F: xr.DataArray) -> None:
    with pytest.raises(TypeError):
        dataset(F, [1, 2, 3])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        dataset([[1]], xr.DataArray([0], dims="row", coords={"row": ["a"]}))  # ty: ignore[invalid-argument-type]


def test_dataset_accepts_singular_information() -> None:
    result = dataset(
        matrix([[0]], ["a"]), xr.DataArray([0], dims="row", coords={"row": ["a"]})
    )
    assert result.fisher.item() == 0
