import numpy as np
import pytest
import xarray as xr

from fisharr import matrix


@pytest.fixture
def F() -> xr.DataArray:
    """Return a correlated positive definite matrix."""
    return matrix([[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]], ["a", "b", "c"])


@pytest.fixture
def J() -> xr.DataArray:
    """Return a rectangular Jacobian with reordered old labels."""
    return xr.DataArray(
        [[1, 2], [2, 0], [0, 1]],
        dims=("old_parameter", "new_parameter"),
        coords={"old_parameter": ["c", "a", "b"], "new_parameter": ["y", "x"]},
    )


@pytest.fixture
def direct() -> xr.DataArray:
    """Return a valid direct DataArray carrying extra metadata."""
    return xr.DataArray(
        np.array([[4, 1], [1, 2]], dtype=np.int32),
        dims=("row", "col"),
        coords={"row": ["a", "b"], "col": ["a", "b"], "unit": ("row", ["s", "m"])},
        name="survey",
        attrs={"source": "example"},
    )
