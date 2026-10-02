from pathlib import Path

import numpy as np
import xarray as xr

from fisharr import (
    combine,
    errors,
    fix,
    gaussian_prior,
    inv,
    marginalize,
    matrix,
    transform,
)


def test_all_operations_leave_inputs_unchanged(
    F: xr.DataArray, J: xr.DataArray
) -> None:
    F = F.assign_coords(unit=("row", ["s", "m", "kg"]))
    F.attrs["source"] = "survey"
    before_F = F.copy(deep=True)
    before_J = J.copy(deep=True)
    prior = gaussian_prior({"c": 1, "d": 2})
    before_prior = prior.copy(deep=True)
    results = [
        fix(F, []),
        marginalize(F, []),
        fix(F, "b"),
        marginalize(F, "b"),
        inv(F),
        errors(F),
        transform(F, J),
        combine(F),
        combine(F, prior),
    ]
    for result in results:
        assert not np.shares_memory(result.values, F.values)
        assert not np.shares_memory(result.values, J.values)
        assert not np.shares_memory(result.values, prior.values)
        expected_coords = (
            {"parameter"} if result.dims == ("parameter",) else {"row", "col"}
        )
        assert set(result.coords) == expected_coords
        assert result.attrs == {}
        assert result.dtype == np.float64
        result.values.flat[0] = 999
        for coordinate in result.coords:
            result.coords[coordinate].values[0] = "z"
    xr.testing.assert_identical(F, before_F)
    xr.testing.assert_identical(J, before_J)
    xr.testing.assert_identical(prior, before_prior)


def test_prior_does_not_mutate_mapping() -> None:
    sigmas = {"a": 2.0, "b": 0.5}
    result = gaussian_prior(sigmas)
    result.values[0, 0] = 99
    assert sigmas == {"a": 2.0, "b": 0.5}


def test_netcdf_round_trip(tmp_path: Path) -> None:
    F = matrix([[4, 1], [1, 2]], ["α", "long parameter name"])
    path = tmp_path / "fisher.nc"
    F.to_netcdf(path, engine="h5netcdf")
    restored = xr.load_dataarray(path, engine="h5netcdf")
    xr.testing.assert_identical(restored, F)
    xr.testing.assert_allclose(inv(restored), inv(F))
    xr.testing.assert_allclose(errors(restored), errors(F))
    xr.testing.assert_allclose(combine(restored), F)
