# fimx

`fimx` provides functions for dense Fisher information matrices stored
as ordinary `xarray.DataArray` objects. NumPy handles the computation; xarray
handles labels, metadata containers, and persistence.

```sh
uv add fimx
# Optional NetCDF backend:
uv add 'fimx[io]'
# Optional Gaussian plotting backend:
uv add 'fimx[plotting]'
```

## Array contract

Every matrix input must have dimensions exactly `("row", "col")`, a nonempty
square shape, and explicit, identical, ordered coordinates on both axes.
Parameter names must be unique strings. Values must be real numeric, finite,
and symmetric (`np.allclose` with `rtol=1e-10`, `atol=1e-12`). Batches, inferred
dimensions, missing coordinates, and automatic coordinate repair are unsupported.
Accepted values are converted to float64 for computation.

Fisher matrices are expected to be positive semidefinite. Ordinary validation
does not check eigenvalues: singular matrices can still be constructed, fixed,
transformed, or combined. `inv` and `errors` require positive definite inputs.
`marginalize` requires only the removed block to be positive definite.

Every function returns a fresh DataArray with canonical parameter coordinates
and independent writable data. Inputs are never mutated. Arbitrary attributes,
names, and auxiliary coordinates are outside the preservation contract.
You can also supply arrays created directly with xarray if they meet the contract.

## Functions

```python
import numpy as np
import xarray as xr
from fimx import (
    matrix,
    fix,
    marginalize,
    inv,
    errors,
    transform,
    combine,
    gaussian_prior,
)

F = matrix([[4, 1], [1, 2]], ["a", "b"])
fixed = fix(F, "b")  # [[4.0]]: b is held fixed
reduced = marginalize(F, "b")  # [[3.5]]: uncertainty in b is integrated out
C = inv(F)  # covariance, with the same row/col labels
sigma = errors(F)  # sqrt(diag(C)), dimension "parameter"

prior = gaussian_prior({"a": 0.5})  # standard deviation 0.5 -> information 4
posterior = combine(F, prior)

# Reuse C without another inverse:
sigma_from_C = xr.DataArray(
    np.sqrt(np.diag(C.values)),
    dims="parameter",
    coords={"parameter": C.row.values},
)
```

| Function | Behavior |
| --- | --- |
| `matrix(values, parameters)` | Construct and validate a canonical matrix. |
| `fix(F, parameters)` | Remove names through a principal submatrix. |
| `marginalize(F, parameters)` | Remove names through the Schur complement. |
| `inv(F)` | Return the inverse using Cholesky. |
| `errors(F)` | Return marginalized standard deviations. |
| `transform(F, jacobian)` | Change variables using `J.T @ F @ J`. |
| `combine(*matrices)` | Sum independent information over the parameter union. |
| `gaussian_prior(sigmas)` | Construct diagonal information `1 / sigma**2`. |

Removal accepts a single name or a sequence and preserves the remaining input
order. Empty selections return fresh copies. Unknown names raise `KeyError`;
duplicate selections or removing every parameter raise `ValueError`.

Combination preserves the first matrix's order, appending newly encountered
parameters in subsequent input order. Missing entries contribute zero. Each
input is validated before alignment, so existing invalid values are never filled.
Calling `combine()` without inputs raises `ValueError`.

Priors preserve mapping order. Their inputs are **standard deviations**, not
diagonal information values. Empty mappings, nonnumeric, nonfinite, zero, and
negative standard deviations raise `ValueError`.

### Changes of variables

The Jacobian must have dimensions `("old", "new")`,
finite real values, and unique string coordinates on both axes. Its orientation
is **`J[i, j] = d theta_i / d phi_j`**, where theta denotes the old parameters
and phi the new ones. All old parameters must be present exactly once; rows
are reordered to match F. New parameters must be nonempty, and their order
determines the output. Rectangular Jacobians are supported.

```python
# theta_a = 2 * phi_x, theta_b = phi_x
# Rows deliberately appear in reverse order to F.
J = xr.DataArray(
    [[1.0], [2.0]],
    dims=("old", "new"),
    coords={"old": ["b", "a"], "new": ["x"]},
)
G = transform(F, J)  # [[22.0]]
```

### Failures

Malformed matrices and Jacobians raise `ValueError`; inputs that must be
DataArrays raise `TypeError` when given other objects. Cholesky failures propagate
as `numpy.linalg.LinAlgError` for singular or indefinite matrices (or removed
blocks). There is no fallback, pseudoinverse, or regularization. Computed matrix
outputs that cannot be represented as finite float64 values raise `ValueError`.

## Metadata and storage

### Forecast Datasets and plotting

Keep fiducials separate from the Fisher matrix, then use `dataset()` to combine
them and other metadata with shared parameter coordinates. Pass a mapping of
variable names to arrays: 1D arrays use `row`, and 2D arrays use `(row, col)`.
Plain arrays follow matrix order; DataArray dimensions are renamed by position
and indexed axes are aligned by label. Unindexed axes follow matrix order.
Xarray validates compatible sizes and coordinates. Additional arrays retain
their dtypes, so strings, units, and nonfinite metadata are supported. The helper
returns independent copies and does not require positive definiteness.
Variable names `fisher`, `row`, and `col` are reserved.

```python
from fimx import dataset, plot

fiducials = xr.DataArray([1.0, 2.0], dims="row", coords={"row": ["a", "b"]})
survey_a = dataset(F, {"fiducials": fiducials, "units": ["km", "s"]})
survey_b = dataset(2 * F, {"fiducials": fiducials})

fig = plot({"Survey A": survey_a, "Survey B": survey_b})
fig.savefig("constraints.pdf")

# Explicit order and GetDist customization:
fig = plot(
    {"Survey A": survey_a, "Survey B": survey_b},
    parameters=["b", "a"],
    backend="getdist",
    filled=False,
    backend_kwargs={"contour_colors": ["C0", "C1"]},
)
```

`plot()` accepts only a nonempty mapping of string labels to Datasets containing
`fisher` and `fiducials`. Directly constructed Datasets may carry additional
metadata. Plotting requires finite, real fiducials on `row`. Each forecast
supplies its own center; mapping order controls overlay order and keys become
legend labels. Inputs are never mutated.

By default, plots use the intersection of parameter names in the first
forecast's order. An explicit `parameters` selection must be nonempty, unique,
and present in every forecast. Each complete Fisher matrix is inverted before
selecting parameters, so omitted parameters are marginalized over. Consequently,
all complete matrices must be positive definite, including nuisance blocks.
Disjoint parameter sets raise `ValueError`; unavailable requested names raise
`KeyError`.

The GetDist backend plots analytic Gaussians without generating random samples.
It returns a Matplotlib `Figure` without showing or saving it. Filled contours
are enabled by default; styles and confidence levels otherwise follow GetDist's
defaults. `backend_kwargs` forwards options to `triangle_plot`, excluding
`roots`, `params`, `legend_labels`, and `filled`, which the wrapper controls.
Unknown backend names raise `ValueError`.

### Command-line corner plots

Install both optional dependency groups and run `fimx-plot` with one or more
NetCDF Datasets containing `fisher` and `fiducials`:

```sh
uv add 'fimx[io,plotting]'
fimx-plot --file survey-a.nc --file survey-b.nc \
    --figure-file constraints.png --figure-dpi 200 \
    --parameters a b --no-filled --backend getdist \
    --backend-kwargs '{"contour_colors": ["C0", "C1"]}'
```

Each input filename stem becomes its legend label, so stems must be unique.
The output defaults to `plot.png` at 150 dpi. `fimx.io.load_dataset(path)`
loads NetCDF Datasets with the optional `h5netcdf` engine.
Use `--parameters` to select and order parameters, `--no-filled` to draw line
contours, `--backend` to select a backend, and `--backend-kwargs` to pass a JSON
object of backend-specific options.

GetDist and Matplotlib are optional and loaded only when plotting is requested.
`fimx.plotting.base.PlotBackend` defines the callable interface for additional
backends. Shared preparation lives in `base`, backend-specific rendering in
`getdist`, and `fimx.plotting.plot` selects the registered implementation.

### Other metadata

Keep parameter metadata in your own Dataset, separately from computation:

```python
ds = xr.Dataset(
    {
        "fisher": F,
        "covariance": C,
        "fiducial": ("parameter", [1.0, 2.0]),
        "unit": ("parameter", ["km", "s"]),
        "label": ("parameter", ["Parameter a", "Parameter b"]),
    },
    coords={"parameter": F.row.values},
)
sigma = errors(ds["fisher"])

# The optional io extra supplies h5netcdf. There is no fimx file format.
F.to_netcdf("fisher.nc", engine="h5netcdf")
restored = xr.load_dataarray("fisher.nc", engine="h5netcdf")
C_restored = inv(restored)
```

## Migration from FisherMatrix

This is a clean API break: the wrapper class and its batch, indexing, cache,
diagnostic, sampling, and metadata infrastructure have been removed.

| Previous API | Replacement |
| --- | --- |
| `FisherMatrix(values, parameters)` | `matrix(values, parameters)` |
| `fm.covariance()` | `inv(F)` |
| `fm.marginalized_errors()` | `errors(F)` |
| Matrix addition | `combine(F1, F2)` |
| Prior application | `combine(F, gaussian_prior({"a": sigma_a}))` |
| Persistence | `F.to_netcdf(...)` and `xr.load_dataarray(...)` |

If an old prior contains diagonal information `p`, pass `sigma = 1 / sqrt(p)`
to `gaussian_prior`, or construct a labeled information matrix and combine it.
Check existing Jacobians against the old-by-new derivative orientation above.
Use xarray directly for selection and metadata; matrix operations still require
both axes to satisfy the full contract.
