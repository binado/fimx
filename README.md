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

## Quickstart

```python
from fimx import combine, errors, fix, gaussian_prior, inv, marginalize, matrix

F = matrix([[4, 1], [1, 2]], ["a", "b"])
fixed = fix(F, "b")  # [[4.0]]: b is held fixed
reduced = marginalize(F, "b")  # [[3.5]]: uncertainty in b is integrated out
C = inv(F)  # covariance, with the same row/col labels
sigma = errors(F)  # sqrt(diag(C)), dimension "parameter"

prior = gaussian_prior({"a": 0.5})  # standard deviation 0.5 -> information 4
posterior = combine(F, prior)
```

| Function | Behavior |
| --- | --- |
| `matrix(values, parameters)` | Construct and validate a canonical matrix. |
| `vector(values, parameters)` | Construct a canonical one-dimensional parameter array. |
| `expand(F, parameters)` | Embed in a larger or reordered parameter set, filling with zeros. |
| `fix(F, parameters)` | Remove names through a principal submatrix. |
| `marginalize(F, parameters)` | Remove names through the Schur complement. |
| `inv(F, method="cholesky", metadata=False)` | Return the inverse; `method` is `cholesky`, `inv`, or `pinv`. With `metadata=True`, `attrs` hold `method`, `condition_number` and `residual`. |
| `errors(F, method="cholesky")` | Return marginalized standard deviations. |
| `transform(F, jacobian)` | Change variables using `J.T @ F @ J`. |
| `combine(*matrices)` | Sum independent information over the parameter union. |
| `gaussian_prior(sigmas)` | Construct diagonal information `1 / sigma**2`. |

## Array contract

- Matrices have dimensions exactly `("row", "col")`, a nonempty square shape,
  and explicit, identical, ordered coordinates of unique string names.
  Batches, inferred dimensions, and coordinate repair are unsupported.
- Values must be real, finite, and symmetric. They are converted to float64.
- Vectors have dimension exactly `("row",)`, a nonempty shape, and explicit
  coordinates of unique string names. Values retain their input dtype.
- Positive semidefiniteness is not checked, so singular matrices can be built,
  fixed, transformed, or combined. `inv` and `errors` require positive
  definite input; `marginalize` only requires the removed block to be.
- Functions return fresh DataArrays and never mutate their inputs. Attributes,
  names, and auxiliary coordinates are not preserved.
- Arrays built directly with xarray work if they meet the contract.

Behavior worth knowing:

- `fix` and `marginalize` accept one name or a sequence and keep the remaining
  order. Unknown names raise `KeyError`; duplicates or removing every
  parameter raise `ValueError`.
- `expand` embeds a matrix in a larger or reordered set of names, filling new
  entries with zeros. The target must contain every existing name. Matrices
  expanded to the same names add with plain `+`, whereas `+` on mismatched
  names silently keeps only the overlap.
- `combine` keeps the first matrix's order and appends new parameters as they
  appear. Missing entries contribute zero.
- `gaussian_prior` takes **standard deviations**, not information values.
- Failures are explicit: `ValueError` for malformed or non-finite data,
  `TypeError` for non-DataArray inputs, and `numpy.linalg.LinAlgError` for
  singular or indefinite matrices. By default `inv` and `errors` use Cholesky and
  require positive definiteness. `method="inv"` only fails on exactly singular
  matrices, and `method="pinv"` uses the pseudoinverse, which assigns zero
  variance to unconstrained directions. There is no regularization.

### Changes of variables

The Jacobian has dimensions `("old", "new")` and orientation
**`J[i, j] = d theta_i / d phi_j`**, where theta are the old parameters and phi
the new ones. Every old parameter must appear exactly once (rows are reordered
to match `F`), and the order of the new parameters sets the output order.
Rectangular Jacobians are supported.

```python
import xarray as xr
from fimx import transform

# theta_a = 2 * phi_x, theta_b = phi_x
# Rows deliberately appear in reverse order to F.
J = xr.DataArray(
    [[1.0], [2.0]],
    dims=("old", "new"),
    coords={"old": ["b", "a"], "new": ["x"]},
)
G = transform(F, J)  # [[22.0]]
```

## Plotting

Plotting needs the `plotting` extra (GetDist and Matplotlib, loaded lazily).

### Forecast Datasets

Keep fiducials separate from the Fisher matrix and bundle them with
`dataset()`. In the mapping of extra variables, 1D arrays use `row` and 2D
arrays use `(row, col)`; plain arrays follow matrix order and indexed axes are
aligned by label. Strings, units, and nonfinite metadata are allowed. The names
`fisher`, `row`, and `col` are reserved.

### Corner plots

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

- `plot()` takes a nonempty mapping of labels to Datasets with `fisher` and
  finite, real `fiducials`. Each forecast is centered on its own fiducials;
  mapping order sets the overlay order and keys become legend labels.
- By default, the plot uses the parameters shared by all forecasts, in the
  first forecast's order. An explicit `parameters` list must be unique and
  present in every forecast.
- Each full Fisher matrix is inverted before selecting parameters, so omitted
  parameters are marginalized and every complete matrix (nuisance blocks
  included) must be positive definite.
- The GetDist backend draws analytic Gaussians, with no sampling, and returns
  a Matplotlib `Figure` without showing or saving it. Filled contours are the
  default. `backend_kwargs` is forwarded to `triangle_plot`, except `roots`,
  `params`, `legend_labels`, and `filled`, which `plot()` controls.
- Additional backends implement `fimx.plotting.base.PlotBackend`.

### Command line

With both extras installed, `fimx-plot` draws a corner plot from NetCDF
Datasets that contain `fisher` and `fiducials`:

```sh
uv add 'fimx[io,plotting]'
fimx-plot --file survey-a.nc --file survey-b.nc \
    --figure-file constraints.png --figure-dpi 200 \
    --parameters a b --no-filled --backend getdist \
    --backend-kwargs '{"contour_colors": ["C0", "C1"]}'
```

Each file stem becomes a legend label, so stems must be unique. The output
defaults to `plot.png` at 150 dpi. `--parameters` selects and orders
parameters. Axis labels come from an optional `labels` variable (see
[Storage](#storage)); labels for shared plotted parameters must agree across
files, and parameter names are used where there are none. `--no-filled` draws
line contours, `--inversion-method` picks `cholesky`, `inv`, or `pinv`,
`--backend` picks a backend, and `--backend-kwargs` takes a JSON object.
`fimx.io.load_dataset(path)` loads the same files from Python.

`fimx-invert` reports matrix conditioning and inversion residuals from either
a standalone matrix DataArray or a Dataset containing `fisher`; an existing
`covariance` is ignored:

```sh
fimx-invert --file fisher.nc
fimx-invert --file forecast.nc --inversion-method inv pinv --json
fimx-invert --file forecast.nc --save forecast-with-covariance.nc
```

All three inversion methods (`cholesky`, `inv`, and `pinv`) are evaluated by
default. Each method reports the maximum absolute element of `F @ F_inv - I`;
methods that cannot invert the matrix report their error while the remaining
methods continue. The report also includes the condition number, numerical
rank, eigenvalue range, and positive-definite status. `--save PATH` writes the
Dataset, with its `fiducials` and `labels`, plus a new `covariance` from the
first successful selected method in the order `cholesky`, `inv`, `pinv`; the
method is reported on stderr and recorded in the covariance's `attrs` together
with its condition number and residual. Nothing is written if every selected
method fails.

## Storage

Matrices are plain xarray objects, so any xarray writer works. The `io` extra
supplies `h5netcdf`.

```python
F.to_netcdf("fisher.nc", engine="h5netcdf")
restored = xr.load_dataarray("fisher.nc", engine="h5netcdf")
C_restored = inv(restored)
```

For forecasts, `fimx.io.save_dataset` writes the Dataset convention read by
`plot`, `fimx-plot` and `fimx-invert`; `fimx.io.load_dataset` reads it back.

```python
from fimx.io import load_dataset, save_dataset

save_dataset(
    "forecast.nc",
    F,
    covariance=inv(F, metadata=True),  # optional, never computed for you
    fiducials=[0.3, 0.7, 1.0],
    labels=[r"$\Omega_m$", r"$h$", r"$\sigma_8$"],
)
forecast = load_dataset("forecast.nc")
```

| Variable | Dimensions | Required | Meaning |
| --- | --- | --- | --- |
| `fisher` | `row`, `col` | yes | Fisher matrix. |
| `covariance` | `row`, `col` | no | Inverse or pseudoinverse of `fisher`; keeps the `attrs` of `inv(..., metadata=True)`. |
| `fiducials` | `row` | no (needed to plot) | Reference parameter values. |
| `labels` | `row` | no | Unique axis labels for plotting. |

`covariance` is checked loosely against `fisher` so that a matrix from another
forecast is rejected; it is not recomputed.

## More

See [CHANGELOG.md](CHANGELOG.md) for release notes. `fimx` is released under
the MIT license; see [LICENSE](LICENSE).
