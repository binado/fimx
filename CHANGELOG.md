# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/). The API is alpha and may
change before 1.0.

## [Unreleased]

## [0.3.0]

### Added

- `correlation()` returning the correlation matrix of a Fisher matrix.
- `fom()` returning the Dark Energy Task Force figure of merit,
  `sqrt(det F)` after optional marginalization.
- `return_diagnostics` argument for `inv()` and `correlation()`, returning
  the result alongside an xarray Dataset with the eigenvalue spectrum,
  condition number, rank, positive-definite and positive-semidefinite status,
  and inversion residual.
- `symmetrize()` returning the symmetric part `(F + F.T) / 2` of a labeled
  matrix as a fresh canonical matrix.
- `fimx.ops` and `fimx.arrays` submodules for operations and construction.
- `vector()` for constructing labeled one-dimensional parameter arrays.
- `fimx.io.save_dataset` to write a forecast with optional `covariance`,
  `fiducials` and plotting `labels`.
- `--save` option for `fimx invert`, which writes the Dataset with a freshly
  computed covariance.
- `plot` and `fimx plot` use an optional `labels` variable for axis labels.
- Optional `cli` extra, which provides the `fimx` command and includes the
  `io` extra.

### Changed

- `fom()` validates positive definiteness using Cholesky and computes the
  result from its diagonal without forming the determinant.
- `correlation()` rejects indefinite input and covariance using the same
  scale-relative eigenvalue tolerance as `inv()` diagnostics. Singular positive
  semidefinite input remains supported with `method="pinv"` when every
  covariance diagonal entry is positive.
- `fimx invert` takes its condition number from inversion diagnostics. The number is
  infinite unless the matrix is positive definite, and the JSON report
  encodes that non-finite value as `null`. Its report and `--save` reuse
  the same computed covariance.
- Internal reorganization: `construction` is renamed `arrays` and hosts
  `gaussian_prior`; `combination` and `parameters` are merged into `ops`. The
  flat top-level API is unchanged.
- `fimx-plot` and `fimx-invert` are subcommands of one `fimx` command:
  `fimx plot` and `fimx invert`.
- `fimx plot --parameters` accepts multiple parameter names by repeating the
  option.
- `fimx invert --inversion-method` takes one method (`cholesky` by default)
  instead of evaluating every method, and the JSON report replaces the
  per-method `methods` object with top-level `method`, `success`,
  `max_abs_residual` and `error` keys.

### Removed

- The `--plot-label-var` option of `fimx plot`; store the labels in a variable
  named `labels` instead.
- The repeatable `--inversion-method` option of `fimx invert` and the "first
  successful selected method" fallback of its `--save` option.

## [0.2.0]

### Added

- `method` argument (`"cholesky"`, `"inv"`, or `"pinv"`) for `inv`, `errors`, and
  `plot`, and a matching `--inversion-method` option for `fimx-plot`.

### Changed

- `plot` now inverts the Fisher matrices itself; `PlotBackend` implementations
  receive prepared means and covariances instead of Datasets.

## [0.1.0]

### Added

- Functions on labeled `xarray.DataArray` Fisher matrices: `matrix`, `expand`,
  `fix`, `marginalize`, `inv`, `errors`, `transform`, `combine`, and
  `gaussian_prior`.
- `dataset()` to bundle a Fisher matrix with fiducials and other metadata.
- `plot()` for corner plots of analytic Gaussians, with a GetDist backend and a
  `PlotBackend` interface for additional backends.
- `fimx-plot` command-line tool and `fimx.io.load_dataset` for NetCDF files.
- Optional `io` and `plotting` extras, and a `py.typed` marker.

[Unreleased]: https://github.com/binado/fimx/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/binado/fimx/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/binado/fimx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/binado/fimx/commits/main
