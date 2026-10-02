# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/). The API is alpha and may
change before 1.0.

## [Unreleased]

### Added

- `vector()` for constructing labeled one-dimensional parameter arrays.
- `metadata` argument for `inv`, which records the method, condition number and
  inverse residual in the result's `attrs`.
- `fimx.io.save_dataset` to write a forecast with optional `covariance`,
  `fiducials` and plotting `labels`.
- `--save` option for `fimx invert`, which writes the Dataset with a freshly
  computed covariance.
- `plot` and `fimx plot` use an optional `labels` variable for axis labels.
- Optional `cli` extra, which provides the `fimx` command and includes the
  `io` extra.

### Changed

- `fimx-plot` and `fimx-invert` are subcommands of one `fimx` command:
  `fimx plot` and `fimx invert`.

### Removed

- The `--plot-label-var` option of `fimx plot`; store the labels in a variable
  named `labels` instead.

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

[Unreleased]: https://github.com/binado/fimx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/binado/fimx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/binado/fimx/commits/main
