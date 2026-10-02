# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/). The API is alpha and may
change before 1.0.

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

[Unreleased]: https://github.com/binado/fimx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/binado/fimx/commits/main
