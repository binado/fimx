## About the package

`fimx` is a Python package providing functions to construct, manipulate, and
analyze dense Fisher matrices represented as labeled `xarray.DataArray`
objects. Matrices have fixed `("row", "col")` dimensions and matching ordered
parameter coordinates. Storage and metadata are delegated to xarray.

The `arrays` module constructs labeled arrays (`matrix`, `vector`,
`gaussian_prior`) and owns the shared validation helpers. The `ops` module
holds Fisher matrix operations (`combine`, `symmetrize`, `expand`, `fix`,
`marginalize`, `transform`, `correlation`, `fom`); `inversion` covers `inv`
and `errors`. The public API is re-exported flat at the top level.

## Project management

- Use `uv` for installing packages and running tools
- Use `uvx ruff format` for formatting
- Use `uvx ruff check` for linting
- Use `uvx ty check` for type-checking
- Use the conventional-commits format for commit messages

## Python style guide

- Fully type-annotate functions and methods
- Use numpy-style docstrings
- Separate methods in different modules according to their functionality, semantics and context

## Testing syle guide

- Use pytest fixtures and `pytest.mark.parametrize` when convenient
- Don't test internal implementation
- Don't write brittle tests with finetuned tolerances for `np.testing.assert_allclose`
