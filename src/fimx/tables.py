"""LaTeX tables of marginalized errors from stored forecast covariances."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .arrays import _real_values
from .datasets import _display_labels, _stored_covariance

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(_LATEX_SPECIAL.get(char, char) for char in text)


def _significant(value: float, digits: int) -> str:
    """Format a number with significant digits, using LaTeX for exponents."""
    text = f"{value:#.{digits}g}"
    mantissa, _, exponent = text.partition("e")
    mantissa = mantissa.removesuffix(".")
    if not exponent:
        return mantissa
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def latex_table(
    forecast: xr.Dataset,
    *,
    parameters: Sequence[str] | None = None,
    parameter_labels: Mapping[str, str] | None = None,
    significant_digits: int = 2,
    fiducials: bool = False,
) -> str:
    r"""Build a LaTeX table of marginalized errors from a stored covariance.

    Parameters
    ----------
    forecast : xarray.Dataset
        Dataset containing a finite, symmetric, positive semidefinite
        ``covariance`` on ``('row', 'col')`` with positive variances.
    parameters : sequence of str, optional
        Unique parameters to tabulate, in order. Defaults to covariance order.
        Selection does not condition on omitted parameters.
    parameter_labels : mapping of str to str, optional
        Display labels overriding the Dataset's optional ``labels`` on ``row``.
        Labels are unique LaTeX math strings without enclosing ``$``.
        Parameters without a display label use their escaped plain names.
    significant_digits : int, default 2
        Positive number of significant digits for every number. Trailing zeros
        are kept; values needing exponents render as ``$1.2\times10^{-4}$``.
    fiducials : bool, default False
        Add a ``Fiducial`` column from the Dataset's ``fiducials`` variable.

    Returns
    -------
    str
        LaTeX ``tabular`` with one row per parameter and the marginalized
        error :math:`\sigma_i = \sqrt{C_{ii}}` in the last column.

    Raises
    ------
    TypeError
        If forecast or parameter_labels has an incorrect type.
    ValueError
        If covariance, labels, parameter selection, ``significant_digits``, or
        requested fiducials are invalid or missing.
    KeyError
        If a selected parameter is absent.
    numpy.linalg.LinAlgError
        If covariance is not positive semidefinite or has nonpositive variances.
    ImportError
        If Jinja2 is unavailable; install ``fimx[tables]``.
    """
    if (
        isinstance(significant_digits, bool)
        or not isinstance(significant_digits, int)
        or significant_digits < 1
    ):
        raise ValueError("significant_digits must be a positive integer.")
    covariance, _, selected = _stored_covariance(forecast, parameters)
    names = covariance.row.values.tolist()
    display = _display_labels(forecast, names, selected, parameter_labels)
    sigma = np.sqrt(np.diag(covariance.sel(row=selected, col=selected).values))
    columns: dict[str, list[str]] = {
        "Parameter": [
            f"${display[n]}$" if n in display else _escape(n) for n in selected
        ]
    }
    if fiducials:
        if "fiducials" not in forecast.data_vars:
            raise ValueError("Dataset must contain a 'fiducials' variable.")
        reference = forecast["fiducials"]
        if reference.dims != ("row",):
            raise ValueError("Fiducial dimensions must be exactly ('row',).")
        values = _real_values(reference.sel(row=selected).values)
        columns["Fiducial"] = [_significant(v, significant_digits) for v in values]
    columns[r"$\sigma$"] = [_significant(v, significant_digits) for v in sigma]
    frame = pd.DataFrame(columns)
    try:
        return frame.to_latex(index=False, escape=False)
    except ImportError as exc:
        raise ImportError(
            "LaTeX tables require Jinja2. Install it with uv add 'fimx[tables]'."
        ) from exc
