"""Command-line interface for Fisher matrix utilities."""

from collections.abc import Sequence

import typer

from .correlation import plot_correlation
from .fix import fix
from .invert import invert
from .marginalize import marginalize
from .plot import plot
from .table import table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Construct, manipulate, and analyze dense Fisher matrices.",
)
plot_app = typer.Typer(
    no_args_is_help=True, help="Plot forecast constraints and correlations."
)
plot_app.command("corner")(plot)
plot_app.command("matrix")(plot_correlation)
app.add_typer(plot_app, name="plot")
app.command()(invert)
app.command()(fix)
app.command()(marginalize)
app.command()(table)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fimx command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to the process command line.
    """
    app(args=None if argv is None else list(argv))


if __name__ == "__main__":
    main()
