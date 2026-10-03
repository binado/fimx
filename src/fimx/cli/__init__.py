"""Command-line interface for Fisher matrix utilities."""

from collections.abc import Sequence

import typer

from .invert import invert
from .plot import plot

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Construct, manipulate, and analyze dense Fisher matrices.",
)
app.command()(plot)
app.command()(invert)


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
