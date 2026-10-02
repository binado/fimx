"""Command-line interface for Fisher matrix utilities."""

from collections.abc import Sequence

import typer
from typer._click import Context as ClickContext
from typer.core import TyperGroup

from .invert import invert
from .plot import plot

_VARIADIC_OPTIONS = {
    "plot": {"--parameters"},
    "invert": {"--inversion-method"},
}


def _expand_variadic(argv: Sequence[str]) -> list[str]:
    """Repeat flags that accept several space-separated values.

    Typer collects list options by repeating the flag. ``fimx`` also accepts
    several values after one flag, as in ``--parameters a b``.
    """
    args = list(argv)
    command = next((arg for arg in args if not arg.startswith("-")), None)
    flags = _VARIADIC_OPTIONS.get(command or "")
    if not flags:
        return args
    expanded: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        expanded.append(token)
        index += 1
        name = token.split("=", 1)[0]
        if name not in flags or "=" in token:
            continue
        values: list[str] = []
        while index < len(args) and not args[index].startswith("-"):
            values.append(args[index])
            index += 1
        if not values:
            continue
        expanded.append(values[0])
        for value in values[1:]:
            expanded.extend((token, value))
    return expanded


class _FimxGroup(TyperGroup):
    """Typer group that accepts space-separated multi-value options."""

    def parse_args(self, ctx: ClickContext, args: list[str]) -> list[str]:
        """Expand variadic options, then parse the command line."""
        return super().parse_args(ctx, _expand_variadic(args))


app = typer.Typer(
    cls=_FimxGroup,
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
