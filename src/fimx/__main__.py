"""Console script for the optional command-line interface."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fimx command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to the process command line.

    Raises
    ------
    SystemExit
        If the ``cli`` extra is not installed.
    """
    # Imported here so this module can explain a missing optional extra.
    try:
        from .cli import main as run
    except ModuleNotFoundError as exc:
        if exc.name != "typer":
            raise
        raise SystemExit(
            "The fimx command requires the cli extra. "
            "Install it with: uv add 'fimx[cli]'"
        ) from exc
    run(argv)


if __name__ == "__main__":
    main()
