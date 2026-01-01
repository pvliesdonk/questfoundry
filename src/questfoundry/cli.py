"""QuestFoundry CLI - typer application entry point."""

import typer
from rich.console import Console

app = typer.Typer(
    name="qf",
    help="QuestFoundry: Pipeline-driven interactive fiction generation.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


@app.command()
def status() -> None:
    """Show pipeline status for current project."""
    console.print("[yellow]Not implemented yet[/yellow]")


if __name__ == "__main__":
    app()
