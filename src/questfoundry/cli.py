"""QuestFoundry CLI - compile and run interactive fiction workflows."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="qf",
    help="QuestFoundry v3 - AI-powered interactive fiction studio",
    no_args_is_help=True,
)
console = Console()


@app.command()
def compile(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes and recompile"),
) -> None:
    """Compile MyST domain files into generated Python code."""
    console.print("[yellow]Compiler not yet implemented[/yellow]")
    if watch:
        console.print("[dim]Watch mode requested[/dim]")


@app.command()
def validate() -> None:
    """Validate domain files without generating code."""
    console.print("[yellow]Validator not yet implemented[/yellow]")


@app.command()
def run(
    loop: str = typer.Argument(..., help="Loop ID to execute (e.g., 'story_spark')"),
) -> None:
    """Execute a workflow loop."""
    console.print(f"[yellow]Runtime not yet implemented for loop: {loop}[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


if __name__ == "__main__":
    app()
