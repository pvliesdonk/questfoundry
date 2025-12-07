"""QuestFoundry CLI - compile and run interactive fiction workflows.

Commands:
    qf ask "message"    Talk to the studio (SR-orchestrated execution)
    qf doctor           Check system status
    qf roles            List available specialist roles
    qf compile          Compile domain files (not yet implemented)
    qf version          Show version information
"""

from __future__ import annotations

import typer
from rich.console import Console

from questfoundry.runtime.cli.main import ask, doctor, roles

app = typer.Typer(
    name="qf",
    help="QuestFoundry v3 - AI-powered interactive fiction studio",
    no_args_is_help=True,
)
console = Console()

# Register runtime commands
app.command()(ask)
app.command()(doctor)
app.command()(roles)


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
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


if __name__ == "__main__":
    app()
