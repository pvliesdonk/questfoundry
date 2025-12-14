"""
QuestFoundry CLI - Cleanroom Rebuild

The entire runtime and CLI are being rebuilt from scratch.
See RUNTIME-CLEANROOM-BRIEF.md for design principles.

Previous implementation archived at: _archive/cli-v3.py
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="qf",
    help="QuestFoundry - AI-powered interactive fiction studio",
    no_args_is_help=True,
)
console = Console()

CLEANROOM_MESSAGE = """
[bold yellow]Cleanroom Rebuild In Progress[/bold yellow]

The runtime and CLI are being rebuilt from scratch based on meta/ schemas.

See [cyan]RUNTIME-CLEANROOM-BRIEF.md[/cyan] for:
- Design principles
- Implementation phases
- Patterns to preserve

Previous implementation archived to [dim]_archive/[/dim]
"""


@app.command()
def status() -> None:
    """Show cleanroom rebuild status."""
    console.print(Panel(CLEANROOM_MESSAGE, title="QuestFoundry", border_style="yellow"))


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__
    console.print(f"QuestFoundry v{__version__} [dim](cleanroom rebuild)[/dim]")


if __name__ == "__main__":
    app()
