"""QuestFoundry CLI - run interactive fiction workflows.

Commands:
    qf ask "message"    Talk to the studio (SR-orchestrated execution)
    qf doctor           Check system status
    qf roles            List available specialist roles
    qf version          Show version information
"""

from __future__ import annotations

import typer
from rich.console import Console

from questfoundry.runtime.cli.main import ask, doctor, roles

app = typer.Typer(
    name="qf",
    help="QuestFoundry - AI-powered interactive fiction studio",
    no_args_is_help=True,
)
console = Console()

# Register runtime commands
app.command()(ask)
app.command()(doctor)
app.command()(roles)


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


if __name__ == "__main__":
    app()
