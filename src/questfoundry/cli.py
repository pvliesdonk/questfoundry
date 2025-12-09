"""QuestFoundry CLI - compile and run interactive fiction workflows.

Commands:
    qf ask "message"    Talk to the studio (SR-orchestrated execution)
    qf doctor           Check system status
    qf roles            List available specialist roles
    qf compile          Compile domain files to generated Python code
    qf validate         Validate domain files without generating
    qf version          Show version information
"""

from __future__ import annotations

from pathlib import Path

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

# Default paths relative to project root
DEFAULT_DOMAIN_DIR = "src/questfoundry/domain"
DEFAULT_OUTPUT_DIR = "src/questfoundry/generated"


@app.command()
def compile(
    domain: str = typer.Option(
        DEFAULT_DOMAIN_DIR, "--domain", "-d", help="Path to domain directory"
    ),
    output: str = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output", "-o", help="Path to output directory"
    ),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes and recompile"),
) -> None:
    """Compile MyST domain files into generated Python code."""
    from questfoundry.compiler.compile import compile_domain

    if watch:
        console.print("[yellow]Watch mode not yet implemented[/yellow]")
        return

    domain_path = Path(domain)
    output_path = Path(output)

    if not domain_path.exists():
        console.print(f"[red]Domain directory not found: {domain_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Compiling domain:[/blue] {domain_path}")
    console.print(f"[blue]Output directory:[/blue] {output_path}")

    try:
        result = compile_domain(domain_path, output_path)
        console.print(f"\n[green]Generated {len(result)} files:[/green]")
        for name, path in sorted(result.items()):
            console.print(f"  {name}: {path}")
    except Exception as e:
        console.print(f"[red]Compilation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    domain: str = typer.Option(
        DEFAULT_DOMAIN_DIR, "--domain", "-d", help="Path to domain directory"
    ),
) -> None:
    """Validate domain files without generating code."""
    from questfoundry.compiler.compile import validate_loops
    from questfoundry.compiler.compile import _parse_role_files, _extract_roles

    domain_path = Path(domain)

    if not domain_path.exists():
        console.print(f"[red]Domain directory not found: {domain_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Validating domain:[/blue] {domain_path}")

    try:
        # Parse roles for loop validation
        roles_path = domain_path / "roles"
        roles_by_id = _parse_role_files(roles_path)
        roles = _extract_roles(roles_by_id)

        # Validate loops against roles
        validate_loops(domain_path, roles)
        console.print("[green]Validation passed[/green]")
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


if __name__ == "__main__":
    app()
