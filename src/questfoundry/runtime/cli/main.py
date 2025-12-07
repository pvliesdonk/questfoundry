"""CLI Main Entry Point - Typer-based command-line interface.

Architecture: SR-orchestrated hub-and-spoke execution.
The Showrunner role coordinates; roles return results via delegation pattern.

Usage
-----
    qf ask "Create a mystery story"
    qf ask "Design a story topology" --trace
    qf doctor  # Check system status
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Create Typer app
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - SR-orchestrated creative fiction engine",
    add_completion=True,
)

# Create Rich console for formatted output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


@app.command()
def ask(
    message: str = typer.Argument(..., help="Your request in natural language"),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Show detailed execution trace"
    ),
    model: str = typer.Option(
        "qwen3:8b", "--model", "-m", help="Ollama model to use"
    ),
    base_url: str = typer.Option(
        "http://localhost:11434", "--base-url", "-b", help="Ollama server URL"
    ),
    max_delegations: int = typer.Option(
        20, "--max-delegations", "-d", help="Maximum delegations before termination"
    ),
) -> None:
    """Talk to the studio in natural language.

    Uses SR-orchestrated hub-and-spoke execution where:
    - Showrunner (SR) receives your request
    - SR delegates to specialist roles (Plotwright, Narrator, etc.)
    - Roles execute and return results
    - SR decides next steps or terminates

    Examples:
        qf ask "Create a mystery story about a haunted lighthouse"
        qf ask "Design a 3-act story structure" --trace
        qf ask "Create a story" --model llama3.1:8b
    """
    try:
        # Set logging level based on trace flag
        if trace:
            logging.getLogger("questfoundry").setLevel(logging.DEBUG)
            console.print(
                Panel(
                    "[cyan bold]Trace Mode[/cyan bold]\n\n"
                    "Detailed execution trace will be displayed.",
                    style="cyan",
                    border_style="cyan",
                )
            )
            console.print()

        # Display request
        console.print(
            Panel(
                f"[bold]Your Request:[/bold]\n{message}",
                style="blue",
                border_style="blue",
            )
        )
        console.print()

        # Import runtime components
        from questfoundry.generated.roles import ALL_ROLES
        from questfoundry.runtime.orchestrator import Orchestrator
        from questfoundry.runtime.providers.ollama import create_ollama_llm

        # Check Ollama availability
        console.print("[dim]Checking Ollama connection...[/dim]")
        from questfoundry.runtime.providers.ollama import check_ollama_available

        if not check_ollama_available(base_url):
            console.print(f"[red]Error: Ollama not available at {base_url}[/red]")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Connected to Ollama at {base_url}[/green]")

        # Create LLM and orchestrator
        console.print(f"[dim]Using model: {model}[/dim]")
        llm = create_ollama_llm(model=model, base_url=base_url)
        orchestrator = Orchestrator(
            roles=ALL_ROLES,
            llm=llm,
            max_delegations=max_delegations,
        )

        # Run the workflow
        console.print()
        console.print("[bold]Running workflow...[/bold]")
        console.print()

        final_state = _run_async(orchestrator.run(message))

        # Display results
        _display_results(final_state)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if trace:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e


def _display_results(state: dict[str, Any]) -> None:
    """Display workflow results in a nice format."""
    metadata = state.get("metadata", {})

    # Check for errors
    if "error" in metadata:
        console.print(
            Panel(
                f"[red bold]Error[/red bold]\n\n{metadata['error']}",
                style="red",
                border_style="red",
            )
        )
        return

    # Display termination info
    termination = metadata.get("termination", {})
    if termination:
        console.print(
            Panel(
                f"[green bold]Workflow Complete[/green bold]\n\n"
                f"[bold]Reason:[/bold] {termination.get('reason', 'Unknown')}\n"
                f"[bold]Summary:[/bold] {termination.get('summary', 'No summary')}",
                style="green",
                border_style="green",
            )
        )

    # Display delegation history
    history = metadata.get("delegation_history", [])
    if history:
        console.print()
        table = Table(title="Delegation History")
        table.add_column("#", style="dim")
        table.add_column("Role", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Task", style="white")

        for i, delegation in enumerate(history, 1):
            result = delegation.get("result", {})
            task = delegation.get("task", "")[:50]
            if len(delegation.get("task", "")) > 50:
                task += "..."
            table.add_row(
                str(i),
                delegation.get("role", "?"),
                result.get("status", "?"),
                task,
            )

        console.print(table)

    # Display artifacts in hot_store
    hot_store = state.get("hot_store", {})
    if hot_store:
        console.print()
        console.print("[bold]Artifacts Created:[/bold]")
        for key, value in hot_store.items():
            if isinstance(value, list):
                console.print(f"  [cyan]{key}[/cyan]: {len(value)} items")
            elif isinstance(value, dict):
                console.print(f"  [cyan]{key}[/cyan]: dict with {len(value)} keys")
            else:
                console.print(f"  [cyan]{key}[/cyan]: {type(value).__name__}")

    # Summary
    total = metadata.get("total_delegations", len(history))
    console.print()
    console.print(f"[dim]Total delegations: {total}[/dim]")


@app.command()
def doctor() -> None:
    """Check system status and configuration.

    Verifies:
    - Ollama availability
    - Available models
    - Role definitions
    - Generated models
    """
    console.print("[bold]QuestFoundry Doctor[/bold]\n")

    # Check Ollama
    console.print("[bold]Checking Ollama...[/bold]")
    try:
        from questfoundry.runtime.providers.ollama import (
            DEFAULT_BASE_URL,
            check_ollama_available,
            list_ollama_models,
        )

        if check_ollama_available(DEFAULT_BASE_URL):
            console.print(f"  [green]✓ Ollama available at {DEFAULT_BASE_URL}[/green]")
            models = list_ollama_models(DEFAULT_BASE_URL)
            if models:
                console.print(f"  [green]✓ {len(models)} models available[/green]")
                for model in models[:5]:
                    console.print(f"    - {model}")
                if len(models) > 5:
                    console.print(f"    [dim]... and {len(models) - 5} more[/dim]")
            else:
                console.print("  [yellow]⚠ No models found[/yellow]")
        else:
            console.print(f"  [red]✗ Ollama not available at {DEFAULT_BASE_URL}[/red]")
            console.print("    [dim]Start with: ollama serve[/dim]")
    except Exception as e:
        console.print(f"  [red]✗ Error checking Ollama: {e}[/red]")

    # Check roles
    console.print()
    console.print("[bold]Checking roles...[/bold]")
    try:
        from questfoundry.generated.roles import ALL_ROLES

        console.print(f"  [green]✓ {len(ALL_ROLES)} roles loaded[/green]")
        for role_id, role in ALL_ROLES.items():
            console.print(f"    - {role.abbr}: {role_id} ({role.archetype})")
    except ImportError as e:
        console.print(f"  [red]✗ Could not load roles: {e}[/red]")

    # Check generated models
    console.print()
    console.print("[bold]Checking generated models...[/bold]")
    try:
        from questfoundry.generated.models import (
            Brief,
            CanonEntry,
            GatecheckReport,
            HookCard,
            Scene,
        )

        models = [Brief, CanonEntry, HookCard, Scene, GatecheckReport]
        console.print(f"  [green]✓ {len(models)} artifact models available[/green]")
        for model in models:
            console.print(f"    - {model.__name__}")
    except ImportError as e:
        console.print(f"  [red]✗ Could not load models: {e}[/red]")

    console.print()


@app.command()
def roles() -> None:
    """List available specialist roles."""
    try:
        from questfoundry.generated.roles import ALL_ROLES

        table = Table(title="Available Roles")
        table.add_column("Code", style="cyan")
        table.add_column("Role ID", style="bold")
        table.add_column("Archetype")
        table.add_column("Mandate", max_width=50)

        for role_id, role in ALL_ROLES.items():
            table.add_row(role.abbr, role_id, role.archetype, role.mandate)

        console.print(table)
    except ImportError as e:
        console.print(f"[red]Error loading roles: {e}[/red]")
        raise typer.Exit(1) from e


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
