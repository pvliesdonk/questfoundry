"""CLI Main Entry Point - Typer-based command-line interface.

Architecture: SR-orchestrated hub-and-spoke execution.
The Showrunner role coordinates; roles return results via delegation pattern.

Usage
-----
::

    qf ask "Create a mystery story"
    qf ask "Design a story topology" --trace
    qf config show       # Show current configuration
    qf config providers  # Check provider status
    qf doctor            # Check system status
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from questfoundry.runtime.cli.config_cmd import config_app
from questfoundry.runtime.config import get_settings

# Create Typer app
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - SR-orchestrated creative fiction engine",
    add_completion=True,
)

# Register config subcommand
app.add_typer(config_app, name="config")

# Create Rich console for formatted output
console = Console()


def _setup_logging(debug: bool = False) -> None:
    """Configure logging based on settings and debug flag."""
    settings = get_settings()
    level = "DEBUG" if debug else settings.logging.level
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s: %(message)s",
    )


def _run_async(coro: Any) -> Any:
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


@app.command()
def ask(
    message: str = typer.Argument(..., help="Your request in natural language"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show detailed execution trace"),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider: google, ollama, openai"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name (provider-specific)"),
    max_delegations: int | None = typer.Option(
        None, "--max-delegations", "-d", help="Maximum delegations before termination"
    ),
    project: str | None = typer.Option(
        None, "--project", help="Project path for cold store (.qfproj file)"
    ),
) -> None:
    """Talk to the studio in natural language.

    Uses SR-orchestrated hub-and-spoke execution where:
    - Showrunner (SR) receives your request
    - SR delegates to specialist roles (Plotwright, Narrator, etc.)
    - Roles execute and return results
    - SR decides next steps or terminates

    Configuration is loaded from:
    1. CLI arguments (highest priority)
    2. Environment variables (QF_* prefix)
    3. .env file
    4. questfoundry.yaml config file
    5. Built-in defaults

    Examples:
        qf ask "Create a mystery story about a haunted lighthouse"
        qf ask "Design a 3-act story structure" --trace
        qf ask "Create a story" --provider google --model gemini-2.5-pro
        qf ask "Create a story" --provider ollama --model qwen3:8b
    """
    try:
        _setup_logging(debug=trace)
        settings = get_settings()

        # Override settings with CLI arguments
        if provider:
            settings.llm.provider = provider  # type: ignore[assignment]
        if model:
            settings.llm.model = model

        effective_max_delegations = max_delegations or settings.runtime.max_delegations

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
        from questfoundry.runtime.providers import create_llm_from_config

        # Check provider availability
        effective_provider = settings.llm.provider
        effective_model = settings.get_llm_model()

        console.print(f"[dim]Provider: {effective_provider}[/dim]")
        console.print(f"[dim]Model: {effective_model}[/dim]")

        if effective_provider == "ollama":
            from questfoundry.runtime.providers import check_ollama_available

            if not check_ollama_available(settings.ollama.host):
                console.print(f"[red]Error: Ollama not available at {settings.ollama.host}[/red]")
                console.print("[dim]Start Ollama with: ollama serve[/dim]")
                raise typer.Exit(1)
            console.print("[green]+ Connected to Ollama[/green]")

        elif effective_provider == "google":
            from questfoundry.runtime.providers import check_google_available

            if not check_google_available():
                console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
                console.print("[dim]Set with: export GOOGLE_API_KEY='your-key'[/dim]")
                raise typer.Exit(1)
            console.print("[green]+ Google AI Studio configured[/green]")

        elif effective_provider == "openai":
            import os

            if not os.getenv("OPENAI_API_KEY"):
                console.print("[red]Error: OPENAI_API_KEY not set[/red]")
                console.print("[dim]Set with: export OPENAI_API_KEY='your-key'[/dim]")
                raise typer.Exit(1)
            console.print("[green]+ OpenAI configured[/green]")

        # Create LLM from configuration
        llm = create_llm_from_config(settings)

        # Set up cold store if project specified
        cold_store = None
        if project:
            from questfoundry.runtime.cold_store import get_cold_store

            cold_store = get_cold_store(project)
            console.print(f"[green]+ Cold store: {project}[/green]")

        # Create orchestrator
        orchestrator = Orchestrator(
            roles=ALL_ROLES,
            llm=llm,
            max_delegations=effective_max_delegations,
            cold_store=cold_store,
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
    - Provider availability (Ollama, Google, OpenAI)
    - Configuration
    - Role definitions
    - Generated models
    """
    import os

    from questfoundry.runtime.providers import (
        check_google_available,
        check_ollama_available,
        list_ollama_models,
    )

    settings = get_settings()

    console.print("[bold]QuestFoundry Doctor[/bold]\n")

    # Show active configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Provider: [cyan]{settings.llm.provider}[/cyan]")
    console.print(f"  Model: [cyan]{settings.get_llm_model()}[/cyan]")
    console.print(f"  Temperature: {settings.llm.temperature}")
    console.print()

    # Check Ollama
    console.print("[bold]Checking Ollama...[/bold]")
    if check_ollama_available(settings.ollama.host):
        console.print(f"  [green]+ Available at {settings.ollama.host}[/green]")
        models = list_ollama_models(settings.ollama.host)
        if models:
            console.print(f"  [green]+ {len(models)} models available[/green]")
            for m in models[:3]:
                console.print(f"    - {m}")
            if len(models) > 3:
                console.print(f"    [dim]... and {len(models) - 3} more[/dim]")
        else:
            console.print("  [yellow]! No models found[/yellow]")
    else:
        console.print(f"  [red]- Not available at {settings.ollama.host}[/red]")
        console.print("    [dim]Start with: ollama serve[/dim]")

    # Check Google
    console.print()
    console.print("[bold]Checking Google AI Studio...[/bold]")
    if check_google_available():
        console.print("  [green]+ GOOGLE_API_KEY configured[/green]")
        console.print(f"  [green]+ Default model: {settings.google.model}[/green]")
    else:
        console.print("  [yellow]- GOOGLE_API_KEY not set[/yellow]")
        console.print("    [dim]Set with: export GOOGLE_API_KEY='your-key'[/dim]")

    # Check OpenAI
    console.print()
    console.print("[bold]Checking OpenAI...[/bold]")
    if os.getenv("OPENAI_API_KEY"):
        console.print("  [green]+ OPENAI_API_KEY configured[/green]")
        console.print(f"  [green]+ Default model: {settings.openai.model}[/green]")
    else:
        console.print("  [yellow]- OPENAI_API_KEY not set[/yellow]")
        console.print("    [dim]Set with: export OPENAI_API_KEY='your-key'[/dim]")

    # Check roles
    console.print()
    console.print("[bold]Checking roles...[/bold]")
    try:
        from questfoundry.generated.roles import ALL_ROLES

        console.print(f"  [green]+ {len(ALL_ROLES)} roles loaded[/green]")
        for role_id, role in ALL_ROLES.items():
            console.print(f"    - {role.abbr}: {role_id} ({role.archetype})")
    except ImportError as e:
        console.print(f"  [red]- Could not load roles: {e}[/red]")

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

        models_list = [Brief, CanonEntry, HookCard, Scene, GatecheckReport]
        console.print(f"  [green]+ {len(models_list)} artifact models available[/green]")
        for m in models_list:
            console.print(f"    - {m.__name__}")
    except ImportError as e:
        console.print(f"  [red]- Could not load models: {e}[/red]")

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


@app.command()
def version() -> None:
    """Show version information."""
    settings = get_settings()

    console.print("[bold]QuestFoundry Runtime[/bold]")
    console.print("Version: 3.0.0-alpha.1")
    console.print("Architecture: SR-Orchestrated Hub-and-Spoke")
    console.print()
    console.print(f"Provider: {settings.llm.provider}")
    console.print(f"Model: {settings.get_llm_model()}")
    console.print()
    console.print("Usage:")
    console.print('  qf ask "your request in natural language"')
    console.print("  qf config show     # Show configuration")
    console.print("  qf doctor          # Check system status")


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
