"""CLI Main Entry Point - Typer-based command-line interface.

Architecture: SR-orchestrated hub-and-spoke execution.
The Showrunner role coordinates; roles return results via delegation pattern.

Usage
-----
::

    qf ask "Create a mystery story"
    qf ask "Design a story topology" -vv
    qf config show       # Show current configuration
    qf config providers  # Check provider status
    qf doctor            # Check system status
"""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from questfoundry.runtime.cli.config_cmd import config_app
from questfoundry.runtime.config import get_settings

# Help panel names for option categorization
PANEL_LLM = "LLM Options"
PANEL_OUTPUT = "Output & Logging"
PANEL_RUNTIME = "Runtime Options"


class ProviderStatus(str, Enum):
    """Provider availability status."""

    UNAVAILABLE = "unavailable"  # No API key / not configured
    NOT_READY = "not_ready"  # Configured but connectivity failed
    READY = "ready"  # Configured and connected


# Create Typer app
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - SR-orchestrated creative fiction engine",
    add_completion=True,
    rich_markup_mode="rich",
)

# Register config subcommand
app.add_typer(config_app, name="config")

# Create Rich console for formatted output
console = Console()


def _verbose_callback(value: int) -> int:
    """Callback for verbose flag counting."""
    return value


def _run_async(coro: Any) -> Any:
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


@app.command()
def ask(
    message: Annotated[str, typer.Argument(help="Your request in natural language")],
    # LLM Options
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider: google, ollama, openai",
            rich_help_panel=PANEL_LLM,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model name (provider-specific)",
            rich_help_panel=PANEL_LLM,
        ),
    ] = None,
    # Output & Logging
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (-v=INFO, -vv=DEBUG, -vvv=TRACE)",
            rich_help_panel=PANEL_OUTPUT,
        ),
    ] = 0,
    log_dir: Annotated[
        Path | None,
        typer.Option(
            "--log-dir",
            help="Directory for structured JSONL logs",
            rich_help_panel=PANEL_OUTPUT,
        ),
    ] = None,
    # Runtime Options
    max_delegations: Annotated[
        int | None,
        typer.Option(
            "--max-delegations",
            "-d",
            help="Maximum delegations before termination",
            rich_help_panel=PANEL_RUNTIME,
        ),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Project path for cold store (.qfproj file)",
            rich_help_panel=PANEL_RUNTIME,
        ),
    ] = None,
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

    [bold]Examples:[/bold]

        qf ask "Create a mystery story about a haunted lighthouse"

        qf ask "Design a 3-act story structure" -vv

        qf ask "Create a story" --provider google --model gemini-2.5-pro

        qf ask "Create a story" -vvv --log-dir ./logs
    """
    from questfoundry.runtime.logging import setup_logging

    try:
        # Set up logging based on verbosity
        # Default to INFO (1) if no -v flags
        effective_verbosity = verbose if verbose > 0 else 1
        setup_logging(verbosity=effective_verbosity, log_dir=log_dir)

        settings = get_settings()

        # Override settings with CLI arguments
        if provider:
            settings.llm.provider = provider  # type: ignore[assignment]
        if model:
            settings.llm.model = model

        effective_max_delegations = max_delegations or settings.runtime.max_delegations

        if verbose >= 2:
            console.print(
                Panel(
                    "[cyan bold]Debug Mode[/cyan bold]\n\n"
                    f"Verbosity level: {verbose}\n"
                    + (f"Log directory: {log_dir}" if log_dir else "No file logging"),
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

        # Validate provider is ready
        status, message_text = _check_provider_status(effective_provider, settings)
        if status == ProviderStatus.UNAVAILABLE:
            console.print(f"[red]Error: {message_text}[/red]")
            raise typer.Exit(1)
        if status == ProviderStatus.NOT_READY:
            console.print(f"[red]Error: {message_text}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]+ {message_text}[/green]")

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
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose >= 2:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e


def _check_provider_status(
    provider: str, settings: Any
) -> tuple[ProviderStatus, str]:
    """Check provider availability with three-state logic.

    Returns
    -------
    tuple[ProviderStatus, str]
        Status enum and human-readable message.
    """
    if provider == "ollama":
        # Check if Ollama host is explicitly configured
        ollama_host = settings.ollama.host
        env_host = os.environ.get("OLLAMA_HOST") or os.environ.get("QF_OLLAMA__HOST")

        # If using default localhost and no explicit OLLAMA_HOST env var, Ollama is unavailable
        if (
            not env_host
            and ollama_host == "http://localhost:11434"
            and not os.environ.get("OLLAMA_HOST")
        ):
            return (
                ProviderStatus.UNAVAILABLE,
                "Ollama not configured. Set OLLAMA_HOST or QF_OLLAMA__HOST environment variable.",
            )

        # Have config, check connectivity
        from questfoundry.runtime.providers import check_ollama_available

        if check_ollama_available(ollama_host):
            return ProviderStatus.READY, f"Connected to Ollama at {ollama_host}"
        return (
            ProviderStatus.NOT_READY,
            f"Ollama not responding at {ollama_host}. Is it running?",
        )

    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return (
                ProviderStatus.UNAVAILABLE,
                "GOOGLE_API_KEY not set. Export it or add to .env file.",
            )
        # Google doesn't have a cheap connectivity test, assume ready if key present
        return ProviderStatus.READY, "Google AI Studio configured"

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return (
                ProviderStatus.UNAVAILABLE,
                "OPENAI_API_KEY not set. Export it or add to .env file.",
            )
        # OpenAI doesn't have a cheap connectivity test, assume ready if key present
        return ProviderStatus.READY, "OpenAI configured"

    return ProviderStatus.UNAVAILABLE, f"Unknown provider: {provider}"


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
def doctor(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity",
            rich_help_panel=PANEL_OUTPUT,
        ),
    ] = 0,
) -> None:
    """Check system status and configuration.

    Verifies:
    - Provider availability (Ollama, Google, OpenAI)
    - Configuration
    - Role definitions
    - Generated models

    Provider States:
    - [green]ready[/green]: Configured and connected
    - [yellow]not ready[/yellow]: Configured but connectivity failed
    - [dim]unavailable[/dim]: Not configured
    """
    settings = get_settings()

    console.print("[bold]QuestFoundry Doctor[/bold]\n")

    # Show active configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Provider: [cyan]{settings.llm.provider}[/cyan]")
    console.print(f"  Model: [cyan]{settings.get_llm_model()}[/cyan]")
    console.print(f"  Temperature: {settings.llm.temperature}")
    console.print()

    # Provider status table
    table = Table(title="Provider Status")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Check Ollama
    ollama_status, ollama_msg = _check_provider_status("ollama", settings)
    if ollama_status == ProviderStatus.READY:
        status_str = "[green]ready[/green]"
        # Get model count if ready
        from questfoundry.runtime.providers import list_ollama_models

        models = list_ollama_models(settings.ollama.host)
        details = f"{len(models)} models available"
        if verbose >= 1 and models:
            details += f"\n    {', '.join(models[:3])}"
            if len(models) > 3:
                details += f" (+{len(models) - 3} more)"
    elif ollama_status == ProviderStatus.NOT_READY:
        status_str = "[yellow]not ready[/yellow]"
        details = ollama_msg
    else:
        status_str = "[dim]unavailable[/dim]"
        details = ollama_msg if verbose >= 1 else "Not configured"
    table.add_row("Ollama", status_str, details)

    # Check Google
    google_status, google_msg = _check_provider_status("google", settings)
    if google_status == ProviderStatus.READY:
        status_str = "[green]ready[/green]"
        details = f"Model: {settings.google.model}"
    elif google_status == ProviderStatus.NOT_READY:
        status_str = "[yellow]not ready[/yellow]"
        details = google_msg
    else:
        status_str = "[dim]unavailable[/dim]"
        details = google_msg if verbose >= 1 else "GOOGLE_API_KEY not set"
    table.add_row("Google AI", status_str, details)

    # Check OpenAI
    openai_status, openai_msg = _check_provider_status("openai", settings)
    if openai_status == ProviderStatus.READY:
        status_str = "[green]ready[/green]"
        details = f"Model: {settings.openai.model}"
    elif openai_status == ProviderStatus.NOT_READY:
        status_str = "[yellow]not ready[/yellow]"
        details = openai_msg
    else:
        status_str = "[dim]unavailable[/dim]"
        details = openai_msg if verbose >= 1 else "OPENAI_API_KEY not set"
    table.add_row("OpenAI", status_str, details)

    console.print(table)
    console.print()

    # Check roles
    console.print("[bold]Checking roles...[/bold]")
    try:
        from questfoundry.generated.roles import ALL_ROLES

        console.print(f"  [green]+ {len(ALL_ROLES)} roles loaded[/green]")
        if verbose >= 1:
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
        if verbose >= 1:
            for m in models_list:
                console.print(f"    - {m.__name__}")
    except ImportError as e:
        console.print(f"  [red]- Could not load models: {e}[/red]")

    console.print()


@app.command()
def roles(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Show more details",
            rich_help_panel=PANEL_OUTPUT,
        ),
    ] = 0,
) -> None:
    """List available specialist roles."""
    try:
        from questfoundry.generated.roles import ALL_ROLES

        table = Table(title="Available Roles")
        table.add_column("Code", style="cyan")
        table.add_column("Role ID", style="bold")
        table.add_column("Archetype")
        table.add_column("Mandate", max_width=50)
        if verbose >= 1:
            table.add_column("Agency")

        for role_id, role in ALL_ROLES.items():
            row = [role.abbr, role_id, role.archetype, role.mandate]
            if verbose >= 1:
                row.append(str(role.agency.value) if hasattr(role.agency, "value") else str(role.agency))
            table.add_row(*row)

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
