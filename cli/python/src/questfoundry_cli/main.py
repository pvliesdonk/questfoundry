"""Main CLI application for QuestFoundry."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from questfoundry.logging_config import setup_logging
from questfoundry.orchestrator import Orchestrator
from questfoundry.providers.config import ProviderConfig
from questfoundry.providers.registry import ProviderRegistry
from questfoundry.state.workspace import WorkspaceManager
from rich.console import Console

from . import __version__
from .config import get_workspace_path, load_env_vars

console = Console()

# Create main app
app = typer.Typer(
    name="qf",
    help="QuestFoundry CLI - Control panel for the agentic runtime",
    no_args_is_help=True,
)

# Global state
_log_level: str = "INFO"
_log_file: Optional[str] = None


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Set log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable verbose output (DEBUG level)"),
    ] = False,
    log_file: Annotated[
        Optional[str],
        typer.Option("--log-file", help="Log to file instead of stderr"),
    ] = None,
) -> None:
    """QuestFoundry CLI - Control panel for the agentic runtime."""
    global _log_level, _log_file

    if version:
        console.print(f"qf version {__version__}")
        raise typer.Exit()

    # Determine log level
    if verbose:
        _log_level = "DEBUG"
    else:
        _log_level = log_level.upper()

    _log_file = log_file

    # Setup logging (will be called again in subcommands with handlers if needed)
    setup_logging(level=_log_level)  # type: ignore


def get_orchestrator(workspace_path: Path | None = None) -> Orchestrator:
    """
    Get a configured Orchestrator instance.

    This helper function:
    1. Loads configuration from ~/.questfoundry/config.toml
    2. Loads API keys from .env files
    3. Initializes ProviderConfig, WorkspaceManager, etc.
    4. Returns a fully configured Orchestrator

    Args:
        workspace_path: Optional workspace path (uses config if not provided)

    Returns:
        Configured Orchestrator instance

    Raises:
        typer.Exit: If configuration is invalid or missing
    """
    import logging
    import sys

    # Setup logging with file handler if specified
    if _log_file:
        file_handler = logging.FileHandler(_log_file)
        setup_logging(level=_log_level, handlers=[file_handler])  # type: ignore
    else:
        setup_logging(level=_log_level)  # type: ignore

    # Load environment variables
    env_vars = load_env_vars()

    # Determine workspace path
    if workspace_path is None:
        workspace_path = get_workspace_path()

    if workspace_path is None:
        workspace_path = Path.cwd()
        console.print(
            f"[yellow]No workspace configured, using current directory: {workspace_path}[/yellow]"
        )

    # Initialize workspace
    try:
        workspace = WorkspaceManager(workspace_path)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize workspace: {e}[/red]")
        raise typer.Exit(1)

    # Initialize provider config
    try:
        provider_config = ProviderConfig()
        provider_registry = ProviderRegistry(config=provider_config)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize providers: {e}[/red]")
        raise typer.Exit(1)

    # Initialize orchestrator
    try:
        orchestrator = Orchestrator(
            workspace=workspace,
            provider_registry=provider_registry,
        )

        # Initialize with default provider
        # Try to get a text provider (OpenAI by default)
        try:
            provider = provider_registry.get_text_provider("openai")
            orchestrator.initialize(provider=provider, provider_name="openai")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not initialize OpenAI provider: {e}[/yellow]"
            )
            console.print(
                "[yellow]Some commands may not work without a configured provider[/yellow]"
            )

        return orchestrator

    except Exception as e:
        console.print(f"[red]Error: Failed to initialize orchestrator: {e}[/red]")
        raise typer.Exit(1)


# Import and register subcommands
from .config_commands import app as config_app
from .run_commands import app as run_app

app.add_typer(run_app, name="run")
app.add_typer(config_app, name="config")

if __name__ == "__main__":
    app()
