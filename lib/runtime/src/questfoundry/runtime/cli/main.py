"""
CLI Main Entry Point - Typer-based command-line interface.

Architecture: Protocol-driven mesh routing via ControlPlane.
The Showrunner role coordinates; routing is envelope-based.
"""

import asyncio
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from questfoundry.runtime.core.control_plane import ControlPlane
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.logging_config import get_logger, setup_logging
from questfoundry.runtime.structured_logging import configure_structured_logging

# Set up Rich logging
setup_logging(level="INFO", show_time=False, show_path=False)
logger = get_logger(__name__)

# Track active trace handlers for cleanup on interrupt
_active_trace_handlers = []


def run_async(coro):
    """Run async coroutine in sync context using asyncio.run()."""
    return asyncio.run(coro)


def cleanup_and_exit(signum=None, frame=None):
    """Gracefully shut down on interrupt (Ctrl-C)."""
    console = Console()
    console.print("\n[yellow]⚠️  Interrupted by user. Cleaning up...[/yellow]")

    # Close any active trace handlers
    for handler in _active_trace_handlers:
        try:
            handler.close()
        except Exception as e:
            logger.warning(f"Error closing trace handler: {e}")

    console.print("[green]✓ Cleanup complete[/green]")
    sys.exit(130)  # Standard exit code for SIGINT


# Register signal handler for Ctrl-C
signal.signal(signal.SIGINT, cleanup_and_exit)

# Create app with shell completion support
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - Protocol-driven mesh for interactive fiction",
    add_completion=True,
)

# Register config subcommand
from questfoundry.runtime.cli.config_cmd import config_app

app.add_typer(config_app, name="config")

# Create console for rich output (force UTF-8 for Unicode support on Windows)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
console = Console()


# ============================================================================
# PRIMARY INTERFACE: Natural Language → Mesh Execution
# ============================================================================


@app.command()
def ask(
    message: str = typer.Argument(..., help="Your request in natural language"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution info"),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Enable trace mode to show agent communication on screen"
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Write trace and debug logs to files (<name>-trace.log, <name>-debug.log)",
    ),
    recursion_limit: int = typer.Option(50, "--recursion-limit", "-r", help="Max graph iterations"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Preferred provider (e.g., 'anthropic:claude-3-haiku-20240307')",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Project identifier for storage (default: $QF_PROJECT_ID or 'default')",
    ),
    project_dir: str | None = typer.Option(
        None,
        "--project-dir",
        help="Base directory for projects (default: $QF_PROJECT_DIR or ~/.questfoundry/projects)",
    ),
    structured_logs: Path | None = typer.Option(
        None,
        "--structured-logs",
        help="Directory for JSONL logs (tools, state, bus, prompts)",
    ),
):
    """
    Talk to the studio in natural language.

    Uses protocol-driven mesh routing where:
    - Routing is determined by message envelope `receiver` field
    - Showrunner coordinates but doesn't bottleneck all communication
    - Roles communicate peer-to-peer via protocol

    Logging modes:
        A) Default: No trace, -v controls screen verbosity
        B) --trace: Show agent communication on screen (with full prompts/responses)
        C) --log-file <name>: Write to <name>-trace.log and <name>-debug.log, no screen trace

    Examples:
        qf ask "Create a mystery story about a haunted lighthouse"
        qf ask "Review the draft and suggest improvements" -v
        qf ask "Create a story" --trace
        qf ask "Create a story" --log-file session1
        qf ask "Create a story" --log-file session1 --trace  # Both file and screen
        qf ask "Create a story" --provider anthropic:claude-3-haiku-20240307
        qf ask "Create a story" --structured-logs ./logs  # Enable structured JSON logging
    """
    try:
        # Set up structured logging if --structured-logs specified
        if structured_logs:
            structured_logs.mkdir(parents=True, exist_ok=True)
            configure_structured_logging(structured_logs)
            logger.info(f"Structured logging configured in: {structured_logs}")

        # Set up file logging if --log-file specified
        file_handler = None
        if log_file:
            from questfoundry.runtime.logging_config import setup_file_logging

            debug_log_path = f"{log_file}-debug.log"
            file_handler = setup_file_logging(debug_log_path, level="DEBUG")
            logger.info(f"Debug logging to: {debug_log_path}")

        # Create trace handler if --trace or --log-file enabled
        trace_handler = None
        if trace or log_file:
            from questfoundry.runtime.core.trace_handler import TraceHandler

            # Determine trace file path
            trace_file_path = Path(f"{log_file}-trace.log") if log_file else None

            # quiet_console: True if only --log-file (no --trace), False if --trace specified
            quiet_console = log_file is not None and not trace

            trace_handler = TraceHandler(
                output_file=trace_file_path,
                console=console,
                verbose=True,
                quiet_console=quiet_console,
            )

            # Register for cleanup on interrupt
            _active_trace_handlers.append(trace_handler)

            # Show appropriate message based on mode
            if trace and log_file:
                # Mode: Both screen and file
                console.print(
                    Panel(
                        "[cyan bold]📡 Trace Mode (Screen + File)[/cyan bold]\n\n"
                        "Agent-to-agent communication displayed on screen.\n"
                        f"Trace log: {log_file}-trace.log\n"
                        f"Debug log: {log_file}-debug.log",
                        style="cyan",
                        border_style="cyan",
                        title="Trace Mode",
                    )
                )
            elif trace:
                # Mode B: Screen only
                console.print(
                    Panel(
                        "[cyan bold]📡 Trace Mode (Screen)[/cyan bold]\n\n"
                        "Agent-to-agent communication will be captured and displayed.\n"
                        "LLM prompts, outputs, and tool calls will be shown.",
                        style="cyan",
                        border_style="cyan",
                        title="Trace Mode",
                    )
                )
            else:
                # Mode C: File only (--log-file without --trace)
                console.print(
                    Panel(
                        "[cyan bold]📁 File Logging Enabled[/cyan bold]\n\n"
                        f"Trace log: {log_file}-trace.log\n"
                        f"Debug log: {log_file}-debug.log\n\n"
                        "[dim]Use --trace to also show on screen[/dim]",
                        style="cyan",
                        border_style="cyan",
                        title="File Logging",
                    )
                )
            console.print()

        # Display customer message
        console.print(
            Panel(f"[bold]Your Request:[/bold]\n{message}", style="blue", border_style="blue")
        )

        # Create ControlPlane with mesh routing and trace handler
        state_manager = StateManager(
            project_id=project,
            project_root=project_dir,
            trace_handler=trace_handler,
        )
        control_plane = ControlPlane(
            state_manager=state_manager,
            preferred_provider=provider,
        )

        logger.info(f"Starting mesh execution with recursion_limit={recursion_limit}")

        # Run the mesh using asyncio.run() to execute async operation
        final_state = asyncio.run(
            control_plane.run(
                human_input=message,
                recursion_limit=recursion_limit,
            )
        )

        # Close trace handler if created
        if trace_handler:
            trace_handler.close()
            _active_trace_handlers.remove(trace_handler)

        # Close file handler if created
        if file_handler:
            import logging

            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            logger.info(f"Debug log complete: {log_file}-debug.log")

        # Display result
        messages = final_state.get("messages", [])
        artifacts = final_state.get("artifacts", {})
        hot_sot = final_state.get("hot_sot", {})

        console.print(
            Panel(
                f"[bold green]✓ Complete[/bold green]\n\n"
                f"[bold]TU ID:[/bold] {final_state.get('tu_id', 'unknown')}\n"
                f"[bold]Messages:[/bold] {len(messages)}\n"
                f"[bold]Artifacts:[/bold] {len(artifacts)}\n"
                f"[bold]Hot SoT keys:[/bold] {len(hot_sot)}",
                style="green",
                border_style="green",
                title="Execution Complete",
            )
        )

        if verbose:
            console.print("\n[bold]Message Flow:[/bold]")
            for i, msg in enumerate(messages[-10:], 1):  # Last 10 messages
                sender = msg.get("sender", "?")
                receiver = msg.get("receiver", "?")
                intent = msg.get("intent", "?")
                console.print(f"  {i}. {sender} → {receiver}: {intent}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# UTILITY COMMANDS: Discovery
# ============================================================================


@app.command()
def list_roles():
    """List all available roles in the mesh."""
    try:
        console.print("[bold]Available Roles:[/bold]\n")

        control_plane = ControlPlane()
        roles = control_plane.get_available_roles()

        registry = SchemaRegistry()
        for role_id in roles:
            try:
                role = registry.load_role(role_id)
                dormancy = "always-on" if role_id in ["showrunner", "gatekeeper"] else "default-on"
                console.print(f"  • [cyan]{role_id}[/cyan]: {role.name} ({dormancy})")
            except Exception:
                console.print(f"  • [dim]{role_id}[/dim]: (not loaded)")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def list_playbooks():
    """List available playbooks (workflow knowledge for Showrunner)."""
    try:
        console.print("[bold]Available Playbooks:[/bold]\n")
        console.print(
            "[dim]These are knowledge resources for the Showrunner, not direct commands.[/dim]\n"
        )

        playbooks = [
            ("story_spark", "Initial story creation and brainstorming"),
            ("hook_harvest", "Collect and triage narrative hooks"),
            ("lore_deepening", "Expand world lore and backstory"),
            ("codex_expansion", "Build encyclopedia entries"),
            ("style_tune_up", "Refine narrative voice and style"),
            ("art_touch_up", "Visual asset refinement"),
            ("audio_pass", "Audio/music production"),
            ("translation_pass", "Localization workflow"),
            ("binding_run", "Export and publishing preparation"),
            ("narration_dry_run", "Test narration generation"),
        ]

        for playbook_id, description in playbooks:
            console.print(f"  • [cyan]{playbook_id}[/cyan]: {description}")

        console.print("\n[dim]The Showrunner consults these via consult_playbook tool.[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def test_schema(
    definition_type: str = typer.Argument("role", help="Type: role or loop"),
    definition_id: str = typer.Argument(..., help="Definition ID (e.g., plotwright)"),
):
    """Test loading and validating a definition."""
    try:
        console.print(f"[bold]Testing {definition_type}: {definition_id}[/bold]\n")

        registry = SchemaRegistry()
        if definition_type == "role":
            role = registry.load_role(definition_id)
            console.print(f"[green]✓ Loaded role: {role.name}[/green]")
            console.print(f"  ID: {role.id}")
            console.print(f"  Type: {role.role_type}")
            console.print(f"  Model Tier: {role.get_model_tier()}")
            console.print(f"  Temperature: {role.get_temperature()}")

        elif definition_type == "loop":
            loop = registry.load_loop(definition_id)
            console.print(f"[green]✓ Loaded loop: {loop.name}[/green]")
            console.print(f"  ID: {loop.id}")
            console.print(f"  Description: {loop.description}")
            console.print(
                "  [dim]Note: Loop topology is informational; routing is envelope-based.[/dim]"
            )

        else:
            console.print(f"[red]Unknown type: {definition_type}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def download_spec(
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Specific spec version to download (e.g., spec-v1.0.0)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if already cached"
    ),
    show_path: bool = typer.Option(
        False, "--show-path", "-p", help="Show the downloaded spec path"
    ),
):
    """
    Download the latest QuestFoundry spec from GitHub releases.

    Examples:
        qf download-spec                    # Download latest spec release
        qf download-spec --tag spec-v1.0.0  # Download specific version
        qf download-spec --force            # Force re-download
    """
    try:
        from questfoundry.runtime.core.spec_fetcher import (
            SpecFetchError,
            download_latest_release_spec,
        )

        version_info = f" ({tag})" if tag else " (latest)"
        console.print(f"\n[bold]Downloading spec{version_info}...[/bold]\n")

        spec_path = download_latest_release_spec(tag=tag, force=force)

        metadata_file = spec_path / ".questfoundry-spec.json"
        version_str = tag or "latest"
        if metadata_file.exists():
            import json

            metadata = json.loads(metadata_file.read_text())
            version_str = metadata.get("tag", version_str)

        console.print(
            Panel(
                f"[bold green]✓ Spec downloaded successfully[/bold green]\n\n"
                f"[bold]Version:[/bold] {version_str}\n"
                f"[bold]Location:[/bold] {spec_path if show_path else '~/.cache/questfoundry/spec/'}",
                style="green",
                border_style="green",
                title="Download Complete",
            )
        )

        console.print("\n[dim]To use: set QF_SPEC_SOURCE=download[/dim]")

    except SpecFetchError as e:
        console.print(f"[red]Failed to download spec: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def doctor(
    show_config: bool = typer.Option(
        False, "--show-config", "-c", help="Show all configuration values"
    ),
    skip_network: bool = typer.Option(
        False, "--skip-network", "-s", help="Skip network connectivity checks"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """
    Check configuration and provider connectivity.

    Validates that providers are properly configured and reachable.
    Shows current configuration and detects common issues.

    Provider statuses:
        + Ready:       Configured and responding
        x Unavailable: Configured but connection failed
        - Unconfigured: No API key or URL set

    Examples:
        qf doctor                # Full check with network tests
        qf doctor --show-config  # Also display all config values
        qf doctor --skip-network # Skip API connectivity tests
        qf doctor --json         # Machine-readable output
    """
    from questfoundry.runtime.cli.doctor import run_doctor

    exit_code = run_doctor(
        console=console,
        show_config=show_config,
        skip_network=skip_network,
        output_json=json_output,
    )
    if exit_code != 0:
        sys.exit(exit_code)


@app.command()
def version():
    """Show version information."""
    import os

    from questfoundry.runtime.core.spec_fetcher import get_spec_source_preference

    console.print("[bold]QuestFoundry Runtime[/bold]")
    console.print("Version: 0.2.0")
    console.print("Architecture: Protocol-Driven Mesh Routing")
    console.print("\nUsage:")
    console.print('  qf ask "your request in natural language"')

    spec_source = get_spec_source_preference()
    console.print(f"\nSpec Source: {spec_source}")
    if spec_source != "auto":
        console.print(f"  [dim](Set via QF_SPEC_SOURCE={os.getenv('QF_SPEC_SOURCE')})[/dim]")


@app.callback()
def main(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v: INFO, -vv: DEBUG, -vvv: full details)",
    ),
):
    """
    QuestFoundry Runtime - Protocol-driven mesh for interactive fiction.

    Primary usage: qf ask "<your request in natural language>"

    Verbosity levels:
        (none): WARNING - Only show warnings and errors
        -v:     INFO - Show general progress information
        -vv:    DEBUG - Show detailed debugging info
        -vvv:   DEBUG + full details - Show everything including library logs
    """
    import logging

    from questfoundry.runtime.logging_config import setup_logging

    if verbose == 0:
        level = "WARNING"
        show_path = False
    elif verbose == 1:
        level = "INFO"
        show_path = False
    elif verbose == 2:
        level = "DEBUG"
        show_path = True
    else:
        level = "DEBUG"
        show_path = True
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("anthropic").setLevel(logging.INFO)

    setup_logging(level=level, show_time=(verbose >= 2), show_path=show_path)


if __name__ == "__main__":
    app()
