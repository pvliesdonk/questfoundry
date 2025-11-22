"""
CLI Main Entry Point - Typer-based command-line interface.

Based on spec: components/cli.md v2.0.0
Architecture: Natural language primary, debug mode secondary.
"""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from questfoundry.runtime.cli.showrunner import ShowrunnerInterface
from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.logging_config import setup_logging, get_logger

# Set up Rich logging
setup_logging(level="INFO", show_time=False, show_path=False)
logger = get_logger(__name__)

# Create app
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - Natural language interface to studio production",
    add_completion=False
)

# Create console for rich output (force UTF-8 for Unicode support on Windows)
if sys.platform == "win32":
    import io
    import codecs
    # Reconfigure stdout to use UTF-8 encoding without closing the underlying buffer
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
console = Console()

# Initialize components (lazy-loaded)
_graph_factory: Optional[GraphFactory] = None
_state_manager: Optional[StateManager] = None
_showrunner: Optional[ShowrunnerInterface] = None


def get_graph_factory() -> GraphFactory:
    """Lazy-load graph factory."""
    global _graph_factory
    if _graph_factory is None:
        _graph_factory = GraphFactory()
    return _graph_factory


def get_state_manager(trace_handler=None) -> StateManager:
    """
    Get or create state manager.

    Args:
        trace_handler: Optional trace handler to pass to new StateManager

    Note: If state_manager already exists and trace_handler is provided,
    creates a new instance with the trace handler.
    """
    global _state_manager

    # If trace handler is provided, always create new instance
    # (don't reuse cached one without trace handler)
    if trace_handler is not None:
        return StateManager(trace_handler=trace_handler)

    # Otherwise use cached instance
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def get_showrunner() -> ShowrunnerInterface:
    """Lazy-load Showrunner interface."""
    global _showrunner
    if _showrunner is None:
        _showrunner = ShowrunnerInterface(
            graph_factory=get_graph_factory(),
            state_manager=get_state_manager()
        )
    return _showrunner


# ============================================================================
# PRIMARY INTERFACE: Natural Language → Showrunner Role
# ============================================================================

@app.command()
def ask(
    message: str = typer.Argument(..., help="Your request in natural language"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution info"),
    trace: bool = typer.Option(False, "--trace", help="Enable trace mode to capture agent communication"),
    trace_file: Optional[str] = typer.Option(None, "--trace-file", help="Write trace to file (requires --trace)")
):
    """
    Primary interface: Talk to the Showrunner in natural language.

    The Showrunner interprets your request and decides which internal studio
    operations to run. You don't need to know about loops or roles.

    Examples:
        qf ask "Can you create a mystery story about a space station?"
        qf ask "I like the scar subplot, can you work that into the narrative?"
        qf ask "This character feels flat, can you give them more depth?"
        qf ask "Review the story and harvest any interesting hooks"
        qf ask "Create a story" --trace
        qf ask "Create a story" --trace --trace-file messages.log
    """
    try:
        # Create trace handler if --trace enabled
        trace_handler = None
        if trace:
            from pathlib import Path
            from questfoundry.runtime.core.trace_handler import TraceHandler

            # Create output file path if trace_file specified
            output_file = Path(trace_file) if trace_file else None

            trace_handler = TraceHandler(
                output_file=output_file,
                console=console,
                verbose=verbose
            )

            console.print(Panel(
                "[cyan bold]📡 Trace Mode Enabled[/cyan bold]\n\n"
                "Agent-to-agent communication will be captured and displayed.\n"
                f"{'Output: Console and file' if output_file else 'Output: Console only'}",
                style="cyan",
                border_style="cyan",
                title="Trace Mode"
            ))
            console.print()

        # Display customer message
        console.print(Panel(
            f"[bold]Your Request:[/bold]\n{message}",
            style="blue",
            border_style="blue"
        ))

        # Get Showrunner with trace handler
        if trace_handler:
            # Create custom Showrunner with trace-enabled state manager
            state_manager = get_state_manager(trace_handler=trace_handler)
            showrunner = ShowrunnerInterface(
                graph_factory=get_graph_factory(),
                state_manager=state_manager
            )
        else:
            showrunner = get_showrunner()

        result = showrunner.interpret_and_execute(message, verbose=verbose)

        # Close trace handler if created
        if trace_handler:
            trace_handler.close()

        # Display result in plain language (no jargon)
        if result.success:
            console.print(Panel(
                result.plain_language_response,
                style="green",
                title="✓ Complete",
                border_style="green"
            ))

            # Show next steps if available
            if result.suggested_next_steps:
                console.print("\n[bold]You might want to:[/bold]")
                for step in result.suggested_next_steps:
                    console.print(f"  • {step}")

        else:
            console.print(Panel(
                result.plain_language_response,
                style="red",
                title="✗ Issue",
                border_style="red"
            ))
            if result.error:
                logger.error(f"Execution error: {result.error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Request failed: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# SECONDARY INTERFACE: Debug Mode (Direct Loop Invocation)
# ============================================================================

@app.command()
def loop(
    loop_id: str = typer.Argument(..., help="Loop ID (e.g., story_spark, hook_harvest)"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Context as key=value pairs"),
    mode: str = typer.Option("workshop", "--mode", "-m", help="Execution mode (workshop or production)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution info"),
    trace: bool = typer.Option(False, "--trace", help="Enable trace mode to capture agent communication"),
    trace_file: Optional[str] = typer.Option(None, "--trace-file", help="Write trace to file (requires --trace)")
):
    """
    Debug/audit mode: Directly invoke a loop (bypasses Showrunner).

    ⚠️  WARNING: This is for debugging, testing, and auditing only.
    You are bypassing the Showrunner's mandate and directly controlling
    internal studio operations. For normal use, prefer: qf ask "..."

    Examples:
        qf loop story_spark --context "scene_text=tense cargo bay scene"
        qf loop hook_harvest --mode workshop --verbose
        qf loop gatecheck --verbose --trace
        qf loop story_spark --trace --trace-file messages.log
    """
    try:
        # Display warning banner
        console.print(Panel(
            "[yellow bold]⚠️  Debug Mode: Bypassing Showrunner Mandate[/yellow bold]\n\n"
            "You are directly invoking internal studio operations.\n"
            "This is useful for debugging, testing, and auditing, but should not be\n"
            "your primary workflow.\n\n"
            "[dim]For normal use, prefer: qf ask \"...\"[/dim]",
            style="yellow",
            border_style="yellow",
            title="Debug Mode"
        ))

        # Parse context string into dict
        context_dict = {}
        if context:
            for pair in context.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    context_dict[key.strip()] = value.strip()

        # Add mode to context
        context_dict["mode"] = mode

        # Display execution info
        console.print(f"\n[bold]Loop:[/bold] {loop_id}")
        console.print(f"[bold]Context:[/bold] {context_dict}")
        if trace:
            trace_target = trace_file if trace_file else "console"
            console.print(f"[bold]Trace:[/bold] enabled → {trace_target}")
        console.print()

        # Create trace handler if --trace enabled
        trace_handler = None
        if trace:
            from pathlib import Path
            from questfoundry.runtime.core.trace_handler import TraceHandler

            # Create output file path if trace_file specified
            output_file = Path(trace_file) if trace_file else None

            trace_handler = TraceHandler(
                output_file=output_file,
                console=console,
                verbose=verbose
            )

            console.print(Panel(
                "[cyan bold]📡 Trace Mode Enabled[/cyan bold]\n\n"
                "Agent-to-agent communication will be captured and displayed.\n"
                f"{'Output: Console and file' if output_file else 'Output: Console only'}",
                style="cyan",
                border_style="cyan",
                title="Trace Mode"
            ))
            console.print()

        # Initialize state with trace handler
        state_manager = get_state_manager(trace_handler=trace_handler)
        state = state_manager.initialize_state(loop_id, context_dict)

        # Create graph factory with state_manager (for tracing)
        if trace_handler:
            # Create new graph factory with trace-enabled state_manager
            graph_factory = GraphFactory(state_manager=state_manager)
        else:
            graph_factory = get_graph_factory()

        graph = graph_factory.create_loop_graph(loop_id)

        logger.info(f"Invoking loop: {loop_id}")
        final_state = graph.invoke(state)

        # Close trace handler if created
        if trace_handler:
            trace_handler.close()

        # Display results (technical format for debug mode)
        console.print(Panel(
            f"[bold green]✓ Loop completed[/bold green]\n\n"
            f"[bold]TU ID:[/bold] {final_state['tu_id']}\n"
            f"[bold]Lifecycle:[/bold] {final_state['tu_lifecycle']}\n"
            f"[bold]Artifacts:[/bold] {len(final_state.get('artifacts', {}))}\n"
            f"[bold]Quality Bars:[/bold] {len(final_state.get('quality_bars', {}))}",
            style="green",
            border_style="green",
            title="Execution Complete"
        ))

        # Show verbose info if requested
        if verbose:
            console.print("\n[bold]Artifacts:[/bold]")
            for artifact_key in final_state.get("artifacts", {}).keys():
                console.print(f"  • {artifact_key}")

            console.print("\n[bold]Quality Bars:[/bold]")
            for bar_name, bar_data in final_state.get("quality_bars", {}).items():
                status = bar_data.get("status", "unknown")
                console.print(f"  • {bar_name}: {status}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Loop execution failed: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# UTILITY COMMANDS: Discovery and Testing
# ============================================================================

@app.command()
def list_loops():
    """List all available loops (for debug mode)."""
    try:
        console.print("[bold]Available Loops:[/bold]\n")

        loops = [
            "story_spark",
            "hook_harvest",
            "lore_deepening",
            "codex_expansion",
            "audio_pass",
            "narration_dry_run",
            "style_tune_up",
            "binding_run",
            "art_touch_up",
            "translation_pass"
        ]

        graph_factory = get_graph_factory()
        for loop_id in loops:
            try:
                loop = graph_factory.schema_registry.load_loop(loop_id)
                console.print(f"  • [cyan]{loop_id}[/cyan]: {loop.description}")
            except Exception:
                console.print(f"  • [red]{loop_id}[/red]: (failed to load)")

        console.print("\n[dim]Use in debug mode: qf loop <loop_id>[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def list_roles():
    """List all available roles."""
    try:
        console.print("[bold]Available Roles:[/bold]\n")

        roles = [
            "plotwright", "scene_smith", "gatekeeper", "style_lead",
            "lore_weaver", "codex_curator", "audio_producer", "audio_director",
            "art_director", "illustrator", "player_narrator", "researcher",
            "translator", "book_binder", "export_service", "showrunner"
        ]

        graph_factory = get_graph_factory()
        for role_id in roles:
            try:
                role = graph_factory.schema_registry.load_role(role_id)
                console.print(f"  • [cyan]{role_id}[/cyan]: {role.name}")
            except Exception:
                console.print(f"  • [red]{role_id}[/red]: (failed to load)")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def test_schema(
    definition_type: str = typer.Argument("role", help="Type: role or loop"),
    definition_id: str = typer.Argument(..., help="Definition ID (e.g., plotwright)")
):
    """Test loading and validating a definition."""
    try:
        console.print(f"[bold]Testing {definition_type}: {definition_id}[/bold]\n")

        graph_factory = get_graph_factory()
        if definition_type == "role":
            role = graph_factory.schema_registry.load_role(definition_id)
            console.print(f"[green]✓ Loaded role: {role.name}[/green]")
            console.print(f"  ID: {role.id}")
            console.print(f"  Type: {role.role_type}")
            console.print(f"  Model: {role.get_model()}")
            console.print(f"  Temperature: {role.get_temperature()}")
            console.print(f"  Max Tokens: {role.get_max_tokens()}")

        elif definition_type == "loop":
            loop = graph_factory.schema_registry.load_loop(definition_id)
            console.print(f"[green]✓ Loaded loop: {loop.name}[/green]")
            console.print(f"  ID: {loop.id}")
            console.print(f"  Description: {loop.description}")
            console.print(f"  Nodes: {loop.get_node_ids()}")
            console.print(f"  Entry Point: {loop.get_entry_node_id()}")

        else:
            console.print(f"[red]Unknown type: {definition_type}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def test_graph(
    loop_id: str = typer.Argument(..., help="Loop ID (e.g., story_spark)")
):
    """Test creating and compiling a graph."""
    try:
        console.print(f"[bold]Testing graph: {loop_id}[/bold]\n")

        # Create graph
        graph_factory = get_graph_factory()
        graph = graph_factory.create_loop_graph(loop_id)

        console.print(f"[green]✓ Graph compiled successfully[/green]")
        console.print(f"  Loop: {loop_id}")
        console.print(f"  Ready to invoke")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Graph test failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def download_spec(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Specific spec version to download (e.g., spec-v1.0.0)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if already cached"),
    show_path: bool = typer.Option(False, "--show-path", "-p", help="Show the downloaded spec path")
):
    """
    Download the latest QuestFoundry spec from GitHub releases.

    This downloads spec releases to ~/.cache/questfoundry/spec/, allowing you to
    use newer specs without reinstalling the package.

    Examples:
        qf download-spec                    # Download latest spec release
        qf download-spec --tag spec-v1.0.0  # Download specific version
        qf download-spec --force            # Force re-download

    Environment Variables:
        QF_SPEC_SOURCE: Control which spec to use (auto/monorepo/bundled/download)
            - Set to 'download' to always use downloaded spec
    """
    try:
        from questfoundry.runtime.core.spec_fetcher import (
            download_latest_release_spec,
            SpecFetchError
        )

        # Show downloading message
        version_info = f" ({tag})" if tag else " (latest)"
        console.print(f"\n[bold]Downloading spec{version_info}...[/bold]\n")

        # Download spec
        spec_path = download_latest_release_spec(tag=tag, force=force)

        # Get version from metadata
        metadata_file = spec_path / ".questfoundry-spec.json"
        version_str = tag or "latest"
        if metadata_file.exists():
            import json
            metadata = json.loads(metadata_file.read_text())
            version_str = metadata.get("tag", version_str)

        # Show success message
        console.print(Panel(
            f"[bold green]✓ Spec downloaded successfully[/bold green]\n\n"
            f"[bold]Version:[/bold] {version_str}\n"
            f"[bold]Location:[/bold] {spec_path if show_path else '~/.cache/questfoundry/spec/'}",
            style="green",
            border_style="green",
            title="Download Complete"
        ))

        console.print("\n[dim]To use the downloaded spec, set: QF_SPEC_SOURCE=download[/dim]")
        console.print("[dim]Or use QF_SPEC_SOURCE=auto to auto-select (monorepo → bundled → download)[/dim]")

    except SpecFetchError as e:
        console.print(Panel(
            f"[red]Failed to download spec:[/red]\n{e}",
            style="red",
            border_style="red",
            title="Download Failed"
        ))
        logger.error(f"Spec download failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def version():
    """Show version information."""
    from questfoundry.runtime.core.spec_fetcher import get_spec_source_preference
    import os

    console.print("[bold]QuestFoundry Runtime[/bold]")
    console.print("Version: 0.1.0")
    console.print("Status: Phase 6 - CLI/Runtime Architecture Fix")
    console.print("\nArchitecture:")
    console.print("  • Primary: Natural language → Showrunner role")
    console.print("  • Debug: Direct loop invocation (qf loop)")

    # Show spec source info
    spec_source = get_spec_source_preference()
    console.print(f"\nSpec Source: {spec_source}")
    if spec_source != "auto":
        console.print(f"  [dim](Set via QF_SPEC_SOURCE={os.getenv('QF_SPEC_SOURCE')})[/dim]")


@app.callback()
def main(
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v: INFO, -vv: DEBUG, -vvv: full details)")
):
    """
    QuestFoundry Runtime - Natural language interface to studio production.

    Primary workflow: qf ask "<your request in plain language>"
    Debug workflow: qf loop <loop_id> --context "key=value"

    Verbosity levels:
        (none): WARNING - Only show warnings and errors
        -v:     INFO - Show general progress information
        -vv:    DEBUG - Show detailed debugging info
        -vvv:   DEBUG + full details - Show everything including library logs
    """
    import logging
    from questfoundry.runtime.logging_config import setup_logging, set_level

    # Map verbosity count to logging level
    if verbose == 0:
        level = "WARNING"
        show_path = False
    elif verbose == 1:
        level = "INFO"
        show_path = False
    elif verbose == 2:
        level = "DEBUG"
        show_path = True
    else:  # verbose >= 3
        level = "DEBUG"
        show_path = True
        # At -vvv, also enable verbose logging from libraries
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("anthropic").setLevel(logging.INFO)

    # Reconfigure logging with appropriate level
    setup_logging(level=level, show_time=(verbose >= 2), show_path=show_path)


if __name__ == "__main__":
    app()
