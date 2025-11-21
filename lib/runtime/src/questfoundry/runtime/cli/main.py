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

# Create console for rich output (force UTF-8 for Unicode support)
import io
console = Console(file=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))

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


def get_state_manager() -> StateManager:
    """Lazy-load state manager."""
    global _state_manager
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution info")
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
    """
    try:
        # Display customer message
        console.print(Panel(
            f"[bold]Your Request:[/bold]\n{message}",
            style="blue",
            border_style="blue"
        ))

        # Get Showrunner and interpret request
        showrunner = get_showrunner()
        result = showrunner.interpret_and_execute(message, verbose=verbose)

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution info")
):
    """
    Debug/audit mode: Directly invoke a loop (bypasses Showrunner).

    ⚠️  WARNING: This is for debugging, testing, and auditing only.
    You are bypassing the Showrunner's mandate and directly controlling
    internal studio operations. For normal use, prefer: qf ask "..."

    Examples:
        qf loop story_spark --context "scene_text=tense cargo bay scene"
        qf loop hook_harvest --mode workshop --verbose
        qf loop gatecheck --verbose
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
        console.print()

        # Initialize state
        state_manager = get_state_manager()
        state = state_manager.initialize_state(loop_id, context_dict)

        # Create and execute loop graph
        graph_factory = get_graph_factory()
        graph = graph_factory.create_loop_graph(loop_id)

        logger.info(f"Invoking loop: {loop_id}")
        final_state = graph.invoke(state)

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
def version():
    """Show version information."""
    console.print("[bold]QuestFoundry Runtime[/bold]")
    console.print("Version: 0.1.0")
    console.print("Status: Phase 6 - CLI/Runtime Architecture Fix")
    console.print("\nArchitecture:")
    console.print("  • Primary: Natural language → Showrunner role")
    console.print("  • Debug: Direct loop invocation (qf loop)")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """
    QuestFoundry Runtime - Natural language interface to studio production.

    Primary workflow: qf ask "<your request in plain language>"
    Debug workflow: qf loop <loop_id> --context "key=value"
    """
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    app()
