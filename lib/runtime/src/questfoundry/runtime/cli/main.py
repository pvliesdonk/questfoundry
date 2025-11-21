"""
CLI Main Entry Point - Typer-based command-line interface.

Based on spec: components/cli.md
Uses Typer framework for natural language-friendly CLI.
"""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from questfoundry.runtime.cli.parser import CLIParser
from questfoundry.runtime.cli.showrunner import Showrunner, ParsedIntent
from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.logging_config import setup_logging, get_logger

# Set up Rich logging
setup_logging(level="INFO", show_time=False, show_path=False)
logger = get_logger(__name__)

# Create app
app = typer.Typer(
    name="qf",
    help="QuestFoundry runtime - Transform YAML roles and loops into executable graphs"
)

# Create console for rich output
console = Console()

# Initialize components
parser = CLIParser()
graph_factory = GraphFactory()
state_manager = StateManager()
showrunner = Showrunner(graph_factory=graph_factory, state_manager=state_manager)


@app.command()
def write(
    text: str = typer.Argument(..., help="Scene or section description"),
    mode: str = typer.Option("workshop", help="Execution mode (workshop or production)")
):
    """
    Write a new scene or section using story_spark loop.

    Example:
        qf write "a tense cargo bay confrontation"
    """
    try:
        console.print(Panel(
            f"[bold]Writing new scene[/bold]\n{text}",
            style="blue"
        ))

        # Create parsed intent
        intent = ParsedIntent(
            action="write",
            args=[text],
            flags={"mode": mode},
            loop_id="story_spark"
        )

        # Execute through Showrunner
        result = showrunner.execute_request(f"write {text}", intent)

        # Display result
        if result.success:
            console.print(Panel(result.summary, style="green", title="✓ Success"))
        else:
            console.print(Panel(result.summary, style="red", title="✗ Error"))
            if result.error:
                logger.error(f"Execution error: {result.error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def review(
    target: str = typer.Argument("story", help="What to review (story, hooks, etc)")
):
    """
    Review and harvest hooks from story using hook_harvest loop.

    Example:
        qf review story
    """
    try:
        console.print(Panel(
            f"[bold]Reviewing {target}[/bold]",
            style="blue"
        ))

        # Create parsed intent
        intent = ParsedIntent(
            action="review",
            args=[target],
            flags={},
            loop_id="hook_harvest"
        )

        # Execute through Showrunner
        result = showrunner.execute_request(f"review {target}", intent)

        # Display result
        if result.success:
            console.print(Panel(result.summary, style="green", title="✓ Success"))
        else:
            console.print(Panel(result.summary, style="red", title="✗ Error"))
            if result.error:
                logger.error(f"Execution error: {result.error}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def list_loops():
    """List all available loops."""
    try:
        console.print("[bold]Available Loops:[/bold]")

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

        for loop_id in loops:
            try:
                loop = graph_factory.schema_registry.load_loop(loop_id)
                console.print(f"  • [cyan]{loop_id}[/cyan]: {loop.description}")
            except Exception:
                console.print(f"  • [red]{loop_id}[/red]: (failed to load)")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def list_roles():
    """List all available roles."""
    try:
        console.print("[bold]Available Roles:[/bold]")

        roles = [
            "plotwright", "scene_smith", "gatekeeper", "style_lead",
            "lore_weaver", "codex_curator", "audio_producer", "audio_director",
            "art_director", "illustrator", "player_narrator", "researcher",
            "translator", "book_binder", "export_service", "showrunner"
        ]

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
        console.print(f"[bold]Testing {definition_type}: {definition_id}[/bold]")

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
        console.print(f"[bold]Testing graph: {loop_id}[/bold]")

        # Create graph
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
    console.print("Status: Phase 5B Implementation")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """QuestFoundry Runtime - Transform YAML definitions into executable loops."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    app()
