"""
CLI Main Entry Point - Typer-based command-line interface.

Based on spec: components/cli.md
Uses Typer framework for natural language-friendly CLI.
"""

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from questfoundry.runtime.cli.parser import CLIParser
from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
        # Parse command
        command = parser.parse(f"write {text}")
        if not command:
            console.print("[red]Error: Could not parse command[/red]")
            sys.exit(1)

        # Add mode to context
        command.context["narration_mode"] = mode

        # Initialize state
        state = state_manager.initialize_state(
            loop_id=command.loop_id,
            context=command.context
        )

        # Create and invoke graph
        graph = graph_factory.create_loop_graph(command.loop_id, command.context)

        console.print(Panel(
            f"[bold]Executing {command.loop_id}[/bold]\nTU: {state['tu_id']}",
            style="blue"
        ))

        # Invoke graph (would do actual LLM calls here)
        # For now, just show state
        console.print(f"\n[green]✓ Created TU {state['tu_id']}[/green]")
        console.print(f"  Loop: {command.loop_id}")
        console.print(f"  Status: {state['tu_lifecycle']}")
        if command.context.get("scene_text"):
            console.print(f"  Content: {command.context['scene_text']}")

        console.print("\n[yellow]Note: Full LLM execution requires API keys and configuration[/yellow]")

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
        # Parse command
        command = parser.parse(f"review {target}")
        if not command:
            console.print("[red]Error: Could not parse command[/red]")
            sys.exit(1)

        # Initialize state
        state = state_manager.initialize_state(
            loop_id=command.loop_id,
            context=command.context
        )

        console.print(Panel(
            f"[bold]Reviewing {target}[/bold]\nTU: {state['tu_id']}",
            style="blue"
        ))

        console.print(f"\n[green]✓ Started review TU {state['tu_id']}[/green]")
        console.print(f"  Loop: {command.loop_id}")
        console.print(f"  Target: {target}")

        console.print("\n[yellow]Note: Full review requires API keys and configuration[/yellow]")

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
