"""Runtime commands for QuestFoundry CLI."""

from pathlib import Path
from typing import Annotated, Optional

import questionary
import typer
from questfoundry.execution.manifest_loader import ManifestLoader, resolve_manifest_location
from questfoundry.state.workspace import WorkspaceManager
from rich.console import Console

from .main import get_orchestrator

console = Console()

app = typer.Typer(
    name="run",
    help="Runtime commands for executing playbooks and interacting with the agent",
)


@app.command()
def init(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to initialize workspace (default: current directory)"),
    ] = None,
) -> None:
    """Initialize a new QuestFoundry workspace."""
    if path is None:
        path = Path.cwd()

    console.print(f"[cyan]Initializing workspace at: {path}[/cyan]")

    try:
        workspace = WorkspaceManager(path)
        workspace.init_workspace(
            name=path.name or "untitled",
            description="QuestFoundry project",
            version="0.1.0",
        )
        console.print(f"[green]✓ Workspace initialized successfully at {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize workspace: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def send(
    message: Annotated[str, typer.Argument(help="Message to send to the agent")],
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace path"),
    ] = None,
) -> None:
    """Send a single message to the agent (non-interactive mode)."""
    console.print(f"[cyan]Sending message: {message}[/cyan]")

    try:
        orchestrator = get_orchestrator(workspace)
        project_info = orchestrator.workspace.get_project_info()

        # Execute the goal
        result = orchestrator.execute_goal(
            goal=message,
            project_id=project_info.name,
        )

        if result.success:
            console.print(f"[green]✓ Task completed successfully[/green]")
            if result.artifacts_created:
                console.print(
                    f"[green]  Created {len(result.artifacts_created)} artifacts[/green]"
                )
        else:
            console.print(f"[red]✗ Task failed: {result.error}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace path"),
    ] = None,
) -> None:
    """Start an interactive chat session with the agent."""
    console.print("[cyan]Starting interactive chat session...[/cyan]")
    console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]\n")

    try:
        orchestrator = get_orchestrator(workspace)
        project_info = orchestrator.workspace.get_project_info()

        while True:
            # Get user input
            message = questionary.text(
                "You:",
                qmark="",
            ).ask()

            if message is None or message.lower() in ["exit", "quit", "q"]:
                console.print("\n[cyan]Goodbye![/cyan]")
                break

            if not message.strip():
                continue

            # Execute the goal
            try:
                result = orchestrator.execute_goal(
                    goal=message,
                    project_id=project_info.name,
                )

                if result.success:
                    console.print(f"[green]✓ Task completed[/green]")
                    if result.artifacts_created:
                        console.print(
                            f"[dim]  Created {len(result.artifacts_created)} artifacts[/dim]"
                        )
                else:
                    console.print(f"[yellow]⚠ Task incomplete: {result.error}[/yellow]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

            console.print()  # Empty line for readability

    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def playbook(
    playbook_id: Annotated[
        Optional[str],
        typer.Argument(help="ID of the playbook to run (or omit for interactive selection)"),
    ] = None,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace path"),
    ] = None,
) -> None:
    """Run a specific playbook by ID."""
    try:
        # If no playbook_id provided, show interactive selector
        if playbook_id is None:
            manifest_dir = resolve_manifest_location()
            loader = ManifestLoader(manifest_dir)
            available = loader.list_available_manifests()

            if not available:
                console.print("[red]No playbooks available[/red]")
                raise typer.Exit(1)

            # Create choices with display names
            choices = []
            for pid in available:
                try:
                    manifest = loader.load_manifest(pid)
                    display_name = manifest.get("display_name", pid)
                    choices.append(
                        questionary.Choice(
                            title=f"{display_name} ({pid})",
                            value=pid,
                        )
                    )
                except Exception:
                    choices.append(questionary.Choice(title=pid, value=pid))

            playbook_id = questionary.select(
                "Select a playbook to run:",
                choices=choices,
            ).ask()

            if playbook_id is None:
                console.print("[yellow]No playbook selected[/yellow]")
                raise typer.Exit()

        console.print(f"[cyan]Running playbook: {playbook_id}[/cyan]")

        orchestrator = get_orchestrator(workspace)
        result = orchestrator.run_playbook(playbook_id)

        if result.success:
            console.print(f"[green]✓ Playbook '{playbook_id}' completed successfully[/green]")
            if result.artifacts_created:
                console.print(
                    f"[green]  Created {len(result.artifacts_created)} artifacts[/green]"
                )
        else:
            console.print(f"[red]✗ Playbook '{playbook_id}' failed: {result.error}[/red]")
            raise typer.Exit(1)

    except KeyError as e:
        console.print(f"[red]Error: Playbook not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
