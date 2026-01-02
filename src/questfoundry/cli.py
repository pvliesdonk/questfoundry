"""QuestFoundry CLI - typer application entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from questfoundry.pipeline import PipelineOrchestrator

app = typer.Typer(
    name="qf",
    help="QuestFoundry: Pipeline-driven interactive fiction generation.",
    no_args_is_help=True,
)
console = Console()


def _get_orchestrator(project_path: Path) -> PipelineOrchestrator:
    """Get a pipeline orchestrator for the project.

    Args:
        project_path: Path to the project directory.

    Returns:
        Configured PipelineOrchestrator.
    """
    from questfoundry.pipeline import PipelineOrchestrator

    return PipelineOrchestrator(project_path)


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Project name")],
    path: Annotated[
        Path, typer.Option("--path", "-p", help="Parent directory for the project")
    ] = Path(),
) -> None:
    """Initialize a new story project.

    Creates a project directory with the necessary structure:
    - project.yaml: Project configuration
    - artifacts/: Generated stage outputs
    """
    import yaml

    from questfoundry.pipeline.config import create_default_config

    # Create project directory
    project_path = path / name
    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{project_path}' already exists")
        raise typer.Exit(1)

    project_path.mkdir(parents=True)

    # Create artifacts directory
    artifacts_dir = project_path / "artifacts"
    artifacts_dir.mkdir()

    # Create project.yaml
    config = create_default_config(name)
    config_data = {
        "name": config.name,
        "version": config.version,
        "pipeline": {
            "stages": config.stages,
        },
        "providers": {
            "default": f"{config.provider.name}/{config.provider.model}",
        },
    }

    config_file = project_path / "project.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Created project: [bold]{name}[/bold]")
    console.print(f"  Location: {project_path.absolute()}")
    console.print()
    console.print("Next steps:")
    console.print(f"  cd {name}")
    console.print('  qf dream "Your story idea..."')


@app.command()
def dream(
    prompt: Annotated[str | None, typer.Argument(help="Story idea or concept")] = None,
    project: Annotated[Path, typer.Option("--project", "-p", help="Project directory")] = Path(),
) -> None:
    """Run DREAM stage - establish creative vision.

    Takes a story idea and generates a creative vision artifact with
    genre, tone, themes, and style direction.
    """
    # Check for project.yaml
    config_file = project / "project.yaml"
    if not config_file.exists():
        console.print(
            "[red]Error:[/red] No project.yaml found. Run 'qf init <name>' first or use --project."
        )
        raise typer.Exit(1)

    # Get prompt interactively if not provided
    if prompt is None:
        prompt = typer.prompt("Enter your story idea")

    console.print()
    console.print("[dim]Running DREAM stage...[/dim]")

    # Run the stage
    orchestrator = _get_orchestrator(project)

    try:
        result = asyncio.run(orchestrator.run_stage("dream", {"user_prompt": prompt}))
    finally:
        asyncio.run(orchestrator.close())

    if result.status == "failed":
        console.print()
        console.print("[red]✗[/red] DREAM stage failed")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(1)

    # Display success
    console.print()
    console.print("[green]✓[/green] DREAM stage completed")
    console.print(f"  Artifact: [cyan]{result.artifact_path}[/cyan]")
    console.print(f"  Tokens: {result.tokens_used:,}")
    console.print(f"  Duration: {result.duration_seconds:.1f}s")

    # Show preview of artifact
    if result.artifact_path and result.artifact_path.exists():
        import yaml

        with result.artifact_path.open() as f:
            artifact = yaml.safe_load(f)

        console.print()
        genre = artifact.get("genre", "unknown")
        subgenre = artifact.get("subgenre")
        genre_display = f"{genre} ({subgenre})" if subgenre else genre

        console.print(f"  Genre: [bold]{genre_display}[/bold]")

        if tones := artifact.get("tone"):
            console.print(f"  Tone: {', '.join(tones)}")

        if themes := artifact.get("themes"):
            console.print(f"  Themes: {', '.join(themes)}")

    console.print()
    console.print("Run: [cyan]qf status[/cyan] to see pipeline state")


@app.command()
def status(
    project: Annotated[Path, typer.Option("--project", "-p", help="Project directory")] = Path(),
) -> None:
    """Show pipeline status for current project."""
    # Check for project.yaml
    config_file = project / "project.yaml"
    if not config_file.exists():
        console.print(
            "[red]Error:[/red] No project.yaml found. Run 'qf init <name>' first or use --project."
        )
        raise typer.Exit(1)

    orchestrator = _get_orchestrator(project)
    pipeline_status = orchestrator.get_status()

    # Create status table
    table = Table(title=f"Pipeline Status: {pipeline_status.project_name}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Run", style="dim")

    status_icons = {
        "completed": "[green]✓[/green] completed",
        "pending": "[dim]○[/dim] pending",
        "failed": "[red]✗[/red] failed",
    }

    for stage_name, info in pipeline_status.stages.items():
        status_display = status_icons.get(info.status, info.status)
        last_run = info.last_run.strftime("%Y-%m-%d %H:%M") if info.last_run else "-"
        table.add_row(stage_name, status_display, last_run)

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
