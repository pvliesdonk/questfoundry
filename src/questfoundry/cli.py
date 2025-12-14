"""
QuestFoundry CLI - Interactive Fiction Studio.

Commands:
- doctor: Validate domain and check provider availability
- config show: Display resolved configuration
- roles: List agents with archetypes and capabilities
- projects: Manage story projects
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="qf",
    help="QuestFoundry - AI-powered interactive fiction studio",
    no_args_is_help=True,
)
console = Console()


# Global options
def verbose_callback(value: int) -> int:
    """Process verbosity level."""
    return value


# =============================================================================
# Main Commands
# =============================================================================


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


@app.command()
def doctor(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """Validate domain and check provider availability."""
    asyncio.run(_doctor_async(domain))


async def _doctor_async(domain_path: Path) -> None:
    """Async implementation of doctor command."""
    from questfoundry.runtime.config import ProviderState, load_config
    from questfoundry.runtime.domain import load_studio

    console.print()
    console.print("[bold]QuestFoundry Doctor[/bold]")
    console.print()

    # Load configuration
    config = load_config()

    # Load domain
    console.print(f"[dim]Loading domain from {domain_path}...[/dim]")
    result = await load_studio(domain_path)

    if not result.success:
        console.print("[red]✗ Domain failed to load[/red]")
        for error in result.errors:
            console.print(f"  [red]• {error.message}[/red]")
        raise typer.Exit(1)

    studio = result.studio
    console.print(f"[green]✓[/green] Domain: [bold]{studio.name}[/bold]")
    if studio.version:
        console.print(f"  Version: {studio.version}")

    # Count entities
    entry_agents = [a for a in studio.agents if a.is_entry_agent]
    entry_names = ", ".join(a.id for a in entry_agents)

    console.print(
        f"  {len(studio.agents)} agents (entry: {entry_names or 'none'})"
    )
    console.print(f"  {len(studio.tools)} tools, {len(studio.stores)} stores, {len(studio.playbooks)} playbooks")
    console.print(f"  {len(studio.artifact_types)} artifact types, {len(studio.asset_types)} asset types")

    # Check providers
    console.print()
    console.print("[bold]Providers:[/bold]")

    for name, provider in config.providers.items():
        if provider.state == ProviderState.AVAILABLE:
            model = provider.default_model or "default"
            if provider.host:
                console.print(
                    f"  [green]✓[/green] {name} @ {provider.host} ({model})"
                )
            else:
                console.print(f"  [green]✓[/green] {name} ({model})")
        elif provider.state == ProviderState.UNCONFIGURED:
            console.print(f"  [yellow]○[/yellow] {name} [dim](unconfigured)[/dim]")
        else:
            console.print(f"  [red]✗[/red] {name} [dim](unavailable)[/dim]")

    # Warnings
    if result.warnings:
        console.print()
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]• {warning.message}[/yellow]")

    console.print()


@app.command("config")
def config_show(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """Show resolved configuration."""
    import yaml

    from questfoundry.runtime.config import load_config

    config = load_config()

    # Build config dict for display
    config_dict = {
        "domain_path": str(config.domain_path),
        "default_provider": config.default_provider,
        "providers": {},
        "model_classes": config.model_classes.mappings,
        "logging": {
            "enabled": config.log_events,
            "path": str(config.log_path) if config.log_path else None,
        },
        "langsmith": {
            "enabled": config.langsmith_enabled,
            "project": config.langsmith_project,
        },
    }

    for name, provider in config.providers.items():
        config_dict["providers"][name] = {
            "state": provider.state.value,
            "host": provider.host,
            "default_model": provider.default_model,
            # Don't show API keys
            "api_key": "***" if provider.api_key else None,
        }

    console.print(Panel(yaml.dump(config_dict, default_flow_style=False), title="Configuration"))


@app.command()
def roles(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """List agents with archetypes and capabilities."""
    asyncio.run(_roles_async(domain))


async def _roles_async(domain_path: Path) -> None:
    """Async implementation of roles command."""
    from questfoundry.runtime.domain import load_studio

    result = await load_studio(domain_path)

    if not result.success:
        console.print("[red]✗ Failed to load domain[/red]")
        for error in result.errors:
            console.print(f"  [red]• {error.message}[/red]")
        raise typer.Exit(1)

    studio = result.studio

    table = Table(title=f"Agents in {studio.name}")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Archetypes", style="green")
    table.add_column("Entry", style="yellow")
    table.add_column("Capabilities", style="dim")

    for agent in sorted(studio.agents, key=lambda a: a.id):
        archetypes = ", ".join(agent.archetypes)
        entry = "✓" if agent.is_entry_agent else ""
        caps = len(agent.capabilities)
        table.add_row(agent.id, agent.name, archetypes, entry, str(caps))

    console.print(table)


# =============================================================================
# Projects Subcommand
# =============================================================================

projects_app = typer.Typer(help="Manage story projects")
app.add_typer(projects_app, name="projects")


@projects_app.command("list")
def projects_list(
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
) -> None:
    """List all projects."""
    from questfoundry.runtime.storage import list_projects

    projects = list_projects(projects_dir)

    if not projects:
        console.print(f"[dim]No projects found in {projects_dir}[/dim]")
        console.print(f"[dim]Create one with: qf projects create <name>[/dim]")
        return

    table = Table(title="Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Studio")
    table.add_column("Created")

    for project in projects:
        if project.info:
            created = project.info.created_at.strftime("%Y-%m-%d")
            table.add_row(
                project.info.id,
                project.info.name,
                project.info.studio_id or "-",
                created,
            )

    console.print(table)


@projects_app.command("create")
def projects_create(
    name: Annotated[str, typer.Argument(help="Project name")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    description: Annotated[
        Optional[str],
        typer.Option("--description", help="Project description"),
    ] = None,
    studio: Annotated[
        str,
        typer.Option("--studio", "-s", help="Studio ID"),
    ] = "questfoundry",
) -> None:
    """Create a new project."""
    from questfoundry.runtime.storage import Project

    # Generate ID from name
    project_id = name.lower().replace(" ", "-").replace("_", "-")
    project_path = projects_dir / project_id

    if project_path.exists():
        console.print(f"[red]✗ Project already exists at {project_path}[/red]")
        raise typer.Exit(1)

    project = Project.create(
        path=project_path,
        name=name,
        description=description,
        studio_id=studio,
    )

    console.print(f"[green]✓[/green] Created project [bold]{name}[/bold]")
    console.print(f"  Path: {project_path}")
    console.print(f"  Studio: {studio}")


@projects_app.command("info")
def projects_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
) -> None:
    """Show project information."""
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id

    try:
        project = Project.open(project_path)
    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1)

    info = project.info
    if not info:
        console.print("[red]✗ Could not load project info[/red]")
        raise typer.Exit(1)

    panel_content = f"""
[bold]Name:[/bold] {info.name}
[bold]ID:[/bold] {info.id}
[bold]Description:[/bold] {info.description or '-'}
[bold]Studio:[/bold] {info.studio_id or '-'}
[bold]Created:[/bold] {info.created_at.strftime('%Y-%m-%d %H:%M')}
[bold]Updated:[/bold] {info.updated_at.strftime('%Y-%m-%d %H:%M')}

[bold]Path:[/bold] {project_path}
[bold]Database:[/bold] {project.db_path}
"""

    console.print(Panel(panel_content.strip(), title=f"Project: {info.name}"))

    # Count artifacts
    artifacts = project.query_artifacts(limit=1000)
    if artifacts:
        console.print()
        console.print(f"[bold]Artifacts:[/bold] {len(artifacts)}")

        # Group by type
        by_type: dict[str, int] = {}
        for a in artifacts:
            t = a["_type"]
            by_type[t] = by_type.get(t, 0) + 1

        for t, count in sorted(by_type.items()):
            console.print(f"  {t}: {count}")

    project.close()


if __name__ == "__main__":
    app()
