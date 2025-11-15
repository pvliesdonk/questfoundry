"""Configuration and utility commands for QuestFoundry CLI."""

from typing import Annotated

import typer
from questfoundry.execution.manifest_loader import ManifestLoader, resolve_manifest_location
from questfoundry.roles.registry import RoleRegistry
from rich.console import Console
from rich.table import Table

from .config import check_api_keys, get_config_value, set_config_value

console = Console()

app = typer.Typer(
    name="config",
    help="Configuration and utility commands",
)


@app.command()
def doctor() -> None:
    """Check system health and configuration."""
    console.print("[cyan]Running QuestFoundry diagnostics...[/cyan]\n")

    # Check 1: Manifest loader
    console.print("[bold]1. Checking for compiled manifests...[/bold]")
    try:
        manifest_dir = resolve_manifest_location()
        loader = ManifestLoader(manifest_dir)
        manifests = loader.list_available_manifests()

        if manifests:
            console.print(
                f"  [green]✓ Found {len(manifests)} playbook manifests in {manifest_dir}[/green]"
            )
        else:
            console.print(
                f"  [yellow]⚠ No manifests found in {manifest_dir}[/yellow]"
            )
            console.print(
                "    [dim]Run 'python lib/python/scripts/bundle_resources.py' to compile manifests[/dim]"
            )
    except Exception as e:
        console.print(f"  [red]✗ Failed to load manifests: {e}[/red]")

    # Check 2: API keys
    console.print("\n[bold]2. Checking API keys...[/bold]")
    api_keys = check_api_keys()

    for key, is_set in api_keys.items():
        if is_set:
            console.print(f"  [green]✓ {key}: Found[/green]")
        else:
            console.print(f"  [yellow]⚠ {key}: Not found[/yellow]")

    if not any(api_keys.values()):
        console.print(
            "\n  [dim]Set API keys in .env file or environment variables[/dim]"
        )

    # Check 3: Workspace configuration
    console.print("\n[bold]3. Checking workspace configuration...[/bold]")
    workspace_path = get_config_value("workspace_path")

    if workspace_path:
        console.print(f"  [green]✓ Workspace path: {workspace_path}[/green]")
    else:
        console.print("  [yellow]⚠ No workspace path configured[/yellow]")
        console.print(
            "    [dim]Set with: qf config set workspace_path /path/to/workspace[/dim]"
        )

    console.print("\n[cyan]Diagnostics complete![/cyan]")


@app.command(name="list-playbooks")
def list_playbooks() -> None:
    """List all available playbooks."""
    try:
        manifest_dir = resolve_manifest_location()
        loader = ManifestLoader(manifest_dir)
        playbook_ids = loader.list_available_manifests()

        if not playbook_ids:
            console.print("[yellow]No playbooks available[/yellow]")
            console.print(
                "[dim]Run 'python lib/python/scripts/bundle_resources.py' to compile manifests[/dim]"
            )
            return

        # Create table
        table = Table(title="Available Playbooks", show_header=True, header_style="bold cyan")
        table.add_column("Playbook ID", style="cyan", no_wrap=True)
        table.add_column("Display Name", style="green")
        table.add_column("Description")
        table.add_column("Steps", justify="right", style="yellow")

        # Load each manifest and display info
        for playbook_id in playbook_ids:
            try:
                manifest = loader.load_manifest(playbook_id)
                display_name = manifest.get("display_name", playbook_id)
                description = manifest.get("description", "")
                steps = len(manifest.get("steps", []))

                # Truncate description if too long
                if len(description) > 60:
                    description = description[:57] + "..."

                table.add_row(playbook_id, display_name, description, str(steps))

            except Exception as e:
                table.add_row(playbook_id, "[red]Error[/red]", str(e), "-")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: Failed to list playbooks: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="list-roles")
def list_roles() -> None:
    """List all available roles."""
    try:
        role_registry = RoleRegistry(provider_registry=None)  # type: ignore
        roles = role_registry.list_roles()

        if not roles:
            console.print("[yellow]No roles available[/yellow]")
            return

        # Create table
        table = Table(title="Available Roles", show_header=True, header_style="bold cyan")
        table.add_column("Role Name", style="cyan", no_wrap=True)
        table.add_column("Class", style="green")

        for role_name in sorted(roles):
            role_class = role_registry._roles.get(role_name)
            class_name = role_class.__name__ if role_class else "Unknown"
            table.add_row(role_name, class_name)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: Failed to list roles: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def set(
    key: Annotated[str, typer.Argument(help="Configuration key")],
    value: Annotated[str, typer.Argument(help="Configuration value")],
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print(f"[green]✓ Set {key} = {value}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Failed to set configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def get(
    key: Annotated[str, typer.Argument(help="Configuration key")],
) -> None:
    """Get a configuration value."""
    try:
        value = get_config_value(key)
        if value is not None:
            console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")
        else:
            console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: Failed to get configuration: {e}[/red]")
        raise typer.Exit(1)
