"""QuestFoundry Config Command - Configuration management.

Provides:
- Generate default configuration files (YAML/TOML)
- Dump current configuration (with resolved env vars)
- Show configuration paths
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from questfoundry.runtime.config import (
    GoogleConfig,
    LLMConfig,
    LoggingConfig,
    OllamaConfig,
    OpenAIConfig,
    PathsConfig,
    RuntimeConfig,
    get_settings,
)

console = Console()

# Create config subcommand group
config_app = typer.Typer(
    name="config",
    help="Configuration management commands",
    no_args_is_help=True,
)


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    return dict(model)


def _get_default_config() -> dict[str, Any]:
    """Get configuration with all default values."""
    return {
        "runtime": _model_to_dict(RuntimeConfig()),
        "llm": _model_to_dict(LLMConfig()),
        "ollama": _model_to_dict(OllamaConfig()),
        "google": _model_to_dict(GoogleConfig()),
        "openai": _model_to_dict(OpenAIConfig()),
        "paths": _model_to_dict(PathsConfig()),
        "logging": _model_to_dict(LoggingConfig()),
    }


def _get_current_config() -> dict[str, Any]:
    """Get current configuration with resolved environment variables."""
    settings = get_settings()
    return {
        "runtime": _model_to_dict(settings.runtime),
        "llm": _model_to_dict(settings.llm),
        "ollama": _model_to_dict(settings.ollama),
        "google": _model_to_dict(settings.google),
        "openai": _model_to_dict(settings.openai),
        "paths": _model_to_dict(settings.paths),
        "logging": _model_to_dict(settings.logging),
    }


def _filter_none_values(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove None values from a dictionary."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            filtered = _filter_none_values(v)
            if filtered:
                result[k] = filtered
        elif v is not None:
            result[k] = v
    return result


def _to_yaml(config: dict[str, Any], with_comments: bool = False) -> str:
    """Convert config dict to YAML string."""
    import yaml

    filtered = _filter_none_values(config)

    if with_comments:
        header = """# QuestFoundry Configuration
# ===========================
#
# This file configures the QuestFoundry runtime.
# All values can be overridden via environment variables with QF_ prefix.
#
# Example: llm.provider -> QF_LLM__PROVIDER
#
# For nested values, use double underscore: QF_GOOGLE__MODEL
#
# API keys should be set via environment variables:
#   export GOOGLE_API_KEY="your-key"
#   export OPENAI_API_KEY="your-key"
#

"""
        return header + yaml.dump(
            filtered, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    else:
        return str(
            yaml.dump(filtered, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )


@config_app.command("init")
def config_init(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (stdout if not specified)",
    ),
    minimal: bool = typer.Option(
        False,
        "--minimal",
        "-m",
        help="Only include commonly changed settings",
    ),
) -> None:
    """Generate a default configuration file.

    Creates a configuration file with all default values that you can
    customize for your project.

    Examples:
        qf config init                      # Print YAML to stdout
        qf config init -o questfoundry.yaml # Write to file
        qf config init --minimal            # Only common settings
    """
    config = _get_default_config()

    if minimal:
        config = {
            "llm": {
                "provider": config["llm"]["provider"],
                "temperature": config["llm"]["temperature"],
            },
            "ollama": {
                "host": config["ollama"]["host"],
                "model": config["ollama"]["model"],
            },
            "google": {
                "model": config["google"]["model"],
            },
            "runtime": {
                "max_delegations": config["runtime"]["max_delegations"],
                "debug": config["runtime"]["debug"],
            },
            "paths": {
                "project_dir": config["paths"]["project_dir"],
            },
        }

    content = _to_yaml(config, with_comments=True)

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Configuration written to:[/green] {output}")
    else:
        print(content)


@config_app.command("dump")
def config_dump(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (stdout if not specified)",
    ),
) -> None:
    """Dump current configuration with resolved values.

    Shows the active configuration including values from environment
    variables and config files. Useful for debugging configuration issues.

    Examples:
        qf config dump                        # Print current config
        qf config dump -o current.yaml        # Save to file
    """
    config = _get_current_config()
    content = _to_yaml(config, with_comments=False)

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]Configuration written to:[/green] {output}")
    else:
        print(content)


@config_app.command("show")
def config_show(
    section: str | None = typer.Argument(
        None,
        help="Configuration section to show (runtime, llm, ollama, google, openai, paths, logging)",
    ),
) -> None:
    """Show current configuration in a readable format.

    Displays the active configuration with nice formatting.
    Optionally filter to a specific section.

    Examples:
        qf config show           # Show all sections
        qf config show llm       # Show only LLM settings
        qf config show google    # Show only Google settings
    """
    settings = get_settings()
    config = _get_current_config()

    valid_sections = {"runtime", "llm", "ollama", "google", "openai", "paths", "logging"}

    if section:
        section = section.lower()
        if section not in valid_sections:
            console.print(f"[red]Unknown section:[/red] {section}")
            console.print(f"Valid sections: {', '.join(sorted(valid_sections))}")
            raise typer.Exit(1)

        console.print(f"\n[bold]{section.title()} Configuration:[/bold]\n")
        section_config = config.get(section, {})
        for key, value in section_config.items():
            if value is not None:
                console.print(f"  {key}: {value}")
    else:
        console.print("\n[bold]Current Configuration:[/bold]\n")

        # Show effective provider
        console.print(f"[cyan]Provider:[/cyan] {settings.llm.provider}")
        console.print(f"[cyan]Model:[/cyan] {settings.get_llm_model()}")
        console.print(f"[cyan]Temperature:[/cyan] {settings.llm.temperature}")
        console.print()

        for section_name in sorted(valid_sections):
            console.print(f"[bold]{section_name}:[/bold]")
            section_config = config.get(section_name, {})
            for key, value in section_config.items():
                if value is not None:
                    console.print(f"  {key}: {value}")
            console.print()


@config_app.command("path")
def config_path() -> None:
    """Show where configuration files are loaded from.

    Displays the search paths for configuration files and indicates
    which files were found and loaded.
    """
    from questfoundry.runtime.config import _find_config_file

    console.print("\n[bold]Configuration File Search Paths:[/bold]\n")

    search_paths = [
        Path.cwd() / "questfoundry.yaml",
        Path.cwd() / "questfoundry.toml",
        Path.cwd() / ".questfoundry.yaml",
        Path.home() / ".config" / "questfoundry" / "config.yaml",
        Path.home() / ".config" / "questfoundry" / "config.toml",
    ]

    found_config = _find_config_file()

    for path in search_paths:
        if path.exists():
            if path == found_config:
                console.print(f"  [green]+[/green] {path} [green](active)[/green]")
            else:
                console.print(f"  [yellow].[/yellow] {path} [dim](exists but shadowed)[/dim]")
        else:
            console.print(f"  [dim]-[/dim] {path}")

    console.print()

    if found_config:
        console.print(f"[bold]Active config file:[/bold] {found_config}")
    else:
        console.print("[dim]No configuration file found. Using defaults + environment.[/dim]")

    # Show .env status
    env_file = Path.cwd() / ".env"
    console.print()
    if env_file.exists():
        console.print(f"[bold].env file:[/bold] {env_file} [green](found)[/green]")
    else:
        console.print("[bold].env file:[/bold] not found in current directory")

    console.print()


@config_app.command("providers")
def config_providers() -> None:
    """Show available LLM providers and their status.

    Checks which providers are configured and available.
    """
    from questfoundry.runtime.providers import check_google_available, check_ollama_available

    console.print("\n[bold]LLM Provider Status:[/bold]\n")

    table = Table()
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Default Model")
    table.add_column("Notes")

    settings = get_settings()

    # Ollama
    ollama_available = check_ollama_available(settings.ollama.host)
    ollama_status = "[green]Ready[/green]" if ollama_available else "[red]Not running[/red]"
    table.add_row(
        "ollama",
        ollama_status,
        settings.ollama.model,
        f"Host: {settings.ollama.host}",
    )

    # Google
    google_available = check_google_available()
    google_status = "[green]Ready[/green]" if google_available else "[yellow]No API key[/yellow]"
    table.add_row(
        "google",
        google_status,
        settings.google.model,
        "Set GOOGLE_API_KEY" if not google_available else "API key configured",
    )

    # OpenAI
    import os

    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    openai_status = "[green]Ready[/green]" if openai_available else "[yellow]No API key[/yellow]"
    table.add_row(
        "openai",
        openai_status,
        settings.openai.model,
        "Set OPENAI_API_KEY" if not openai_available else "API key configured",
    )

    console.print(table)

    console.print()
    console.print(f"[bold]Active provider:[/bold] {settings.llm.provider}")
    console.print(f"[bold]Active model:[/bold] {settings.get_llm_model()}")
    console.print()
