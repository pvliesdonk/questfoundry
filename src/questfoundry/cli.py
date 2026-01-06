"""QuestFoundry CLI - typer application entry point."""

from __future__ import annotations

import asyncio
import atexit
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from questfoundry.observability import close_file_logging, configure_logging, get_logger


def _is_interactive_tty() -> bool:
    """Check if stdin/stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


# Load environment variables from .env file
load_dotenv()

if TYPE_CHECKING:
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent

    from questfoundry.pipeline import PipelineOrchestrator, StageResult

app = typer.Typer(
    name="qf",
    help="QuestFoundry: Pipeline-driven interactive fiction generation.",
    no_args_is_help=True,
)
console = Console()

# Default directory for projects
DEFAULT_PROJECTS_DIR = Path("projects")

# Default prompt for interactive mode when no prompt is provided
DEFAULT_INTERACTIVE_DREAM_PROMPT = (
    "I'd like to create a new interactive fiction story. "
    "Please help me develop the creative vision."
)

# Global state for logging flags (set by callback, used by commands)
_verbose: int = 0
_log_enabled: bool = False
_projects_dir: Path = DEFAULT_PROJECTS_DIR


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase verbosity: -v for INFO, -vv for DEBUG.",
        ),
    ] = 0,
    log: Annotated[
        bool,
        typer.Option(
            "--log",
            help="Enable file logging to {project}/logs/ (debug.jsonl, llm_calls.jsonl).",
        ),
    ] = False,
    projects_dir: Annotated[
        Path,
        typer.Option(
            "--projects-dir",
            "-d",
            help="Base directory for projects (default: ./projects).",
            envvar="QF_PROJECTS_DIR",
        ),
    ] = DEFAULT_PROJECTS_DIR,
) -> None:
    """QuestFoundry: Pipeline-driven interactive fiction generation."""
    global _verbose, _log_enabled, _projects_dir
    _verbose = verbose
    _log_enabled = log
    _projects_dir = projects_dir

    # Configure console logging (file logging configured later when project is known)
    configure_logging(verbosity=verbose)


def _configure_project_logging(project_path: Path) -> None:
    """Configure file logging if --log flag was set.

    Args:
        project_path: Path to the project directory.
    """
    if _log_enabled:
        configure_logging(verbosity=_verbose, log_to_file=True, project_path=project_path)
        atexit.register(close_file_logging)


def _resolve_project_path(project: Path | None) -> Path:
    """Resolve project path from argument.

    Resolution order:
    1. If project is None, use current directory
    2. If project exists as given, use it
    3. If project is a name (no path separators), look in _projects_dir

    Args:
        project: Project path or name from CLI argument.

    Returns:
        Resolved project path.
    """
    if project is None:
        return Path()

    # If path exists as given, use it
    if project.exists():
        return project

    # If it's a simple name (no path sep), try in projects dir
    if len(project.parts) == 1:
        projects_path = _projects_dir / project
        if projects_path.exists():
            return projects_path

    # Return as-is (will fail in _require_project with helpful error)
    return project


def _require_project(project_path: Path) -> None:
    """Verify project.yaml exists, exit with error if not.

    Args:
        project_path: Path to the project directory.
    """
    config_file = project_path / "project.yaml"
    if not config_file.exists():
        console.print(
            "[red]Error:[/red] No project.yaml found. Run 'qf init <name>' first or use --project."
        )
        raise typer.Exit(1)


def _get_orchestrator(
    project_path: Path, provider_override: str | None = None
) -> PipelineOrchestrator:
    """Get a pipeline orchestrator for the project.

    Args:
        project_path: Path to the project directory.
        provider_override: Optional provider string (e.g., "openai/gpt-4o") to override config.

    Returns:
        Configured PipelineOrchestrator.
    """
    from questfoundry.pipeline import PipelineOrchestrator

    return PipelineOrchestrator(
        project_path,
        provider_override=provider_override,
        enable_llm_logging=_log_enabled,
    )


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Project name")],
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Parent directory for the project (default: --projects-dir).",
        ),
    ] = None,
) -> None:
    """Initialize a new story project.

    Creates a project directory with the necessary structure:
    - project.yaml: Project configuration
    - artifacts/: Generated stage outputs
    """
    from ruamel.yaml import YAML

    from questfoundry.pipeline.config import create_default_config

    # Use global projects dir if no path specified
    parent_dir = path if path is not None else _projects_dir
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Create project directory
    project_path = parent_dir / name
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
    yaml_writer = YAML()
    yaml_writer.default_flow_style = False
    with config_file.open("w") as f:
        yaml_writer.dump(config_data, f)

    console.print(f"[green]✓[/green] Created project: [bold]{name}[/bold]")
    console.print(f"  Location: {project_path.absolute()}")
    console.print()
    console.print("Next steps:")
    console.print(f"  cd {name}")
    console.print('  qf dream "Your story idea..."')


@app.command()
def dream(
    prompt: Annotated[str | None, typer.Argument(help="Story idea or concept")] = None,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="LLM provider (e.g., ollama/qwen3:8b, openai/gpt-4o)"),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive conversation mode. Defaults to auto-detect based on TTY.",
        ),
    ] = None,
) -> None:
    """Run DREAM stage - establish creative vision.

    Takes a story idea and generates a creative vision artifact with
    genre, tone, themes, and style direction.

    By default, interactive mode is auto-detected based on whether the
    terminal is a TTY. Use --interactive/-i to force interactive mode,
    or --no-interactive/-I to force direct mode.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    # Get logger after logging is fully configured
    log = get_logger(__name__)

    # Determine interactive mode: explicit flag > TTY detection
    use_interactive = interactive if interactive is not None else _is_interactive_tty()

    # Handle prompt: if not provided, interactive mode uses default, non-interactive fails
    if prompt is None:
        if use_interactive:
            # In interactive mode, start with a guiding prompt that invites conversation
            prompt = DEFAULT_INTERACTIVE_DREAM_PROMPT
        else:
            # Non-interactive requires explicit prompt (fail fast, don't hang in CI/CD)
            console.print("[red]Error:[/red] Prompt required in non-interactive mode.")
            console.print("Provide a prompt argument or use --interactive/-i flag.")
            raise typer.Exit(1)

    log.info("stage_start", stage="dream")
    log.debug("user_prompt", prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt)

    # Build context
    context: dict[str, object] = {"user_prompt": prompt, "interactive": use_interactive}
    if use_interactive:
        log.debug("interactive_mode", mode="enabled")

        session: PromptSession[str] = PromptSession(multiline=True)
        bindings = KeyBindings()

        def _submit(event: KeyPressEvent) -> None:  # pragma: no cover - UI behavior
            """Enter submits the current buffer."""
            event.current_buffer.validate_and_handle()

        bindings.add("enter")(_submit)

        def _insert_newline(event: KeyPressEvent) -> None:  # pragma: no cover - UI behavior
            """Ctrl+Enter (Ctrl+J) inserts a newline."""
            event.current_buffer.insert_text("\n")

        bindings.add("c-j")(_insert_newline)

        async def _async_user_input() -> str | None:
            """Get user input asynchronously with prompt_toolkit."""
            console.print()
            try:
                loop = asyncio.get_running_loop()

                def _prompt() -> str:
                    with patch_stdout():
                        return cast(
                            "str",
                            session.prompt(
                                HTML("<b><ansicyan>You</ansicyan></b>: "),
                                multiline=True,
                                key_bindings=bindings,
                            ),
                        )

                user_input = await loop.run_in_executor(None, _prompt)
                return user_input if user_input.strip() else None
            except (EOFError, KeyboardInterrupt):
                return None

        def _display_assistant_message(content: str) -> None:
            """Display assistant message with richer formatting."""
            console.print()
            renderable = Markdown(content)
            panel = Panel.fit(
                renderable,
                title="Assistant",
                title_align="left",
                border_style="green",
            )
            console.print(panel)

        context["user_input_fn"] = _async_user_input
        context["on_assistant_message"] = _display_assistant_message

        thinking_indicator = "[dim]••• thinking •••[/dim]"

        def _on_llm_start(_: str) -> None:
            console.print(thinking_indicator, end="\r")

        def _on_llm_end(_: str) -> None:
            console.print(" " * len("••• thinking •••"), end="\r")

        context["on_llm_start"] = _on_llm_start
        context["on_llm_end"] = _on_llm_end
    else:
        log.debug("interactive_mode", mode="disabled")

    async def _run_dream() -> StageResult:
        """Run DREAM stage and close orchestrator."""
        orchestrator = _get_orchestrator(project_path, provider_override=provider)
        log.debug("provider_configured", provider=f"{orchestrator.config.provider.name}")
        try:
            return await orchestrator.run_stage("dream", context)
        finally:
            await orchestrator.close()

    console.print()

    if use_interactive:
        # Interactive mode: no spinner, direct output
        console.print("[dim]Starting interactive DREAM stage...[/dim]")
        console.print("[dim]The AI will discuss your story idea with you.[/dim]")
        console.print()
        result = asyncio.run(_run_dream())
    else:
        # Non-interactive: use progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Running DREAM stage...", total=None)
            result = asyncio.run(_run_dream())

    if result.status == "failed":
        log.error("stage_failed", stage="dream", errors=result.errors)
        console.print()
        console.print("[red]✗[/red] DREAM stage failed")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(1)

    log.info(
        "stage_complete",
        stage="dream",
        tokens=result.tokens_used,
        duration=result.duration_seconds,
    )

    # Display success
    console.print()
    console.print("[green]✓[/green] DREAM stage completed")
    console.print(f"  Artifact: [cyan]{result.artifact_path}[/cyan]")
    console.print(f"  Tokens: {result.tokens_used:,}")
    console.print(f"  Duration: {result.duration_seconds:.1f}s")

    if _log_enabled:
        console.print(f"  Logs: [dim]{project_path / 'logs'}[/dim]")

    # Show preview of artifact
    if result.artifact_path and result.artifact_path.exists():
        from ruamel.yaml import YAML

        yaml_reader = YAML()
        with result.artifact_path.open() as f:
            artifact = yaml_reader.load(f)

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
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
) -> None:
    """Show pipeline status for current project."""
    project_path = _resolve_project_path(project)
    _require_project(project_path)

    orchestrator = _get_orchestrator(project_path)
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


@app.command()
def doctor(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
) -> None:
    """Check configuration and provider connectivity.

    Validates environment variables, tests provider connections,
    and optionally checks project configuration.
    """
    console.print("[bold]QuestFoundry Doctor[/bold]")
    console.print()

    all_ok = True

    # Check configuration
    all_ok &= _check_configuration()

    # Check provider connectivity
    all_ok &= asyncio.run(_check_providers())

    # Check project (if specified or current dir has project.yaml)
    project_path = _resolve_project_path(project)
    if (project_path / "project.yaml").exists():
        all_ok &= _check_project(project_path)

    console.print()
    if all_ok:
        console.print("[green]All checks passed![/green]")
    else:
        console.print("[yellow]Some checks failed or were skipped.[/yellow]")
        raise typer.Exit(1)


def _check_configuration() -> bool:
    """Check environment configuration."""
    import os

    console.print("[bold]Configuration[/bold]")

    checks = [
        ("OLLAMA_HOST", os.getenv("OLLAMA_HOST")),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        ("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")),
        ("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY")),
    ]

    any_provider = False
    for name, value in checks:
        if value:
            # Mask secrets
            if "KEY" in name:
                display = f"{value[:7]}...{value[-3:]}" if len(value) > 10 else "(set)"
            else:
                display = value
            console.print(f"  [green]✓[/green] {name}: {display}")
            any_provider = True
        else:
            console.print(f"  [dim]○[/dim] {name}: not configured")

    console.print()
    return any_provider  # At least one provider configured


async def _check_providers() -> bool:
    """Check provider connectivity."""
    import os

    console.print("[bold]Provider Connectivity[/bold]")

    all_ok = True

    # Check Ollama
    if os.getenv("OLLAMA_HOST"):
        all_ok &= await _check_ollama()
    else:
        console.print("  [dim]○[/dim] ollama: Skipped (OLLAMA_HOST not set)")

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        all_ok &= await _check_openai()
    else:
        console.print("  [dim]○[/dim] openai: Skipped (not configured)")

    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        all_ok &= await _check_anthropic()
    else:
        console.print("  [dim]○[/dim] anthropic: Skipped (not configured)")

    console.print()
    return all_ok


async def _check_ollama() -> bool:
    """Check Ollama connectivity and list models."""
    import json
    import os

    import httpx

    host = os.getenv("OLLAMA_HOST")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if models:
                    model_list = ", ".join(models[:5])
                    if len(models) > 5:
                        model_list += f", +{len(models) - 5} more"
                    console.print(f"  [green]✓[/green] ollama: Connected ({model_list})")
                else:
                    console.print("  [yellow]![/yellow] ollama: Connected (no models pulled)")
                return True
            else:
                console.print(f"  [red]✗[/red] ollama: HTTP {response.status_code}")
                return False
    except httpx.ConnectError:
        console.print(f"  [red]✗[/red] ollama: Connection refused ({host})")
        return False
    except httpx.TimeoutException:
        console.print(f"  [red]✗[/red] ollama: Connection timeout ({host})")
        return False
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] ollama: Request error - {e}")
        return False
    except json.JSONDecodeError:
        console.print("  [red]✗[/red] ollama: Invalid JSON response")
        return False


async def _check_openai() -> bool:
    """Check OpenAI API key validity."""
    import os

    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                console.print("  [green]✓[/green] openai: Connected (API key valid)")
                return True
            elif response.status_code == 401:
                console.print("  [red]✗[/red] openai: Invalid API key")
                return False
            else:
                console.print(f"  [red]✗[/red] openai: HTTP {response.status_code}")
                return False
    except httpx.TimeoutException:
        console.print("  [red]✗[/red] openai: Connection timeout")
        return False
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] openai: Request error - {e}")
        return False


async def _check_anthropic() -> bool:
    """Check Anthropic API key validity."""
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    # Anthropic doesn't have a simple models endpoint,
    # so we just verify the key format
    if api_key and api_key.startswith("sk-ant-"):
        console.print("  [green]✓[/green] anthropic: API key configured")
        return True
    elif api_key:
        console.print("  [yellow]![/yellow] anthropic: Unusual key format")
        return False
    return False


def _check_project(project_path: Path) -> bool:
    """Check project configuration."""
    from questfoundry.pipeline.config import ProjectConfigError, load_project_config

    console.print("[bold]Project[/bold]")

    all_ok = True

    # Check project.yaml
    config_file = project_path / "project.yaml"
    if config_file.exists():
        console.print("  [green]✓[/green] project.yaml: Found")

        # Load and validate config
        try:
            config = load_project_config(project_path)
            console.print(f"  [green]✓[/green] Project name: {config.name}")
            console.print(
                f"  [green]✓[/green] Default provider: {config.provider.name}/{config.provider.model}"
            )
        except ProjectConfigError as e:
            console.print(f"  [red]✗[/red] Config error: {e}")
            all_ok = False
    else:
        console.print("  [dim]○[/dim] project.yaml: Not found (not in a project)")

    # Check artifacts directory
    artifacts_dir = project_path / "artifacts"
    if artifacts_dir.exists():
        artifact_count = len(list(artifacts_dir.glob("*.yaml")))
        console.print(f"  [green]✓[/green] Artifacts directory: {artifact_count} artifact(s)")
    else:
        console.print("  [dim]○[/dim] Artifacts directory: Not found")

    console.print()
    return all_ok


if __name__ == "__main__":
    app()
