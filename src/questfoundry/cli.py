"""QuestFoundry CLI - typer application entry point."""

from __future__ import annotations

import asyncio
import atexit
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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

# Type alias for artifact preview functions
PreviewFn = Callable[[dict[str, Any]], None]

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

DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT = (
    "Let's brainstorm story elements based on the creative vision. "
    "Help me develop characters, locations, and dramatic dilemmas."
)

DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT = (
    "Generate characters, locations, objects, factions, and dramatic dilemmas "
    "based on the creative vision from the DREAM stage."
)

DEFAULT_INTERACTIVE_SEED_PROMPT = (
    "Let's triage this brainstorm into a committed story structure. "
    "Help me decide which entities to keep, which dilemmas to explore, "
    "and what the initial beats should be."
)

DEFAULT_NONINTERACTIVE_SEED_PROMPT = (
    "Triage the brainstorm into committed story structure: curate entities, "
    "decide which dilemmas to explore as paths, create initial beats, "
    "and sketch convergence points."
)

DEFAULT_GROW_PROMPT = (
    "Build the complete branching structure from the SEED graph: "
    "enumerate arcs, create passages, derive choices and codewords."
)

# Pipeline stage order and configuration
STAGE_ORDER = ["dream", "brainstorm", "seed", "grow"]

# Stage prompt configuration for the run command
# Maps stage name to (default_interactive_prompt, default_noninteractive_prompt)
STAGE_PROMPTS: dict[str, tuple[str, str | None]] = {
    "dream": (
        DEFAULT_INTERACTIVE_DREAM_PROMPT,
        None,  # DREAM requires explicit prompt
    ),
    "brainstorm": (
        DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT,
        DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT,
    ),
    "seed": (
        DEFAULT_INTERACTIVE_SEED_PROMPT,
        DEFAULT_NONINTERACTIVE_SEED_PROMPT,
    ),
    "grow": (
        DEFAULT_GROW_PROMPT,
        DEFAULT_GROW_PROMPT,  # Same for both modes (GROW ignores prompt)
    ),
}

# Message shown after SEED stage completes
PATH_FREEZE_MESSAGE = "[yellow]PATH FREEZE:[/yellow] No new paths can be created after SEED."

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
    project_path: Path,
    provider_override: str | None = None,
    provider_discuss_override: str | None = None,
    provider_summarize_override: str | None = None,
    provider_serialize_override: str | None = None,
) -> PipelineOrchestrator:
    """Get a pipeline orchestrator for the project.

    Args:
        project_path: Path to the project directory.
        provider_override: Optional provider string (e.g., "openai/gpt-5-mini") to override config.
        provider_discuss_override: Optional provider override for discuss phase.
        provider_summarize_override: Optional provider override for summarize phase.
        provider_serialize_override: Optional provider override for serialize phase.

    Returns:
        Configured PipelineOrchestrator.
    """
    from questfoundry.pipeline import PipelineOrchestrator

    return PipelineOrchestrator(
        project_path,
        provider_override=provider_override,
        provider_discuss_override=provider_discuss_override,
        provider_summarize_override=provider_summarize_override,
        provider_serialize_override=provider_serialize_override,
        enable_llm_logging=_log_enabled,
    )


# =============================================================================
# Stage Command Helpers
# =============================================================================


def _setup_interactive_context(console_: Console) -> dict[str, Any]:
    """Set up interactive mode context with user input and callbacks.

    Creates prompt_toolkit session, key bindings, and all the callbacks
    needed for interactive conversation mode.

    Args:
        console_: Rich console for output.

    Returns:
        Context dict with user_input_fn, on_assistant_message,
        on_llm_start, on_llm_end callbacks configured.
    """
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
        console_.print()
        try:
            loop = asyncio.get_running_loop()

            def _prompt() -> str:
                with patch_stdout():
                    prompt_text: str = session.prompt(
                        HTML("<b><ansicyan>You</ansicyan></b>: "),
                        multiline=True,
                        key_bindings=bindings,
                    )
                    return prompt_text

            user_input = await loop.run_in_executor(None, _prompt)
            return user_input if user_input.strip() else None
        except (EOFError, KeyboardInterrupt):
            return None

    def _display_assistant_message(content: str) -> None:
        """Display assistant message with richer formatting."""
        console_.print()
        renderable = Markdown(content)
        panel = Panel.fit(
            renderable,
            title="Assistant",
            title_align="left",
            border_style="green",
        )
        console_.print(panel)

    thinking_indicator = "[dim]••• thinking •••[/dim]"

    def _on_llm_start(_: str) -> None:
        console_.print(thinking_indicator, end="\r")

    def _on_llm_end(_: str) -> None:
        console_.print(" " * len("••• thinking •••"), end="\r")

    return {
        "user_input_fn": _async_user_input,
        "on_assistant_message": _display_assistant_message,
        "on_llm_start": _on_llm_start,
        "on_llm_end": _on_llm_end,
    }


async def _run_stage_async(
    stage_name: str,
    project_path: Path,
    context: dict[str, Any],
    provider: str | None,
    provider_discuss: str | None = None,
    provider_summarize: str | None = None,
    provider_serialize: str | None = None,
) -> StageResult:
    """Run a stage asynchronously and close orchestrator.

    Args:
        stage_name: Name of the stage to run.
        project_path: Path to the project directory.
        context: Context dict for the stage.
        provider: Optional provider override for all phases.
        provider_discuss: Optional provider override for discuss phase.
        provider_summarize: Optional provider override for summarize phase.
        provider_serialize: Optional provider override for serialize phase.

    Returns:
        StageResult from the stage execution.
    """
    log = get_logger(__name__)
    orchestrator = _get_orchestrator(
        project_path,
        provider_override=provider,
        provider_discuss_override=provider_discuss,
        provider_summarize_override=provider_summarize,
        provider_serialize_override=provider_serialize,
    )
    log.debug("provider_configured", provider=f"{orchestrator.config.provider.name}")
    try:
        return await orchestrator.run_stage(stage_name, context)
    finally:
        await orchestrator.close()


def _run_stage_command(
    stage_name: str,
    project_path: Path,
    prompt: str | None,
    provider: str | None,
    interactive: bool | None,
    default_interactive_prompt: str,
    default_noninteractive_prompt: str | None,
    preview_fn: PreviewFn | None,
    next_step_hint: str | None = None,
    provider_discuss: str | None = None,
    provider_summarize: str | None = None,
    provider_serialize: str | None = None,
    resume_from: str | None = None,
) -> None:
    """Common logic for running a stage command.

    This is the main helper that consolidates the shared logic across
    dream, brainstorm, and seed commands.

    Args:
        stage_name: Name of the stage (e.g., "dream", "brainstorm", "seed").
        project_path: Path to the project directory.
        prompt: User-provided prompt, or None to use defaults.
        provider: Optional provider override string for all phases.
        interactive: Explicit interactive mode flag, or None for auto-detect.
        default_interactive_prompt: Default prompt for interactive mode.
        default_noninteractive_prompt: Default prompt for non-interactive mode.
            If None, non-interactive mode requires an explicit prompt.
        preview_fn: Optional function to display artifact preview.
        next_step_hint: Optional hint about next step (e.g., "qf brainstorm").
        provider_discuss: Optional provider override for discuss phase.
        provider_summarize: Optional provider override for summarize phase.
        provider_serialize: Optional provider override for serialize phase.
    """
    log = get_logger(__name__)

    # Determine interactive mode: explicit flag > TTY detection
    use_interactive = interactive if interactive is not None else _is_interactive_tty()

    # Handle prompt: apply defaults based on mode
    if prompt is None:
        if use_interactive:
            prompt = default_interactive_prompt
        elif default_noninteractive_prompt is not None:
            prompt = default_noninteractive_prompt
        else:
            # Non-interactive requires explicit prompt (e.g., DREAM stage)
            console.print("[red]Error:[/red] Prompt required in non-interactive mode.")
            console.print("Provide a prompt argument or use --interactive/-i flag.")
            raise typer.Exit(1)

    log.info("stage_start", stage=stage_name)
    log.debug("user_prompt", prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt)

    # Build context
    context: dict[str, Any] = {"user_prompt": prompt, "interactive": use_interactive}
    if resume_from:
        context["resume_from"] = resume_from

    # Add phase progress callback for GROW stage (shows phase-by-phase progress)
    if stage_name == "grow":

        def _on_phase_progress(phase: str, status: str, detail: str | None) -> None:
            """Display phase progress for GROW stage."""
            status_icon = "[green]✓[/green]" if status == "completed" else "[yellow]○[/yellow]"
            detail_str = f" ({detail})" if detail else ""
            console.print(f"  {status_icon} {phase}{detail_str}")

        context["on_phase_progress"] = _on_phase_progress

    if use_interactive:
        log.debug("interactive_mode", mode="enabled")
        context.update(_setup_interactive_context(console))
    else:
        log.debug("interactive_mode", mode="disabled")

    console.print()

    if use_interactive:
        # Interactive mode: no spinner, direct output
        console.print(f"[dim]Starting interactive {stage_name.upper()} stage...[/dim]")
        console.print("[dim]The AI will discuss with you.[/dim]")
        console.print("[dim]Type [bold]/done[/bold] or press Enter on empty line to finish.[/dim]")
        console.print()
        result = asyncio.run(
            _run_stage_async(
                stage_name,
                project_path,
                context,
                provider,
                provider_discuss,
                provider_summarize,
                provider_serialize,
            )
        )
    else:
        # Non-interactive mode
        if stage_name == "grow":
            # GROW shows phase-by-phase progress instead of spinner
            console.print(f"[dim]Running {stage_name.upper()} stage...[/dim]")
            result = asyncio.run(
                _run_stage_async(
                    stage_name,
                    project_path,
                    context,
                    provider,
                    provider_discuss,
                    provider_summarize,
                    provider_serialize,
                )
            )
        else:
            # Other stages use progress spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Running {stage_name.upper()} stage...", total=None)
                result = asyncio.run(
                    _run_stage_async(
                        stage_name,
                        project_path,
                        context,
                        provider,
                        provider_discuss,
                        provider_summarize,
                        provider_serialize,
                    )
                )

    if result.status == "failed":
        log.error("stage_failed", stage=stage_name, errors=result.errors)
        console.print()
        console.print(f"[red]✗[/red] {stage_name.upper()} stage failed")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")
        raise typer.Exit(1)

    log.info(
        "stage_complete",
        stage=stage_name,
        tokens=result.tokens_used,
        duration=result.duration_seconds,
    )

    # Display success
    console.print()
    console.print(f"[green]✓[/green] {stage_name.upper()} stage completed")
    console.print(f"  Artifact: [cyan]{result.artifact_path}[/cyan]")
    console.print(f"  Tokens: {result.tokens_used:,}")
    console.print(f"  Duration: {result.duration_seconds:.1f}s")

    if _log_enabled:
        console.print(f"  Logs: [dim]{project_path / 'logs'}[/dim]")

    # Show preview of artifact
    if preview_fn and result.artifact_path and result.artifact_path.exists():
        from ruamel.yaml import YAML

        yaml_reader = YAML()
        with result.artifact_path.open() as f:
            artifact = yaml_reader.load(f)

        console.print()
        preview_fn(artifact)

    console.print()
    if next_step_hint:
        console.print(f"Run: [cyan]{next_step_hint}[/cyan]")
    else:
        console.print("Run: [cyan]qf status[/cyan] to see pipeline state")


# =============================================================================
# Artifact Preview Functions
# =============================================================================


def _preview_dream_artifact(artifact: dict[str, Any]) -> None:
    """Display preview of DREAM artifact."""
    genre = artifact.get("genre", "unknown")
    subgenre = artifact.get("subgenre")
    genre_display = f"{genre} ({subgenre})" if subgenre else genre

    console.print(f"  Genre: [bold]{genre_display}[/bold]")

    if tones := artifact.get("tone"):
        console.print(f"  Tone: {', '.join(tones)}")

    if themes := artifact.get("themes"):
        console.print(f"  Themes: {', '.join(themes)}")


def _preview_brainstorm_artifact(artifact: dict[str, Any]) -> None:
    """Display preview of BRAINSTORM artifact."""
    entities = artifact.get("entities", [])
    dilemmas = artifact.get("dilemmas", artifact.get("tensions", []))

    console.print(f"  Entities: [bold]{len(entities)}[/bold]")

    # Group entities by category
    entity_categories: dict[str, list[str]] = {}
    for entity in entities:
        category = entity.get("entity_category", "unknown")
        entity_categories.setdefault(category, []).append(entity.get("entity_id", "?"))

    for category, ids in entity_categories.items():
        ids_display = ", ".join(ids[:5])
        if len(ids) > 5:
            ids_display += f", +{len(ids) - 5} more"
        console.print(f"    {category}: {ids_display}")

    console.print(f"  Dilemmas: [bold]{len(dilemmas)}[/bold]")
    for dilemma in dilemmas:
        console.print(f"    • {dilemma.get('question', '?')}")


def _preview_seed_artifact(artifact: dict[str, Any]) -> None:
    """Display preview of SEED artifact."""
    entities = artifact.get("entities", [])
    paths = artifact.get("paths", artifact.get("threads", []))
    beats = artifact.get("initial_beats", [])

    # Count retained vs cut entities
    retained = sum(1 for e in entities if e.get("disposition") == "retained")
    cut = sum(1 for e in entities if e.get("disposition") == "cut")

    console.print(f"  Entities: [bold]{retained}[/bold] retained, [dim]{cut}[/dim] cut")

    console.print(f"  Paths: [bold]{len(paths)}[/bold]")
    for path in paths[:3]:
        importance = path.get("path_importance", path.get("thread_importance", "?"))
        importance_style = "bold green" if importance == "major" else "dim"
        console.print(
            f"    • [{importance_style}]{importance}[/{importance_style}] {path.get('name', '?')}"
        )
    if len(paths) > 3:
        console.print(f"    ... and {len(paths) - 3} more")

    console.print(f"  Initial beats: [bold]{len(beats)}[/bold]")


def _preview_grow_artifact(artifact: dict[str, Any]) -> None:
    """Display preview of GROW artifact."""
    arc_count = artifact.get("arc_count", 0)
    passage_count = artifact.get("passage_count", 0)
    choice_count = artifact.get("choice_count", 0)
    codeword_count = artifact.get("codeword_count", 0)
    overlay_count = artifact.get("overlay_count", 0)
    spine = artifact.get("spine_arc_id", "?")

    console.print(f"  Arcs: [bold]{arc_count}[/bold] (spine: {spine})")
    console.print(f"  Passages: [bold]{passage_count}[/bold]")
    console.print(f"  Choices: [bold]{choice_count}[/bold]")
    console.print(f"  Codewords: [bold]{codeword_count}[/bold]")
    if overlay_count:
        console.print(f"  Overlays: [bold]{overlay_count}[/bold]")

    # Show phase summary
    phases = artifact.get("phases_completed", [])
    if phases:
        failed = [p for p in phases if p.get("status") == "failed"]
        warnings = [p for p in phases if "warning" in (p.get("detail") or "").lower()]
        if failed:
            console.print(f"  [red]Failed phases: {len(failed)}[/red]")
        elif warnings:
            console.print(f"  [yellow]Warnings: {len(warnings)} phases[/yellow]")
        else:
            console.print(f"  Phases: [green]{len(phases)} completed[/green]")


# Stage preview function mapping (defined after functions exist)
STAGE_PREVIEW_FNS: dict[str, PreviewFn] = {
    "dream": _preview_dream_artifact,
    "brainstorm": _preview_brainstorm_artifact,
    "seed": _preview_seed_artifact,
    "grow": _preview_grow_artifact,
}


# =============================================================================
# CLI Commands
# =============================================================================


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
        typer.Option(
            "--provider", help="LLM provider for all phases (e.g., ollama/qwen3:4b-instruct-32k)"
        ),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option("--provider-discuss", help="LLM provider for discuss phase"),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option("--provider-summarize", help="LLM provider for summarize phase"),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option("--provider-serialize", help="LLM provider for serialize phase"),
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

    _run_stage_command(
        stage_name="dream",
        project_path=project_path,
        prompt=prompt,
        provider=provider,
        interactive=interactive,
        default_interactive_prompt=DEFAULT_INTERACTIVE_DREAM_PROMPT,
        default_noninteractive_prompt=None,  # DREAM requires explicit prompt
        preview_fn=_preview_dream_artifact,
        next_step_hint="qf brainstorm",
        provider_discuss=provider_discuss,
        provider_summarize=provider_summarize,
        provider_serialize=provider_serialize,
    )


@app.command()
def brainstorm(
    prompt: Annotated[str | None, typer.Argument(help="Guidance for brainstorming")] = None,
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
        typer.Option(
            "--provider", help="LLM provider for all phases (e.g., ollama/qwen3:4b-instruct-32k)"
        ),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option("--provider-discuss", help="LLM provider for discuss phase"),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option("--provider-summarize", help="LLM provider for summarize phase"),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option("--provider-serialize", help="LLM provider for serialize phase"),
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
    """Run BRAINSTORM stage - generate entities and dilemmas.

    Takes the creative vision from DREAM and generates raw creative
    material: characters, locations, objects, factions, and dramatic
    dilemmas (binary questions with two answers each).

    Requires DREAM stage to have completed first.

    By default, interactive mode is auto-detected based on whether the
    terminal is a TTY. Use --interactive/-i to force interactive mode,
    or --no-interactive/-I to force direct mode.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    _run_stage_command(
        stage_name="brainstorm",
        project_path=project_path,
        prompt=prompt,
        provider=provider,
        interactive=interactive,
        default_interactive_prompt=DEFAULT_INTERACTIVE_BRAINSTORM_PROMPT,
        default_noninteractive_prompt=DEFAULT_NONINTERACTIVE_BRAINSTORM_PROMPT,
        preview_fn=_preview_brainstorm_artifact,
        next_step_hint="qf seed",
        provider_discuss=provider_discuss,
        provider_summarize=provider_summarize,
        provider_serialize=provider_serialize,
    )


@app.command()
def seed(
    prompt: Annotated[str | None, typer.Argument(help="Guidance for seeding")] = None,
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
        typer.Option(
            "--provider", help="LLM provider for all phases (e.g., ollama/qwen3:4b-instruct-32k)"
        ),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option("--provider-discuss", help="LLM provider for discuss phase"),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option("--provider-summarize", help="LLM provider for summarize phase"),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option("--provider-serialize", help="LLM provider for serialize phase"),
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
    """Run SEED stage - triage brainstorm into story structure.

    Takes the entities and dilemmas from BRAINSTORM and transforms
    them into committed structure: curated entities, paths with
    consequences, and initial beats.

    CRITICAL: After SEED, no new paths can be created (PATH FREEZE).

    Requires BRAINSTORM stage to have completed first.

    By default, interactive mode is auto-detected based on whether the
    terminal is a TTY. Use --interactive/-i to force interactive mode,
    or --no-interactive/-I to force direct mode.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    _run_stage_command(
        stage_name="seed",
        project_path=project_path,
        prompt=prompt,
        provider=provider,
        interactive=interactive,
        default_interactive_prompt=DEFAULT_INTERACTIVE_SEED_PROMPT,
        default_noninteractive_prompt=DEFAULT_NONINTERACTIVE_SEED_PROMPT,
        preview_fn=_preview_seed_artifact,
        provider_discuss=provider_discuss,
        provider_summarize=provider_summarize,
        provider_serialize=provider_serialize,
    )

    # SEED-specific message about path freeze
    console.print(PATH_FREEZE_MESSAGE)


@app.command()
def grow(
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
        typer.Option(
            "--provider", help="LLM provider for all phases (e.g., ollama/qwen3:4b-instruct-32k)"
        ),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option("--provider-discuss", help="LLM provider for discuss phase"),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option("--provider-summarize", help="LLM provider for summarize phase"),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option("--provider-serialize", help="LLM provider for serialize phase"),
    ] = None,
    resume_from: Annotated[
        str | None,
        typer.Option("--resume-from", help="Resume from named phase (skips earlier phases)"),
    ] = None,
) -> None:
    """Run GROW stage - build complete branching structure.

    Takes the paths and beats from SEED and builds the full
    branching story graph: arcs, passages, choices, codewords,
    and state overlays.

    This stage runs 15 phases (mostly deterministic, some LLM-assisted)
    and manages graph mutations internally.

    Requires SEED stage to have completed first.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    _run_stage_command(
        stage_name="grow",
        project_path=project_path,
        prompt=None,  # Uses DEFAULT_GROW_PROMPT via STAGE_PROMPTS
        provider=provider,
        interactive=False,  # GROW is never interactive (deterministic, graph-driven)
        default_interactive_prompt=DEFAULT_GROW_PROMPT,
        default_noninteractive_prompt=DEFAULT_GROW_PROMPT,
        preview_fn=_preview_grow_artifact,
        next_step_hint="qf fill",
        provider_discuss=provider_discuss,
        provider_summarize=provider_summarize,
        provider_serialize=provider_serialize,
        resume_from=resume_from,
    )


@app.command()
def run(
    to_stage: Annotated[
        str,
        typer.Option(
            "--to",
            "-t",
            help="Run stages up to and including this stage (dream, brainstorm, seed, grow).",
        ),
    ],
    from_stage: Annotated[
        str | None,
        typer.Option(
            "--from",
            "-f",
            help="Start from this stage (default: first incomplete stage).",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help="Initial prompt for DREAM stage (required if DREAM will run).",
        ),
    ] = None,
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
        typer.Option(
            "--provider", help="LLM provider for all phases (e.g., ollama/qwen3:4b-instruct-32k)"
        ),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option("--provider-discuss", help="LLM provider for discuss phase"),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option("--provider-summarize", help="LLM provider for summarize phase"),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option("--provider-serialize", help="LLM provider for serialize phase"),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive mode. Defaults to non-interactive for batch execution.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-run already completed stages."),
    ] = False,
) -> None:
    """Run multiple pipeline stages sequentially.

    Runs all stages from the starting point up to and including the
    target stage. By default, skips already-completed stages and uses
    non-interactive mode for batch execution.

    Examples:
        qf run --to seed --prompt "A mystery story"
        qf run --to brainstorm --from dream --force
        qf run --to seed --prompt "A mystery" --interactive
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    log = get_logger(__name__)

    # Validate stage names
    if to_stage not in STAGE_ORDER:
        console.print(f"[red]Error:[/red] Unknown stage '{to_stage}'")
        console.print(f"Valid stages: {', '.join(STAGE_ORDER)}")
        raise typer.Exit(1)

    if from_stage is not None and from_stage not in STAGE_ORDER:
        console.print(f"[red]Error:[/red] Unknown stage '{from_stage}'")
        console.print(f"Valid stages: {', '.join(STAGE_ORDER)}")
        raise typer.Exit(1)

    to_idx = STAGE_ORDER.index(to_stage)
    from_idx = STAGE_ORDER.index(from_stage) if from_stage else 0

    if from_idx > to_idx:
        console.print("[red]Error:[/red] --from stage must come before --to stage")
        raise typer.Exit(1)

    # Get current pipeline status
    orchestrator = _get_orchestrator(project_path, provider_override=provider)
    pipeline_status = orchestrator.get_status()

    # Determine which stages to run
    stages_to_run: list[str] = []
    for idx in range(from_idx, to_idx + 1):
        stage_name = STAGE_ORDER[idx]
        stage_info = pipeline_status.stages.get(stage_name)

        if force or stage_info is None or stage_info.status != "completed":
            stages_to_run.append(stage_name)
        else:
            log.debug("stage_skipped", stage=stage_name, reason="already completed")

    if not stages_to_run:
        console.print("[green]✓[/green] All stages already completed!")
        console.print("Use [cyan]--force[/cyan] to re-run completed stages.")
        return

    # Determine interactive mode: explicit flag > default to non-interactive for batch execution
    use_interactive = interactive if interactive is not None else False

    # Check if DREAM will run and prompt is required (only in non-interactive mode)
    # In interactive mode, DREAM has a default prompt
    if "dream" in stages_to_run and prompt is None and not use_interactive:
        console.print("[red]Error:[/red] DREAM stage requires a prompt in non-interactive mode.")
        console.print("Use [cyan]--prompt[/cyan] to provide a story idea,")
        console.print("or use [cyan]--interactive[/cyan] to discuss interactively.")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold]Running stages:[/bold] {' → '.join(s.upper() for s in stages_to_run)}")
    console.print()

    # Run each stage
    for stage_name in stages_to_run:
        default_interactive_prompt, default_noninteractive_prompt = STAGE_PROMPTS[stage_name]

        # Use provided prompt only for DREAM, defaults for others
        stage_prompt = prompt if stage_name == "dream" else None

        # Determine next step hint
        stage_idx = STAGE_ORDER.index(stage_name)
        if stage_idx < len(STAGE_ORDER) - 1:
            next_stage = STAGE_ORDER[stage_idx + 1]
            next_step_hint = f"qf {next_stage}"
        else:
            next_step_hint = None

        try:
            _run_stage_command(
                stage_name=stage_name,
                project_path=project_path,
                prompt=stage_prompt,
                provider=provider,
                interactive=use_interactive,
                default_interactive_prompt=default_interactive_prompt,
                default_noninteractive_prompt=default_noninteractive_prompt,
                preview_fn=STAGE_PREVIEW_FNS.get(stage_name),
                next_step_hint=next_step_hint,
                provider_discuss=provider_discuss,
                provider_summarize=provider_summarize,
                provider_serialize=provider_serialize,
            )

            # SEED-specific message
            if stage_name == "seed":
                console.print(PATH_FREEZE_MESSAGE)

        except typer.Exit as e:
            if e.exit_code != 0:
                console.print()
                console.print(f"[red]Pipeline stopped at {stage_name.upper()} stage.[/red]")
                raise

    console.print()
    console.print("[green]✓[/green] [bold]Pipeline run complete![/bold]")


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
