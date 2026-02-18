"""QuestFoundry CLI - typer application entry point."""

from __future__ import annotations

import asyncio
import atexit
import sys
from enum import StrEnum
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
from rich.table import Table

from questfoundry.observability import close_file_logging, configure_logging, get_logger


def _is_interactive_tty() -> bool:
    """Check if stdin/stdout are connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


# Load environment variables from .env file
load_dotenv()

if TYPE_CHECKING:
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent

    from questfoundry.inspection import InspectionReport
    from questfoundry.pipeline import PipelineOrchestrator, StageResult

# Shared CLI option types (used across multiple commands)
MinPriorityOption = Annotated[
    int,
    typer.Option(
        "--min-priority",
        min=1,
        max=3,
        help="Only generate images with this priority or higher (1=must-have, 2=important, 3=all).",
    ),
]

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
DEFAULT_FILL_PROMPT = (
    "Generate prose for all passages: determine voice document, "
    "write prose per passage, review quality, and revise flagged passages."
)

DEFAULT_DRESS_PROMPT = "Establish art direction, generate illustration briefs and codex entries."

# Pipeline stage order and configuration
STAGE_ORDER = ["dream", "brainstorm", "seed", "grow", "fill", "dress", "ship"]

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
    "fill": (
        DEFAULT_FILL_PROMPT,
        DEFAULT_FILL_PROMPT,  # Same for both modes (FILL ignores prompt)
    ),
    "dress": (
        DEFAULT_DRESS_PROMPT,
        DEFAULT_DRESS_PROMPT,  # Same for both modes
    ),
    "ship": (
        "Export to playable format.",
        "Export to playable format.",  # SHIP is deterministic, prompt unused
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
            "[red]Error:[/red] No project.yaml found. "
            "Run 'qf init <name>' first, use --project, or use 'qf run --init'."
        )
        raise typer.Exit(1)


def _ensure_project(
    project_path: Path,
    *,
    auto_init: bool,
    provider: str | None,
) -> Path:
    """Ensure a project exists, optionally creating it.

    Behavior:
    - If project.yaml exists: return project_path as-is.
    - If --init: auto-create without prompting.
    - If interactive TTY: prompt user to create.
    - Otherwise: fail with error.

    Args:
        project_path: Resolved project path (may not exist yet).
        auto_init: Whether --init flag was passed.
        provider: Provider string for project creation.

    Returns:
        Path to the (possibly newly created) project directory.

    Raises:
        typer.Exit: If project doesn't exist and user declines or non-interactive.
    """
    config_file = project_path / "project.yaml"
    if config_file.exists():
        return project_path

    name = project_path.name

    if auto_init:
        # For simple names (no path separators), default to _projects_dir
        # so that `qf run --init --project foo` creates projects/foo/
        parent = project_path.parent
        if len(project_path.parts) == 1:
            parent = _projects_dir
        project_path = _init_project(name, parent, provider=provider)
        console.print(f"[green]✓[/green] Created project: [bold]{name}[/bold]")
        return project_path

    if _is_interactive_tty():
        if typer.confirm(f"Project '{name}' doesn't exist. Create it?", default=True):
            parent = project_path.parent
            if len(project_path.parts) == 1:
                parent = _projects_dir
            project_path = _init_project(name, parent, provider=provider)
            console.print(f"[green]✓[/green] Created project: [bold]{name}[/bold]")
            return project_path
        raise typer.Exit(0)

    # Non-interactive without --init
    console.print(
        "[red]Error:[/red] No project.yaml found. "
        "Use --init to create, or run 'qf init <name>' first."
    )
    raise typer.Exit(1)


def _get_orchestrator(
    project_path: Path,
    provider_override: str | None = None,
    provider_discuss_override: str | None = None,
    provider_summarize_override: str | None = None,
    provider_serialize_override: str | None = None,
    image_provider_override: str | None = None,
) -> PipelineOrchestrator:
    """Get a pipeline orchestrator for the project.

    Args:
        project_path: Path to the project directory.
        provider_override: Optional provider string (e.g., "openai/gpt-5-mini") to override config.
        provider_discuss_override: Optional provider override for discuss/creative phase.
        provider_summarize_override: Optional provider override for summarize/balanced phase.
        provider_serialize_override: Optional provider override for serialize/structured phase.
        image_provider_override: Optional image provider override
            (e.g., "openai/gpt-image-1", "placeholder").

    Returns:
        Configured PipelineOrchestrator.
    """
    from questfoundry.pipeline import PipelineOrchestrator

    # Pass legacy phase names — orchestrator maps them to role names internally
    return PipelineOrchestrator(
        project_path,
        provider_override=provider_override,
        provider_discuss_override=provider_discuss_override,
        provider_summarize_override=provider_summarize_override,
        provider_serialize_override=provider_serialize_override,
        image_provider_override=image_provider_override,
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
    image_provider: str | None = None,
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
        image_provider: Optional image provider override for DRESS stage.

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
        image_provider_override=image_provider,
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
    next_step_hint: str | None = None,
    provider_discuss: str | None = None,
    provider_summarize: str | None = None,
    provider_serialize: str | None = None,
    resume_from: str | None = None,
    image_provider: str | None = None,
    image_budget: int = 0,
    min_priority: int = 3,
    two_step: bool | None = None,
    language: str | None = None,
    skip_codex: bool = False,
) -> None:
    """Common logic for running a stage command.

    This is the main helper that consolidates the shared logic across
    dream, brainstorm, seed, and dress commands.

    Args:
        stage_name: Name of the stage (e.g., "dream", "brainstorm", "seed").
        project_path: Path to the project directory.
        prompt: User-provided prompt, or None to use defaults.
        provider: Optional provider override string for all phases.
        interactive: Explicit interactive mode flag, or None for auto-detect.
        default_interactive_prompt: Default prompt for interactive mode.
        default_noninteractive_prompt: Default prompt for non-interactive mode.
            If None, non-interactive mode requires an explicit prompt.
        next_step_hint: Optional hint about next step (e.g., "qf brainstorm").
        provider_discuss: Optional provider override for discuss phase.
        provider_summarize: Optional provider override for summarize phase.
        provider_serialize: Optional provider override for serialize phase.
        resume_from: Phase name to resume execution from.
        image_provider: Image provider spec for DRESS stage (e.g., ``openai/gpt-image-1``).
        min_priority: Only generate briefs with this priority or higher (1-3).
        two_step: Whether to use two-step prose generation in FILL stage.
        language: ISO 639-1 language code override (e.g., "nl", "ja").
        skip_codex: Skip codex generation in DRESS stage.
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
    if image_provider:
        context["image_provider"] = image_provider
    if image_budget > 0:
        context["image_budget"] = image_budget
    if min_priority < 3:
        context["min_priority"] = min_priority
    if stage_name == "fill" and two_step is not None:
        context["two_step"] = two_step
    if language:
        context["language"] = language
    if skip_codex:
        context["skip_codex"] = True

    # Add phase progress callback (used by GROW, and optionally by other stages)
    def _on_phase_progress(phase: str, status: str, detail: str | None) -> None:
        """Display phase progress for stages that emit phase progress events."""
        detail_str = f" ({detail})" if detail else ""

        if stage_name in ("grow", "fill", "dress"):
            status_icon = "[green]✓[/green]" if status == "completed" else "[yellow]○[/yellow]"
            console.print(f"  {status_icon} {phase}{detail_str}")
            return

        # DREAM / BRAINSTORM / SEED
        if status == "completed":
            status_icon = "[green]✓[/green]"
            console.print(f"{stage_name.upper()}: {phase} {status_icon}{detail_str}")
            return

        if status == "retry":
            # Special-case to match the style requested in #298
            if phase.lower().startswith("outer loop retry"):
                console.print(f"[yellow][{phase}: {detail}][/yellow]")
            else:
                console.print(f"[yellow]{stage_name.upper()}: {phase} ↻{detail_str}[/yellow]")
            return

        # Fallback for unknown statuses
        console.print(f"{stage_name.upper()}: {phase}{detail_str}")

    context["on_phase_progress"] = _on_phase_progress

    # Connectivity retry hook — prompt user when the LLM server goes down
    if _is_interactive_tty():

        async def _on_connectivity_error(failed: int, total: int, error_sample: str) -> bool:
            """Prompt user to retry after LLM connectivity failure."""
            console.print()
            console.print(
                f"[red bold]Connectivity failure:[/red bold] {failed}/{total} batch items failed."
            )
            console.print(f"[dim]Error: {error_sample}[/dim]")
            console.print("[yellow]Verify the LLM server is running before retrying.[/yellow]")
            return await asyncio.to_thread(typer.confirm, "Retry failed items?", default=True)

        context["on_connectivity_error"] = _on_connectivity_error

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
                image_provider,
            )
        )
    else:
        # Non-interactive mode
        if stage_name in ("grow", "fill", "dress"):
            # GROW/FILL/DRESS show phase-by-phase progress instead of spinner
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
            # Other stages show phase-level progress via on_phase_progress
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
    console.print(f"  Tokens: {result.tokens_used:,}")
    console.print(f"  Duration: {result.duration_seconds:.1f}s")

    if _log_enabled:
        console.print(f"  Logs: [dim]{project_path / 'logs'}[/dim]")

    if result.summary_lines:
        console.print("  Summary:")
        for line in result.summary_lines:
            console.print(f"   - {line}")

    console.print()
    if next_step_hint:
        console.print(f"Run: [cyan]{next_step_hint}[/cyan]")
    else:
        console.print("Run: [cyan]qf status[/cyan] to see pipeline state")


# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def version() -> None:
    """Show version information."""
    from questfoundry import __version__

    console.print(f"QuestFoundry v{__version__}")


def _init_project(
    name: str,
    parent_dir: Path,
    provider: str | None = None,
) -> Path:
    """Create a new project directory with config and artifacts.

    Args:
        name: Project name.
        parent_dir: Parent directory for the project.
        provider: Optional provider string (e.g., "openai/gpt-4o").

    Returns:
        Path to the created project directory.

    Raises:
        typer.Exit: If the directory already exists.
    """
    from ruamel.yaml import YAML

    from questfoundry.pipeline.config import create_default_config

    parent_dir.mkdir(parents=True, exist_ok=True)

    project_path = parent_dir / name
    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{project_path}' already exists")
        raise typer.Exit(1)

    project_path.mkdir(parents=True)

    # Create artifacts directory
    artifacts_dir = project_path / "artifacts"
    artifacts_dir.mkdir()

    # Create project.yaml
    config = create_default_config(name, provider=provider)
    config_data = {
        "name": config.name,
        "version": config.version,
        "providers": {
            "default": f"{config.provider.name}/{config.provider.model}",
        },
    }

    config_file = project_path / "project.yaml"
    yaml_writer = YAML()
    yaml_writer.default_flow_style = False
    with config_file.open("w") as f:
        yaml_writer.dump(config_data, f)

    return project_path


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
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Default LLM provider (e.g., ollama/qwen3:4b-instruct-32k, openai/gpt-4o).",
        ),
    ] = None,
) -> None:
    """Initialize a new story project.

    Creates a project directory with the necessary structure:
    - project.yaml: Project configuration
    - artifacts/: Generated stage outputs
    """
    parent_dir = path if path is not None else _projects_dir
    project_path = _init_project(name, parent_dir, provider=provider)

    console.print(f"[green]✓[/green] Created project: [bold]{name}[/bold]")
    console.print(f"  Location: {project_path.absolute()}")
    console.print(f"  Provider: {provider or 'default'}")
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive conversation mode. Defaults to auto-detect based on TTY.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
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
        next_step_hint="qf brainstorm",
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        language=language,
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive conversation mode. Defaults to auto-detect based on TTY.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
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
        next_step_hint="qf seed",
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        language=language,
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive conversation mode. Defaults to auto-detect based on TTY.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
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
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        language=language,
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    resume_from: Annotated[
        str | None,
        typer.Option("--resume-from", help="Resume from named phase (skips earlier phases)"),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
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
        next_step_hint="qf fill",
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        resume_from=resume_from,
        language=language,
    )


@app.command()
def fill(
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    resume_from: Annotated[
        str | None,
        typer.Option("--resume-from", help="Resume from named phase (skips earlier phases)"),
    ] = None,
    two_step: Annotated[
        bool | None,
        typer.Option(
            "--two-step/--no-two-step",
            help="Two-step prose generation: write prose first, then extract entities. "
            "Improves quality by removing JSON constraints from creative output. "
            "Defaults to fill.two_step in project.yaml (true if unset).",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
    ] = None,
) -> None:
    """Run FILL stage - generate prose for all passages.

    Takes the branching structure from GROW and generates prose for
    each passage: determines voice document, writes prose per passage,
    reviews quality, and revises flagged passages.

    This stage runs 4 phases (voice, generate, review, revision)
    and manages graph mutations internally.

    Requires GROW stage to have completed first.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    _run_stage_command(
        stage_name="fill",
        project_path=project_path,
        prompt=None,  # Uses DEFAULT_FILL_PROMPT via STAGE_PROMPTS
        provider=provider,
        interactive=False,  # FILL is never interactive (graph-driven)
        default_interactive_prompt=DEFAULT_FILL_PROMPT,
        default_noninteractive_prompt=DEFAULT_FILL_PROMPT,
        next_step_hint="qf dress",
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        resume_from=resume_from,
        two_step=two_step,
        language=language,
    )


@app.command()
def dress(
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
    ] = None,
    image_provider: Annotated[
        str | None,
        typer.Option("--image-provider", help="Image provider (e.g., openai/gpt-image-1)"),
    ] = None,
    image_budget: Annotated[
        int,
        typer.Option("--image-budget", help="Max images to generate (0=all selected briefs)"),
    ] = 0,
    resume_from: Annotated[
        str | None,
        typer.Option("--resume-from", help="Resume from named phase (skips earlier phases)"),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
    ] = None,
    min_priority: MinPriorityOption = 2,
    no_codex: Annotated[
        bool,
        typer.Option("--no-codex", help="Skip codex generation (Phase 2)"),
    ] = False,
) -> None:
    """Run DRESS stage - art direction, illustrations, and codex.

    Establishes visual identity, generates illustration briefs for
    passages, creates codex entries for entities, and optionally
    generates images via an image provider.

    By default, only generates briefs for priority 1-2 passages
    (--min-priority 2). Use --min-priority 3 to generate briefs
    for all passages.

    Requires FILL stage to have completed first.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    _run_stage_command(
        stage_name="dress",
        project_path=project_path,
        prompt=None,
        provider=provider,
        interactive=False,
        default_interactive_prompt=DEFAULT_DRESS_PROMPT,
        default_noninteractive_prompt=DEFAULT_DRESS_PROMPT,
        next_step_hint="qf ship",
        provider_discuss=provider_creative or provider_discuss,
        provider_summarize=provider_balanced or provider_summarize,
        provider_serialize=provider_structured or provider_serialize,
        resume_from=resume_from,
        image_provider=image_provider,
        image_budget=image_budget,
        min_priority=min_priority,
        language=language,
        skip_codex=no_codex,
    )


def _run_ship(
    project_path: Path,
    export_format: str = "twee",
    output_dir: Path | None = None,
    *,
    embed_assets: bool = False,
) -> Path:
    """Run SHIP stage and return the output file path.

    Args:
        project_path: Path to the project directory.
        export_format: Export format name (json, twee, html, pdf).
        output_dir: Custom output directory.

    Returns:
        Path to the exported file.

    Raises:
        typer.Exit: On ShipStageError.
    """
    from questfoundry.pipeline.stages.ship import ShipStage, ShipStageError

    console.print()
    console.print(f"[dim]Exporting story as {export_format}...[/dim]")

    try:
        stage = ShipStage(project_path)
        output_file = stage.execute(
            export_format=export_format,
            output_dir=output_dir,
            embed_assets=embed_assets,
        )
    except ShipStageError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print()
    console.print(f"[green]✓[/green] SHIP stage completed ({export_format})")
    console.print(f"  Output: [cyan]{output_file}[/cyan]")
    return output_file


@app.command()
def ship(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    export_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Export format: json, twee, html, pdf.",
        ),
    ] = "twee",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Custom output directory (default: PROJECT/exports/FORMAT/).",
        ),
    ] = None,
    embed: Annotated[
        bool,
        typer.Option(
            "--embed",
            help="Embed image assets as base64 data URLs in HTML output.",
        ),
    ] = False,
) -> None:
    """Run SHIP stage - export story to playable format.

    Exports the completed story graph to a playable format.
    SHIP is deterministic and does not use an LLM.

    Supported formats:
      - twee: SugarCube 2 Twee source (default)
      - html: Standalone playable HTML file
      - json: Structured JSON data

    Requires FILL stage to have completed first (all passages need prose).
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    log = get_logger(__name__)
    log.info("ship_cli_start", format=export_format)

    _run_ship(
        project_path,
        export_format=export_format,
        output_dir=output,
        embed_assets=embed,
    )


@app.command("generate-images")
def generate_images(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    image_provider: Annotated[
        str | None,
        typer.Option(
            "--image-provider",
            help="Image provider (e.g., openai/gpt-image-1, placeholder).",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="LLM provider for prompt distillation (e.g., openai/gpt-4o).",
        ),
    ] = None,
    image_budget: Annotated[
        int,
        typer.Option("--image-budget", help="Max images to generate (0=all selected briefs)."),
    ] = 0,
    min_priority: MinPriorityOption = 3,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate images even if illustrations already exist."),
    ] = False,
) -> None:
    """Generate images for an existing DRESS project.

    Runs only the image generation phase (Phase 4) of the DRESS stage.
    Requires that 'qf dress' has already been run to create briefs
    and selections. By default, briefs that already have illustrations
    are skipped; use --force to regenerate them.

    The image provider is resolved in order: --image-provider flag,
    QF_IMAGE_PROVIDER env var, providers.image in project.yaml.

    The LLM provider (for prompt distillation) is resolved in order:
    --provider flag, QF_PROVIDER env var, providers.default in project.yaml.

    Examples:
        qf generate-images --project my-story --image-provider placeholder
        qf generate-images -p my-story --image-provider a1111/sdxl_base --provider openai/gpt-4o
    """
    import os

    from questfoundry.pipeline.config import load_project_config
    from questfoundry.pipeline.stages.dress import DressStage, DressStageError

    project_path = _resolve_project_path(project)
    _require_project(project_path)
    _configure_project_logging(project_path)

    log = get_logger(__name__)
    config = load_project_config(project_path)

    # Resolve image provider: CLI flag → env var → project.yaml
    resolved_provider = (
        image_provider
        or os.environ.get("QF_IMAGE_PROVIDER")
        or config.providers.get_image_provider()
    )

    if not resolved_provider:
        console.print(
            "[red]Error:[/red] No image provider specified. "
            "Use --image-provider, set QF_IMAGE_PROVIDER, "
            "or add providers.image to project.yaml."
        )
        raise typer.Exit(1)

    # Resolve LLM provider for prompt distillation (only A1111 needs this
    # for condensing prose briefs into SD tags): CLI flag → env var → project.yaml
    llm_provider_spec = provider or os.environ.get("QF_PROVIDER") or config.providers.default

    llm_model = None
    if llm_provider_spec and resolved_provider.startswith("a1111"):
        from questfoundry.providers.factory import create_chat_model, get_default_model

        if "/" in llm_provider_spec:
            prov_name, model_name = llm_provider_spec.split("/", 1)
            if not prov_name or not model_name:
                console.print(
                    f"[red]Error:[/red] Invalid provider spec: '{llm_provider_spec}'. "
                    "Expected format 'provider/model'."
                )
                raise typer.Exit(1)
        else:
            prov_name = llm_provider_spec
            default_model = get_default_model(prov_name)
            if not default_model:
                console.print(
                    f"[red]Error:[/red] Provider '{prov_name}' requires an explicit model. "
                    f"Use --provider {prov_name}/<model>."
                )
                raise typer.Exit(1)
            model_name = default_model

        llm_model = create_chat_model(prov_name, model_name)
        log.info("distiller_llm_created", provider=prov_name, model=model_name)

    log.info(
        "generate_images_start",
        provider=resolved_provider,
        budget=image_budget,
        min_priority=min_priority,
        distiller_llm=llm_provider_spec,
    )

    stage = DressStage(
        project_path=project_path,
        image_provider=resolved_provider,
    )

    def _on_phase_progress(phase: str, status: str, detail: str | None) -> None:
        detail_str = f" ({detail})" if detail else ""
        if status == "completed":
            status_icon = "[green]✓[/green]"
        elif status == "failed":
            status_icon = "[red]✗[/red]"
        else:
            status_icon = "[yellow]○[/yellow]"
        console.print(f"  {status_icon} {phase}{detail_str}")

    # Warn if requesting briefs that were never generated
    from questfoundry.graph.graph import Graph

    try:
        _graph = Graph.load(project_path)
        brief_config = _graph.get_node("dress_meta::brief_config")
        if brief_config:
            dress_min = brief_config.get("min_priority", 3)
            if min_priority > dress_min:
                console.print(
                    f"[yellow]Warning:[/yellow] Requesting priority {min_priority} briefs, "
                    f"but dress was run with --min-priority {dress_min}. "
                    f"Priority {dress_min + 1}-{min_priority} briefs were not generated.\n"
                    f"Re-run [bold]qf dress --min-priority {min_priority}[/bold] to generate them."
                )
    except Exception as e:
        log.debug("priority_check_failed", error=str(e))

    console.print()
    console.print(f"[dim]Generating images with {resolved_provider}...[/dim]")

    try:
        result = asyncio.run(
            stage.run_generate_only(
                project_path,
                image_budget=image_budget,
                min_priority=min_priority,
                force=force,
                on_phase_progress=_on_phase_progress,
                model=llm_model,
            )
        )
    except DressStageError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print()
    if result.status == "failed":
        console.print(f"[red]✗[/red] Image generation failed: {result.detail}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Image generation complete: {result.detail}")
    if _log_enabled:
        console.print(f"  Logs: [dim]{project_path / 'logs'}[/dim]")


@app.command()
def run(
    to_stage: Annotated[
        str,
        typer.Option(
            "--to",
            "-t",
            help="Run stages up to and including this stage (dream, brainstorm, seed, grow, fill).",
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
    provider_creative: Annotated[
        str | None,
        typer.Option("--provider-creative", help="LLM provider for creative/discuss role"),
    ] = None,
    provider_balanced: Annotated[
        str | None,
        typer.Option("--provider-balanced", help="LLM provider for balanced/summarize role"),
    ] = None,
    provider_structured: Annotated[
        str | None,
        typer.Option("--provider-structured", help="LLM provider for structured/serialize role"),
    ] = None,
    provider_discuss: Annotated[
        str | None,
        typer.Option(
            "--provider-discuss", help="LLM provider for discuss phase (legacy)", hidden=True
        ),
    ] = None,
    provider_summarize: Annotated[
        str | None,
        typer.Option(
            "--provider-summarize", help="LLM provider for summarize phase (legacy)", hidden=True
        ),
    ] = None,
    provider_serialize: Annotated[
        str | None,
        typer.Option(
            "--provider-serialize", help="LLM provider for serialize phase (legacy)", hidden=True
        ),
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
    init: Annotated[
        bool,
        typer.Option("--init", help="Create project if it doesn't exist."),
    ] = False,
    image_provider: Annotated[
        str | None,
        typer.Option(
            "--image-provider",
            help="Image provider for DRESS stage (e.g., a1111, openai/gpt-image-1).",
        ),
    ] = None,
    image_budget: Annotated[
        int,
        typer.Option(
            "--image-budget",
            help="Max images to generate in DRESS stage (0=all selected briefs).",
        ),
    ] = 0,
    min_priority: MinPriorityOption = 3,
    two_step: Annotated[
        bool | None,
        typer.Option(
            "--two-step/--no-two-step",
            help="Two-step FILL prose generation. Defaults to fill.two_step in project.yaml.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Output language (ISO 639-1 code, e.g., nl, ja, de)"),
    ] = None,
) -> None:
    """Run multiple pipeline stages sequentially.

    Runs all stages from the starting point up to and including the
    target stage. By default, skips already-completed stages and uses
    non-interactive mode for batch execution.

    Examples:
        qf run --to seed --prompt "A mystery story"
        qf run --to brainstorm --from dream --force
        qf run --to seed --prompt "A mystery" --init --project my-story
    """
    project_path = _resolve_project_path(project)
    project_path = _ensure_project(project_path, auto_init=init, provider=provider)
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
        # SHIP is deterministic (no LLM) — handle separately
        if stage_name == "ship":
            try:
                _run_ship(project_path, export_format="twee")
            except typer.Exit:
                console.print()
                console.print("[red]Pipeline stopped at SHIP stage.[/red]")
                raise
            continue

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
                next_step_hint=next_step_hint,
                provider_discuss=provider_creative or provider_discuss,
                provider_summarize=provider_balanced or provider_summarize,
                provider_serialize=provider_structured or provider_serialize,
                image_provider=image_provider,
                image_budget=image_budget,
                min_priority=min_priority,
                two_step=two_step,
                language=language,
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

    status_icons = {
        "completed": "[green]✓[/green] completed",
        "pending": "[dim]○[/dim] pending",
        "failed": "[red]✗[/red] failed",
    }

    for stage_name, info in pipeline_status.stages.items():
        status_display = status_icons.get(info.status, info.status)
        table.add_row(stage_name, status_display)

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
    providers_ok, discovered_models = asyncio.run(_check_providers())
    all_ok &= providers_ok

    # Show available models if any providers connected
    if discovered_models:
        _show_available_models(discovered_models)

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

    log = get_logger(__name__)
    console.print("[bold]Configuration[/bold]")

    checks = [
        ("OLLAMA_HOST", os.getenv("OLLAMA_HOST")),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        ("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")),
        ("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY")),
        ("LANGSMITH_API_KEY", os.getenv("LANGSMITH_API_KEY")),
        ("A1111_HOST", os.getenv("A1111_HOST")),
    ]

    any_provider = False
    configured = []
    for name, value in checks:
        if value:
            # Mask secrets
            if "KEY" in name:
                display = f"{value[:7]}...{value[-3:]}" if len(value) > 10 else "(set)"
            else:
                display = value
            console.print(f"  [green]✓[/green] {name}: {display}")
            any_provider = True
            configured.append(name)
        else:
            console.print(f"  [dim]○[/dim] {name}: not configured")

    log.info("doctor_configuration", configured=configured, any_provider=any_provider)
    console.print()
    return any_provider  # At least one provider configured


async def _check_providers() -> tuple[bool, dict[str, list[str]]]:
    """Check provider connectivity and collect discovered models.

    Returns:
        Tuple of (all_ok, discovered_models) where discovered_models
        maps provider name to list of model identifiers.
    """
    import os

    log = get_logger(__name__)
    console.print("[bold]Provider Connectivity[/bold]")

    all_ok = True
    discovered: dict[str, list[str]] = {}
    skipped: list[str] = []

    # Check Ollama
    if os.getenv("OLLAMA_HOST"):
        ok, models = await _check_ollama()
        all_ok &= ok
        if models:
            discovered["ollama"] = models
    else:
        console.print("  [dim]○[/dim] ollama: Skipped (OLLAMA_HOST not set)")
        skipped.append("ollama")

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        ok, models = await _check_openai()
        all_ok &= ok
        if models:
            discovered["openai"] = models
    else:
        console.print("  [dim]○[/dim] openai: Skipped (not configured)")
        skipped.append("openai")

    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        ok, models = await _check_anthropic()
        all_ok &= ok
        if models:
            discovered["anthropic"] = models
    else:
        console.print("  [dim]○[/dim] anthropic: Skipped (not configured)")
        skipped.append("anthropic")

    # Check Google
    if os.getenv("GOOGLE_API_KEY"):
        ok, models = await _check_google()
        all_ok &= ok
        if models:
            discovered["google"] = models
    else:
        console.print("  [dim]○[/dim] google: Skipped (GOOGLE_API_KEY not set)")
        skipped.append("google")

    # Check A1111
    if os.getenv("A1111_HOST"):
        all_ok &= await _check_a1111()
    else:
        console.print("  [dim]○[/dim] a1111: Skipped (A1111_HOST not set)")
        skipped.append("a1111")

    log.info(
        "doctor_providers",
        all_ok=all_ok,
        discovered=list(discovered.keys()),
        skipped=skipped,
    )
    console.print()
    return all_ok, discovered


async def _check_ollama() -> tuple[bool, list[str]]:
    """Check Ollama connectivity and list models.

    Returns:
        Tuple of (connected, model_names).
    """
    import json
    import os

    import httpx

    log = get_logger(__name__)
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
                    log.info("doctor_ollama", status="connected", models=len(models))
                else:
                    console.print("  [yellow]![/yellow] ollama: Connected (no models pulled)")
                    log.warning("doctor_ollama", status="connected", models=0)
                return True, models
            else:
                console.print(f"  [red]✗[/red] ollama: HTTP {response.status_code}")
                log.error("doctor_ollama", status="http_error", code=response.status_code)
                return False, []
    except httpx.ConnectError:
        console.print(f"  [red]✗[/red] ollama: Connection refused ({host})")
        log.error("doctor_ollama", status="connection_refused", host=host)
        return False, []
    except httpx.TimeoutException:
        console.print(f"  [red]✗[/red] ollama: Connection timeout ({host})")
        log.error("doctor_ollama", status="timeout", host=host)
        return False, []
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] ollama: Request error - {e}")
        log.error("doctor_ollama", status="request_error", error=str(e))
        return False, []
    except json.JSONDecodeError:
        console.print("  [red]✗[/red] ollama: Invalid JSON response")
        log.error("doctor_ollama", status="invalid_json")
        return False, []


async def _check_openai() -> tuple[bool, list[str]]:
    """Check OpenAI API key validity and discover models.

    Returns:
        Tuple of (connected, model_names).
    """
    import json
    import os

    import httpx

    log = get_logger(__name__)
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                data = response.json()
                # Filter to chat models (skip embeddings, fine-tunes, etc.)
                all_models = [m.get("id", "") for m in data.get("data", [])]
                chat_prefixes = ("gpt-", "o1", "o3", "o4")
                models = sorted(m for m in all_models if m.startswith(chat_prefixes))
                count = len(models)
                console.print(
                    f"  [green]✓[/green] openai: Connected "
                    f"({count} chat model{'s' if count != 1 else ''} available)"
                )
                log.info("doctor_openai", status="connected", models=count)
                return True, models
            elif response.status_code == 401:
                console.print("  [red]✗[/red] openai: Invalid API key")
                log.error("doctor_openai", status="invalid_key")
                return False, []
            else:
                console.print(f"  [red]✗[/red] openai: HTTP {response.status_code}")
                log.error("doctor_openai", status="http_error", code=response.status_code)
                return False, []
    except httpx.TimeoutException:
        console.print("  [red]✗[/red] openai: Connection timeout")
        log.error("doctor_openai", status="timeout")
        return False, []
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] openai: Request error - {e}")
        log.error("doctor_openai", status="request_error", error=str(e))
        return False, []
    except json.JSONDecodeError:
        console.print("  [red]✗[/red] openai: Invalid JSON response")
        log.error("doctor_openai", status="invalid_json")
        return False, []


async def _check_anthropic() -> tuple[bool, list[str]]:
    """Check Anthropic API key validity and return known models.

    Anthropic doesn't have a public models list endpoint, so we make a
    cheap API call to validate the key, then return models from the
    KNOWN_MODELS registry.

    Returns:
        Tuple of (connected, model_names).
    """
    import os

    import httpx

    from questfoundry.providers.model_info import KNOWN_MODELS

    log = get_logger(__name__)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return False, []

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Intentionally malformed request (max_tokens=0) to validate the key.
    # A 400 error confirms connectivity and valid auth.
    data = {"model": "claude-3-haiku-20240307", "max_tokens": 0, "messages": []}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
            )

        if response.status_code == 400:
            console.print("  [green]✓[/green] anthropic: Connected (API key valid)")
            models = list(KNOWN_MODELS.get("anthropic", {}).keys())
            log.info("doctor_anthropic", status="connected", models=len(models))
            return True, models
        elif response.status_code in (401, 403):
            console.print("  [red]✗[/red] anthropic: Invalid API key")
            log.error("doctor_anthropic", status="invalid_key")
            return False, []
        else:
            console.print(f"  [red]✗[/red] anthropic: API error HTTP {response.status_code}")
            log.error("doctor_anthropic", status="http_error", code=response.status_code)
            return False, []
    except httpx.TimeoutException:
        console.print("  [red]✗[/red] anthropic: Connection timeout")
        log.error("doctor_anthropic", status="timeout")
        return False, []
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] anthropic: Request error - {e}")
        log.error("doctor_anthropic", status="request_error", error=str(e))
        return False, []


async def _check_google() -> tuple[bool, list[str]]:
    """Check Google Gemini API key validity and discover models.

    Returns:
        Tuple of (connected, model_names).
    """
    import json
    import os

    import httpx

    log = get_logger(__name__)
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key},
            )
            if response.status_code == 200:
                data = response.json()
                models = sorted(
                    m.get("name", "").removeprefix("models/")
                    for m in data.get("models", [])
                    if "gemini" in m.get("name", "").lower()
                )
                count = len(models)
                console.print(
                    f"  [green]✓[/green] google: Connected "
                    f"({count} Gemini model{'s' if count != 1 else ''} available)"
                )
                log.info("doctor_google", status="connected", models=count)
                return True, models
            elif response.status_code == 400:
                console.print("  [red]✗[/red] google: Invalid API key")
                log.error("doctor_google", status="invalid_key")
                return False, []
            else:
                console.print(f"  [red]✗[/red] google: HTTP {response.status_code}")
                log.error("doctor_google", status="http_error", code=response.status_code)
                return False, []
    except httpx.TimeoutException:
        console.print("  [red]✗[/red] google: Connection timeout")
        log.error("doctor_google", status="timeout")
        return False, []
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] google: Request error - {e}")
        log.error("doctor_google", status="request_error", error=str(e))
        return False, []
    except json.JSONDecodeError:
        console.print("  [red]✗[/red] google: Invalid JSON response")
        log.error("doctor_google", status="invalid_json")
        return False, []


async def _check_a1111() -> bool:
    """Check A1111 (Stable Diffusion WebUI) connectivity and active checkpoint."""
    import os

    import httpx

    log = get_logger(__name__)
    host = os.getenv("A1111_HOST", "").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{host}/sdapi/v1/options")
            if response.status_code == 200:
                data = response.json()
                checkpoint = data.get("sd_model_checkpoint", "unknown")
                console.print(f"  [green]✓[/green] a1111: Connected (checkpoint: {checkpoint})")
                log.info("doctor_a1111", status="connected", checkpoint=checkpoint)
                return True
            else:
                console.print(f"  [red]✗[/red] a1111: HTTP {response.status_code}")
                log.error("doctor_a1111", status="http_error", code=response.status_code)
                return False
    except httpx.ConnectError:
        console.print(f"  [red]✗[/red] a1111: Connection refused ({host})")
        log.error("doctor_a1111", status="connection_refused", host=host)
        return False
    except httpx.TimeoutException:
        console.print(f"  [red]✗[/red] a1111: Connection timeout ({host})")
        log.error("doctor_a1111", status="timeout", host=host)
        return False
    except httpx.RequestError as e:
        console.print(f"  [red]✗[/red] a1111: Request error - {e}")
        log.error("doctor_a1111", status="request_error", error=str(e))
        return False


def _show_available_models(discovered: dict[str, list[str]]) -> None:
    """Display discovered models with capabilities from KNOWN_MODELS registry.

    Args:
        discovered: Maps provider name to list of model identifiers.
    """
    from questfoundry.providers.model_info import KNOWN_MODELS

    console.print("[bold]Available Models[/bold]")

    table = Table(show_header=True, header_style="bold", pad_edge=False, box=None)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Context", justify="right", style="dim")
    table.add_column("Vision", justify="center")
    table.add_column("Tools", justify="center")
    table.add_column("Verbose", justify="center")
    table.add_column("Reason", justify="center")

    for provider in sorted(discovered.keys()):
        models = discovered[provider]
        known = KNOWN_MODELS.get(provider, {})

        for model_name in sorted(models):
            props = known.get(model_name)
            if props:
                ctx = _format_context_window(props.context_window)
                vision = "[green]✓[/green]" if props.supports_vision else "[dim]-[/dim]"
                tools = "[green]✓[/green]" if props.supports_tools else "[dim]-[/dim]"
                verbose = "[green]✓[/green]" if props.supports_verbosity else "[dim]-[/dim]"
                reason = "[green]✓[/green]" if props.supports_reasoning_effort else "[dim]-[/dim]"
            else:
                ctx = "[dim]?[/dim]"
                vision = "[dim]?[/dim]"
                tools = "[dim]?[/dim]"
                verbose = "[dim]?[/dim]"
                reason = "[dim]?[/dim]"

            table.add_row(provider, model_name, ctx, vision, tools, verbose, reason)

    console.print(table)
    console.print()


def _format_context_window(tokens: int) -> str:
    """Format token count as human-readable string.

    Args:
        tokens: Number of tokens.

    Returns:
        Formatted string (e.g., '32K', '128K', '1M').
    """
    if tokens >= 1_000_000:
        value = tokens // 1_000_000
        return f"{value:g}M"
    if tokens >= 1_000:
        value = tokens // 1_000
        return f"{value:g}K"
    return str(tokens)


def _check_project(project_path: Path) -> bool:
    """Check project configuration."""
    from questfoundry.pipeline.config import ProjectConfigError, load_project_config

    log = get_logger(__name__)
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
            log.info(
                "doctor_project",
                status="valid",
                name=config.name,
                provider=f"{config.provider.name}/{config.provider.model}",
            )
        except ProjectConfigError as e:
            console.print(f"  [red]✗[/red] Config error: {e}")
            log.error("doctor_project", status="config_error", error=str(e))
            all_ok = False
    else:
        console.print("  [dim]○[/dim] project.yaml: Not found (not in a project)")
        log.info("doctor_project", status="no_project")

    # Check artifacts directory
    artifacts_dir = project_path / "artifacts"
    if artifacts_dir.exists():
        artifact_count = len(list(artifacts_dir.glob("*.yaml")))
        console.print(f"  [green]✓[/green] Artifacts directory: {artifact_count} artifact(s)")
        log.info("doctor_artifacts", count=artifact_count)
    else:
        console.print("  [dim]○[/dim] Artifacts directory: Not found")

    console.print()
    return all_ok


class _GraphFormat(StrEnum):
    """Output format for the graph command."""

    dot = "dot"
    mermaid = "mermaid"
    json = "json"


@app.command(name="graph")
def graph_cmd(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    fmt: Annotated[
        _GraphFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format.",
        ),
    ] = _GraphFormat.dot,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file (stdout if not specified)."),
    ] = None,
    spine_only: Annotated[
        bool,
        typer.Option("--spine-only", help="Only show passages on the spine arc."),
    ] = False,
    no_labels: Annotated[
        bool,
        typer.Option("--no-labels", help="Omit choice labels on edges."),
    ] = False,
) -> None:
    """Visualize story graph as DOT, Mermaid, or JSON."""
    project_path = _resolve_project_path(project)
    _require_project(project_path)

    from questfoundry.graph.graph import Graph
    from questfoundry.visualization import build_story_graph, render_dot, render_mermaid

    graph = Graph.load(project_path)
    sg = build_story_graph(graph, spine_only=spine_only)

    if fmt == _GraphFormat.dot:
        result = render_dot(sg, no_labels=no_labels)
    elif fmt == _GraphFormat.mermaid:
        result = render_mermaid(sg, no_labels=no_labels)
    else:
        import dataclasses
        import json

        result = json.dumps(dataclasses.asdict(sg), indent=2)

    if output:
        output.write_text(result)
        console.print(f"Written to {output}")
    else:
        console.print(result, soft_wrap=True, markup=False, highlight=False)


@app.command()
def audit(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option("--stage", "-s", help="Filter by stage name."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option("--phase", help="Filter by phase name."),
    ] = None,
    operation: Annotated[
        str | None,
        typer.Option("--operation", "-o", help="Filter by operation type."),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Filter by target ID (substring match)."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of results."),
    ] = 50,
    summary: Annotated[
        bool,
        typer.Option("--summary", help="Show mutation count summary instead of details."),
    ] = False,
) -> None:
    """Query the graph mutation audit trail.

    Requires a SQLite-backed graph (graph.db). Shows mutations recorded
    during stage execution with stage, phase, operation, and target details.
    """
    project_path = _resolve_project_path(project)
    _require_project(project_path)

    db_path = project_path / "graph.db"
    if not db_path.exists():
        console.print(
            "[yellow]No graph.db found.[/yellow] "
            "Mutation audit is only available for SQLite-backed graphs.",
        )
        raise typer.Exit(1)

    import sqlite3

    from questfoundry.graph.audit import mutation_summary, query_mutations

    try:
        if summary:
            result = mutation_summary(db_path)
            console.print(f"\n[bold]Total mutations:[/bold] {result['total']}\n")

            if result["by_stage"]:
                stage_table = Table(title="By Stage")
                stage_table.add_column("Stage", style="cyan")
                stage_table.add_column("Count", justify="right")
                for s, cnt in result["by_stage"].items():
                    stage_table.add_row(s, str(cnt))
                console.print(stage_table)

            if result["by_operation"]:
                op_table = Table(title="By Operation")
                op_table.add_column("Operation", style="green")
                op_table.add_column("Count", justify="right")
                for op, cnt in result["by_operation"].items():
                    op_table.add_row(op, str(cnt))
                console.print(op_table)
        else:
            mutations = query_mutations(
                db_path,
                stage=stage,
                phase=phase,
                operation=operation,
                target=target,
                limit=limit,
            )

            if not mutations:
                console.print("[dim]No mutations found matching filters.[/dim]")
                return

            table = Table(title=f"Mutations ({len(mutations)} shown)")
            table.add_column("ID", style="dim", justify="right")
            table.add_column("Timestamp", style="dim")
            table.add_column("Stage", style="cyan")
            table.add_column("Phase", style="blue")
            table.add_column("Operation", style="green")
            table.add_column("Target", style="yellow")

            for m in mutations:
                table.add_row(
                    str(m["id"]),
                    m["timestamp"][:19] if m["timestamp"] else "",
                    m["stage"] or "",
                    m["phase"] or "",
                    m["operation"],
                    m["target_id"],
                )
            console.print(table)
    except sqlite3.Error as e:
        console.print(f"[red]Database error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def inspect(
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-p",
            help="Project directory. Can be a path or name (looks in --projects-dir).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of Rich tables."),
    ] = False,
) -> None:
    """Inspect project quality: prose, branching, coverage, and validation."""
    project_path = _resolve_project_path(project)
    _require_project(project_path)

    from questfoundry.inspection import inspect_project

    report = inspect_project(project_path)

    if json_output:
        import dataclasses
        import json

        # Emit machine-readable JSON. Rich will otherwise wrap long lines at the
        # terminal width (80), which can split string values and break JSON.
        console.print(
            json.dumps(dataclasses.asdict(report), indent=2),
            soft_wrap=True,
            markup=False,
            highlight=False,
        )
        return

    _render_inspection_report(report)


def _render_inspection_report(report: InspectionReport) -> None:
    """Render an InspectionReport using Rich tables and panels."""
    s = report.summary
    console.print()
    console.print(f"[bold]Project Inspection: {s.project_name}[/bold]")
    console.print(f"  Last stage: [cyan]{s.last_stage or 'none'}[/cyan]")
    console.print(f"  Nodes: [bold]{s.total_nodes}[/bold]  Edges: [bold]{s.total_edges}[/bold]")
    console.print()

    # Node counts table
    if s.node_counts:
        table = Table(title="Node Counts")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="bold", justify="right")
        for ntype, count in s.node_counts.items():
            table.add_row(ntype, str(count))
        console.print(table)
        console.print()

    # Prose stats
    if report.prose:
        p = report.prose
        console.print("[bold]Prose Quality[/bold]")
        console.print(
            f"  Passages: [bold]{p.passages_with_prose}[/bold]/{p.total_passages} with prose"
        )
        if p.passages_with_prose:
            console.print(
                f"  Words: [bold]{p.total_words}[/bold] total, "
                f"avg {p.avg_words:.0f}, range {p.min_words}-{p.max_words}"
            )
        if p.lexical_diversity is not None:
            console.print(f"  Lexical diversity: [bold]{p.lexical_diversity:.3f}[/bold]")
        if p.flagged_passages:
            for fp in p.flagged_passages:
                flag = fp.get("flag", "no prose")
                console.print(f"  [yellow]![/yellow] {fp['id']}: {flag}")
        console.print()

    # Branching stats
    if report.branching:
        b = report.branching
        console.print("[bold]Branching Structure[/bold]")
        console.print(
            f"  Choices: [bold]{b.total_choices}[/bold] "
            f"({b.meaningful_choices} meaningful, {b.contextual_choices} contextual, "
            f"{b.continue_choices} continue)"
        )
        console.print(
            f"  Dilemmas: [bold]{b.total_dilemmas}[/bold] "
            f"({b.fully_explored} fully explored, {b.partially_explored} partial)"
        )
        console.print(f"  Start passages: {b.start_passages}  Endings: {b.ending_passages}")
        console.print()

    # Branching quality score
    if report.branching_quality:
        q = report.branching_quality
        console.print("[bold]Branching Quality[/bold]")
        if q.policy_distribution:
            dist = ", ".join(f"{k}={v}" for k, v in q.policy_distribution.items())
            console.print(f"  Policy distribution: {dist}")
        console.print(f"  Avg exclusive beats per branch: [bold]{q.avg_exclusive_beats}[/bold]")
        console.print(f"  Meaningful choice ratio: [bold]{q.meaningful_choice_ratio}[/bold]")
        console.print(
            f"  Terminal passages: [bold]{q.terminal_count}[/bold]  "
            f"Ending variants: [bold]{q.ending_variants}[/bold]"
        )
        console.print()

    # Coverage stats
    c = report.coverage
    if c.entity_count:
        console.print("[bold]Coverage[/bold]")
        entity_types_str = ", ".join(f"{t}: {n}" for t, n in c.entity_types.items())
        console.print(f"  Entities: [bold]{c.entity_count}[/bold] ({entity_types_str})")
        console.print(
            f"  Codex: [bold]{c.codex_entries}[/bold] entries "
            f"across {c.entities_with_codex} entities"
        )
        console.print(
            f"  Illustrations: [bold]{c.illustration_briefs}[/bold] briefs, "
            f"{c.illustration_nodes} rendered, {c.asset_files} asset files"
        )
        console.print()

    # Prose neutrality
    if report.prose_neutrality:
        pn = report.prose_neutrality
        console.print("[bold]Prose Neutrality[/bold]")
        console.print(
            f"  Shared passages: [bold]{pn.shared_passages}[/bold]  "
            f"Routed: [bold]{pn.routed_passages}[/bold]"
        )
        if pn.unrouted_heavy:
            for pid in pn.unrouted_heavy:
                console.print(f"  [red]![/red] {pid}: heavy/high without routing")
        if pn.unrouted_light:
            for pid in pn.unrouted_light:
                console.print(f"  [yellow]![/yellow] {pid}: light without routing")
        if not pn.unrouted_heavy and not pn.unrouted_light:
            console.print("  [green]All shared passages satisfy prose-layer contracts[/green]")
        console.print()

    # Validation checks
    if report.validation_checks:
        severity_icons = {
            "pass": "[green]✓[/green]",
            "warn": "[yellow]![/yellow]",
            "fail": "[red]✗[/red]",
        }
        table = Table(title="Validation Checks")
        table.add_column("Check", style="cyan")
        table.add_column("", width=3)
        table.add_column("Message")
        for check in report.validation_checks:
            icon = severity_icons.get(check["severity"], "?")
            table.add_row(check["name"], icon, check.get("message", ""))
        console.print(table)
        console.print()


if __name__ == "__main__":
    app()
