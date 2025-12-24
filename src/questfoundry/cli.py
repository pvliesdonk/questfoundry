"""
QuestFoundry CLI - Interactive Fiction Studio.

Commands:
- doctor: Validate domain and check provider availability
- config show: Display resolved configuration
- roles: List agents with archetypes and capabilities
- projects: Manage story projects
- ask: Interactive session with agents
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from questfoundry.runtime import AgentRuntime
    from questfoundry.runtime.messaging import AsyncMessageBroker
    from questfoundry.runtime.models import Agent, Studio
    from questfoundry.runtime.observability import EventLogger, TracingManager
    from questfoundry.runtime.providers import LLMProvider, OllamaProvider
    from questfoundry.runtime.session import Session
    from questfoundry.runtime.storage import Project
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from questfoundry.cli_output import DelegationResultInfo, StatusReporter, create_log_handler

# Load .env file for environment variables (OLLAMA_HOST, API keys, etc.)
load_dotenv()

# Global verbosity level (set by callback)
_verbosity: int = 0

# Maximum recursion depth for orchestrator follow-up turns
MAX_ORCHESTRATOR_FOLLOW_UP_DEPTH = 10

# Main output console (stdout)
console = Console()

# Log console (stderr) - for logs, separated from user output
log_console = Console(stderr=True)

# Status reporter for structured output
_status_reporter: StatusReporter | None = None


def _configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Routes all logs to stderr via log_console, keeping stdout clean
    for structured status output.
    """
    global _verbosity, _status_reporter
    _verbosity = verbosity

    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:  # 2+
        level = logging.DEBUG

    # Configure root logger with Rich handler on STDERR
    # This keeps stdout clean for structured status output
    handler = create_log_handler(log_console, verbosity)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )

    # Also set questfoundry logger
    logging.getLogger("questfoundry").setLevel(level)

    # Filter noisy libraries at lower verbosities
    if verbosity < 2:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("langsmith").setLevel(logging.WARNING)

    # Create status reporter
    _status_reporter = StatusReporter(verbosity=verbosity, show_summary=True)


def _verbosity_callback(_ctx: typer.Context, value: int) -> int:
    """Process verbosity level and configure logging."""
    _configure_logging(value)
    return value


app = typer.Typer(
    name="qf",
    help="QuestFoundry - AI-powered interactive fiction studio",
    no_args_is_help=True,
)


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
    assert studio is not None  # Guaranteed by result.success

    console.print(f"[green]✓[/green] Domain: [bold]{studio.name}[/bold]")
    if studio.version:
        console.print(f"  Version: {studio.version}")

    # Count entities
    entry_agents = [a for a in studio.agents if a.is_entry_agent]
    entry_names = ", ".join(a.id for a in entry_agents)

    console.print(f"  {len(studio.agents)} agents (entry: {entry_names or 'none'})")
    console.print(
        f"  {len(studio.tools)} tools, {len(studio.stores)} stores, {len(studio.playbooks)} playbooks"
    )
    console.print(
        f"  {len(studio.artifact_types)} artifact types, {len(studio.asset_types)} asset types"
    )

    # Check providers
    console.print()
    console.print("[bold]Providers:[/bold]")

    for name, provider in config.providers.items():
        if provider.state == ProviderState.AVAILABLE:
            model = provider.default_model or "default"
            if provider.host:
                console.print(f"  [green]✓[/green] {name} @ {provider.host} ({model})")
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
    _domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """Show resolved configuration."""
    import yaml

    from questfoundry.runtime.config import load_config

    # Note: _domain reserved for future use
    config = load_config()

    # Build providers dict for display
    providers_dict: dict[str, dict[str, str | None]] = {}
    for name, provider in config.providers.items():
        providers_dict[name] = {
            "state": provider.state.value,
            "host": provider.host,
            "default_model": provider.default_model,
            # Don't show API keys
            "api_key": "***" if provider.api_key else None,
        }

    # Build config dict for display
    config_dict = {
        "domain_path": str(config.domain_path),
        "default_provider": config.default_provider,
        "providers": providers_dict,
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
    assert studio is not None  # Guaranteed by result.success

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

projects_app = typer.Typer(help="Manage story projects", no_args_is_help=True)
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
        console.print("[dim]Create one with: qf projects create <name>[/dim]")
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
        str | None,
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

    Project.create(
        path=project_path,
        name=name,
        description=description,
        studio_id=studio,
    )

    console.print(f"[green]✓[/green] Created project [bold]{name}[/bold]")
    console.print(f"  Path: {project_path}")
    console.print(f"  Studio: {studio}")


# =============================================================================
# Ask Command - Interactive Session
# =============================================================================


@app.command()
def ask(
    project: Annotated[
        str | None,
        typer.Argument(help="Project ID (auto-creates if doesn't exist)"),
    ] = None,
    prompt: Annotated[str | None, typer.Argument(help="Prompt (omit for REPL mode)")] = None,
    entry_agent: Annotated[
        str | None,
        typer.Option("--entry-agent", "-e", help="Entry agent ID"),
    ] = None,
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
    projects_dir: Annotated[
        Path,
        typer.Option("--projects-dir", help="Projects directory"),
    ] = Path("projects"),
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="LLM provider (e.g., ollama, anthropic)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use"),
    ] = None,
    log: Annotated[
        bool,
        typer.Option("--log", "-l", help="Write JSONL event log to project_dir/logs/events.jsonl"),
    ] = False,
    interactive: Annotated[
        bool | None,
        typer.Option(
            "--interactive/--no-interactive",
            "-i/-I",
            help="Enable/disable interactive mode (default: auto-detect TTY)",
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream/--no-stream",
            help="Enable/disable streaming output (default: streaming enabled)",
        ),
    ] = True,
    from_checkpoint: Annotated[
        str | None,
        typer.Option(
            "--from-checkpoint",
            "-c",
            help="Resume from checkpoint ID (use 'latest' for most recent)",
        ),
    ] = None,
    verbose: Annotated[  # noqa: ARG001 - callback sets global _verbosity
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (-v: info+tokens, -vv: debug+timing, -vvv: full prompts)",
            callback=_verbosity_callback,
        ),
    ] = 0,
) -> None:
    """Interactive session or single-shot query with an agent."""
    # Auto-create project if not specified or doesn't exist
    if project is None:
        project = "default"
    project_was_created = _ensure_project(project, projects_dir)

    # Determine interactive mode: explicit flag > TTY detection
    is_interactive = interactive if interactive is not None else sys.stdin.isatty()

    if prompt:
        asyncio.run(
            _ask_single(
                project,
                prompt,
                entry_agent,
                domain,
                projects_dir,
                provider,
                model,
                log,
                is_interactive,
                stream,
                from_checkpoint,
                project_was_created,
            )
        )
    else:
        asyncio.run(
            _ask_repl(
                project,
                entry_agent,
                domain,
                projects_dir,
                provider,
                model,
                log,
                is_interactive,
                stream,
                from_checkpoint,
                project_was_created,
            )
        )


def _ensure_project(project_id: str, projects_dir: Path) -> bool:
    """Create the project if it doesn't exist.

    Returns:
        True if a new project was created, False if it already existed.
    """
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    if not project_path.exists():
        # Generate a nice name from the ID
        name = project_id.replace("-", " ").replace("_", " ").title()
        if project_id == "default":
            name = "Default Project"
            description = "Auto-created default project for quick sessions"
        else:
            description = f"Auto-created project: {name}"

        Project.create(
            path=project_path,
            name=name,
            description=description,
            studio_id="questfoundry",
        )
        return True
    return False


async def _setup_runtime(
    project_id: str,
    entry_agent_id: str | None,
    domain_path: Path,
    projects_dir: Path,
    provider_name: str | None,
    model: str | None,
    log: bool = False,
    interactive: bool = True,
    from_checkpoint: str | None = None,
    project_created: bool = False,
) -> tuple[
    Project,
    Studio,
    AgentRuntime,
    Agent,
    Session,
    LLMProvider,
    EventLogger | None,
    Path | None,
    AsyncMessageBroker,
    TracingManager | None,
]:
    """
    Set up the runtime components for an ask session.

    Returns:
        Tuple of (project, studio, runtime, agent, session, provider, event_logger, log_path, broker, tracing_manager)
    """
    from questfoundry.runtime import AgentRuntime, load_studio
    from questfoundry.runtime.config import load_config
    from questfoundry.runtime.observability import EventLogger, TracingManager
    from questfoundry.runtime.providers import OllamaProvider
    from questfoundry.runtime.session import Session
    from questfoundry.runtime.storage import Project

    logger = logging.getLogger("questfoundry.cli")

    # Load configuration
    logger.debug("Loading configuration...")
    config = load_config()

    # Load project
    project_path = projects_dir / project_id
    logger.debug(f"Opening project at {project_path}")
    try:
        project = Project.open(project_path)
    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        console.print(f"[dim]Expected at: {project_path}[/dim]")
        raise typer.Exit(1) from None

    # Display project status
    if _status_reporter and project.info:
        if project_created:
            _status_reporter.project_created(project.info.name, project.path)
        else:
            summary = project.get_status_summary()
            _status_reporter.project_resumed(project.info.name, summary)

    # Load domain
    logger.debug(f"Loading domain from {domain_path}")
    result = await load_studio(domain_path)
    if not result.success or not result.studio:
        console.print("[red]✗ Failed to load domain[/red]")
        for error in result.errors:
            console.print(f"  [red]• {error.message}[/red]")
        project.close()
        raise typer.Exit(1)

    studio: Studio = result.studio
    logger.info(f"Loaded studio: {studio.name} ({len(studio.agents)} agents)")

    # Determine which provider to use
    selected_provider = provider_name or config.default_provider
    provider_config = config.providers.get(selected_provider)

    if not provider_config:
        console.print(f"[red]✗ Provider not found: {selected_provider}[/red]")
        available = list(config.providers.keys())
        console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        project.close()
        raise typer.Exit(1)

    model_to_use: str = model or provider_config.default_model or "qwen3:8b"

    # Create provider based on type
    from questfoundry.runtime.providers import OpenAIProvider

    provider: OllamaProvider | OpenAIProvider
    if selected_provider == "openai":
        logger.debug("Connecting to OpenAI API")
        try:
            provider = OpenAIProvider()
        except Exception as e:
            console.print(f"[red]✗ OpenAI provider error: {e}[/red]")
            project.close()
            raise typer.Exit(1) from None
    else:
        # Default to Ollama for ollama or any other provider
        host: str = provider_config.host or "http://localhost:11434"
        logger.debug(f"Connecting to {selected_provider} at {host}")
        provider = OllamaProvider(host=host)

    # Check provider availability
    if not await provider.check_availability():
        if selected_provider == "openai":
            console.print("[red]✗ OpenAI API not available[/red]")
            console.print("[dim]Check OPENAI_API_KEY environment variable[/dim]")
        else:
            host = provider_config.host or "http://localhost:11434"
            console.print(f"[red]✗ {selected_provider} not available at {host}[/red]")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
        project.close()
        raise typer.Exit(1)

    logger.info(f"Using provider: {selected_provider}, model: {model_to_use}")

    # Create event logger if logging enabled
    event_logger: EventLogger | None = None
    log_path: Path | None = None
    if log:
        # Log to project_dir/logs/events.jsonl
        log_path = project_path / "logs" / "events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event_logger = EventLogger(log_path, direct_file=True)
        logger.info(f"Event logging to: {log_path}")

    # Create LangSmith tracing manager (auto-detects LANGSMITH_TRACING env var)
    tracing_manager = TracingManager(project_name="questfoundry")
    if tracing_manager.enabled:
        logger.info("LangSmith tracing enabled")

    # Create message broker for delegation routing
    from questfoundry.runtime.messaging import AsyncMessageBroker

    broker = AsyncMessageBroker(project=project)
    logger.debug("Created message broker for delegation routing")

    # Create checkpoint manager for auto-checkpointing and resumption
    from questfoundry.runtime.checkpoint import CheckpointManager

    checkpoint_manager = CheckpointManager(project)
    logger.debug("Created checkpoint manager")

    # Get context size from provider (model-specific)
    context_limit = await provider.get_context_size(model_to_use)
    if context_limit:
        logger.info("Model %s context size: %d tokens", model_to_use, context_limit)
    else:
        logger.debug("Could not determine context size for %s, using default", model_to_use)

    # Create tool call callback for UI updates
    def on_tool_call(
        tool_id: str,
        success: bool,
        agent_id: str,
        turn_number: int | None,
        execution_time_ms: float | None,
        result: Any,
    ) -> None:
        """Callback from runtime when a tool is executed."""
        if _status_reporter:
            _status_reporter.tool_call(
                tool_id=tool_id,
                success=success,
                agent_id=agent_id,
                turn_number=turn_number or 0,
                execution_time_ms=execution_time_ms,
                result_data=result if isinstance(result, dict) else None,
            )
            # Track artifact creation from save_artifact tool
            if tool_id == "save_artifact" and success and result:
                _status_reporter.artifact_created(
                    artifact_id=result.get("artifact_id", "unknown"),
                    artifact_type=result.get("artifact", {}).get("_type", "unknown"),
                    store=result.get("store", "unknown"),
                    created_by=agent_id,
                    turn_number=turn_number or 0,
                    lifecycle_state=result.get("lifecycle_state", "draft"),
                )

    # Create runtime
    runtime = AgentRuntime(
        provider=provider,
        studio=studio,
        domain_path=domain_path,
        project=project,
        model=model_to_use,
        event_logger=event_logger,
        tracing_manager=tracing_manager if tracing_manager.enabled else None,
        broker=broker,
        interactive=interactive,
        checkpoint_manager=checkpoint_manager,
        context_limit=context_limit,
        on_tool_call=on_tool_call,
    )

    # Get entry agent
    if entry_agent_id:
        agent = runtime.get_agent(entry_agent_id)
        if not agent:
            console.print(f"[red]✗ Agent not found: {entry_agent_id}[/red]")
            available = [a.id for a in studio.agents]
            console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            project.close()
            raise typer.Exit(1)
    else:
        agent = runtime.get_entry_agent()
        if not agent:
            console.print("[red]✗ No entry agent defined in studio[/red]")
            project.close()
            raise typer.Exit(1)

    logger.info(f"Entry agent: {agent.name} ({agent.id})")

    # Handle checkpoint resumption or create new session
    if from_checkpoint:
        # Load checkpoint
        if from_checkpoint == "latest":
            checkpoint = checkpoint_manager.get_latest_checkpoint()
            if not checkpoint:
                console.print("[red]✗ No checkpoints found[/red]")
                project.close()
                raise typer.Exit(1)
        else:
            checkpoint = checkpoint_manager.load_checkpoint(from_checkpoint)
            if not checkpoint:
                console.print(f"[red]✗ Checkpoint not found: {from_checkpoint}[/red]")
                project.close()
                raise typer.Exit(1)

        console.print(f"[dim]Resuming from checkpoint: {checkpoint.id}[/dim]")
        console.print(
            f"[dim]  Session: {checkpoint.session_id}, Turn: {checkpoint.turn_number}[/dim]"
        )

        # Load the existing session
        session = Session.load(project=project, session_id=checkpoint.session_id)
        if not session:
            # Session not found - checkpoint is orphaned, cannot restore
            console.print(
                f"[red]✗ Session not found: {checkpoint.session_id}[/red]\n"
                "[dim]The checkpoint references a session that no longer exists in the database.[/dim]"
            )
            project.close()
            raise typer.Exit(1)
        logger.info(f"Restored session: {session.id}")

        # Restore checkpoint state (mailbox states, etc.)
        restore_info = await checkpoint_manager.restore_from_checkpoint(
            checkpoint=checkpoint,
            broker=broker,
        )
        logger.info(
            "Restored checkpoint: %d mailboxes, %d messages",
            restore_info["mailboxes_restored"],
            restore_info["messages_restored"],
        )

        # Restore context usage tracking to runtime
        if checkpoint.context_usage:
            runtime.restore_context_usage(checkpoint.context_usage)
    else:
        # Create new session
        session = Session.create(project=project, entry_agent=agent.id)
        logger.debug(f"Created session: {session.id}")

    # Log session start
    if event_logger:
        event_logger.session_start(session_id=session.id, agent_id=agent.id, project_id=project_id)

    return (
        project,
        studio,
        runtime,
        agent,
        session,
        provider,
        event_logger,
        log_path,
        broker,
        tracing_manager,
    )


async def _handle_orchestrator_follow_up(
    runtime: AgentRuntime,
    agent: Agent,
    session: Session,
    respond_fn: Callable[..., Awaitable[str]],
    current_depth: int,
) -> str:
    """
    Handle orchestrator follow-up turns after delegations complete.

    This is the hub-and-spoke pattern: after specialists return, the
    orchestrator gets follow-up turns to decide next steps.

    Args:
        runtime: Agent runtime
        agent: The orchestrator agent
        session: Current session
        respond_fn: Response function (_stream_response or _invoke_response)
        current_depth: Current recursion depth

    Returns:
        Additional content from follow-up turns (empty string if none)
    """
    logger = logging.getLogger("questfoundry.cli")

    if current_depth >= MAX_ORCHESTRATOR_FOLLOW_UP_DEPTH:
        logger.warning(
            "Max orchestrator follow-up depth (%d) reached, stopping recursion",
            MAX_ORCHESTRATOR_FOLLOW_UP_DEPTH,
        )
        return ""

    if not runtime._is_orchestrator(agent):
        return ""

    responses = await runtime.get_delegation_responses(agent.id)
    if not responses:
        return ""

    # Display structured results from each delegation
    if _status_reporter and responses:
        console.print()
        console.print("[dim]─── Delegation Results ───[/dim]")
        for msg in responses:
            payload = msg.payload or {}
            success = payload.get("success", False)
            result_data = payload.get("result", {})

            # Extract structured fields from return_to_orchestrator
            task_completion = result_data.get(
                "task_completion", "completed" if success else "failed"
            )
            result_info = result_data.get("result", {})
            assessment = result_info.get("assessment", "info")
            recommendation = result_data.get("recommendation", "proceed")
            summary = result_info.get("summary") or result_data.get("summary", "No summary")

            # Artifacts
            artifacts = payload.get("artifacts_produced", [])
            ready_for_review = result_data.get("artifacts_ready_for_review", [])

            # Blockers/issues
            details = result_info.get("details", [])
            blockers = []
            for d in details:
                if isinstance(d, dict):
                    blockers.append(d.get("description", str(d)))
                else:
                    blockers.append(str(d))

            result_info_obj = DelegationResultInfo(
                from_agent=msg.from_agent,
                task_completion=task_completion,
                assessment=assessment,
                recommendation=recommendation,
                summary=summary,
                artifacts_produced=artifacts,
                artifacts_ready_for_review=ready_for_review,
                blockers=blockers,
            )
            _status_reporter.delegation_result(result_info_obj)
        console.print()

    # Build follow-up prompt with delegation results
    follow_up_prompt = runtime.build_delegation_response_prompt(responses)
    logger.debug(
        "Orchestrator %s has %d delegation responses, giving follow-up turn (depth %d)",
        agent.id,
        len(responses),
        current_depth + 1,
    )

    # Report follow-up turn start (turn number will be updated after activation)
    if _status_reporter:
        _status_reporter.turn_start(
            turn_number=session.turn_count + 1,
            agent_id=agent.id,
            agent_name=agent.name,
        )

    # Recursively invoke the follow-up response with incremented depth
    follow_up_content = await respond_fn(
        runtime,
        agent,
        follow_up_prompt,
        session,
        process_delegations=True,
        _follow_up_depth=current_depth + 1,
    )
    return "\n\n" + follow_up_content


async def _stream_response(
    runtime: AgentRuntime,
    agent: Agent,
    user_input: str,
    session: Session,
    process_delegations: bool = True,
    _follow_up_depth: int = 0,
) -> str:
    """Stream a response from the agent and return full content."""
    import time

    from rich.live import Live
    from rich.text import Text

    logger = logging.getLogger("questfoundry.cli")

    full_content = ""
    start_time = time.time()
    final_usage = None

    # Show prompt at -vvv
    if _verbosity >= 3:
        context = runtime.build_context(agent)
        messages = runtime.build_messages(agent, user_input, context)
        console.print()
        console.print("[dim]─── Prompt ───[/dim]")
        for msg in messages:
            role_color = {"system": "yellow", "user": "green", "assistant": "blue"}.get(
                msg.role, "white"
            )
            console.print(f"[{role_color}]{msg.role}:[/{role_color}]")
            # Truncate long system prompts
            content = msg.content
            if msg.role == "system" and len(content) > 2000:
                content = content[:2000] + f"\n... ({len(msg.content)} chars total)"
            console.print(f"[dim]{content}[/dim]")
            console.print()
        console.print("[dim]─── Response ───[/dim]")

    with Live(Text(""), console=console, refresh_per_second=10) as live:
        async for chunk in runtime.activate_streaming(agent, user_input, session):
            if chunk.done:
                final_usage = chunk
            else:
                full_content += chunk.content
                live.update(Text(full_content))

    duration_ms = (time.time() - start_time) * 1000

    # Get turn number from session (turn was created during streaming)
    turn_number = session.turn_count

    # Build usage from streaming chunk for turn completion
    from questfoundry.runtime.session import TokenUsage

    usage = None
    if final_usage:
        usage = TokenUsage(
            prompt_tokens=final_usage.prompt_tokens,
            completion_tokens=final_usage.completion_tokens,
            total_tokens=final_usage.total_tokens,
        )

    # Report turn completion via status reporter
    if _status_reporter:
        _status_reporter.turn_complete(usage=usage, duration_ms=duration_ms)

    # Show token usage at -v+ (to stderr)
    if _verbosity >= 1 and final_usage:
        tokens_info = []
        if final_usage.prompt_tokens:
            tokens_info.append(f"prompt: {final_usage.prompt_tokens}")
        if final_usage.completion_tokens:
            tokens_info.append(f"completion: {final_usage.completion_tokens}")
        if final_usage.total_tokens:
            tokens_info.append(f"total: {final_usage.total_tokens}")
        if tokens_info:
            log_console.print(f"[dim]tokens: {', '.join(tokens_info)}[/dim]")

    # Show timing at -vv+ (to stderr)
    if _verbosity >= 2:
        log_console.print(f"[dim]time: {duration_ms:.0f}ms | model: {runtime._model}[/dim]")
        logger.debug(f"Response generated in {duration_ms:.0f}ms")

    # Process any pending delegations and handle orchestrator follow-up
    if process_delegations:
        # Track turn count before delegations to detect new turns
        turn_count_before = session.turn_count

        delegation_results = await runtime.process_pending_delegations(session)
        if delegation_results:
            for dr in delegation_results:
                to_agent = dr.get("to_agent", "unknown")
                task = dr.get("task", "")
                success = dr.get("success", False)

                # Report via status reporter
                if _status_reporter:
                    _status_reporter.delegation_complete(
                        from_agent=agent.id,
                        to_agent=to_agent,
                        task=task,
                        success=success,
                        turn_number=turn_number,
                        error=dr.get("error") if not success else None,
                    )

            # Report delegated agent turns to StatusReporter for summary table
            if _status_reporter:
                for turn in session.turns[turn_count_before:]:
                    delegated_agent = runtime.get_agent(turn.agent_id)
                    if delegated_agent:
                        _status_reporter.turn_start(
                            turn_number=turn.turn_number,
                            agent_id=turn.agent_id,
                            agent_name=delegated_agent.name,
                        )
                        _status_reporter.turn_complete(usage=turn.usage)

        # Hub-and-spoke: Give orchestrator follow-up turns after delegations complete
        follow_up_content = await _handle_orchestrator_follow_up(
            runtime, agent, session, _stream_response, _follow_up_depth
        )
        full_content += follow_up_content

    # Check if session should be marked as completed
    # For streaming, we check if the current turn called a session-terminating tool
    current_turn = session.current_turn
    if current_turn:
        for tc in current_turn.tool_calls:
            if tc.tool_id == "terminate_session" and tc.success:
                session.complete()
                break

    return full_content


async def _invoke_response(
    runtime: AgentRuntime,
    agent: Agent,
    user_input: str,
    session: Session,
    process_delegations: bool = True,
    _follow_up_depth: int = 0,
) -> str:
    """Invoke agent without streaming (single response)."""
    import time

    logger = logging.getLogger("questfoundry.cli")

    start_time = time.time()

    # Show prompt at -vvv
    if _verbosity >= 3:
        context = runtime.build_context(agent)
        messages = runtime.build_messages(agent, user_input, context)
        console.print()
        console.print("[dim]─── Prompt ───[/dim]")
        for msg in messages:
            role_color = {"system": "yellow", "user": "green", "assistant": "blue"}.get(
                msg.role, "white"
            )
            console.print(f"[{role_color}]{msg.role}:[/{role_color}]")
            content = msg.content
            if msg.role == "system" and len(content) > 2000:
                content = content[:2000] + f"\n... ({len(msg.content)} chars total)"
            console.print(f"[dim]{content}[/dim]")
            console.print()
        console.print("[dim]─── Response ───[/dim]")

    # Non-streaming activation
    result = await runtime.activate(agent, user_input, session)
    full_content = result.content

    # Print the response
    console.print(full_content)

    duration_ms = (time.time() - start_time) * 1000

    # Note: Tool calls and artifacts are now reported via the on_tool_call callback
    # which is called during runtime._execute_tool_calls()

    # Report turn completion
    if _status_reporter:
        _status_reporter.turn_complete(usage=result.usage, duration_ms=duration_ms)

    # Show token usage at -v+ (also goes to log_console now)
    if _verbosity >= 1 and result.usage:
        tokens_info = []
        if result.usage.prompt_tokens:
            tokens_info.append(f"prompt: {result.usage.prompt_tokens}")
        if result.usage.completion_tokens:
            tokens_info.append(f"completion: {result.usage.completion_tokens}")
        if result.usage.total_tokens:
            tokens_info.append(f"total: {result.usage.total_tokens}")
        if tokens_info:
            log_console.print(f"[dim]tokens: {', '.join(tokens_info)}[/dim]")

    # Show timing at -vv+
    if _verbosity >= 2:
        log_console.print(f"[dim]time: {duration_ms:.0f}ms | model: {runtime._model}[/dim]")
        logger.debug(f"Response generated in {duration_ms:.0f}ms")

    # Process any pending delegations and handle orchestrator follow-up
    if process_delegations:
        # Track turn count before delegations to detect new turns
        turn_count_before = session.turn_count

        delegation_results = await runtime.process_pending_delegations(session)
        if delegation_results:
            for dr in delegation_results:
                to_agent = dr.get("to_agent", "unknown")
                task = dr.get("task", "")
                success = dr.get("success", False)

                # Report via status reporter
                if _status_reporter:
                    _status_reporter.delegation_complete(
                        from_agent=agent.id,
                        to_agent=to_agent,
                        task=task,
                        success=success,
                        turn_number=result.turn.turn_number,
                        error=dr.get("error") if not success else None,
                    )

            # Report delegated agent turns to StatusReporter for summary table
            if _status_reporter:
                for turn in session.turns[turn_count_before:]:
                    delegated_agent = runtime.get_agent(turn.agent_id)
                    if delegated_agent:
                        _status_reporter.turn_start(
                            turn_number=turn.turn_number,
                            agent_id=turn.agent_id,
                            agent_name=delegated_agent.name,
                        )
                        _status_reporter.turn_complete(usage=turn.usage)

        # Hub-and-spoke: Give orchestrator follow-up turns after delegations complete
        follow_up_content = await _handle_orchestrator_follow_up(
            runtime, agent, session, _invoke_response, _follow_up_depth
        )
        full_content += follow_up_content

    # Check if session should be marked as completed
    if result.session_terminated:
        session.complete()

    return full_content


async def _handle_clarification_requests(
    broker: AsyncMessageBroker,
) -> list[dict[str, Any]]:
    """
    Check for and handle customer-facing messages from agents.

    Handles all message types from the communicate tool:
    - PROGRESS_UPDATE (status): Displayed immediately
    - COMPLETION_SIGNAL (notification): Displayed immediately
    - ESCALATION (error): Displayed immediately
    - CLARIFICATION_REQUEST (question): Prompts for user input

    Returns list of clarification requests that were found (for logging/debugging).
    The requests are handled interactively - user input is collected and responses sent.
    """
    from questfoundry.runtime.messaging import create_message
    from questfoundry.runtime.messaging.types import MessageType

    logger = logging.getLogger("questfoundry.cli")

    # Get the "customer" mailbox (special mailbox for human communication)
    mailbox = await broker.get_mailbox("customer")
    pending = await mailbox.get_all_pending()

    if not pending:
        return []

    # First, display all non-interactive messages (status, notification, error)
    for msg in pending:
        payload = msg.payload
        message_text = payload.get("message", "")
        from_agent = msg.from_agent

        if msg.type == MessageType.PROGRESS_UPDATE:
            # Status update - display inline
            console.print(f"[dim][{from_agent}][/dim] {message_text}")

        elif msg.type == MessageType.COMPLETION_SIGNAL:
            # Notification - display with highlight
            artifacts = payload.get("artifacts", [])
            console.print()
            console.print(
                Panel(
                    message_text,
                    title=f"[green]📢 {from_agent}[/green]",
                    border_style="green",
                )
            )
            if artifacts:
                console.print("[dim]Artifacts:[/dim]", ", ".join(artifacts))

        elif msg.type == MessageType.ESCALATION:
            # Error - display with appropriate severity styling
            severity = payload.get("severity", "info")
            style = {"info": "blue", "warning": "yellow", "error": "red"}.get(severity, "white")
            console.print()
            console.print(
                Panel(
                    message_text,
                    title=f"[{style}]⚠️ {from_agent} ({severity})[/{style}]",
                    border_style=style,
                )
            )

    # Now filter for interactive messages (questions)
    clarification_requests = [
        msg for msg in pending if msg.type == MessageType.CLARIFICATION_REQUEST
    ]

    if not clarification_requests:
        return []

    handled = []

    for request in clarification_requests:
        payload = request.payload
        # communicate tool uses "message" field for the question text
        question = payload.get("message", "") or payload.get("question", "")
        context = payload.get("context")
        options = payload.get("options", [])
        default_option = payload.get("default_option")

        # Display the clarification request
        console.print()
        console.print(
            Panel(
                f"[bold]{question}[/bold]",
                title=f"[yellow]Question from {request.from_agent}[/yellow]",
                border_style="yellow",
            )
        )

        if context:
            console.print(f"[dim]Context: {context}[/dim]")

        # Display options if provided
        if options:
            console.print()
            console.print("[bold]Options:[/bold]")
            for i, opt in enumerate(options, 1):
                opt_id = opt.get("id", str(i))
                opt_desc = opt.get("description", "")
                opt_impl = opt.get("implications", "")
                marker = " [dim](recommended)[/dim]" if opt_id == default_option else ""
                console.print(f"  [{i}] [cyan]{opt_id}[/cyan]: {opt_desc}{marker}")
                if opt_impl:
                    console.print(f"      [dim]→ {opt_impl}[/dim]")
            console.print()
            console.print("[dim]Enter option number, option ID, or type your own answer[/dim]")

        # Capture user response
        try:
            user_response = console.input("[bold yellow]Answer> [/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            user_response = ""

        # Resolve option selection if applicable
        selected_option = None
        answer = user_response

        if options and user_response.strip():
            # Check if user entered a number
            if user_response.strip().isdigit():
                idx = int(user_response.strip()) - 1
                if 0 <= idx < len(options):
                    selected_option = options[idx].get("id")
                    answer = options[idx].get("description", selected_option)

            # Check if user entered an option ID
            elif not selected_option:
                for opt in options:
                    if opt.get("id", "").lower() == user_response.strip().lower():
                        selected_option = opt.get("id")
                        answer = opt.get("description", selected_option)
                        break

        # Use default if no response and default exists
        if not user_response.strip() and default_option:
            selected_option = default_option
            for opt in options:
                if opt.get("id") == default_option:
                    answer = opt.get("description", default_option)
                    break
            console.print(f"[dim]Using default: {default_option}[/dim]")

        # Build response payload
        response_payload: dict[str, Any] = {
            "answer": answer,
            "question": question,  # Echo back for context
        }
        if selected_option:
            response_payload["selected_option"] = selected_option

        # Send clarification response back to the agent
        response_msg = create_message(
            message_type=MessageType.CLARIFICATION_RESPONSE,
            from_agent="customer",
            to_agent=request.from_agent,
            payload=response_payload,
            correlation_id=request.correlation_id,
            in_reply_to=request.id,
        )

        await broker.send(response_msg)

        logger.info(
            "Clarification response sent to %s: %s",
            request.from_agent,
            answer[:50] + "..." if len(answer) > 50 else answer,
        )

        # Mark request as processed (remove from mailbox)
        # The get_all_pending already removed them; we just don't put them back

        handled.append(
            {
                "from_agent": request.from_agent,
                "question": question,
                "answer": answer,
                "selected_option": selected_option,
            }
        )

    return handled


async def _ask_single(
    project_id: str,
    prompt: str,
    entry_agent_id: str | None,
    domain_path: Path,
    projects_dir: Path,
    provider_name: str | None,
    model: str | None,
    log: bool = False,
    interactive: bool = True,
    stream: bool = True,
    from_checkpoint: str | None = None,
    project_created: bool = False,
) -> None:
    """Execute a single-shot query."""
    from questfoundry.runtime.providers import ContextOverflowError, ProviderError

    (
        project,
        _studio,
        runtime,
        agent,
        session,
        provider,
        event_logger,
        log_path,
        broker,
        tracing_manager,
    ) = await _setup_runtime(
        project_id,
        entry_agent_id,
        domain_path,
        projects_dir,
        provider_name,
        model,
        log,
        interactive,
        from_checkpoint,
        project_created,
    )

    # Select response function based on streaming preference
    respond = _stream_response if stream else _invoke_response

    # Report session start via status reporter
    if _status_reporter:
        _status_reporter.session_start(
            session_id=session.id,
            project_id=project_id,
            agent_name=agent.name,
            agent_id=agent.id,
            model=runtime._model,
        )

    # Wrap execution in LangSmith session tracing
    with (
        tracing_manager.session(session.id, agent.id, project_id)
        if tracing_manager
        else nullcontext()
    ):
        try:
            # Report turn start
            if _status_reporter:
                _status_reporter.turn_start(
                    turn_number=session.turn_count + 1,
                    agent_id=agent.id,
                    agent_name=agent.name,
                )

            await respond(runtime, agent, prompt, session)
            console.print()

            # Handle clarification requests in a loop
            # Agent may ask multiple questions before completing
            while True:
                clarifications = await _handle_clarification_requests(broker)
                if not clarifications:
                    break

                # Re-activate agent with the clarification response context
                # The response was sent to agent's mailbox - agent will receive it
                for clarification in clarifications:
                    from_agent = clarification["from_agent"]
                    answer = clarification["answer"]

                    console.print()
                    console.print(f"[dim]{from_agent} (continuing):[/dim]")

                    # Build a context message for the agent with the clarification response
                    context_msg = f"[Customer response to your question: {answer}]"
                    await respond(runtime, agent, context_msg, session)
                    console.print()

        except ContextOverflowError as e:
            console.print(f"[red]✗ Context overflow: {e}[/red]")
        except ProviderError as e:
            console.print(f"[red]✗ Provider error: {e}[/red]")
        finally:
            # End session tracing
            if tracing_manager:
                tracing_manager.end_session(turn_count=session.turn_count)
            # Log session end and flush
            if event_logger:
                event_logger.session_complete(session_id=session.id, turn_count=session.turn_count)
                if _verbosity >= 1:
                    log_console.print(f"[dim]Events logged to: {log_path}[/dim]")
            # Report session end with summary
            if _status_reporter:
                _status_reporter.session_end(turn_count=session.turn_count)
            await provider.close()
            project.close()


async def _ask_repl(
    project_id: str,
    entry_agent_id: str | None,
    domain_path: Path,
    projects_dir: Path,
    provider_name: str | None,
    model: str | None,
    log: bool = False,
    interactive: bool = True,
    stream: bool = True,
    from_checkpoint: str | None = None,
    project_created: bool = False,
) -> None:
    """Run interactive REPL mode."""
    from questfoundry.runtime.providers import ContextOverflowError, ProviderError
    from questfoundry.runtime.session.turn import TurnStatus

    # Select response function based on streaming preference
    respond = _stream_response if stream else _invoke_response

    (
        project,
        _studio,
        runtime,
        agent,
        session,
        provider,
        event_logger,
        log_path,
        broker,
        tracing_manager,
    ) = await _setup_runtime(
        project_id,
        entry_agent_id,
        domain_path,
        projects_dir,
        provider_name,
        model,
        log,
        interactive,
        from_checkpoint,
        project_created,
    )

    # Report session start via status reporter
    if _status_reporter:
        _status_reporter.session_start(
            session_id=session.id,
            project_id=project_id,
            agent_name=agent.name,
            agent_id=agent.id,
            model=runtime._model,
        )
    else:
        # Fallback to old-style header if no status reporter
        console.print()
        console.print("[bold]QuestFoundry Interactive Session[/bold]")
        console.print(f"Project: [cyan]{project_id}[/cyan]")
        console.print(f"Agent: [green]{agent.name}[/green] ({agent.id})")

    # Show extra debug info
    if _verbosity >= 1:
        if log_path:
            log_console.print(f"[dim]Logging: {log_path}[/dim]")
        if tracing_manager:
            log_console.print("[dim]LangSmith tracing: enabled[/dim]")
    console.print("[dim]Type 'exit' or Ctrl+D to quit[/dim]")
    console.print()

    # Wrap execution in LangSmith session tracing
    with (
        tracing_manager.session(session.id, agent.id, project_id)
        if tracing_manager
        else nullcontext()
    ):
        try:
            while True:
                try:
                    # Get input
                    user_input = console.input("[bold]> [/bold]")

                    # Check for exit
                    if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                        break

                    if not user_input.strip():
                        continue

                    # Report turn start
                    if _status_reporter:
                        _status_reporter.turn_start(
                            turn_number=session.turn_count + 1,
                            agent_id=agent.id,
                            agent_name=agent.name,
                        )

                    # Get response (streaming or non-streaming based on flag)
                    await respond(runtime, agent, user_input, session)
                    console.print()

                    # Handle clarification requests
                    # Agent may ask questions before completing
                    while True:
                        clarifications = await _handle_clarification_requests(broker)
                        if not clarifications:
                            break

                        # Re-activate agent with the clarification response context
                        for clarification in clarifications:
                            from_agent = clarification["from_agent"]
                            answer = clarification["answer"]

                            console.print()
                            console.print(f"[dim]{from_agent} (continuing):[/dim]")

                            # Build context message for the agent
                            context_msg = f"[Customer response to your question: {answer}]"
                            await respond(runtime, agent, context_msg, session)
                            console.print()

                except ContextOverflowError as e:
                    console.print(f"[red]✗ Context overflow: {e}[/red]")
                    console.print("[dim]Try a shorter message or start a new session[/dim]")
                    console.print()
                except ProviderError as e:
                    console.print(f"[red]✗ Provider error: {e}[/red]")
                    console.print()

        except (KeyboardInterrupt, EOFError):
            # Cancel any in-progress turn
            current = session.current_turn
            if current and current.status == TurnStatus.STREAMING:
                session.cancel_turn(current, "Interrupted by user")
            console.print("\n[dim]Cancelled[/dim]")

        finally:
            # End session tracing
            if tracing_manager:
                tracing_manager.end_session(turn_count=session.turn_count)
            # Log session end and flush
            if event_logger:
                event_logger.session_complete(session_id=session.id, turn_count=session.turn_count)
                if _verbosity >= 1:
                    log_console.print(f"[dim]Events logged to: {log_path}[/dim]")
            # Report session end with summary
            if _status_reporter:
                _status_reporter.session_end(turn_count=session.turn_count)
            else:
                console.print()
                console.print(f"[dim]Session ended: {session.turn_count} turns[/dim]")
            await provider.close()
            project.close()


# =============================================================================
# Projects Subcommand
# =============================================================================


# =============================================================================
# Artifacts Subcommand
# =============================================================================

artifacts_app = typer.Typer(help="View artifacts in a project", no_args_is_help=True)
app.add_typer(artifacts_app, name="artifacts")


@artifacts_app.command("list")
def artifacts_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    artifact_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by artifact type"),
    ] = None,
    store: Annotated[
        str | None,
        typer.Option("--store", "-s", help="Filter by store"),
    ] = None,
    lifecycle_state: Annotated[
        str | None,
        typer.Option("--state", help="Filter by lifecycle state"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum artifacts to show"),
    ] = 50,
) -> None:
    """List artifacts in a project."""
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)

        artifacts = project.query_artifacts(
            artifact_type=artifact_type,
            store=store,
            lifecycle_state=lifecycle_state,
            limit=limit,
        )

        if not artifacts:
            console.print("[dim]No artifacts found[/dim]")
            return

        table = Table(title=f"Artifacts in {project_id}")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Store")
        table.add_column("State")
        table.add_column("Version")
        table.add_column("Updated")
        table.add_column("Created By", style="dim")

        for artifact in artifacts:
            updated = artifact.get("_updated_at", "")
            if updated:
                # Shorten timestamp
                updated = updated[:16].replace("T", " ")

            table.add_row(
                artifact.get("_id", "-"),
                artifact.get("_type", "-"),
                artifact.get("_store", "-"),
                artifact.get("_lifecycle_state", "-"),
                str(artifact.get("_version", 1)),
                updated,
                artifact.get("_created_by") or "-",
            )

        console.print(table)
        console.print(f"[dim]Showing {len(artifacts)} artifact(s)[/dim]")

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


@artifacts_app.command("show")
def artifacts_show(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    artifact_id: Annotated[str, typer.Argument(help="Artifact ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    version: Annotated[
        int | None,
        typer.Option("--version", "-v", help="Show specific version"),
    ] = None,
) -> None:
    """Show an artifact's details."""
    import json

    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)

        if version is not None:
            # Get specific version
            version_data = project.get_artifact_at_version(artifact_id, version)
            if not version_data:
                console.print(f"[red]✗ Version {version} not found for: {artifact_id}[/red]")
                raise typer.Exit(1)

            console.print(Panel(f"[dim]Version {version}[/dim]", title=f"Artifact: {artifact_id}"))
            console.print(f"[bold]Created At:[/bold] {version_data.get('created_at') or '-'}")
            console.print(f"[bold]Created By:[/bold] {version_data.get('created_by') or '-'}")
            console.print()
            console.print("[bold]Data:[/bold]")
            console.print(json.dumps(version_data.get("data", {}), indent=2))
        else:
            # Get current artifact
            artifact = project.get_artifact(artifact_id)
            if not artifact:
                console.print(f"[red]✗ Artifact not found: {artifact_id}[/red]")
                raise typer.Exit(1)

            # Separate system fields from user data
            user_data = {k: v for k, v in artifact.items() if not k.startswith("_")}

            console.print(Panel(f"[cyan]{artifact_id}[/cyan]", title="Artifact"))

            # System metadata
            console.print("[bold]Metadata:[/bold]")
            console.print(f"  Type: [green]{artifact.get('_type') or '-'}[/green]")
            console.print(f"  Store: {artifact.get('_store') or '-'}")
            console.print(f"  State: {artifact.get('_lifecycle_state') or '-'}")
            console.print(f"  Version: {artifact.get('_version', 1)}")
            console.print(f"  Created: {artifact.get('_created_at') or '-'}")
            console.print(f"  Updated: {artifact.get('_updated_at') or '-'}")
            console.print(f"  Created By: {artifact.get('_created_by') or '-'}")
            console.print()

            # User data
            console.print("[bold]Data:[/bold]")
            console.print(json.dumps(user_data, indent=2))

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


@artifacts_app.command("versions")
def artifacts_versions(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    artifact_id: Annotated[str, typer.Argument(help="Artifact ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum versions to show"),
    ] = 20,
) -> None:
    """Show version history for an artifact."""
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)

        # Check artifact exists
        artifact = project.get_artifact(artifact_id)
        if not artifact:
            console.print(f"[red]✗ Artifact not found: {artifact_id}[/red]")
            raise typer.Exit(1)

        versions = project.get_artifact_versions(artifact_id, limit=limit)

        if not versions:
            console.print(f"[dim]No version history for: {artifact_id}[/dim]")
            console.print(f"[dim]Current version: {artifact.get('_version', 1)}[/dim]")
            return

        table = Table(title=f"Version History: {artifact_id}")
        table.add_column("Version", style="cyan")
        table.add_column("Created At")
        table.add_column("Created By")

        for ver in versions:
            table.add_row(
                str(ver.get("version", "-")),
                ver.get("created_at") or "-",
                ver.get("created_by") or "-",
            )

        console.print(table)
        console.print(f"[dim]Current version: {artifact.get('_version', 1)}[/dim]")
        console.print(
            f"[dim]Use 'qf artifacts show {project_id} {artifact_id} --version N' "
            "to view a version[/dim]"
        )

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


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
        raise typer.Exit(1) from None

    info = project.info
    if not info:
        console.print("[red]✗ Could not load project info[/red]")
        raise typer.Exit(1)

    panel_content = f"""
[bold]Name:[/bold] {info.name}
[bold]ID:[/bold] {info.id}
[bold]Description:[/bold] {info.description or "-"}
[bold]Studio:[/bold] {info.studio_id or "-"}
[bold]Created:[/bold] {info.created_at.strftime("%Y-%m-%d %H:%M")}
[bold]Updated:[/bold] {info.updated_at.strftime("%Y-%m-%d %H:%M")}

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


# =============================================================================
# Checkpoints Subcommand
# =============================================================================

checkpoints_app = typer.Typer(help="Manage session checkpoints", no_args_is_help=True)
app.add_typer(checkpoints_app, name="checkpoints")


@checkpoints_app.command("list")
def checkpoints_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    session_id: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Filter by session ID"),
    ] = None,
) -> None:
    """List checkpoints in a project."""
    from questfoundry.runtime.checkpoint import CheckpointManager
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)
        manager = CheckpointManager(project)

        checkpoints = manager.list_checkpoints(session_id=session_id)

        if not checkpoints:
            console.print("[dim]No checkpoints found[/dim]")
            return

        table = Table(title=f"Checkpoints in {project_id}")
        table.add_column("ID", style="cyan")
        table.add_column("Session", style="dim")
        table.add_column("Turn", style="green", justify="right")
        table.add_column("Created")
        table.add_column("Summary")

        for cp in checkpoints:
            created = cp.created_at.strftime("%Y-%m-%d %H:%M")
            summary = cp.summary[:40] + "..." if cp.summary and len(cp.summary) > 40 else cp.summary
            table.add_row(
                cp.id,
                cp.session_id[:12] + "..." if len(cp.session_id) > 12 else cp.session_id,
                str(cp.turn_number),
                created,
                summary or "-",
            )

        console.print(table)
        console.print(f"[dim]Showing {len(checkpoints)} checkpoint(s)[/dim]")

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


@checkpoints_app.command("show")
def checkpoints_show(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    checkpoint_id: Annotated[str, typer.Argument(help="Checkpoint ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
) -> None:
    """Show checkpoint details."""
    from questfoundry.runtime.checkpoint import CheckpointManager
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)
        manager = CheckpointManager(project)

        checkpoint = manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            console.print(f"[red]✗ Checkpoint not found: {checkpoint_id}[/red]")
            raise typer.Exit(1)

        console.print(Panel(f"[cyan]{checkpoint.id}[/cyan]", title="Checkpoint"))

        # Metadata
        console.print("[bold]Metadata:[/bold]")
        console.print(f"  Session: {checkpoint.session_id}")
        console.print(f"  Turn: [green]{checkpoint.turn_number}[/green]")
        console.print(f"  Status: {checkpoint.session_status.value}")
        console.print(f"  Entry Agent: {checkpoint.entry_agent}")
        console.print(f"  Created: {checkpoint.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"  Schema Version: {checkpoint.schema_version}")

        # Summary
        if checkpoint.summary:
            console.print()
            console.print(f"[bold]Summary:[/bold] {checkpoint.summary}")

        # Mailbox States
        console.print()
        console.print("[bold]Mailbox States:[/bold]")
        for agent_id, messages in checkpoint.mailbox_states.items():
            console.print(f"  {agent_id}: {len(messages)} pending message(s)")

        # Active Delegations
        if checkpoint.active_delegations:
            console.print()
            console.print("[bold]Active Delegations:[/bold]")
            for deleg in checkpoint.active_delegations:
                console.print(
                    f"  - {deleg.delegation_id}: {deleg.from_agent} → {deleg.to_agent} "
                    f"({deleg.status})"
                )

        # Playbook Instances
        if checkpoint.playbook_instances:
            console.print()
            console.print(f"[bold]Playbook Instances:[/bold] {len(checkpoint.playbook_instances)}")
            for pb in checkpoint.playbook_instances:
                console.print(f"  - {pb.get('playbook_id', '-')}: {pb.get('status', '-')}")

        # Context Usage
        if checkpoint.context_usage:
            console.print()
            console.print("[bold]Context Usage:[/bold]")
            for agent_id, usage in checkpoint.context_usage.items():
                pct = usage.usage_percent
                console.print(
                    f"  {agent_id}: {usage.total_tokens:,} / {usage.limit:,} tokens ({pct:.1f}%)"
                )

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


@checkpoints_app.command("delete")
def checkpoints_delete(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    checkpoint_id: Annotated[str, typer.Argument(help="Checkpoint ID")],
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Delete a checkpoint."""
    from questfoundry.runtime.checkpoint import CheckpointManager
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id
    project = None

    try:
        project = Project.open(project_path)
        manager = CheckpointManager(project)

        # Check if checkpoint exists
        checkpoint = manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            console.print(f"[red]✗ Checkpoint not found: {checkpoint_id}[/red]")
            raise typer.Exit(1)

        # Confirm deletion
        if not force:
            console.print(f"Checkpoint: [cyan]{checkpoint_id}[/cyan]")
            console.print(f"Session: {checkpoint.session_id}")
            console.print(f"Turn: {checkpoint.turn_number}")
            confirm = typer.confirm("Are you sure you want to delete this checkpoint?")
            if not confirm:
                console.print("[dim]Cancelled[/dim]")
                return

        # Delete
        if manager.delete_checkpoint(checkpoint_id):
            console.print(f"[green]✓[/green] Deleted checkpoint: {checkpoint_id}")
        else:
            console.print(f"[red]✗ Failed to delete checkpoint: {checkpoint_id}[/red]")
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None
    finally:
        if project:
            project.close()


# =============================================================================
# Export Command (Stub)
# =============================================================================


# =============================================================================
# Validate Subcommand
# =============================================================================

validate_app = typer.Typer(help="Validate domain and schemas", no_args_is_help=True)
app.add_typer(validate_app, name="validate")


@validate_app.command("semantic")
def validate_semantic(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
    strict: Annotated[
        bool,
        typer.Option("--strict", "-s", help="Treat warnings as errors"),
    ] = False,
) -> None:
    """Check for semantic ambiguity issues in tool interfaces.

    Validates tool schemas against semantic conventions defined in
    meta/docs/semantic-conventions.md. Checks for:

    - Banned field names (color codes like 'green', 'yellow', 'red')
    - Discouraged field names ('hint', 'any', 'completing')
    - Biasing language in tool descriptions (emphatic caps, role-specific language)

    Exit codes:
    - 0: No errors (warnings allowed unless --strict)
    - 1: Errors found or warnings with --strict
    """
    from questfoundry.runtime.validation.semantic import (
        Severity,
        validate_domain_semantics,
    )

    console.print()
    console.print("[bold]Semantic Validation[/bold]")
    console.print(f"[dim]Domain: {domain}[/dim]")
    console.print()

    result = validate_domain_semantics(domain)

    if not result.issues:
        console.print(f"[green]✓[/green] {result.files_checked} files checked, no issues found")
        return

    # Group issues by severity
    errors = [i for i in result.issues if i.severity == Severity.ERROR]
    warnings = [i for i in result.issues if i.severity == Severity.WARNING]
    infos = [i for i in result.issues if i.severity == Severity.INFO]

    # Display issues
    for issue in result.issues:
        if issue.severity == Severity.ERROR:
            icon = "[red]✗[/red]"
            style = "red"
        elif issue.severity == Severity.WARNING:
            icon = "[yellow]⚠[/yellow]"
            style = "yellow"
        else:
            icon = "[blue]ℹ[/blue]"
            style = "dim"

        console.print(f"{icon} [{style}]{issue.message}[/{style}]")
        console.print(f"   [dim]Location: {issue.location}[/dim]")
        console.print(f"   [dim]Rule: {issue.rule}[/dim]")
        if issue.suggestion:
            console.print(f"   [cyan]Suggestion: {issue.suggestion}[/cyan]")
        console.print()

    # Summary
    console.print(f"[bold]Summary:[/bold] {result.files_checked} files checked")
    if errors:
        console.print(f"  [red]{len(errors)} error(s)[/red]")
    if warnings:
        console.print(f"  [yellow]{len(warnings)} warning(s)[/yellow]")
    if infos:
        console.print(f"  [dim]{len(infos)} info(s)[/dim]")

    # Determine exit code
    if errors:
        raise typer.Exit(1)
    if strict and warnings:
        console.print()
        console.print("[yellow]Failing due to --strict flag[/yellow]")
        raise typer.Exit(1)


# =============================================================================
# Corpus Subcommand
# =============================================================================

corpus_app = typer.Typer(help="Manage corpus index for search", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("build")
def corpus_build(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force rebuild even if files unchanged"),
    ] = False,
    embeddings: Annotated[
        bool,
        typer.Option("--embeddings", "-e", help="Build vector embeddings for semantic search"),
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Embedding provider: 'ollama' or 'openai'"),
    ] = None,
) -> None:
    """Build or update the corpus search index.

    Parses all corpus files in domain/knowledge/corpus/ and indexes
    them in a SQLite database for fast search. Only files that have
    changed since the last build are re-indexed (unless --force).

    Use --embeddings to also generate vector embeddings for semantic
    search. Use --provider to select 'ollama' (local) or 'openai'.
    If not specified, auto-detects available provider.
    """
    import asyncio

    from questfoundry.runtime.corpus.index import CorpusIndex

    corpus_dir = domain / "knowledge" / "corpus"
    if not corpus_dir.exists():
        console.print(f"[red]✗ Corpus directory not found: {corpus_dir}[/red]")
        raise typer.Exit(1)

    index_path = CorpusIndex.get_index_path(domain)
    index = CorpusIndex(index_path)

    console.print(f"[dim]Building index from {corpus_dir}...[/dim]")

    try:
        count = index.build(corpus_dir, force=force)

        if count > 0:
            console.print(f"[green]✓[/green] Indexed {count} file(s)")
        else:
            console.print("[dim]No files needed indexing (all up to date)[/dim]")

        status = index.get_status()
        console.print(
            f"[dim]Total: {status.file_count} files, {status.section_count} sections[/dim]"
        )

        # Build vector embeddings if requested
        if embeddings:
            console.print()
            console.print("[dim]Building vector embeddings...[/dim]")

            async def build_vectors(provider_name: str | None) -> int:
                from questfoundry.runtime.corpus.embeddings import get_embedding_provider
                from questfoundry.runtime.corpus.vector_index import VectorIndex

                embed_provider = await get_embedding_provider(provider_name=provider_name)
                if embed_provider is None:
                    console.print("[red]✗ No embedding provider available[/red]")
                    if provider_name:
                        console.print(f"[dim]Provider '{provider_name}' not available[/dim]")
                    else:
                        console.print(
                            "[dim]Install Ollama with nomic-embed-text or set OPENAI_API_KEY[/dim]"
                        )
                    return 0

                console.print(
                    f"[dim]Using {embed_provider.model} ({embed_provider.dimension}d)[/dim]"
                )

                vector_index = VectorIndex(index_path, dimension=embed_provider.dimension)
                if not vector_index.is_available:
                    console.print("[red]✗ sqlite-vec not available[/red]")
                    console.print("[dim]Install with: pip install sqlite-vec[/dim]")
                    return 0

                try:
                    embedded = await vector_index.build_vectors(embed_provider, force=force)
                    return embedded
                finally:
                    vector_index.close()

            embedded_count = asyncio.run(build_vectors(provider))
            if embedded_count > 0:
                console.print(f"[green]✓[/green] Embedded {embedded_count} section(s)")
            elif embedded_count == 0 and status.section_count > 0:
                console.print("[dim]Vectors already up to date (use --force to rebuild)[/dim]")

        console.print(f"[dim]Index: {index_path}[/dim]")

    finally:
        index.close()


@corpus_app.command("status")
def corpus_status(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """Show corpus index status.

    Displays information about the indexed corpus including file counts,
    section counts, and cluster breakdown. Also checks for stale files
    that need re-indexing, and shows vector embedding status if available.
    """
    from questfoundry.runtime.corpus.index import CorpusIndex
    from questfoundry.runtime.corpus.vector_index import VectorIndex

    index_path = CorpusIndex.get_index_path(domain)
    corpus_dir = domain / "knowledge" / "corpus"

    if not index_path.exists():
        console.print("[yellow]○[/yellow] Corpus index not found")
        console.print(f"[dim]Expected: {index_path}[/dim]")
        if corpus_dir.exists():
            console.print("[dim]Run 'qf corpus build' to create index[/dim]")
        return

    index = CorpusIndex(index_path)
    vector_index = VectorIndex(index_path)

    try:
        status = index.get_status(corpus_dir if corpus_dir.exists() else None)

        console.print()
        console.print("[bold]Corpus Index Status[/bold]")
        console.print()
        console.print(f"[green]✓[/green] Index exists: {index_path}")
        console.print(f"  Files: {status.file_count}")
        console.print(f"  Sections: {status.section_count}")

        # Vector index status
        console.print()
        console.print("[bold]Vector Search:[/bold]")
        vec_status = vector_index.get_status()
        if not vec_status["available"]:
            console.print(
                f"  [dim]Not available ({vec_status.get('reason', 'sqlite-vec not loaded')})[/dim]"
            )
        elif "embeddings" in vec_status:
            # Multi-model format
            embeddings = vec_status["embeddings"]
            if not embeddings:
                console.print("  [dim]No embeddings (run 'qf corpus build --embeddings')[/dim]")
            else:
                console.print(f"  [green]✓[/green] {len(embeddings)} model(s):")
                for emb in embeddings:
                    console.print(
                        f"    • {emb['model']} ({emb['dimension']}d): {emb['vector_count']} vectors"
                    )
                    if emb.get("indexed_at"):
                        console.print(f"      Built: {emb['indexed_at']}")
        elif vec_status.get("vector_count", 0) == 0:
            console.print("  [dim]No embeddings (run 'qf corpus build --embeddings')[/dim]")
        else:
            # Single model format (legacy)
            console.print(f"  [green]✓[/green] {vec_status['vector_count']} embeddings")
            if vec_status.get("model"):
                console.print(f"  Model: {vec_status['model']} ({vec_status['dimension']}d)")
            if vec_status.get("indexed_at"):
                console.print(f"  Built: {vec_status['indexed_at']}")

        if status.clusters:
            console.print()
            console.print("[bold]Clusters:[/bold]")
            for cluster, count in sorted(status.clusters.items()):
                console.print(f"  {cluster}: {count} file(s)")

        if status.stale_files:
            console.print()
            console.print(
                f"[yellow]⚠ {len(status.stale_files)} stale file(s) need re-indexing[/yellow]"
            )
            for f in status.stale_files[:5]:
                console.print(f"  [dim]{f}[/dim]")
            if len(status.stale_files) > 5:
                console.print(f"  [dim]... and {len(status.stale_files) - 5} more[/dim]")
            console.print("[dim]Run 'qf corpus build' to update[/dim]")

    finally:
        index.close()
        vector_index.close()


@corpus_app.command("validate")
def corpus_validate(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
) -> None:
    """Validate corpus frontmatter against schema.

    Checks each corpus file for:
    - Valid YAML frontmatter
    - Required fields (title, summary, topics, cluster)
    - Field value constraints (length, valid cluster names)
    - Proper section structure
    """
    from questfoundry.runtime.corpus.parser import parse_corpus_file

    corpus_dir = domain / "knowledge" / "corpus"
    if not corpus_dir.exists():
        console.print(f"[red]✗ Corpus directory not found: {corpus_dir}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print("[bold]Corpus Validation[/bold]")
    console.print(f"[dim]Directory: {corpus_dir}[/dim]")
    console.print()

    files = list(corpus_dir.glob("*.md"))
    if not files:
        console.print("[yellow]○[/yellow] No corpus files found")
        return

    errors_found = False
    valid_count = 0

    for file_path in sorted(files):
        result = parse_corpus_file(file_path)

        if result is None:
            console.print(f"[red]✗[/red] {file_path.name}")
            console.print("   [red]Could not parse file (missing or invalid frontmatter)[/red]")
            errors_found = True
            continue

        # Validate frontmatter
        validation_errors = result.frontmatter.validate()

        if validation_errors:
            console.print(f"[yellow]⚠[/yellow] {file_path.name}")
            for error in validation_errors:
                console.print(f"   [yellow]{error}[/yellow]")
            errors_found = True
        else:
            console.print(f"[green]✓[/green] {file_path.name}")
            valid_count += 1

    console.print()
    console.print(f"[bold]Summary:[/bold] {valid_count}/{len(files)} files valid")

    if errors_found:
        console.print()
        console.print("[yellow]Some files have validation issues[/yellow]")
        raise typer.Exit(1)


# =============================================================================
# Export Command (Stub)
# =============================================================================


# =============================================================================
# Agent Subcommand
# =============================================================================

agent_app = typer.Typer(help="Inspect agent configuration and prompts", no_args_is_help=True)
app.add_typer(agent_app, name="agent")


@agent_app.command("prompt")
def agent_prompt(
    agent_id: Annotated[str, typer.Argument(help="Agent ID (e.g., showrunner, scene_smith)")],
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
    section: Annotated[
        str | None,
        typer.Option(
            "--section",
            "-s",
            help="Show only specific section (constitution, knowledge, playbooks, agents, tools, stores, artifacts, menu)",
        ),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", "-r", help="Output raw text (no panels/formatting)"),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", "-f", help="Show complete system prompt as sent to LLM"),
    ] = False,
) -> None:
    """Show the system prompt for an agent (same as runtime uses).

    This command displays the actual system prompt that would be sent to the LLM,
    using the same code path as the runtime. Includes:

    - Constitution (inviolable principles)
    - Must-know entries (critical knowledge, inlined)
    - Playbooks menu (for orchestrators)
    - Agents menu (for orchestrators)
    - Tools list
    - Stores access
    - Artifact types
    - Knowledge menu

    Use --full to see the complete prompt as one block.
    Use --section to inspect specific parts.

    Examples:
        qf agent prompt showrunner
        qf agent prompt showrunner --full --raw > showrunner_prompt.txt
        qf agent prompt scene_smith --section tools
    """
    asyncio.run(_agent_prompt_async(agent_id, domain, section, raw, full))


async def _agent_prompt_async(
    agent_id: str,
    domain_path: Path,
    section: str | None,
    raw: bool,
    full: bool,
) -> None:
    """Async implementation of agent prompt command.

    Uses the same ContextBuilder and build_prompt as the runtime to ensure
    the CLI shows exactly what the LLM receives.
    """
    from questfoundry.runtime.agent.context import ContextBuilder
    from questfoundry.runtime.agent.prompt import build_prompt
    from questfoundry.runtime.domain import load_studio
    from questfoundry.runtime.tools.registry import ToolRegistry

    # Load domain
    result = await load_studio(domain_path)
    if not result.success or not result.studio:
        console.print("[red]✗ Failed to load domain[/red]")
        for error in result.errors:
            console.print(f"  [red]• {error.message}[/red]")
        raise typer.Exit(1)

    studio = result.studio

    # Find agent
    agent = next((a for a in studio.agents if a.id == agent_id), None)
    if not agent:
        console.print(f"[red]✗ Agent not found: {agent_id}[/red]")
        available = [a.id for a in studio.agents]
        console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    # Build context using the SAME code path as runtime
    context_builder = ContextBuilder(domain_path=domain_path)
    context = context_builder.build(agent, studio)

    # Get tool schemas (same as runtime)
    tool_registry = ToolRegistry(studio, domain_path=domain_path)
    tool_schemas = tool_registry.get_langchain_tools(agent, session_id=None)

    # Build prompt using the SAME code path as runtime
    prompt = build_prompt(
        agent=agent,
        constitution_text=context.constitution_text,
        must_know_entries=context.must_know_entries,
        role_specific_menu=context.role_specific_menu,
        tool_schemas=tool_schemas,
        playbooks_menu=context.playbooks_menu,
        stores_menu=context.stores_menu,
        artifact_types_menu=context.artifact_types_menu,
        agents_menu=context.agents_menu,
    )

    # Full mode: show complete prompt as sent to LLM
    if full:
        if raw:
            console.print(prompt.text)
        else:
            console.print()
            console.print(f"[bold]Complete System Prompt for {agent.name}[/bold] ({agent.id})")
            console.print(f"[dim]Archetypes: {', '.join(agent.archetypes)}[/dim]")
            console.print(f"[dim]Token estimate: ~{len(prompt.text) // 4}[/dim]")
            console.print()
            console.print(Panel(prompt.text, title="System Prompt", border_style="cyan"))
        return

    # Display header
    if not raw:
        console.print()
        console.print(f"[bold]System Prompt for {agent.name}[/bold] ({agent.id})")
        console.print(f"[dim]Archetypes: {', '.join(agent.archetypes)}[/dim]")
        console.print()

    # Extract sections from the prompt for display
    sections_found: dict[str, str] = {}

    # Parse sections from prompt text
    prompt_text = prompt.text
    section_markers = [
        ("constitution", "## Constitution"),
        ("knowledge", "## Critical Knowledge"),
        ("playbooks", "## Available Playbooks"),
        ("agents", "## Available Agents"),
        ("tools", "## Your Tools"),
        ("stores", "## Your Store Access"),
        ("artifacts", "## Artifact Types"),
        ("menu", "## Knowledge Menu"),
    ]

    for sec_name, marker in section_markers:
        start = prompt_text.find(marker)
        if start >= 0:
            # Find end (next ## or end of text)
            end = len(prompt_text)
            for _, other_marker in section_markers:
                if other_marker != marker:
                    other_start = prompt_text.find(other_marker, start + len(marker))
                    if other_start > start and other_start < end:
                        end = other_start
            sections_found[sec_name] = prompt_text[start:end].strip()

    # Show requested section or all
    section_titles = {
        "constitution": ("Constitution", "blue"),
        "knowledge": ("Critical Knowledge", "green"),
        "playbooks": ("Available Playbooks", "magenta"),
        "agents": ("Available Agents", "magenta"),
        "tools": ("Your Tools", "cyan"),
        "stores": ("Store Access", "yellow"),
        "artifacts": ("Artifact Types", "yellow"),
        "menu": ("Knowledge Menu", "dim"),
    }

    if section:
        if section in sections_found:
            if raw:
                console.print(sections_found[section])
            else:
                title, color = section_titles.get(section, (section, "white"))
                console.print(Panel(sections_found[section], title=title, border_style=color))
        else:
            console.print(f"[yellow]⚠ Section '{section}' not found in prompt[/yellow]")
            console.print(f"[dim]Available: {', '.join(sections_found.keys())}[/dim]")
        return

    # Show all sections
    for sec_name in [
        "constitution",
        "knowledge",
        "playbooks",
        "agents",
        "tools",
        "stores",
        "artifacts",
        "menu",
    ]:
        if sec_name in sections_found:
            if raw:
                console.print(sections_found[sec_name])
                console.print()
            else:
                title, color = section_titles.get(sec_name, (sec_name, "white"))
                console.print(Panel(sections_found[sec_name], title=title, border_style=color))

    # Summary stats
    if not raw:
        console.print()
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Token estimate: ~{len(prompt.text) // 4}")
        console.print(f"  Sections: {', '.join(sections_found.keys())}")
        console.print(f"  Tools: {len(tool_schemas)}")
        if context.playbooks_menu:
            console.print(f"  Playbooks: {len(context.playbooks_menu)}")
        if context.agents_menu:
            console.print(f"  Delegation targets: {len(context.agents_menu)}")


@agent_app.command("list")
def agent_list(
    domain: Annotated[
        Path,
        typer.Option("--domain", "-d", help="Path to domain directory"),
    ] = Path("domain-v4"),
    archetype: Annotated[
        str | None,
        typer.Option("--archetype", "-a", help="Filter by archetype"),
    ] = None,
) -> None:
    """List all agents in the domain.

    Similar to 'qf roles' but with more filtering options.
    """
    asyncio.run(_agent_list_async(domain, archetype))


async def _agent_list_async(domain_path: Path, archetype: str | None) -> None:
    """Async implementation of agent list command."""
    from questfoundry.runtime.domain import load_studio

    result = await load_studio(domain_path)
    if not result.success or not result.studio:
        console.print("[red]✗ Failed to load domain[/red]")
        for error in result.errors:
            console.print(f"  [red]• {error.message}[/red]")
        raise typer.Exit(1)

    studio = result.studio
    agents = studio.agents

    # Filter by archetype if specified
    if archetype:
        agents = [a for a in agents if archetype in a.archetypes]
        if not agents:
            console.print(f"[yellow]No agents with archetype '{archetype}'[/yellow]")
            all_archetypes = set()
            for a in studio.agents:
                all_archetypes.update(a.archetypes)
            console.print(f"[dim]Available archetypes: {', '.join(sorted(all_archetypes))}[/dim]")
            return

    table = Table(title=f"Agents in {studio.name}")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Archetypes", style="green")
    table.add_column("Entry", style="yellow")
    table.add_column("Knowledge")

    for agent in sorted(agents, key=lambda a: a.id):
        archetypes = ", ".join(agent.archetypes)
        entry = "✓" if agent.is_entry_agent else ""

        # Count knowledge requirements
        knowledge_info = []
        if agent.knowledge_requirements:
            kr = agent.knowledge_requirements
            if kr.constitution:
                knowledge_info.append("C")
            if kr.must_know:
                knowledge_info.append(f"M:{len(kr.must_know)}")
            if kr.should_know:
                knowledge_info.append(f"S:{len(kr.should_know)}")
            if kr.role_specific:
                knowledge_info.append(f"R:{len(kr.role_specific)}")

        table.add_row(
            agent.id,
            agent.name,
            archetypes,
            entry,
            " ".join(knowledge_info) if knowledge_info else "-",
        )

    console.print(table)
    console.print()
    console.print("[dim]Legend: C=constitution, M=must_know, S=should_know, R=role_specific[/dim]")


# =============================================================================
# Export Command (Stub)
# =============================================================================


@app.command()
def export(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    output_path: Annotated[
        Path | None,
        typer.Argument(help="Output file path (default: project_id.epub/.html)"),
    ] = None,
    projects_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Projects directory"),
    ] = Path("projects"),
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format (epub, html, markdown)"),
    ] = "epub",
) -> None:
    """Export a project to a publishable format.

    This command exports artifacts from a project into a cohesive
    document suitable for publication or sharing. The Book Binder
    agent handles the actual export process.

    Supported formats:
    - epub: E-book format (default)
    - html: Web-ready HTML
    - markdown: Plain markdown with frontmatter
    """
    from questfoundry.runtime.storage import Project

    project_path = projects_dir / project_id

    # Check if project exists
    project = Project(project_path)
    if not project.exists():
        console.print(f"[red]✗ Project not found: {project_id}[/red]")
        raise typer.Exit(1) from None

    # Validate format
    supported_formats = ["epub", "html", "markdown"]
    if output_format not in supported_formats:
        console.print(f"[red]✗ Unsupported format: {output_format}[/red]")
        console.print(f"[dim]Supported: {', '.join(supported_formats)}[/dim]")
        raise typer.Exit(1)

    # Determine output path
    if output_path is None:
        ext_map = {"epub": ".epub", "html": ".html", "markdown": ".md"}
        output_path = Path(f"{project_id}{ext_map[output_format]}")

    # Stub implementation - show not-yet-implemented message
    console.print()
    console.print(
        Panel(
            "[yellow]Export functionality is not yet implemented.[/yellow]\n\n"
            "This command will invoke the Book Binder agent to:\n"
            "• Compile cold-store artifacts into a cohesive document\n"
            "• Apply formatting for the target format\n"
            "• Generate table of contents and navigation\n"
            "• Package assets (images, styles) as needed\n\n"
            f"[dim]Project: {project_id}\n"
            f"Format: {output_format}\n"
            f"Output: {output_path}[/dim]",
            title="Export (Coming Soon)",
            border_style="yellow",
        )
    )

    console.print()
    console.print(
        "[dim]Track progress: https://github.com/pvliesdonk/questfoundry/issues/150[/dim]"
    )


if __name__ == "__main__":
    app()
