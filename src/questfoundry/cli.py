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
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from questfoundry.runtime import AgentRuntime
    from questfoundry.runtime.messaging import AsyncMessageBroker
    from questfoundry.runtime.models import Agent, Studio
    from questfoundry.runtime.observability import EventLogger, TracingManager
    from questfoundry.runtime.providers import OllamaProvider
    from questfoundry.runtime.session import Session
    from questfoundry.runtime.storage import Project
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Global verbosity level (set by callback)
_verbosity: int = 0

console = Console()


def _configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    global _verbosity
    _verbosity = verbosity

    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:  # 2+
        level = logging.DEBUG

    # Configure root logger with Rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbosity >= 2)],
        force=True,
    )

    # Also set questfoundry logger
    logging.getLogger("questfoundry").setLevel(level)


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
    _ensure_project(project, projects_dir)

    if prompt:
        asyncio.run(
            _ask_single(project, prompt, entry_agent, domain, projects_dir, provider, model, log)
        )
    else:
        asyncio.run(_ask_repl(project, entry_agent, domain, projects_dir, provider, model, log))


def _ensure_project(project_id: str, projects_dir: Path) -> None:
    """Create the project if it doesn't exist."""
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

        console.print(f"[dim]Creating project '{project_id}'...[/dim]")
        Project.create(
            path=project_path,
            name=name,
            description=description,
            studio_id="questfoundry",
        )
        console.print(f"[green]✓[/green] Created project [bold]{name}[/bold]")
        console.print()


async def _setup_runtime(
    project_id: str,
    entry_agent_id: str | None,
    domain_path: Path,
    projects_dir: Path,
    provider_name: str | None,
    model: str | None,
    log: bool = False,
) -> tuple[
    Project,
    Studio,
    AgentRuntime,
    Agent,
    Session,
    OllamaProvider,
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

    host: str = provider_config.host or "http://localhost:11434"
    model_to_use: str = model or provider_config.default_model or "qwen3:8b"

    # Create provider
    logger.debug(f"Connecting to {selected_provider} at {host}")
    provider = OllamaProvider(host=host)

    # Check provider availability
    if not await provider.check_availability():
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

    # Create runtime
    runtime = AgentRuntime(
        provider=provider,
        studio=studio,
        domain_path=domain_path,
        model=model_to_use,
        event_logger=event_logger,
        tracing_manager=tracing_manager if tracing_manager.enabled else None,
        broker=broker,
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

    # Create session
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


async def _stream_response(
    runtime: AgentRuntime,
    agent: Agent,
    user_input: str,
    session: Session,
    process_delegations: bool = True,
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

    # Show token usage at -v+
    if _verbosity >= 1 and final_usage:
        tokens_info = []
        if final_usage.prompt_tokens:
            tokens_info.append(f"prompt: {final_usage.prompt_tokens}")
        if final_usage.completion_tokens:
            tokens_info.append(f"completion: {final_usage.completion_tokens}")
        if final_usage.total_tokens:
            tokens_info.append(f"total: {final_usage.total_tokens}")
        if tokens_info:
            console.print(f"[dim]tokens: {', '.join(tokens_info)}[/dim]")

    # Show timing at -vv+
    if _verbosity >= 2:
        console.print(f"[dim]time: {duration_ms:.0f}ms | model: {runtime._model}[/dim]")
        logger.debug(f"Response generated in {duration_ms:.0f}ms")

    # Process any pending delegations
    if process_delegations:
        delegation_results = await runtime.process_pending_delegations(session)
        if delegation_results:
            console.print()
            console.print(f"[dim]── Processed {len(delegation_results)} delegation(s) ──[/dim]")
            for result in delegation_results:
                status = "[green]✓[/green]" if result.get("success") else "[red]✗[/red]"
                to_agent = result.get("to_agent", "unknown")
                task_preview = result.get("task", "")[:40]
                console.print(f"  {status} {to_agent}: {task_preview}...")
                if result.get("success") and _verbosity >= 2:
                    # Show delegatee response at -vv+
                    response_preview = str(result.get("result", ""))[:200]
                    console.print(f"    [dim]{response_preview}...[/dim]")
                elif not result.get("success"):
                    console.print(f"    [red]{result.get('error', 'Unknown error')}[/red]")

    return full_content


async def _handle_clarification_requests(
    broker: AsyncMessageBroker,
) -> list[dict[str, Any]]:
    """
    Check for and handle clarification requests from agents.

    Returns list of clarification requests that were found (for logging/debugging).
    The requests are handled interactively - user input is collected and responses sent.
    """
    from questfoundry.runtime.messaging import create_message
    from questfoundry.runtime.messaging.types import MessageType

    logger = logging.getLogger("questfoundry.cli")

    # Get the "customer" mailbox (special mailbox for human communication)
    mailbox = await broker.get_mailbox("customer")
    pending = await mailbox.get_all_pending()

    clarification_requests = [
        msg for msg in pending if msg.type == MessageType.CLARIFICATION_REQUEST
    ]

    if not clarification_requests:
        return []

    handled = []

    for request in clarification_requests:
        payload = request.payload
        question = payload.get("question", "")
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
        project_id, entry_agent_id, domain_path, projects_dir, provider_name, model, log
    )

    # Wrap execution in LangSmith session tracing
    with (
        tracing_manager.session(session.id, agent.id, project_id)
        if tracing_manager
        else nullcontext()
    ):
        try:
            console.print(f"[dim]{agent.name}:[/dim]")
            await _stream_response(runtime, agent, prompt, session)
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
                    await _stream_response(runtime, agent, context_msg, session)
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
                console.print(f"[dim]Events logged to: {log_path}[/dim]")
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
) -> None:
    """Run interactive REPL mode."""
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
        project_id, entry_agent_id, domain_path, projects_dir, provider_name, model, log
    )

    # Print header
    console.print()
    console.print("[bold]QuestFoundry Interactive Session[/bold]")
    console.print(f"Project: [cyan]{project_id}[/cyan]")
    console.print(f"Agent: [green]{agent.name}[/green] ({agent.id})")
    if log_path:
        console.print(f"Logging: [dim]{log_path}[/dim]")
    if tracing_manager:
        console.print("[dim]LangSmith tracing: enabled[/dim]")
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

                    # Stream response
                    console.print(f"[dim]{agent.name}:[/dim]")
                    await _stream_response(runtime, agent, user_input, session)
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
                            await _stream_response(runtime, agent, context_msg, session)
                            console.print()

                except ContextOverflowError as e:
                    console.print(f"[red]✗ Context overflow: {e}[/red]")
                    console.print("[dim]Try a shorter message or start a new session[/dim]")
                    console.print()
                except ProviderError as e:
                    console.print(f"[red]✗ Provider error: {e}[/red]")
                    console.print()

        except (KeyboardInterrupt, EOFError):
            console.print()

        finally:
            # End session tracing
            if tracing_manager:
                tracing_manager.end_session(turn_count=session.turn_count)
            # Log session end and flush
            if event_logger:
                event_logger.session_complete(session_id=session.id, turn_count=session.turn_count)
                console.print(f"[dim]Events logged to: {log_path}[/dim]")
            # Show session summary
            console.print()
            console.print(f"[dim]Session ended: {session.turn_count} turns[/dim]")
            await provider.close()
            project.close()


# =============================================================================
# Projects Subcommand
# =============================================================================


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


if __name__ == "__main__":
    app()
