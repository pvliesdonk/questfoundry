"""
CLI output management for QuestFoundry.

Provides structured status output on stdout while routing logs to stderr.
Implements:
- Turn/role status indicators
- Artifact creation tracking
- Summary table generation
- Stream separation (output: stdout, logs: stderr)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.runtime.session import TokenUsage
    from questfoundry.runtime.storage import ProjectStatusSummary


@dataclass
class ArtifactEvent:
    """Record of an artifact creation during a session."""

    artifact_id: str
    artifact_type: str
    store: str
    created_by: str
    turn_number: int
    lifecycle_state: str = "draft"  # draft, review, approved, cold
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCallEvent:
    """Record of a tool call during a session."""

    tool_id: str
    success: bool
    turn_number: int
    agent_id: str
    execution_time_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DelegationEvent:
    """Record of a delegation during a session."""

    from_agent: str
    to_agent: str
    task_preview: str
    success: bool
    turn_number: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DelegationResultInfo:
    """Structured result from return_to_orchestrator."""

    from_agent: str
    task_completion: str  # completed | blocked
    assessment: str  # pass | partial | fail | skip | info
    recommendation: str  # proceed | rework | escalate | hold
    summary: str
    artifacts_produced: list[str] = field(default_factory=list)
    artifacts_ready_for_review: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class TurnSummary:
    """Summary of a single turn."""

    turn_number: int
    agent_id: str
    agent_name: str
    tool_calls: int = 0
    artifacts_created: int = 0
    delegations: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None


class StatusReporter:
    """
    Manages structured CLI output with stream separation.

    - Status updates go to stdout (main console)
    - Logs go to stderr (log console)
    - Tracks events for summary table
    """

    def __init__(
        self,
        verbosity: int = 0,
        show_summary: bool = True,
        quiet: bool = False,
    ):
        """
        Initialize the status reporter.

        Args:
            verbosity: Verbosity level (0=WARN, 1=INFO, 2=DEBUG)
            show_summary: Whether to show summary table at end
            quiet: If True, suppress all status output
        """
        # Main output console (stdout)
        self.console = Console(file=sys.stdout)

        # Log console (stderr) - for logs and debug info
        self.log_console = Console(file=sys.stderr, stderr=True)

        self.verbosity = verbosity
        self.show_summary = show_summary
        self.quiet = quiet

        # Tracking state
        self._artifacts: list[ArtifactEvent] = []
        self._tool_calls: list[ToolCallEvent] = []
        self._delegations: list[DelegationEvent] = []
        self._turns: list[TurnSummary] = []
        self._current_turn: TurnSummary | None = None

        # Session metadata
        self._session_id: str | None = None
        self._project_id: str | None = None
        self._agent_id: str | None = None
        self._playbook_id: str | None = None
        self._start_time: datetime | None = None

    def _now(self) -> datetime:
        """Get current time. Override in tests for deterministic output."""
        return datetime.now()

    # -------------------------------------------------------------------------
    # Project Status
    # -------------------------------------------------------------------------

    def project_created(self, project_name: str, project_path: Path) -> None:
        """Report new project creation."""
        if self.quiet:
            return

        self.console.print()
        self.console.print(f"[bold green]Creating new project[/bold green] '{project_name}'")
        self.console.print(f"  [dim]Created at {project_path}/[/dim]")

    def project_resumed(self, project_name: str, summary: ProjectStatusSummary) -> None:
        """Report existing project resumption with summary."""
        if self.quiet:
            return

        self.console.print()
        self.console.print(f"[bold blue]Resuming project[/bold blue] '{project_name}'")

        # Check if there's any activity
        if summary.session_count == 0:
            self.console.print("  [dim]No sessions yet[/dim]")
            return

        # Session info
        turns_text = "turn" if summary.total_turns == 1 else "turns"
        sessions_text = "session" if summary.session_count == 1 else "sessions"
        self.console.print(
            f"  [dim]Sessions:[/dim] {summary.session_count} {sessions_text} "
            f"({summary.total_turns} {turns_text} total)"
        )

        # Last activity
        if summary.last_activity:
            elapsed = self._now() - summary.last_activity
            if elapsed.total_seconds() < 60:
                ago = "just now"
            elif elapsed.total_seconds() < 3600:
                mins = int(elapsed.total_seconds() // 60)
                ago = f"{mins} minute{'s' if mins != 1 else ''} ago"
            elif elapsed.total_seconds() < 86400:
                hours = int(elapsed.total_seconds() // 3600)
                ago = f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                days = int(elapsed.total_seconds() // 86400)
                ago = f"{days} day{'s' if days != 1 else ''} ago"
            self.console.print(f"  [dim]Last active:[/dim] {ago}")

        # Artifact counts
        if summary.artifacts_by_store:
            parts = []
            for store, count in sorted(summary.artifacts_by_store.items()):
                parts.append(f"{count} in {store}")
            self.console.print(f"  [dim]Artifacts:[/dim] {', '.join(parts)}")

        # Active playbook
        if summary.active_playbook:
            playbook_id, phase = summary.active_playbook
            self.console.print(f"  [dim]Active:[/dim] {playbook_id} (phase: {phase})")

    # -------------------------------------------------------------------------
    # Session Lifecycle
    # -------------------------------------------------------------------------

    def session_start(
        self,
        session_id: str,
        project_id: str,
        agent_name: str,
        agent_id: str,
        playbook_id: str | None = None,
        model: str | None = None,
    ) -> None:
        """Report session start."""
        self._session_id = session_id
        self._project_id = project_id
        self._playbook_id = playbook_id
        self._agent_id = agent_id
        self._start_time = self._now()

        if self.quiet:
            return

        # Minimal header on stdout
        self.console.print()
        self.console.print(f"[bold]Session[/bold] {project_id}")
        info_parts = [f"[cyan]{agent_name}[/cyan] ({agent_id})"]
        if playbook_id:
            info_parts.append(f"playbook: [green]{playbook_id}[/green]")
        if model and self.verbosity >= 1:
            info_parts.append(f"model: [dim]{model}[/dim]")
        self.console.print(" | ".join(info_parts))
        self.console.print()

    def session_end(self, turn_count: int) -> None:
        """Report session end and show summary if enabled."""
        if self.quiet:
            return

        # Show summary table
        if self.show_summary and self._turns:
            self._print_summary_table()

        # Final status
        duration = ""
        if self._start_time:
            elapsed = (self._now() - self._start_time).total_seconds()
            if elapsed < 60:
                duration = f" in {elapsed:.1f}s"
            else:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                duration = f" in {mins}m {secs}s"

        self.console.print()
        self.console.print(f"[dim]Session complete: {turn_count} turns{duration}[/dim]")

    # -------------------------------------------------------------------------
    # Turn Lifecycle
    # -------------------------------------------------------------------------

    def turn_start(
        self,
        turn_number: int,
        agent_id: str,
        agent_name: str,
        playbook_id: str | None = None,
    ) -> None:
        """Report turn start."""
        self._current_turn = TurnSummary(
            turn_number=turn_number,
            agent_id=agent_id,
            agent_name=agent_name,
        )

        if self.quiet:
            return

        # Status line format: [Turn N] Agent Name (playbook)
        parts = [f"[bold cyan]Turn {turn_number}[/bold cyan]"]
        parts.append(f"[green]{agent_name}[/green]")
        if playbook_id:
            parts.append(f"[dim]({playbook_id})[/dim]")

        self.console.print(" ".join(parts))

    def turn_complete(
        self,
        usage: TokenUsage | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Report turn completion."""
        if self._current_turn:
            if usage:
                self._current_turn.prompt_tokens = usage.prompt_tokens
                self._current_turn.completion_tokens = usage.completion_tokens
            self._current_turn.duration_ms = duration_ms
            self._turns.append(self._current_turn)
            self._current_turn = None

    # -------------------------------------------------------------------------
    # Tool Events
    # -------------------------------------------------------------------------

    def tool_call(
        self,
        tool_id: str,
        success: bool,
        agent_id: str,
        turn_number: int,
        execution_time_ms: float | None = None,
        result_preview: str | None = None,
        result_data: dict[str, Any] | None = None,
    ) -> None:
        """Report a tool call.

        Args:
            tool_id: The tool that was called
            success: Whether the tool executed without error
            agent_id: Agent that called the tool
            turn_number: Current turn number
            execution_time_ms: Execution time
            result_preview: Deprecated - use result_data instead
            result_data: Full tool result data dict (may contain action_outcome, etc.)
        """
        event = ToolCallEvent(
            tool_id=tool_id,
            success=success,
            turn_number=turn_number,
            agent_id=agent_id,
            execution_time_ms=execution_time_ms,
        )
        self._tool_calls.append(event)

        # Track on current turn if available
        if self._current_turn:
            self._current_turn.tool_calls += 1

        if self.quiet:
            return

        # Extract semantic fields from result_data
        action_outcome = None
        rejection_reason = None
        if result_data:
            action_outcome = result_data.get("action_outcome") or result_data.get("result")
            rejection_reason = result_data.get("rejection_reason")

        # Show tool calls at INFO level (-v) or always show failures/rejections
        show_tool = self.verbosity >= 1 or not success or action_outcome == "rejected"
        if show_tool:
            # Build status indicator: execution success vs action outcome
            if not success:
                # Tool execution failed
                status_icon = "[red]✗ error[/red]"
            elif action_outcome == "rejected":
                status_icon = "[yellow]↩ rejected[/yellow]"
            elif action_outcome == "committed" or action_outcome == "saved":
                status_icon = "[green]✓ committed[/green]"
            elif action_outcome == "deferred":
                status_icon = "[blue]⏳ deferred[/blue]"
            elif action_outcome:
                status_icon = f"[green]✓[/green] {action_outcome}"
            else:
                status_icon = "[green]✓[/green]" if success else "[red]✗[/red]"

            time_str = f" [dim]({execution_time_ms:.0f}ms)[/dim]" if execution_time_ms else ""
            self.console.print(f"  [dim]tool:[/dim] {tool_id} {status_icon}{time_str}")

            # Show rejection reason
            if rejection_reason:
                self.console.print(f"    [yellow]reason: {rejection_reason}[/yellow]")

            # Show recovery action if present
            recovery_action = result_data.get("recovery_action") if result_data else None
            if recovery_action and self.verbosity >= 1:
                self.console.print(f"    [cyan]action: {recovery_action}[/cyan]")

            # Show result preview at higher verbosity (legacy support)
            if result_preview and self.verbosity >= 2:
                preview = (
                    result_preview[:80] + "..." if len(result_preview) > 80 else result_preview
                )
                self.console.print(f"    [dim]{preview}[/dim]")

    # -------------------------------------------------------------------------
    # Artifact Events
    # -------------------------------------------------------------------------

    def artifact_created(
        self,
        artifact_id: str,
        artifact_type: str,
        store: str,
        created_by: str,
        turn_number: int,
        lifecycle_state: str = "draft",
    ) -> None:
        """Report artifact creation."""
        event = ArtifactEvent(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            store=store,
            created_by=created_by,
            turn_number=turn_number,
            lifecycle_state=lifecycle_state,
        )
        self._artifacts.append(event)

        if self._current_turn:
            self._current_turn.artifacts_created += 1

        if self.quiet:
            return

        # Lifecycle state indicator
        state_colors = {
            "draft": "yellow",
            "review": "blue",
            "approved": "green",
            "cold": "cyan",
        }
        state_color = state_colors.get(lifecycle_state, "dim")

        # Always show artifact creation (this is what user wants to see)
        self.console.print(
            f"  [bold green]+[/bold green] {artifact_type}: "
            f"[cyan]{artifact_id}[/cyan] "
            f"[{state_color}]({lifecycle_state})[/{state_color}] "
            f"[dim]-> {store}[/dim]"
        )

    # -------------------------------------------------------------------------
    # Delegation Events
    # -------------------------------------------------------------------------

    def delegation_start(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        turn_number: int,
    ) -> None:
        """Report delegation start."""
        if self.quiet:
            return

        task_preview = task[:50] + "..." if len(task) > 50 else task
        turn_info = f" (turn {turn_number})" if self.verbosity >= 2 else ""
        self.console.print(
            f"  [dim]delegate:[/dim] {from_agent} [yellow]->[/yellow] {to_agent}{turn_info}"
        )
        if self.verbosity >= 1:
            self.console.print(f"    [dim]{task_preview}[/dim]")

    def delegation_complete(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        success: bool,
        turn_number: int,
        error: str | None = None,
    ) -> None:
        """Report delegation completion."""
        task_preview = task[:80] + "..." if len(task) > 80 else task

        event = DelegationEvent(
            from_agent=from_agent,
            to_agent=to_agent,
            task_preview=task_preview,
            success=success,
            turn_number=turn_number,
        )
        self._delegations.append(event)

        if self._current_turn:
            self._current_turn.delegations += 1

        if self.quiet:
            return

        # Show delegation with task - this is key info users want to see
        status_icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        self.console.print(f"  [dim]delegate:[/dim] [cyan]{to_agent}[/cyan] {status_icon}")
        # Always show task preview (this is what SR asked specialist to do)
        self.console.print(f"    [dim]task:[/dim] {task_preview}")
        if not success and error:
            self.console.print(f"    [red]error: {error}[/red]")

    def delegation_result(self, result: DelegationResultInfo) -> None:
        """
        Display structured result from return_to_orchestrator.

        Shows what the specialist accomplished: assessment, recommendation,
        summary, and any artifacts or blockers.
        """
        if self.quiet:
            return

        # Assessment color coding
        assessment_colors = {
            "pass": "green",
            "partial": "yellow",
            "fail": "red",
            "skip": "dim",
            "info": "blue",
        }
        assessment_color = assessment_colors.get(result.assessment, "white")

        # Recommendation color coding
        rec_colors = {
            "proceed": "green",
            "rework": "yellow",
            "escalate": "red",
            "hold": "yellow",
        }
        rec_color = rec_colors.get(result.recommendation, "white")

        # Task completion indicator
        completion_icon = "✓" if result.task_completion == "completed" else "⏸"
        completion_color = "green" if result.task_completion == "completed" else "yellow"

        # Header line with agent and completion status
        self.console.print(
            f"  [{completion_color}]{completion_icon}[/{completion_color}] "
            f"[cyan]{result.from_agent}[/cyan] "
            f"[{assessment_color}]{result.assessment}[/{assessment_color}] → "
            f"[{rec_color}]{result.recommendation}[/{rec_color}]"
        )

        # Summary (always shown)
        summary_text = result.summary[:100] + "..." if len(result.summary) > 100 else result.summary
        self.console.print(f"    [dim]{summary_text}[/dim]")

        # Artifacts produced (if any)
        if result.artifacts_produced:
            artifacts_str = ", ".join(result.artifacts_produced[:3])
            if len(result.artifacts_produced) > 3:
                artifacts_str += f" (+{len(result.artifacts_produced) - 3} more)"
            self.console.print(f"    [green]+[/green] artifacts: {artifacts_str}")

        # Blockers (if any)
        if result.blockers:
            blockers_str = "; ".join(result.blockers[:2])
            if len(result.blockers) > 2:
                blockers_str += f" (+{len(result.blockers) - 2} more)"
            self.console.print(f"    [yellow]![/yellow] blockers: {blockers_str}")

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------

    def _print_summary_table(self) -> None:
        """Print the session summary table."""
        table = Table(title="Session Summary", show_header=True, header_style="bold")
        table.add_column("Turn", style="cyan", justify="right")
        table.add_column("Agent", style="green")
        table.add_column("Tools", justify="right")
        table.add_column("Artifacts", justify="right")
        table.add_column("Tokens", justify="right", style="dim")

        # Aggregate tool/artifact counts by turn from event lists
        # This handles out-of-order tracking for delegated agents
        tools_by_turn: dict[int, int] = {}
        for tc in self._tool_calls:
            tools_by_turn[tc.turn_number] = tools_by_turn.get(tc.turn_number, 0) + 1

        artifacts_by_turn: dict[int, int] = {}
        for art in self._artifacts:
            artifacts_by_turn[art.turn_number] = artifacts_by_turn.get(art.turn_number, 0) + 1

        total_tokens = 0
        total_artifacts = 0
        total_tools = 0

        for turn in self._turns:
            tokens_str = ""
            # Handle None/0 values correctly - 0 is valid, None means unknown
            prompt = turn.prompt_tokens if turn.prompt_tokens is not None else 0
            completion = turn.completion_tokens if turn.completion_tokens is not None else 0
            if turn.prompt_tokens is not None or turn.completion_tokens is not None:
                tokens = prompt + completion
                total_tokens += tokens
                tokens_str = f"{tokens:,}"

            # Get counts from aggregated event lists
            turn_tools = tools_by_turn.get(turn.turn_number, 0)
            turn_artifacts = artifacts_by_turn.get(turn.turn_number, 0)

            artifacts_str = str(turn_artifacts) if turn_artifacts else "-"
            tools_str = str(turn_tools) if turn_tools else "-"

            total_artifacts += turn_artifacts
            total_tools += turn_tools

            table.add_row(
                str(turn.turn_number),
                turn.agent_name,
                tools_str,
                artifacts_str,
                tokens_str,
            )

        # Add totals row
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{len(self._turns)} turns[/bold]",
            f"[bold]{total_tools}[/bold]",
            f"[bold]{total_artifacts}[/bold]",
            f"[bold]{total_tokens:,}[/bold]" if total_tokens else "-",
        )

        self.console.print()
        self.console.print(table)

        # Artifacts summary if any were created
        if self._artifacts:
            self.console.print()
            self.console.print("[bold]Artifacts Created:[/bold]")
            state_colors = {
                "draft": "yellow",
                "review": "blue",
                "approved": "green",
                "cold": "cyan",
            }
            for art in self._artifacts:
                state_color = state_colors.get(art.lifecycle_state, "dim")
                self.console.print(
                    f"  [cyan]{art.artifact_id}[/cyan] ({art.artifact_type}) "
                    f"[{state_color}]{art.lifecycle_state}[/{state_color}] "
                    f"[dim]-> {art.store}[/dim]"
                )

    # -------------------------------------------------------------------------
    # Generic Output
    # -------------------------------------------------------------------------

    def print(self, message: str, style: str | None = None) -> None:
        """Print a message to stdout."""
        if self.quiet:
            return
        self.console.print(message, style=style)

    def log(self, message: str, level: str = "info") -> None:
        """Print a log message to stderr."""
        styles = {
            "debug": "dim",
            "info": "blue",
            "warning": "yellow",
            "error": "red",
        }
        style = styles.get(level, "")
        self.log_console.print(f"[{style}]{message}[/{style}]")

    def agent_response(self, agent_name: str) -> None:
        """Print agent response header."""
        if self.quiet:
            return
        self.console.print(f"[dim]{agent_name}:[/dim]")

    def print_response(self, content: str) -> None:
        """Print agent response content."""
        if self.quiet:
            return
        self.console.print(content)


def create_log_handler(log_console: Console, verbosity: int) -> Any:
    """
    Create a RichHandler that outputs to stderr.

    Args:
        log_console: Console instance configured for stderr
        verbosity: Verbosity level

    Returns:
        RichHandler configured for the log console
    """
    from rich.logging import RichHandler

    return RichHandler(
        console=log_console,
        rich_tracebacks=False,  # Disabled - rich tracebacks are too verbose
        show_path=verbosity >= 2,
        show_time=verbosity >= 1,
    )
