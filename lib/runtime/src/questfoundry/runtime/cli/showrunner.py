"""
Showrunner Agent - Translation layer between human requests and studio protocol.

Based on spec: components/showrunner_agent.md
FLEXIBLE component - natural language interface design.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Parsed command intent."""
    action: str
    args: List[str]
    flags: Dict[str, str]
    loop_id: str


@dataclass
class LoopExecutionPlan:
    """Plan for loop execution."""
    loop_id: str
    context: Dict[str, Any]
    dependencies: List[str]
    mode: str = "workshop"


@dataclass
class ExecutionResult:
    """Result of loop execution."""
    success: bool
    summary: str
    artifacts: Dict[str, Any]
    tu_id: str
    quality_status: Dict[str, str]
    next_steps: List[str]
    error: Optional[str] = None


class Showrunner:
    """
    Orchestrate AI agents on behalf of human authors.

    The Showrunner translates natural language requests into studio protocol
    execution, manages loop orchestration, and presents results in human-friendly format.
    """

    def __init__(
        self,
        graph_factory: Optional[GraphFactory] = None,
        state_manager: Optional[StateManager] = None
    ):
        """
        Initialize Showrunner.

        Args:
            graph_factory: GraphFactory instance (creates new if not provided)
            state_manager: StateManager instance (creates new if not provided)
        """
        self.graph_factory = graph_factory or GraphFactory()
        self.state_manager = state_manager or StateManager()

    def execute_request(
        self,
        command: str,
        parsed_intent: ParsedIntent,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a human request through studio loops.

        Steps:
        1. Determine which loop(s) to run
        2. Prepare context for loop
        3. Create and execute loop
        4. Translate results
        5. Return formatted output

        Args:
            command: Original human command
            parsed_intent: Parsed command (action, args, flags)
            user_context: Optional project context

        Returns:
            ExecutionResult with artifacts and summary
        """
        try:
            logger.info(f"[bold]Executing:[/bold] {command}")

            # 1. Map intent to loop execution plan
            plan = self.map_intent_to_loop(parsed_intent, user_context)

            # 2. Execute dependencies first (if any)
            for dep_loop_id in plan.dependencies:
                logger.info(f"Executing dependency: [bold cyan]{dep_loop_id}[/bold cyan]")
                self._execute_single_loop(dep_loop_id, {})

            # 3. Execute primary loop
            final_state = self._execute_single_loop(plan.loop_id, plan.context)

            # 4. Translate results to human-friendly format
            result = self.translate_results(final_state, plan.loop_id, command)

            logger.info(f"[bold green]Request completed successfully[/bold green]: {result.tu_id}")
            return result

        except Exception as e:
            logger.error(f"Request execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                summary=f"Execution failed: {str(e)}",
                artifacts={},
                tu_id="",
                quality_status={},
                next_steps=[],
                error=str(e)
            )

    def map_intent_to_loop(
        self,
        intent: ParsedIntent,
        user_context: Optional[Dict[str, Any]] = None
    ) -> LoopExecutionPlan:
        """
        Determine which loop(s) to execute based on intent.

        Intent Mapping:
        - "write <text>" → story_spark
        - "review story" → hook_harvest
        - "add lore <topic>" → lore_deepening
        - "expand codex <entry>" → codex_expansion
        - "tune style" → style_tune_up
        - "add art <desc>" → art_touch_up
        - "add audio <desc>" → audio_pass
        - "translate <lang>" → translation_pass
        - "narrate <scene>" → narration_dry_run
        - "export <format>" → binding_run

        Args:
            intent: Parsed command intent
            user_context: Optional project state

        Returns:
            LoopExecutionPlan with loop_id, context, dependencies
        """
        # Context is already prepared by CLI, just pass through
        context = self.prepare_context(intent, intent.loop_id, user_context)

        # Check for special cases requiring dependencies
        dependencies: List[str] = []
        if intent.loop_id == "narration_dry_run":
            # Narration requires binding first
            dependencies.append("binding_run")

        return LoopExecutionPlan(
            loop_id=intent.loop_id,
            context=context,
            dependencies=dependencies,
            mode=intent.flags.get("mode", "workshop")
        )

    def prepare_context(
        self,
        intent: ParsedIntent,
        loop_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare context dict for loop initialization.

        Extract parameters from intent and user context,
        format them for loop's expected context schema.

        Args:
            intent: Parsed command intent
            loop_id: Target loop identifier
            user_context: Optional project state

        Returns:
            Context dict ready for StateManager.initialize_state()
        """
        context: Dict[str, Any] = {}

        # Add user context if provided
        if user_context:
            context.update(user_context)

        # Add intent-specific context
        if loop_id == "story_spark":
            # Extract scene text from args
            scene_text = " ".join(intent.args) if intent.args else ""
            context["scene_text"] = scene_text
            context["mode"] = intent.flags.get("mode", "workshop")

        elif loop_id == "hook_harvest":
            # Review mode - fetch hot artifacts
            context["mode"] = "review"
            # In real implementation, would fetch hot artifacts from storage
            context["hot_artifacts"] = []

        elif loop_id == "lore_deepening":
            # Extract lore topic
            topic = " ".join(intent.args) if intent.args else ""
            context["topic"] = topic

        elif loop_id == "translation_pass":
            # Extract target language
            target_language = intent.args[0] if intent.args else "Spanish"
            context["target_language"] = target_language

        elif loop_id == "binding_run":
            # Extract format
            format_type = intent.args[0] if intent.args else "markdown"
            context["format"] = format_type

        # Add mode from flags
        if "mode" in intent.flags:
            context["mode"] = intent.flags["mode"]

        logger.debug(f"Prepared context for {loop_id}: {list(context.keys())}")
        return context

    def _execute_single_loop(
        self,
        loop_id: str,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> StudioState:
        """
        Execute a single loop and return final state.

        Args:
            loop_id: Loop pattern identifier
            context: Prepared context dict
            progress_callback: Optional function to call with progress updates

        Returns:
            Final StudioState after loop completion
        """
        logger.info(f"[bold cyan]>[/bold cyan] Starting loop: [bold]{loop_id}[/bold]")

        # 1. Initialize state
        state = self.state_manager.initialize_state(loop_id, context)

        # 2. Create loop graph
        graph = self.graph_factory.create_loop_graph(loop_id)

        # 3. Execute graph
        logger.info(f"Invoking compiled graph for {loop_id}")

        # Call progress callback if provided
        if progress_callback:
            progress_callback(f"Starting {loop_id}")

        # Invoke the graph with the initial state
        final_state = graph.invoke(state)

        logger.info(f"[bold green]DONE[/bold green] Completed loop: [bold]{loop_id}[/bold] | TU: {final_state['tu_id']}")

        if progress_callback:
            progress_callback(f"Completed {loop_id}")

        return final_state

    def translate_results(
        self,
        state: StudioState,
        loop_id: str,
        original_command: str
    ) -> ExecutionResult:
        """
        Translate studio state into human-readable results.

        Extract artifacts, format summary, suggest next steps.

        Args:
            state: Final StudioState from loop execution
            loop_id: Which loop was executed
            original_command: Original human command

        Returns:
            ExecutionResult with human-friendly summary and next steps
        """
        tu_id = state["tu_id"]
        lifecycle = state["tu_lifecycle"]
        artifacts = state["artifacts"]

        # Extract quality status
        quality_status = {}
        for bar_name, bar_data in state["quality_bars"].items():
            status = bar_data.get("status", "unknown")
            emoji = self._status_emoji(status)
            quality_status[bar_name] = f"{emoji} {status}"

        # Generate summary based on loop type
        summary_lines = [
            f"✓ Completed {loop_id}",
            f"Trace Unit: {tu_id}",
            f"Status: {lifecycle}",
            ""
        ]

        # Add artifact summary
        if artifacts:
            summary_lines.append("Artifacts created:")
            for artifact_key, artifact_data in artifacts.items():
                content = artifact_data.get("content", "")
                content_preview = content[:80] + "..." if len(content) > 80 else content
                summary_lines.append(f"  • {artifact_key}: {content_preview}")
            summary_lines.append("")

        # Add quality status
        if quality_status:
            summary_lines.append("Quality Status:")
            for bar, status in quality_status.items():
                summary_lines.append(f"  • {bar}: {status}")
            summary_lines.append("")

        # Suggest next steps
        next_steps = self._suggest_next_steps(loop_id, state)
        if next_steps:
            summary_lines.append("Next steps:")
            for step in next_steps:
                summary_lines.append(f"  • {step}")

        summary = "\n".join(summary_lines)

        return ExecutionResult(
            success=True,
            summary=summary,
            artifacts=artifacts,
            tu_id=tu_id,
            quality_status=quality_status,
            next_steps=next_steps,
            error=None
        )

    def _status_emoji(self, status: str) -> str:
        """Get emoji for quality bar status."""
        status_map = {
            "green": "🟢",
            "yellow": "🟡",
            "red": "🔴",
            "gray": "⚫",
            "unknown": "⚪"
        }
        return status_map.get(status.lower(), "⚪")

    def _suggest_next_steps(self, loop_id: str, state: StudioState) -> List[str]:
        """
        Suggest next commands based on loop type and state.

        Args:
            loop_id: Which loop was executed
            state: Final state

        Returns:
            List of suggested next command strings
        """
        suggestions = []
        lifecycle = state["tu_lifecycle"]

        if loop_id == "story_spark":
            if lifecycle == "hot-proposed":
                suggestions.append("Run 'qf review story' to refine and approve")
                suggestions.append(f"Run 'qf show {state['tu_id']}' to view full content")
                suggestions.append("Run 'qf add lore <topic>' if you need backstory")

        elif loop_id == "hook_harvest":
            suggestions.append("Run 'qf export epub' to preview accepted content")
            suggestions.append("Run 'qf tune style' to fix any style issues")

        elif loop_id == "binding_run":
            suggestions.append("Run 'qf narrate <chapter>' to generate audio preview")
            suggestions.append("Run 'qf export <format>' for final output")

        return suggestions
