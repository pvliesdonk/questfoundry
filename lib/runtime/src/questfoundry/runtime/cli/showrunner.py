"""
Showrunner Interface - LLM-backed role with customer communication mandate.

Based on spec: components/showrunner_agent.md v2.0.0
Architecture: Showrunner is a role (from showrunner.yaml), not infrastructure.

NOTE: This is a transitional implementation. The full LLM-backed interpretation
will be added in Phase 6B once the interpret_customer_directive tool is defined.
For now, uses deterministic mapping but maintains the correct interface structure.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


@dataclass
class ShowrunnerResponse:
    """Response from Showrunner after interpreting customer directive."""
    success: bool
    plain_language_response: str
    loops_executed: List[str]
    suggested_next_steps: List[str]
    error: Optional[str] = None


@dataclass
class InterpretationResult:
    """Internal result from interpreting a customer directive."""
    loops_sequenced: List[str]
    plain_language_explanation: str
    context_for_loops: Dict[str, Any]


class ShowrunnerInterface:
    """
    Interface to the Showrunner role (LLM-backed agent with customer mandate).

    The Showrunner is defined in spec/05-definitions/roles/showrunner.yaml and
    loaded like any other role. Its mandate is to interpret customer directives
    and coordinate internal studio operations.

    Key Principles:
    - Showrunner is a role, not infrastructure
    - Customer talks in natural language (no jargon)
    - Showrunner decides which loops to run
    - Internal operations are hidden from customer
    - Responses are plain language only
    """

    def __init__(
        self,
        graph_factory: Optional[GraphFactory] = None,
        state_manager: Optional[StateManager] = None,
        role_profile: Optional[Any] = None
    ):
        """
        Initialize Showrunner interface.

        Args:
            graph_factory: GraphFactory instance (creates new if not provided)
            state_manager: StateManager instance (creates new if not provided)
            role_profile: Showrunner role profile from showrunner.yaml (loads if not provided)
        """
        self.state_manager = state_manager or StateManager()
        # Pass state_manager to graph_factory for message tracing
        self.graph_factory = graph_factory or GraphFactory(state_manager=self.state_manager)

        # Load Showrunner role profile
        if role_profile is None:
            try:
                self.role = self.graph_factory.schema_registry.load_role("showrunner")
                logger.info(f"Loaded Showrunner role: {self.role.name}")
            except Exception as e:
                logger.warning(f"Could not load showrunner.yaml: {e}")
                self.role = None
        else:
            self.role = role_profile

    def interpret_and_execute(
        self,
        customer_message: str,
        verbose: bool = False
    ) -> ShowrunnerResponse:
        """
        Interpret customer directive and execute appropriate work.

        This is the primary method called by the CLI. It:
        1. Interprets the customer's natural language request
        2. Decides which loops to run (currently deterministic, will be LLM)
        3. Executes loops internally
        4. Translates results to plain language

        Args:
            customer_message: Natural language request from customer
            verbose: Whether to show detailed execution info

        Returns:
            ShowrunnerResponse with plain language response and next steps
        """
        try:
            logger.info(f"[bold]Showrunner interpreting:[/bold] {customer_message}")

            # 1. Interpret customer directive
            # TODO: Replace with LLM-backed interpretation using interpret_customer_directive tool
            interpretation = self._interpret_directive_deterministic(customer_message)

            if verbose:
                logger.info(f"Interpretation: {interpretation.loops_sequenced}")
                logger.info(f"Context: {list(interpretation.context_for_loops.keys())}")

            # 2. Execute loops internally (customer doesn't see this)
            loops_executed = []
            final_states = {}

            for loop_id in interpretation.loops_sequenced:
                try:
                    logger.info(f"[cyan]Executing:[/cyan] {loop_id}")
                    final_state = self._execute_loop_internal(
                        loop_id,
                        interpretation.context_for_loops
                    )
                    loops_executed.append(loop_id)
                    final_states[loop_id] = final_state

                    if verbose:
                        tu_id = final_state.get("tu_id", "unknown")
                        logger.info(f"[green]Completed:[/green] {loop_id} → TU {tu_id}")

                except Exception as e:
                    logger.error(f"Loop {loop_id} failed: {e}", exc_info=verbose)
                    return ShowrunnerResponse(
                        success=False,
                        plain_language_response=f"I ran into an issue while working on that: {str(e)}",
                        loops_executed=loops_executed,
                        suggested_next_steps=[],
                        error=str(e)
                    )

            # 3. Translate results to plain language
            plain_response = self._generate_plain_language_response(
                interpretation,
                final_states,
                customer_message
            )

            # 4. Suggest next steps
            next_steps = self._suggest_next_steps(
                interpretation.loops_sequenced,
                final_states
            )

            logger.info("[bold green]Request completed successfully[/bold green]")

            return ShowrunnerResponse(
                success=True,
                plain_language_response=plain_response,
                loops_executed=loops_executed,
                suggested_next_steps=next_steps,
                error=None
            )

        except Exception as e:
            logger.error(f"Showrunner failed: {e}", exc_info=True)
            return ShowrunnerResponse(
                success=False,
                plain_language_response=f"I apologize, but I encountered an error: {str(e)}",
                loops_executed=[],
                suggested_next_steps=[],
                error=str(e)
            )

    def _interpret_directive_deterministic(
        self,
        customer_message: str
    ) -> InterpretationResult:
        """
        Deterministic interpretation of customer directive.

        TODO: Replace with LLM-backed interpretation in Phase 6B.
        This is a transitional implementation that maintains the correct interface.

        Args:
            customer_message: Natural language request

        Returns:
            InterpretationResult with loops to run and context
        """
        message_lower = customer_message.lower()
        context: Dict[str, Any] = {}

        # Pattern matching for common requests
        if any(word in message_lower for word in ["write", "create", "draft", "scene"]):
            # Customer wants to create content
            loops = ["story_spark"]
            context["scene_text"] = customer_message
            context["mode"] = "workshop"
            explanation = "I'll work on creating that content for you."

        elif any(word in message_lower for word in ["review", "harvest", "refine", "improve"]):
            # Customer wants review/refinement
            loops = ["hook_harvest"]
            context["mode"] = "review"
            explanation = "I'll review the story and identify interesting narrative hooks."

        elif any(word in message_lower for word in ["lore", "backstory", "background", "worldbuilding"]):
            # Customer wants lore development
            loops = ["lore_deepening"]
            context["topic"] = customer_message
            explanation = "I'll develop that aspect of the world's lore."

        elif any(word in message_lower for word in ["codex", "encyclopedia", "reference"]):
            # Customer wants codex expansion
            loops = ["codex_expansion"]
            context["entry"] = customer_message
            explanation = "I'll expand the codex with that information."

        elif any(word in message_lower for word in ["style", "voice", "tone"]):
            # Customer wants style tuning
            loops = ["style_tune_up"]
            explanation = "I'll review and tune the narrative style."

        elif any(word in message_lower for word in ["translate", "translation", "language"]):
            # Customer wants translation
            loops = ["translation_pass"]
            # Extract target language from message
            for lang in ["spanish", "french", "german", "italian", "portuguese"]:
                if lang in message_lower:
                    context["target_language"] = lang.capitalize()
                    break
            explanation = f"I'll translate the content to {context.get('target_language', 'the requested language')}."

        elif any(word in message_lower for word in ["export", "publish", "bind"]):
            # Customer wants binding/export
            loops = ["binding_run"]
            context["format"] = "markdown"
            explanation = "I'll prepare the content for export."

        else:
            # Default: treat as a writing request
            loops = ["story_spark"]
            context["scene_text"] = customer_message
            context["mode"] = "workshop"
            explanation = "I'll work on that for you."

        return InterpretationResult(
            loops_sequenced=loops,
            plain_language_explanation=explanation,
            context_for_loops=context
        )

    def _execute_loop_internal(
        self,
        loop_id: str,
        context: Dict[str, Any]
    ) -> StudioState:
        """
        Execute a single loop internally (customer doesn't see this).

        Args:
            loop_id: Loop identifier
            context: Context dict for loop initialization

        Returns:
            Final StudioState after loop completion
        """
        # Initialize state
        state = self.state_manager.initialize_state(loop_id, context)

        # Create and execute loop graph
        graph = self.graph_factory.create_loop_graph(loop_id)
        final_state = graph.invoke(state)

        return final_state

    def _generate_plain_language_response(
        self,
        interpretation: InterpretationResult,
        final_states: Dict[str, StudioState],
        original_message: str
    ) -> str:
        """
        Generate plain language response for customer (no jargon).

        Args:
            interpretation: The interpretation result
            final_states: Final states from executed loops
            original_message: Original customer message

        Returns:
            Plain language response string
        """
        # Start with the explanation
        response_lines = [interpretation.plain_language_explanation, ""]

        # Add summary of what was created (no technical jargon)
        for loop_id, state in final_states.items():
            artifacts = state.get("artifacts", {})
            if artifacts:
                response_lines.append("Created:")
                for artifact_key, artifact_data in artifacts.items():
                    # Translate artifact keys to customer-friendly names
                    friendly_name = self._translate_artifact_name(artifact_key)
                    response_lines.append(f"  • {friendly_name}")
                response_lines.append("")

        # Add quality status in plain language
        for loop_id, state in final_states.items():
            quality_bars = state.get("quality_bars", {})
            if quality_bars:
                issues = []
                for bar_name, bar_data in quality_bars.items():
                    status = bar_data.get("status", "unknown")
                    if status in ["yellow", "red"]:
                        friendly_bar = self._translate_bar_name(bar_name)
                        issues.append(friendly_bar)

                if issues:
                    response_lines.append("Areas that need attention:")
                    for issue in issues:
                        response_lines.append(f"  • {issue}")
                    response_lines.append("")
                else:
                    response_lines.append("Everything looks good!")
                    response_lines.append("")

        return "\n".join(response_lines).strip()

    def _suggest_next_steps(
        self,
        loops_executed: List[str],
        final_states: Dict[str, StudioState]
    ) -> List[str]:
        """
        Suggest next steps in plain language (no commands).

        Args:
            loops_executed: List of loops that were executed
            final_states: Final states from loops

        Returns:
            List of suggested next step strings
        """
        suggestions = []

        # Suggest based on what was just done
        if "story_spark" in loops_executed:
            suggestions.append("Review and refine the content")
            suggestions.append("Add backstory or lore details")

        if "hook_harvest" in loops_executed:
            suggestions.append("Export the content to see how it looks")
            suggestions.append("Fine-tune the narrative style")

        if "lore_deepening" in loops_executed:
            suggestions.append("Update the codex with the new lore")
            suggestions.append("Review how it integrates with the story")

        if "binding_run" in loops_executed:
            suggestions.append("Preview the exported content")
            suggestions.append("Generate narration for testing")

        # Check quality bars for suggestions
        for loop_id, state in final_states.items():
            quality_bars = state.get("quality_bars", {})
            for bar_name, bar_data in quality_bars.items():
                status = bar_data.get("status", "unknown")
                if status == "yellow" or status == "red":
                    friendly_bar = self._translate_bar_name(bar_name)
                    suggestions.append(f"Address {friendly_bar}")

        return suggestions[:5]  # Limit to 5 suggestions

    def _translate_artifact_name(self, artifact_key: str) -> str:
        """Translate technical artifact key to customer-friendly name."""
        translations = {
            "hot_draft": "Draft content",
            "cold_canon": "Final content",
            "hooks": "Narrative hooks",
            "lore_entry": "Lore entry",
            "codex_entry": "Codex entry",
            "style_report": "Style analysis",
            "translation": "Translated content",
            "view": "Exported content"
        }
        return translations.get(artifact_key, artifact_key.replace("_", " ").title())

    def _translate_bar_name(self, bar_name: str) -> str:
        """Translate technical quality bar name to customer-friendly description."""
        translations = {
            "integrity": "content consistency",
            "reachability": "story flow",
            "nonlinearity": "narrative choices",
            "gateways": "progression logic",
            "style": "writing style",
            "determinism": "asset consistency",
            "presentation": "formatting and accessibility"
        }
        return translations.get(bar_name.lower(), bar_name.replace("_", " "))


# Legacy classes for backward compatibility during transition
# TODO: Remove in Phase 6C after all references are updated

@dataclass
class ParsedIntent:
    """Deprecated: Legacy class for backward compatibility."""
    action: str
    args: List[str]
    flags: Dict[str, str]
    loop_id: str


@dataclass
class ExecutionResult:
    """Deprecated: Legacy class for backward compatibility."""
    success: bool
    summary: str
    artifacts: Dict[str, Any]
    tu_id: str
    quality_status: Dict[str, str]
    next_steps: List[str]
    error: Optional[str] = None


# Legacy Showrunner class (for backward compatibility)
# TODO: Remove in Phase 6C
class Showrunner:
    """
    Deprecated: Legacy orchestrator class.

    Use ShowrunnerInterface instead. This class is kept for backward
    compatibility during transition.
    """

    def __init__(self, graph_factory=None, state_manager=None):
        """Initialize with a ShowrunnerInterface internally."""
        self._interface = ShowrunnerInterface(
            graph_factory=graph_factory,
            state_manager=state_manager
        )
        logger.warning(
            "Showrunner class is deprecated. Use ShowrunnerInterface instead."
        )

    def execute_request(self, command: str, parsed_intent: ParsedIntent, user_context=None):
        """Legacy execute_request method."""
        # Convert to natural language and use new interface
        result = self._interface.interpret_and_execute(command)

        # Convert ShowrunnerResponse to ExecutionResult
        return ExecutionResult(
            success=result.success,
            summary=result.plain_language_response,
            artifacts={},
            tu_id="legacy",
            quality_status={},
            next_steps=result.suggested_next_steps,
            error=result.error
        )
