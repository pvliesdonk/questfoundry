"""
Turn validation for orchestrator enforcement.

Implements the tools-only output constraint for orchestrator agents:
- Orchestrators MUST end every turn with a terminating tool call
- Terminating tools have `terminates_turn: true` in their definition
- Non-terminating tool calls alone are insufficient

From meta/schemas/core/tool-definition.schema.json:
"terminates_turn: If true, calling this tool ends the agent's turn.
Runtime enforces: orchestrator agents MUST end every turn with a
terminating tool call (e.g., delegate, communicate)."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, Studio
    from questfoundry.runtime.providers import ToolCallRequest


@dataclass
class TurnValidationConfig:
    """Configuration for turn validation and retry behavior."""

    # Maximum retries before giving up on orchestrator enforcement
    max_retries: int = 3

    # Whether to allow non-terminating tool calls as intermediate steps
    allow_intermediate_tools: bool = True

    # Whether to include tool list in nudge messages
    include_tool_hints: bool = True


@dataclass
class TurnValidationResult:
    """Result of validating a turn."""

    # Whether the turn is valid
    valid: bool

    # Whether a terminating tool was called
    has_terminating_tool: bool = False

    # The terminating tool ID if called
    terminating_tool_id: str | None = None

    # List of non-terminating tools called
    non_terminating_tools: list[str] | None = None

    # Nudge message if invalid (for retry)
    nudge_message: str | None = None

    # Whether this is an orchestrator (for context)
    is_orchestrator: bool = False


class TurnValidator:
    """
    Validates that orchestrator turns comply with tools-only output constraint.

    Orchestrator agents (archetype: "orchestrator") must end every turn with
    a tool call that has `terminates_turn: true`. This enforces the hub-and-spoke
    delegation pattern where orchestrators coordinate but don't produce content.

    Usage:
        validator = TurnValidator(studio)
        result = validator.validate_turn(agent, tool_calls)
        if not result.valid:
            # Retry with result.nudge_message
    """

    def __init__(
        self,
        studio: Studio,
        config: TurnValidationConfig | None = None,
    ) -> None:
        """
        Initialize turn validator.

        Args:
            studio: Studio definition with tool definitions
            config: Optional validation configuration
        """
        self._studio = studio
        self._config = config or TurnValidationConfig()

        # Build lookup of terminating tools
        self._terminating_tools: set[str] = set()
        for tool in studio.tools:
            if tool.terminates_turn:
                self._terminating_tools.add(tool.id)

    @property
    def terminating_tools(self) -> set[str]:
        """Get the set of tool IDs that terminate turns."""
        return self._terminating_tools

    def is_orchestrator(self, agent: Agent) -> bool:
        """Check if an agent is an orchestrator."""
        return "orchestrator" in [
            a.value if hasattr(a, "value") else str(a) for a in agent.archetypes
        ]

    def has_tools_only_constraint(self, agent: Agent) -> bool:
        """
        Check if agent has a tools-only output constraint with runtime enforcement.

        This looks for constraints with enforcement="runtime" that indicate
        the agent must use tools-only output.
        """
        for constraint in agent.constraints:
            if constraint.enforcement.value == "runtime" and (
                "tools_only" in constraint.id or "tools-only" in constraint.id
            ):
                return True
        return False

    def requires_terminating_tool(self, agent: Agent) -> bool:
        """
        Check if an agent requires a terminating tool to end its turn.

        Returns True if:
        - Agent is an orchestrator archetype, OR
        - Agent has a tools_only constraint with runtime enforcement
        """
        return self.is_orchestrator(agent) or self.has_tools_only_constraint(agent)

    def validate_turn(
        self,
        agent: Agent,
        tool_calls: list[ToolCallRequest] | None,
    ) -> TurnValidationResult:
        """
        Validate that a turn complies with orchestrator enforcement rules.

        Args:
            agent: The agent whose turn is being validated
            tool_calls: List of tool calls made during the turn

        Returns:
            TurnValidationResult with validation status and nudge if needed
        """
        is_orch = self.requires_terminating_tool(agent)

        # Non-orchestrators don't require terminating tools
        if not is_orch:
            return TurnValidationResult(
                valid=True,
                has_terminating_tool=False,
                is_orchestrator=False,
            )

        # No tool calls at all
        if not tool_calls:
            return TurnValidationResult(
                valid=False,
                has_terminating_tool=False,
                is_orchestrator=True,
                nudge_message=self._build_no_tools_nudge(),
            )

        # Check for terminating tools
        terminating_tool_id = None
        non_terminating = []

        for tc in tool_calls:
            if tc.name in self._terminating_tools:
                terminating_tool_id = tc.name
            else:
                non_terminating.append(tc.name)

        # Has terminating tool - valid
        if terminating_tool_id:
            return TurnValidationResult(
                valid=True,
                has_terminating_tool=True,
                terminating_tool_id=terminating_tool_id,
                non_terminating_tools=non_terminating if non_terminating else None,
                is_orchestrator=True,
            )

        # Has non-terminating tools only - invalid
        return TurnValidationResult(
            valid=False,
            has_terminating_tool=False,
            non_terminating_tools=non_terminating,
            is_orchestrator=True,
            nudge_message=self._build_missing_terminator_nudge(non_terminating),
        )

    def _build_no_tools_nudge(self) -> str:
        """Build nudge message when no tools were called."""
        nudge = (
            "Your response contained no tool calls. "
            "As an orchestrator, you MUST use tools for ALL output.\n\n"
            "You MUST either:\n"
            "1. Call `delegate` to assign work to a specialist agent, OR\n"
            "2. Call `communicate` to send information to another agent, OR\n"
            "3. Call `terminate` if the workflow is complete\n\n"
            "Do NOT generate prose directly. "
            "Do NOT respond with plain text. "
            "You MUST make a tool call that ends your turn."
        )

        if self._config.include_tool_hints and self._terminating_tools:
            tools_list = ", ".join(f"`{t}`" for t in sorted(self._terminating_tools))
            nudge += f"\n\nTerminating tools available: {tools_list}"

        return nudge

    def _build_missing_terminator_nudge(
        self,
        non_terminating: list[str],
    ) -> str:
        """Build nudge when tools were called but none terminate the turn."""
        tools_used = ", ".join(f"`{t}`" for t in non_terminating)
        nudge = (
            f"You called {tools_used}, but none of these tools terminate your turn. "
            "As an orchestrator, you MUST end your turn with a terminating tool.\n\n"
            "After using intermediate tools, you MUST:\n"
            "1. Call `delegate` to assign the next task, OR\n"
            "2. Call `communicate` to relay information, OR\n"
            "3. Call `terminate` if the workflow is complete\n\n"
            "Your turn is not complete until you call a terminating tool."
        )

        if self._config.include_tool_hints and self._terminating_tools:
            tools_list = ", ".join(f"`{t}`" for t in sorted(self._terminating_tools))
            nudge += f"\n\nTerminating tools available: {tools_list}"

        return nudge

    def get_terminating_tool_names(self) -> list[str]:
        """Get list of terminating tool names for display."""
        return sorted(self._terminating_tools)


def create_turn_validator(
    studio: Studio,
    config: TurnValidationConfig | None = None,
) -> TurnValidator:
    """
    Create a turn validator for a studio.

    Args:
        studio: Studio definition
        config: Optional validation configuration

    Returns:
        Configured TurnValidator
    """
    return TurnValidator(studio, config)
