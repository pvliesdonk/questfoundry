"""Finalization tools for structured output.

This module provides tools that signal completion of a stage's
conversation phase and capture structured artifact data.

Each stage has its own finalization tool (submit_dream, submit_brainstorm, etc.)
that validates and captures the structured output.
"""

from __future__ import annotations

from typing import Any

from questfoundry.tools.base import Tool, ToolDefinition
from questfoundry.tools.generated import SUBMIT_DREAM_PARAMS


class SubmitDreamTool:
    """Tool for finalizing DREAM stage output.

    When called, signals that the creative vision has been finalized
    and provides the structured artifact data for validation.

    The tool schema matches the dream.schema.json artifact format,
    ensuring the LLM produces valid structured output.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding.

        Uses generated schema from schemas/dream.schema.json to ensure
        the tool definition stays in sync with artifact validation.
        """
        return ToolDefinition(
            name="submit_dream",
            description=(
                "Submit the finalized creative vision. Call this when you have "
                "discussed and refined the story concept with the user and are "
                "ready to lock in the creative direction. "
                "IMPORTANT: You must include type='dream' and version=1."
            ),
            parameters=SUBMIT_DREAM_PARAMS,
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        """Execute the finalization tool.

        Note: Actual validation happens in ConversationRunner.
        This method is called after successful validation.

        Args:
            arguments: The finalized artifact data.

        Returns:
            Confirmation message.
        """
        return "Creative vision submitted for validation."


class ReadyToSummarizeTool:
    """Signal tool for LLM to indicate discussion is complete.

    When called during the Discuss phase, signals that the LLM believes
    the conversation has reached a good stopping point and is ready
    to move to the Summarize phase.

    This is a no-op tool - it doesn't perform any action, just signals
    intent to transition phases.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="ready_to_summarize",
            description=(
                "Signal that the discussion is complete and you are ready to "
                "summarize the creative direction. Call this when you have "
                "gathered enough information to proceed with summarizing."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        """Execute the signal tool.

        Returns:
            JSON confirmation of the transition signal.
        """
        return '{"result": "proceed"}'


class SubmitBrainstormTool:
    """Tool for finalizing BRAINSTORM stage output.

    Captures raw creative material including characters, settings,
    plot hooks, and other story elements.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="submit_brainstorm",
            description=(
                "Submit the brainstorm results. Call this when you have "
                "generated sufficient raw creative material for the story."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                        "description": "Character concepts",
                    },
                    "settings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                        "description": "Setting/location concepts",
                    },
                    "plot_hooks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Potential plot hooks and story seeds",
                    },
                    "conflicts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Conflict ideas",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional creative notes",
                    },
                },
                "required": ["characters", "plot_hooks"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        """Execute the finalization tool."""
        return "Brainstorm material submitted for validation."


# Registry of finalization tools by stage name
# Each value is a callable that returns a Tool-compatible instance
FINALIZATION_TOOLS: dict[str, type[SubmitDreamTool] | type[SubmitBrainstormTool]] = {
    "dream": SubmitDreamTool,
    "brainstorm": SubmitBrainstormTool,
    # Future stages: seed, grow, fill, ship
}


def get_finalization_tool(stage: str) -> Tool | None:
    """Get the finalization tool for a stage.

    Args:
        stage: Stage name (e.g., "dream", "brainstorm").

    Returns:
        Instantiated finalization tool, or None if stage not found.
    """
    tool_class = FINALIZATION_TOOLS.get(stage)
    if tool_class is not None:
        return tool_class()
    return None
