"""Finalization tools for structured output.

This module provides tools that signal completion of a stage's
conversation phase and capture structured artifact data.

Each stage has its own finalization tool (submit_dream, submit_brainstorm, etc.)
that validates and captures the structured output.
"""

from __future__ import annotations

from typing import Any

from questfoundry.tools.base import Tool, ToolDefinition


class SubmitDreamTool:
    """Tool for finalizing DREAM stage output.

    When called, signals that the creative vision has been finalized
    and provides the structured artifact data for validation.

    The tool schema matches the dream.schema.json artifact format,
    ensuring the LLM produces valid structured output.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="submit_dream",
            description=(
                "Submit the finalized creative vision. Call this when you have "
                "discussed and refined the story concept with the user and are "
                "ready to lock in the creative direction."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Primary genre (e.g., 'fantasy', 'mystery', 'sci-fi', 'horror')",
                    },
                    "subgenre": {
                        "type": "string",
                        "description": "Optional genre refinement (e.g., 'urban fantasy', 'cozy mystery')",
                    },
                    "tone": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tone descriptors (e.g., ['dark', 'suspenseful', 'romantic'])",
                    },
                    "audience": {
                        "type": "string",
                        "description": "Target audience: 'all ages', 'young adult', 'adult', or 'mature'",
                    },
                    "themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Core thematic elements (e.g., ['redemption', 'found family'])",
                    },
                    "style_notes": {
                        "type": "string",
                        "description": "Prose style guidance (e.g., 'flowery and descriptive' or 'terse and punchy')",
                    },
                    "scope": {
                        "type": "object",
                        "description": "Story scope parameters",
                        "required": ["target_word_count", "estimated_passages"],
                        "properties": {
                            "target_word_count": {
                                "type": "integer",
                                "description": "Approximate total word count (e.g., 15000)",
                            },
                            "estimated_passages": {
                                "type": "integer",
                                "description": "Target number of scenes/passages (e.g., 25)",
                            },
                            "branching_depth": {
                                "type": "string",
                                "description": "Branching complexity: 'light', 'moderate', 'heavy', 'extensive'",
                            },
                            "estimated_playtime_minutes": {
                                "type": "integer",
                                "description": "Target reading/play time in minutes",
                            },
                        },
                    },
                    "content_notes": {
                        "type": "object",
                        "description": "Content guidance",
                        "properties": {
                            "includes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Content to include (e.g., ['mild violence', 'romance'])",
                            },
                            "excludes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Content to avoid (e.g., ['graphic violence', 'explicit content'])",
                            },
                        },
                    },
                },
                "required": ["genre", "tone", "audience", "themes"],
            },
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
