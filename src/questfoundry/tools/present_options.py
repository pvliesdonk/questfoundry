"""Present structured options to user during interactive conversation.

This tool allows the LLM to present structured choices during the discuss
phase, providing a better UX than freeform questions for decisions like
genre, tone, or scope selection.

The tool:
1. Displays formatted options to the user
2. Collects their selection (number or freeform text)
3. Returns the selection to the LLM as structured JSON

Requires interactive mode with callbacks set via interactive_context.
In non-interactive mode, returns "skipped" to let LLM proceed autonomously.

Example tool call:
    {
        "question": "What genre fits your haunted mansion idea?",
        "options": [
            {"label": "Mystery", "description": "Focus on clues and deduction", "recommended": true},
            {"label": "Horror", "description": "Focus on fear and dread"},
            {"label": "Gothic Romance", "description": "Atmosphere and emotion"}
        ]
    }
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

from questfoundry.observability.logging import get_logger
from questfoundry.tools.base import Tool, ToolDefinition
from questfoundry.tools.interactive_context import get_interactive_callbacks

log = get_logger(__name__)


class Option(TypedDict, total=False):
    """A single option in the choices list."""

    label: str  # Required: display text (1-5 words)
    description: str  # Optional: explains implications
    recommended: bool  # Optional: mark as recommended choice


class PresentOptionsInput(TypedDict):
    """Input schema for present_options tool."""

    question: str  # The question to ask
    options: list[Option]  # 2-4 options to present


# JSON Schema for LLM tool binding
PRESENT_OPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["question", "options"],
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to ask the user. Should be clear and specific.",
        },
        "options": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "description": "2-4 options for the user to choose from.",
            "items": {
                "type": "object",
                "required": ["label"],
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short option name (1-5 words).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief explanation of this choice.",
                    },
                    "recommended": {
                        "type": "boolean",
                        "description": "True if this is your recommended option.",
                    },
                },
            },
        },
    },
}


class PresentOptionsTool:
    """Tool for presenting structured choices to users.

    This tool bridges the LLM's desire to present options with the
    interactive CLI's ability to collect user input. It uses the
    callback bridge in interactive_context to access display and
    input functions.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        return ToolDefinition(
            name="present_options",
            description=(
                "Present structured choices to the user. Use when offering "
                "clear alternatives for genre, tone, scope, or similar decisions. "
                "User can select by number or type a custom response."
            ),
            parameters=PRESENT_OPTIONS_SCHEMA,
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool synchronously.

        In non-interactive mode, returns immediately with guidance.
        In interactive mode, runs the async logic via the event loop.

        Args:
            arguments: Tool arguments with question and options.

        Returns:
            JSON string following ADR-008 format.
        """
        callbacks = get_interactive_callbacks()

        if callbacks is None:
            log.debug("present_options_skipped", reason="not_interactive")
            return json.dumps(
                {
                    "result": "skipped",
                    "reason": "Not in interactive mode",
                    "action": "Proceed with your best judgment on this decision.",
                }
            )

        # Run async logic - we need to get into the running event loop
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context - use run_coroutine_threadsafe
            # to run our coroutine from within sync code
            future = asyncio.run_coroutine_threadsafe(
                self._execute_async(arguments, callbacks),
                loop,
            )
            return future.result(timeout=300)  # 5 minute timeout for user input
        except RuntimeError:
            # No running loop - shouldn't happen in normal use
            log.warning("present_options_no_loop")
            return json.dumps(
                {
                    "result": "error",
                    "error": "No event loop available",
                    "action": "Proceed with your best judgment on this decision.",
                }
            )

    async def _execute_async(
        self,
        arguments: dict[str, Any],
        callbacks: Any,  # InteractiveCallbacks but avoid circular import
    ) -> str:
        """Async implementation that uses interactive callbacks.

        Args:
            arguments: Tool arguments with question and options.
            callbacks: InteractiveCallbacks with user_input_fn and display_fn.

        Returns:
            JSON string with user's selection.
        """
        question = arguments.get("question", "Please choose an option:")
        options: list[dict[str, Any]] = arguments.get("options", [])

        if not options or len(options) < 2:
            return json.dumps(
                {
                    "result": "error",
                    "error": "At least 2 options required",
                    "action": "Ask the question directly without structured options.",
                }
            )

        # Format and display the options
        display_text = self._format_options(question, options)
        callbacks.display_fn(display_text)

        # Collect user input
        user_response = await callbacks.user_input_fn()

        # Parse the selection
        selection = self._parse_selection(user_response, options)

        log.debug(
            "present_options_complete",
            question=question[:50],
            selection=selection[:50] if isinstance(selection, str) else str(selection),
        )

        return json.dumps(
            {
                "result": "success",
                "question": question,
                "selected": selection,
                "action": "User has made their selection. Continue based on their choice.",
            }
        )

    def _format_options(self, question: str, options: list[dict[str, Any]]) -> str:
        """Format question and options for display.

        Args:
            question: The question to display.
            options: List of option dicts with label, description, recommended.

        Returns:
            Formatted markdown string for display.
        """
        lines = [f"**{question}**\n"]

        for i, opt in enumerate(options, 1):
            label = opt.get("label", f"Option {i}")
            description = opt.get("description", "")
            recommended = opt.get("recommended", False)

            # Build option line
            rec_marker = " *(Recommended)*" if recommended else ""
            lines.append(f"  **[{i}]** {label}{rec_marker}")

            if description:
                lines.append(f"      {description}")

        lines.append("")
        lines.append("  **[0]** Something else...")
        lines.append("")
        lines.append("*Enter number or type your own response:*")

        return "\n".join(lines)

    def _parse_selection(
        self,
        response: str | None,
        options: list[dict[str, Any]],
    ) -> str:
        """Parse user response into a selection.

        Args:
            response: User's raw input (number, label, or freeform text).
            options: Original options list for lookup.

        Returns:
            The selected option label or the user's freeform response.
        """
        if not response or not response.strip():
            # Default to recommended option, or first option
            for opt in options:
                if opt.get("recommended"):
                    label: str = opt.get("label", "")
                    return label
            first_label: str = options[0].get("label", "") if options else ""
            return first_label

        response = response.strip()

        # Handle numeric selection
        if response.isdigit():
            idx = int(response)
            if idx == 0:
                # User wants to type something else - return marker
                return "[custom]"
            if 1 <= idx <= len(options):
                selected_label: str = options[idx - 1].get("label", f"Option {idx}")
                return selected_label

        # Check if response matches an option label (case-insensitive)
        response_lower = response.lower()
        for opt in options:
            opt_label: str = opt.get("label", "")
            if opt_label.lower() == response_lower:
                return opt_label

        # Treat as freeform custom response
        return response


# Implement Tool protocol
assert isinstance(PresentOptionsTool(), Tool)
