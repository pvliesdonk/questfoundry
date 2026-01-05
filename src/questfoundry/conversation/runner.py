"""Conversation runner for interactive stages.

This module provides the ConversationRunner class that manages
multi-turn LLM interactions with the 3-phase pattern:
Discuss → Summarize → Serialize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from questfoundry.conversation.state import ConversationState
from questfoundry.tools import ReadyToSummarizeTool, Tool

# Default configuration values
DEFAULT_MAX_DISCUSS_TURNS = 10
DEFAULT_VALIDATION_RETRIES = 3
USER_DONE_SIGNAL = "/done"

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from questfoundry.providers.base import LLMProvider, Message
    from questfoundry.tools import ToolCall


class ConversationError(Exception):
    """Raised when conversation fails.

    Attributes:
        message: Error description.
        state: Conversation state at time of failure.
    """

    def __init__(self, message: str, state: ConversationState | None = None) -> None:
        self.state = state
        super().__init__(message)


@dataclass
class ValidationErrorDetail:
    """Details of a single validation error.

    Attributes:
        field: The field path that failed validation (e.g., "genre", "scope.target_word_count").
        issue: Description of what went wrong (used as "problem" in feedback).
        provided: The value that was provided (if any).
        error_type: Pydantic error type code (e.g., "missing", "string_too_short").
            Used for reliable categorization instead of string matching on issue text.
        requirement: Human-readable description of what the field requires.
            Derived from error type and schema constraints.
    """

    field: str
    issue: str
    provided: Any = None
    error_type: str | None = None
    requirement: str | None = None


@dataclass
class ValidationResult:
    """Result of validating tool call arguments.

    Attributes:
        valid: Whether validation passed.
        error: Error message if validation failed (for display/logging).
        errors: Structured list of validation errors (preferred over error string).
        data: Validated/transformed data if validation passed.
        expected_fields: Set of valid field names (for detecting unknown fields).
    """

    valid: bool
    error: str | None = None
    errors: list[ValidationErrorDetail] | None = None
    data: dict[str, Any] | None = None
    expected_fields: set[str] | None = None


class ConversationRunner:
    """Manages multi-turn LLM conversations with the 3-phase pattern.

    The runner implements the Discuss → Summarize → Serialize pattern:

    1. **Discuss phase**: LLM discusses with user using research tools.
       Ends when user types /done, LLM calls ready_to_summarize(), or max turns.

    2. **Summarize phase**: LLM generates a summary with no tools.
       Distills the conversation into key decisions.

    3. **Serialize phase**: LLM calls finalization tool with structured data.
       Validation with retry loop; fails explicitly if no tool call.

    Both direct and interactive modes use this same code path:
    - Direct mode: max_discuss_turns=1, user_input_fn=None
    - Interactive mode: max_discuss_turns=10, user_input_fn provided

    Example:
        >>> runner = ConversationRunner(
        ...     provider=provider,
        ...     research_tools=[SearchCorpusTool()],
        ...     finalization_tool=SubmitDreamTool(),
        ... )
        >>> artifact, state = await runner.run(
        ...     initial_messages=[system_msg, user_msg],
        ...     user_input_fn=lambda: input("> "),
        ...     validator=validate_dream,
        ... )
    """

    def __init__(
        self,
        provider: LLMProvider,
        research_tools: list[Tool],
        finalization_tool: Tool,
        max_discuss_turns: int = DEFAULT_MAX_DISCUSS_TURNS,
        validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    ) -> None:
        """Initialize the conversation runner.

        Args:
            provider: LLM provider for completions.
            research_tools: Tools available during Discuss phase only.
            finalization_tool: Tool available during Serialize phase only.
            max_discuss_turns: Max turns in Discuss phase (1 for direct mode).
            validation_retries: Max validation retries in Serialize phase.

        Raises:
            TypeError: If finalization_tool doesn't implement the Tool protocol.
        """
        # Validate finalization tool implements Tool protocol
        if not isinstance(finalization_tool, Tool):
            raise TypeError(
                f"finalization_tool {finalization_tool!r} doesn't implement Tool protocol"
            )

        # Validate research tools
        for tool in research_tools:
            if not isinstance(tool, Tool):
                raise TypeError(f"Research tool {tool!r} doesn't implement Tool protocol")

        self._provider = provider
        self._research_tools = research_tools
        self._finalization_tool = finalization_tool
        self._max_discuss_turns = max_discuss_turns
        self._validation_retries = validation_retries

        # Build tool lookup maps for each phase
        self._ready_tool = ReadyToSummarizeTool()
        self._discuss_tools = {t.definition.name: t for t in research_tools}
        self._discuss_tools[self._ready_tool.definition.name] = self._ready_tool
        self._discuss_tool_defs = [t.definition for t in research_tools] + [
            self._ready_tool.definition
        ]

        self._finalization_tool_name = finalization_tool.definition.name
        self._finalization_tool_def = finalization_tool.definition

    async def run(
        self,
        initial_messages: list[Message],
        user_input_fn: Callable[[], Awaitable[str | None]] | None = None,
        validator: Callable[[dict[str, Any]], ValidationResult] | None = None,
        summary_prompt: str | None = None,
        on_assistant_message: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any], ConversationState]:
        """Run the 3-phase conversation.

        Args:
            initial_messages: Starting messages (typically system + user).
            user_input_fn: Async function to get user input. If None,
                runs in direct mode (no user interaction in Discuss).
            validator: Optional function to validate finalization data.
            summary_prompt: Optional guidance for the Summarize phase.
            on_assistant_message: Optional callback for assistant messages.

        Returns:
            Tuple of (artifact_data, conversation_state).

        Raises:
            ConversationError: If any phase fails.
        """
        state = ConversationState(messages=list(initial_messages), phase="discuss")

        # Phase 1: Discuss
        await self._discuss_phase(state, user_input_fn, on_assistant_message)

        # Phase 2: Summarize
        state.phase = "summarize"
        summary = await self._summarize_phase(state, summary_prompt, on_assistant_message)

        # Phase 3: Serialize
        state.phase = "serialize"
        artifact = await self._serialize_phase(state, summary, validator, on_assistant_message)

        return artifact, state

    async def _discuss_phase(
        self,
        state: ConversationState,
        user_input_fn: Callable[[], Awaitable[str | None]] | None,
        on_assistant_message: Callable[[str], None] | None,
    ) -> None:
        """Run the Discuss phase with research tools only.

        Exit conditions:
        - User types /done
        - LLM calls ready_to_summarize()
        - max_discuss_turns reached (auto-transition)

        Args:
            state: Conversation state to update.
            user_input_fn: Function to get user input, or None for direct mode.
            on_assistant_message: Callback for assistant messages.
        """
        while state.turn_count < self._max_discuss_turns:
            # Call LLM with research tools only
            response = await self._provider.complete(
                messages=state.messages,
                tools=self._discuss_tool_defs if self._discuss_tool_defs else None,
                tool_choice="auto",
            )
            state.llm_calls += 1
            state.tokens_used += response.tokens_used

            # Add assistant message
            if response.content:
                state.add_message({"role": "assistant", "content": response.content})
                if on_assistant_message is not None:
                    on_assistant_message(response.content)

            # Handle tool calls
            if response.has_tool_calls and response.tool_calls:
                ready_to_proceed = await self._handle_discuss_tools(response.tool_calls, state)
                if ready_to_proceed:
                    return  # LLM signaled ready to summarize
                # Tool was executed - loop back to see LLM response to tool result
                # without incrementing turn count (tool calls don't count as turns)
                continue

            # Get user input (if interactive)
            if user_input_fn is not None:
                user_input = await user_input_fn()
                if user_input:
                    # Check for /done signal
                    if user_input.strip().lower() == USER_DONE_SIGNAL:
                        return  # User signaled ready to summarize
                    state.add_message({"role": "user", "content": user_input})

            state.turn_count += 1

        # Max turns reached - auto-transition to summarize

    async def _handle_discuss_tools(
        self,
        tool_calls: list[ToolCall],
        state: ConversationState,
    ) -> bool:
        """Process tool calls during Discuss phase.

        Only allows research tools and ready_to_summarize.
        Rejects finalization tool calls (must be in Serialize phase).

        Args:
            tool_calls: Tool calls from LLM response.
            state: Conversation state to update.

        Returns:
            True if ready_to_summarize was called, False otherwise.
        """
        for tc in tool_calls:
            # Check for ready_to_summarize signal
            if tc.name == "ready_to_summarize":
                result = self._ready_tool.execute(tc.arguments)
                state.add_tool_result(tc.id, result)
                return True

            # Reject finalization tool in discuss phase
            if tc.name == self._finalization_tool_name:
                state.add_tool_result(
                    tc.id,
                    json.dumps(
                        {
                            "error": "Cannot call finalization tool during discussion phase. "
                            "Use ready_to_summarize() when done discussing, then the "
                            "finalization tool will be available in the serialize phase."
                        }
                    ),
                )
                continue

            # Execute research tool
            tool = self._discuss_tools.get(tc.name)
            if tool is None:
                state.add_tool_result(tc.id, f"Error: Unknown tool '{tc.name}'")
                continue

            try:
                result = tool.execute(tc.arguments)
                state.add_tool_result(tc.id, result)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                state.add_tool_result(tc.id, f"Error executing {tc.name}: {e}")

        return False

    async def _summarize_phase(
        self,
        state: ConversationState,
        summary_prompt: str | None,
        on_assistant_message: Callable[[str], None] | None,
    ) -> str:
        """Run the Summarize phase with no tools.

        Generates a summary of the discussion for the Serialize phase.

        Args:
            state: Conversation state to update.
            summary_prompt: Optional guidance for summarization.
            on_assistant_message: Callback for assistant messages.

        Returns:
            The generated summary text.

        Raises:
            ConversationError: If summarization fails.
        """
        # Build summarize prompt
        if summary_prompt:
            summarize_instruction = summary_prompt
        else:
            summarize_instruction = (
                "Summarize the key decisions and creative direction from our discussion. "
                "Be concise but capture the essential elements that should inform the final output."
            )

        state.add_message({"role": "user", "content": summarize_instruction})

        # Call LLM with no tools
        response = await self._provider.complete(
            messages=state.messages,
            tools=None,
            tool_choice=None,
        )
        state.llm_calls += 1
        state.tokens_used += response.tokens_used

        if not response.content:
            raise ConversationError(
                "Summarize phase produced no content",
                state,
            )

        state.add_message({"role": "assistant", "content": response.content})
        if on_assistant_message is not None:
            on_assistant_message(response.content)

        return response.content

    async def _serialize_phase(
        self,
        state: ConversationState,
        summary: str,  # noqa: ARG002 - summary already in messages, param kept for API clarity
        validator: Callable[[dict[str, Any]], ValidationResult] | None,
        on_assistant_message: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        """Run the Serialize phase with finalization tool only.

        Converts the summary to structured output via tool call.
        Includes validation retry loop.

        Args:
            state: Conversation state to update.
            summary: Summary from previous phase.
            validator: Optional validation function.
            on_assistant_message: Callback for assistant messages.

        Returns:
            Validated artifact data.

        Raises:
            ConversationError: If serialization or validation fails.
        """
        # Build serialize instruction
        serialize_instruction = (
            f"Based on the summary above, call {self._finalization_tool_name}() "
            f"with the complete structured data."
        )
        state.add_message({"role": "user", "content": serialize_instruction})

        retries = 0
        while retries <= self._validation_retries:
            # Call LLM with finalization tool only, forced
            response = await self._provider.complete(
                messages=state.messages,
                tools=[self._finalization_tool_def],
                tool_choice=self._finalization_tool_name,
            )
            state.llm_calls += 1
            state.tokens_used += response.tokens_used

            # Add any assistant content
            if response.content:
                state.add_message({"role": "assistant", "content": response.content})
                if on_assistant_message is not None:
                    on_assistant_message(response.content)

            # Extract tool call
            tool_call = None
            if response.has_tool_calls and response.tool_calls:
                for tc in response.tool_calls:
                    if tc.name == self._finalization_tool_name:
                        tool_call = tc
                        break

            # No tool call = explicit failure (no YAML fallback)
            if tool_call is None:
                raise ConversationError(
                    f"Provider did not return {self._finalization_tool_name} tool call. "
                    "Ensure you are using a tool-capable provider/model.",
                    state,
                )

            data = tool_call.arguments

            # Validate if validator provided
            if validator is not None:
                result = validator(data)
                if not result.valid:
                    retries += 1
                    if retries > self._validation_retries:
                        raise ConversationError(
                            f"Validation failed after {self._validation_retries} retries",
                            state,
                        )

                    # Build feedback and retry
                    feedback = self._build_validation_feedback(
                        data, result, self._finalization_tool_name, include_schema=True
                    )
                    state.add_tool_result(tool_call.id, json.dumps(feedback, indent=2))
                    continue

                # Use validated/transformed data if provided
                data = result.data if result.data is not None else data

            # Validation passed - confirm and return
            result_message = self._finalization_tool.execute(data)
            state.add_tool_result(tool_call.id, result_message)
            return data

        # Should not reach here, but just in case
        raise ConversationError(
            f"Validation failed after {self._validation_retries} retries",
            state,
        )

    def _build_validation_feedback(
        self,
        data: dict[str, Any],
        result: ValidationResult,
        tool_name: str,
        include_schema: bool = False,
    ) -> dict[str, Any]:
        """Build structured validation feedback per ADR-007.

        Creates feedback optimized for LLM comprehension with:
        - Semantic result enum (not boolean)
        - Categorized issues (invalid/missing/unknown)
        - Field-specific requirements
        - Action instruction last (recency effect)
        - Optionally includes schema for serialize phase (small models)

        Args:
            data: The submitted data that failed validation.
            result: ValidationResult with structured errors.
            tool_name: Name of the finalization tool for action text.
            include_schema: If True, include full schema in feedback.

        Returns:
            Structured feedback dict ready for JSON serialization.
        """
        # Categorize errors into invalid vs missing
        invalid_fields: list[dict[str, Any]] = []
        missing_fields: list[dict[str, Any]] = []

        # Known Pydantic v2 error types for missing fields
        missing_error_types = {"missing", "value_error.missing"}

        if result.errors:
            for err in result.errors:
                # Determine if this is a "missing" or "invalid" error
                if err.error_type in missing_error_types:
                    is_missing = True
                else:
                    # Defensive fallback for unknown error types
                    issue_lower = err.issue.lower()
                    is_missing = "required" in issue_lower or "missing" in issue_lower

                if is_missing:
                    missing_fields.append(
                        {
                            "field": err.field,
                            "requirement": err.requirement or "required field",
                        }
                    )
                else:
                    invalid_fields.append(
                        {
                            "field": err.field,
                            "provided": err.provided,
                            "problem": err.issue,
                            "requirement": err.requirement or "see tool definition",
                        }
                    )

        # Detect unknown fields (submitted but not in expected schema)
        unknown_fields = self._detect_unknown_fields(data, result.expected_fields)

        # Calculate total issue count
        issue_count = len(invalid_fields) + len(missing_fields) + len(unknown_fields)

        # Build action text
        action_parts = [f"Call {tool_name}() with corrected data."]
        if unknown_fields:
            unknown_list = ", ".join(f"'{f}'" for f in unknown_fields[:3])
            if len(unknown_fields) > 3:
                unknown_list += f" and {len(unknown_fields) - 3} more"
            action_parts.append(
                f"Unknown fields {unknown_list} may be typos - use EXACT field names from schema below."
            )

        # Build feedback structure (ordered per ADR-007: result → issues → action)
        feedback: dict[str, Any] = {
            "result": "validation_failed",
            "issues": {
                "invalid": invalid_fields,
                "missing": missing_fields,
                "unknown": unknown_fields,
            },
            "issue_count": issue_count,
            "action": " ".join(action_parts),
        }

        # Include schema for serialize phase - helps small models
        if include_schema:
            feedback["schema"] = self._finalization_tool_def.parameters

        return feedback

    def _detect_unknown_fields(
        self,
        data: dict[str, Any],
        expected_fields: set[str] | None,
        prefix: str = "",
    ) -> list[str]:
        """Detect fields in data that are not in the expected schema.

        Recursively checks nested objects. Helps identify wrong field names
        (e.g., 'passages' instead of 'estimated_passages').

        Args:
            data: Submitted data to check.
            expected_fields: Set of valid field paths.
            prefix: Current field path prefix for recursion.

        Returns:
            List of unknown field paths.
        """
        if expected_fields is None:
            return []

        # Pre-compute parent prefixes for O(1) lookup
        parent_prefixes = {ef.rsplit(".", 1)[0] for ef in expected_fields if "." in ef}

        unknown: list[str] = []

        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key

            # Check if this exact path or a parent path is expected
            path_is_expected = field_path in expected_fields
            is_parent_of_expected = field_path in parent_prefixes

            if not path_is_expected and not is_parent_of_expected:
                unknown.append(field_path)
            elif isinstance(value, dict) and is_parent_of_expected:
                # Recurse into nested objects
                nested_unknown = self._detect_unknown_fields(value, expected_fields, field_path)
                unknown.extend(nested_unknown)

        return unknown
