"""Conversation runner for interactive stages.

This module provides the ConversationRunner class that manages
multi-turn LLM interactions with tool calling support, enabling
conversational refinement before structured output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from questfoundry.conversation.state import ConversationState
from questfoundry.tools import Tool

# Default configuration values
DEFAULT_MAX_TURNS = 10
DEFAULT_VALIDATION_RETRIES = 3

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
    """Manages multi-turn LLM conversations with tool calling.

    The runner implements the Discuss → Freeze → Serialize pattern:
    1. Conversation phase: LLM discusses with user, uses research tools
    2. Finalization: LLM calls finalization tool with structured data
    3. Validation: Data is validated with optional retry loop

    Attributes:
        finalization_tool: Name of the tool that signals completion.
        max_turns: Maximum conversation turns before timeout.
        validation_retries: Maximum validation retry attempts.

    Example:
        >>> runner = ConversationRunner(
        ...     provider=provider,
        ...     tools=[SubmitDreamTool(), SearchCorpusTool()],
        ...     finalization_tool="submit_dream",
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
        tools: list[Tool],
        finalization_tool: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    ) -> None:
        """Initialize the conversation runner.

        Args:
            provider: LLM provider for completions.
            tools: List of tools available during conversation.
            finalization_tool: Name of the tool that ends the conversation.
            max_turns: Maximum turns before timeout (default 10).
            validation_retries: Max validation retries (default 3).

        Raises:
            TypeError: If any tool doesn't implement the Tool protocol.
        """
        # Validate all tools implement the Tool protocol
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    f"Tool {tool!r} doesn't implement Tool protocol "
                    f"(missing 'definition' property or 'execute' method)"
                )

        self._provider = provider
        self._tools = {t.definition.name: t for t in tools}
        self._tool_definitions = [t.definition for t in tools]
        self._finalization_tool = finalization_tool
        self._max_turns = max_turns
        self._validation_retries = validation_retries

    async def run(
        self,
        initial_messages: list[Message],
        user_input_fn: Callable[[], Awaitable[str | None]] | None = None,
        validator: Callable[[dict[str, Any]], ValidationResult] | None = None,
        on_assistant_message: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any], ConversationState]:
        """Run the conversation until finalization.

        Args:
            initial_messages: Starting messages (typically system + user).
            user_input_fn: Async function to get user input. If None or
                returns None/empty, conversation continues without user input.
            validator: Optional function to validate finalization data.
                Called with tool arguments, returns ValidationResult.
            on_assistant_message: Optional callback for assistant messages.
                Called with the message content for display purposes.

        Returns:
            Tuple of (artifact_data, conversation_state).

        Raises:
            ConversationError: If max turns exceeded or validation fails
                after all retries.
        """
        state = ConversationState(messages=list(initial_messages))

        while state.turn_count < self._max_turns:
            # Call LLM with tools
            response = await self._provider.complete(
                messages=state.messages,
                tools=self._tool_definitions,
                tool_choice="auto",
            )
            state.llm_calls += 1
            state.tokens_used += response.tokens_used

            # Add assistant message BEFORE processing tool calls
            # This preserves any explanatory text the LLM provides alongside tool calls
            if response.content:
                state.add_message({"role": "assistant", "content": response.content})
                if on_assistant_message is not None:
                    on_assistant_message(response.content)

            # Handle tool calls
            if response.has_tool_calls:
                result = await self._handle_tool_calls(
                    response.tool_calls or [],
                    state,
                    validator,
                )
                if result is not None:
                    # Finalization tool was called and validated
                    return result, state

            # Check if we should get user input
            if user_input_fn is not None:
                user_input = await user_input_fn()
                if user_input:
                    state.add_message({"role": "user", "content": user_input})

            state.turn_count += 1

        raise ConversationError(
            f"Maximum turns ({self._max_turns}) exceeded without finalization",
            state,
        )

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        state: ConversationState,
        validator: Callable[[dict[str, Any]], ValidationResult] | None,
    ) -> dict[str, Any] | None:
        """Process tool calls from LLM response.

        Args:
            tool_calls: List of tool calls from LLM.
            state: Current conversation state.
            validator: Optional validator for finalization tool.

        Returns:
            Finalization data if finalization tool was called successfully,
            None otherwise.
        """
        for tc in tool_calls:
            if tc.name == self._finalization_tool:
                # Handle finalization with validation
                return await self._handle_finalization(tc, state, validator)
            else:
                # Execute research tool
                result = self._execute_tool(tc)
                state.add_tool_result(tc.id, result)

        return None

    async def _handle_finalization(
        self,
        tool_call: ToolCall,
        state: ConversationState,
        validator: Callable[[dict[str, Any]], ValidationResult] | None,
    ) -> dict[str, Any]:
        """Handle finalization tool call with validation retry.

        Args:
            tool_call: The finalization tool call.
            state: Current conversation state.
            validator: Optional validator function.

        Returns:
            Validated artifact data.

        Raises:
            ConversationError: If validation fails after all retries.
        """
        data = tool_call.arguments
        retries = 0

        while retries < self._validation_retries:
            # Validate if validator provided
            if validator is not None:
                result = validator(data)
                if not result.valid:
                    # Build structured feedback per ADR-007
                    feedback = self._build_validation_feedback(
                        data, result, self._finalization_tool
                    )

                    # Check if we've exhausted retries BEFORE making another LLM call
                    retries += 1
                    if retries >= self._validation_retries:
                        raise ConversationError(
                            f"Validation failed after {self._validation_retries} retries",
                            state,
                        )

                    state.add_tool_result(tool_call.id, json.dumps(feedback, indent=2))

                    # Request retry from LLM
                    response = await self._provider.complete(
                        messages=state.messages,
                        tools=self._tool_definitions,
                        tool_choice=self._finalization_tool,  # Force retry of same tool
                    )
                    state.llm_calls += 1
                    state.tokens_used += response.tokens_used

                    # Extract new tool call
                    found_new_call = False
                    if response.has_tool_calls and response.tool_calls:
                        for tc in response.tool_calls:
                            if tc.name == self._finalization_tool:
                                data = tc.arguments
                                tool_call = tc
                                found_new_call = True
                                break

                    if not found_new_call:
                        # LLM didn't provide expected tool call - fail fast
                        # Continuing would revalidate same stale data (infinite loop)
                        if response.content:
                            state.add_message({"role": "assistant", "content": response.content})
                        raise ConversationError(
                            f"LLM failed to call {self._finalization_tool} on retry {retries}",
                            state,
                        )
                    continue
                else:
                    # Use validated/transformed data if provided
                    data = result.data if result.data is not None else data

            # Validation passed or no validator - call tool.execute() for confirmation
            tool = self._tools.get(self._finalization_tool)
            if tool is not None:
                result_message = tool.execute(data)
            else:
                result_message = "Artifact submitted successfully."
            state.add_tool_result(tool_call.id, result_message)
            return data

        raise ConversationError(
            f"Validation failed after {self._validation_retries} retries",
            state,
        )

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a non-finalization tool.

        Args:
            tool_call: The tool call to execute.

        Returns:
            Tool execution result as string.
        """
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return f"Error: Unknown tool '{tool_call.name}'"

        try:
            return tool.execute(tool_call.arguments)
        except (KeyboardInterrupt, SystemExit):
            raise  # Don't catch system signals
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def _build_validation_feedback(
        self,
        data: dict[str, Any],
        result: ValidationResult,
        tool_name: str,
    ) -> dict[str, Any]:
        """Build structured validation feedback per ADR-007.

        Creates feedback optimized for LLM comprehension with:
        - Semantic result enum (not boolean)
        - Categorized issues (invalid/missing/unknown)
        - Field-specific requirements
        - Action instruction last (recency effect)

        Args:
            data: The submitted data that failed validation.
            result: ValidationResult with structured errors.
            tool_name: Name of the finalization tool for action text.

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
                f"Unknown fields {unknown_list} may be typos - check tool definition for correct names."
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
            expected_fields: Set of valid field paths (e.g., {"genre", "scope.target_word_count"}).
            prefix: Current field path prefix for recursion.

        Returns:
            List of unknown field paths.
        """
        if expected_fields is None:
            return []

        # Pre-compute parent prefixes for O(1) lookup instead of O(m) per field
        parent_prefixes = {ef.rsplit(".", 1)[0] for ef in expected_fields if "." in ef}

        unknown: list[str] = []

        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key

            # Check if this exact path or a parent path is expected
            # e.g., "scope" is expected if "scope.target_word_count" is in expected_fields
            path_is_expected = field_path in expected_fields
            is_parent_of_expected = field_path in parent_prefixes

            if not path_is_expected and not is_parent_of_expected:
                unknown.append(field_path)
            elif isinstance(value, dict) and is_parent_of_expected:
                # Recurse into nested objects
                nested_unknown = self._detect_unknown_fields(value, expected_fields, field_path)
                unknown.extend(nested_unknown)

        return unknown
