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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from questfoundry.providers.base import LLMProvider, Message
    from questfoundry.tools import Tool, ToolCall


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
class ValidationResult:
    """Result of validating tool call arguments.

    Attributes:
        valid: Whether validation passed.
        error: Error message if validation failed.
        data: Validated/transformed data if validation passed.
    """

    valid: bool
    error: str | None = None
    data: dict[str, Any] | None = None


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
        max_turns: int = 10,
        validation_retries: int = 3,
    ) -> None:
        """Initialize the conversation runner.

        Args:
            provider: LLM provider for completions.
            tools: List of tools available during conversation.
            finalization_tool: Name of the tool that ends the conversation.
            max_turns: Maximum turns before timeout (default 10).
            validation_retries: Max validation retries (default 3).
        """
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
    ) -> tuple[dict[str, Any], ConversationState]:
        """Run the conversation until finalization.

        Args:
            initial_messages: Starting messages (typically system + user).
            user_input_fn: Async function to get user input. If None or
                returns None/empty, conversation continues without user input.
            validator: Optional function to validate finalization data.
                Called with tool arguments, returns ValidationResult.

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

            # Add assistant message if there's content
            if response.content:
                state.add_message({"role": "assistant", "content": response.content})

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
                    # Validate-with-feedback pattern: structured error response
                    # Lets LLM understand exactly what failed and how to fix it
                    feedback = {
                        "success": False,
                        "error": "Validation failed for submitted artifact",
                        "error_count": len(result.error.split(";")) if result.error else 1,
                        "invalid_fields": [],
                        "missing_fields": [],
                        "submitted_data": data,
                        "hint": f"Call {self._finalization_tool}() again with corrected data. Fix only the errors listed.",
                    }

                    # Parse validation errors into structured format
                    if result.error:
                        for err in result.error.replace("Validation errors: ", "").split("; "):
                            if ": " in err:
                                field, issue = err.split(": ", 1)
                                if "required" in issue.lower() or "missing" in issue.lower():
                                    feedback["missing_fields"].append(field)
                                else:
                                    feedback["invalid_fields"].append(
                                        {
                                            "field": field,
                                            "provided": data.get(field.split(".")[0]),
                                            "issue": issue,
                                        }
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
                    if response.has_tool_calls and response.tool_calls:
                        for tc in response.tool_calls:
                            if tc.name == self._finalization_tool:
                                data = tc.arguments
                                tool_call = tc
                                break
                    retries += 1
                    continue
                else:
                    # Use validated/transformed data if provided
                    data = result.data if result.data is not None else data

            # Validation passed or no validator
            state.add_tool_result(tool_call.id, "Artifact submitted successfully.")
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
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"
