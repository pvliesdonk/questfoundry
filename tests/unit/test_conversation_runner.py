"""Tests for ConversationRunner."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.conversation import (
    ConversationError,
    ConversationRunner,
    ConversationState,
    ValidationResult,
)
from questfoundry.providers.base import LLMResponse
from questfoundry.tools import ToolCall, ToolDefinition

# --- Helper Classes ---


class MockFinalizationTool:
    """Mock finalization tool for testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="submit_test",
            description="Submit test data",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                },
                "required": ["data"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        return "Submitted successfully"


class MockResearchTool:
    """Mock research tool for testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        return f"Search results for: {arguments.get('query', '')}"


# --- ConversationState Tests ---


def test_conversation_state_creation() -> None:
    """ConversationState can be created with initial messages."""
    messages = [{"role": "system", "content": "Hello"}]
    state = ConversationState(messages=messages)

    assert state.messages == messages
    assert state.turn_count == 0
    assert state.llm_calls == 0
    assert state.tokens_used == 0


def test_conversation_state_add_message() -> None:
    """ConversationState.add_message appends to messages."""
    state = ConversationState(messages=[])
    state.add_message({"role": "user", "content": "Hello"})

    assert len(state.messages) == 1
    assert state.messages[0]["content"] == "Hello"


def test_conversation_state_add_tool_result() -> None:
    """ConversationState.add_tool_result adds tool response message."""
    state = ConversationState(messages=[])
    state.add_tool_result("call_123", "Tool output")

    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "tool"
    assert state.messages[0]["tool_call_id"] == "call_123"
    assert state.messages[0]["content"] == "Tool output"


# --- ValidationResult Tests ---


def test_validation_result_valid() -> None:
    """ValidationResult with valid=True."""
    result = ValidationResult(valid=True, data={"key": "value"})

    assert result.valid is True
    assert result.error is None
    assert result.data == {"key": "value"}


def test_validation_result_invalid() -> None:
    """ValidationResult with valid=False."""
    result = ValidationResult(valid=False, error="Missing required field")

    assert result.valid is False
    assert result.error == "Missing required field"
    assert result.data is None


# --- ConversationRunner Initialization Tests ---


def test_runner_initialization() -> None:
    """ConversationRunner can be initialized."""
    provider = MagicMock()
    tools = [MockFinalizationTool()]

    runner = ConversationRunner(
        provider=provider,
        tools=tools,
        finalization_tool="submit_test",
        max_turns=5,
        validation_retries=2,
    )

    assert runner._max_turns == 5
    assert runner._validation_retries == 2
    assert runner._finalization_tool == "submit_test"


def test_runner_builds_tool_definitions() -> None:
    """ConversationRunner builds tool definitions list."""
    provider = MagicMock()
    tools = [MockFinalizationTool(), MockResearchTool()]

    runner = ConversationRunner(
        provider=provider,
        tools=tools,
        finalization_tool="submit_test",
    )

    assert len(runner._tool_definitions) == 2
    assert runner._tool_definitions[0].name == "submit_test"
    assert runner._tool_definitions[1].name == "search"


# --- ConversationRunner.run Tests ---


@pytest.mark.asyncio
async def test_runner_immediate_finalization() -> None:
    """Runner returns immediately when finalization tool is called."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "test_value"})
            ],
        )
    )

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
    )

    initial = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
    ]

    artifact, state = await runner.run(initial)

    assert artifact == {"data": "test_value"}
    assert state.llm_calls == 1
    assert state.tokens_used == 100


@pytest.mark.asyncio
async def test_runner_conversation_before_finalization() -> None:
    """Runner handles conversation turns before finalization."""
    # First call: assistant message (no tool call)
    # Second call: finalization tool
    responses = [
        LLMResponse(
            content="Let me think about this...",
            model="test",
            tokens_used=50,
            finish_reason="stop",
        ),
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "final"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
    )

    # No user input function - auto-continue
    artifact, state = await runner.run([{"role": "system", "content": "System"}])

    assert artifact == {"data": "final"}
    assert state.llm_calls == 2
    assert state.tokens_used == 150
    assert state.turn_count == 1


@pytest.mark.asyncio
async def test_runner_with_user_input() -> None:
    """Runner calls user_input_fn for each turn."""
    responses = [
        LLMResponse(
            content="What genre?",
            model="test",
            tokens_used=50,
            finish_reason="stop",
        ),
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "fantasy"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    user_inputs = ["Fantasy", None]  # Second call returns None
    input_index = [0]

    async def mock_user_input() -> str | None:
        idx = input_index[0]
        input_index[0] += 1
        return user_inputs[idx] if idx < len(user_inputs) else None

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
    )

    _artifact, state = await runner.run(
        [{"role": "system", "content": "System"}],
        user_input_fn=mock_user_input,
    )

    # Check user message was added
    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "Fantasy"


@pytest.mark.asyncio
async def test_runner_max_turns_exceeded() -> None:
    """Runner raises ConversationError when max turns exceeded."""
    # Always return content without tool calls
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="Still thinking...",
            model="test",
            tokens_used=20,
            finish_reason="stop",
        )
    )

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
        max_turns=3,
    )

    with pytest.raises(ConversationError) as exc_info:
        await runner.run([{"role": "system", "content": "System"}])

    assert "Maximum turns" in str(exc_info.value)
    assert exc_info.value.state is not None
    assert exc_info.value.state.turn_count == 3


@pytest.mark.asyncio
async def test_runner_research_tool_execution() -> None:
    """Runner executes research tools and adds results to conversation."""
    responses = [
        # First: research tool call
        LLMResponse(
            content="",
            model="test",
            tokens_used=50,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_search", name="search", arguments={"query": "noir"})
            ],
        ),
        # Second: finalization
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_final", name="submit_test", arguments={"data": "result"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool(), MockResearchTool()],
        finalization_tool="submit_test",
    )

    _artifact, state = await runner.run([{"role": "system", "content": "System"}])

    # Check tool result was added
    tool_messages = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1
    assert "Search results for: noir" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_runner_validation_success() -> None:
    """Runner uses validator and returns on success."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "valid"})
            ],
        )
    )

    def validator(data: dict) -> ValidationResult:
        if data.get("data") == "valid":
            return ValidationResult(valid=True, data=data)
        return ValidationResult(valid=False, error="Invalid data")

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
    )

    artifact, _state = await runner.run(
        [{"role": "system", "content": "System"}],
        validator=validator,
    )

    assert artifact == {"data": "valid"}


@pytest.mark.asyncio
async def test_runner_validation_retry() -> None:
    """Runner retries on validation failure."""
    # First attempt fails validation, second succeeds
    responses = [
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "invalid"})
            ],
        ),
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_2", name="submit_test", arguments={"data": "valid"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    call_count = [0]

    def validator(data: dict) -> ValidationResult:
        call_count[0] += 1
        if data.get("data") == "valid":
            return ValidationResult(valid=True, data=data)
        return ValidationResult(valid=False, error="Data must be 'valid'")

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
        validation_retries=3,
    )

    artifact, state = await runner.run(
        [{"role": "system", "content": "System"}],
        validator=validator,
    )

    assert artifact == {"data": "valid"}
    assert call_count[0] == 2  # Validator called twice
    assert state.llm_calls == 2  # Two LLM calls


@pytest.mark.asyncio
async def test_runner_validation_exhausted() -> None:
    """Runner raises ConversationError when validation retries exhausted."""
    # Always fails validation
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="submit_test", arguments={"data": "always_bad"})
            ],
        )
    )

    def validator(data: dict[str, Any]) -> ValidationResult:  # noqa: ARG001
        return ValidationResult(valid=False, error="Always fails")

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
        validation_retries=2,
    )

    with pytest.raises(ConversationError) as exc_info:
        await runner.run(
            [{"role": "system", "content": "System"}],
            validator=validator,
        )

    assert "Validation failed" in str(exc_info.value)
    assert "2 retries" in str(exc_info.value)


@pytest.mark.asyncio
async def test_runner_unknown_tool() -> None:
    """Runner handles unknown tool gracefully."""
    responses = [
        LLMResponse(
            content="",
            model="test",
            tokens_used=50,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="unknown_tool", arguments={})
            ],
        ),
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_2", name="submit_test", arguments={"data": "ok"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool()],
        finalization_tool="submit_test",
    )

    _artifact, state = await runner.run([{"role": "system", "content": "System"}])

    # Check error message was added
    tool_messages = [m for m in state.messages if m.get("role") == "tool"]
    assert any("Unknown tool" in m["content"] for m in tool_messages)


@pytest.mark.asyncio
async def test_runner_tool_execution_error() -> None:
    """Runner handles tool execution errors gracefully."""

    class FailingTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="failing", description="Always fails")

        def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
            raise ValueError("Tool error")

    responses = [
        LLMResponse(
            content="",
            model="test",
            tokens_used=50,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", name="failing", arguments={})
            ],
        ),
        LLMResponse(
            content="",
            model="test",
            tokens_used=100,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_2", name="submit_test", arguments={"data": "ok"})
            ],
        ),
    ]

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=responses)

    runner = ConversationRunner(
        provider=provider,
        tools=[MockFinalizationTool(), FailingTool()],
        finalization_tool="submit_test",
    )

    _artifact, state = await runner.run([{"role": "system", "content": "System"}])

    # Check error message was added
    tool_messages = [m for m in state.messages if m.get("role") == "tool"]
    assert any("Error executing" in m["content"] for m in tool_messages)


# --- ConversationError Tests ---


def test_conversation_error_with_state() -> None:
    """ConversationError preserves state."""
    state = ConversationState(messages=[{"role": "user", "content": "test"}])
    error = ConversationError("Test error", state)

    assert str(error) == "Test error"
    assert error.state is state
    assert len(error.state.messages) == 1


def test_conversation_error_without_state() -> None:
    """ConversationError works without state."""
    error = ConversationError("Test error")

    assert str(error) == "Test error"
    assert error.state is None
