"""Tests for ConversationRunner and related types."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from questfoundry.conversation import (
    ConversationError,
    ConversationRunner,
    ConversationState,
    ValidationErrorDetail,
    ValidationResult,
)
from questfoundry.providers.base import LLMResponse
from questfoundry.tools import ToolCall, ToolDefinition

# --- Test Fixtures ---


class MockTool:
    """A simple mock tool for testing."""

    def __init__(self, name: str = "mock_tool", result: str = "Mock result") -> None:
        self._name = name
        self._result = result

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="A mock tool for testing",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        return self._result


class FinalizationTool:
    """A mock finalization tool for testing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="submit_test",
            description="Submit test data",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        return "Test submitted successfully."


def create_mock_provider(responses: list[LLMResponse]) -> Mock:
    """Create a mock LLM provider that returns given responses in sequence."""
    provider = Mock()
    provider.complete = AsyncMock(side_effect=responses)
    provider.default_model = "test-model"
    return provider


# --- ConversationState Tests ---


def test_conversation_state_init() -> None:
    """ConversationState initializes with given messages."""
    messages = [{"role": "system", "content": "Hello"}]
    state = ConversationState(messages=messages)

    assert state.messages == messages
    assert state.turn_count == 0
    assert state.tokens_used == 0
    assert state.llm_calls == 0


def test_conversation_state_add_message() -> None:
    """add_message appends to messages list."""
    state = ConversationState(messages=[])
    state.add_message({"role": "user", "content": "Test"})

    assert len(state.messages) == 1
    assert state.messages[0]["content"] == "Test"


def test_conversation_state_add_tool_result() -> None:
    """add_tool_result adds tool message with call_id."""
    state = ConversationState(messages=[])
    state.add_tool_result("call_123", "Result text")

    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "tool"
    assert state.messages[0]["content"] == "Result text"
    assert state.messages[0]["tool_call_id"] == "call_123"


# --- ValidationResult Tests ---


def test_validation_result_valid() -> None:
    """ValidationResult with valid=True and data."""
    result = ValidationResult(valid=True, data={"key": "value"})

    assert result.valid
    assert result.data == {"key": "value"}
    assert result.error is None
    assert result.errors is None


def test_validation_result_invalid_with_error() -> None:
    """ValidationResult with valid=False and error string."""
    result = ValidationResult(valid=False, error="Something went wrong")

    assert not result.valid
    assert result.error == "Something went wrong"


def test_validation_result_invalid_with_structured_errors() -> None:
    """ValidationResult with valid=False and structured errors."""
    errors = [
        ValidationErrorDetail(field="name", issue="is required", provided=None),
        ValidationErrorDetail(field="age", issue="must be positive", provided=-5),
    ]
    result = ValidationResult(valid=False, errors=errors)

    assert not result.valid
    assert len(result.errors) == 2
    assert result.errors[0].field == "name"
    assert result.errors[1].provided == -5


# --- ValidationErrorDetail Tests ---


def test_validation_error_detail() -> None:
    """ValidationErrorDetail holds field, issue, and provided value."""
    error = ValidationErrorDetail(field="genre", issue="cannot be empty", provided="")

    assert error.field == "genre"
    assert error.issue == "cannot be empty"
    assert error.provided == ""


def test_validation_error_detail_default_provided() -> None:
    """ValidationErrorDetail defaults provided to None."""
    error = ValidationErrorDetail(field="test", issue="error")

    assert error.provided is None


# --- ConversationRunner Initialization Tests ---


def test_runner_init_validates_tools() -> None:
    """ConversationRunner validates tools implement Tool protocol."""
    provider = create_mock_provider([])

    # Valid tool should work
    runner = ConversationRunner(
        provider=provider,
        tools=[MockTool()],
        finalization_tool="submit_test",
    )
    assert runner._finalization_tool == "submit_test"


def test_runner_init_rejects_invalid_tool() -> None:
    """ConversationRunner rejects tools that don't implement protocol."""
    provider = create_mock_provider([])

    class InvalidTool:
        pass  # Missing definition and execute

    with pytest.raises(TypeError, match="doesn't implement Tool protocol"):
        ConversationRunner(
            provider=provider,
            tools=[InvalidTool()],  # type: ignore[list-item]
            finalization_tool="submit_test",
        )


def test_runner_init_default_config() -> None:
    """ConversationRunner uses default max_turns and validation_retries."""
    provider = create_mock_provider([])
    runner = ConversationRunner(
        provider=provider,
        tools=[MockTool()],
        finalization_tool="submit_test",
    )

    assert runner._max_turns == 10
    assert runner._validation_retries == 3


def test_runner_init_custom_config() -> None:
    """ConversationRunner accepts custom max_turns and validation_retries."""
    provider = create_mock_provider([])
    runner = ConversationRunner(
        provider=provider,
        tools=[MockTool()],
        finalization_tool="submit_test",
        max_turns=5,
        validation_retries=2,
    )

    assert runner._max_turns == 5
    assert runner._validation_retries == 2


# --- ConversationRunner.run() Tests ---


@pytest.mark.asyncio
async def test_runner_simple_finalization() -> None:
    """Runner completes when finalization tool is called with valid data."""
    tool_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "test"})
    response = LLMResponse(
        content="Here's my submission",
        model="test",
        tokens_used=100,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    provider = create_mock_provider([response])

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "test"}
    assert state.llm_calls == 1
    assert state.tokens_used == 100


@pytest.mark.asyncio
async def test_runner_with_validation_success() -> None:
    """Runner validates finalization data when validator provided."""
    tool_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "test"})
    response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    provider = create_mock_provider([response])

    def validator(data: dict[str, Any]) -> ValidationResult:
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    result, _state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    assert result == {"value": "test"}


@pytest.mark.asyncio
async def test_runner_validation_retry_success() -> None:
    """Runner retries validation when it fails, then succeeds."""
    # First attempt fails, second succeeds
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})

    first_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    second_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider([first_response, second_response])

    call_count = 0

    def validator(data: dict[str, Any]) -> ValidationResult:
        nonlocal call_count
        call_count += 1
        if data.get("value") == "":
            return ValidationResult(
                valid=False,
                errors=[ValidationErrorDetail(field="value", issue="cannot be empty", provided="")],
            )
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    assert result == {"value": "fixed"}
    assert call_count == 2
    assert state.llm_calls == 2


@pytest.mark.asyncio
async def test_runner_validation_exhausts_retries() -> None:
    """Runner raises ConversationError after exhausting validation retries."""
    tool_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    # Always return failing response
    provider = create_mock_provider([response, response, response, response])

    def validator(_data: dict[str, Any]) -> ValidationResult:
        return ValidationResult(
            valid=False,
            errors=[ValidationErrorDetail(field="value", issue="cannot be empty", provided="")],
        )

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
        validation_retries=2,
    )

    with pytest.raises(ConversationError, match="Validation failed after 2 retries"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
            validator=validator,
        )


@pytest.mark.asyncio
async def test_runner_max_turns_exceeded() -> None:
    """Runner raises ConversationError when max turns exceeded."""
    # Response without tool calls - conversation continues
    response = LLMResponse(
        content="Let me think...",
        model="test",
        tokens_used=20,
        finish_reason="stop",
        tool_calls=None,
    )
    provider = create_mock_provider([response] * 5)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
        max_turns=3,
    )

    with pytest.raises(ConversationError, match=r"Maximum turns.*exceeded"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
        )


@pytest.mark.asyncio
async def test_runner_llm_fails_to_call_tool() -> None:
    """Runner raises ConversationError when LLM doesn't call finalization on retry."""
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    first_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    # Second response has no tool calls
    second_response = LLMResponse(
        content="I can't do that",
        model="test",
        tokens_used=30,
        finish_reason="stop",
        tool_calls=None,
    )
    provider = create_mock_provider([first_response, second_response])

    def validator(_data: dict[str, Any]) -> ValidationResult:
        return ValidationResult(
            valid=False,
            errors=[ValidationErrorDetail(field="value", issue="error", provided="")],
        )

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    with pytest.raises(ConversationError, match=r"LLM failed to call"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
            validator=validator,
        )


@pytest.mark.asyncio
async def test_runner_executes_research_tool() -> None:
    """Runner executes non-finalization tools and continues."""
    research_call = ToolCall(id="call_1", name="research", arguments={"query": "test"})
    finalization_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "done"})

    first_response = LLMResponse(
        content="Let me research",
        model="test",
        tokens_used=30,
        finish_reason="tool_calls",
        tool_calls=[research_call],
    )
    second_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalization_call],
    )
    provider = create_mock_provider([first_response, second_response])

    research_tool = MockTool(name="research", result="Research results")

    runner = ConversationRunner(
        provider=provider,
        tools=[research_tool, FinalizationTool()],
        finalization_tool="submit_test",
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "done"}
    assert state.llm_calls == 2
    # Check research tool result was added
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1


@pytest.mark.asyncio
async def test_runner_calls_on_assistant_message() -> None:
    """Runner calls on_assistant_message callback with content."""
    tool_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "test"})
    response = LLMResponse(
        content="Here's my response",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    provider = create_mock_provider([response])

    messages_received: list[str] = []

    def on_message(content: str) -> None:
        messages_received.append(content)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        on_assistant_message=on_message,
    )

    assert "Here's my response" in messages_received


@pytest.mark.asyncio
async def test_runner_tool_execution_error() -> None:
    """Runner handles tool execution errors gracefully."""

    class FailingTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="failing", description="Fails", parameters={})

        def execute(self, _arguments: dict[str, Any]) -> str:
            raise RuntimeError("Tool crashed")

    failing_call = ToolCall(id="call_1", name="failing", arguments={})
    finalization_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "done"})

    first_response = LLMResponse(
        content="",
        model="test",
        tokens_used=30,
        finish_reason="tool_calls",
        tool_calls=[failing_call],
    )
    second_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalization_call],
    )
    provider = create_mock_provider([first_response, second_response])

    runner = ConversationRunner(
        provider=provider,
        tools=[FailingTool(), FinalizationTool()],
        finalization_tool="submit_test",
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    # Should complete despite tool error
    assert result == {"value": "done"}
    # Error should be in tool results
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert any("Error executing" in m.get("content", "") for m in tool_results)


# --- ConversationError Tests ---


def test_conversation_error_with_state() -> None:
    """ConversationError preserves state."""
    state = ConversationState(messages=[{"role": "user", "content": "Test"}])
    state.llm_calls = 5

    error = ConversationError("Something failed", state)

    assert str(error) == "Something failed"
    assert error.state is not None
    assert error.state.llm_calls == 5


def test_conversation_error_without_state() -> None:
    """ConversationError works without state."""
    error = ConversationError("Something failed")

    assert str(error) == "Something failed"
    assert error.state is None


# --- expected_fields in Feedback Tests ---


def test_validation_result_expected_fields() -> None:
    """ValidationResult supports expected_fields attribute."""
    result = ValidationResult(
        valid=False,
        error="Missing fields",
        expected_fields=["name", "age", "email"],
    )

    assert result.expected_fields == ["name", "age", "email"]


def test_validation_result_expected_fields_default() -> None:
    """ValidationResult defaults expected_fields to None."""
    result = ValidationResult(valid=True)

    assert result.expected_fields is None


@pytest.mark.asyncio
async def test_runner_feedback_includes_expected_fields() -> None:
    """Runner includes expected_fields in validation feedback when provided."""
    import json

    # First attempt fails with expected_fields, second succeeds
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})

    first_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    second_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider([first_response, second_response])

    def validator(data: dict[str, Any]) -> ValidationResult:
        if data.get("value") == "":
            return ValidationResult(
                valid=False,
                errors=[ValidationErrorDetail(field="value", issue="cannot be empty", provided="")],
                expected_fields=["value", "description", "priority"],
            )
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    _result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    # Find the tool result message with feedback
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1

    # Parse the feedback JSON
    feedback_msg = tool_results[0]["content"]
    feedback = json.loads(feedback_msg)

    assert "expected_fields" in feedback
    assert feedback["expected_fields"] == ["value", "description", "priority"]


@pytest.mark.asyncio
async def test_runner_feedback_omits_expected_fields_when_none() -> None:
    """Runner omits expected_fields from feedback when validator doesn't provide it."""
    import json

    # First attempt fails without expected_fields, second succeeds
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})

    first_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    second_response = LLMResponse(
        content="",
        model="test",
        tokens_used=50,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider([first_response, second_response])

    def validator(data: dict[str, Any]) -> ValidationResult:
        if data.get("value") == "":
            return ValidationResult(
                valid=False,
                errors=[ValidationErrorDetail(field="value", issue="cannot be empty", provided="")],
                # No expected_fields
            )
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        tools=[FinalizationTool()],
        finalization_tool="submit_test",
    )

    _result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    # Find the tool result message with feedback
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1

    # Parse the feedback JSON
    feedback_msg = tool_results[0]["content"]
    feedback = json.loads(feedback_msg)

    # expected_fields should not be present when not provided
    assert "expected_fields" not in feedback
