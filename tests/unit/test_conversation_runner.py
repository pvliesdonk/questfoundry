"""Tests for ConversationRunner 3-phase pattern."""

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


class MockResearchTool:
    """A simple mock research tool for testing."""

    def __init__(self, name: str = "research_tool", result: str = "Research result") -> None:
        self._name = name
        self._result = result

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="A mock research tool for testing",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        return self._result


class MockFinalizationTool:
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
    """ConversationState initializes with given messages and phase."""
    messages = [{"role": "system", "content": "Hello"}]
    state = ConversationState(messages=messages, phase="discuss")

    assert state.messages == messages
    assert state.turn_count == 0
    assert state.tokens_used == 0
    assert state.llm_calls == 0
    assert state.phase == "discuss"


def test_conversation_state_default_phase() -> None:
    """ConversationState defaults to 'discuss' phase."""
    state = ConversationState(messages=[])
    assert state.phase == "discuss"


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


def test_runner_init_validates_finalization_tool() -> None:
    """ConversationRunner validates finalization_tool implements Tool protocol."""
    provider = create_mock_provider([])

    # Valid finalization tool should work
    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
    )
    assert runner._finalization_tool_name == "submit_test"


def test_runner_init_rejects_invalid_finalization_tool() -> None:
    """ConversationRunner rejects finalization_tool that doesn't implement protocol."""
    provider = create_mock_provider([])

    class InvalidTool:
        pass  # Missing definition and execute

    with pytest.raises(TypeError, match="doesn't implement Tool protocol"):
        ConversationRunner(
            provider=provider,
            research_tools=[],
            finalization_tool=InvalidTool(),  # type: ignore[arg-type]
        )


def test_runner_init_validates_research_tools() -> None:
    """ConversationRunner validates research tools implement Tool protocol."""
    provider = create_mock_provider([])

    class InvalidTool:
        pass

    with pytest.raises(TypeError, match="doesn't implement Tool protocol"):
        ConversationRunner(
            provider=provider,
            research_tools=[InvalidTool()],  # type: ignore[list-item]
            finalization_tool=MockFinalizationTool(),
        )


def test_runner_init_default_config() -> None:
    """ConversationRunner uses default max_discuss_turns and validation_retries."""
    provider = create_mock_provider([])
    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
    )

    assert runner._max_discuss_turns == 10
    assert runner._validation_retries == 3


def test_runner_init_custom_config() -> None:
    """ConversationRunner accepts custom max_discuss_turns and validation_retries."""
    provider = create_mock_provider([])
    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=5,
        validation_retries=2,
    )

    assert runner._max_discuss_turns == 5
    assert runner._validation_retries == 2


# --- 3-Phase Pattern Tests ---


@pytest.mark.asyncio
async def test_runner_complete_3phase_flow() -> None:
    """Runner executes all 3 phases: Discuss → Summarize → Serialize."""
    # Phase 1: Discuss - LLM responds, no tool calls
    discuss_response = LLMResponse(
        content="I understand your story idea.",
        model="test",
        tokens_used=50,
        finish_reason="stop",
        tool_calls=None,
    )
    # Phase 2: Summarize - LLM generates summary
    summarize_response = LLMResponse(
        content="Summary: A fantasy story about heroism.",
        model="test",
        tokens_used=30,
        finish_reason="stop",
        tool_calls=None,
    )
    # Phase 3: Serialize - LLM calls finalization tool
    finalize_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "done"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,  # Direct mode
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "done"}
    assert state.llm_calls == 3  # Discuss + Summarize + Serialize
    assert state.phase == "serialize"


@pytest.mark.asyncio
async def test_runner_discuss_phase_with_research_tool() -> None:
    """Discuss phase executes research tools and continues conversation."""
    # Discuss: LLM calls research tool
    research_call = ToolCall(id="call_1", name="research", arguments={"query": "fantasy"})
    discuss_response1 = LLMResponse(
        content="Let me research that.",
        model="test",
        tokens_used=20,
        finish_reason="tool_calls",
        tool_calls=[research_call],
    )
    # Discuss: LLM responds after research (max_turns=1 reached)
    discuss_response2 = LLMResponse(
        content="Based on research...",
        model="test",
        tokens_used=30,
        finish_reason="stop",
        tool_calls=None,
    )
    # Summarize
    summarize_response = LLMResponse(
        content="Summary from research.",
        model="test",
        tokens_used=20,
        finish_reason="stop",
    )
    # Serialize
    finalize_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "researched"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider(
        [discuss_response1, discuss_response2, summarize_response, serialize_response]
    )

    research_tool = MockResearchTool(name="research", result="Fantasy genre info")

    runner = ConversationRunner(
        provider=provider,
        research_tools=[research_tool],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "researched"}
    # Check research tool was executed
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert any("Fantasy genre info" in m.get("content", "") for m in tool_results)


@pytest.mark.asyncio
async def test_runner_discuss_phase_ready_to_summarize_tool() -> None:
    """LLM calling ready_to_summarize() triggers transition to Summarize phase."""
    # Discuss: LLM calls ready_to_summarize
    ready_call = ToolCall(id="call_1", name="ready_to_summarize", arguments={})
    discuss_response = LLMResponse(
        content="I'm ready to summarize.",
        model="test",
        tokens_used=20,
        finish_reason="tool_calls",
        tool_calls=[ready_call],
    )
    # Summarize
    summarize_response = LLMResponse(
        content="Summary of discussion.",
        model="test",
        tokens_used=30,
        finish_reason="stop",
    )
    # Serialize
    finalize_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "via_ready"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=10,  # Many turns available
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "via_ready"}
    # Should have transitioned early, not used all turns
    assert state.turn_count < 10


@pytest.mark.asyncio
async def test_runner_discuss_phase_user_done_signal() -> None:
    """User typing /done triggers transition to Summarize phase."""
    # Discuss: LLM responds
    discuss_response = LLMResponse(
        content="What else would you like?",
        model="test",
        tokens_used=20,
        finish_reason="stop",
        tool_calls=None,
    )
    # Summarize
    summarize_response = LLMResponse(
        content="Summary.",
        model="test",
        tokens_used=30,
        finish_reason="stop",
    )
    # Serialize
    finalize_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "user_done"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    # User input function that returns /done
    async def user_input():
        return "/done"

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=10,
    )

    result, _state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        user_input_fn=user_input,
    )

    assert result == {"value": "user_done"}


@pytest.mark.asyncio
async def test_runner_discuss_phase_rejects_finalization_tool() -> None:
    """Finalization tool called in Discuss phase is rejected with error message."""
    # Discuss: LLM incorrectly calls finalization tool
    wrong_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "too_early"})
    discuss_response1 = LLMResponse(
        content="Let me submit directly.",
        model="test",
        tokens_used=20,
        finish_reason="tool_calls",
        tool_calls=[wrong_call],
    )
    # Discuss continues (max turns reached)
    discuss_response2 = LLMResponse(
        content="Okay, understood.",
        model="test",
        tokens_used=15,
        finish_reason="stop",
        tool_calls=None,
    )
    # Summarize
    summarize_response = LLMResponse(
        content="Summary.",
        model="test",
        tokens_used=30,
        finish_reason="stop",
    )
    # Serialize - now finalization is allowed
    finalize_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "correct"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider(
        [discuss_response1, discuss_response2, summarize_response, serialize_response]
    )

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "correct"}
    # Check error was returned for premature finalization
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert any("Cannot call finalization tool" in m.get("content", "") for m in tool_results)


@pytest.mark.asyncio
async def test_runner_summarize_phase_no_tools() -> None:
    """Summarize phase has no tools available."""
    # Discuss
    discuss_response = LLMResponse(
        content="Discussion done.",
        model="test",
        tokens_used=20,
        finish_reason="stop",
    )
    # Summarize - just text, no tools
    summarize_response = LLMResponse(
        content="Here is my summary of our discussion.",
        model="test",
        tokens_used=50,
        finish_reason="stop",
    )
    # Serialize
    finalize_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "done"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    runner = ConversationRunner(
        provider=provider,
        research_tools=[MockResearchTool()],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
    )

    assert result == {"value": "done"}
    # Summarize phase should have added the summary to messages
    assert any("summary" in m.get("content", "").lower() for m in state.messages)


@pytest.mark.asyncio
async def test_runner_serialize_phase_no_tool_call_fails() -> None:
    """Serialize phase without tool call raises ConversationError."""
    # Discuss
    discuss_response = LLMResponse(
        content="Done.",
        model="test",
        tokens_used=10,
        finish_reason="stop",
    )
    # Summarize
    summarize_response = LLMResponse(
        content="Summary.",
        model="test",
        tokens_used=20,
        finish_reason="stop",
    )
    # Serialize - no tool call!
    serialize_response = LLMResponse(
        content="I can't call the tool.",
        model="test",
        tokens_used=30,
        finish_reason="stop",
        tool_calls=None,
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    with pytest.raises(ConversationError, match=r"did not return.*tool call"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
        )


# --- Validation Tests ---


@pytest.mark.asyncio
async def test_runner_serialize_validation_success() -> None:
    """Serialize phase validates data when validator provided."""
    # Discuss
    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    # Summarize
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    # Serialize
    finalize_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "valid"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    def validator(data: dict[str, Any]) -> ValidationResult:
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    result, _ = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    assert result == {"value": "valid"}


@pytest.mark.asyncio
async def test_runner_serialize_validation_retry_success() -> None:
    """Serialize phase retries on validation failure."""
    # Discuss
    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    # Summarize
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    # Serialize - first attempt fails
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    serialize_response1 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    # Serialize - second attempt succeeds
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})
    serialize_response2 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider(
        [discuss_response, summarize_response, serialize_response1, serialize_response2]
    )

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
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    assert result == {"value": "fixed"}
    assert call_count == 2
    assert state.llm_calls == 4  # Discuss + Summarize + 2x Serialize


@pytest.mark.asyncio
async def test_runner_serialize_validation_exhausts_retries() -> None:
    """Serialize phase raises error after exhausting validation retries."""
    # Discuss
    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    # Summarize
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    # Serialize - always fails
    bad_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[bad_call],
    )
    provider = create_mock_provider(
        [
            discuss_response,
            summarize_response,
            serialize_response,
            serialize_response,
            serialize_response,
            serialize_response,
        ]
    )

    def validator(_data: dict[str, Any]) -> ValidationResult:
        return ValidationResult(
            valid=False,
            errors=[ValidationErrorDetail(field="value", issue="error", provided="")],
        )

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
        validation_retries=2,
    )

    with pytest.raises(ConversationError, match="Validation failed after 2 retries"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
            validator=validator,
        )


# --- Callback Tests ---


@pytest.mark.asyncio
async def test_runner_calls_on_assistant_message() -> None:
    """Runner calls on_assistant_message callback for each phase."""
    discuss_response = LLMResponse(
        content="Discuss content",
        model="test",
        tokens_used=20,
        finish_reason="stop",
    )
    summarize_response = LLMResponse(
        content="Summary content",
        model="test",
        tokens_used=30,
        finish_reason="stop",
    )
    finalize_call = ToolCall(id="call_1", name="submit_test", arguments={"value": "done"})
    serialize_response = LLMResponse(
        content="Serialize content",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider([discuss_response, summarize_response, serialize_response])

    messages_received: list[str] = []

    def on_message(content: str) -> None:
        messages_received.append(content)

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        on_assistant_message=on_message,
    )

    assert "Discuss content" in messages_received
    assert "Summary content" in messages_received


# --- Error Handling Tests ---


@pytest.mark.asyncio
async def test_runner_summarize_empty_content_fails() -> None:
    """Summarize phase fails if LLM returns empty content."""
    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    summarize_response = LLMResponse(content="", model="test", tokens_used=10, finish_reason="stop")
    provider = create_mock_provider([discuss_response, summarize_response])

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    with pytest.raises(ConversationError, match="Summarize phase produced no content"):
        await runner.run(
            initial_messages=[{"role": "user", "content": "Start"}],
        )


@pytest.mark.asyncio
async def test_runner_tool_execution_error() -> None:
    """Runner handles research tool execution errors gracefully."""

    class FailingTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="failing", description="Fails", parameters={})

        def execute(self, _arguments: dict[str, Any]) -> str:
            raise RuntimeError("Tool crashed")

    failing_call = ToolCall(id="call_1", name="failing", arguments={})
    discuss_response1 = LLMResponse(
        content="",
        model="test",
        tokens_used=10,
        finish_reason="tool_calls",
        tool_calls=[failing_call],
    )
    discuss_response2 = LLMResponse(
        content="Recovered",
        model="test",
        tokens_used=15,
        finish_reason="stop",
    )
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    finalize_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "done"})
    serialize_response = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[finalize_call],
    )
    provider = create_mock_provider(
        [discuss_response1, discuss_response2, summarize_response, serialize_response]
    )

    runner = ConversationRunner(
        provider=provider,
        research_tools=[FailingTool()],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
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


# --- Validation Feedback Tests ---


@pytest.mark.asyncio
async def test_runner_feedback_includes_schema() -> None:
    """Feedback includes tool schema to help small models."""
    import json

    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    first_call = ToolCall(id="call_1", name="submit_test", arguments={"value": ""})
    serialize_response1 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})
    serialize_response2 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider(
        [discuss_response, summarize_response, serialize_response1, serialize_response2]
    )

    def validator(data: dict[str, Any]) -> ValidationResult:
        if data.get("value") == "":
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationErrorDetail(field="value", issue="required", error_type="missing")
                ],
            )
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    _result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    # Find validation feedback
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    # Filter to only JSON feedback (not confirmation messages)
    feedback_msgs = [m for m in tool_results if m.get("content", "").startswith("{")]
    assert len(feedback_msgs) >= 1

    feedback = json.loads(feedback_msgs[0]["content"])
    assert "schema" in feedback
    assert feedback["schema"]["type"] == "object"


@pytest.mark.asyncio
async def test_runner_detects_unknown_fields() -> None:
    """Runner detects fields not in expected_fields."""
    import json

    discuss_response = LLMResponse(content="D", model="test", tokens_used=10, finish_reason="stop")
    summarize_response = LLMResponse(
        content="S", model="test", tokens_used=10, finish_reason="stop"
    )
    first_call = ToolCall(
        id="call_1", name="submit_test", arguments={"value": "", "typo_field": "oops"}
    )
    serialize_response1 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[first_call],
    )
    second_call = ToolCall(id="call_2", name="submit_test", arguments={"value": "fixed"})
    serialize_response2 = LLMResponse(
        content="",
        model="test",
        tokens_used=40,
        finish_reason="tool_calls",
        tool_calls=[second_call],
    )
    provider = create_mock_provider(
        [discuss_response, summarize_response, serialize_response1, serialize_response2]
    )

    def validator(data: dict[str, Any]) -> ValidationResult:
        if data.get("value") == "":
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationErrorDetail(field="value", issue="required", error_type="missing")
                ],
                expected_fields={"value"},
            )
        return ValidationResult(valid=True, data=data)

    runner = ConversationRunner(
        provider=provider,
        research_tools=[],
        finalization_tool=MockFinalizationTool(),
        max_discuss_turns=1,
    )

    _result, state = await runner.run(
        initial_messages=[{"role": "user", "content": "Start"}],
        validator=validator,
    )

    # Find validation feedback
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    feedback_msgs = [m for m in tool_results if m.get("content", "").startswith("{")]
    feedback = json.loads(feedback_msgs[0]["content"])

    # typo_field should be detected as unknown
    assert "typo_field" in feedback["issues"]["unknown"]
