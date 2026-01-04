"""Tests for LLM provider base types."""

from __future__ import annotations

from questfoundry.providers import (
    LLMResponse,
    ProviderConnectionError,
    ProviderError,
    ProviderModelError,
    ProviderRateLimitError,
)
from questfoundry.tools import ToolCall, ToolDefinition

# --- LLMResponse Tests ---


def test_llm_response_is_complete_stop() -> None:
    """Response with 'stop' finish reason is complete."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="stop",
    )
    assert response.is_complete


def test_llm_response_is_complete_end_turn() -> None:
    """Response with 'end_turn' finish reason is complete."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="end_turn",
    )
    assert response.is_complete


def test_llm_response_not_complete_length() -> None:
    """Response with 'length' finish reason is not complete."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="length",
    )
    assert not response.is_complete


def test_llm_response_not_complete_unknown() -> None:
    """Response with unknown finish reason is not complete."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="unknown",
    )
    assert not response.is_complete


# --- Exception Tests ---


def test_provider_error_format() -> None:
    """ProviderError formats message with provider name."""
    error = ProviderError("test", "Something went wrong")
    assert str(error) == "[test] Something went wrong"
    assert error.provider == "test"


def test_provider_connection_error_is_provider_error() -> None:
    """ProviderConnectionError is a ProviderError."""
    error = ProviderConnectionError("test", "Connection failed")
    assert isinstance(error, ProviderError)
    assert error.provider == "test"


def test_provider_rate_limit_error_is_provider_error() -> None:
    """ProviderRateLimitError is a ProviderError."""
    error = ProviderRateLimitError("test", "Rate limit exceeded")
    assert isinstance(error, ProviderError)
    assert error.provider == "test"


def test_provider_model_error_is_provider_error() -> None:
    """ProviderModelError is a ProviderError."""
    error = ProviderModelError("test", "Model not found")
    assert isinstance(error, ProviderError)
    assert error.provider == "test"


# --- Tool-Related LLMResponse Tests ---


def test_llm_response_with_tool_calls_not_complete() -> None:
    """Response with tool_calls is not complete even with 'stop' finish_reason."""
    tool_call = ToolCall(id="call_1", name="test_tool", arguments={})
    response = LLMResponse(
        content="Let me call a tool",
        model="test",
        tokens_used=10,
        finish_reason="stop",
        tool_calls=[tool_call],
    )
    assert not response.is_complete
    assert response.has_tool_calls


def test_llm_response_empty_tool_calls_is_complete() -> None:
    """Response with empty tool_calls list is still complete."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="stop",
        tool_calls=[],
    )
    # Empty list is falsy, so is_complete should be True
    assert response.is_complete
    assert not response.has_tool_calls


def test_llm_response_has_tool_calls_true() -> None:
    """has_tool_calls returns True when tool_calls is populated."""
    tool_call = ToolCall(id="call_1", name="submit_dream", arguments={"title": "Test"})
    response = LLMResponse(
        content="",
        model="test",
        tokens_used=10,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    assert response.has_tool_calls


def test_llm_response_has_tool_calls_false_none() -> None:
    """has_tool_calls returns False when tool_calls is None."""
    response = LLMResponse(
        content="Hello",
        model="test",
        tokens_used=10,
        finish_reason="stop",
        tool_calls=None,
    )
    assert not response.has_tool_calls


# --- ToolDefinition Tests ---


def test_tool_definition_creation() -> None:
    """ToolDefinition can be created with all fields."""
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert "properties" in tool.parameters


def test_tool_definition_default_parameters() -> None:
    """ToolDefinition has sensible default for parameters."""
    tool = ToolDefinition(
        name="simple_tool",
        description="Tool with no params",
    )
    assert tool.name == "simple_tool"
    # Parameters defaults to empty object schema
    assert tool.parameters == {"type": "object", "properties": {}}


# --- ToolCall Tests ---


def test_tool_call_creation() -> None:
    """ToolCall can be created with required fields."""
    call = ToolCall(
        id="call_abc123",
        name="submit_dream",
        arguments={"title": "My Story", "genre": "Fantasy"},
    )
    assert call.id == "call_abc123"
    assert call.name == "submit_dream"
    assert call.arguments == {"title": "My Story", "genre": "Fantasy"}


def test_tool_call_empty_arguments() -> None:
    """ToolCall can have empty arguments."""
    call = ToolCall(id="call_1", name="get_status", arguments={})
    assert call.arguments == {}
