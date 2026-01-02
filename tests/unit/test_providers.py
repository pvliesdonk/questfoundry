"""Tests for LLM provider base types."""

from __future__ import annotations

from questfoundry.providers import (
    LLMResponse,
    ProviderConnectionError,
    ProviderError,
    ProviderModelError,
    ProviderRateLimitError,
)

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
