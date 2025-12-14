"""Tests for base provider types and interfaces."""

from __future__ import annotations

import pytest

from questfoundry.runtime.providers import (
    ContextOverflowError,
    InvokeOptions,
    LLMMessage,
    LLMResponse,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
)


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_create_system_message(self) -> None:
        """System messages have role='system'."""
        msg = LLMMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_create_user_message(self) -> None:
        """User messages have role='user'."""
        msg = LLMMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_create_assistant_message(self) -> None:
        """Assistant messages have role='assistant'."""
        msg = LLMMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_minimal_response(self) -> None:
        """Response requires content, model, and provider."""
        response = LLMResponse(
            content="Hello!",
            model="qwen3:8b",
            provider="ollama",
        )
        assert response.content == "Hello!"
        assert response.model == "qwen3:8b"
        assert response.provider == "ollama"
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
        assert response.total_tokens is None
        assert response.duration_ms is None
        assert response.raw is None

    def test_create_full_response(self) -> None:
        """Response can include all metadata."""
        response = LLMResponse(
            content="Hello!",
            model="qwen3:8b",
            provider="ollama",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            duration_ms=1234.5,
            raw={"debug": "data"},
        )
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 50
        assert response.total_tokens == 150
        assert response.duration_ms == 1234.5
        assert response.raw == {"debug": "data"}


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_create_content_chunk(self) -> None:
        """Content chunks have text and done=False."""
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.done is False
        assert chunk.prompt_tokens is None

    def test_create_final_chunk(self) -> None:
        """Final chunk has done=True and optional usage stats."""
        chunk = StreamChunk(
            content="",
            done=True,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert chunk.content == ""
        assert chunk.done is True
        assert chunk.prompt_tokens == 100
        assert chunk.completion_tokens == 50
        assert chunk.total_tokens == 150


class TestInvokeOptions:
    """Tests for InvokeOptions dataclass."""

    def test_default_options(self) -> None:
        """Options have sensible defaults."""
        options = InvokeOptions()
        assert options.temperature == 0.7
        assert options.max_tokens is None
        assert options.stop_sequences == []
        assert options.timeout_seconds == 120.0

    def test_custom_options(self) -> None:
        """Options can be customized."""
        options = InvokeOptions(
            temperature=0.3,
            max_tokens=1000,
            stop_sequences=["###", "END"],
            timeout_seconds=60.0,
        )
        assert options.temperature == 0.3
        assert options.max_tokens == 1000
        assert options.stop_sequences == ["###", "END"]
        assert options.timeout_seconds == 60.0


class TestExceptions:
    """Tests for provider exceptions."""

    def test_provider_error_hierarchy(self) -> None:
        """All provider errors inherit from ProviderError."""
        assert issubclass(ProviderUnavailableError, ProviderError)
        assert issubclass(ProviderConfigError, ProviderError)
        assert issubclass(ContextOverflowError, ProviderError)

    def test_provider_error(self) -> None:
        """ProviderError can be raised with message."""
        with pytest.raises(ProviderError, match="Test error"):
            raise ProviderError("Test error")

    def test_provider_unavailable_error(self) -> None:
        """ProviderUnavailableError for connectivity issues."""
        with pytest.raises(ProviderUnavailableError, match="not available"):
            raise ProviderUnavailableError("Ollama not available at localhost")

    def test_provider_config_error(self) -> None:
        """ProviderConfigError for configuration issues."""
        with pytest.raises(ProviderConfigError, match="Missing API key"):
            raise ProviderConfigError("Missing API key for OpenAI")

    def test_context_overflow_error(self) -> None:
        """ContextOverflowError for prompt size issues."""
        with pytest.raises(ContextOverflowError, match="exceeds"):
            raise ContextOverflowError("Prompt (50000 tokens) exceeds context (32768)")
