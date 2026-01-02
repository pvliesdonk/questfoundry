"""Tests for LLM providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from questfoundry.providers import (
    LLMResponse,
    Message,
    OllamaProvider,
    OpenAIProvider,
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


# --- OllamaProvider Tests ---


def test_ollama_requires_host() -> None:
    """OllamaProvider raises error without OLLAMA_HOST."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderError) as exc_info:
            OllamaProvider()

        assert "OLLAMA_HOST not configured" in str(exc_info.value)


def test_ollama_env_host() -> None:
    """OllamaProvider uses OLLAMA_HOST env var."""
    with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:8080"}):
        provider = OllamaProvider()
        assert provider.host == "http://custom:8080"


def test_ollama_custom_host() -> None:
    """OllamaProvider uses custom host parameter."""
    provider = OllamaProvider(host="http://myhost:1234")
    assert provider.host == "http://myhost:1234"


def test_ollama_default_model() -> None:
    """OllamaProvider has correct default model."""
    provider = OllamaProvider(host="http://test:11434")
    assert provider.default_model == "qwen3:8b"


def test_ollama_custom_model() -> None:
    """OllamaProvider uses custom default model."""
    provider = OllamaProvider(host="http://test:11434", default_model="llama3:8b")
    assert provider.default_model == "llama3:8b"


@pytest.mark.asyncio
async def test_ollama_complete_success() -> None:
    """OllamaProvider successfully completes a request."""
    provider = OllamaProvider(host="http://test:11434")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Hello, world!"},
        "eval_count": 10,
        "prompt_eval_count": 5,
        "done": True,
        "done_reason": "stop",
    }

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        result = await provider.complete(messages)

        assert result.content == "Hello, world!"
        assert result.model == "qwen3:8b"
        assert result.tokens_used == 15
        assert result.finish_reason == "stop"
        assert result.is_complete


@pytest.mark.asyncio
async def test_ollama_complete_custom_model() -> None:
    """OllamaProvider uses custom model when specified."""
    provider = OllamaProvider(host="http://test:11434")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Response"},
        "done": True,
        "done_reason": "stop",
    }

    with patch.object(
        provider._client, "post", new=AsyncMock(return_value=mock_response)
    ) as mock_post:
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        result = await provider.complete(messages, model="llama3:70b")

        assert result.model == "llama3:70b"
        # Verify the model was passed in the request
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "llama3:70b"


@pytest.mark.asyncio
async def test_ollama_complete_connection_error() -> None:
    """OllamaProvider raises ProviderConnectionError on connection failure."""
    provider = OllamaProvider(host="http://test:11434")

    with patch.object(
        provider._client, "post", new=AsyncMock(side_effect=httpx.ConnectError("Failed"))
    ):
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.complete(messages)

        assert exc_info.value.provider == "ollama"
        assert "Failed to connect" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_complete_timeout() -> None:
    """OllamaProvider raises ProviderConnectionError on timeout."""
    provider = OllamaProvider(host="http://test:11434")

    with patch.object(
        provider._client, "post", new=AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    ):
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.complete(messages)

        assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_complete_model_not_found() -> None:
    """OllamaProvider raises ProviderModelError when model not found."""
    provider = OllamaProvider(host="http://test:11434")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model not found"

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ProviderModelError) as exc_info:
            await provider.complete(messages, model="nonexistent")

        assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_complete_api_error() -> None:
    """OllamaProvider raises ProviderError on API error."""
    provider = OllamaProvider(host="http://test:11434")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(messages)

        assert "status 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_context_manager() -> None:
    """OllamaProvider works as async context manager."""
    async with OllamaProvider(host="http://test:11434") as provider:
        assert provider.default_model == "qwen3:8b"


@pytest.mark.asyncio
async def test_ollama_list_models() -> None:
    """OllamaProvider lists available models."""
    provider = OllamaProvider(host="http://test:11434")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3:8b"},
            {"name": "qwen3:8b"},
            {"name": "mistral:7b"},
        ]
    }

    with patch.object(provider._client, "get", new=AsyncMock(return_value=mock_response)):
        models = await provider.list_models()

        assert "llama3:8b" in models
        assert "qwen3:8b" in models
        assert len(models) == 3


# --- OpenAIProvider Tests ---


def test_openai_requires_api_key() -> None:
    """OpenAIProvider raises error without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderError) as exc_info:
            OpenAIProvider()

        assert "API key required" in str(exc_info.value)


def test_openai_uses_env_key() -> None:
    """OpenAIProvider uses OPENAI_API_KEY env var."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider()
        assert provider.default_model == "gpt-4o-mini"


def test_openai_custom_model() -> None:
    """OpenAIProvider uses custom default model."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider(default_model="gpt-4o")
        assert provider.default_model == "gpt-4o"


@pytest.mark.asyncio
async def test_openai_complete_success() -> None:
    """OpenAIProvider successfully completes a request."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello from GPT!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 25},
        }

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
            messages: list[Message] = [{"role": "user", "content": "Hi"}]
            result = await provider.complete(messages)

            assert result.content == "Hello from GPT!"
            assert result.model == "gpt-4o-mini"
            assert result.tokens_used == 25
            assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_complete_rate_limit() -> None:
    """OpenAIProvider raises ProviderRateLimitError on 429."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
            messages: list[Message] = [{"role": "user", "content": "Hi"}]
            with pytest.raises(ProviderRateLimitError):
                await provider.complete(messages)


@pytest.mark.asyncio
async def test_openai_complete_auth_error() -> None:
    """OpenAIProvider raises ProviderError on 401."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "invalid-key"}):
        provider = OpenAIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_response)):
            messages: list[Message] = [{"role": "user", "content": "Hi"}]
            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(messages)

            assert "Invalid API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_context_manager() -> None:
    """OpenAIProvider works as async context manager."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        async with OpenAIProvider() as provider:
            assert provider.default_model == "gpt-4o-mini"
