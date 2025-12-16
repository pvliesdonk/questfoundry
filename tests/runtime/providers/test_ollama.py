"""Tests for Ollama provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.runtime.providers import (
    InvokeOptions,
    LLMMessage,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
)
from questfoundry.runtime.providers.ollama import OllamaProvider


class TestOllamaProviderBasics:
    """Tests for OllamaProvider basic functionality."""

    def test_provider_name(self) -> None:
        """Provider has correct name."""
        provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_default_host(self) -> None:
        """Default host is localhost:11434."""
        provider = OllamaProvider()
        assert provider.host == "http://localhost:11434"

    def test_custom_host(self) -> None:
        """Host can be customized."""
        provider = OllamaProvider(host="http://192.168.1.100:11434")
        assert provider.host == "http://192.168.1.100:11434"

    def test_host_trailing_slash_stripped(self) -> None:
        """Trailing slash is stripped from host."""
        provider = OllamaProvider(host="http://localhost:11434/")
        assert provider.host == "http://localhost:11434"


class TestOllamaProviderAvailability:
    """Tests for OllamaProvider availability checks."""

    @pytest.mark.asyncio
    async def test_check_availability_success(self) -> None:
        """Returns True when Ollama is reachable."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.check_availability()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_failure(self) -> None:
        """Returns False when Ollama is not reachable."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_get_client.return_value = mock_client

            result = await provider.check_availability()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_non_200(self) -> None:
        """Returns False when Ollama returns non-200."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await provider.check_availability()
            assert result is False


class TestOllamaProviderListModels:
    """Tests for OllamaProvider model listing."""

    @pytest.mark.asyncio
    async def test_list_models_success(self) -> None:
        """Returns model list when Ollama is available."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:8b"},
                {"name": "llama3.2:3b"},
            ]
        }

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            models = await provider.list_models()
            assert models == ["qwen3:8b", "llama3.2:3b"]

    @pytest.mark.asyncio
    async def test_list_models_empty(self) -> None:
        """Returns empty list when no models installed."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            models = await provider.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_error(self) -> None:
        """Returns empty list on error."""
        provider = OllamaProvider()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_get_client.return_value = mock_client

            models = await provider.list_models()
            assert models == []


class TestOllamaProviderInvoke:
    """Tests for OllamaProvider invoke method."""

    @pytest.mark.asyncio
    async def test_invoke_unavailable(self) -> None:
        """Raises ProviderUnavailableError when Ollama is down."""
        provider = OllamaProvider()

        with (
            patch.object(provider, "check_availability", return_value=False),
            pytest.raises(ProviderUnavailableError, match="not available"),
        ):
            await provider.invoke(
                messages=[LLMMessage(role="user", content="Hello")],
                model="qwen3:8b",
            )

    @pytest.mark.asyncio
    async def test_invoke_success(self) -> None:
        """Returns response on successful invocation."""
        provider = OllamaProvider()

        # Mock langchain response
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help you?"
        mock_response.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

        with (
            patch.object(provider, "check_availability", return_value=True),
            patch("questfoundry.runtime.providers.ollama.ChatOllama") as mock_chat,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_chat.return_value = mock_llm

            response = await provider.invoke(
                messages=[
                    LLMMessage(role="system", content="Be helpful."),
                    LLMMessage(role="user", content="Hello"),
                ],
                model="qwen3:8b",
            )

            assert response.content == "Hello! How can I help you?"
            assert response.model == "qwen3:8b"
            assert response.provider == "ollama"
            assert response.prompt_tokens == 10
            assert response.completion_tokens == 5
            assert response.total_tokens == 15
            assert response.duration_ms is not None

    @pytest.mark.asyncio
    async def test_invoke_with_options(self) -> None:
        """Options are passed to ChatOllama."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_response.usage_metadata = None

        with (
            patch.object(provider, "check_availability", return_value=True),
            patch("questfoundry.runtime.providers.ollama.ChatOllama") as mock_chat,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_chat.return_value = mock_llm

            await provider.invoke(
                messages=[LLMMessage(role="user", content="Hello")],
                model="qwen3:8b",
                options=InvokeOptions(
                    temperature=0.3,
                    max_tokens=500,
                    stop_sequences=["###"],
                ),
            )

            # Verify ChatOllama was called with correct params
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["num_predict"] == 500
            assert call_kwargs["stop"] == ["###"]

    @pytest.mark.asyncio
    async def test_invoke_timeout(self) -> None:
        """Raises ProviderError on timeout."""
        provider = OllamaProvider()

        with (
            patch.object(provider, "check_availability", return_value=True),
            patch("questfoundry.runtime.providers.ollama.ChatOllama") as mock_chat,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = TimeoutError()
            mock_chat.return_value = mock_llm

            with pytest.raises(ProviderError, match="timed out"):
                await provider.invoke(
                    messages=[LLMMessage(role="user", content="Hello")],
                    model="qwen3:8b",
                    options=InvokeOptions(timeout_seconds=1.0),
                )


class TestOllamaProviderStream:
    """Tests for OllamaProvider stream method."""

    @pytest.mark.asyncio
    async def test_stream_unavailable(self) -> None:
        """Raises ProviderUnavailableError when Ollama is down."""
        provider = OllamaProvider()

        with (
            patch.object(provider, "check_availability", return_value=False),
            pytest.raises(ProviderUnavailableError, match="not available"),
        ):
            async for _chunk in provider.stream(
                messages=[LLMMessage(role="user", content="Hello")],
                model="qwen3:8b",
            ):
                pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_stream_success(self) -> None:
        """Yields chunks and final done chunk."""
        provider = OllamaProvider()

        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.content = " world"
        chunk2.usage_metadata = None

        chunk3 = MagicMock()
        chunk3.content = "!"
        chunk3.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
        }

        async def mock_astream(_messages: list, **_kwargs: object) -> AsyncMock:  # type: ignore[misc]
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        with (
            patch.object(provider, "check_availability", return_value=True),
            patch("questfoundry.runtime.providers.ollama.ChatOllama") as mock_chat,
        ):
            mock_llm = MagicMock()
            mock_llm.astream = mock_astream
            mock_chat.return_value = mock_llm

            chunks: list[StreamChunk] = []
            async for chunk in provider.stream(
                messages=[LLMMessage(role="user", content="Hello")],
                model="qwen3:8b",
            ):
                chunks.append(chunk)

            # Should have 4 chunks: 3 content + 1 final
            assert len(chunks) == 4

            # Content chunks
            assert chunks[0].content == "Hello"
            assert chunks[0].done is False
            assert chunks[1].content == " world"
            assert chunks[1].done is False
            assert chunks[2].content == "!"
            assert chunks[2].done is False

            # Final chunk with usage
            assert chunks[3].content == ""
            assert chunks[3].done is True
            assert chunks[3].prompt_tokens == 10
            assert chunks[3].completion_tokens == 3
            assert chunks[3].total_tokens == 13

    @pytest.mark.asyncio
    async def test_stream_error(self) -> None:
        """Raises ProviderError on streaming failure."""
        provider = OllamaProvider()

        async def mock_astream(_messages: list) -> AsyncMock:  # type: ignore[misc]
            raise Exception("Stream error")
            yield  # Make it a generator  # noqa: B033

        with (
            patch.object(provider, "check_availability", return_value=True),
            patch("questfoundry.runtime.providers.ollama.ChatOllama") as mock_chat,
        ):
            mock_llm = MagicMock()
            mock_llm.astream = mock_astream
            mock_chat.return_value = mock_llm

            with pytest.raises(ProviderError, match="streaming failed"):
                async for _chunk in provider.stream(
                    messages=[LLMMessage(role="user", content="Hello")],
                    model="qwen3:8b",
                ):
                    pass


class TestOllamaProviderClose:
    """Tests for OllamaProvider cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up_client(self) -> None:
        """close() releases HTTP client."""
        provider = OllamaProvider()

        # Simulate a client being created
        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_called_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self) -> None:
        """close() is safe to call without client."""
        provider = OllamaProvider()
        assert provider._client is None

        # Should not raise
        await provider.close()
        assert provider._client is None
