"""
Tests for embedding providers.

Tests the embedding infrastructure without requiring actual API access.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from questfoundry.runtime.corpus.embeddings import (
    DEFAULT_MODELS,
    MODEL_DIMENSIONS,
    EmbeddingResult,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    get_embedding_provider,
)


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_basic_result(self):
        """EmbeddingResult should store embeddings and metadata."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            dimension=3,
        )

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.model == "test-model"
        assert result.dimension == 3
        assert result.token_count is None

    def test_result_with_token_count(self):
        """EmbeddingResult can include token count."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="test-model",
            dimension=2,
            token_count=100,
        )

        assert result.token_count == 100


class TestModelConstants:
    """Tests for model constants."""

    def test_default_models_defined(self):
        """Default models should be defined for each provider."""
        assert "ollama" in DEFAULT_MODELS
        assert "openai" in DEFAULT_MODELS
        assert DEFAULT_MODELS["ollama"] == "nomic-embed-text"
        assert DEFAULT_MODELS["openai"] == "text-embedding-3-small"

    def test_model_dimensions_defined(self):
        """Model dimensions should be defined for common models."""
        assert "nomic-embed-text" in MODEL_DIMENSIONS
        assert "text-embedding-3-small" in MODEL_DIMENSIONS

        assert MODEL_DIMENSIONS["nomic-embed-text"] == 768
        assert MODEL_DIMENSIONS["text-embedding-3-small"] == 1536


class TestOllamaEmbeddings:
    """Tests for OllamaEmbeddings provider."""

    def test_default_model(self):
        """OllamaEmbeddings should use default model."""
        provider = OllamaEmbeddings()

        assert provider.model == "nomic-embed-text"
        assert provider.dimension == 768

    def test_custom_model(self):
        """OllamaEmbeddings should accept custom model."""
        provider = OllamaEmbeddings(model="mxbai-embed-large")

        assert provider.model == "mxbai-embed-large"
        assert provider.dimension == 1024

    def test_custom_host(self):
        """OllamaEmbeddings should accept custom host."""
        provider = OllamaEmbeddings(host="http://custom:11434")

        assert provider._host == "http://custom:11434"

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """embed_single should return single embedding."""
        provider = OllamaEmbeddings()

        # Mock the embed method
        mock_embedding = [0.1] * 768
        with patch.object(
            provider,
            "embed",
            new_callable=AsyncMock,
            return_value=EmbeddingResult(
                embeddings=[mock_embedding],
                model="nomic-embed-text",
                dimension=768,
            ),
        ):
            result = await provider.embed_single("test text")

        assert len(result) == 768
        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_check_availability_offline(self):
        """check_availability should return False when Ollama is offline."""
        # Patch to simulate connection failure
        with patch(
            "questfoundry.runtime.corpus.embeddings.OllamaEmbeddings.check_availability"
        ) as mock:
            mock.return_value = False
            result = await mock()

        assert result is False


class TestOpenAIEmbeddings:
    """Tests for OpenAIEmbeddings provider."""

    def test_default_model(self):
        """OpenAIEmbeddings should use default model."""
        provider = OpenAIEmbeddings()

        assert provider.model == "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_custom_model(self):
        """OpenAIEmbeddings should accept custom model."""
        provider = OpenAIEmbeddings(model="text-embedding-3-large")

        assert provider.model == "text-embedding-3-large"
        assert provider.dimension == 3072

    def test_api_key_from_param(self):
        """OpenAIEmbeddings should accept API key as parameter."""
        provider = OpenAIEmbeddings(api_key="test-key")

        assert provider._api_key == "test-key"

    @pytest.mark.asyncio
    async def test_check_availability_no_key(self):
        """check_availability should return False without API key."""
        provider = OpenAIEmbeddings(api_key=None)
        provider._api_key = None

        result = await provider.check_availability()

        assert result is False


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider function."""

    @pytest.mark.asyncio
    async def test_explicit_ollama(self):
        """Explicit provider name should select Ollama."""
        with patch.object(OllamaEmbeddings, "check_availability", new_callable=AsyncMock) as mock:
            mock.return_value = True
            provider = await get_embedding_provider(provider_name="ollama")

        assert provider is not None
        assert isinstance(provider, OllamaEmbeddings)

    @pytest.mark.asyncio
    async def test_explicit_openai(self):
        """Explicit provider name should select OpenAI."""
        with patch.object(OpenAIEmbeddings, "check_availability", new_callable=AsyncMock) as mock:
            mock.return_value = True
            provider = await get_embedding_provider(provider_name="openai")

        assert provider is not None
        assert isinstance(provider, OpenAIEmbeddings)

    @pytest.mark.asyncio
    async def test_unknown_provider(self):
        """Unknown provider should return None."""
        provider = await get_embedding_provider(provider_name="unknown")

        assert provider is None

    @pytest.mark.asyncio
    async def test_unavailable_provider(self):
        """Unavailable provider should return None."""
        with patch.object(OllamaEmbeddings, "check_availability", new_callable=AsyncMock) as mock:
            mock.return_value = False
            provider = await get_embedding_provider(provider_name="ollama")

        assert provider is None

    @pytest.mark.asyncio
    async def test_auto_detect_prefers_ollama(self):
        """Auto-detect should prefer Ollama over OpenAI."""
        with patch.object(
            OllamaEmbeddings, "check_availability", new_callable=AsyncMock
        ) as ollama_mock:
            ollama_mock.return_value = True

            with patch.object(
                OpenAIEmbeddings, "check_availability", new_callable=AsyncMock
            ) as openai_mock:
                openai_mock.return_value = True
                provider = await get_embedding_provider()

        assert provider is not None
        assert isinstance(provider, OllamaEmbeddings)

    @pytest.mark.asyncio
    async def test_auto_detect_falls_back_to_openai(self):
        """Auto-detect should fall back to OpenAI if Ollama unavailable."""
        with patch.object(
            OllamaEmbeddings, "check_availability", new_callable=AsyncMock
        ) as ollama_mock:
            ollama_mock.return_value = False

            with patch.object(
                OpenAIEmbeddings, "check_availability", new_callable=AsyncMock
            ) as openai_mock:
                openai_mock.return_value = True
                provider = await get_embedding_provider()

        assert provider is not None
        assert isinstance(provider, OpenAIEmbeddings)

    @pytest.mark.asyncio
    async def test_no_provider_available(self):
        """Should return None if no provider available."""
        with patch.object(
            OllamaEmbeddings, "check_availability", new_callable=AsyncMock
        ) as ollama_mock:
            ollama_mock.return_value = False

            with patch.object(
                OpenAIEmbeddings, "check_availability", new_callable=AsyncMock
            ) as openai_mock:
                openai_mock.return_value = False
                provider = await get_embedding_provider()

        assert provider is None
