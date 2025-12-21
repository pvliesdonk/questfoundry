"""
Embedding providers for corpus vector search.

Supports multiple backends:
- Ollama (local, recommended for dev)
- OpenAI (cloud, requires API key)

Provider selection:
1. QF_EMBEDDING_PROVIDER env var
2. Fall back to default LLM provider
3. Default to Ollama if available
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Default embedding models per provider
DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}

# Embedding dimensions per model
MODEL_DIMENSIONS = {
    # Ollama models
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""

    embeddings: list[list[float]]
    model: str
    dimension: int
    token_count: int | None = None


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name being used."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Embed multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with vectors and metadata
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = await self.embed([text])
        return result.embeddings[0]

    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if the provider is available."""
        ...


class OllamaEmbeddings(EmbeddingProvider):
    """
    Ollama embedding provider.

    Uses local Ollama instance for embeddings.
    Recommended model: nomic-embed-text (768 dimensions)
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ):
        """
        Initialize Ollama embeddings.

        Args:
            model: Embedding model name (default: nomic-embed-text)
            host: Ollama host URL (default: http://localhost:11434)
        """
        self._model = model or DEFAULT_MODELS["ollama"]
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._dimension = MODEL_DIMENSIONS.get(self._model, 768)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts using Ollama."""
        import httpx

        embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    f"{self._host}/api/embeddings",
                    json={
                        "model": self._model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])

        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model,
            dimension=self._dimension,
        )

    async def check_availability(self) -> bool:
        """Check if Ollama is available with the embedding model."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self._host}/api/tags")
                if response.status_code != 200:
                    return False

                # Check if model is available
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]

                # Model names may include :latest suffix
                model_base = self._model.split(":")[0]
                return any(m.startswith(model_base) for m in models)

        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI embedding provider.

    Requires OPENAI_API_KEY environment variable.
    Default model: text-embedding-3-small (1536 dimensions)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            model: Embedding model name (default: text-embedding-3-small)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env)
        """
        self._model = model or DEFAULT_MODELS["openai"]
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._dimension = MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts using OpenAI API."""
        import httpx

        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        token_count = data.get("usage", {}).get("total_tokens")

        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model,
            dimension=self._dimension,
            token_count=token_count,
        )

    async def check_availability(self) -> bool:
        """Check if OpenAI API is available."""
        if not self._api_key:
            return False

        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return response.status_code == 200

        except Exception as e:
            logger.debug(f"OpenAI availability check failed: {e}")
            return False


async def get_embedding_provider(
    provider_name: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider | None:
    """
    Get an embedding provider based on configuration.

    Selection order:
    1. Explicit provider_name parameter
    2. QF_EMBEDDING_PROVIDER environment variable
    3. QF_LLM_PROVIDER environment variable (follow LLM provider)
    4. Auto-detect available provider (Ollama preferred)

    Args:
        provider_name: Explicit provider name ("ollama" or "openai")
        model: Optional model override

    Returns:
        Configured EmbeddingProvider or None if none available
    """
    # Determine provider
    name = provider_name or os.getenv("QF_EMBEDDING_PROVIDER") or os.getenv("QF_LLM_PROVIDER")

    if name:
        name = name.lower()
        provider: EmbeddingProvider
        if name == "ollama":
            provider = OllamaEmbeddings(model=model)
        elif name == "openai":
            provider = OpenAIEmbeddings(model=model)
        else:
            logger.warning(f"Unknown embedding provider: {name}")
            return None

        if await provider.check_availability():
            return provider
        logger.warning(f"Embedding provider {name} not available")
        return None

    # Auto-detect: try Ollama first, then OpenAI
    ollama = OllamaEmbeddings(model=model if model else None)
    if await ollama.check_availability():
        logger.info(f"Using Ollama embeddings ({ollama.model})")
        return ollama

    openai = OpenAIEmbeddings(model=model if model else None)
    if await openai.check_availability():
        logger.info(f"Using OpenAI embeddings ({openai.model})")
        return openai

    logger.warning("No embedding provider available")
    return None
