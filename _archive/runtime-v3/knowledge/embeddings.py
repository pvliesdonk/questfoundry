"""Embeddings provider abstraction for corpus vector search.

Supports multiple embedding backends:
- ollama: Local embeddings via Ollama (lightweight, no torch)
- openai: OpenAI embeddings API
- sentence-transformers: Local embeddings with torch (heavy, optional)

The default is 'ollama' which requires only a running Ollama instance,
avoiding the ~2GB PyTorch dependency.

Usage:
    from questfoundry.runtime.knowledge.embeddings import get_embeddings

    # Auto-detect available provider
    embeddings = get_embeddings()

    # Or specify explicitly
    embeddings = get_embeddings(provider="ollama", model="nomic-embed-text")
    embeddings = get_embeddings(provider="openai", model="text-embedding-3-small")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingsProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...


# Default models for each provider
DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",  # 768 dims, good quality, fast
    "openai": "text-embedding-3-small",  # 1536 dims
    "sentence-transformers": "all-MiniLM-L6-v2",  # 384 dims
}

# Embedding dimensions by model
MODEL_DIMENSIONS = {
    # Ollama models
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Sentence-transformers models
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


def get_embedding_dimension(model: str) -> int:
    """Get the embedding dimension for a model.

    Args:
        model: Model name

    Returns:
        Embedding dimension (default 768 if unknown)
    """
    return MODEL_DIMENSIONS.get(model, 768)


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingsProvider:
    """Get an embeddings provider.

    Args:
        provider: Provider name ('ollama', 'openai', 'sentence-transformers')
                  If None, auto-detects available provider.
        model: Model name (uses provider default if None)
        base_url: Override base URL for API providers
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingsProvider instance

    Raises:
        RuntimeError: If no provider is available
    """
    if provider is None:
        provider = _detect_provider()

    if model is None:
        model = DEFAULT_MODELS.get(provider, "nomic-embed-text")

    logger.debug(f"Creating embeddings provider: {provider} with model {model}")

    if provider == "ollama":
        return _create_ollama_embeddings(model, base_url, **kwargs)
    elif provider == "openai":
        return _create_openai_embeddings(model, **kwargs)
    elif provider == "sentence-transformers":
        return _create_sentence_transformer_embeddings(model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def _detect_provider() -> str:
    """Detect the best available embeddings provider.

    Priority:
    1. ollama (if OLLAMA_BASE_URL is set or langchain_ollama is installed)
    2. openai (if OPENAI_API_KEY is set and langchain_openai is installed)
    3. sentence-transformers (if installed)
    """
    import importlib.util
    import os

    # Check for Ollama
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST"):
        logger.debug("Detected Ollama via environment variable")
        return "ollama"

    # Check if langchain_ollama is installed
    if importlib.util.find_spec("langchain_ollama") is not None:
        logger.debug("Ollama embeddings available")
        return "ollama"

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY") and importlib.util.find_spec("langchain_openai") is not None:
        logger.debug("OpenAI embeddings available")
        return "openai"

    # Fall back to sentence-transformers
    if importlib.util.find_spec("sentence_transformers") is not None:
        logger.debug("Sentence-transformers available (heavy dependency)")
        return "sentence-transformers"

    raise RuntimeError(
        "No embeddings provider available. Options:\n"
        "  1. Run Ollama locally (recommended, lightweight)\n"
        "  2. Set OPENAI_API_KEY for OpenAI embeddings\n"
        "  3. Install questfoundry[rag-local] for sentence-transformers (heavy)"
    )


def _create_ollama_embeddings(
    model: str,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingsProvider:
    """Create Ollama embeddings provider."""
    import os

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as e:
        raise RuntimeError(
            "langchain_ollama not installed. Install with: "
            "uv pip install langchain-ollama"
        ) from e

    # Use environment variable if base_url not specified
    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")

    ollama_kwargs: dict[str, Any] = {"model": model}
    if base_url:
        ollama_kwargs["base_url"] = base_url

    ollama_kwargs.update(kwargs)

    return OllamaEmbeddings(**ollama_kwargs)


def _create_openai_embeddings(
    model: str,
    **kwargs: Any,
) -> EmbeddingsProvider:
    """Create OpenAI embeddings provider."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise RuntimeError(
            "langchain_openai not installed. Install with: "
            "uv pip install langchain-openai"
        ) from e

    return OpenAIEmbeddings(model=model, **kwargs)


def _create_sentence_transformer_embeddings(
    model: str,
) -> EmbeddingsProvider:
    """Create sentence-transformers embeddings provider.

    This wraps SentenceTransformer in a LangChain-compatible interface.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with: "
            "uv pip install questfoundry[rag]"
        ) from e

    class SentenceTransformerEmbeddings:
        """LangChain-compatible wrapper for SentenceTransformer."""

        def __init__(self, model_name: str):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        def embed_query(self, text: str) -> list[float]:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

    return SentenceTransformerEmbeddings(model)
