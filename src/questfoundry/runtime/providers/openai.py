"""OpenAI LLM provider using langchain-openai.

Provides access to OpenAI models (GPT-4, GPT-4o, etc.) with native tool calling support.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODEL = "gpt-4o"

# Supported models with tool calling
SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
]


def check_openai_available(api_key: str | None = None) -> bool:
    """Check if OpenAI API is available.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key. If None, uses OPENAI_API_KEY env var.

    Returns
    -------
    bool
        True if API key is configured and valid.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        logger.debug("No OpenAI API key configured")
        return False

    # Just check key exists - actual validation happens on first call
    return True


def create_openai_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an OpenAI LLM instance.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-4o", "gpt-4-turbo").
    api_key : str | None
        OpenAI API key. If None, uses OPENAI_API_KEY env var.
    temperature : float
        Sampling temperature (0-2).
    max_tokens : int | None
        Maximum output tokens.
    **kwargs
        Additional arguments for ChatOpenAI.

    Returns
    -------
    BaseChatModel
        LangChain ChatOpenAI instance with tool calling support.

    Raises
    ------
    ImportError
        If langchain-openai is not installed.
    ValueError
        If API key is not configured.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai not installed. Install with: pip install langchain-openai"
        ) from e

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

    if model not in SUPPORTED_MODELS:
        logger.warning(f"Model '{model}' not in validated list. Supported: {SUPPORTED_MODELS}")

    logger.info(f"Creating OpenAI LLM: model={model}, temperature={temperature}")

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": key,
    }

    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    llm_kwargs.update(kwargs)

    return ChatOpenAI(**llm_kwargs)


def list_openai_models() -> list[dict[str, Any]]:
    """List supported OpenAI models with metadata.

    Returns
    -------
    list[dict[str, Any]]
        List of model info dicts.
    """
    return [
        {
            "model_id": "gpt-4o",
            "name": "GPT-4o",
            "context_window": 128000,
            "description": "Most capable model, best for complex tasks",
        },
        {
            "model_id": "gpt-4o-mini",
            "name": "GPT-4o Mini",
            "context_window": 128000,
            "description": "Smaller, faster, cheaper GPT-4o variant",
        },
        {
            "model_id": "gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "context_window": 128000,
            "description": "High capability with vision support",
        },
        {
            "model_id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "context_window": 16385,
            "description": "Fast and cost-effective",
        },
    ]


class OpenAIProvider:
    """OpenAI LLM provider class.

    Provides a consistent interface for OpenAI model access.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize provider.

        Parameters
        ----------
        api_key : str | None
            OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        """Check if provider is available."""
        return check_openai_available(self.api_key)

    def get_llm(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get LLM instance.

        Parameters
        ----------
        model : str
            Model name.
        temperature : float
            Sampling temperature.
        **kwargs
            Additional arguments.

        Returns
        -------
        BaseChatModel
            LangChain ChatOpenAI instance.
        """
        return create_openai_llm(
            model=model,
            api_key=self.api_key,
            temperature=temperature,
            **kwargs,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        return list_openai_models()
