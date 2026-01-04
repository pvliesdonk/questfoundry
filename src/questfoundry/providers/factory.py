"""Factory for creating LLM providers."""

from __future__ import annotations

import os
from typing import Any

from questfoundry.providers.base import ProviderError
from questfoundry.providers.langchain_wrapper import LangChainProvider


def create_provider(
    provider_name: str,
    model: str,
    **kwargs: Any,
) -> LangChainProvider:
    """Create a LangChain-backed provider.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic).
        model: Model name/identifier.
        **kwargs: Additional provider-specific options.

    Returns:
        Configured LangChainProvider.

    Raises:
        ProviderError: If provider unavailable or misconfigured.
    """
    provider_name = provider_name.lower()

    if provider_name == "ollama":
        return _create_ollama(model, **kwargs)
    elif provider_name == "openai":
        return _create_openai(model, **kwargs)
    elif provider_name == "anthropic":
        return _create_anthropic(model, **kwargs)
    else:
        raise ProviderError(provider_name, f"Unknown provider: {provider_name}")


def _create_ollama(model: str, **kwargs: Any) -> LangChainProvider:
    """Create Ollama provider."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ProviderError(
            "ollama",
            "langchain-ollama not installed. Run: uv add langchain-ollama",
        ) from e

    host = kwargs.get("host") or os.getenv("OLLAMA_HOST")
    if not host:
        raise ProviderError(
            "ollama",
            "OLLAMA_HOST not configured. Set OLLAMA_HOST environment variable.",
        )

    chat_model = ChatOllama(
        model=model,
        base_url=host,
        temperature=kwargs.get("temperature", 0.7),
    )

    return LangChainProvider(chat_model, model)


def _create_openai(model: str, **kwargs: Any) -> LangChainProvider:
    """Create OpenAI provider."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ProviderError(
            "openai",
            "langchain-openai not installed. Run: uv add langchain-openai",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError(
            "openai",
            "API key required. Set OPENAI_API_KEY environment variable.",
        )

    chat_model = ChatOpenAI(
        model=model,
        api_key=api_key,  # type: ignore[arg-type]
        temperature=kwargs.get("temperature", 0.7),
    )

    return LangChainProvider(chat_model, model)


def _create_anthropic(model: str, **kwargs: Any) -> LangChainProvider:
    """Create Anthropic provider."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ProviderError(
            "anthropic",
            "langchain-anthropic not installed. Run: uv add langchain-anthropic",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError(
            "anthropic",
            "API key required. Set ANTHROPIC_API_KEY environment variable.",
        )

    chat_model = ChatAnthropic(
        model=model,  # type: ignore[call-arg]
        api_key=api_key,  # type: ignore[arg-type]
        temperature=kwargs.get("temperature", 0.7),
    )

    return LangChainProvider(chat_model, model)
