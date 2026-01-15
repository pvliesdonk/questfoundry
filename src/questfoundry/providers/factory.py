"""Factory for creating LLM providers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger
from questfoundry.providers.base import ProviderError
from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel

log = get_logger(__name__)

# Provider default models - None means model must be explicitly specified
PROVIDER_DEFAULTS: dict[str, str | None] = {
    "ollama": None,  # Require explicit model due to distribution issues
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}


def get_default_model(provider_name: str) -> str | None:
    """Get default model for a provider.

    Returns None for providers that require explicit model specification.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic).

    Returns:
        Default model name, or None if provider requires explicit model.
    """
    return PROVIDER_DEFAULTS.get(provider_name.lower())


def create_chat_model(
    provider_name: str,
    model: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain BaseChatModel directly.

    This is the primary way to get a chat model for use with LangChain agents
    and the stage protocol.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic).
        model: Model name/identifier.
        **kwargs: Additional provider-specific options.

    Returns:
        Configured BaseChatModel.

    Raises:
        ProviderError: If provider unavailable or misconfigured.
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower == "ollama":
        chat_model = _create_ollama_base_model(model, **kwargs)
    elif provider_name_lower == "openai":
        chat_model = _create_openai_base_model(model, **kwargs)
    elif provider_name_lower == "anthropic":
        chat_model = _create_anthropic_base_model(model, **kwargs)
    else:
        log.error("provider_unknown", provider=provider_name_lower)
        raise ProviderError(provider_name_lower, f"Unknown provider: {provider_name_lower}")

    log.info("chat_model_created", provider=provider_name_lower, model=model)
    return chat_model


def create_model_for_structured_output(
    provider_name: str,
    model_name: str | None = None,
    schema: type[BaseModel] | None = None,
    strategy: StructuredOutputStrategy | None = None,
    **kwargs: Any,
) -> BaseChatModel | Runnable[Any, Any]:
    """Create a chat model configured for structured output.

    This is a convenience function for creating a model that enforces
    structured output according to a Pydantic schema. It wraps the base
    model with LangChain's structured output support.

    Args:
        provider_name: Provider (ollama, openai, anthropic).
        model_name: Model name. Uses provider default if None.
        schema: Pydantic schema for structured output validation. If None,
            returns an unstructured BaseChatModel.
        strategy: Output strategy (auto-selected if None).
        **kwargs: Additional model kwargs (temperature, api_key, host, etc.).

    Returns:
        BaseChatModel if no schema provided, or Runnable with structured output
        if schema is provided.

    Raises:
        ProviderError: If provider is unavailable or misconfigured.

    Example:
        ```python
        from pydantic import BaseModel

        class StoryOutline(BaseModel):
            title: str
            genre: str
            plot_points: list[str]

        model = create_model_for_structured_output(
            "ollama",
            model_name="qwen3:8b",
            schema=StoryOutline,
        )
        ```
    """
    provider_name_lower = provider_name.lower()

    # Get base model based on provider
    if provider_name_lower == "ollama":
        base_model = _create_ollama_base_model(model_name or "qwen3:8b", **kwargs)
    elif provider_name_lower == "openai":
        base_model = _create_openai_base_model(model_name or "gpt-4o-mini", **kwargs)
    elif provider_name_lower == "anthropic":
        base_model = _create_anthropic_base_model(model_name or "claude-3-5-sonnet", **kwargs)
    else:
        log.error("provider_unknown", provider=provider_name_lower)
        raise ProviderError(provider_name_lower, f"Unknown provider: {provider_name_lower}")

    # Apply structured output if schema provided
    if schema is not None:
        base_model = with_structured_output(  # type: ignore[assignment]
            base_model,
            schema,
            strategy=strategy,
            provider_name=provider_name_lower,
        )

    log.info(
        "model_created_structured",
        provider=provider_name_lower,
        model=model_name,
        has_schema=schema is not None,
    )
    return base_model


def _create_ollama_base_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create base Ollama chat model (unstructured)."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        log.error("provider_import_error", provider="ollama", package="langchain-ollama")
        raise ProviderError(
            "ollama",
            "langchain-ollama not installed. Run: uv add langchain-ollama",
        ) from e

    host = kwargs.get("host") or os.getenv("OLLAMA_HOST")
    if not host:
        log.error("provider_config_error", provider="ollama", missing="OLLAMA_HOST")
        raise ProviderError(
            "ollama",
            "OLLAMA_HOST not configured. Set OLLAMA_HOST environment variable.",
        )

    # Cast needed: ChatOllama runtime type is BaseChatModel, but mypy can't verify
    chat_model: BaseChatModel = ChatOllama(
        model=model,
        base_url=host,
        temperature=kwargs.get("temperature", 0.7),
        num_ctx=kwargs.get("num_ctx", 32768),  # Default 32k to avoid truncation
    )
    return chat_model


def _create_openai_base_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create base OpenAI chat model (unstructured)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        log.error("provider_import_error", provider="openai", package="langchain-openai")
        raise ProviderError(
            "openai",
            "langchain-openai not installed. Run: uv add langchain-openai",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("provider_config_error", provider="openai", missing="OPENAI_API_KEY")
        raise ProviderError(
            "openai",
            "API key required. Set OPENAI_API_KEY environment variable.",
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,  # type: ignore[arg-type]
        temperature=kwargs.get("temperature", 0.7),
    )


def _create_anthropic_base_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create base Anthropic chat model (unstructured)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        log.error("provider_import_error", provider="anthropic", package="langchain-anthropic")
        raise ProviderError(
            "anthropic",
            "langchain-anthropic not installed. Run: uv add langchain-anthropic",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("provider_config_error", provider="anthropic", missing="ANTHROPIC_API_KEY")
        raise ProviderError(
            "anthropic",
            "API key required. Set ANTHROPIC_API_KEY environment variable.",
        )

    return ChatAnthropic(
        model=model,  # type: ignore[call-arg]
        api_key=api_key,  # type: ignore[arg-type]
        temperature=kwargs.get("temperature", 0.7),
    )
