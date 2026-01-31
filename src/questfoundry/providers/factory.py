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
    "openai": "gpt-5-mini",
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
            model_name="qwen3:4b-instruct-32k",
            schema=StoryOutline,
        )
        ```
    """
    provider_name_lower = provider_name.lower()

    # Resolve model name: use provided, then provider default, then convenience fallback
    resolved_model = model_name or get_default_model(provider_name_lower)
    # Fallback for providers where get_default_model returns None (e.g., ollama)
    if resolved_model is None:
        fallback_models = {"ollama": "qwen3:4b-instruct-32k"}
        resolved_model = fallback_models.get(provider_name_lower)
        if resolved_model is None:
            raise ProviderError(
                provider_name_lower, f"No default model for provider: {provider_name_lower}"
            )

    # Get base model based on provider
    if provider_name_lower == "ollama":
        base_model = _create_ollama_base_model(resolved_model, **kwargs)
    elif provider_name_lower == "openai":
        base_model = _create_openai_base_model(resolved_model, **kwargs)
    elif provider_name_lower == "anthropic":
        base_model = _create_anthropic_base_model(resolved_model, **kwargs)
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
    """Create base Ollama chat model (unstructured).

    Args:
        model: Model name.
        **kwargs: Model options including:
            - host: Ollama server URL (or use OLLAMA_HOST env var)
            - temperature: Sampling temperature (optional, uses model default if not provided)
            - top_p: Nucleus sampling parameter
            - seed: Random seed for reproducibility
            - num_ctx: Context window size (default 32768)

    Returns:
        Configured ChatOllama model.

    Raises:
        ProviderError: If OLLAMA_HOST not configured.
    """
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

    # Build model kwargs - only include parameters that are provided
    model_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": host,
        "num_ctx": kwargs.get("num_ctx", 32768),  # Default 32k to avoid truncation
    }

    # Temperature is provided by phase settings; if absent, model uses its default
    if "temperature" in kwargs:
        model_kwargs["temperature"] = kwargs["temperature"]

    if "top_p" in kwargs:
        model_kwargs["top_p"] = kwargs["top_p"]

    if "seed" in kwargs:
        model_kwargs["seed"] = kwargs["seed"]

    chat_model: BaseChatModel = ChatOllama(**model_kwargs)
    return chat_model


async def unload_ollama_model(chat_model: BaseChatModel) -> None:
    """Send keep_alive=0 to Ollama to immediately unload a model from VRAM.

    Used between pipeline phases when switching to a different Ollama model,
    so the outgoing model frees GPU memory for the incoming one.

    Safe to call on non-Ollama models (silently returns).

    Args:
        chat_model: The model to unload. Must have ``base_url`` and ``model``
            attributes (ChatOllama instances do).
    """
    import httpx

    base_url = getattr(chat_model, "base_url", None)
    model_name = getattr(chat_model, "model", None)
    if not base_url or not model_name:
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
            )
        log.info("ollama_model_unloaded", model=model_name)
    except Exception as e:
        log.warning("ollama_unload_failed", model=model_name, error=str(e))


def _is_reasoning_model(model: str) -> bool:
    """Check if model is an OpenAI reasoning model (o1/o3 families).

    Reasoning models have different API constraints:
    - No temperature parameter (they control their own reasoning)
    - No tool/function calling support
    - Use max_completion_tokens instead of max_tokens

    Args:
        model: Model name to check.

    Returns:
        True if model is from the o1 or o3 family (e.g., o1, o1-mini, o3).
    """
    model_lower = model.lower()
    return model_lower.startswith("o1") or model_lower.startswith("o3")


def _create_openai_base_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create base OpenAI chat model (unstructured).

    Args:
        model: Model name.
        **kwargs: Model options including:
            - api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            - temperature: Sampling temperature (optional, ignored for reasoning models)
            - top_p: Nucleus sampling parameter
            - seed: Random seed for reproducibility

    Returns:
        Configured ChatOpenAI model.

    Raises:
        ProviderError: If API key not configured.
    """
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

    # Build model kwargs
    model_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
    }

    # Reasoning models (o1, o1-mini, o3, etc.) don't support temperature
    if not _is_reasoning_model(model):
        if "temperature" in kwargs:
            model_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            model_kwargs["top_p"] = kwargs["top_p"]
    else:
        log.debug("reasoning_model_detected", model=model, note="skipping temperature parameter")

    # Seed is supported for all OpenAI models
    if "seed" in kwargs:
        model_kwargs["seed"] = kwargs["seed"]

    return ChatOpenAI(**model_kwargs)


def _create_anthropic_base_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create base Anthropic chat model (unstructured).

    Args:
        model: Model name.
        **kwargs: Model options including:
            - api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            - temperature: Sampling temperature (optional, uses model default if not provided)
            - top_p: Nucleus sampling parameter

    Note:
        Anthropic does not support the seed parameter. It is filtered out
        by PhaseSettings.to_model_kwargs() before reaching this function.

    Returns:
        Configured ChatAnthropic model.

    Raises:
        ProviderError: If API key not configured.
    """
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

    # Build model kwargs
    model_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
    }

    if "temperature" in kwargs:
        model_kwargs["temperature"] = kwargs["temperature"]

    if "top_p" in kwargs:
        model_kwargs["top_p"] = kwargs["top_p"]

    # Note: seed is not supported by Anthropic - filtered upstream

    return ChatAnthropic(**model_kwargs)
