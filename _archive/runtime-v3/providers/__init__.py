"""LLM providers for the runtime.

Supported providers:
- ollama: Local inference via Ollama
- google: Google AI Studio (Gemini)
- openai: OpenAI API (via langchain-openai)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel

from questfoundry.runtime.providers.ollama import (
    check_ollama_available,
    create_ollama_llm,
    list_ollama_models,
)

if TYPE_CHECKING:
    from questfoundry.runtime.config import QuestFoundrySettings

__all__ = [
    # Ollama
    "create_ollama_llm",
    "check_ollama_available",
    "list_ollama_models",
    # Google
    "create_google_llm",
    "check_google_available",
    "list_google_models",
    # OpenAI
    "create_openai_llm",
    "check_openai_available",
    "list_openai_models",
    # Factory
    "create_llm_from_config",
]


# Lazy imports for optional dependencies
def create_google_llm(
    model: str = "gemini-2.5-pro-preview-05-06",
    temperature: float = 0.7,
    api_key: str | None = None,
    thinking_budget: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a Google AI Studio (Gemini) LLM. See providers.google for details."""
    from questfoundry.runtime.providers.google import (
        create_google_llm as _create_google_llm,
    )

    return _create_google_llm(
        model=model,
        temperature=temperature,
        api_key=api_key,
        thinking_budget=thinking_budget,
        **kwargs,
    )


def check_google_available(api_key: str | None = None) -> bool:
    """Check if Google AI Studio is available."""
    from questfoundry.runtime.providers.google import (
        check_google_available as _check_google_available,
    )

    return _check_google_available(api_key)


def list_google_models() -> list[str]:
    """List available Gemini models."""
    from questfoundry.runtime.providers.google import (
        list_google_models as _list_google_models,
    )

    return _list_google_models()


def create_openai_llm(
    model: str = "gpt-4o",
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create an OpenAI LLM. See providers.openai for details."""
    from questfoundry.runtime.providers.openai import (
        create_openai_llm as _create_openai_llm,
    )

    return _create_openai_llm(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def check_openai_available(api_key: str | None = None) -> bool:
    """Check if OpenAI API is available."""
    from questfoundry.runtime.providers.openai import (
        check_openai_available as _check_openai_available,
    )

    return _check_openai_available(api_key)


def list_openai_models() -> list[dict[str, Any]]:
    """List available OpenAI models."""
    from questfoundry.runtime.providers.openai import (
        list_openai_models as _list_openai_models,
    )

    return _list_openai_models()


def create_llm_from_config(settings: QuestFoundrySettings | None = None) -> BaseChatModel:
    """Create an LLM based on configuration.

    This factory function creates the appropriate LLM based on the
    provider setting in the configuration.

    Parameters
    ----------
    settings : QuestFoundrySettings | None
        Settings to use. If None, loads from get_settings().

    Returns
    -------
    BaseChatModel
        Configured LLM instance.

    Raises
    ------
    ValueError
        If provider is not supported.
    ImportError
        If required provider package is not installed.

    Examples
    --------
    Use configured provider::

        from questfoundry.runtime.providers import create_llm_from_config

        llm = create_llm_from_config()

    With custom settings::

        from questfoundry.runtime.config import get_settings

        settings = get_settings()
        settings.llm.provider = "google"
        llm = create_llm_from_config(settings)
    """
    if settings is None:
        from questfoundry.runtime.config import get_settings

        settings = get_settings()

    provider = settings.llm.provider
    model = settings.get_llm_model()
    temperature = settings.llm.temperature

    if provider == "ollama":
        return create_ollama_llm(
            model=model,
            base_url=settings.ollama.host,
            temperature=temperature,
        )

    elif provider == "google":
        return create_google_llm(
            model=model,
            temperature=temperature,
            thinking_budget=settings.google.thinking_budget,
        )

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI support. "
                "Install with: pip install langchain-openai"
            ) from e

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
        }
        if settings.openai.api_base:
            kwargs["base_url"] = settings.openai.api_base

        return ChatOpenAI(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, google, openai")
