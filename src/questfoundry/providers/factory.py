"""Factory for creating LLM providers.

Uses LangChain's init_chat_model abstraction for unified provider instantiation.
Provider-specific logic (Ollama num_ctx detection, reasoning model handling)
is applied as pre-processing before the unified call.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger
from questfoundry.providers.base import ProviderError
from questfoundry.providers.settings import filter_model_kwargs
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
    "google": "gemini-2.5-flash",
}

# Known provider names for validation
_KNOWN_PROVIDERS = frozenset({"ollama", "openai", "anthropic", "google"})

# Fallback models for providers where PROVIDER_DEFAULTS returns None
_FALLBACK_MODELS: dict[str, str] = {
    "ollama": "qwen3:4b-instruct-32k",
}


def get_default_model(provider_name: str) -> str | None:
    """Get default model for a provider.

    Returns None for providers that require explicit model specification.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic, google).

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
    and the stage protocol. Uses init_chat_model for unified instantiation
    with provider-specific pre-processing.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic, google).
        model: Model name/identifier.
        **kwargs: Additional provider-specific options.

    Returns:
        Configured BaseChatModel.

    Raises:
        ProviderError: If provider unavailable or misconfigured.
    """
    provider = _normalize_provider(provider_name)

    # Validate provider is known
    if provider not in _KNOWN_PROVIDERS:
        log.error("provider_unknown", provider=provider)
        raise ProviderError(provider, f"Unknown provider: {provider}")

    # Pre-processing: resolve provider-specific configuration
    kwargs = _preprocess_provider_kwargs(provider, model, kwargs)

    # Filter unsupported parameters (logs warnings for dropped params)
    kwargs = filter_model_kwargs(provider, model, kwargs)

    # Map internal provider name to init_chat_model's expected name
    provider_for_init = _map_provider_for_init(provider)

    # Create model using unified LangChain abstraction
    try:
        chat_model = _init_chat_model_safe(provider_for_init, model, **kwargs)
    except ImportError as e:
        package = _get_package_for_provider(provider)
        log.error("provider_import_error", provider=provider, package=package)
        raise ProviderError(
            provider,
            f"{package} not installed. Run: uv add {package}",
        ) from e

    log.info("chat_model_created", provider=provider, model=model)
    return chat_model


def _init_chat_model_safe(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    """Safely call init_chat_model with import error handling.

    Args:
        provider: Provider name for init_chat_model (e.g., 'google_genai').
        model: Model name.
        **kwargs: Model kwargs.

    Returns:
        Configured BaseChatModel.

    Raises:
        ImportError: If provider package is not installed.
    """
    from langchain.chat_models import init_chat_model

    # init_chat_model returns Any, but we know it returns BaseChatModel
    result: BaseChatModel = init_chat_model(model=model, model_provider=provider, **kwargs)
    return result


def _preprocess_provider_kwargs(
    provider: str,
    model: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Apply provider-specific pre-processing to kwargs.

    Handles:
    - Ollama: num_ctx detection, OLLAMA_HOST env var, base_url mapping
    - OpenAI: OPENAI_API_KEY env var
    - Anthropic: ANTHROPIC_API_KEY env var
    - Google: GOOGLE_API_KEY env var, google_api_key → api_key mapping

    Args:
        provider: Normalized provider name.
        model: Model name.
        kwargs: Input kwargs (will be copied, not mutated).

    Returns:
        Processed kwargs ready for filtering.

    Raises:
        ProviderError: If required configuration is missing.
    """
    kwargs = dict(kwargs)  # Don't mutate input

    if provider == "ollama":
        # Resolve host from kwargs or env var
        host = kwargs.pop("host", None) or os.getenv("OLLAMA_HOST")
        if not host:
            log.error("provider_config_error", provider="ollama", missing="OLLAMA_HOST")
            raise ProviderError(
                "ollama",
                "OLLAMA_HOST not configured. Set OLLAMA_HOST environment variable.",
            )
        kwargs["base_url"] = host

        # Query Ollama for num_ctx if not explicitly provided
        if "num_ctx" not in kwargs:
            num_ctx = _query_ollama_num_ctx(host, model)
            kwargs["num_ctx"] = num_ctx if num_ctx else 32_768

    elif provider == "openai":
        # Resolve API key from kwargs or env var (handles api_key=None case)
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.error("provider_config_error", provider="openai", missing="OPENAI_API_KEY")
            raise ProviderError(
                "openai",
                "API key required. Set OPENAI_API_KEY environment variable.",
            )
        kwargs["api_key"] = api_key

    elif provider == "anthropic":
        # Resolve API key from kwargs or env var (handles api_key=None case)
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            log.error("provider_config_error", provider="anthropic", missing="ANTHROPIC_API_KEY")
            raise ProviderError(
                "anthropic",
                "API key required. Set ANTHROPIC_API_KEY environment variable.",
            )
        kwargs["api_key"] = api_key

    elif provider == "google":
        # Resolve API key from kwargs or env var (accept google_api_key or api_key)
        api_key = (
            kwargs.pop("google_api_key", None)
            or kwargs.get("api_key")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            log.error("provider_config_error", provider="google", missing="GOOGLE_API_KEY")
            raise ProviderError(
                "google",
                "API key required. Set GOOGLE_API_KEY environment variable.",
            )
        kwargs["api_key"] = api_key

    return kwargs


def _map_provider_for_init(provider: str) -> str:
    """Map internal provider name to init_chat_model's expected name.

    Args:
        provider: Internal provider name (ollama, openai, anthropic, google).

    Returns:
        Provider name expected by init_chat_model.
    """
    # init_chat_model expects 'google_genai' not 'google'
    if provider == "google":
        return "google_genai"
    return provider


def _get_package_for_provider(provider: str) -> str:
    """Get the LangChain package name for a provider.

    Args:
        provider: Provider name.

    Returns:
        Package name for installation.
    """
    packages = {
        "ollama": "langchain-ollama",
        "openai": "langchain-openai",
        "anthropic": "langchain-anthropic",
        "google": "langchain-google-genai",
    }
    return packages.get(provider, f"langchain-{provider}")


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
        provider_name: Provider (ollama, openai, anthropic, google).
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
    provider = _normalize_provider(provider_name)

    # Resolve model name: use provided, then provider default, then convenience fallback
    resolved_model = model_name or get_default_model(provider)
    # Fallback for providers where get_default_model returns None (e.g., ollama)
    if resolved_model is None:
        resolved_model = _FALLBACK_MODELS.get(provider)
        if resolved_model is None:
            raise ProviderError(provider, f"No default model for provider: {provider}")

    # Create base model using the unified factory
    base_model = create_chat_model(provider, resolved_model, **kwargs)

    # Apply structured output if schema provided
    if schema is not None:
        base_model = with_structured_output(  # type: ignore[assignment]
            base_model,
            schema,
            strategy=strategy,
            provider_name=provider,
        )

    log.info(
        "model_created_structured",
        provider=provider,
        model=resolved_model,
        has_schema=schema is not None,
    )
    return base_model


def _normalize_provider(provider_name: str) -> str:
    """Normalize provider name, resolving aliases.

    Args:
        provider_name: Raw provider name (e.g., "gemini", "Google", "openai").

    Returns:
        Canonical lowercase provider name.
    """
    name = provider_name.lower()
    if name == "gemini":
        return "google"
    return name


def _query_ollama_num_ctx(host: str, model: str) -> int | None:
    """Query Ollama /api/show to get the model's configured num_ctx.

    Parses the ``parameters`` field from the response which contains the
    Modelfile-configured values (e.g., ``num_ctx  32768``).

    Args:
        host: Ollama server base URL.
        model: Model name.

    Returns:
        The num_ctx value from the model's configuration, or None if the
        query fails or the value is not found.
    """
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{host}/api/show", json={"model": model})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("ollama_show_failed", model=model, error=str(exc))
        return None

    # Parse the 'parameters' field: newline-separated "key  value" pairs
    parameters = data.get("parameters", "")
    for line in parameters.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "num_ctx":
            try:
                num_ctx = int(parts[-1])
                log.info("ollama_num_ctx_from_model", model=model, num_ctx=num_ctx)
                return num_ctx
            except ValueError:
                pass

    # Fallback: check model_info for architecture context_length
    model_info = data.get("model_info", {})
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            log.info("ollama_num_ctx_from_arch", model=model, num_ctx=value)
            return value

    return None


async def unload_ollama_model(chat_model: BaseChatModel) -> None:
    """Unload an Ollama model from VRAM and wait for it to complete.

    Used between pipeline phases when switching to a different Ollama model,
    so the outgoing model frees GPU memory for the incoming one.

    The unload is done in two steps:
    1. Send keep_alive=0 to trigger the unload
    2. Poll /api/ps until the model is no longer listed (max 30s)

    Safe to call on non-Ollama models (silently returns).

    Args:
        chat_model: The model to unload. Must have ``base_url`` and ``model``
            attributes (ChatOllama instances do).
    """
    import asyncio
    import time

    import httpx

    base_url = getattr(chat_model, "base_url", None)
    model_name = getattr(chat_model, "model", None)
    if not base_url or not model_name:
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Step 1: Request unload
            await client.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
            )

            # Step 2: Wait for model to actually unload (poll /api/ps)
            # Ollama unloads asynchronously, so we need to verify it's gone
            max_wait = 30.0  # seconds
            poll_interval = 0.5  # seconds
            start_time = time.monotonic()

            while (time.monotonic() - start_time) < max_wait:
                try:
                    response = await client.get(f"{base_url}/api/ps")
                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except (ValueError, TypeError):
                            # Malformed JSON response, keep polling
                            log.debug(
                                "ollama_ps_parse_error",
                                model=model_name,
                                status=response.status_code,
                            )
                            await asyncio.sleep(poll_interval)
                            continue

                        loaded_models = [m.get("name", "") for m in data.get("models", [])]
                        if model_name not in loaded_models:
                            # Model successfully unloaded
                            elapsed = time.monotonic() - start_time
                            log.info("ollama_model_unloaded", model=model_name, wait_time=elapsed)
                            return
                except Exception:
                    pass  # Ignore polling errors, keep trying

                await asyncio.sleep(poll_interval)

            # Timed out waiting for unload
            log.warning(
                "ollama_unload_timeout",
                model=model_name,
                timeout=max_wait,
            )

    except Exception as e:
        log.warning("ollama_unload_failed", model=model_name, error=str(e))
