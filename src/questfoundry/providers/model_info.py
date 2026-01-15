"""Model information and capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Model capabilities and limits.

    Attributes:
        context_window: Maximum input tokens the model can process.
        supports_tools: Whether the model supports tool/function calling.
        supports_vision: Whether the model can process images.
        max_output_tokens: Maximum tokens in model response (None if unknown).
    """

    context_window: int
    supports_tools: bool = True
    supports_vision: bool = False
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ModelProperties:
    """Known properties for a specific model.

    Used in KNOWN_MODELS registry to consolidate model metadata.
    """

    context_window: int
    supports_vision: bool = False


# Known model properties by provider and model name.
# Consolidates context window and capability information.
KNOWN_MODELS: dict[str, dict[str, ModelProperties]] = {
    "ollama": {
        "qwen3:8b": ModelProperties(context_window=32_768),
        "qwen2.5:7b": ModelProperties(context_window=32_768),
        "llama3:8b": ModelProperties(context_window=8_192),
        "llama3.1:8b": ModelProperties(context_window=128_000),
        "mistral:7b": ModelProperties(context_window=32_768),
        "deepseek-coder:6.7b": ModelProperties(context_window=16_384),
    },
    "openai": {
        "gpt-4o": ModelProperties(context_window=128_000, supports_vision=True),
        "gpt-4o-mini": ModelProperties(context_window=128_000, supports_vision=True),
        "gpt-4-turbo": ModelProperties(context_window=128_000, supports_vision=True),
        "gpt-4": ModelProperties(context_window=8_192),
        "gpt-3.5-turbo": ModelProperties(context_window=16_385),
        "o1": ModelProperties(context_window=200_000),
        "o1-mini": ModelProperties(context_window=128_000),
    },
    "anthropic": {
        "claude-sonnet-4-20250514": ModelProperties(context_window=200_000, supports_vision=True),
        "claude-opus-4-20250514": ModelProperties(context_window=200_000, supports_vision=True),
        "claude-3-5-sonnet-latest": ModelProperties(context_window=200_000, supports_vision=True),
        "claude-3-5-sonnet-20241022": ModelProperties(context_window=200_000, supports_vision=True),
        "claude-3-opus-20240229": ModelProperties(context_window=200_000, supports_vision=True),
        "claude-3-haiku-20240307": ModelProperties(context_window=200_000, supports_vision=True),
    },
}

# Default context window when model is not in known list.
# Conservative value to avoid context overflow.
DEFAULT_CONTEXT_WINDOW = 32_768


def get_model_info(provider: str, model: str) -> ModelInfo:
    """Get model information from known values or defaults.

    Args:
        provider: Provider name (e.g., "ollama", "openai", "anthropic").
        model: Model name (e.g., "gpt-4o", "qwen3:8b").

    Returns:
        ModelInfo with context window and capabilities.
    """
    provider_lower = provider.lower()
    provider_models = KNOWN_MODELS.get(provider_lower, {})
    props = provider_models.get(model)

    if props is not None:
        context_window = props.context_window
        supports_vision = props.supports_vision
    else:
        context_window = DEFAULT_CONTEXT_WINDOW
        supports_vision = False

    return ModelInfo(
        context_window=context_window,
        supports_tools=True,  # All supported providers have tool support
        supports_vision=supports_vision,
    )
