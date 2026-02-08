"""Model information and capabilities."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Model capabilities and limits.

    Attributes:
        context_window: Maximum input tokens the model can process.
        supports_tools: Whether the model supports tool/function calling.
        supports_vision: Whether the model can process images.
        max_output_tokens: Maximum tokens in model response (None if unknown).
        max_concurrency: Maximum concurrent LLM calls for batching.
    """

    context_window: int
    supports_tools: bool = True
    supports_vision: bool = False
    max_output_tokens: int | None = None
    max_concurrency: int = 2


@dataclass(frozen=True)
class ModelProperties:
    """Known properties for a specific model.

    Used in KNOWN_MODELS registry to consolidate model metadata.

    Attributes:
        context_window: Maximum input tokens the model can process.
        supports_vision: Whether the model can process images.
        supports_tools: Whether the model supports tool/function calling.
        capability_tier: Model capability classification ("small" or "large").
            Small models (<=13B params) copy exemplar content verbatim;
            large models can abstract style from content.
    """

    context_window: int
    supports_vision: bool = False
    supports_tools: bool = True  # Most models support tools; o1 family doesn't
    capability_tier: str = "small"  # "small" or "large"


# Known model properties by provider and model name.
# Consolidates context window and capability information.
KNOWN_MODELS: dict[str, dict[str, ModelProperties]] = {
    "ollama": {
        "qwen3:4b-instruct-32k": ModelProperties(context_window=32_768),
        "qwen3:8b": ModelProperties(context_window=32_768),
        "qwen2.5:7b": ModelProperties(context_window=32_768),
        "llama3:8b": ModelProperties(context_window=8_192),
        "llama3.1:8b": ModelProperties(context_window=128_000),
        "mistral:7b": ModelProperties(context_window=32_768),
        "deepseek-coder:6.7b": ModelProperties(context_window=16_384),
    },
    "openai": {
        "gpt-5-mini": ModelProperties(
            context_window=1_000_000, supports_vision=True, capability_tier="large"
        ),
        "gpt-4o": ModelProperties(
            context_window=128_000, supports_vision=True, capability_tier="large"
        ),
        "gpt-4o-mini": ModelProperties(
            context_window=128_000, supports_vision=True, capability_tier="large"
        ),
        "gpt-4-turbo": ModelProperties(
            context_window=128_000, supports_vision=True, capability_tier="large"
        ),
        "gpt-4": ModelProperties(context_window=8_192, capability_tier="large"),
        "gpt-3.5-turbo": ModelProperties(context_window=16_385, capability_tier="large"),
        # Reasoning models: no tool support, no temperature parameter
        "o1": ModelProperties(
            context_window=200_000, supports_tools=False, capability_tier="large"
        ),
        "o1-mini": ModelProperties(
            context_window=128_000, supports_tools=False, capability_tier="large"
        ),
        "o1-preview": ModelProperties(
            context_window=128_000, supports_tools=False, capability_tier="large"
        ),
        "o3": ModelProperties(
            context_window=200_000, supports_tools=False, capability_tier="large"
        ),
        "o3-mini": ModelProperties(
            context_window=200_000, supports_tools=False, capability_tier="large"
        ),
    },
    "anthropic": {
        "claude-sonnet-4-20250514": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
        "claude-opus-4-20250514": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
        "claude-3-5-sonnet-latest": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
        "claude-3-5-sonnet-20241022": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
        "claude-3-opus-20240229": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
        "claude-3-haiku-20240307": ModelProperties(
            context_window=200_000, supports_vision=True, capability_tier="large"
        ),
    },
    "google": {
        "gemini-2.5-flash": ModelProperties(
            context_window=1_000_000, supports_vision=True, capability_tier="large"
        ),
        "gemini-2.5-pro": ModelProperties(
            context_window=1_000_000, supports_vision=True, capability_tier="large"
        ),
        "gemini-2.0-flash": ModelProperties(
            context_window=1_000_000, supports_vision=True, capability_tier="large"
        ),
    },
}

# Default context window when model is not in known list.
# Conservative value to avoid context overflow.
DEFAULT_CONTEXT_WINDOW = 32_768


# Default max concurrency per provider.
# Ollama: local GPU, limited parallelism.
# OpenAI/Anthropic: cloud APIs with higher rate limits.
_PROVIDER_MAX_CONCURRENCY: dict[str, int] = {
    "ollama": 2,
    "openai": 20,
    "anthropic": 10,
    "google": 20,
}


def get_model_info(provider: str, model: str) -> ModelInfo:
    """Get model information from known values or defaults.

    Args:
        provider: Provider name (e.g., "ollama", "openai", "anthropic").
        model: Model name (e.g., "gpt-5-mini", "qwen3:4b-instruct-32k").

    Returns:
        ModelInfo with context window and capabilities.
    """
    provider_lower = provider.lower()
    provider_models = KNOWN_MODELS.get(provider_lower, {})
    props = provider_models.get(model)

    if props is not None:
        context_window = props.context_window
        supports_vision = props.supports_vision
        supports_tools = props.supports_tools
    else:
        context_window = DEFAULT_CONTEXT_WINDOW
        supports_vision = False
        supports_tools = True  # Default to True for unknown models

    # Concurrency: env var override > provider default > fallback
    env_concurrency = os.environ.get("QF_MAX_CONCURRENCY")
    if env_concurrency is not None:
        try:
            max_concurrency = int(env_concurrency)
            if max_concurrency <= 0:
                max_concurrency = 1
        except ValueError:
            max_concurrency = _PROVIDER_MAX_CONCURRENCY.get(provider_lower, 2)
    else:
        max_concurrency = _PROVIDER_MAX_CONCURRENCY.get(provider_lower, 2)

    return ModelInfo(
        context_window=context_window,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        max_concurrency=max_concurrency,
    )


# Name patterns that indicate small models (<=13B parameters).
_SMALL_MODEL_PATTERNS = ("1b", "2b", "3b", "4b", "7b", "8b", "13b")

# Cloud providers where even "mini" models are large enough to abstract style.
_LARGE_PROVIDERS = frozenset({"openai", "anthropic", "google"})


def get_capability_tier(provider: str, model: str) -> str:
    """Resolve the model's capability tier.

    Determines whether a model is "small" (copies exemplar content verbatim)
    or "large" (can abstract style from exemplar content).

    Resolution order:
    1. Explicit annotation in KNOWN_MODELS registry
    2. Name-pattern heuristic for unknown models
    3. Cloud provider default (large) or conservative default (small)

    Args:
        provider: Provider name (e.g., "ollama", "openai").
        model: Model name (e.g., "qwen3:4b-instruct-32k", "gpt-4o").

    Returns:
        "small" or "large".
    """
    provider_lower = provider.lower()
    props = KNOWN_MODELS.get(provider_lower, {}).get(model)
    if props is not None:
        return props.capability_tier

    # Heuristic for unknown models
    model_lower = model.lower()
    if any(f"{p}" in model_lower for p in _SMALL_MODEL_PATTERNS):
        return "small"

    # Cloud providers default to large
    if provider_lower in _LARGE_PROVIDERS:
        return "large"

    # Conservative default: unknown local models are small
    return "small"
