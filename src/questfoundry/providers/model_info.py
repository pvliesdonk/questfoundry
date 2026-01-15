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


# Known context windows by provider and model.
# Used as fallback when API doesn't provide this information.
KNOWN_CONTEXT_WINDOWS: dict[str, dict[str, int]] = {
    "ollama": {
        "qwen3:8b": 32_768,
        "qwen2.5:7b": 32_768,
        "llama3:8b": 8_192,
        "llama3.1:8b": 128_000,
        "mistral:7b": 32_768,
        "deepseek-coder:6.7b": 16_384,
    },
    "openai": {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
        "o1": 200_000,
        "o1-mini": 128_000,
    },
    "anthropic": {
        "claude-sonnet-4-20250514": 200_000,
        "claude-opus-4-20250514": 200_000,
        "claude-3-5-sonnet-latest": 200_000,
        "claude-3-5-sonnet-20241022": 200_000,
        "claude-3-opus-20240229": 200_000,
        "claude-3-haiku-20240307": 200_000,
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
    provider_models = KNOWN_CONTEXT_WINDOWS.get(provider_lower, {})
    context_window = provider_models.get(model, DEFAULT_CONTEXT_WINDOW)

    # Vision support for known multimodal models
    supports_vision = model in {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    }

    return ModelInfo(
        context_window=context_window,
        supports_tools=True,  # All supported providers have tool support
        supports_vision=supports_vision,
    )
