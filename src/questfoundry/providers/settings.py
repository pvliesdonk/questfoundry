"""Phase-specific model settings with provider-aware temperature mapping.

Temperature scales differ across providers - Anthropic's 1.0 is more conservative
than OpenAI's 1.0. This module provides semantic creativity levels that map to
provider-appropriate temperature values.

See: https://gist.github.com/pvliesdonk/35b36897a42c41b371c8898bfec55882
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from questfoundry.observability.logging import get_logger

log = get_logger(__name__)


class CreativityLevel(Enum):
    """Semantic creativity levels that map to provider-specific temperatures.

    These levels abstract away provider-specific temperature scales:
    - Anthropic's 1.0 is more conservative than OpenAI's 1.0
    - Ollama models vary, but typically use 0.0-1.0 scale
    """

    DETERMINISTIC = "deterministic"  # Structured output, minimal variance
    FOCUSED = "focused"  # Consistent but some variation
    BALANCED = "balanced"  # Default, good for most tasks
    CREATIVE = "creative"  # Exploratory, high variance


# Provider-specific temperature mappings
# Based on empirical testing and provider documentation
TEMPERATURE_MAP: dict[str, dict[CreativityLevel, float]] = {
    "openai": {
        CreativityLevel.DETERMINISTIC: 0.0,
        CreativityLevel.FOCUSED: 0.3,
        CreativityLevel.BALANCED: 0.7,
        CreativityLevel.CREATIVE: 0.9,
    },
    "anthropic": {
        CreativityLevel.DETERMINISTIC: 0.0,
        CreativityLevel.FOCUSED: 0.3,
        CreativityLevel.BALANCED: 0.5,
        CreativityLevel.CREATIVE: 0.8,
    },
    "ollama": {
        CreativityLevel.DETERMINISTIC: 0.0,
        CreativityLevel.FOCUSED: 0.2,
        CreativityLevel.BALANCED: 0.5,
        CreativityLevel.CREATIVE: 0.8,
    },
    "google": {
        CreativityLevel.DETERMINISTIC: 0.0,
        CreativityLevel.FOCUSED: 0.3,
        CreativityLevel.BALANCED: 0.7,
        CreativityLevel.CREATIVE: 0.9,
    },
}

# Role defaults (semantic level, not raw temperature)
# Based on agent prompt engineering best practices:
# - creative: high creativity for exploration and prose generation
# - balanced: balanced for coherent narratives and summarization
# - structured: deterministic for structured output and serialization
PHASE_CREATIVITY: dict[str, CreativityLevel] = {
    # Role-based names (primary)
    "creative": CreativityLevel.CREATIVE,
    "balanced": CreativityLevel.BALANCED,
    "structured": CreativityLevel.DETERMINISTIC,
    # Legacy phase names (aliases)
    "discuss": CreativityLevel.CREATIVE,
    "summarize": CreativityLevel.BALANCED,
    "serialize": CreativityLevel.DETERMINISTIC,
}


def get_temperature_for_phase(phase: str, provider: str) -> float:
    """Get provider-appropriate temperature for a phase.

    Args:
        phase: Pipeline phase (discuss, summarize, serialize).
        provider: Provider name (ollama, openai, anthropic).

    Returns:
        Temperature value appropriate for the provider's scale.
    """
    level = PHASE_CREATIVITY.get(phase, CreativityLevel.BALANCED)
    provider_key = provider.lower()
    temps = TEMPERATURE_MAP.get(provider_key, TEMPERATURE_MAP["ollama"])
    return temps[level]


def get_max_temperature(provider: str) -> float:
    """Get the maximum valid temperature for a provider.

    Args:
        provider: Provider name (ollama, openai, anthropic).

    Returns:
        Maximum temperature value for the provider.
    """
    provider_limits: dict[str, float] = {
        "openai": 2.0,
        "anthropic": 1.0,
        "ollama": 2.0,  # Ollama allows > 1.0 though rarely useful
        "google": 2.0,
    }
    return provider_limits.get(provider.lower(), 1.0)


@dataclass
class PhaseSettings:
    """Model settings for a specific pipeline phase.

    Attributes:
        temperature: Override temperature, or None to use phase/provider default.
        top_p: Nucleus sampling parameter, or None to use provider default.
        seed: Random seed for reproducibility, or None.
    """

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def to_model_kwargs(self, phase: str, provider: str) -> dict[str, Any]:
        """Convert to kwargs for model creation.

        Applies provider-specific temperature clamping when user provides
        explicit values that exceed provider limits.

        Args:
            phase: Pipeline phase (discuss, summarize, serialize).
            provider: Provider name for temperature mapping.

        Returns:
            Dictionary of model kwargs (temperature, top_p, seed).
        """
        kwargs: dict[str, Any] = {}

        # Use explicit temperature if set, otherwise use phase/provider default
        if self.temperature is not None:
            max_temp = get_max_temperature(provider)
            if self.temperature > max_temp:
                log.warning(
                    "temperature_clamped",
                    provider=provider,
                    requested=self.temperature,
                    max=max_temp,
                )
                kwargs["temperature"] = max_temp
            else:
                kwargs["temperature"] = self.temperature
        else:
            kwargs["temperature"] = get_temperature_for_phase(phase, provider)

        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if self.seed is not None:
            # Anthropic and Google don't support seed - log warning and skip
            if provider.lower() in ("anthropic", "google"):
                log.warning(
                    "seed_not_supported",
                    provider=provider,
                    note=f"{provider.lower()} does not support seed parameter, ignoring",
                )
            else:
                kwargs["seed"] = self.seed

        return kwargs

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PhaseSettings:
        """Create from config dict.

        Args:
            data: Dictionary with optional temperature, top_p, seed keys.

        Returns:
            PhaseSettings instance.

        Raises:
            ValueError: If values are out of valid range.
        """
        if not data:
            return cls()

        temperature = data.get("temperature")
        top_p = data.get("top_p")
        seed = data.get("seed")

        # Validate temperature (non-negative)
        if temperature is not None and temperature < 0:
            msg = f"temperature must be non-negative, got {temperature}"
            raise ValueError(msg)

        # Validate top_p (0 to 1 inclusive)
        if top_p is not None and not (0 <= top_p <= 1):
            msg = f"top_p must be between 0 and 1, got {top_p}"
            raise ValueError(msg)

        # Validate seed (must be an integer)
        if seed is not None and not isinstance(seed, int):
            msg = f"seed must be an integer, got {type(seed).__name__}"
            raise ValueError(msg)

        return cls(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )


def get_default_phase_settings(_phase: str) -> PhaseSettings:
    """Get default settings for a phase.

    This returns an empty PhaseSettings - the actual temperature will be
    computed based on the phase and provider when to_model_kwargs is called.

    Args:
        _phase: Pipeline phase name (unused - defaults are computed dynamically).

    Returns:
        PhaseSettings with no overrides (uses phase/provider defaults).
    """
    # Return empty settings - defaults are computed dynamically
    # based on phase and provider in to_model_kwargs
    return PhaseSettings()
