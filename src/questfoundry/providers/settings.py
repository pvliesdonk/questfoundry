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


# ---------------------------------------------------------------------------
# Provider Capabilities Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCapabilities:
    """What parameters a provider actually supports at construction time.

    Used to filter kwargs before passing to init_chat_model, preventing
    silent failures and runtime errors from unsupported parameters.
    """

    temperature: bool = True
    top_p: bool = True
    seed: bool = False
    stop: bool = True
    max_tokens_param: str = "max_tokens"  # Ollama uses "num_predict"
    supports_runtime_binding: bool = True  # Ollama does NOT respect bind()


PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "ollama": ProviderCapabilities(
        seed=True,
        max_tokens_param="num_predict",
        supports_runtime_binding=False,  # bind() kwargs ignored!
    ),
    "openai": ProviderCapabilities(
        seed=True,
        supports_runtime_binding=True,
    ),
    "anthropic": ProviderCapabilities(
        seed=False,  # Not supported by Anthropic API
        supports_runtime_binding=True,
    ),
    "google": ProviderCapabilities(
        seed=False,  # Not supported by Google Gemini API
        supports_runtime_binding=True,
    ),
}


@dataclass(frozen=True)
class ModelVariant:
    """Special model characteristics that affect parameter handling.

    Some models (like OpenAI's o1/o3 reasoning models) reject standard
    parameters at runtime even when the provider generally supports them.
    """

    rejects_temperature: bool = False
    rejects_stop: bool = False
    rejects_top_p: bool = False
    supports_reasoning_effort: bool = False
    supports_verbosity: bool = False


def _detect_model_variant(provider: str, model: str) -> ModelVariant:
    """Detect special model characteristics.

    Args:
        provider: Normalized provider name.
        model: Model name/identifier.

    Returns:
        ModelVariant describing the model's parameter restrictions.
    """
    model_lower = model.lower()

    # OpenAI reasoning models (o1, o3 families) don't support temperature/top_p
    if provider == "openai" and (model_lower.startswith("o1") or model_lower.startswith("o3")):
        return ModelVariant(
            rejects_temperature=True,
            rejects_top_p=True,
            supports_reasoning_effort=True,
        )

    # GPT-5 family: supports verbosity + reasoning_effort, rejects stop sequences
    if provider == "openai" and model_lower.startswith("gpt-5"):
        return ModelVariant(
            rejects_stop=True,
            supports_reasoning_effort=True,
            supports_verbosity=True,
        )

    return ModelVariant()


def get_provider_capabilities(provider: str) -> ProviderCapabilities:
    """Get capabilities for a provider.

    Args:
        provider: Provider name (ollama, openai, anthropic, google).

    Returns:
        ProviderCapabilities for the provider. Returns default capabilities
        (temperature, top_p, stop supported; seed not supported) for unknown
        providers.
    """
    return PROVIDER_CAPABILITIES.get(provider.lower(), ProviderCapabilities())


def filter_model_kwargs(
    provider: str,
    model: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Filter kwargs to only include supported parameters.

    Checks both provider-level capabilities and model-specific restrictions.
    Logs warnings for dropped parameters (graceful degradation).

    Args:
        provider: Normalized provider name.
        model: Model name/identifier.
        kwargs: Raw kwargs to filter.

    Returns:
        Filtered kwargs with only supported parameters.
    """
    caps = get_provider_capabilities(provider)
    variant = _detect_model_variant(provider, model)
    filtered: dict[str, Any] = {}

    for key, value in kwargs.items():
        # Skip None values (explicit None is treated as "not provided")
        if value is None:
            log.debug("param_is_none", param=key, action="skipping")
            continue

        # Check provider capabilities for seed
        if key == "seed" and not caps.seed:
            log.warning(
                "param_not_supported",
                param=key,
                provider=provider,
                reason="seed_not_supported",
            )
            continue

        # Check model variant restrictions
        if key == "temperature" and variant.rejects_temperature:
            log.warning(
                "param_rejected_by_model",
                param=key,
                model=model,
                reason="reasoning_model_controls_temperature",
            )
            continue

        if key == "top_p" and variant.rejects_top_p:
            log.warning(
                "param_rejected_by_model",
                param=key,
                model=model,
                reason="reasoning_model_controls_sampling",
            )
            continue

        if key == "stop" and variant.rejects_stop:
            log.warning(
                "param_rejected_by_model",
                param=key,
                model=model,
                reason="stop_sequences_not_supported",
            )
            continue

        # Filter reasoning_effort for models that don't support it
        if key == "reasoning_effort" and not variant.supports_reasoning_effort:
            log.warning(
                "param_not_supported",
                param=key,
                model=model,
                reason="model_does_not_support_reasoning_effort",
            )
            continue

        # Filter verbosity for models that don't support it.
        # Also strip verbosity from model_kwargs sub-dict.
        if (
            key == "model_kwargs"
            and isinstance(value, dict)
            and "verbosity" in value
            and not variant.supports_verbosity
        ):
            log.warning(
                "param_not_supported",
                param="verbosity",
                model=model,
                reason="model_does_not_support_verbosity",
            )
            value = {k: v for k, v in value.items() if k != "verbosity"}
            if not value:
                continue  # Skip empty model_kwargs

        # Translate max_tokens to provider-specific param name
        if key == "max_tokens" and caps.max_tokens_param != "max_tokens":
            filtered[caps.max_tokens_param] = value
            continue

        filtered[key] = value

    return filtered


VALID_REASONING_EFFORT = {"none", "minimal", "low", "medium", "high", "xhigh"}
VALID_VERBOSITY = {"low", "medium", "high"}


@dataclass
class PhaseSettings:
    """Model settings for a specific pipeline phase.

    Attributes:
        temperature: Override temperature, or None to use phase/provider default.
        top_p: Nucleus sampling parameter, or None to use provider default.
        seed: Random seed for reproducibility, or None.
        reasoning_effort: OpenAI reasoning depth (GPT-5/o1/o3), or None.
        verbosity: OpenAI output verbosity (GPT-5 family), or None.
    """

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None

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

        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
            # GPT-5 family rejects temperature/top_p when reasoning_effort != "none"
            if self.reasoning_effort != "none":
                for param in ("temperature", "top_p"):
                    if param in kwargs:
                        log.debug(
                            "param_suppressed_by_reasoning_effort",
                            param=param,
                            reasoning_effort=self.reasoning_effort,
                        )
                        del kwargs[param]

        if self.verbosity is not None:
            kwargs["model_kwargs"] = {"verbosity": self.verbosity}

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
        reasoning_effort = data.get("reasoning_effort")
        verbosity = data.get("verbosity")

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

        # Validate reasoning_effort
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).lower()
            if reasoning_effort not in VALID_REASONING_EFFORT:
                msg = f"reasoning_effort must be one of {sorted(VALID_REASONING_EFFORT)}, got '{reasoning_effort}'"
                raise ValueError(msg)

        # Validate verbosity
        if verbosity is not None:
            verbosity = str(verbosity).lower()
            if verbosity not in VALID_VERBOSITY:
                msg = f"verbosity must be one of {sorted(VALID_VERBOSITY)}, got '{verbosity}'"
                raise ValueError(msg)

        return cls(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
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
