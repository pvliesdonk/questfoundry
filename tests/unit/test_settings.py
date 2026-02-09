"""Tests for phase-specific model settings."""

from __future__ import annotations

import pytest

from questfoundry.providers.settings import (
    PHASE_CREATIVITY,
    TEMPERATURE_MAP,
    VALID_REASONING_EFFORT,
    VALID_VERBOSITY,
    CreativityLevel,
    ModelVariant,
    PhaseSettings,
    _detect_model_variant,
    filter_model_kwargs,
    get_default_phase_settings,
    get_max_temperature,
    get_temperature_for_phase,
)


class TestCreativityLevel:
    """Tests for CreativityLevel enum."""

    def test_all_levels_defined(self) -> None:
        """All creativity levels are defined."""
        assert CreativityLevel.DETERMINISTIC.value == "deterministic"
        assert CreativityLevel.FOCUSED.value == "focused"
        assert CreativityLevel.BALANCED.value == "balanced"
        assert CreativityLevel.CREATIVE.value == "creative"


class TestTemperatureMap:
    """Tests for temperature mapping configuration."""

    def test_all_providers_have_mappings(self) -> None:
        """All supported providers have temperature mappings."""
        assert "openai" in TEMPERATURE_MAP
        assert "anthropic" in TEMPERATURE_MAP
        assert "ollama" in TEMPERATURE_MAP
        assert "google" in TEMPERATURE_MAP

    def test_all_levels_mapped_for_each_provider(self) -> None:
        """Each provider has mappings for all creativity levels."""
        for provider, temps in TEMPERATURE_MAP.items():
            for level in CreativityLevel:
                assert level in temps, f"{provider} missing {level}"

    def test_deterministic_is_zero(self) -> None:
        """Deterministic level is 0.0 for all providers."""
        for provider, temps in TEMPERATURE_MAP.items():
            assert temps[CreativityLevel.DETERMINISTIC] == 0.0, f"{provider}"

    def test_creative_is_highest(self) -> None:
        """Creative level has highest temperature for each provider."""
        for provider, temps in TEMPERATURE_MAP.items():
            creative_temp = temps[CreativityLevel.CREATIVE]
            for level, temp in temps.items():
                assert temp <= creative_temp, f"{provider} {level}"


class TestPhaseCreativity:
    """Tests for phase creativity defaults."""

    def test_discuss_is_creative(self) -> None:
        """Discuss phase uses creative temperature."""
        assert PHASE_CREATIVITY["discuss"] == CreativityLevel.CREATIVE

    def test_summarize_is_balanced(self) -> None:
        """Summarize phase uses balanced temperature."""
        assert PHASE_CREATIVITY["summarize"] == CreativityLevel.BALANCED

    def test_serialize_is_deterministic(self) -> None:
        """Serialize phase uses deterministic temperature."""
        assert PHASE_CREATIVITY["serialize"] == CreativityLevel.DETERMINISTIC


class TestGetTemperatureForPhase:
    """Tests for get_temperature_for_phase function."""

    @pytest.mark.parametrize(
        ("phase", "provider", "expected"),
        [
            # Discuss phase (CREATIVE)
            ("discuss", "openai", 0.9),
            ("discuss", "anthropic", 0.8),
            ("discuss", "ollama", 0.8),
            # Summarize phase (BALANCED)
            ("summarize", "openai", 0.7),
            ("summarize", "anthropic", 0.5),
            ("summarize", "ollama", 0.5),
            # Google (same scale as OpenAI)
            ("discuss", "google", 0.9),
            ("summarize", "google", 0.7),
            ("serialize", "google", 0.0),
            # Serialize phase (DETERMINISTIC)
            ("serialize", "openai", 0.0),
            ("serialize", "anthropic", 0.0),
            ("serialize", "ollama", 0.0),
        ],
    )
    def test_phase_provider_temperatures(self, phase: str, provider: str, expected: float) -> None:
        """Correct temperature for each phase/provider combination."""
        assert get_temperature_for_phase(phase, provider) == expected

    def test_unknown_phase_uses_balanced(self) -> None:
        """Unknown phase falls back to BALANCED."""
        temp = get_temperature_for_phase("unknown_phase", "openai")
        assert temp == TEMPERATURE_MAP["openai"][CreativityLevel.BALANCED]

    def test_unknown_provider_uses_ollama_defaults(self) -> None:
        """Unknown provider falls back to Ollama defaults."""
        temp = get_temperature_for_phase("discuss", "unknown_provider")
        assert temp == TEMPERATURE_MAP["ollama"][CreativityLevel.CREATIVE]

    def test_case_insensitive_provider(self) -> None:
        """Provider lookup is case insensitive."""
        assert get_temperature_for_phase("discuss", "OPENAI") == 0.9
        assert get_temperature_for_phase("discuss", "OpenAI") == 0.9


class TestGetMaxTemperature:
    """Tests for get_max_temperature function."""

    def test_openai_max(self) -> None:
        """OpenAI has max temperature of 2.0."""
        assert get_max_temperature("openai") == 2.0

    def test_anthropic_max(self) -> None:
        """Anthropic has max temperature of 1.0."""
        assert get_max_temperature("anthropic") == 1.0

    def test_ollama_max(self) -> None:
        """Ollama has max temperature of 2.0."""
        assert get_max_temperature("ollama") == 2.0

    def test_google_max(self) -> None:
        """Google has max temperature of 2.0."""
        assert get_max_temperature("google") == 2.0

    def test_unknown_provider_defaults_to_1(self) -> None:
        """Unknown provider defaults to 1.0 max."""
        assert get_max_temperature("unknown") == 1.0

    def test_case_insensitive(self) -> None:
        """Provider lookup is case insensitive."""
        assert get_max_temperature("ANTHROPIC") == 1.0


class TestPhaseSettings:
    """Tests for PhaseSettings dataclass."""

    def test_default_values(self) -> None:
        """Default PhaseSettings has all None values."""
        settings = PhaseSettings()
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.seed is None
        assert settings.reasoning_effort is None
        assert settings.verbosity is None

    def test_from_dict_empty(self) -> None:
        """from_dict with empty dict returns defaults."""
        settings = PhaseSettings.from_dict({})
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.seed is None
        assert settings.reasoning_effort is None
        assert settings.verbosity is None

    def test_from_dict_none(self) -> None:
        """from_dict with None returns defaults."""
        settings = PhaseSettings.from_dict(None)
        assert settings.temperature is None

    def test_from_dict_with_values(self) -> None:
        """from_dict parses all values."""
        settings = PhaseSettings.from_dict(
            {
                "temperature": 0.5,
                "top_p": 0.9,
                "seed": 42,
            }
        )
        assert settings.temperature == 0.5
        assert settings.top_p == 0.9
        assert settings.seed == 42

    def test_from_dict_partial(self) -> None:
        """from_dict handles partial values."""
        settings = PhaseSettings.from_dict({"temperature": 0.3})
        assert settings.temperature == 0.3
        assert settings.top_p is None
        assert settings.seed is None

    def test_from_dict_rejects_negative_temperature(self) -> None:
        """from_dict rejects negative temperature."""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            PhaseSettings.from_dict({"temperature": -0.5})

    def test_from_dict_accepts_zero_temperature(self) -> None:
        """from_dict accepts zero temperature."""
        settings = PhaseSettings.from_dict({"temperature": 0.0})
        assert settings.temperature == 0.0

    def test_from_dict_rejects_top_p_below_zero(self) -> None:
        """from_dict rejects top_p below 0."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            PhaseSettings.from_dict({"top_p": -0.1})

    def test_from_dict_rejects_top_p_above_one(self) -> None:
        """from_dict rejects top_p above 1."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            PhaseSettings.from_dict({"top_p": 1.5})

    def test_from_dict_accepts_boundary_top_p(self) -> None:
        """from_dict accepts top_p at boundaries (0 and 1)."""
        settings = PhaseSettings.from_dict({"top_p": 0.0})
        assert settings.top_p == 0.0
        settings = PhaseSettings.from_dict({"top_p": 1.0})
        assert settings.top_p == 1.0

    def test_from_dict_rejects_non_integer_seed(self) -> None:
        """from_dict rejects non-integer seed."""
        with pytest.raises(ValueError, match="seed must be an integer"):
            PhaseSettings.from_dict({"seed": 42.5})

    def test_from_dict_accepts_integer_seed(self) -> None:
        """from_dict accepts integer seed."""
        settings = PhaseSettings.from_dict({"seed": 42})
        assert settings.seed == 42

    # --- reasoning_effort ---

    @pytest.mark.parametrize("value", sorted(VALID_REASONING_EFFORT))
    def test_from_dict_accepts_valid_reasoning_effort(self, value: str) -> None:
        """from_dict accepts all valid reasoning_effort values."""
        settings = PhaseSettings.from_dict({"reasoning_effort": value})
        assert settings.reasoning_effort == value

    def test_from_dict_normalizes_reasoning_effort_case(self) -> None:
        """from_dict lowercases reasoning_effort."""
        settings = PhaseSettings.from_dict({"reasoning_effort": "HIGH"})
        assert settings.reasoning_effort == "high"

    def test_from_dict_rejects_invalid_reasoning_effort(self) -> None:
        """from_dict rejects invalid reasoning_effort values."""
        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            PhaseSettings.from_dict({"reasoning_effort": "ultra"})

    # --- verbosity ---

    @pytest.mark.parametrize("value", sorted(VALID_VERBOSITY))
    def test_from_dict_accepts_valid_verbosity(self, value: str) -> None:
        """from_dict accepts all valid verbosity values."""
        settings = PhaseSettings.from_dict({"verbosity": value})
        assert settings.verbosity == value

    def test_from_dict_normalizes_verbosity_case(self) -> None:
        """from_dict lowercases verbosity."""
        settings = PhaseSettings.from_dict({"verbosity": "Low"})
        assert settings.verbosity == "low"

    def test_from_dict_rejects_invalid_verbosity(self) -> None:
        """from_dict rejects invalid verbosity values."""
        with pytest.raises(ValueError, match="verbosity must be one of"):
            PhaseSettings.from_dict({"verbosity": "extreme"})

    def test_from_dict_with_all_fields(self) -> None:
        """from_dict parses all fields including reasoning_effort and verbosity."""
        settings = PhaseSettings.from_dict(
            {
                "temperature": 0.5,
                "top_p": 0.9,
                "seed": 42,
                "reasoning_effort": "high",
                "verbosity": "low",
            }
        )
        assert settings.temperature == 0.5
        assert settings.top_p == 0.9
        assert settings.seed == 42
        assert settings.reasoning_effort == "high"
        assert settings.verbosity == "low"


class TestPhaseSettingsToModelKwargs:
    """Tests for PhaseSettings.to_model_kwargs method."""

    def test_uses_phase_default_when_no_override(self) -> None:
        """Uses phase/provider default temperature when not overridden."""
        settings = PhaseSettings()
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["temperature"] == 0.9  # CREATIVE for OpenAI

    def test_uses_explicit_temperature(self) -> None:
        """Uses explicit temperature when provided."""
        settings = PhaseSettings(temperature=0.5)
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["temperature"] == 0.5

    def test_includes_top_p_when_set(self) -> None:
        """Includes top_p in kwargs when set."""
        settings = PhaseSettings(top_p=0.95)
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["top_p"] == 0.95

    def test_excludes_top_p_when_none(self) -> None:
        """Excludes top_p from kwargs when None."""
        settings = PhaseSettings()
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert "top_p" not in kwargs

    def test_includes_seed_for_openai(self) -> None:
        """Includes seed in kwargs for OpenAI."""
        settings = PhaseSettings(seed=42)
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["seed"] == 42

    def test_includes_seed_for_ollama(self) -> None:
        """Includes seed in kwargs for Ollama."""
        settings = PhaseSettings(seed=42)
        kwargs = settings.to_model_kwargs("discuss", "ollama")
        assert kwargs["seed"] == 42

    def test_excludes_seed_for_anthropic(self) -> None:
        """Excludes seed from kwargs for Anthropic (not supported)."""
        settings = PhaseSettings(seed=42)
        kwargs = settings.to_model_kwargs("discuss", "anthropic")
        assert "seed" not in kwargs

    def test_excludes_seed_for_google(self) -> None:
        """Excludes seed from kwargs for Google (not supported)."""
        settings = PhaseSettings(seed=42)
        kwargs = settings.to_model_kwargs("discuss", "google")
        assert "seed" not in kwargs

    def test_clamps_temperature_for_anthropic(self) -> None:
        """Clamps temperature to provider max."""
        settings = PhaseSettings(temperature=1.5)
        kwargs = settings.to_model_kwargs("discuss", "anthropic")
        assert kwargs["temperature"] == 1.0  # Clamped to Anthropic max

    def test_does_not_clamp_within_range(self) -> None:
        """Does not clamp temperature within provider range."""
        settings = PhaseSettings(temperature=1.5)
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["temperature"] == 1.5  # OpenAI allows up to 2.0

    # --- reasoning_effort in kwargs ---

    def test_includes_reasoning_effort(self) -> None:
        """Includes reasoning_effort in kwargs when set."""
        settings = PhaseSettings(reasoning_effort="high")
        kwargs = settings.to_model_kwargs("serialize", "openai")
        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_suppresses_temperature(self) -> None:
        """reasoning_effort != 'none' suppresses temperature from kwargs."""
        settings = PhaseSettings(temperature=0.5, reasoning_effort="high")
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert "temperature" not in kwargs
        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_suppresses_top_p(self) -> None:
        """reasoning_effort != 'none' suppresses top_p from kwargs."""
        settings = PhaseSettings(top_p=0.9, reasoning_effort="medium")
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert "top_p" not in kwargs
        assert kwargs["reasoning_effort"] == "medium"

    def test_reasoning_effort_none_keeps_temperature(self) -> None:
        """reasoning_effort='none' does NOT suppress temperature."""
        settings = PhaseSettings(temperature=0.5, reasoning_effort="none")
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert kwargs["temperature"] == 0.5
        assert kwargs["reasoning_effort"] == "none"

    def test_reasoning_effort_suppresses_phase_default_temperature(self) -> None:
        """reasoning_effort suppresses even auto-calculated phase temperature."""
        settings = PhaseSettings(reasoning_effort="high")
        kwargs = settings.to_model_kwargs("discuss", "openai")
        # Phase default would add temperature=0.9, but reasoning_effort suppresses it
        assert "temperature" not in kwargs

    # --- verbosity in kwargs ---

    def test_includes_verbosity_in_model_kwargs(self) -> None:
        """Includes verbosity inside model_kwargs sub-dict."""
        settings = PhaseSettings(verbosity="low")
        kwargs = settings.to_model_kwargs("summarize", "openai")
        assert kwargs["model_kwargs"] == {"verbosity": "low"}

    def test_no_model_kwargs_without_verbosity(self) -> None:
        """model_kwargs key absent when verbosity is None."""
        settings = PhaseSettings()
        kwargs = settings.to_model_kwargs("discuss", "openai")
        assert "model_kwargs" not in kwargs


class TestGetDefaultPhaseSettings:
    """Tests for get_default_phase_settings function."""

    def test_returns_empty_settings(self) -> None:
        """Returns PhaseSettings with no overrides."""
        settings = get_default_phase_settings("discuss")
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.seed is None

    def test_works_for_all_phases(self) -> None:
        """Works for all standard phases."""
        for phase in ["discuss", "summarize", "serialize"]:
            settings = get_default_phase_settings(phase)
            assert isinstance(settings, PhaseSettings)

    def test_works_for_unknown_phase(self) -> None:
        """Works for unknown phases (returns defaults)."""
        settings = get_default_phase_settings("unknown")
        assert isinstance(settings, PhaseSettings)


class TestDetectModelVariant:
    """Tests for _detect_model_variant with GPT-5 and reasoning models."""

    def test_gpt5_mini_supports_verbosity(self) -> None:
        """GPT-5-mini supports verbosity."""
        variant = _detect_model_variant("openai", "gpt-5-mini")
        assert variant.supports_verbosity is True

    def test_gpt5_mini_supports_reasoning_effort(self) -> None:
        """GPT-5-mini supports reasoning_effort."""
        variant = _detect_model_variant("openai", "gpt-5-mini")
        assert variant.supports_reasoning_effort is True

    def test_gpt5_mini_rejects_stop(self) -> None:
        """GPT-5 family rejects stop sequences."""
        variant = _detect_model_variant("openai", "gpt-5-mini")
        assert variant.rejects_stop is True

    def test_gpt5_does_not_reject_temperature(self) -> None:
        """GPT-5 family allows temperature (unlike o1/o3)."""
        variant = _detect_model_variant("openai", "gpt-5-mini")
        assert variant.rejects_temperature is False

    def test_o1_no_verbosity(self) -> None:
        """o1 models do NOT support verbosity."""
        variant = _detect_model_variant("openai", "o1")
        assert variant.supports_verbosity is False

    def test_o1_supports_reasoning_effort(self) -> None:
        """o1 models support reasoning_effort."""
        variant = _detect_model_variant("openai", "o1")
        assert variant.supports_reasoning_effort is True

    def test_gpt4o_no_special_features(self) -> None:
        """GPT-4o has no special reasoning/verbosity support."""
        variant = _detect_model_variant("openai", "gpt-4o")
        assert variant.supports_verbosity is False
        assert variant.supports_reasoning_effort is False
        assert variant.rejects_temperature is False

    def test_non_openai_provider_default(self) -> None:
        """Non-OpenAI providers return default variant."""
        variant = _detect_model_variant("anthropic", "claude-3-opus")
        assert variant == ModelVariant()


class TestFilterModelKwargs:
    """Tests for filter_model_kwargs with reasoning_effort and verbosity."""

    def test_passes_reasoning_effort_for_gpt5(self) -> None:
        """reasoning_effort passes through for GPT-5-mini."""
        result = filter_model_kwargs("openai", "gpt-5-mini", {"reasoning_effort": "high"})
        assert result["reasoning_effort"] == "high"

    def test_filters_reasoning_effort_for_gpt4o(self) -> None:
        """reasoning_effort is filtered out for GPT-4o (not supported)."""
        result = filter_model_kwargs("openai", "gpt-4o", {"reasoning_effort": "high"})
        assert "reasoning_effort" not in result

    def test_filters_reasoning_effort_for_ollama(self) -> None:
        """reasoning_effort is filtered out for Ollama models."""
        result = filter_model_kwargs("ollama", "qwen3:4b", {"reasoning_effort": "medium"})
        assert "reasoning_effort" not in result

    def test_passes_verbosity_for_gpt5(self) -> None:
        """verbosity in model_kwargs passes through for GPT-5-mini."""
        result = filter_model_kwargs("openai", "gpt-5-mini", {"model_kwargs": {"verbosity": "low"}})
        assert result["model_kwargs"] == {"verbosity": "low"}

    def test_filters_verbosity_for_gpt4o(self) -> None:
        """verbosity in model_kwargs is stripped for GPT-4o."""
        result = filter_model_kwargs("openai", "gpt-4o", {"model_kwargs": {"verbosity": "low"}})
        # model_kwargs should be stripped entirely since verbosity was the only key
        assert "model_kwargs" not in result

    def test_filters_verbosity_keeps_other_model_kwargs(self) -> None:
        """Strips only verbosity from model_kwargs, keeps other keys."""
        result = filter_model_kwargs(
            "openai",
            "gpt-4o",
            {"model_kwargs": {"verbosity": "low", "other_param": "value"}},
        )
        assert result["model_kwargs"] == {"other_param": "value"}

    def test_passes_reasoning_effort_for_o1(self) -> None:
        """reasoning_effort passes through for o1 models."""
        result = filter_model_kwargs("openai", "o1", {"reasoning_effort": "high"})
        assert result["reasoning_effort"] == "high"

    def test_filters_verbosity_for_o1(self) -> None:
        """verbosity in model_kwargs is stripped for o1 (only GPT-5 supports it)."""
        result = filter_model_kwargs("openai", "o1", {"model_kwargs": {"verbosity": "low"}})
        assert "model_kwargs" not in result
