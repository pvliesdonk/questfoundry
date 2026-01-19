"""Tests for phase-specific model settings."""

from __future__ import annotations

import pytest

from questfoundry.providers.settings import (
    PHASE_CREATIVITY,
    TEMPERATURE_MAP,
    CreativityLevel,
    PhaseSettings,
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

    def test_from_dict_empty(self) -> None:
        """from_dict with empty dict returns defaults."""
        settings = PhaseSettings.from_dict({})
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.seed is None

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
