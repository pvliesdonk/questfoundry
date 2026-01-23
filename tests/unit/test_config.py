"""Tests for pipeline configuration."""

from __future__ import annotations

from unittest.mock import patch

from questfoundry.pipeline.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ProjectConfig,
    ProvidersConfig,
    create_default_config,
)

# --- Tests for ProvidersConfig ---


class TestProvidersConfig:
    """Tests for ProvidersConfig class."""

    def test_from_dict_default_only(self) -> None:
        """Parse config with only default provider."""
        data = {"default": "ollama/qwen3:4b-instruct-32k"}
        config = ProvidersConfig.from_dict(data)

        assert config.default == "ollama/qwen3:4b-instruct-32k"
        assert config.discuss is None
        assert config.summarize is None
        assert config.serialize is None

    def test_from_dict_with_phase_overrides(self) -> None:
        """Parse config with phase-specific overrides."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "discuss": "ollama/qwen3:4b-instruct-32k",
            "summarize": "openai/gpt-5-mini",
            "serialize": "openai/o1-mini",
        }
        config = ProvidersConfig.from_dict(data)

        assert config.default == "ollama/qwen3:4b-instruct-32k"
        assert config.discuss == "ollama/qwen3:4b-instruct-32k"
        assert config.summarize == "openai/gpt-5-mini"
        assert config.serialize == "openai/o1-mini"

    def test_from_dict_empty_uses_defaults(self) -> None:
        """Empty dict uses system defaults."""
        config = ProvidersConfig.from_dict({})

        assert config.default == f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"
        assert config.discuss is None
        assert config.summarize is None
        assert config.serialize is None

    def test_get_discuss_provider_from_config(self) -> None:
        """get_discuss_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            discuss="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_discuss_provider() == "openai/gpt-5-mini"

    def test_get_discuss_provider_fallback_to_default(self) -> None:
        """get_discuss_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_discuss_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_discuss_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            discuss="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_DISCUSS": "anthropic/claude-3"}):
            assert config.get_discuss_provider() == "openai/gpt-5-mini"

    def test_get_summarize_provider_from_config(self) -> None:
        """get_summarize_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            summarize="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_summarize_provider() == "openai/gpt-5-mini"

    def test_get_summarize_provider_fallback_to_default(self) -> None:
        """get_summarize_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_summarize_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_summarize_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            summarize="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_SUMMARIZE": "anthropic/claude-3"}):
            assert config.get_summarize_provider() == "openai/gpt-5-mini"

    def test_get_serialize_provider_from_config(self) -> None:
        """get_serialize_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            serialize="openai/o1-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_serialize_provider() == "openai/o1-mini"

    def test_get_serialize_provider_fallback_to_default(self) -> None:
        """get_serialize_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_serialize_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_serialize_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            serialize="openai/o1-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_SERIALIZE": "openai/o3-mini"}):
            assert config.get_serialize_provider() == "openai/o1-mini"


# --- Tests for ProjectConfig with hybrid providers ---


class TestProjectConfigHybridProviders:
    """Tests for ProjectConfig with hybrid provider support."""

    def test_from_dict_backward_compatible(self) -> None:
        """Parsing old format with only default provider still works."""
        data = {
            "name": "test-project",
            "providers": {
                "default": "ollama/qwen3:4b-instruct-32k",
            },
        }
        config = ProjectConfig.from_dict(data)

        # Legacy field populated
        assert config.provider.name == "ollama"
        assert config.provider.model == "qwen3:4b-instruct-32k"

        # New field also populated
        assert config.providers.default == "ollama/qwen3:4b-instruct-32k"
        assert config.providers.discuss is None
        assert config.providers.serialize is None

    def test_from_dict_with_hybrid_providers(self) -> None:
        """Parsing new format with phase-specific providers."""
        data = {
            "name": "hybrid-project",
            "providers": {
                "default": "ollama/qwen3:4b-instruct-32k",
                "discuss": "ollama/qwen3:4b-instruct-32k",
                "summarize": "openai/gpt-5-mini",
                "serialize": "openai/o1-mini",
            },
        }
        config = ProjectConfig.from_dict(data)

        # Legacy field uses default
        assert config.provider.name == "ollama"
        assert config.provider.model == "qwen3:4b-instruct-32k"

        # New field has all values
        assert config.providers.default == "ollama/qwen3:4b-instruct-32k"
        assert config.providers.discuss == "ollama/qwen3:4b-instruct-32k"
        assert config.providers.summarize == "openai/gpt-5-mini"
        assert config.providers.serialize == "openai/o1-mini"

    def test_from_dict_provider_without_model(self) -> None:
        """Provider string without model uses provider-specific default model."""
        data = {
            "name": "test",
            "providers": {
                "default": "openai",  # No model specified
            },
        }
        config = ProjectConfig.from_dict(data)

        # Should use provider-specific default (gpt-5-mini for openai), not DEFAULT_MODEL
        assert config.provider.name == "openai"
        assert config.provider.model == "gpt-5-mini"  # OpenAI's default model

    def test_from_dict_unknown_provider_without_model_uses_default(self) -> None:
        """Unknown provider without model falls back to DEFAULT_MODEL."""
        data = {
            "name": "test",
            "providers": {
                "default": "custom-llm",  # Unknown provider, no model
            },
        }
        config = ProjectConfig.from_dict(data)

        # Unknown providers fall back to DEFAULT_MODEL
        assert config.provider.name == "custom-llm"
        assert config.provider.model == DEFAULT_MODEL


# --- Tests for create_default_config ---


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_create_default_config_no_provider(self) -> None:
        """Create config with system defaults."""
        config = create_default_config("test-project")

        assert config.name == "test-project"
        assert config.provider.name == DEFAULT_PROVIDER
        assert config.provider.model == DEFAULT_MODEL
        assert config.providers.default == f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"

    def test_create_default_config_with_provider(self) -> None:
        """Create config with explicit provider."""
        config = create_default_config("test-project", provider="openai/gpt-5-mini")

        assert config.name == "test-project"
        assert config.provider.name == "openai"
        assert config.provider.model == "gpt-5-mini"
        assert config.providers.default == "openai/gpt-5-mini"

    def test_create_default_config_provider_without_model(self) -> None:
        """Create config with provider but no model uses provider-specific default."""
        config = create_default_config("test-project", provider="anthropic")

        # Should use provider-specific default (claude-sonnet-4), not DEFAULT_MODEL
        assert config.provider.name == "anthropic"
        assert config.provider.model == "claude-sonnet-4-20250514"  # Anthropic's default
        assert config.providers.default == "anthropic"

    def test_create_default_config_unknown_provider_uses_default_model(self) -> None:
        """Unknown provider without model falls back to DEFAULT_MODEL."""
        config = create_default_config("test-project", provider="custom-provider")

        # Unknown providers without a model fall back to DEFAULT_MODEL
        assert config.provider.name == "custom-provider"
        assert config.provider.model == DEFAULT_MODEL
        assert config.providers.default == "custom-provider"


# --- Tests for ProvidersConfig phase settings ---


class TestProvidersConfigSettings:
    """Tests for ProvidersConfig settings functionality."""

    def test_from_dict_with_settings(self) -> None:
        """Parse config with phase-specific settings."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "settings": {
                "discuss": {"temperature": 0.95, "top_p": 0.9},
                "serialize": {"seed": 42},
            },
        }
        config = ProvidersConfig.from_dict(data)

        assert "discuss" in config.settings
        assert config.settings["discuss"].temperature == 0.95
        assert config.settings["discuss"].top_p == 0.9
        assert "serialize" in config.settings
        assert config.settings["serialize"].seed == 42

    def test_from_dict_without_settings(self) -> None:
        """Parse config without settings uses empty dict."""
        data = {"default": "ollama/qwen3:4b-instruct-32k"}
        config = ProvidersConfig.from_dict(data)

        assert config.settings == {}

    def test_get_phase_settings_returns_configured(self) -> None:
        """get_phase_settings returns configured settings."""
        from questfoundry.providers.settings import PhaseSettings

        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            settings={"discuss": PhaseSettings(temperature=0.95)},
        )

        settings = config.get_phase_settings("discuss")
        assert settings.temperature == 0.95

    def test_get_phase_settings_returns_defaults_when_not_configured(self) -> None:
        """get_phase_settings returns defaults when phase not in settings."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        settings = config.get_phase_settings("discuss")
        # Default settings have no explicit temperature (uses phase/provider default)
        assert settings.temperature is None

    def test_get_phase_settings_merges_with_defaults(self) -> None:
        """get_phase_settings merges configured with defaults."""
        from questfoundry.providers.settings import PhaseSettings

        # Configure only temperature, leave top_p as default
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            settings={"discuss": PhaseSettings(temperature=0.95)},
        )

        settings = config.get_phase_settings("discuss")
        assert settings.temperature == 0.95
        assert settings.top_p is None  # Not configured, uses default


class TestProjectConfigWithSettings:
    """Tests for ProjectConfig with phase settings support."""

    def test_from_dict_parses_settings(self) -> None:
        """ProjectConfig.from_dict parses nested settings."""
        data = {
            "name": "test-project",
            "providers": {
                "default": "ollama/qwen3:4b-instruct-32k",
                "settings": {
                    "discuss": {"temperature": 0.9},
                    "summarize": {"temperature": 0.5},
                    "serialize": {"temperature": 0.0, "seed": 42},
                },
            },
        }
        config = ProjectConfig.from_dict(data)

        assert config.providers.settings["discuss"].temperature == 0.9
        assert config.providers.settings["summarize"].temperature == 0.5
        assert config.providers.settings["serialize"].temperature == 0.0
        assert config.providers.settings["serialize"].seed == 42
