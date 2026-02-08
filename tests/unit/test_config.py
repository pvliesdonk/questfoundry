"""Tests for pipeline configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from questfoundry.pipeline.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    FillConfig,
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
        assert config.creative is None
        assert config.balanced is None
        assert config.structured is None

    def test_from_dict_with_role_overrides(self) -> None:
        """Parse config with role-specific overrides."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "creative": "ollama/qwen3:4b-instruct-32k",
            "balanced": "openai/gpt-5-mini",
            "structured": "openai/o1-mini",
        }
        config = ProvidersConfig.from_dict(data)

        assert config.default == "ollama/qwen3:4b-instruct-32k"
        assert config.creative == "ollama/qwen3:4b-instruct-32k"
        assert config.balanced == "openai/gpt-5-mini"
        assert config.structured == "openai/o1-mini"

    def test_from_dict_with_legacy_phase_names(self) -> None:
        """Parse config with legacy phase names (backwards compatibility)."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "discuss": "ollama/qwen3:4b-instruct-32k",
            "summarize": "openai/gpt-5-mini",
            "serialize": "openai/o1-mini",
        }
        config = ProvidersConfig.from_dict(data)

        # Legacy names map to role names
        assert config.creative == "ollama/qwen3:4b-instruct-32k"
        assert config.balanced == "openai/gpt-5-mini"
        assert config.structured == "openai/o1-mini"
        # Legacy aliases still work
        assert config.discuss == "ollama/qwen3:4b-instruct-32k"
        assert config.summarize == "openai/gpt-5-mini"
        assert config.serialize == "openai/o1-mini"

    def test_from_dict_role_names_override_legacy(self) -> None:
        """Role names take precedence when both are specified."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "discuss": "openai/old",
            "creative": "openai/new",
        }
        config = ProvidersConfig.from_dict(data)

        assert config.creative == "openai/new"

    def test_from_dict_empty_uses_defaults(self) -> None:
        """Empty dict uses system defaults."""
        config = ProvidersConfig.from_dict({})

        assert config.default == f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"
        assert config.creative is None
        assert config.balanced is None
        assert config.structured is None

    def test_get_creative_provider_from_config(self) -> None:
        """get_creative_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            creative="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_creative_provider() == "openai/gpt-5-mini"

    def test_get_creative_provider_fallback_to_default(self) -> None:
        """get_creative_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_creative_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_creative_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            creative="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_CREATIVE": "anthropic/claude-3"}):
            assert config.get_creative_provider() == "openai/gpt-5-mini"

    def test_get_balanced_provider_from_config(self) -> None:
        """get_balanced_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            balanced="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_balanced_provider() == "openai/gpt-5-mini"

    def test_get_balanced_provider_fallback_to_default(self) -> None:
        """get_balanced_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_balanced_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_balanced_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            balanced="openai/gpt-5-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_BALANCED": "anthropic/claude-3"}):
            assert config.get_balanced_provider() == "openai/gpt-5-mini"

    def test_get_structured_provider_from_config(self) -> None:
        """get_structured_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            structured="openai/o1-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_structured_provider() == "openai/o1-mini"

    def test_get_structured_provider_fallback_to_default(self) -> None:
        """get_structured_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_structured_provider() == "ollama/qwen3:4b-instruct-32k"

    def test_get_structured_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            structured="openai/o1-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_STRUCTURED": "openai/o3-mini"}):
            assert config.get_structured_provider() == "openai/o1-mini"

    def test_legacy_aliases_work(self) -> None:
        """Legacy property aliases (discuss, summarize, serialize) return role values."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            creative="openai/creative",
            balanced="openai/balanced",
            structured="openai/structured",
        )

        assert config.discuss == "openai/creative"
        assert config.summarize == "openai/balanced"
        assert config.serialize == "openai/structured"

    def test_legacy_getter_aliases_work(self) -> None:
        """Legacy getter methods delegate to role-based methods."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            creative="openai/creative",
            balanced="openai/balanced",
            structured="openai/structured",
        )

        assert config.get_discuss_provider() == "openai/creative"
        assert config.get_summarize_provider() == "openai/balanced"
        assert config.get_serialize_provider() == "openai/structured"

    def test_from_dict_with_image_provider(self) -> None:
        """Parse config with image provider set."""
        data = {
            "default": "ollama/qwen3:4b-instruct-32k",
            "image": "openai/gpt-image-1",
        }
        config = ProvidersConfig.from_dict(data)

        assert config.image == "openai/gpt-image-1"
        assert config.get_image_provider() == "openai/gpt-image-1"

    def test_from_dict_without_image_provider(self) -> None:
        """Image provider defaults to None (opt-in)."""
        config = ProvidersConfig.from_dict({"default": "ollama/qwen3:4b-instruct-32k"})

        assert config.image is None
        assert config.get_image_provider() is None

    def test_get_image_provider_ignores_env(self) -> None:
        """ProvidersConfig returns config value, not env var (SRP: orchestrator handles env)."""
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            image="openai/gpt-image-1",
        )

        with patch.dict("os.environ", {"QF_IMAGE_PROVIDER": "placeholder"}):
            assert config.get_image_provider() == "openai/gpt-image-1"


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
        assert config.providers.creative is None
        assert config.providers.structured is None

    def test_from_dict_with_legacy_phase_names(self) -> None:
        """Parsing config with legacy phase names still works."""
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

        # Mapped to role names
        assert config.providers.creative == "ollama/qwen3:4b-instruct-32k"
        assert config.providers.balanced == "openai/gpt-5-mini"
        assert config.providers.structured == "openai/o1-mini"
        # Legacy aliases still work
        assert config.providers.discuss == "ollama/qwen3:4b-instruct-32k"
        assert config.providers.summarize == "openai/gpt-5-mini"
        assert config.providers.serialize == "openai/o1-mini"

    def test_from_dict_with_role_names(self) -> None:
        """Parsing config with role-based names."""
        data = {
            "name": "role-project",
            "providers": {
                "default": "ollama/qwen3:4b-instruct-32k",
                "creative": "openai/gpt-5-mini",
                "balanced": "openai/gpt-5-mini",
                "structured": "openai/o3-mini",
            },
        }
        config = ProjectConfig.from_dict(data)

        assert config.providers.creative == "openai/gpt-5-mini"
        assert config.providers.balanced == "openai/gpt-5-mini"
        assert config.providers.structured == "openai/o3-mini"

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

    def test_get_role_settings_returns_configured(self) -> None:
        """get_role_settings returns configured settings."""
        from questfoundry.providers.settings import PhaseSettings

        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            settings={"discuss": PhaseSettings(temperature=0.95)},
        )

        settings = config.get_role_settings("discuss")
        assert settings.temperature == 0.95

    def test_get_role_settings_returns_defaults_when_not_configured(self) -> None:
        """get_role_settings returns defaults when phase not in settings."""
        config = ProvidersConfig(default="ollama/qwen3:4b-instruct-32k")

        settings = config.get_role_settings("discuss")
        # Default settings have no explicit temperature (uses phase/provider default)
        assert settings.temperature is None

    def test_get_role_settings_merges_with_defaults(self) -> None:
        """get_role_settings merges configured with defaults."""
        from questfoundry.providers.settings import PhaseSettings

        # Configure only temperature, leave top_p as default
        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            settings={"discuss": PhaseSettings(temperature=0.95)},
        )

        settings = config.get_role_settings("discuss")
        assert settings.temperature == 0.95
        assert settings.top_p is None  # Not configured, uses default

    def test_get_role_settings_resolves_role_aliases(self) -> None:
        """get_role_settings resolves legacy phase names to role names in settings."""
        from questfoundry.providers.settings import PhaseSettings

        config = ProvidersConfig(
            default="ollama/qwen3:4b-instruct-32k",
            settings={"creative": PhaseSettings(temperature=0.95)},
        )

        # Looking up "discuss" should find "creative" settings via alias
        settings = config.get_role_settings("discuss")
        assert settings.temperature == 0.95


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


# --- Tests for user config loading ---


class TestUserConfig:
    """Tests for global user config loading."""

    def test_load_user_config_missing_file(self, tmp_path: Path) -> None:
        """Returns None when config file doesn't exist."""

        from questfoundry.pipeline.user_config import load_user_config

        result = load_user_config(config_dir=tmp_path / "nonexistent")
        assert result is None

    def test_load_user_config_valid(self, tmp_path: Path) -> None:
        """Loads valid user config."""

        from ruamel.yaml import YAML

        from questfoundry.pipeline.user_config import load_user_config

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        yaml = YAML()
        yaml.dump(
            {
                "providers": {
                    "default": "openai/gpt-5-mini",
                    "creative": "openai/gpt-5-mini",
                    "structured": "openai/o3-mini",
                }
            },
            config_path,
        )

        result = load_user_config(config_dir=config_dir)
        assert result is not None
        assert result.default == "openai/gpt-5-mini"
        assert result.creative == "openai/gpt-5-mini"
        assert result.structured == "openai/o3-mini"

    def test_load_user_config_empty_file(self, tmp_path: Path) -> None:
        """Returns None for empty config file."""

        from questfoundry.pipeline.user_config import load_user_config

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("")

        result = load_user_config(config_dir=config_dir)
        assert result is None

    def test_load_user_config_no_providers(self, tmp_path: Path) -> None:
        """Returns None when config has no providers section."""

        from ruamel.yaml import YAML

        from questfoundry.pipeline.user_config import load_user_config

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml = YAML()
        yaml.dump({"other_setting": "value"}, config_dir / "config.yaml")

        result = load_user_config(config_dir=config_dir)
        assert result is None


class TestProjectConfigLanguage:
    """Tests for ProjectConfig language field."""

    def test_default_language_is_english(self) -> None:
        data = {
            "name": "test",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
        }
        config = ProjectConfig.from_dict(data)
        assert config.language == "en"

    def test_language_from_dict(self) -> None:
        data = {
            "name": "test",
            "language": "nl",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
        }
        config = ProjectConfig.from_dict(data)
        assert config.language == "nl"

    def test_language_german(self) -> None:
        data = {
            "name": "test",
            "language": "de",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
        }
        config = ProjectConfig.from_dict(data)
        assert config.language == "de"


class TestFillConfig:
    """Tests for FillConfig parsing."""

    def test_default_two_step_is_true(self) -> None:
        """FillConfig defaults to two_step=True."""
        config = FillConfig()
        assert config.two_step is True

    def test_from_dict_two_step_false(self) -> None:
        """FillConfig.from_dict reads two_step."""
        config = FillConfig.from_dict({"two_step": False})
        assert config.two_step is False

    def test_from_dict_empty(self) -> None:
        """Empty dict uses defaults."""
        config = FillConfig.from_dict({})
        assert config.two_step is True

    def test_default_exemplar_strategy_is_auto(self) -> None:
        """FillConfig defaults to exemplar_strategy='auto'."""
        config = FillConfig()
        assert config.exemplar_strategy == "auto"

    def test_from_dict_exemplar_strategy(self) -> None:
        """FillConfig.from_dict reads exemplar_strategy."""
        config = FillConfig.from_dict({"exemplar_strategy": "corpus_only"})
        assert config.exemplar_strategy == "corpus_only"

    def test_from_dict_exemplar_strategy_full(self) -> None:
        """FillConfig.from_dict reads exemplar_strategy=full."""
        config = FillConfig.from_dict({"exemplar_strategy": "full"})
        assert config.exemplar_strategy == "full"

    def test_from_dict_empty_preserves_exemplar_default(self) -> None:
        """Empty dict keeps exemplar_strategy default."""
        config = FillConfig.from_dict({})
        assert config.exemplar_strategy == "auto"


class TestProjectConfigFill:
    """Tests for ProjectConfig fill section."""

    def test_default_fill_config(self) -> None:
        """ProjectConfig defaults to FillConfig defaults."""
        data = {
            "name": "test",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
        }
        config = ProjectConfig.from_dict(data)
        assert config.fill.two_step is True

    def test_fill_two_step_from_yaml(self) -> None:
        """ProjectConfig parses fill.two_step from dict."""
        data = {
            "name": "test",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
            "fill": {"two_step": False},
        }
        config = ProjectConfig.from_dict(data)
        assert config.fill.two_step is False

    def test_fill_exemplar_strategy_from_yaml(self) -> None:
        """ProjectConfig parses fill.exemplar_strategy from dict."""
        data = {
            "name": "test",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
            "fill": {"exemplar_strategy": "corpus_only"},
        }
        config = ProjectConfig.from_dict(data)
        assert config.fill.exemplar_strategy == "corpus_only"

    def test_fill_section_missing_uses_defaults(self) -> None:
        """Missing fill section uses FillConfig defaults."""
        data = {
            "name": "test",
            "providers": {"default": "ollama/qwen3:4b-instruct-32k"},
        }
        config = ProjectConfig.from_dict(data)
        assert isinstance(config.fill, FillConfig)
        assert config.fill.two_step is True
        assert config.fill.exemplar_strategy == "auto"
