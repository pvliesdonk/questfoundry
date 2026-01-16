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
        data = {"default": "ollama/qwen3:8b"}
        config = ProvidersConfig.from_dict(data)

        assert config.default == "ollama/qwen3:8b"
        assert config.discuss is None
        assert config.summarize is None
        assert config.serialize is None

    def test_from_dict_with_phase_overrides(self) -> None:
        """Parse config with phase-specific overrides."""
        data = {
            "default": "ollama/qwen3:8b",
            "discuss": "ollama/qwen3:8b",
            "summarize": "openai/gpt-4o",
            "serialize": "openai/o1-mini",
        }
        config = ProvidersConfig.from_dict(data)

        assert config.default == "ollama/qwen3:8b"
        assert config.discuss == "ollama/qwen3:8b"
        assert config.summarize == "openai/gpt-4o"
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
            default="ollama/qwen3:8b",
            discuss="openai/gpt-4o",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_discuss_provider() == "openai/gpt-4o"

    def test_get_discuss_provider_fallback_to_default(self) -> None:
        """get_discuss_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:8b")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_discuss_provider() == "ollama/qwen3:8b"

    def test_get_discuss_provider_env_override(self) -> None:
        """Environment variable overrides config value."""
        config = ProvidersConfig(
            default="ollama/qwen3:8b",
            discuss="openai/gpt-4o",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_DISCUSS": "anthropic/claude-3"}):
            assert config.get_discuss_provider() == "anthropic/claude-3"

    def test_get_summarize_provider_from_config(self) -> None:
        """get_summarize_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:8b",
            summarize="openai/gpt-4o",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_summarize_provider() == "openai/gpt-4o"

    def test_get_summarize_provider_fallback_to_default(self) -> None:
        """get_summarize_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:8b")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_summarize_provider() == "ollama/qwen3:8b"

    def test_get_summarize_provider_env_override(self) -> None:
        """Environment variable overrides config value."""
        config = ProvidersConfig(
            default="ollama/qwen3:8b",
            summarize="openai/gpt-4o",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_SUMMARIZE": "anthropic/claude-3"}):
            assert config.get_summarize_provider() == "anthropic/claude-3"

    def test_get_serialize_provider_from_config(self) -> None:
        """get_serialize_provider returns config value when set."""
        config = ProvidersConfig(
            default="ollama/qwen3:8b",
            serialize="openai/o1-mini",
        )

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_serialize_provider() == "openai/o1-mini"

    def test_get_serialize_provider_fallback_to_default(self) -> None:
        """get_serialize_provider falls back to default when not set."""
        config = ProvidersConfig(default="ollama/qwen3:8b")

        with patch.dict("os.environ", {}, clear=True):
            assert config.get_serialize_provider() == "ollama/qwen3:8b"

    def test_get_serialize_provider_env_override(self) -> None:
        """Environment variable overrides config value."""
        config = ProvidersConfig(
            default="ollama/qwen3:8b",
            serialize="openai/o1-mini",
        )

        with patch.dict("os.environ", {"QF_PROVIDER_SERIALIZE": "openai/o3-mini"}):
            assert config.get_serialize_provider() == "openai/o3-mini"


# --- Tests for ProjectConfig with hybrid providers ---


class TestProjectConfigHybridProviders:
    """Tests for ProjectConfig with hybrid provider support."""

    def test_from_dict_backward_compatible(self) -> None:
        """Parsing old format with only default provider still works."""
        data = {
            "name": "test-project",
            "providers": {
                "default": "ollama/qwen3:8b",
            },
        }
        config = ProjectConfig.from_dict(data)

        # Legacy field populated
        assert config.provider.name == "ollama"
        assert config.provider.model == "qwen3:8b"

        # New field also populated
        assert config.providers.default == "ollama/qwen3:8b"
        assert config.providers.discuss is None
        assert config.providers.serialize is None

    def test_from_dict_with_hybrid_providers(self) -> None:
        """Parsing new format with phase-specific providers."""
        data = {
            "name": "hybrid-project",
            "providers": {
                "default": "ollama/qwen3:8b",
                "discuss": "ollama/qwen3:8b",
                "summarize": "openai/gpt-4o",
                "serialize": "openai/o1-mini",
            },
        }
        config = ProjectConfig.from_dict(data)

        # Legacy field uses default
        assert config.provider.name == "ollama"
        assert config.provider.model == "qwen3:8b"

        # New field has all values
        assert config.providers.default == "ollama/qwen3:8b"
        assert config.providers.discuss == "ollama/qwen3:8b"
        assert config.providers.summarize == "openai/gpt-4o"
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

        # Should use provider-specific default (gpt-4o for openai), not DEFAULT_MODEL
        assert config.provider.name == "openai"
        assert config.provider.model == "gpt-4o"  # OpenAI's default model

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
        config = create_default_config("test-project", provider="openai/gpt-4o")

        assert config.name == "test-project"
        assert config.provider.name == "openai"
        assert config.provider.model == "gpt-4o"
        assert config.providers.default == "openai/gpt-4o"

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
