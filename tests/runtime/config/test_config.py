"""
Tests for the runtime configuration system.

Tests cover:
- Default configuration
- Provider configuration
- Model class resolution
- YAML config loading
- Environment variable handling
- CLI override application
"""

import os
from pathlib import Path

from questfoundry.runtime.config import (
    ModelClassMapping,
    ProviderConfig,
    ProviderState,
    RuntimeConfig,
    load_config,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_provider_config_defaults(self):
        """ProviderConfig has sensible defaults."""
        config = ProviderConfig(name="test")

        assert config.name == "test"
        assert config.state == ProviderState.UNCONFIGURED
        assert config.host is None
        assert config.api_key is None
        assert config.default_model is None

    def test_provider_config_with_host(self):
        """ProviderConfig can be configured with host (for Ollama)."""
        config = ProviderConfig(
            name="ollama",
            host="http://localhost:11434",
            default_model="qwen3:8b",
        )

        assert config.host == "http://localhost:11434"
        assert config.default_model == "qwen3:8b"

    def test_provider_config_with_api_key(self):
        """ProviderConfig can be configured with API key (for cloud providers)."""
        config = ProviderConfig(
            name="openai",
            api_key="sk-test-key",
            default_model="gpt-4o",
        )

        assert config.api_key == "sk-test-key"
        assert config.default_model == "gpt-4o"


class TestModelClassMapping:
    """Tests for ModelClassMapping."""

    def test_model_class_mapping_resolve(self):
        """ModelClassMapping resolves model classes to concrete models."""
        mapping = ModelClassMapping(
            mappings={
                "creative": {
                    "ollama": "qwen3:8b",
                    "openai": "gpt-4o",
                },
                "fast": {
                    "ollama": "qwen3:4b",
                    "openai": "gpt-4o-mini",
                },
            }
        )

        assert mapping.resolve("creative", "ollama") == "qwen3:8b"
        assert mapping.resolve("creative", "openai") == "gpt-4o"
        assert mapping.resolve("fast", "ollama") == "qwen3:4b"

    def test_model_class_mapping_unknown_class(self):
        """ModelClassMapping returns None for unknown model class."""
        mapping = ModelClassMapping(mappings={})

        assert mapping.resolve("unknown", "ollama") is None

    def test_model_class_mapping_unknown_provider(self):
        """ModelClassMapping returns None for unknown provider."""
        mapping = ModelClassMapping(mappings={"creative": {"ollama": "qwen3:8b"}})

        assert mapping.resolve("creative", "unknown") is None


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_runtime_config_defaults(self):
        """RuntimeConfig has sensible defaults."""
        config = RuntimeConfig()

        assert config.domain_path == Path("domain-v4")
        assert config.default_provider == "ollama"
        assert config.log_events is False
        assert config.langsmith_enabled is False

    def test_get_provider(self):
        """RuntimeConfig.get_provider returns provider by name."""
        config = RuntimeConfig(
            providers={
                "ollama": ProviderConfig(name="ollama", host="http://localhost:11434"),
                "openai": ProviderConfig(name="openai", api_key="sk-test"),
            }
        )

        ollama = config.get_provider("ollama")
        assert ollama is not None
        assert ollama.name == "ollama"

        openai = config.get_provider("openai")
        assert openai is not None
        assert openai.name == "openai"

    def test_get_provider_default(self):
        """RuntimeConfig.get_provider returns default provider when name is None."""
        config = RuntimeConfig(
            default_provider="ollama",
            providers={
                "ollama": ProviderConfig(name="ollama", host="http://localhost:11434"),
            },
        )

        provider = config.get_provider(None)
        assert provider is not None
        assert provider.name == "ollama"

    def test_get_model_for_class(self):
        """RuntimeConfig.get_model_for_class resolves model class."""
        config = RuntimeConfig(
            default_provider="ollama",
            model_classes=ModelClassMapping(
                mappings={"creative": {"ollama": "qwen3:8b", "openai": "gpt-4o"}}
            ),
        )

        # Uses default provider
        assert config.get_model_for_class("creative") == "qwen3:8b"

        # Explicit provider
        assert config.get_model_for_class("creative", "openai") == "gpt-4o"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_defaults(self):
        """load_config returns default configuration."""
        config = load_config()

        assert config.default_provider == "ollama"
        assert "ollama" in config.providers
        assert "openai" in config.providers
        assert "google" in config.providers

    def test_load_config_ollama_defaults(self):
        """load_config sets up Ollama with defaults."""
        config = load_config()

        ollama = config.providers.get("ollama")
        assert ollama is not None
        assert ollama.default_model == "qwen3:8b"
        # Host comes from env or defaults to localhost
        assert ollama.host is not None

    def test_load_config_model_classes(self):
        """load_config sets up default model class mappings."""
        config = load_config()

        # Creative class should map to models for each provider
        assert config.get_model_for_class("creative", "ollama") == "qwen3:8b"
        assert config.get_model_for_class("creative", "openai") == "gpt-4o"

        # Fast class
        assert config.get_model_for_class("fast", "ollama") == "qwen3:4b"
        assert config.get_model_for_class("fast", "openai") == "gpt-4o-mini"

    def test_load_config_yaml(self, tmp_path: Path):
        """load_config merges YAML configuration."""
        yaml_content = """
domain_path: custom-domain
default_provider: openai
providers:
  ollama:
    host: http://custom:11434
model_classes:
  custom:
    ollama: custom-model
    openai: gpt-custom
"""
        config_file = tmp_path / "qf.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_path=config_file)

        assert config.domain_path == Path("custom-domain")
        assert config.default_provider == "openai"
        assert config.providers["ollama"].host == "http://custom:11434"
        assert config.get_model_for_class("custom", "ollama") == "custom-model"

    def test_load_config_cli_overrides(self):
        """load_config applies CLI overrides."""
        config = load_config(
            cli_overrides={
                "domain": "cli-domain",
                "provider": "openai",
                "log": True,
            }
        )

        assert config.domain_path == Path("cli-domain")
        assert config.default_provider == "openai"
        assert config.log_events is True

    def test_load_config_env_file(self, tmp_path: Path, monkeypatch):
        """load_config loads .env file."""
        env_content = "OLLAMA_HOST=http://env-host:11434\n"
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # Clear existing env var if set
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        _config = load_config(env_file=env_file)

        # The env file should set OLLAMA_HOST (load_config loads dotenv)
        assert os.getenv("OLLAMA_HOST") == "http://env-host:11434"

    def test_provider_states_updated(self):
        """load_config updates provider states based on configuration."""
        config = load_config()

        # Ollama should be available (has host)
        ollama = config.providers.get("ollama")
        assert ollama is not None
        assert ollama.state == ProviderState.AVAILABLE

        # OpenAI should be unconfigured (no API key by default)
        openai = config.providers.get("openai")
        assert openai is not None
        # State depends on whether OPENAI_API_KEY is set in environment
        if os.getenv("OPENAI_API_KEY"):
            assert openai.state == ProviderState.AVAILABLE
        else:
            assert openai.state == ProviderState.UNCONFIGURED
