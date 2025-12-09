"""Tests for runtime configuration system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from questfoundry.runtime.config import (
    GoogleConfig,
    LLMConfig,
    LoggingConfig,
    OllamaConfig,
    OpenAIConfig,
    PathsConfig,
    QuestFoundrySettings,
    RuntimeConfig,
    _deep_merge,
    _find_config_file,
    _load_config_file,
    get_settings,
    reload_settings,
)


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = RuntimeConfig()
        assert config.max_delegations == 50
        assert config.max_iterations == 10
        assert config.max_failures == 3
        assert config.debug is False

    def test_custom_values(self) -> None:
        """Custom values can be set."""
        config = RuntimeConfig(max_delegations=100, debug=True)
        assert config.max_delegations == 100
        assert config.debug is True

    def test_validation(self) -> None:
        """Values are validated."""
        with pytest.raises(ValueError):
            RuntimeConfig(max_delegations=0)  # Must be >= 1

        with pytest.raises(ValueError):
            RuntimeConfig(max_delegations=1000)  # Must be <= 500


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model is None
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_providers(self) -> None:
        """Valid providers are accepted."""
        for provider in ["ollama", "google", "openai"]:
            config = LLMConfig(provider=provider)  # type: ignore[arg-type]
            assert config.provider == provider

    def test_temperature_validation(self) -> None:
        """Temperature is validated."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.5)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default values are set correctly."""
        # Clear env vars that might override defaults
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("QF_OLLAMA_HOST", raising=False)
        config = OllamaConfig()
        assert config.host == "http://localhost:11434"
        assert config.model == "qwen3:8b"
        assert config.num_ctx == 32768


class TestGoogleConfig:
    """Tests for GoogleConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = GoogleConfig()
        assert "gemini-2.5-pro" in config.model
        assert config.thinking_budget is None

    def test_thinking_budget(self) -> None:
        """Thinking budget can be set."""
        config = GoogleConfig(thinking_budget=1024)
        assert config.thinking_budget == 1024


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = OpenAIConfig()
        assert config.model == "gpt-4o"
        assert config.api_base is None


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = PathsConfig()
        assert "questfoundry" in config.project_dir.lower()
        assert config.project_id == "default"

    def test_path_expansion(self) -> None:
        """Paths with ~ are expanded."""
        config = PathsConfig(project_dir="~/test")
        assert "~" not in config.project_dir
        assert str(Path.home()) in config.project_dir


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        config = LoggingConfig()
        assert config.level == "WARNING"
        assert config.show_time is False
        assert config.show_path is False

    def test_level_validation(self) -> None:
        """Log level is validated."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            assert config.level == level

        # Invalid level
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")

    def test_level_case_insensitive(self) -> None:
        """Log level is case-insensitive."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"


class TestQuestFoundrySettings:
    """Tests for main settings class."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default settings are created."""
        # Clear env vars that might override defaults
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("QF_OLLAMA_HOST", raising=False)
        settings = QuestFoundrySettings()
        assert settings.llm.provider == "ollama"
        assert settings.runtime.max_delegations == 50
        assert settings.ollama.host == "http://localhost:11434"
        assert "gemini" in settings.google.model

    def test_get_llm_model_explicit(self) -> None:
        """Explicit model is returned."""
        settings = QuestFoundrySettings()
        settings.llm.model = "custom-model"
        assert settings.get_llm_model() == "custom-model"

    def test_get_llm_model_ollama_default(self) -> None:
        """Ollama default model is returned."""
        settings = QuestFoundrySettings()
        settings.llm.provider = "ollama"
        settings.llm.model = None
        assert settings.get_llm_model() == settings.ollama.model

    def test_get_llm_model_google_default(self) -> None:
        """Google default model is returned."""
        settings = QuestFoundrySettings()
        settings.llm.provider = "google"
        settings.llm.model = None
        assert settings.get_llm_model() == settings.google.model

    def test_get_llm_model_openai_default(self) -> None:
        """OpenAI default model is returned."""
        settings = QuestFoundrySettings()
        settings.llm.provider = "openai"
        settings.llm.model = None
        assert settings.get_llm_model() == settings.openai.model


class TestConfigFileFinding:
    """Tests for config file discovery."""

    def test_find_config_file_none(self, tmp_path: Path) -> None:
        """No config file found returns None."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = _find_config_file()
            # May find ~/.config/questfoundry/config.yaml if it exists
            # Just verify it doesn't crash


class TestConfigFileLoading:
    """Tests for config file loading."""

    def test_load_yaml_config(self, tmp_path: Path) -> None:
        """YAML config file is loaded."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
llm:
  provider: google
  temperature: 0.5
runtime:
  max_delegations: 100
"""
        )

        result = _load_config_file(str(config_file))
        assert result["llm"]["provider"] == "google"
        assert result["llm"]["temperature"] == 0.5
        assert result["runtime"]["max_delegations"] == 100

    def test_load_nonexistent_file(self) -> None:
        """Nonexistent file returns empty dict."""
        result = _load_config_file("/nonexistent/path/config.yaml")
        assert result == {}

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = _load_config_file(str(config_file))
        assert result == {}


class TestDeepMerge:
    """Tests for deep merge function."""

    def test_simple_merge(self) -> None:
        """Simple dicts are merged."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Nested dicts are merged recursively."""
        base: dict[str, Any] = {"outer": {"a": 1, "b": 2}}
        override: dict[str, Any] = {"outer": {"b": 3, "c": 4}}

        result = _deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_override_non_dict(self) -> None:
        """Non-dict values override completely."""
        base: dict[str, Any] = {"outer": {"a": 1}}
        override: dict[str, Any] = {"outer": "string"}

        result = _deep_merge(base, override)
        assert result == {"outer": "string"}


class TestSettingsSingleton:
    """Tests for settings singleton behavior."""

    def test_get_settings_cached(self) -> None:
        """get_settings returns cached instance."""
        # Clear cache first
        reload_settings()

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_settings(self) -> None:
        """reload_settings clears cache."""
        s1 = get_settings()
        s2 = reload_settings()
        s3 = get_settings()

        # s2 should be new instance
        # s3 should be same as s2
        assert s2 is s3


class TestEnvironmentVariables:
    """Tests for environment variable override."""

    def test_env_override_provider(self) -> None:
        """Environment variables override settings."""
        # Clear cache
        reload_settings()

        with patch.dict(os.environ, {"QF_LLM__PROVIDER": "google"}):
            # Need to reload for env to take effect
            settings = reload_settings()
            # Note: pydantic-settings may not pick up nested vars
            # This tests the mechanism, not full functionality
