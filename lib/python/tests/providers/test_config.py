"""Tests for provider configuration"""

import tempfile
from pathlib import Path

import pytest

from questfoundry.providers.config import ProviderConfig


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Set required environment variables for tests by default"""
    # Set a dummy OPENAI_API_KEY for tests that don't care about the actual value
    # Individual tests can explicitly delete it to test missing env var behavior
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_file(temp_config_dir):
    """Create a sample config file"""
    config_path = temp_config_dir / ".questfoundry" / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_content = """
providers:
  text:
    default: openai
    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3
  image:
    default: dalle
    dalle:
      api_key: ${OPENAI_API_KEY}
      model: dall-e-3
"""
    config_path.write_text(config_content)
    return config_path


def test_config_loads_from_file(sample_config_file):
    """Test loading configuration from file"""
    config = ProviderConfig(sample_config_file)

    assert config.get_default_provider("text") == "openai"
    assert config.get_default_provider("image") == "dalle"


def test_config_env_var_substitution(sample_config_file, monkeypatch):
    """Test environment variable substitution"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-123")

    config = ProviderConfig(sample_config_file)
    openai_config = config.get_provider_config("text", "openai")

    assert openai_config["api_key"] == "test-api-key-123"


def test_config_missing_env_var(sample_config_file, monkeypatch):
    """Test missing environment variables raise ValueError"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Should raise ValueError when trying to load config with missing env var
    with pytest.raises(
        ValueError, match="Environment variable 'OPENAI_API_KEY' is not set"
    ):
        ProviderConfig(sample_config_file)


def test_config_get_provider_config(sample_config_file):
    """Test getting provider-specific configuration"""
    config = ProviderConfig(sample_config_file)

    ollama_config = config.get_provider_config("text", "ollama")
    assert ollama_config["base_url"] == "http://localhost:11434"
    assert ollama_config["model"] == "llama3"


def test_config_get_provider_config_not_found(sample_config_file):
    """Test getting non-existent provider raises KeyError"""
    config = ProviderConfig(sample_config_file)

    with pytest.raises(KeyError, match="provider 'nonexistent' not found"):
        config.get_provider_config("text", "nonexistent")


def test_config_list_providers(sample_config_file):
    """Test listing available providers"""
    config = ProviderConfig(sample_config_file)

    text_providers = config.list_providers("text")
    assert "openai" in text_providers
    assert "ollama" in text_providers
    assert "default" not in text_providers  # 'default' should be filtered out


def test_config_set_default_provider(temp_config_dir):
    """Test setting default provider"""
    config_path = temp_config_dir / "config.yml"
    config = ProviderConfig(config_path)

    config.set_default_provider("text", "ollama")
    assert config.get_default_provider("text") == "ollama"


def test_config_save(temp_config_dir):
    """Test saving configuration"""
    config_path = temp_config_dir / "config.yml"
    config = ProviderConfig(config_path)

    config.set_default_provider("text", "ollama")
    config.save()

    # Load again to verify persistence
    config2 = ProviderConfig(config_path)
    assert config2.get_default_provider("text") == "ollama"


def test_config_default_when_file_missing(temp_config_dir):
    """Test default configuration when file doesn't exist"""
    config_path = temp_config_dir / "nonexistent.yml"
    config = ProviderConfig(config_path)

    # Should have default configuration
    assert config.get_default_provider("text") == "openai"
    assert config.get_default_provider("image") == "dalle"


def test_config_invalid_yaml(temp_config_dir):
    """Test handling of invalid YAML"""
    config_path = temp_config_dir / "config.yml"
    config_path.write_text("invalid: yaml: content:")

    with pytest.raises(ValueError, match="Invalid YAML"):
        ProviderConfig(config_path)
