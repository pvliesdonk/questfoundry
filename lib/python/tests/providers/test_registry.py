"""Tests for provider registry"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from questfoundry.providers import ProviderConfig, ProviderRegistry, TextProvider


class MockTextProvider(TextProvider):
    """Mock text provider for testing"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.generate_called = False
        self.stream_called = False

    def validate_config(self) -> None:
        """Validate configuration"""
        if "required_key" in self.config and not self.config["required_key"]:
            raise ValueError("required_key is empty")

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate mock text"""
        self.generate_called = True
        return f"Mock response to: {prompt}"

    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate mock streaming text"""
        self.stream_called = True
        for word in ["Mock", "streaming", "response"]:
            yield word


@pytest.fixture
def temp_config():
    """Create temporary configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text("""
providers:
  text:
    default: mock
    mock:
      required_key: test_value
""")
        yield ProviderConfig(config_path)


def test_registry_register_text_provider(temp_config):
    """Test registering a text provider"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock", MockTextProvider)

    assert "mock" in registry.list_text_providers()


def test_registry_get_text_provider(temp_config):
    """Test getting a text provider instance"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock", MockTextProvider)

    provider = registry.get_text_provider("mock")
    assert isinstance(provider, MockTextProvider)
    assert provider.generate_called is False


def test_registry_get_default_text_provider(temp_config):
    """Test getting default text provider"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock", MockTextProvider)

    # Default is 'mock' according to config
    provider = registry.get_text_provider()
    assert isinstance(provider, MockTextProvider)


def test_registry_get_nonexistent_provider(temp_config):
    """Test getting non-existent provider raises ValueError"""
    registry = ProviderRegistry(temp_config)

    with pytest.raises(ValueError, match="not registered"):
        registry.get_text_provider("nonexistent")


def test_registry_caches_instances(temp_config):
    """Test that registry caches provider instances"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock", MockTextProvider)

    provider1 = registry.get_text_provider("mock")
    provider2 = registry.get_text_provider("mock")

    # Should be same instance
    assert provider1 is provider2


def test_registry_validates_config_on_get(temp_config):
    """Test that validate_config is called when getting provider"""
    registry = ProviderRegistry(temp_config)

    # Register provider that will fail validation
    class FailingProvider(MockTextProvider):
        def validate_config(self) -> None:
            raise ValueError("Config validation failed")

    registry.register_text_provider("failing", FailingProvider)

    # Update config to have failing provider
    temp_config._config["providers"]["text"]["failing"] = {"required_key": ""}

    with pytest.raises(ValueError, match="Config validation failed"):
        registry.get_text_provider("failing")


def test_registry_close_all(temp_config):
    """Test closing all provider instances"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock", MockTextProvider)

    # Get provider to create instance
    provider = registry.get_text_provider("mock")
    assert provider is not None

    # Close all
    registry.close_all()

    # Should create new instance on next get
    provider2 = registry.get_text_provider("mock")
    assert provider2 is not provider


def test_registry_list_providers(temp_config):
    """Test listing registered providers"""
    registry = ProviderRegistry(temp_config)
    registry.register_text_provider("mock1", MockTextProvider)
    registry.register_text_provider("mock2", MockTextProvider)

    providers = registry.list_text_providers()
    assert "mock1" in providers
    assert "mock2" in providers


def test_registry_no_default_configured():
    """Test error when no default provider configured"""
    # Create config with no default
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text("""
providers:
  text: {}
""")
        config = ProviderConfig(config_path)
        registry = ProviderRegistry(config)

        with pytest.raises(ValueError, match="No default .* provider configured"):
            registry.get_text_provider()
