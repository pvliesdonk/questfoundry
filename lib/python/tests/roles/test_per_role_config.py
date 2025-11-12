"""Tests for per-role provider configuration."""

from unittest.mock import Mock

import pytest

from questfoundry.providers.base import TextProvider
from questfoundry.providers.config import ProviderConfig
from questfoundry.providers.registry import ProviderRegistry
from questfoundry.roles.base import Role, RoleContext, RoleResult
from questfoundry.roles.showrunner import Showrunner


class MockTextProvider(TextProvider):
    """Mock text provider for testing."""

    def validate_config(self) -> None:
        """Validate configuration."""
        pass

    def generate_text(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate text (mock implementation)."""
        return f"Response to: {prompt}"

    def generate_text_streaming(self, prompt: str, model: str | None = None, **kwargs):
        """Generate text streaming (mock implementation)."""
        yield "chunk1"
        yield "chunk2"


class MockRole(Role):
    """Mock role for testing."""

    @property
    def role_name(self) -> str:
        """Role identifier."""
        return "mock_role"

    @property
    def display_name(self) -> str:
        """Human-readable role name."""
        return "Mock Role"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """Execute a mock task."""
        return RoleResult(success=True, output="Mock execution")


class TestProviderConfigRoleSupport:
    """Tests for ProviderConfig role-specific methods."""

    def test_get_role_config_not_configured(self) -> None:
        """get_role_config returns empty dict for unconfigured role."""
        config = ProviderConfig()
        role_config = config.get_role_config("nonexistent_role")
        assert role_config == {}

    def test_get_role_config_with_config(self, tmp_path) -> None:
        """get_role_config returns configuration when present."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o

roles:
  plotwright:
    text_provider: ollama
    cache:
      ttl_seconds: 3600
    rate_limit:
      requests_per_minute: 30
""")

        config = ProviderConfig(config_file)
        role_config = config.get_role_config("plotwright")

        assert role_config["text_provider"] == "ollama"
        assert role_config["cache"]["ttl_seconds"] == 3600
        assert role_config["rate_limit"]["requests_per_minute"] == 30

    def test_get_role_provider_default_fallback(self, tmp_path) -> None:
        """get_role_provider falls back to default if role not configured."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
""")

        config = ProviderConfig(config_file)
        provider = config.get_role_provider("any_role", "text")
        assert provider == "openai"

    def test_get_role_provider_role_specific(self, tmp_path) -> None:
        """get_role_provider returns role-specific provider when configured."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3

roles:
  plotwright:
    text_provider: ollama
""")

        config = ProviderConfig(config_file)
        provider = config.get_role_provider("plotwright", "text")
        assert provider == "ollama"

    def test_get_role_provider_config_basic(self, tmp_path) -> None:
        """get_role_provider_config returns provider config for role."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3

roles:
  plotwright:
    text_provider: ollama
""")

        config = ProviderConfig(config_file)
        provider_config = config.get_role_provider_config("plotwright", "text")

        assert provider_config["base_url"] == "http://localhost:11434"
        assert provider_config["model"] == "llama3"

    def test_get_role_provider_config_with_overrides(self, tmp_path) -> None:
        """get_role_provider_config merges provider config with role overrides."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
      cache_enabled: false

roles:
  plotwright:
    text_provider: openai
    cache:
      ttl_seconds: 3600
    rate_limit:
      requests_per_minute: 30
""")

        config = ProviderConfig(config_file)
        provider_config = config.get_role_provider_config("plotwright", "text")

        # Base provider config should be included
        assert provider_config["api_key"] == "test_key"
        assert provider_config["model"] == "gpt-4o"

        # Role overrides should be included
        assert provider_config["cache"]["ttl_seconds"] == 3600
        assert provider_config["rate_limit"]["requests_per_minute"] == 30

    def test_get_role_provider_config_not_found(self, tmp_path) -> None:
        """get_role_provider_config returns empty dict if provider not found."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai

roles:
  plotwright:
    text_provider: nonexistent
""")

        config = ProviderConfig(config_file)
        provider_config = config.get_role_provider_config("plotwright", "text")
        assert provider_config == {}

    def test_list_roles_empty(self) -> None:
        """list_roles returns empty list when no roles configured."""
        config = ProviderConfig()
        roles = config.list_roles()
        assert roles == []

    def test_list_roles_with_config(self, tmp_path) -> None:
        """list_roles returns configured role names."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai

roles:
  plotwright:
    text_provider: ollama
  illustrator:
    text_provider: openai
  lore_weaver:
    cache:
      ttl_seconds: 7200
""")

        config = ProviderConfig(config_file)
        roles = config.list_roles()
        assert set(roles) == {"plotwright", "illustrator", "lore_weaver"}


class TestRoleWithRoleConfig:
    """Tests for Role class with role_config parameter."""

    def test_role_init_with_role_config(self) -> None:
        """Role accepts and stores role_config parameter."""
        provider = MockTextProvider({})
        role_config = {
            "text_provider": "ollama",
            "cache": {"ttl_seconds": 3600},
        }

        role = MockRole(provider=provider, role_config=role_config)

        assert role.role_config == role_config
        assert role.config == {}

    def test_role_init_without_role_config(self) -> None:
        """Role defaults role_config to empty dict if not provided."""
        provider = MockTextProvider({})
        role = MockRole(provider=provider)
        assert role.role_config == {}

    def test_role_init_with_both_configs(self) -> None:
        """Role can have both config and role_config."""
        provider = MockTextProvider({})
        task_config = {"max_tokens": 2000, "temperature": 0.7}
        role_config = {
            "text_provider": "ollama",
            "cache": {"ttl_seconds": 3600},
        }

        role = MockRole(
            provider=provider,
            config=task_config,
            role_config=role_config,
        )

        assert role.config == task_config
        assert role.role_config == role_config

    def test_role_config_independent_from_task_config(self) -> None:
        """Role config and task config are independent."""
        provider = MockTextProvider({})
        role_config = {"text_provider": "ollama"}
        task_config = {"max_tokens": 2000}

        role = MockRole(
            provider=provider,
            config=task_config,
            role_config=role_config,
        )

        # Modifying one shouldn't affect the other
        role.config["custom"] = "value"
        assert "custom" not in role.role_config


class TestShowrunnerRoleInitialization:
    """Tests for Showrunner role initialization methods."""

    def test_get_provider_for_role_default(self, tmp_path) -> None:
        """get_provider_for_role uses default when role not configured."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        # Mock the registry to return our mock provider
        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        result = showrunner.get_provider_for_role(registry, "text")
        assert result is provider

    def test_get_provider_for_role_role_specific(self, tmp_path) -> None:
        """get_provider_for_role uses role-specific provider when configured."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3

roles:
  showrunner:
    text_provider: ollama
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        result = showrunner.get_provider_for_role(registry, "text")
        assert result is provider
        # Verify ollama was requested
        registry.get_text_provider.assert_called_once_with("ollama")

    def test_initialize_role_with_config_basic(self, tmp_path) -> None:
        """initialize_role_with_config creates role with provider config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        role = showrunner.initialize_role_with_config(
            MockRole,
            registry,
            config={"max_tokens": 2000},
        )

        assert isinstance(role, MockRole)
        assert role.provider is provider
        assert role.config == {"max_tokens": 2000}

    def test_initialize_role_with_config_role_specific(self, tmp_path) -> None:
        """initialize_role_with_config uses role-specific configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3

roles:
  mock_role:
    text_provider: ollama
    cache:
      ttl_seconds: 3600
    rate_limit:
      requests_per_minute: 30
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        role = showrunner.initialize_role_with_config(MockRole, registry)

        assert isinstance(role, MockRole)
        assert role.role_config["text_provider"] == "ollama"
        assert role.role_config["cache"]["ttl_seconds"] == 3600
        assert role.role_config["rate_limit"]["requests_per_minute"] == 30

    def test_initialize_role_with_config_session_and_callback(self, tmp_path) -> None:
        """initialize_role_with_config passes session and callback."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        mock_session = Mock()
        mock_callback = Mock()

        role = showrunner.initialize_role_with_config(
            MockRole,
            registry,
            session=mock_session,
            human_callback=mock_callback,
        )

        assert role.session is mock_session
        assert role.human_callback is mock_callback

    def test_initialize_role_with_config_spec_path(self, tmp_path) -> None:
        """initialize_role_with_config passes spec_path."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        registry.get_text_provider.return_value = provider

        spec_path = tmp_path / "spec"

        role = showrunner.initialize_role_with_config(
            MockRole,
            registry,
            spec_path=spec_path,
        )

        assert role.spec_path == spec_path

    def test_initialize_role_with_config_no_default_provider(self, tmp_path) -> None:
        """initialize_role_with_config raises error if no default provider."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text: {}
""")

        config = ProviderConfig(config_file)
        provider = MockTextProvider({})
        showrunner = Showrunner(provider=provider)

        registry = Mock(spec=ProviderRegistry)
        registry.config = config
        error_msg = "No default text provider configured"
        registry.get_text_provider.side_effect = ValueError(error_msg)

        with pytest.raises(ValueError, match="Failed to initialize"):
            showrunner.initialize_role_with_config(MockRole, registry)


class TestPerRoleConfigIntegration:
    """Integration tests for per-role configuration."""

    def test_role_cache_from_role_config(self, tmp_path) -> None:
        """Role can access cache settings from role_config."""
        provider = MockTextProvider({})
        role_config = {
            "cache": {
                "ttl_seconds": 7200,
                "backend": "file",
            }
        }

        role = MockRole(provider=provider, role_config=role_config)

        # Role should be able to access cache settings
        assert role.role_config["cache"]["ttl_seconds"] == 7200
        assert role.role_config["cache"]["backend"] == "file"

    def test_role_rate_limit_from_role_config(self, tmp_path) -> None:
        """Role can access rate limiting settings from role_config."""
        provider = MockTextProvider({})
        role_config = {
            "rate_limit": {
                "requests_per_minute": 30,
                "tokens_per_hour": 10000,
            }
        }

        role = MockRole(provider=provider, role_config=role_config)

        # Role should be able to access rate limiting settings
        assert role.role_config["rate_limit"]["requests_per_minute"] == 30
        assert role.role_config["rate_limit"]["tokens_per_hour"] == 10000

    def test_provider_config_multiple_roles_independent(self, tmp_path) -> None:
        """Different roles can have independent configurations."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
    ollama:
      base_url: http://localhost:11434
      model: llama3

roles:
  plotwright:
    text_provider: ollama
    cache:
      ttl_seconds: 3600
  illustrator:
    text_provider: openai
    cache:
      ttl_seconds: 86400
""")

        config = ProviderConfig(config_file)

        plotwright_config = config.get_role_config("plotwright")
        illustrator_config = config.get_role_config("illustrator")

        assert plotwright_config["text_provider"] == "ollama"
        assert illustrator_config["text_provider"] == "openai"
        assert plotwright_config["cache"]["ttl_seconds"] == 3600
        assert illustrator_config["cache"]["ttl_seconds"] == 86400

    def test_role_inherits_base_provider_config(self, tmp_path) -> None:
        """Role-specific config can inherit from base provider config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
providers:
  text:
    default: openai
    openai:
      api_key: test_key
      model: gpt-4o
      temperature: 0.7

roles:
  plotwright:
    text_provider: openai
    cache:
      ttl_seconds: 3600
""")

        config = ProviderConfig(config_file)
        provider_config = config.get_role_provider_config("plotwright", "text")

        # Should have base provider config
        assert provider_config["api_key"] == "test_key"
        assert provider_config["model"] == "gpt-4o"

        # Should also have role overrides
        assert provider_config["cache"]["ttl_seconds"] == 3600
