"""Configuration management for QuestFoundry providers"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ProviderConfig:
    """
    Manages provider configuration with environment variable substitution.

    Configuration files use YAML format and support ${ENV_VAR} syntax
    for environment variable substitution.

    Example config.yml:
        providers:
          text:
            default: openai
            openai:
              api_key: ${OPENAI_API_KEY}
              model: gpt-4o
            ollama:
              base_url: http://localhost:11434
              model: llama3
    """

    ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_path: Path | str | None = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config file. If None, looks for
                        .questfoundry/config.yml in current directory.
        """
        if config_path is None:
            config_path = Path.cwd() / ".questfoundry" / "config.yml"
            logger.debug("No config path specified, using default: %s", config_path)
        else:
            config_path = Path(config_path)

        logger.debug("Initializing ProviderConfig from path: %s", config_path)
        self.config_path = config_path
        self._config: dict[str, Any] = {}

        if config_path.exists():
            logger.trace("Config file exists, loading configuration")
            self.load()
            logger.info("Configuration loaded successfully from %s", config_path)
        else:
            logger.debug(
                "Config file not found at %s, using default configuration", config_path
            )
            self._config = self._get_default_config()
            logger.trace("Default configuration initialized")

    def load(self) -> None:
        """
        Load configuration from file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid YAML
        """
        logger.debug("Loading configuration from file: %s", self.config_path)

        if not self.config_path.exists():
            logger.error("Config file not found: %s", self.config_path)
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            logger.trace("Reading and parsing YAML configuration file")
            with open(self.config_path) as f:
                raw_config = yaml.safe_load(f)
                logger.trace("Substituting environment variables in configuration")
                self._config = self._substitute_env_vars(raw_config or {})
            logger.info("Configuration loaded and environment variables substituted")
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in config file: %s", str(e), exc_info=True)
            raise ValueError(f"Invalid YAML in config file: {e}") from e
        except Exception as e:
            logger.error("Error loading configuration: %s", str(e), exc_info=True)
            raise

    def save(self) -> None:
        """
        Save configuration to file.

        Creates parent directories if they don't exist.
        """
        logger.debug("Saving configuration to file: %s", self.config_path)
        logger.trace("Creating parent directories if needed")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.trace("Writing YAML configuration to file")
            with open(self.config_path, "w") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
            logger.info("Configuration saved successfully to %s", self.config_path)
        except Exception as e:
            logger.error("Error saving configuration: %s", str(e), exc_info=True)
            raise

    def get_provider_config(
        self, provider_type: str, provider_name: str
    ) -> dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_type: Type of provider ('text' or 'image')
            provider_name: Name of provider (e.g., 'openai', 'ollama')

        Returns:
            Provider configuration dictionary

        Raises:
            KeyError: If provider not found in configuration
        """
        providers = self._config.get("providers", {})
        provider_configs = providers.get(provider_type, {})

        if provider_name not in provider_configs:
            raise KeyError(
                f"{provider_type} provider '{provider_name}' not found in configuration"
            )

        # Filter out 'default' key
        config = provider_configs.get(provider_name, {})
        if isinstance(config, dict):
            return config
        return {}

    def get_default_provider(self, provider_type: str) -> str | None:
        """
        Get default provider name for a type.

        Args:
            provider_type: Type of provider ('text' or 'image')

        Returns:
            Default provider name, or None if not configured
        """
        providers = self._config.get("providers", {})
        provider_configs = providers.get(provider_type, {})
        default = provider_configs.get("default")
        return str(default) if default is not None else None

    def set_default_provider(self, provider_type: str, provider_name: str) -> None:
        """
        Set default provider for a type.

        Args:
            provider_type: Type of provider ('text' or 'image')
            provider_name: Name of provider
        """
        providers = self._config.setdefault("providers", {})
        provider_config = providers.setdefault(provider_type, {})
        provider_config["default"] = provider_name

    def list_providers(self, provider_type: str) -> list[str]:
        """
        List available providers of a given type.

        Args:
            provider_type: Type of provider ('text' or 'image')

        Returns:
            List of provider names
        """
        providers = self._config.get("providers", {})
        provider_configs = providers.get(provider_type, {})

        # Exclude 'default' key from list
        return [name for name in provider_configs.keys() if name != "default"]

    def get_role_config(self, role_name: str) -> dict[str, Any]:
        """
        Get configuration for a specific role.

        Roles can have per-role provider selection, caching config,
        rate limiting config, and other role-specific settings.

        Args:
            role_name: Name of the role (e.g., 'plotwright', 'illustrator')

        Returns:
            Role configuration dictionary (empty dict if not configured)

        Example:
            config = provider_config.get_role_config("plotwright")
            # Returns: {
            #   "text_provider": "ollama",
            #   "cache": {"ttl_seconds": 3600},
            #   "rate_limit": {"requests_per_minute": 30}
            # }
        """
        roles = self._config.get("roles", {})
        return roles.get(role_name, {})

    def get_role_provider(self, role_name: str, provider_type: str) -> str | None:
        """
        Get the provider to use for a specific role and provider type.

        First checks role-specific configuration, then falls back to default.

        Args:
            role_name: Name of the role
            provider_type: Type of provider ('text' or 'image')

        Returns:
            Provider name to use for this role, or None if not configured

        Example:
            provider = config.get_role_provider("plotwright", "text")
            # Returns: "ollama" if configured, else falls back to default
        """
        # Check role-specific config first
        role_config = self.get_role_config(role_name)

        # Look for role-specific provider selection
        # Supports both "text_provider" and "image_provider" keys
        provider_key = f"{provider_type}_provider"
        if provider_key in role_config:
            return str(role_config[provider_key])

        # Fall back to default provider
        return self.get_default_provider(provider_type)

    def get_role_provider_config(
        self, role_name: str, provider_type: str
    ) -> dict[str, Any]:
        """
        Get the full provider configuration for a role.

        This combines the provider configuration with any role-specific
        overrides (cache settings, rate limiting, etc).

        Args:
            role_name: Name of the role
            provider_type: Type of provider ('text' or 'image')

        Returns:
            Combined provider configuration dictionary

        Example:
            config = provider_config.get_role_provider_config("plotwright", "text")
            # Returns: {
            #   "api_key": "...",
            #   "model": "llama3",
            #   "cache": {"ttl_seconds": 3600},
            #   "rate_limit": {"requests_per_minute": 30}
            # }
        """
        # Get the provider name for this role
        provider_name = self.get_role_provider(role_name, provider_type)
        if not provider_name:
            return {}

        # Get base provider configuration
        try:
            provider_config = self.get_provider_config(provider_type, provider_name)
        except KeyError:
            # Provider not found in configuration
            return {}

        # Get role-specific config
        role_config = self.get_role_config(role_name)

        # Deep merge configurations (role config takes precedence)
        merged_config = self._deep_merge(provider_config, role_config)

        return merged_config

    def _deep_merge(
        self, base: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Deep merge overrides into base dictionary.

        Nested dictionaries are merged rather than replaced, allowing
        role-specific overrides to only change specific keys while
        preserving base configuration values.

        Args:
            base: Base configuration dictionary
            overrides: Override configuration dictionary

        Returns:
            Merged configuration with overrides applied

        Example:
            base = {"cache": {"ttl": 3600}, "model": "gpt-4"}
            overrides = {"cache": {"enabled": True}}
            result = _deep_merge(base, overrides)
            # Returns: {"cache": {"ttl": 3600, "enabled": True}, "model": "gpt-4"}
        """
        result = {**base}

        for key, value in overrides.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Deep merge for nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Replace for non-dict values or new keys
                result[key] = value

        return result

    def list_roles(self) -> list[str]:
        """
        List all roles that have specific configuration.

        Returns:
            List of role names with custom configuration
        """
        roles = self._config.get("roles", {})
        return list(roles.keys())

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Replaces ${ENV_VAR} with os.environ['ENV_VAR'].

        Args:
            obj: Configuration object (dict, list, str, etc.)

        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_in_string(obj)
        return obj

    def _substitute_env_var_in_string(self, value: str) -> str:
        """
        Substitute environment variables in a string.

        Args:
            value: String potentially containing ${ENV_VAR} patterns

        Returns:
            String with variables substituted

        Raises:
            ValueError: If a referenced environment variable is not set
        """

        def replace_match(match: re.Match[str]) -> str:
            env_var = match.group(1)
            logger.trace("Substituting environment variable: %s", env_var)
            env_value = os.environ.get(env_var)
            if env_value is None:
                logger.error("Required environment variable not set: %s", env_var)
                raise ValueError(
                    f"Environment variable '{env_var}' is not set. "
                    f"Please set it before loading configuration."
                )
            logger.trace("Environment variable %s substituted successfully", env_var)
            return env_value

        return self.ENV_VAR_PATTERN.sub(replace_match, value)

    def _get_default_config(self) -> dict[str, Any]:
        """
        Get default configuration structure.

        Returns:
            Default configuration dictionary
        """
        return {
            "providers": {
                "text": {
                    "default": "openai",
                },
                "image": {
                    "default": "dalle",
                },
            }
        }
