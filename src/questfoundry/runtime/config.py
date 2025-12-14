"""
Runtime configuration system.

Precedence (lowest to highest):
1. Built-in defaults
2. .env file
3. Environment variables
4. qf.yaml config file
5. CLI arguments

Configuration includes:
- Provider settings (ollama host, api keys)
- Model class mappings (creative → concrete model per provider)
- Per-agent overrides
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ProviderState(str, Enum):
    """Provider availability state."""

    UNCONFIGURED = "unconfigured"  # No credentials/config
    UNAVAILABLE = "unavailable"  # Configured but not reachable
    AVAILABLE = "available"  # Ready to use


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    state: ProviderState = ProviderState.UNCONFIGURED
    host: str | None = None  # For Ollama
    api_key: str | None = None  # For cloud providers
    default_model: str | None = None


@dataclass
class ModelClassMapping:
    """Maps abstract model classes to concrete models per provider."""

    # Abstract class name → {provider: concrete_model}
    mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, model_class: str, provider: str) -> str | None:
        """Resolve a model class to a concrete model for a provider."""
        if model_class in self.mappings:
            return self.mappings[model_class].get(provider)
        return None


@dataclass
class RuntimeConfig:
    """Complete runtime configuration."""

    # Domain path
    domain_path: Path = field(default_factory=lambda: Path("domain-v4"))

    # Provider configurations
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Model class mappings
    model_classes: ModelClassMapping = field(default_factory=ModelClassMapping)

    # Default provider to use
    default_provider: str = "ollama"

    # Per-agent overrides: agent_id → {setting: value}
    agent_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Logging
    log_events: bool = False
    log_path: Path | None = None

    # LangSmith
    langsmith_enabled: bool = False
    langsmith_project: str | None = None

    def get_provider(self, name: str | None = None) -> ProviderConfig | None:
        """Get provider config by name, or default provider."""
        name = name or self.default_provider
        return self.providers.get(name)

    def get_model_for_class(
        self, model_class: str, provider: str | None = None
    ) -> str | None:
        """Resolve model class to concrete model."""
        provider = provider or self.default_provider
        return self.model_classes.resolve(model_class, provider)


def _default_config() -> RuntimeConfig:
    """Create default configuration."""
    return RuntimeConfig(
        providers={
            "ollama": ProviderConfig(
                name="ollama",
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                default_model="qwen3:8b",
            ),
            "openai": ProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                default_model="gpt-4o",
            ),
            "google": ProviderConfig(
                name="google",
                api_key=os.getenv("GOOGLE_API_KEY"),
                default_model="gemini-1.5-pro",
            ),
        },
        model_classes=ModelClassMapping(
            mappings={
                "creative": {
                    "ollama": "qwen3:8b",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                },
                "creative-xl": {
                    "ollama": "qwen3:32b",
                    "openai": "gpt-4o",
                    "google": "gemini-1.5-pro",
                },
                "fast": {
                    "ollama": "qwen3:4b",
                    "openai": "gpt-4o-mini",
                    "google": "gemini-1.5-flash",
                },
            }
        ),
    )


def load_config(
    config_path: Path | None = None,
    env_file: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RuntimeConfig:
    """
    Load configuration with proper precedence.

    Args:
        config_path: Path to qf.yaml config file
        env_file: Path to .env file
        cli_overrides: CLI argument overrides

    Returns:
        Merged RuntimeConfig
    """
    # Load .env file
    if env_file and env_file.exists():
        load_dotenv(env_file)
    elif Path(".env").exists():
        load_dotenv(Path(".env"))

    # Start with defaults
    config = _default_config()

    # Load YAML config if provided
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        _merge_yaml_config(config, yaml_config)

    # Apply CLI overrides
    if cli_overrides:
        _apply_cli_overrides(config, cli_overrides)

    # Update provider states based on configuration
    _update_provider_states(config)

    return config


def _merge_yaml_config(config: RuntimeConfig, yaml_config: dict[str, Any]) -> None:
    """Merge YAML config into RuntimeConfig."""
    if "domain_path" in yaml_config:
        config.domain_path = Path(yaml_config["domain_path"])

    if "default_provider" in yaml_config:
        config.default_provider = yaml_config["default_provider"]

    if "providers" in yaml_config:
        for name, provider_cfg in yaml_config["providers"].items():
            if name in config.providers:
                if "host" in provider_cfg:
                    config.providers[name].host = provider_cfg["host"]
                if "api_key" in provider_cfg:
                    config.providers[name].api_key = provider_cfg["api_key"]
                if "default_model" in provider_cfg:
                    config.providers[name].default_model = provider_cfg["default_model"]

    if "model_classes" in yaml_config:
        for class_name, mappings in yaml_config["model_classes"].items():
            config.model_classes.mappings[class_name] = mappings

    if "agent_overrides" in yaml_config:
        config.agent_overrides = yaml_config["agent_overrides"]

    if "logging" in yaml_config:
        config.log_events = yaml_config["logging"].get("enabled", False)
        if "path" in yaml_config["logging"]:
            config.log_path = Path(yaml_config["logging"]["path"])

    if "langsmith" in yaml_config:
        config.langsmith_enabled = yaml_config["langsmith"].get("enabled", False)
        config.langsmith_project = yaml_config["langsmith"].get("project")


def _apply_cli_overrides(config: RuntimeConfig, overrides: dict[str, Any]) -> None:
    """Apply CLI overrides to config."""
    if "domain" in overrides:
        config.domain_path = Path(overrides["domain"])

    if "provider" in overrides:
        config.default_provider = overrides["provider"]

    if "log" in overrides:
        config.log_events = overrides["log"]


def _update_provider_states(config: RuntimeConfig) -> None:
    """Update provider states based on configuration."""
    for name, provider in config.providers.items():
        if name == "ollama":
            # Ollama just needs a host
            if provider.host:
                provider.state = ProviderState.AVAILABLE
            else:
                provider.state = ProviderState.UNCONFIGURED
        else:
            # Cloud providers need API keys
            if provider.api_key:
                provider.state = ProviderState.AVAILABLE
            else:
                provider.state = ProviderState.UNCONFIGURED
