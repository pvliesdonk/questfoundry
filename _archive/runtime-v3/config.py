"""Centralized configuration for QuestFoundry Runtime.

Configuration is loaded from multiple sources in priority order (highest first):
1. CLI arguments (via typer options)
2. Environment variables (QF_* prefix)
3. .env file (if present in current directory or project root)
4. Config file (questfoundry.yaml, questfoundry.toml, or ~/.config/questfoundry/config.yaml)
5. Built-in defaults

Environment Variables
---------------------
All settings can be overridden via environment variables with QF_ prefix::

    # LLM settings
    export QF_LLM_PROVIDER="google"
    export QF_LLM_MODEL="gemini-2.5-pro"
    export QF_LLM_TEMPERATURE=0.7
    export GOOGLE_API_KEY="your-api-key"

    # Runtime settings
    export QF_RUNTIME_MAX_DELEGATIONS=50
    export QF_RUNTIME_MAX_ITERATIONS=10

    # Ollama settings (for local inference)
    export QF_OLLAMA_HOST="http://localhost:11434"
    export QF_OLLAMA_MODEL="qwen3:8b"

    # SearXNG settings (optional, for Lorekeeper web search)
    export QF_SEARXNG__URL="http://localhost:8080"

Config File
-----------
Create `questfoundry.yaml` in your project root::

    llm:
      provider: google
      model: gemini-2.5-pro
      temperature: 0.7

    runtime:
      max_delegations: 50
      debug: false

    paths:
      project_dir: ~/.questfoundry/projects
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Categories
# =============================================================================


class RuntimeConfig(BaseModel):
    """Execution limits and safety thresholds."""

    max_delegations: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum SR delegations before forced termination",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum tool iterations per role execution",
    )
    max_failures: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum consecutive tool call failures before giving up",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose output",
    )


class LLMConfig(BaseModel):
    """LLM provider and model configuration.

    Supports multiple providers:
    - google: Google AI Studio (Gemini)
    - ollama: Local Ollama inference
    - openai: OpenAI API
    """

    provider: Literal["google", "ollama", "openai"] = Field(
        default="ollama",
        description="LLM provider: google, ollama, or openai",
    )
    model: str | None = Field(
        default=None,
        description="Model name (provider-specific). If None, uses provider default.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0.0 = deterministic, 2.0 = creative)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens for generation",
    )


def _get_ollama_host_default() -> str:
    """Get Ollama host from environment, with fallback to localhost."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""

    host: str = Field(
        default_factory=_get_ollama_host_default,
        description="Ollama server URL",
    )

    model: str = Field(
        default="qwen3:8b",
        description="Default Ollama model",
    )
    num_ctx: int = Field(
        default=32768,
        ge=1024,
        le=131072,
        description="Context window size",
    )


class GoogleConfig(BaseModel):
    """Google AI Studio configuration.

    Requires GOOGLE_API_KEY environment variable to be set.
    """

    model: str = Field(
        default="gemini-2.5-pro-preview-05-06",
        description="Gemini model name",
    )
    thinking_budget: int | None = Field(
        default=None,
        description="Thinking budget in tokens (None=default, 0=disabled, -1=dynamic)",
    )


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    model: str = Field(
        default="gpt-4o",
        description="OpenAI model name",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL (for Azure or proxies)",
    )


class SearXNGConfig(BaseModel):
    """SearXNG web search configuration.

    SearXNG is a self-hosted metasearch engine. The Lorekeeper role
    can use it for research during world-building and lore creation.

    This is optional - if not configured, Lorekeeper will operate
    without web search capabilities.
    """

    url: str | None = Field(
        default=None,
        description="SearXNG instance URL (e.g., 'http://localhost:8080'). "
        "If not set, web search is disabled.",
    )
    timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Request timeout in seconds",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return",
    )

    @property
    def enabled(self) -> bool:
        """Check if SearXNG is configured and enabled."""
        return bool(self.url)


class PathsConfig(BaseModel):
    """File system paths and locations."""

    project_dir: str = Field(
        default="~/.questfoundry/projects",
        description="Base directory for project storage",
    )
    project_id: str = Field(
        default="default",
        description="Default project identifier",
    )
    config_file: str | None = Field(
        default=None,
        description="Path to config file (auto-detected if not specified)",
    )

    @field_validator("project_dir")
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand ~ and environment variables in paths."""
        return str(Path(v).expanduser())


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="WARNING",
        description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    show_time: bool = Field(
        default=False,
        description="Show timestamp in console output",
    )
    show_path: bool = Field(
        default=False,
        description="Show file path and line number",
    )
    structured_logs_dir: str | None = Field(
        default=None,
        description="Directory for structured JSONL logs",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Log level must be one of {valid}, got {v}")
        return v.upper()


# =============================================================================
# Main Settings Class
# =============================================================================


class QuestFoundrySettings(BaseSettings):
    """Main configuration for QuestFoundry Runtime.

    Configuration is loaded from multiple sources in priority order:
    1. CLI arguments (passed directly to methods)
    2. Environment variables (QF_* prefix)
    3. .env file (if present)
    4. Config file (questfoundry.yaml/toml)
    5. Built-in defaults

    Examples
    --------
    Get settings::

        from questfoundry.runtime.config import get_settings

        settings = get_settings()
        print(settings.llm.provider)
        print(settings.runtime.max_delegations)

    Override with environment::

        export QF_LLM_PROVIDER=google
        export QF_LLM_MODEL=gemini-2.5-pro
    """

    model_config = SettingsConfigDict(
        env_prefix="QF_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Configuration categories
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    searxng: SearXNGConfig = Field(default_factory=SearXNGConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def __init__(self, **data: Any) -> None:
        """Initialize settings, loading from config file if available."""
        # Try to load config file first
        config_from_file = _load_config_file(data.get("paths", {}).get("config_file"))

        # Merge: file config < env vars < explicit data
        merged = _deep_merge(config_from_file, data)
        super().__init__(**merged)

    def get_llm_model(self) -> str:
        """Get the effective model name for the configured provider."""
        if self.llm.model:
            return self.llm.model

        # Return provider-specific default
        if self.llm.provider == "google":
            return self.google.model
        elif self.llm.provider == "ollama":
            return self.ollama.model
        elif self.llm.provider == "openai":
            return self.openai.model
        else:
            return "qwen3:8b"  # Fallback


# =============================================================================
# Config File Loading
# =============================================================================


def _find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Searches in order:
    1. ./questfoundry.yaml
    2. ./questfoundry.toml
    3. ./.questfoundry.yaml
    4. ~/.config/questfoundry/config.yaml
    5. ~/.config/questfoundry/config.toml
    """
    locations = [
        Path.cwd() / "questfoundry.yaml",
        Path.cwd() / "questfoundry.toml",
        Path.cwd() / ".questfoundry.yaml",
        Path.home() / ".config" / "questfoundry" / "config.yaml",
        Path.home() / ".config" / "questfoundry" / "config.toml",
    ]

    for path in locations:
        if path.exists():
            logger.debug(f"Found config file: {path}")
            return path

    return None


def _load_config_file(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML or TOML file.

    Parameters
    ----------
    config_path : str | None
        Explicit path to config file, or None to auto-detect.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary, or empty dict if no file found.
    """
    path = Path(config_path).expanduser() if config_path else _find_config_file()

    if not path or not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            import yaml

            config = yaml.safe_load(content) or {}
            logger.info(f"Loaded config from {path}")
            return dict(config)

        elif path.suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]

            config = tomllib.loads(content)
            logger.info(f"Loaded config from {path}")
            return dict(config)

        else:
            logger.warning(f"Unknown config file format: {path.suffix}")
            return {}

    except Exception as e:
        logger.error(f"Failed to load config file {path}: {e}")
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# Singleton Access
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> QuestFoundrySettings:
    """Get the global settings instance (cached).

    This function returns a cached singleton instance of the settings.
    The first call loads and validates all configuration.
    Subsequent calls return the cached instance.

    To reload settings (e.g., after environment changes), use reload_settings().

    Returns
    -------
    QuestFoundrySettings
        Cached settings instance.

    Examples
    --------
    ::

        settings = get_settings()
        print(settings.llm.provider)  # 'ollama'
        print(settings.runtime.max_delegations)  # 50
    """
    return QuestFoundrySettings()


def reload_settings() -> QuestFoundrySettings:
    """Reload settings, clearing the cache.

    Call this if you need to reload configuration after changing
    environment variables or config files.

    Returns
    -------
    QuestFoundrySettings
        Fresh settings instance.
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# Environment Variable Helpers
# =============================================================================


def get_env_str(name: str, default: str | None = None) -> str | None:
    """Get string from environment variable.

    Parameters
    ----------
    name : str
        Environment variable name (without QF_ prefix).
    default : str | None
        Default value if not set.

    Returns
    -------
    str | None
        Value from environment or default.
    """
    return os.getenv(f"QF_{name}", default)


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable.

    Truthy values: "true", "1", "yes", "on"
    Falsy values: "false", "0", "no", "off", ""

    Parameters
    ----------
    name : str
        Environment variable name (without QF_ prefix).
    default : bool
        Default value if not set.

    Returns
    -------
    bool
        Boolean value from environment or default.
    """
    env_val = os.getenv(f"QF_{name}")
    if env_val is None:
        return default
    return env_val.lower() in ("true", "1", "yes", "on")
