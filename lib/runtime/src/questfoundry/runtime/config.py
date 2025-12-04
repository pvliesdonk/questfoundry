"""
Centralized configuration for QuestFoundry Runtime.

Configuration is loaded from multiple sources in priority order (highest first):
1. CLI arguments (via typer options)
2. Environment variables (QF_* prefix)
3. .env file (if present in current directory or project root)
4. Config file (questfoundry.yaml, questfoundry.toml, or ~/.config/questfoundry/config.yaml)
5. Built-in defaults

## Categories

- **runtime**: Execution limits, iteration counts, safety thresholds
- **llm**: Model selection, provider configuration, API settings
- **memory**: Prompt size limits, conversation memory caps
- **paths**: Project directories, spec locations, cache paths
- **logging**: Log levels, file paths, structured logging
- **network**: Timeouts, endpoints, external service URLs

## Environment Variables

All settings can be overridden via environment variables with QF_ prefix:

```bash
# Runtime settings
export QF_RECURSION_LIMIT=50
export QF_MAX_FAILURES=3
export QF_MAX_ITERATIONS=5

# LLM settings
export QF_DEFAULT_MODEL="claude-3-5-sonnet-20241022"
export QF_DEFAULT_TEMPERATURE=0.7
export QF_LLM_PROVIDER="anthropic"

# Memory settings
export QF_PROMPT_ERROR_THRESHOLD=32000
export QF_MEMORY_CAP=8000

# Path settings
export QF_PROJECT_DIR="~/.questfoundry/projects"
export QF_SPEC_SOURCE="auto"
export QF_SPEC_CACHE_DIR="~/.cache/questfoundry/spec"

# Logging settings
export QF_LOG_LEVEL="INFO"
export QF_DEBUG=false
```

## Config File

Create `questfoundry.yaml` in your project root:

```yaml
runtime:
  recursion_limit: 50
  max_failures: 3

llm:
  default_model: "claude-3-5-sonnet-20241022"
  default_temperature: 0.7

paths:
  project_dir: "~/.questfoundry/projects"
```
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Categories
# =============================================================================


class RuntimeConfig(BaseModel):
    """Execution limits and safety thresholds.

    These settings control how the runtime executes roles and prevents
    infinite loops or runaway executions.
    """

    recursion_limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum graph iterations before forced termination",
    )
    max_failures: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum consecutive tool call failures before giving up",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum tool iterations per role execution (bind_tools/react)",
    )
    max_parallel_roles: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum roles to execute in parallel",
    )
    max_ping_pong: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum identical message exchanges before intervention",
    )
    max_role_executions: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum times a single role can execute before forced termination",
    )
    execution_reset_threshold: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Total executions before halving per-role counts (prevents stuck sessions)",
    )
    max_consecutive_role_executions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum consecutive executions of the same role (fairness)",
    )
    max_validation_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum validation retry attempts before escalation",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose output",
    )


class LLMConfig(BaseModel):
    """LLM and model configuration.

    Settings for model selection, provider preferences, and generation parameters.
    """

    default_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default model to use when not specified by role",
    )
    default_model_tier: str = Field(
        default="creative-writing",
        description="Default model tier for role capability matching",
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for generation (0.0 = deterministic, 2.0 = creative)",
    )
    default_max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Default maximum tokens for generation",
    )
    provider: str | None = Field(
        default=None,
        description="Preferred LLM provider (anthropic, openai, google, ollama, litellm)",
    )
    model_tiers_config: str | None = Field(
        default=None,
        description="Path to custom model tiers YAML config file",
    )
    ollama_host: str | None = Field(
        default=None,
        description="Ollama server endpoint URL (e.g., http://localhost:11434)",
    )
    ollama_num_ctx: int = Field(
        default=32768,
        ge=1024,
        le=131072,
        description="Ollama context window size",
    )
    litellm_api_base: str | None = Field(
        default=None,
        description="LiteLLM proxy API base URL",
    )
    litellm_api_key: str | None = Field(
        default=None,
        description="LiteLLM API key (use 'sk-1234' for local proxy)",
    )
    bind_tools_denylist: list[str] = Field(
        default_factory=lambda: [
            "llama-3.2-1b",
            "llama-3.2-3b",
            "llama3.2:1b",
            "llama3.2:3b",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "phi3:mini",
            "gemma2:2b",
        ],
        description="Models that should use text-based tool calling instead of bind_tools",
    )


class MemoryConfig(BaseModel):
    """Prompt and memory size limits.

    These settings prevent context overflow and control how much conversation
    history is retained between role executions.
    """

    prompt_error_threshold: int = Field(
        default=32000,
        ge=1000,
        le=500000,
        description="Prompt size (chars) above which to log error (~8k tokens)",
    )
    prompt_warning_threshold: int = Field(
        default=16000,
        ge=1000,
        le=500000,
        description="Prompt size (chars) above which to log warning (~4k tokens)",
    )
    memory_cap: int = Field(
        default=8000,
        ge=1000,
        le=100000,
        description="Maximum characters for prior conversation history (~2k tokens)",
    )
    summarize_messages_threshold: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of messages after which to summarize conversation",
    )
    summarize_chars_threshold: int = Field(
        default=12000,
        ge=1000,
        le=100000,
        description="Character count after which to summarize conversation",
    )


class PathsConfig(BaseModel):
    """File system paths and locations.

    Settings for project storage, spec resolution, and cache directories.
    """

    project_dir: str = Field(
        default="~/.questfoundry/projects",
        description="Base directory for project storage",
    )
    project_id: str = Field(
        default="default",
        description="Default project identifier",
    )
    spec_source: str = Field(
        default="auto",
        description="Spec source preference: auto, monorepo, bundled, or download",
    )
    spec_cache_dir: str = Field(
        default="~/.cache/questfoundry/spec",
        description="Directory for downloaded spec cache",
    )
    config_file: str | None = Field(
        default=None,
        description="Path to config file (auto-detected if not specified)",
    )

    @field_validator("project_dir", "spec_cache_dir")
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand ~ and environment variables in paths."""
        return str(Path(v).expanduser())

    @field_validator("spec_source")
    @classmethod
    def validate_spec_source(cls, v: str) -> str:
        """Validate spec source is one of the allowed values."""
        valid = {"auto", "monorepo", "bundled", "download"}
        if v.lower() not in valid:
            raise ValueError(f"spec_source must be one of {valid}, got {v}")
        return v.lower()


class LoggingConfig(BaseModel):
    """Logging configuration.

    Settings for log levels, output destinations, and structured logging.
    """

    level: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    show_time: bool = Field(
        default=True,
        description="Show timestamp in console log output",
    )
    show_path: bool = Field(
        default=False,
        description="Show file path and line number in log output",
    )
    rich_tracebacks: bool = Field(
        default=True,
        description="Use Rich formatted tracebacks",
    )
    log_file: str | None = Field(
        default=None,
        description="Path to log file (creates <name>-debug.log and <name>-trace.log)",
    )
    structured_logs_dir: str | None = Field(
        default=None,
        description="Directory for structured JSONL logs",
    )
    httpx_level: str = Field(
        default="WARNING",
        description="Log level for httpx library",
    )
    openai_level: str = Field(
        default="WARNING",
        description="Log level for openai library",
    )
    anthropic_level: str = Field(
        default="WARNING",
        description="Log level for anthropic library",
    )
    reasoning_enabled: bool = Field(
        default=False,
        description="Enable extraction and logging of agent reasoning to qf.reasoning domain",
    )

    @field_validator("level", "httpx_level", "openai_level", "anthropic_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Log level must be one of {valid}, got {v}")
        return v.upper()


class NetworkConfig(BaseModel):
    """Network and external service configuration.

    Settings for timeouts, external API endpoints, and service URLs.
    """

    spec_fetch_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout (seconds) for spec fetch API calls",
    )
    spec_download_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Timeout (seconds) for spec archive download",
    )
    search_timeout: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="Timeout (seconds) for web search requests",
    )
    searxng_url: str | None = Field(
        default=None,
        description="SearXNG instance URL for web search",
    )
    searxng_api_token: str | None = Field(
        default=None,
        description="SearXNG API token (if required)",
    )
    elevenlabs_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout (seconds) for ElevenLabs API calls",
    )
    elevenlabs_default_voice: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",
        description="Default ElevenLabs voice ID",
    )
    elevenlabs_model: str = Field(
        default="eleven_turbo_v2",
        description="Default ElevenLabs model",
    )
    elevenlabs_stability: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="ElevenLabs voice stability (0.0-1.0)",
    )
    elevenlabs_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="ElevenLabs voice similarity boost (0.0-1.0)",
    )
    dalle_model: str = Field(
        default="gpt-image-1",
        description="DALL-E model for image generation",
    )
    dalle_size: str = Field(
        default="1024x1024",
        description="Default DALL-E image size",
    )
    gemini_image_model: str = Field(
        default="imagen-3.0-generate-002",
        description="Gemini/Imagen model for image generation",
    )


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

    Example usage:
        >>> from questfoundry.runtime.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.runtime.recursion_limit)
        50
        >>> print(settings.llm.default_model)
        'claude-3-5-sonnet-20241022'
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
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    def __init__(self, **data: Any) -> None:
        """Initialize settings, loading from config file if available."""
        # Try to load config file first
        config_from_file = _load_config_file(data.get("paths", {}).get("config_file"))

        # Merge: file config < env vars < explicit data
        merged = _deep_merge(config_from_file, data)
        super().__init__(**merged)


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

    Args:
        config_path: Explicit path to config file, or None to auto-detect

    Returns:
        Dictionary of configuration values, or empty dict if no file found
    """
    if config_path:
        path = Path(config_path).expanduser()
    else:
        path = _find_config_file()

    if not path or not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            import yaml

            config = yaml.safe_load(content) or {}
            logger.info(f"Loaded config from {path}")
            return config

        elif path.suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]

            config = tomllib.loads(content)
            logger.info(f"Loaded config from {path}")
            return config

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

    Returns:
        Cached QuestFoundrySettings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.runtime.max_failures)
        3
    """
    return QuestFoundrySettings()


def reload_settings() -> QuestFoundrySettings:
    """Reload settings, clearing the cache.

    Call this if you need to reload configuration after changing
    environment variables or config files.

    Returns:
        Fresh QuestFoundrySettings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# Environment Variable Helpers (for backward compatibility)
# =============================================================================


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable with fallback to settings.

    This is a compatibility helper for code that still uses os.getenv().
    Prefer accessing settings directly: get_settings().runtime.max_failures

    Args:
        name: Environment variable name (without QF_ prefix)
        default: Default value if not set

    Returns:
        Integer value from environment or settings
    """
    env_val = os.getenv(f"QF_{name}")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(f"Invalid integer value for QF_{name}: {env_val}")
    return default


def get_env_float(name: str, default: float) -> float:
    """Get float from environment variable with fallback.

    Args:
        name: Environment variable name (without QF_ prefix)
        default: Default value if not set

    Returns:
        Float value from environment or default
    """
    env_val = os.getenv(f"QF_{name}")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            logger.warning(f"Invalid float value for QF_{name}: {env_val}")
    return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable.

    Truthy values: "true", "1", "yes", "on"
    Falsy values: "false", "0", "no", "off", ""

    Args:
        name: Environment variable name (without QF_ prefix)
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    env_val = os.getenv(f"QF_{name}")
    if env_val is None:
        return default
    return env_val.lower() in ("true", "1", "yes", "on")


def get_env_str(name: str, default: str | None = None) -> str | None:
    """Get string from environment variable.

    Args:
        name: Environment variable name (without QF_ prefix)
        default: Default value if not set

    Returns:
        String value from environment or default
    """
    return os.getenv(f"QF_{name}", default)
