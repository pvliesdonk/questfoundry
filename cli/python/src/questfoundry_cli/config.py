"""Configuration file management for QuestFoundry CLI."""

import os
from pathlib import Path
from typing import Any

import toml
from rich.console import Console

console = Console()

CONFIG_DIR = Path.home() / ".questfoundry"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def ensure_config_dir() -> None:
    """Ensure configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """
    Load configuration from config file.

    Returns:
        Configuration dictionary
    """
    ensure_config_dir()

    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return toml.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")
        return {}


def save_config(config: dict[str, Any]) -> None:
    """
    Save configuration to config file.

    Args:
        config: Configuration dictionary to save
    """
    ensure_config_dir()

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        console.print(f"[green]✓ Configuration saved to {CONFIG_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]Error: Failed to save config: {e}[/red]")
        raise


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.

    Args:
        key: Configuration key (supports dot notation for nested keys)
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    config = load_config()

    # Support dot notation for nested keys
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def set_config_value(key: str, value: Any) -> None:
    """
    Set a configuration value.

    Args:
        key: Configuration key (supports dot notation for nested keys)
        value: Value to set
    """
    config = load_config()

    # Support dot notation for nested keys
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value
    save_config(config)


def get_workspace_path() -> Path | None:
    """
    Get the configured workspace path.

    Returns:
        Workspace path or None if not configured
    """
    workspace_path = get_config_value("workspace_path")
    if workspace_path:
        return Path(workspace_path)
    return None


def get_env_path() -> Path:
    """
    Get the path to the .env file.

    Returns:
        Path to .env file (in workspace or current directory)
    """
    workspace_path = get_workspace_path()
    if workspace_path and workspace_path.exists():
        return workspace_path / ".env"
    return Path.cwd() / ".env"


def load_env_vars() -> dict[str, str]:
    """
    Load environment variables from .env file.

    Returns:
        Dictionary of environment variables
    """
    from dotenv import dotenv_values

    env_path = get_env_path()
    if env_path.exists():
        return dict(dotenv_values(env_path))
    return {}


def check_api_keys() -> dict[str, bool]:
    """
    Check which API keys are configured.

    Returns:
        Dictionary mapping provider names to whether their API key is set
    """
    env_vars = load_env_vars()

    # Also check actual environment variables
    keys_to_check = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]

    status = {}
    for key in keys_to_check:
        # Check .env file first, then actual environment
        value = env_vars.get(key) or os.environ.get(key)
        status[key] = bool(value and value.strip())

    return status
