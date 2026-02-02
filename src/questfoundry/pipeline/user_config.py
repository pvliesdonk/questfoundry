"""Global user configuration loading.

Reads user-level provider defaults from ~/.config/questfoundry/config.yaml.
This is the lowest-priority source in the provider resolution chain, below
CLI flags, environment variables, and project config.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.pipeline.config import ProvidersConfig

log = get_logger(__name__)

# XDG-compliant default config directory
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "questfoundry"


def load_user_config(config_dir: Path | None = None) -> ProvidersConfig | None:
    """Load global user provider configuration.

    Args:
        config_dir: Override config directory (for testing).
            Defaults to ~/.config/questfoundry/.

    Returns:
        ProvidersConfig from user config, or None if file doesn't exist.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    config_path = config_dir / "config.yaml"

    if not config_path.exists():
        return None

    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    from questfoundry.pipeline.config import ProvidersConfig

    yaml = YAML()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)
    except OSError as e:
        log.warning("user_config_load_failed", path=str(config_path), error=str(e))
        return None
    except YAMLError as e:
        log.warning("user_config_parse_failed", path=str(config_path), error=str(e))
        return None

    if data is None:
        return None

    providers_data = dict(data).get("providers", {})
    if not providers_data:
        return None

    # Let ValueError / ValidationError propagate — invalid user config
    # should surface clearly, not be silently swallowed.
    config = ProvidersConfig.from_dict(providers_data)
    log.debug("user_config_loaded", path=str(config_path))
    return config
