"""Load QuestFoundry spec resources with automatic fallback.

This module provides functions to load spec resources (schemas, protocols,
definitions) with automatic fallback between three modes:

1. **Monorepo mode** (development): Read from ../../spec/ if available
2. **Bundled mode** (production): Use importlib.resources from package
3. **Download mode** (optional): Fetch from GitHub releases if enabled

The loader tries monorepo first, then bundled, then download (if enabled).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

try:
    from importlib.resources import files
except ImportError as e:
    # Python 3.11 should have this, but provide a clear error
    raise ImportError(
        "importlib.resources.files requires Python 3.9+. Please upgrade to Python 3.11 or later."
    ) from e

SpecLayer = Literal["schemas", "protocol", "definitions"]


class SpecLoadError(RuntimeError):
    """Raised when a spec resource cannot be loaded."""


def _get_monorepo_spec_root() -> Path | None:
    """Get the monorepo spec/ root if available."""
    # From lib/runtime/src/questfoundry/runtime/spec_loader.py
    # Navigate to monorepo root: ../../../../../../spec/
    runtime_module = Path(__file__).parent
    monorepo_root = runtime_module.parent.parent.parent.parent.parent
    spec_root = monorepo_root / "spec"

    # Validate this is actually the monorepo spec directory
    if spec_root.is_dir() and (spec_root / "05-definitions").is_dir():
        return spec_root

    return None


def _load_from_monorepo(layer: SpecLayer, resource_path: str) -> str:
    """Load a resource from the monorepo spec directory."""
    spec_root = _get_monorepo_spec_root()
    if spec_root is None:
        raise SpecLoadError("Monorepo spec/ directory not found")

    # Map layer names to spec directory names
    layer_map = {
        "schemas": "03-schemas",
        "protocol": "04-protocol",
        "definitions": "05-definitions",
    }

    layer_dir = spec_root / layer_map[layer]
    resource_file = layer_dir / resource_path

    if not resource_file.exists():
        raise SpecLoadError(f"Resource not found in monorepo: {resource_file}")

    return resource_file.read_text(encoding="utf-8")


def _load_from_bundle(layer: SpecLayer, resource_path: str) -> str:
    """Load a resource from bundled package resources."""
    try:
        # Access bundled resources: questfoundry.runtime.resources.{layer}/
        resource_package = files("questfoundry.runtime.resources") / layer
        resource_file = resource_package / resource_path

        # For Python 3.11+, read_text() works on Traversable
        return resource_file.read_text(encoding="utf-8")
    except (FileNotFoundError, AttributeError, TypeError) as exc:
        raise SpecLoadError(
            f"Resource not found in bundled package: {layer}/{resource_path}"
        ) from exc


def load_spec_resource(
    layer: SpecLayer,
    resource_path: str,
    *,
    allow_download: bool = False,
) -> str:
    """Load a spec resource with automatic fallback.

    Args:
        layer: Which spec layer to load from ("schemas", "protocol", "definitions")
        resource_path: Relative path within the layer (e.g., "role.schema.json")
        allow_download: If True, attempt to download from GitHub releases as last resort

    Returns:
        The resource content as a string

    Raises:
        SpecLoadError: If the resource cannot be loaded from any source

    Examples:
        >>> # Load a JSON schema
        >>> schema_json = load_spec_resource("schemas", "role.schema.json")
        >>> schema = json.loads(schema_json)

        >>> # Load a YAML definition
        >>> role_yaml = load_spec_resource("definitions", "roles/historian.yaml")
    """
    # Try monorepo first (development mode)
    try:
        return _load_from_monorepo(layer, resource_path)
    except SpecLoadError:
        pass  # Fall through to bundled mode

    # Try bundled resources (production mode)
    try:
        return _load_from_bundle(layer, resource_path)
    except SpecLoadError:
        pass  # Fall through to download mode if enabled

    # Try download mode if enabled
    if allow_download:
        # TODO: Implement download from GitHub releases
        # This would use questfoundry_compiler.spec_fetcher
        raise SpecLoadError(f"Download mode not yet implemented for {layer}/{resource_path}")

    # All modes failed
    raise SpecLoadError(
        f"Unable to load {layer}/{resource_path} from any source. "
        f"Ensure you're in the monorepo or have bundled resources installed."
    )


def load_json_schema(schema_name: str) -> dict[str, Any]:
    """Load a JSON schema from Layer 3 (03-schemas).

    Args:
        schema_name: Name of the schema file (e.g., "role.schema.json")

    Returns:
        Parsed JSON schema as a dictionary

    Examples:
        >>> role_schema = load_json_schema("role.schema.json")
        >>> assert "$schema" in role_schema
    """
    content = load_spec_resource("schemas", schema_name)
    return json.loads(content)


def load_protocol_flow(flow_name: str) -> str:
    """Load a protocol flow from Layer 4 (04-protocol).

    Args:
        flow_name: Name of the protocol flow file

    Returns:
        Protocol flow content as a string

    Examples:
        >>> flow = load_protocol_flow("01-initialization.md")
    """
    return load_spec_resource("protocol", flow_name)


def load_definition(definition_path: str) -> str:
    """Load a YAML definition from Layer 5 (05-definitions).

    Args:
        definition_path: Relative path within definitions (e.g., "roles/historian.yaml")

    Returns:
        Definition content as a string (typically YAML)

    Examples:
        >>> role_yaml = load_definition("roles/historian.yaml")
        >>> import yaml
        >>> role_data = yaml.safe_load(role_yaml)
    """
    return load_spec_resource("definitions", definition_path)


def is_monorepo_available() -> bool:
    """Check if running in monorepo mode.

    Returns:
        True if the monorepo spec/ directory is available, False otherwise

    Examples:
        >>> if is_monorepo_available():
        ...     print("Running in development mode")
        ... else:
        ...     print("Running with bundled resources")
    """
    return _get_monorepo_spec_root() is not None
