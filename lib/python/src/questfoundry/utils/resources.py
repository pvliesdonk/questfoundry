"""Resource loading utilities for schemas and prompts"""

import json
from pathlib import Path
from typing import Any

# Define the root of the spec relative to the project root.
# This assumes the code is run from the mono-repo root.
_SPEC_ROOT = Path("spec")
_SCHEMA_DIR = _SPEC_ROOT / "03-schemas"
_PROMPT_DIR = _SPEC_ROOT / "05-prompts"


def _validate_safe_path(base_dir: Path, target_path: Path, resource_type: str) -> None:
    """
    Validate that target path is within base directory (prevent path traversal).

    Args:
        base_dir: Base directory that should contain the target
        target_path: Path to validate
        resource_type: Type of resource for error message

    Raises:
        ValueError: If path traversal is detected
    """
    try:
        # resolve() follows symlinks and normalizes the path
        resolved_target = target_path.resolve()
        resolved_base = base_dir.resolve()

        # Check if target is relative to base
        resolved_target.relative_to(resolved_base)
    except (ValueError, RuntimeError):
        raise ValueError(
            f"Invalid {resource_type} path: path traversal detected. "
            f"Path must be within the spec directory."
        )


def get_schema(schema_name: str) -> dict[str, Any]:
    """
    Load a schema from the spec directory.

    Args:
        schema_name: Name of the schema (without .schema.json extension)

    Returns:
        Schema dictionary

    Raises:
        FileNotFoundError: If schema doesn't exist
        ValueError: If path traversal is detected
    """
    schema_file = _SCHEMA_DIR / f"{schema_name}.schema.json"

    # Validate path to prevent directory traversal
    _validate_safe_path(_SCHEMA_DIR, schema_file, "schema")

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_name}")

    with open(schema_file) as f:
        schema: dict[str, Any] = json.load(f)
        return schema


def get_prompt(role_name: str) -> str:
    """
    Load a prompt from the spec directory.

    Args:
        role_name: Name of the role

    Returns:
        Prompt text

    Raises:
        FileNotFoundError: If prompt doesn't exist
        ValueError: If path traversal is detected
    """
    prompt_file = _PROMPT_DIR / role_name / "system_prompt.md"

    # Validate path to prevent directory traversal
    _validate_safe_path(_PROMPT_DIR, prompt_file, "prompt")

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {role_name}")

    return prompt_file.read_text()


def list_schemas() -> list[str]:
    """List available schemas"""
    return [f.stem for f in _SCHEMA_DIR.glob("*.schema.json")]


def list_prompts() -> list[str]:
    """List available prompt roles"""
    return [d.name for d in _PROMPT_DIR.iterdir() if d.is_dir()]
