"""Resource loading utilities for schemas and prompts"""

import json
from importlib.resources import files
from typing import Any


def _validate_safe_name(name: str, resource_type: str) -> None:
    """
    Validate that resource name doesn't contain path traversal attempts.

    Args:
        name: Resource name to validate
        resource_type: Type of resource for error message

    Raises:
        ValueError: If path traversal is detected
    """
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError(
            f"Invalid {resource_type} name: path traversal detected. "
            f"Name must not contain path separators or '..'."
        )


def get_schema(schema_name: str) -> dict[str, Any]:
    """
    Load a schema from bundled resources.

    Args:
        schema_name: Name of the schema (without .schema.json extension)

    Returns:
        Schema dictionary

    Raises:
        FileNotFoundError: If schema doesn't exist
        ValueError: If path traversal is detected
    """
    _validate_safe_name(schema_name, "schema")

    schema_file = f"{schema_name}.schema.json"
    schemas_package = files("questfoundry.resources.schemas")

    resource = schemas_package.joinpath(schema_file)
    try:
        schema_text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Schema not found: {schema_name}") from e

    schema: dict[str, Any] = json.loads(schema_text)
    return schema


def get_prompt(role_name: str) -> str:
    """
    Load a prompt from bundled resources.

    Args:
        role_name: Name of the role

    Returns:
        Prompt text

    Raises:
        FileNotFoundError: If prompt doesn't exist
        ValueError: If path traversal is detected
    """
    _validate_safe_name(role_name, "prompt")

    prompts_package = files("questfoundry.resources.prompts")
    role_dir = prompts_package.joinpath(role_name)
    prompt_file = role_dir.joinpath("system_prompt.md")

    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Prompt not found: {role_name}") from e


def list_schemas() -> list[str]:
    """
    List available schemas.

    Returns:
        List of schema names (without .schema.json extension)

    Raises:
        FileNotFoundError: If resources package is not properly installed
    """
    schemas_package = files("questfoundry.resources.schemas")
    return [
        f.name.replace(".schema.json", "")
        for f in schemas_package.iterdir()
        if f.name.endswith(".schema.json")
    ]


def list_prompts() -> list[str]:
    """
    List available prompt roles.

    Returns:
        List of role names

    Raises:
        FileNotFoundError: If resources package is not properly installed
    """
    prompts_package = files("questfoundry.resources.prompts")
    return [
        d.name
        for d in prompts_package.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]
