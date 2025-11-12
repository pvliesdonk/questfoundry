"""Schema validation utilities"""

import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from ..utils.resources import get_schema


def validate_instance(instance: dict[str, Any], schema_name: str) -> bool:
    """
    Validate an instance against a schema.

    Args:
        instance: Data to validate
        schema_name: Name of the schema to validate against

    Returns:
        True if valid, False otherwise

    Raises:
        FileNotFoundError: If schema doesn't exist
        ValueError: If path traversal detected
    """
    schema = get_schema(schema_name)
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


def validate_instance_detailed(
    instance: dict[str, Any], schema_name: str
) -> dict[str, Any]:
    """
    Validate instance and return detailed error information.

    Args:
        instance: Data to validate
        schema_name: Name of the schema to validate against

    Returns:
        Dictionary with validation results including errors

    Raises:
        FileNotFoundError: If schema doesn't exist
        ValueError: If path traversal detected
    """
    schema = get_schema(schema_name)
    try:
        jsonschema.validate(instance, schema)
        return {"valid": True, "errors": []}
    except jsonschema.ValidationError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "message": str(e.message),
            "path": list(e.path),
        }


def validate_schema(schema_path: str | Path) -> bool:
    """
    Validate that a schema file is valid JSON Schema Draft 2020-12.

    Args:
        schema_path: Path to schema file

    Returns:
        True if schema is valid, False otherwise

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema is invalid JSON
    """
    if isinstance(schema_path, str):
        schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {e}") from e

    # Basic validation: ensure it has required schema properties
    required_fields = ["$schema", "$id"]
    for field in required_fields:
        if field not in schema:
            return False

    # Validate schema structure using Draft 2020-12 validator
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except jsonschema.SchemaError:
        return False
