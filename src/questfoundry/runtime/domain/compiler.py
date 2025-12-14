"""
Schema compiler - converts FieldDefinition to JSON Schema.

This module takes the FieldDefinition structure from artifact types
and produces JSON Schema Draft 2020-12 compatible schemas for validation.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.models.enums import FieldType
from questfoundry.runtime.models.fields import FieldDefinition


def compile_schema(
    fields: list[FieldDefinition],
    title: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Compile a list of FieldDefinitions into a JSON Schema.

    Args:
        fields: List of field definitions
        title: Optional schema title
        description: Optional schema description

    Returns:
        JSON Schema dict (Draft 2020-12 compatible)
    """
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "required": [],
    }

    if title:
        schema["title"] = title
    if description:
        schema["description"] = description

    for field in fields:
        prop_schema = _compile_field(field)
        schema["properties"][field.name] = prop_schema

        if field.required:
            schema["required"].append(field.name)

    # Remove empty required array
    if not schema["required"]:
        del schema["required"]

    return schema


def _compile_field(field: FieldDefinition) -> dict[str, Any]:
    """Compile a single FieldDefinition to a JSON Schema property."""
    schema: dict[str, Any] = {}

    if field.description:
        schema["description"] = field.description

    if field.default is not None:
        schema["default"] = field.default

    # Handle type-specific compilation
    match field.type:
        case FieldType.STRING:
            schema["type"] = "string"
            _add_string_constraints(schema, field)

        case FieldType.TEXT:
            schema["type"] = "string"
            _add_string_constraints(schema, field)

        case FieldType.INTEGER:
            schema["type"] = "integer"
            _add_numeric_constraints(schema, field)

        case FieldType.NUMBER:
            schema["type"] = "number"
            _add_numeric_constraints(schema, field)

        case FieldType.BOOLEAN:
            schema["type"] = "boolean"

        case FieldType.DATE:
            schema["type"] = "string"
            schema["format"] = "date"

        case FieldType.DATETIME:
            schema["type"] = "string"
            schema["format"] = "date-time"

        case FieldType.URI:
            schema["type"] = "string"
            schema["format"] = "uri"

        case FieldType.ARRAY:
            schema["type"] = "array"
            _add_array_constraints(schema, field)

        case FieldType.OBJECT:
            schema["type"] = "object"
            _add_object_constraints(schema, field)

        case FieldType.REF:
            # References are stored as strings (IDs)
            schema["type"] = "string"
            if field.ref_target:
                schema["$comment"] = f"Reference to {field.ref_target}"

        case _:
            # Unknown type, default to string
            schema["type"] = "string"

    return schema


def _add_string_constraints(schema: dict[str, Any], field: FieldDefinition) -> None:
    """Add string-specific constraints to schema."""
    if field.enum:
        schema["enum"] = field.enum

    if field.format:
        schema["format"] = field.format

    if field.min_length is not None:
        schema["minLength"] = field.min_length

    if field.max_length is not None:
        schema["maxLength"] = field.max_length


def _add_numeric_constraints(schema: dict[str, Any], field: FieldDefinition) -> None:
    """Add numeric constraints to schema."""
    if field.min is not None:
        schema["minimum"] = field.min

    if field.max is not None:
        schema["maximum"] = field.max


def _add_array_constraints(schema: dict[str, Any], field: FieldDefinition) -> None:
    """Add array-specific constraints to schema."""
    if field.min_length is not None:
        schema["minItems"] = field.min_length

    if field.max_length is not None:
        schema["maxItems"] = field.max_length

    # Determine items schema
    if field.items:
        # Complex items (nested object/array)
        schema["items"] = _compile_field(field.items)
    elif field.items_type:
        # Simple scalar items
        schema["items"] = _scalar_type_schema(field.items_type)
    else:
        # Default to any
        schema["items"] = {}


def _add_object_constraints(schema: dict[str, Any], field: FieldDefinition) -> None:
    """Add object-specific constraints to schema."""
    if field.properties:
        schema["properties"] = {}
        required = []

        for prop in field.properties:
            schema["properties"][prop.name] = _compile_field(prop)
            if prop.required:
                required.append(prop.name)

        if required:
            schema["required"] = required


def _scalar_type_schema(field_type: FieldType) -> dict[str, Any]:
    """Get JSON Schema for a scalar type."""
    match field_type:
        case FieldType.STRING:
            return {"type": "string"}
        case FieldType.TEXT:
            return {"type": "string"}
        case FieldType.INTEGER:
            return {"type": "integer"}
        case FieldType.NUMBER:
            return {"type": "number"}
        case FieldType.BOOLEAN:
            return {"type": "boolean"}
        case FieldType.DATE:
            return {"type": "string", "format": "date"}
        case FieldType.DATETIME:
            return {"type": "string", "format": "date-time"}
        case FieldType.URI:
            return {"type": "string", "format": "uri"}
        case FieldType.REF:
            return {"type": "string"}
        case _:
            return {"type": "string"}


def compile_artifact_type_schema(
    artifact_type_id: str,
    artifact_type_name: str,
    fields: list[FieldDefinition],
    description: str | None = None,
) -> dict[str, Any]:
    """
    Compile a complete schema for an artifact type.

    Adds standard artifact system fields (_id, _type, _version, etc.)
    alongside the user-defined fields.

    Args:
        artifact_type_id: Artifact type ID
        artifact_type_name: Human-readable name
        fields: User-defined field definitions
        description: Optional description

    Returns:
        Complete JSON Schema for the artifact type
    """
    # Start with user-defined fields
    schema = compile_schema(
        fields=fields,
        title=artifact_type_name,
        description=description,
    )

    # Add system fields
    system_fields = {
        "_id": {
            "type": "string",
            "description": "Unique artifact identifier",
        },
        "_type": {
            "type": "string",
            "const": artifact_type_id,
            "description": "Artifact type identifier",
        },
        "_version": {
            "type": "integer",
            "minimum": 1,
            "description": "Artifact version number",
        },
        "_created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp",
        },
        "_updated_at": {
            "type": "string",
            "format": "date-time",
            "description": "Last update timestamp",
        },
        "_created_by": {
            "type": "string",
            "description": "Agent ID that created this artifact",
        },
        "_lifecycle_state": {
            "type": "string",
            "description": "Current lifecycle state",
        },
    }

    # Merge system fields into properties
    schema["properties"] = {**system_fields, **schema.get("properties", {})}

    # System fields that are always required
    system_required = ["_id", "_type", "_version"]
    existing_required = schema.get("required", [])
    schema["required"] = system_required + existing_required

    return schema
