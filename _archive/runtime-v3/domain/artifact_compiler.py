"""Compile ArtifactType definitions into Pydantic models.

This module dynamically generates Pydantic model classes from ArtifactType
definitions loaded from domain-v4/. This allows the domain to define new
artifact types without requiring code changes.

Usage:
    from questfoundry.runtime.domain.artifact_compiler import compile_artifact_type
    from questfoundry.runtime.domain.loader import load_studio

    studio = load_studio("domain-v4/studio.json")
    artifact_type = studio.artifact_types["section"]
    SectionModel = compile_artifact_type(artifact_type)

    # Now use SectionModel like any Pydantic model
    section = SectionModel(anchor="001", title="Test", prose="...")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, create_model

from questfoundry.runtime.domain.metamodel import (
    ArtifactType,
    FieldDefinition,
    FieldType,
)

logger = logging.getLogger(__name__)

# Cache for compiled models
_MODEL_CACHE: dict[str, type[BaseModel]] = {}

# Cache for dynamically created enum types
_ENUM_CACHE: dict[str, type[Enum]] = {}


def _make_enum_name(artifact_id: str, field_name: str) -> str:
    """Generate a unique enum name for a field."""
    # Convert to PascalCase
    artifact_part = "".join(word.capitalize() for word in artifact_id.split("_"))
    field_part = "".join(word.capitalize() for word in field_name.split("_"))
    return f"{artifact_part}{field_part}Enum"


def _get_or_create_enum(
    artifact_id: str, field_name: str, values: list[str]
) -> type[Enum]:
    """Get or create an Enum type for a field's allowed values."""
    cache_key = f"{artifact_id}.{field_name}"

    if cache_key in _ENUM_CACHE:
        return _ENUM_CACHE[cache_key]

    enum_name = _make_enum_name(artifact_id, field_name)

    # Create enum members: uppercase key -> original value
    members = {v.upper().replace("-", "_").replace(" ", "_"): v for v in values}

    enum_type = Enum(enum_name, members, type=str)  # type: ignore
    _ENUM_CACHE[cache_key] = enum_type

    return enum_type


def _field_type_to_python(
    field: FieldDefinition,
    artifact_id: str,
    parent_field_name: str = "",
) -> type:
    """Map a FieldDefinition to a Python type.

    Handles:
    - Basic types (string, integer, boolean, etc.)
    - Enum constraints
    - Arrays with items_type or complex items
    - Objects with properties (creates nested models)
    """
    # Handle enum constraint first (for string fields with enum)
    if field.enum and field.type in (FieldType.STRING, FieldType.TEXT):
        return _get_or_create_enum(artifact_id, field.name, field.enum)

    # Basic type mapping
    type_map: dict[FieldType, type] = {
        FieldType.STRING: str,
        FieldType.TEXT: str,
        FieldType.INTEGER: int,
        FieldType.NUMBER: float,
        FieldType.BOOLEAN: bool,
        FieldType.DATE: str,  # ISO date string
        FieldType.DATETIME: str,  # ISO datetime string
        FieldType.URI: str,
        FieldType.REF: str,  # Reference ID string
    }

    if field.type in type_map:
        return type_map[field.type]

    # Handle array type
    if field.type == FieldType.ARRAY:
        if field.items_type:
            # Simple array with scalar items
            item_type = type_map.get(field.items_type, Any)
            return list[item_type]  # type: ignore

        if field.items:
            # Complex array with nested object items
            if field.items.type == FieldType.OBJECT and field.items.properties:
                # Create a nested model for the array items
                nested_model = _compile_nested_object(
                    artifact_id=artifact_id,
                    field_name=field.name,
                    properties=field.items.properties,
                )
                return list[nested_model]  # type: ignore
            else:
                # Simple nested item
                item_type = _field_type_to_python(field.items, artifact_id, field.name)
                return list[item_type]  # type: ignore

        # Fallback: list of Any
        return list[Any]

    # Handle object type
    if field.type == FieldType.OBJECT:
        if field.properties:
            # Create a nested model
            return _compile_nested_object(
                artifact_id=artifact_id,
                field_name=field.name,
                properties=field.properties,
            )
        # Fallback: dict
        return dict[str, Any]

    # Unknown type, default to Any
    logger.warning(f"Unknown field type: {field.type}, defaulting to Any")
    return Any


def _compile_nested_object(
    artifact_id: str,
    field_name: str,
    properties: list[FieldDefinition],
) -> type[BaseModel]:
    """Compile a nested object field into a Pydantic model."""
    # Generate model name
    artifact_part = "".join(word.capitalize() for word in artifact_id.split("_"))
    field_part = "".join(word.capitalize() for word in field_name.split("_"))
    model_name = f"{artifact_part}{field_part}"

    # Check cache
    cache_key = f"_nested.{artifact_id}.{field_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Build field definitions
    field_definitions: dict[str, Any] = {}

    for prop in properties:
        python_type = _field_type_to_python(prop, artifact_id, field_name)
        field_info = _create_field_info(prop)

        if prop.required:
            field_definitions[prop.name] = (python_type, field_info)
        else:
            field_definitions[prop.name] = (python_type | None, field_info)

    # Create the model
    model = create_model(model_name, **field_definitions)

    # Cache it
    _MODEL_CACHE[cache_key] = model

    return model


def _create_field_info(field: FieldDefinition) -> Any:
    """Create a Pydantic Field() from a FieldDefinition."""
    kwargs: dict[str, Any] = {}

    if field.description:
        kwargs["description"] = field.description

    if field.min_length is not None:
        kwargs["min_length"] = field.min_length

    if field.max_length is not None:
        kwargs["max_length"] = field.max_length

    if field.min is not None:
        kwargs["ge"] = field.min

    if field.max is not None:
        kwargs["le"] = field.max

    # Handle default value
    if field.default is not None:
        kwargs["default"] = field.default
    elif not field.required:
        kwargs["default"] = None
    else:
        kwargs["default"] = ...

    return Field(**kwargs)


def compile_artifact_type(
    artifact_type: ArtifactType,
    *,
    force_recompile: bool = False,
) -> type[BaseModel]:
    """Compile an ArtifactType definition into a Pydantic model class.

    Parameters
    ----------
    artifact_type : ArtifactType
        The artifact type definition from domain-v4.
    force_recompile : bool, optional
        If True, bypass cache and recompile. Default False.

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class.

    Example
    -------
    >>> studio = load_studio("domain-v4/studio.json")
    >>> Section = compile_artifact_type(studio.artifact_types["section"])
    >>> section = Section(anchor="001", title="Test", prose="...", choices=[...])
    """
    cache_key = artifact_type.id

    if not force_recompile and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    logger.debug(f"Compiling artifact type: {artifact_type.id}")

    # Build field definitions
    field_definitions: dict[str, Any] = {}

    for field in artifact_type.fields:
        python_type = _field_type_to_python(field, artifact_type.id)
        field_info = _create_field_info(field)

        if field.required:
            field_definitions[field.name] = (python_type, field_info)
        else:
            field_definitions[field.name] = (python_type | None, field_info)

    # Generate model name (PascalCase)
    model_name = "".join(word.capitalize() for word in artifact_type.id.split("_"))

    # Build docstring
    docstring = f"{artifact_type.name}.\n\n"
    if artifact_type.description:
        docstring += f"{artifact_type.description}\n\n"
    docstring += f"Category: {artifact_type.category}\n"
    if artifact_type.default_store:
        docstring += f"Default Store: {artifact_type.default_store}\n"
    if artifact_type.lifecycle:
        states = [s.id for s in artifact_type.lifecycle.states]
        docstring += f"Lifecycle States: {', '.join(states)}\n"

    # Create the model
    model = create_model(
        model_name,
        __doc__=docstring,
        **field_definitions,
    )

    # Add lifecycle states as class attribute for convenience
    if artifact_type.lifecycle:
        model.LIFECYCLE_STATES = [s.id for s in artifact_type.lifecycle.states]  # type: ignore
        model.INITIAL_STATE = artifact_type.lifecycle.initial_state  # type: ignore

    # Cache it
    _MODEL_CACHE[cache_key] = model

    logger.debug(f"Compiled artifact type: {artifact_type.id} -> {model_name}")

    return model


def get_compiled_model(artifact_type_id: str) -> type[BaseModel] | None:
    """Get a previously compiled model by artifact type ID.

    Returns None if the model hasn't been compiled yet.
    Use compile_artifact_type() to compile from an ArtifactType definition.
    """
    return _MODEL_CACHE.get(artifact_type_id)


def clear_cache() -> None:
    """Clear all cached compiled models.

    Useful for testing or when domain definitions change.
    """
    _MODEL_CACHE.clear()
    _ENUM_CACHE.clear()
    logger.debug("Cleared artifact compiler cache")


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the cache."""
    return {
        "models": len(_MODEL_CACHE),
        "enums": len(_ENUM_CACHE),
    }
