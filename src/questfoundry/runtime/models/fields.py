"""
Field definitions for artifact types.

This module handles the recursive FieldDefinition structure used
in artifact-type.schema.json for defining artifact fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from questfoundry.runtime.models.enums import FieldType


class FieldDefinition(BaseModel):
    """
    Recursive field definition from _definitions.schema.json#/$defs/field_definition.

    Nested objects/arrays are represented via:
    - properties: list[FieldDefinition] for object types
    - items: FieldDefinition for array of complex items
    - items_type: FieldType for array of scalar items
    """

    name: str
    type: FieldType
    description: str | None = None
    required: bool = False
    default: Any = None

    # Object type: nested field definitions
    properties: list[FieldDefinition] | None = None

    # Array type: item structure
    items: FieldDefinition | None = None  # Complex items (object/nested)
    items_type: FieldType | None = None  # Simple scalar items

    # String type constraints
    enum: list[str] | None = None
    format: str | None = None  # email, uri, uuid, date, datetime, markdown

    # Numeric constraints
    min: float | None = Field(default=None, alias="min")
    max: float | None = Field(default=None, alias="max")

    # Length constraints (string, text, array)
    min_length: int | None = None
    max_length: int | None = None

    # Reference type
    ref_target: str | None = None  # What entity type is referenced

    model_config = {"populate_by_name": True}


# Enable forward reference resolution for recursive type
FieldDefinition.model_rebuild()
