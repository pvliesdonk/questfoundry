"""Dynamic Pydantic model generation from domain-v4 artifact type definitions.

This module generates Pydantic model classes at runtime from the artifact type
JSON schemas in domain-v4/artifact-types/. This replaces the need for compiled
models in generated/.

Usage:
    from questfoundry.runtime.domain.artifact_models import get_artifact_model, ARTIFACT_REGISTRY

    # Get a model class
    SectionModel = get_artifact_model("section")

    # Validate data
    section = SectionModel(anchor="001", title="Test", prose="...", choices=[...])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, create_model

from questfoundry.runtime.domain.models import ArtifactField, ArtifactType

logger = logging.getLogger(__name__)

# Cache for generated models
_MODEL_CACHE: dict[str, type[BaseModel]] = {}
_ARTIFACT_TYPES: dict[str, ArtifactType] | None = None


def _get_python_type(field: ArtifactField) -> type:
    """Map artifact field type strings to Python types."""
    type_map = {
        "string": str,
        "text": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    base_type = type_map.get(field.type, Any)

    # For arrays, try to get items type
    if field.type == "array":
        if field.items_type:
            item_type = type_map.get(field.items_type, Any)
            return list[item_type]  # type: ignore
        return list[Any]

    return base_type


def _create_field_info(field: ArtifactField) -> tuple[type, Any]:
    """Create Pydantic field info from ArtifactField.

    Returns (type_annotation, Field(...)) tuple for create_model.
    """
    python_type = _get_python_type(field)

    # Build Field kwargs
    field_kwargs: dict[str, Any] = {}

    if field.description:
        field_kwargs["description"] = field.description

    if field.min_length is not None:
        field_kwargs["min_length"] = field.min_length

    if field.max_length is not None:
        field_kwargs["max_length"] = field.max_length

    # Handle required vs optional
    if field.required:
        field_kwargs["default"] = ...
        return (python_type, Field(**field_kwargs))
    else:
        # Make optional with None default
        return (python_type | None, Field(default=None, **field_kwargs))


def create_artifact_model(artifact_type: ArtifactType) -> type[BaseModel]:
    """Create a Pydantic model class from an ArtifactType definition.

    Parameters
    ----------
    artifact_type : ArtifactType
        The artifact type definition from domain-v4.

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class.
    """
    # Check cache first
    if artifact_type.id in _MODEL_CACHE:
        return _MODEL_CACHE[artifact_type.id]

    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}

    for field in artifact_type.fields:
        field_definitions[field.name] = _create_field_info(field)

    # Extract lifecycle states for class attribute
    lifecycle_states: list[str] = []
    if artifact_type.lifecycle:
        lifecycle_states = [s.id for s in artifact_type.lifecycle.states]

    # Create the model class
    model_name = artifact_type.name.replace(" ", "")

    # Create model with proper docstring
    model = create_model(
        model_name,
        __doc__=f"""{artifact_type.name}.

{artifact_type.description or ''}

Category: {artifact_type.category}
Default Store: {artifact_type.default_store or 'workspace'}
""",
        **field_definitions,
    )

    # Add lifecycle as class variable (can't do this in create_model directly)
    model.LIFECYCLE = lifecycle_states  # type: ignore

    # Cache it
    _MODEL_CACHE[artifact_type.id] = model

    return model


def _load_artifact_types() -> dict[str, ArtifactType]:
    """Load artifact types from domain-v4."""
    global _ARTIFACT_TYPES

    if _ARTIFACT_TYPES is not None:
        return _ARTIFACT_TYPES

    from questfoundry.runtime.domain import load_studio

    # Find domain-v4 path
    domain_v4_path = Path(__file__).parents[4] / "domain-v4" / "studio.json"

    if not domain_v4_path.exists():
        logger.warning(f"domain-v4 not found at {domain_v4_path}")
        _ARTIFACT_TYPES = {}
        return _ARTIFACT_TYPES

    studio = load_studio(domain_v4_path)
    _ARTIFACT_TYPES = studio.artifact_types

    logger.debug(f"Loaded {len(_ARTIFACT_TYPES)} artifact types from domain-v4")
    return _ARTIFACT_TYPES


def get_artifact_model(artifact_type_id: str) -> type[BaseModel] | None:
    """Get the Pydantic model for an artifact type.

    Parameters
    ----------
    artifact_type_id : str
        Artifact type ID (e.g., "section", "hook_card", "codex_entry").

    Returns
    -------
    type[BaseModel] | None
        The generated Pydantic model class, or None if not found.
    """
    artifact_types = _load_artifact_types()

    # Try exact match
    if artifact_type_id in artifact_types:
        return create_artifact_model(artifact_types[artifact_type_id])

    # Try lowercase
    artifact_type_id_lower = artifact_type_id.lower()
    if artifact_type_id_lower in artifact_types:
        return create_artifact_model(artifact_types[artifact_type_id_lower])

    # Try with underscores to hyphens
    artifact_type_id_hyphen = artifact_type_id.replace("_", "-")
    if artifact_type_id_hyphen in artifact_types:
        return create_artifact_model(artifact_types[artifact_type_id_hyphen])

    return None


def get_artifact_registry() -> dict[str, type[BaseModel]]:
    """Get a registry of all artifact type models.

    Returns
    -------
    dict[str, type[BaseModel]]
        Mapping of artifact type IDs to their Pydantic model classes.
    """
    artifact_types = _load_artifact_types()

    registry: dict[str, type[BaseModel]] = {}
    for artifact_id, artifact_type in artifact_types.items():
        registry[artifact_id] = create_artifact_model(artifact_type)

    return registry


# Compatibility alias for code expecting ARTIFACT_REGISTRY
def get_ARTIFACT_REGISTRY() -> dict[str, type[BaseModel]]:
    """Compatibility function for code expecting ARTIFACT_REGISTRY.

    Note: Returns a fresh dict each time. For caching, access _MODEL_CACHE directly
    after calling get_artifact_registry().
    """
    return get_artifact_registry()


# For simple imports, provide a property-like module-level access
class _ArtifactRegistryProxy:
    """Lazy proxy for ARTIFACT_REGISTRY."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_artifact_registry(), name)

    def __getitem__(self, key: str) -> type[BaseModel]:
        return get_artifact_registry()[key]

    def __contains__(self, key: str) -> bool:
        return key in get_artifact_registry()

    def __iter__(self):
        return iter(get_artifact_registry())

    def items(self):
        return get_artifact_registry().items()

    def keys(self):
        return get_artifact_registry().keys()

    def values(self):
        return get_artifact_registry().values()

    def get(self, key: str, default=None):
        return get_artifact_registry().get(key, default)


ARTIFACT_REGISTRY = _ArtifactRegistryProxy()
