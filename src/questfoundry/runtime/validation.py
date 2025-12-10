"""Artifact validation with LLM-friendly error formatting.

This module provides validation utilities that produce detailed, actionable
feedback when LLM-generated data doesn't match artifact schemas.

The key insight: LLMs need specific feedback about what's wrong and how to fix it.
Generic Pydantic errors are not helpful - we need to tell them:
- Which fields are missing
- Which fields have invalid values and why
- What optional fields are available
- Clear hints for how to proceed

Usage
-----
    from questfoundry.runtime.validation import validate_artifact, format_validation_errors

    result = validate_artifact("hook_card", {"title": "My Hook"})
    if not result["success"]:
        # result contains missing_fields, invalid_fields, hint
        print(result["hint"])
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# Artifact type -> Pydantic model class mapping
# Populated lazily to avoid import issues
_ARTIFACT_MODELS: dict[str, type[BaseModel]] | None = None


def _get_artifact_models() -> dict[str, type[BaseModel]]:
    """Get mapping of artifact type names to Pydantic model classes."""
    global _ARTIFACT_MODELS
    if _ARTIFACT_MODELS is not None:
        return _ARTIFACT_MODELS

    _ARTIFACT_MODELS = {}

    try:
        from questfoundry.generated.models import (
            Act,
            AudioPlan,
            Beat,
            Brief,
            CanonEntry,
            Chapter,
            Character,
            Event,
            Fact,
            GatecheckReport,
            HookCard,
            Item,
            Location,
            Relationship,
            Scene,
            Sequence,
            Shotlist,
            Timeline,
            TranslationPack,
        )

        # Map various names to models (snake_case, kebab-case, etc.)
        _ARTIFACT_MODELS.update(
            {
                # Core workflow artifacts
                "brief": Brief,
                "briefs": Brief,
                "canon_entry": CanonEntry,
                "canon-entry": CanonEntry,
                "canon_entries": CanonEntry,
                "gatecheck_report": GatecheckReport,
                "gatecheck-report": GatecheckReport,
                "gatecheck": GatecheckReport,
                "hook_card": HookCard,
                "hook-card": HookCard,
                "hook": HookCard,
                "hooks": HookCard,
                # Narrative structure
                "act": Act,
                "acts": Act,
                "chapter": Chapter,
                "chapters": Chapter,
                "scene": Scene,
                "scenes": Scene,
                "sequence": Sequence,
                "sequences": Sequence,
                "beat": Beat,
                "beats": Beat,
                # World building
                "character": Character,
                "characters": Character,
                "location": Location,
                "locations": Location,
                "item": Item,
                "items": Item,
                "relationship": Relationship,
                "relationships": Relationship,
                "event": Event,
                "events": Event,
                "fact": Fact,
                "facts": Fact,
                "timeline": Timeline,
                "timelines": Timeline,
                # Production artifacts
                "shotlist": Shotlist,
                "shotlists": Shotlist,
                "audio_plan": AudioPlan,
                "audio-plan": AudioPlan,
                "audio_plans": AudioPlan,
                "translation_pack": TranslationPack,
                "translation-pack": TranslationPack,
                "translation_packs": TranslationPack,
            }
        )
        logger.debug(f"Loaded {len(_ARTIFACT_MODELS)} artifact model mappings")
    except ImportError as e:
        logger.warning(f"Could not load generated models: {e}")

    return _ARTIFACT_MODELS


def get_artifact_model(artifact_type: str) -> type[BaseModel] | None:
    """Get the Pydantic model for an artifact type.

    Parameters
    ----------
    artifact_type : str
        Artifact type name (e.g., "hook_card", "brief", "scene").

    Returns
    -------
    type[BaseModel] | None
        The Pydantic model class, or None if not found.
    """
    models = _get_artifact_models()
    # Try exact match first, then lowercase
    return models.get(artifact_type) or models.get(artifact_type.lower())


def format_validation_errors(
    exc: ValidationError,
    artifact_type: str,
    model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Format Pydantic ValidationError into LLM-friendly feedback.

    Instead of raw Pydantic error strings with URLs and truncated input dumps,
    this produces a clean structured response that helps the LLM understand
    exactly what needs to be fixed.

    Parameters
    ----------
    exc : ValidationError
        The Pydantic validation error.
    artifact_type : str
        Name of the artifact being validated.
    model : type[BaseModel] | None
        The Pydantic model (used to extract field info).

    Returns
    -------
    dict[str, Any]
        Structured feedback with:
        - success: False
        - artifact_type: The artifact being validated
        - error_count: Number of validation errors
        - missing_fields: List of required fields not provided
        - invalid_fields: List of fields with value errors
        - optional_fields: List of optional fields available
        - required_fields: List of required fields
        - hint: Concise instruction for the LLM
    """
    missing_fields: list[str] = []
    invalid_fields: list[dict[str, str]] = []

    for error in exc.errors():
        # Build field path (e.g., "header.status" for nested fields)
        field_path = ".".join(str(loc) for loc in error["loc"])
        error_type = error["type"]
        msg = error["msg"]

        if error_type == "missing":
            missing_fields.append(field_path)
        else:
            # Simplify common Pydantic messages
            clean_msg = msg
            if "String should have at least" in msg:
                clean_msg = msg.replace("String should have at least", "minimum length")
            elif "String should have at most" in msg:
                clean_msg = msg.replace("String should have at most", "maximum length")
            elif "Input should be" in msg:
                clean_msg = msg.replace("Input should be", "expected")
            elif "Field required" in msg:
                clean_msg = "This field is required"

            invalid_fields.append(
                {
                    "field": field_path,
                    "issue": clean_msg,
                    "error_type": error_type,
                }
            )

    # Extract field info from model if available
    required_field_names: list[str] = []
    optional_field_names: list[str] = []

    if model is not None:
        for field_name, field_info in model.model_fields.items():
            if field_info.is_required():
                required_field_names.append(field_name)
            else:
                optional_field_names.append(field_name)

    # Build hint based on error types
    hints = []
    if missing_fields:
        hints.append(f"Add missing required fields: {', '.join(missing_fields)}")
    if invalid_fields:
        hints.append("Fix invalid field values (see invalid_fields for details)")
    hints.append("Use consult_schema tool to check field requirements")

    return {
        "success": False,
        "artifact_type": artifact_type,
        "error_count": len(exc.errors()),
        "missing_fields": missing_fields if missing_fields else None,
        "invalid_fields": invalid_fields if invalid_fields else None,
        "required_fields": required_field_names if required_field_names else None,
        "optional_fields": optional_field_names if optional_field_names else None,
        "hint": ". ".join(hints),
    }


def validate_artifact(
    artifact_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Validate artifact data against its schema.

    Parameters
    ----------
    artifact_type : str
        Artifact type name (e.g., "hook_card", "brief").
    data : dict[str, Any]
        The artifact data to validate.

    Returns
    -------
    dict[str, Any]
        On success: {"success": True, "validated": <normalized data>}
        On failure: LLM-friendly feedback dict (see format_validation_errors)
    """
    model = get_artifact_model(artifact_type)
    if model is None:
        # No model found - pass through without validation
        logger.debug(f"No validation model for artifact type: {artifact_type}")
        return {"success": True, "validated": data}

    try:
        validated = model(**data)
        # Return normalized data (with defaults applied, types coerced, enums as strings)
        return {
            "success": True,
            "validated": validated.model_dump(exclude_unset=True, mode="json"),
        }
    except ValidationError as e:
        return format_validation_errors(e, artifact_type, model)


# Hot store key -> artifact type mapping
# Maps common hot_sot keys to their artifact types for auto-detection
HOT_SOT_KEY_TO_ARTIFACT: dict[str, str] = {
    "hooks": "hook_card",
    "briefs": "brief",
    "scenes": "scene",
    "canon_entries": "canon_entry",
    "gatecheck_reports": "gatecheck_report",
    # Singular forms
    "hook": "hook_card",
    "brief": "brief",
    "scene": "scene",
}

# Prefixes for numbered artifact keys (e.g., scene_1, act_1, chapter_2)
ARTIFACT_KEY_PREFIXES: dict[str, str] = {
    "scene_": "scene",
    "act_": "act",
    "chapter_": "chapter",
    "hook_": "hook_card",
    "brief_": "brief",
    "gatecheck_": "gatecheck_report",
    "character_": "character",
    "location_": "location",
    "item_": "item",
    "event_": "event",
    "fact_": "fact",
    "relationship_": "relationship",
    "timeline_": "timeline",
    "sequence_": "sequence",
    "beat_": "beat",
    "shotlist_": "shotlist",
    "audio_plan_": "audio_plan",
    "translation_pack_": "translation_pack",
}


def detect_artifact_type(key: str) -> str | None:
    """Detect artifact type from hot_sot key.

    Parameters
    ----------
    key : str
        The hot_sot key (e.g., "hooks", "briefs", "scene_1", "act_2").

    Returns
    -------
    str | None
        The artifact type, or None if not a known artifact key.
    """
    if not key:
        return None

    # Extract top-level key (e.g., "hooks.0" -> "hooks")
    top_key = key.split(".")[0]

    # Try exact match first
    if top_key in HOT_SOT_KEY_TO_ARTIFACT:
        return HOT_SOT_KEY_TO_ARTIFACT[top_key]

    # Try prefix match for numbered keys (e.g., scene_1, act_2)
    for prefix, artifact_type in ARTIFACT_KEY_PREFIXES.items():
        if top_key.startswith(prefix):
            return artifact_type

    return None
