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
import re
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# Artifact type -> Pydantic model class mapping
# Uses the generated ARTIFACT_REGISTRY from the domain compiler
_ARTIFACT_MODELS: dict[str, type[BaseModel]] | None = None


def _get_artifact_models() -> dict[str, type[BaseModel]]:
    """Get mapping of artifact type names to Pydantic model classes.

    Uses the generated ARTIFACT_REGISTRY from the domain compiler,
    which maps artifact IDs (snake_case) to their Pydantic model classes.
    """
    global _ARTIFACT_MODELS
    if _ARTIFACT_MODELS is not None:
        return _ARTIFACT_MODELS

    _ARTIFACT_MODELS = {}

    try:
        from questfoundry.generated.models.artifacts import ARTIFACT_REGISTRY

        # Copy the generated registry
        _ARTIFACT_MODELS.update(ARTIFACT_REGISTRY)

        # Add common aliases (plural forms, kebab-case, etc.)
        alias_map = {
            # Plurals
            "briefs": "brief",
            "acts": "act",
            "chapters": "chapter",
            "scenes": "scene",
            "sequences": "sequence",
            "beats": "beat",
            "hooks": "hook_card",
            "characters": "character",
            "locations": "location",
            "items": "item",
            "relationships": "relationship",
            "events": "event",
            "facts": "fact",
            "timelines": "timeline",
            "shotlists": "shotlist",
            "audio_plans": "audio_plan",
            "translation_packs": "translation_pack",
            "canon_entries": "canon_entry",
            "gatecheck_reports": "gatecheck_report",
            # Kebab-case
            "canon-entry": "canon_entry",
            "gatecheck-report": "gatecheck_report",
            "hook-card": "hook_card",
            "audio-plan": "audio_plan",
            "translation-pack": "translation_pack",
            # Short forms
            "hook": "hook_card",
            "gatecheck": "gatecheck_report",
        }

        for alias, canonical in alias_map.items():
            if canonical in ARTIFACT_REGISTRY:
                _ARTIFACT_MODELS[alias] = ARTIFACT_REGISTRY[canonical]

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
        On failure: LLM-friendly feedback dict with actionable hints
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
        result = format_validation_errors(e, artifact_type, model)
        # Enhance with 9.4 validate-with-feedback fields for common mismatches
        _enhance_with_schema_guidance(artifact_type, data, result)
        return result


def _enhance_with_schema_guidance(
    artifact_type: str, data: dict[str, Any], result: dict[str, Any]
) -> None:
    """Enhance validation error with 9.4 validate-with-feedback fields.

    Adds actionable hints based on common mistakes. Does NOT hardcode
    domain-specific enum values - those come from Pydantic error messages.
    """
    # Scene-specific: help with common structural mistakes
    if artifact_type == "scene":
        # Build specific hints based on what's wrong (structural, not enum values)
        hints = []
        gates = data.get("gates", [])
        for i, gate in enumerate(gates):
            if isinstance(gate, dict):
                if "gate_id" in gate and "key" not in gate:
                    hints.append(f"gates[{i}]: rename 'gate_id' to 'key'")
                if "condition" in gate and "gate_type" not in gate:
                    hints.append(f"gates[{i}]: add 'gate_type' field")

        choices = data.get("choices", [])
        for i, choice in enumerate(choices):
            if isinstance(choice, str):
                hints.append(
                    f"choices[{i}]: convert string '{choice}' to "
                    f"{{'label': '{choice}', 'target': '<scene_id>'}}"
                )

        if hints:
            # Prepend specific fixes to existing hint
            fix_text = "; ".join(hints)
            if data.get("content"):
                fix_text += ". OR omit gates/choices (they are optional)"
            result["hint"] = f"{fix_text}. {result.get('hint', '')}"


def detect_artifact_type(key: str) -> str | None:
    """Detect artifact type from hot_sot key.

    Uses the generated ARTIFACT_REGISTRY to detect types from keys like:
    - Exact matches: "scene", "act", "chapter"
    - Numbered keys: "scene_1", "act_2", "chapter_3"
    - Collection keys: "scenes", "acts", "hooks"

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

    # Get the models (which includes both registry and aliases)
    models = _get_artifact_models()

    # Try exact match first (handles plurals and aliases via _get_artifact_models)
    if top_key in models:
        # Return the canonical artifact type ID
        model = models[top_key]
        # Find the canonical name from ARTIFACT_REGISTRY
        try:
            from questfoundry.generated.models.artifacts import ARTIFACT_REGISTRY

            for artifact_id, cls in ARTIFACT_REGISTRY.items():
                if cls is model:
                    return artifact_id
        except ImportError:
            pass
        # Fallback: return the key if it's a known model
        return top_key

    # Try prefix match for numbered keys (e.g., scene_1, act_2)
    # Extract the prefix before any trailing digits/underscores
    match = re.match(r"^([a-z_]+?)_?\d+$", top_key)
    if match:
        prefix = match.group(1)
        # Check if prefix matches any registry entry
        try:
            from questfoundry.generated.models.artifacts import ARTIFACT_REGISTRY

            if prefix in ARTIFACT_REGISTRY:
                return prefix
        except ImportError:
            pass

    return None
