"""Validate-with-feedback pattern for LLM self-correction.

This module implements action-first structured feedback that helps LLMs
understand validation errors and make targeted corrections.

Key principles (from v4 ARCHITECTURE-v3.md Section 9.4):
- Action-first: Recovery directive at top, not buried
- Fuzzy matching: Detect field name typos and suggest corrections
- Semantic: Separate outcome, reason, and action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Common synonyms that LLMs might use instead of expected field names
FIELD_SYNONYMS: dict[str, list[str]] = {
    "prose": ["content", "text", "body", "writing"],
    "title": ["name", "heading", "label", "section_title"],
    "anchor": ["hook", "opening", "start"],
    "choices": ["options", "branches", "decisions"],
    "genre": ["category", "type", "style"],
    "tone": ["mood", "atmosphere", "feeling"],
    "themes": ["topics", "motifs", "elements"],
    "audience": ["target", "readers", "demographic"],
}


def _similarity_ratio(a: str, b: str) -> float:
    """Calculate string similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_field_correction(
    provided_field: str,
    expected_fields: set[str],
    threshold: float = 0.6,
) -> str | None:
    """Find a correction for a misnamed field.

    Checks for:
    1. Suffix match (e.g., section_title -> title)
    2. Prefix match (e.g., title_text -> title)
    3. Synonym match (e.g., content -> prose)
    4. Fuzzy similarity match

    Args:
        provided_field: The field name that was provided.
        expected_fields: Set of valid field names.
        threshold: Minimum similarity ratio for fuzzy match.

    Returns:
        Suggested field name if a match is found, None otherwise.
    """
    provided_lower = provided_field.lower()

    for expected in expected_fields:
        expected_lower = expected.lower()

        # Exact match (shouldn't happen, but handle it)
        if provided_lower == expected_lower:
            return None

        # Suffix match: provided ends with expected (e.g., section_title -> title)
        if provided_lower.endswith(f"_{expected_lower}") or provided_lower.endswith(expected_lower):
            return expected

        # Prefix match: provided starts with expected (e.g., title_text -> title)
        if provided_lower.startswith(f"{expected_lower}_") or provided_lower.startswith(
            expected_lower
        ):
            return expected

    # Check synonyms
    for expected, synonyms in FIELD_SYNONYMS.items():
        if expected in expected_fields and provided_lower in [s.lower() for s in synonyms]:
            return expected

    # Fuzzy match as last resort
    best_match = None
    best_ratio = threshold

    for expected in expected_fields:
        ratio = _similarity_ratio(provided_field, expected)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = expected

    return best_match


@dataclass
class ValidationFeedback:
    """Structured feedback for LLM validation errors.

    Implements the action-first pattern where the recovery directive
    is the first thing the LLM reads, not buried in error details.

    Attributes:
        action_outcome: What happened - "saved" or "rejected".
        rejection_reason: Why rejected (if rejected).
        recovery_action: Clear directive for what to do next.
        field_corrections: Map of provided field -> "rename to 'expected'".
        missing_required: List of required fields that are missing.
        invalid_fields: List of fields with invalid values.
        error_count: Total number of errors.
        errors: Detailed error list (for debugging).
    """

    action_outcome: str
    rejection_reason: str | None = None
    recovery_action: str | None = None
    field_corrections: dict[str, str] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    invalid_fields: list[dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.action_outcome == "saved"

    @classmethod
    def success(cls) -> ValidationFeedback:
        """Create a success feedback instance."""
        return cls(action_outcome="saved")

    @classmethod
    def from_pydantic_errors(
        cls,
        errors: Sequence[Mapping[str, Any]],
        provided_fields: set[str],
        required_fields: set[str],
        artifact_type: str = "artifact",  # noqa: ARG003 - reserved for future logging
    ) -> ValidationFeedback:
        """Create feedback from Pydantic validation errors.

        Args:
            errors: Pydantic validation errors (from e.errors()).
            provided_fields: Set of field names that were provided.
            required_fields: Set of required field names.
            artifact_type: Type of artifact being validated (for logging).

        Returns:
            ValidationFeedback with structured error information.
        """
        field_corrections: dict[str, str] = {}
        missing_required: list[str] = []
        invalid_fields: list[dict[str, Any]] = []
        error_messages: list[str] = []

        # Process each error
        for error in errors:
            loc = error.get("loc", ())
            msg = error.get("msg", "")
            error_type = error.get("type", "")

            # Build field path
            field_path = ".".join(str(part) for part in loc) if loc else ""

            # Format error message
            if field_path:
                error_messages.append(f"{field_path}: {msg}")
            else:
                error_messages.append(msg)

            # Categorize error
            if error_type == "missing" or "required" in msg.lower() or "missing" in msg.lower():
                if field_path:
                    missing_required.append(field_path)
            elif field_path:
                invalid_fields.append(
                    {
                        "field": field_path,
                        "issue": msg,
                        "type": error_type,
                    }
                )

        # Find field corrections for extra/unknown fields
        for provided in provided_fields:
            if provided not in required_fields:
                correction = _find_field_correction(provided, required_fields)
                if correction:
                    field_corrections[provided] = f"rename to '{correction}'"

        # Build recovery action (action-first!)
        actions: list[str] = []
        if field_corrections:
            actions.append(f"Rename {len(field_corrections)} field(s)")
        if missing_required:
            actions.append(f"add {len(missing_required)} missing field(s)")
        if invalid_fields:
            actions.append(f"fix {len(invalid_fields)} invalid field(s)")

        recovery_action = ", then ".join(actions) + ", then retry." if actions else "Review errors and retry."

        return cls(
            action_outcome="rejected",
            rejection_reason="validation_failed",
            recovery_action=recovery_action,
            field_corrections=field_corrections,
            missing_required=missing_required,
            invalid_fields=invalid_fields,
            error_count=len(errors),
            errors=error_messages,
        )

    @classmethod
    def from_error_strings(
        cls,
        errors: list[str],
        provided_fields: set[str] | None = None,
        required_fields: set[str] | None = None,
    ) -> ValidationFeedback:
        """Create feedback from a list of error message strings.

        Args:
            errors: List of error message strings.
            provided_fields: Optional set of provided field names.
            required_fields: Optional set of required field names.

        Returns:
            ValidationFeedback with structured error information.
        """
        field_corrections: dict[str, str] = {}
        missing_required: list[str] = []
        invalid_fields: list[dict[str, Any]] = []

        for error in errors:
            # Parse "field: message" format
            if ": " in error:
                field_path, msg = error.split(": ", 1)

                if "required" in msg.lower() or "missing" in msg.lower():
                    missing_required.append(field_path)
                else:
                    invalid_fields.append({"field": field_path, "issue": msg})
            else:
                # Can't parse, treat as general error
                invalid_fields.append({"field": "", "issue": error})

        # Find field corrections if we have the field sets
        if provided_fields and required_fields:
            for provided in provided_fields:
                if provided not in required_fields:
                    correction = _find_field_correction(provided, required_fields)
                    if correction:
                        field_corrections[provided] = f"rename to '{correction}'"

        # Build recovery action
        actions: list[str] = []
        if field_corrections:
            actions.append(f"Rename {len(field_corrections)} field(s)")
        if missing_required:
            actions.append(f"add {len(missing_required)} missing field(s)")
        if invalid_fields:
            actions.append(f"fix {len(invalid_fields)} invalid field(s)")

        recovery_action = ", then ".join(actions) + ", then retry." if actions else "Review errors and retry."

        return cls(
            action_outcome="rejected",
            rejection_reason="validation_failed",
            recovery_action=recovery_action,
            field_corrections=field_corrections,
            missing_required=missing_required,
            invalid_fields=invalid_fields,
            error_count=len(errors),
            errors=errors,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns dict with action_outcome first for action-first pattern.
        """
        result: dict[str, Any] = {"action_outcome": self.action_outcome}

        if self.rejection_reason:
            result["rejection_reason"] = self.rejection_reason
        if self.recovery_action:
            result["recovery_action"] = self.recovery_action
        if self.field_corrections:
            result["field_corrections"] = self.field_corrections
        if self.missing_required:
            result["missing_required"] = self.missing_required
        if self.invalid_fields:
            result["invalid_fields"] = self.invalid_fields
        if self.error_count:
            result["error_count"] = self.error_count
        if self.errors:
            result["errors"] = self.errors

        return result

    def to_json(self) -> str:
        """Convert to JSON string for LLM feedback."""
        import json

        return json.dumps(self.to_dict(), indent=2)
