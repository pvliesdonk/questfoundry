"""Stage mutation appliers.

Each stage produces structured output that the runtime applies as graph
mutations. This module contains the logic for each stage's mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

import contextlib
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    format_entity_id,
    is_entity_id,
    parse_scoped_id,
    strip_scope_prefix,
)

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Display limits for error messages
_MAX_ERRORS_DISPLAY = 8
_MAX_AVAILABLE_DISPLAY = 5
_MAX_SIMILARITY_SUGGESTIONS = 3  # Max ranked suggestions in "Did you mean?" output

# Similarity thresholds for ID suggestions.
# These values were empirically tuned based on real failure cases:
# - 0.85: Very close matches (e.g., "hollow_archive" vs "the_hollow_archive" = 87.5%)
#   warrant a prescriptive "Use X instead" since we're confident it's correct.
# - 0.6: Moderate matches (e.g., "the_archive" vs "the_hollow_archive" = 75%)
#   get ranked suggestions to help without being overly prescriptive.
# - Below 0.6: Too dissimilar to suggest confidently; just show sorted list.
_HIGH_CONFIDENCE_THRESHOLD = 0.85  # Single prescriptive suggestion
_MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # Ranked suggestions with scores


def _sort_by_similarity(invalid_id: str, available: list[str]) -> list[tuple[str, float]]:
    """Sort available IDs by similarity to invalid_id, highest first.

    Uses SequenceMatcher ratio for fuzzy matching. Strips scope prefixes
    (e.g., "entity::") before comparison to handle both scoped and unscoped IDs.

    Performance: Reuses a single SequenceMatcher instance with set_seq2() for
    efficiency when comparing against many available IDs. Typical list sizes
    are 10-30 entities, so O(n) comparisons are acceptable.

    Args:
        invalid_id: The invalid ID that was provided.
        available: List of valid IDs to compare against.

    Returns:
        List of (id, score) tuples sorted by score descending.
    """

    invalid_raw_lower = strip_scope_prefix(invalid_id).lower()
    matcher = SequenceMatcher(a=invalid_raw_lower)
    scored = []
    for aid in available:
        matcher.set_seq2(strip_scope_prefix(aid).lower())
        scored.append((aid, matcher.ratio()))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def _format_available_with_suggestions(invalid_id: str, available: list[str]) -> str:
    """Format available IDs with similarity-based suggestions.

    Applies confidence-gated feedback:
    - >= 0.85: Single prescriptive suggestion ("Use X instead")
    - 0.6-0.85: Ranked list with scores ("Did you mean?")
    - < 0.6: Sorted list, no suggestions

    Args:
        invalid_id: The invalid ID that was provided.
        available: List of valid IDs.

    Returns:
        Formatted string with suggestions appropriate to confidence level.
    """
    if not available:
        return ""

    sorted_ids = _sort_by_similarity(invalid_id, available)
    if not sorted_ids:
        return ""

    best_id, best_score = sorted_ids[0]

    # High confidence: single prescriptive suggestion
    if best_score >= _HIGH_CONFIDENCE_THRESHOLD:
        return f"Use '{best_id}' instead."

    # Medium confidence: ranked suggestions
    if best_score >= _MEDIUM_CONFIDENCE_THRESHOLD:
        lines = ["Did you mean one of these?"]
        for sid, score in sorted_ids[:_MAX_SIMILARITY_SUGGESTIONS]:
            lines.append(f"      - {sid} ({int(score * 100)}%)")
        return "\n".join(lines)

    # Low confidence: sorted list only (most similar first)
    display = [sid for sid, _ in sorted_ids[:_MAX_AVAILABLE_DISPLAY]]
    suffix = "..." if len(available) > _MAX_AVAILABLE_DISPLAY else ""
    return f"Available IDs (most similar first): {', '.join(display)}{suffix}"


def _format_error_available(provided: str, available: list[str]) -> str | None:
    """Format available IDs for an error, using similarity if provided value exists.

    This is a shared helper for BrainstormMutationError and SeedMutationError to
    avoid code duplication (DRY principle).

    Args:
        provided: The invalid ID that was provided (may be empty).
        available: List of valid IDs.

    Returns:
        Formatted suggestion string, or None if no available IDs.
    """
    if not available:
        return None

    if provided:
        suggestion = _format_available_with_suggestions(provided, available)
        if suggestion:
            return suggestion

    # Fallback for errors without provided value
    avail = available[:_MAX_AVAILABLE_DISPLAY]
    suffix = "..." if len(available) > _MAX_AVAILABLE_DISPLAY else ""
    return f"Available: {', '.join(avail)}{suffix}"


# Error message patterns for categorization.
# Using constants makes the categorization explicit and testable.
# Future work: Replace string matching with structured error codes (see issue #216).
# These patterns MUST match the error messages produced by validate_seed_mutations().
_PATTERN_SEMANTIC_BRAINSTORM = "not in brainstorm"
_PATTERN_SEMANTIC_SEED = "not defined in seed"
_PATTERN_COMPLETENESS = "missing decision"


class SeedErrorCategory(Enum):
    """Categories of SEED validation errors for targeted retry strategies.

    Different error types require different recovery approaches:
    - INNER: Schema/type errors - retry with Pydantic feedback
    - SEMANTIC: Invalid ID references - retry with valid ID list
    - COMPLETENESS: Missing items - retry with manifest counts
    - FATAL: Reserved for unrecoverable errors (corruption, impossible states)
    """

    INNER = auto()  # Schema/type error in a single section
    SEMANTIC = auto()  # Invalid ID reference (phantom IDs)
    COMPLETENESS = auto()  # Missing entity/dilemma decisions
    CROSS_REFERENCE = (
        auto()
    )  # Cross-section ID mismatch (e.g. path answer_id not in dilemma explored)
    # FATAL is reserved for future use - e.g., graph corruption that requires
    # manual intervention. Currently no errors are classified as FATAL since
    # all known error types can be retried with appropriate feedback.
    FATAL = auto()


class MutationError(ValueError):
    """Error during mutation application."""

    pass


@dataclass
class BrainstormValidationError:
    """Semantic error for BRAINSTORM internal consistency.

    Attributes:
        field_path: Path to the invalid field (e.g., "dilemmas.0.central_entity_ids").
        issue: Description of what's wrong.
        available: List of valid IDs that could be used instead.
        provided: The value that was provided.
    """

    field_path: str
    issue: str
    available: list[str] = field(default_factory=list)
    provided: str = ""


class BrainstormMutationError(MutationError):
    """BRAINSTORM output failed internal consistency validation.

    This error contains structured validation errors that can be formatted
    as feedback for the LLM to retry with correct values.
    """

    def __init__(self, errors: list[BrainstormValidationError]) -> None:
        self.errors = errors
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format errors for exception message with similarity-based suggestions."""
        lines = ["BRAINSTORM has invalid internal references:"]
        for e in self.errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.field_path}: {e.issue}")
            suggestion = _format_error_available(e.provided, e.available)
            if suggestion:
                lines.append(f"    {suggestion}")
        if len(self.errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(self.errors) - _MAX_ERRORS_DISPLAY} more errors")
        lines.append("Use entity_id values from the entities list.")
        return "\n".join(lines)

    def to_feedback(self) -> str:
        """Format for LLM retry feedback.

        Returns:
            Human-readable error message for LLM to fix.
        """
        return self._format_message()


@dataclass
class SeedValidationError:
    """Semantic error with context for LLM feedback.

    Attributes:
        field_path: Path to the invalid field (e.g., "paths.0.dilemma_id").
        issue: Description of what's wrong.
        available: List of valid IDs that could be used instead.
        provided: The value that was provided.
        category: Error category set at creation time. When set, categorize_error()
            uses this directly instead of pattern matching on the issue string.
    """

    field_path: str
    issue: str
    available: list[str] = field(default_factory=list)
    provided: str = ""
    category: SeedErrorCategory | None = None


def categorize_error(error: SeedValidationError) -> SeedErrorCategory:
    """Categorize a SEED validation error for targeted retry strategy.

    Error categories determine how to recover:
    - SEMANTIC: Invalid ID reference → retry with valid ID list
    - COMPLETENESS: Missing decisions → retry with manifest counts
    - INNER: Everything else → retry with Pydantic feedback

    Uses the explicit category field when set (preferred), falling back to
    pattern matching on the issue string for backwards compatibility.

    Args:
        error: SeedValidationError to categorize.

    Returns:
        SeedErrorCategory indicating the error type.
    """
    # Prefer explicit category set at creation time
    if error.category is not None:
        return error.category

    # Fallback: pattern matching for errors without explicit category
    issue = error.issue.lower()

    if _PATTERN_SEMANTIC_BRAINSTORM in issue or _PATTERN_SEMANTIC_SEED in issue:
        return SeedErrorCategory.SEMANTIC

    if _PATTERN_COMPLETENESS in issue:
        return SeedErrorCategory.COMPLETENESS

    return SeedErrorCategory.INNER


def categorize_errors(
    errors: list[SeedValidationError],
) -> dict[SeedErrorCategory, list[SeedValidationError]]:
    """Group errors by category for targeted retry strategies.

    Args:
        errors: List of SeedValidationError objects.

    Returns:
        Dict mapping categories to their errors.
    """
    by_category: dict[SeedErrorCategory, list[SeedValidationError]] = {}
    for error in errors:
        category = categorize_error(error)
        by_category.setdefault(category, []).append(error)
    return by_category


class SeedMutationError(MutationError):
    """SEED mutation failed due to semantic validation errors.

    This error contains structured validation errors that can be formatted
    as feedback for the LLM to retry with correct values.
    """

    def __init__(self, errors: list[SeedValidationError]) -> None:
        self.errors = errors
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format errors for exception message with similarity-based suggestions."""
        lines = ["SEED has invalid cross-references:"]
        for e in self.errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.field_path}: {e.issue}")
            suggestion = _format_error_available(e.provided, e.available)
            if suggestion:
                lines.append(f"    {suggestion}")
        if len(self.errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(self.errors) - _MAX_ERRORS_DISPLAY} more errors")
        lines.append("Use EXACT IDs from BRAINSTORM.")
        return "\n".join(lines)

    def to_feedback(self) -> str:
        """Format for LLM retry feedback.

        Returns:
            Human-readable error message for LLM to fix.
        """
        return self._format_message()


class GrowErrorCategory(Enum):
    """Categories of GROW validation errors for targeted recovery.

    Different error types require different recovery approaches:
    - STRUCTURAL: DAG violations or missing commits beats
    - COMBINATORIAL: Too many arcs (exponential blowup)
    - REFERENCE: Invalid node/edge references
    - FATAL: Unrecoverable graph state
    """

    STRUCTURAL = auto()
    COMBINATORIAL = auto()
    REFERENCE = auto()
    FATAL = auto()


@dataclass
class GrowValidationError:
    """Semantic error for GROW phase validation.

    Attributes:
        field_path: Path to the invalid field (e.g., "arc.paths.0").
        issue: Description of what's wrong.
        available: List of valid IDs that could be used instead.
        provided: The value that was provided.
        category: Error category for targeted recovery.
    """

    field_path: str
    issue: str
    available: list[str] = field(default_factory=list)
    provided: str = ""
    category: GrowErrorCategory | None = None


class GrowMutationError(MutationError):
    """GROW phase failed due to structural or reference errors.

    This error contains structured validation errors with categories
    to guide error recovery strategies.
    """

    def __init__(self, errors: list[GrowValidationError]) -> None:
        self.errors = errors
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format errors for exception message."""
        lines = ["GROW has validation errors:"]
        for e in self.errors[:_MAX_ERRORS_DISPLAY]:
            category_label = f" [{e.category.name}]" if e.category else ""
            lines.append(f"  - {e.field_path}: {e.issue}{category_label}")
            suggestion = _format_error_available(e.provided, e.available)
            if suggestion:
                lines.append(f"    {suggestion}")
        if len(self.errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(self.errors) - _MAX_ERRORS_DISPLAY} more errors")
        return "\n".join(lines)

    def to_feedback(self) -> str:
        """Format for error reporting.

        Returns:
            Human-readable error message describing validation failures.
        """
        return self._format_message()


def _require_field(item: dict[str, Any], field: str, context: str) -> Any:
    """Require a field exists in a dict, raising clear error if missing.

    Args:
        item: Dictionary to check.
        field: Field name to require.
        context: Description for error message (e.g., "entity", "dilemma").

    Returns:
        The field value.

    Raises:
        MutationError: If field is missing.
    """
    if field not in item:
        raise MutationError(f"{context} missing required '{field}' field")
    return item[field]


def _normalize_id(provided_id: str, expected_scope: str) -> tuple[str, str | None]:
    """Normalize a potentially scoped ID for validation.

    Accepts both scoped and unscoped IDs. If scoped, validates the scope matches
    expected_scope. For entities, accepts category prefixes (character::, location::,
    object::, faction::) in addition to the generic "entity::" prefix.

    Args:
        provided_id: The ID from LLM output (may or may not have scope prefix).
        expected_scope: Expected scope type (e.g., "entity", "dilemma", "path").

    Returns:
        Tuple of (normalized_id, error_message).
        - If valid: (raw_id, None)
        - If wrong scope: (provided_id, error message describing mismatch)

    Examples:
        >>> _normalize_id("character::hero", "entity")
        ('hero', None)
        >>> _normalize_id("location::manor", "entity")
        ('manor', None)
        >>> _normalize_id("hero", "entity")
        ('hero', None)
        >>> _normalize_id("dilemma::hero", "entity")
        ('dilemma::hero', "Wrong scope prefix: expected entity category (character/location/object/faction), got 'dilemma::'")
    """
    if "::" in provided_id:
        scope, raw_id = parse_scoped_id(provided_id)
        # For entities, accept category prefixes (character::, location::, etc.)
        # as well as legacy "entity::" prefix
        if expected_scope == "entity":
            if scope in ENTITY_CATEGORIES or scope == "entity":
                return raw_id, None
            return (
                provided_id,
                f"Wrong scope prefix: expected entity category (character/location/object/faction), got '{scope}::'",
            )
        if scope != expected_scope:
            return (
                provided_id,
                f"Wrong scope prefix: expected '{expected_scope}::', got '{scope}::'",
            )
        return raw_id, None
    return provided_id, None


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from a dictionary for cleaner storage.

    Args:
        data: Dictionary to clean.

    Returns:
        New dictionary with None values removed.
    """
    return {k: v for k, v in data.items() if v is not None}


def _backfill_explored_from_paths(output: dict[str, Any]) -> None:
    """Backfill empty explored arrays from path answer_ids (in-place mutation).

    Handles cases where LLM serialized paths but left explored empty.
    The path's answer_id IS what was explored - if a path exists for
    an answer, that answer was necessarily explored.

    This runs BEFORE validation to fix data integrity issues in
    graphs where dilemmas have `explored: []` but paths exist
    with valid `answer_id` values.

    Args:
        output: SEED stage output dictionary (mutated in place).
    """
    # Build mapping: dilemma_id -> set of answer_ids from paths
    path_answers: dict[str, set[str]] = {}
    for path in output.get("paths", []):
        dilemma_id = strip_scope_prefix(path.get("dilemma_id", ""))
        answer_id = path.get("answer_id")
        if answer_id:
            path_answers.setdefault(dilemma_id, set()).add(answer_id)

    # Backfill empty explored arrays
    for dilemma in output.get("dilemmas", []):
        explored = dilemma.get("explored", [])
        if not explored:
            dilemma_id = strip_scope_prefix(dilemma.get("dilemma_id", ""))
            path_answer_ids = path_answers.get(dilemma_id, set())
            if path_answer_ids:
                dilemma["explored"] = sorted(path_answer_ids)


# Registry of stages with mutation handlers
# Note: GROW is not included because it modifies the graph directly during execution,
# not via post-stage mutation application.
_MUTATION_STAGES = frozenset({"dream", "brainstorm", "seed"})


def has_mutation_handler(stage: str) -> bool:
    """Check if a stage has a mutation handler.

    Use this to determine whether to call apply_mutations() for a stage.
    This is the single source of truth for which stages support graph mutations.

    Args:
        stage: Stage name to check.

    Returns:
        True if the stage has a mutation handler, False otherwise.
    """
    return stage in _MUTATION_STAGES


def apply_mutations(graph: Graph, stage: str, output: dict[str, Any]) -> None:
    """Apply stage output as graph mutations.

    Routes to the appropriate stage-specific mutation function.

    Args:
        graph: Graph to mutate.
        stage: Stage name (dream, brainstorm, seed, etc.).
        output: Stage output data.

    Raises:
        ValueError: If stage is unknown.
        MutationError: If output is malformed.
    """
    mutation_funcs = {
        "dream": apply_dream_mutations,
        "brainstorm": apply_brainstorm_mutations,
        "seed": apply_seed_mutations,
    }

    if stage not in mutation_funcs:
        raise ValueError(f"Unknown stage: {stage}")

    mutation_funcs[stage](graph, output)


def apply_dream_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply DREAM stage output to graph.

    Creates or replaces the "vision" node with the dream artifact data.

    Args:
        graph: Graph to mutate.
        output: DREAM stage output (DreamArtifact fields).
    """
    # Extract fields, ensuring we have the right structure
    # The output from DREAM stage is a DreamArtifact-like dict
    vision_data = {
        "type": "vision",
        "genre": output.get("genre"),
        "subgenre": output.get("subgenre"),
        "tone": output.get("tone", []),
        "themes": output.get("themes", []),
        "audience": output.get("audience"),
        "style_notes": output.get("style_notes"),
        "scope": output.get("scope"),
        "content_notes": output.get("content_notes"),
        # POV hints (optional, may be None)
        "pov_style": output.get("pov_style"),
        "protagonist_defined": output.get("protagonist_defined", False),
    }

    # Remove None values for cleaner storage
    vision_data = _clean_dict(vision_data)

    # Use upsert to allow re-running DREAM stage (replaces existing vision)
    graph.upsert_node("vision", vision_data)


def validate_brainstorm_mutations(output: dict[str, Any]) -> list[BrainstormValidationError]:
    """Validate BRAINSTORM output internal consistency.

    Checks that the output is self-consistent (no graph needed):
    1. All central_entity_ids in dilemmas exist in entities list
    2. All answer IDs within a dilemma are unique
    3. Each dilemma has exactly one is_default_path=true answer

    Args:
        output: BRAINSTORM stage output (entities, dilemmas).

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[BrainstormValidationError] = []

    # Validate entities and build ID set in single pass
    entities = output.get("entities", [])
    entity_ids: set[str] = set()
    for i, entity in enumerate(entities):
        entity_id = entity.get("entity_id")
        if not entity_id:  # None or empty string
            errors.append(
                BrainstormValidationError(
                    field_path=f"entities.{i}.entity_id",
                    issue="Entity has missing or empty entity_id",
                    available=[],
                    provided=repr(entity_id),
                )
            )
        else:
            entity_ids.add(strip_scope_prefix(entity_id))
    sorted_entity_ids = sorted(entity_ids)

    # Validate each dilemma
    for i, dilemma in enumerate(output.get("dilemmas", [])):
        raw_dilemma_id = dilemma.get("dilemma_id", f"<index {i}>")
        dilemma_id = strip_scope_prefix(raw_dilemma_id)

        # 1. Check central_entity_ids reference valid entities
        for eid in dilemma.get("central_entity_ids", []):
            normalized_eid = strip_scope_prefix(eid)
            if normalized_eid not in entity_ids:
                errors.append(
                    BrainstormValidationError(
                        field_path=f"dilemmas.{i}.central_entity_ids",
                        issue=f"Entity '{normalized_eid}' not in entities list",
                        available=sorted_entity_ids,
                        provided=eid,
                    )
                )

        # 2. Check answer IDs are unique within this dilemma
        answers = dilemma.get("answers", [])
        answer_ids = [strip_scope_prefix(a_id) for a in answers if (a_id := a.get("answer_id"))]
        answer_id_counts = Counter(answer_ids)
        for answer_id, count in answer_id_counts.items():
            if count > 1:
                errors.append(
                    BrainstormValidationError(
                        field_path=f"dilemmas.{i}.answers",
                        issue=f"Duplicate answer_id '{answer_id}' appears {count} times in dilemma '{dilemma_id}'",
                        available=[],
                        provided=answer_id,
                    )
                )

        # 3. Check exactly one answer has is_default_path=True
        # (missing or False both count as non-default - Pydantic validation ensures
        # the field exists for valid BrainstormOutput, this handles edge cases)
        default_count = sum(1 for a in answers if a.get("is_default_path"))
        if default_count != 1:
            issue = (
                f"No answer has is_default_path=true in dilemma '{dilemma_id}'"
                if default_count == 0
                else f"Multiple answers have is_default_path=true in dilemma '{dilemma_id}'"
            )
            errors.append(
                BrainstormValidationError(
                    field_path=f"dilemmas.{i}.answers",
                    issue=issue,
                    available=[],
                    provided=f"found {default_count} defaults",
                )
            )

    # 4. Validate is_protagonist constraints
    protagonist_count = 0
    protagonist_indices: list[int] = []
    for i, entity in enumerate(entities):
        if entity.get("is_protagonist"):
            protagonist_count += 1
            protagonist_indices.append(i)
            # is_protagonist only valid for character entities
            entity_category = entity.get("entity_category", "")
            if entity_category != "character":
                errors.append(
                    BrainstormValidationError(
                        field_path=f"entities.{i}.is_protagonist",
                        issue=f"is_protagonist=true only valid for character entities, not '{entity_category}'",
                        available=["character"],
                        provided=entity_category,
                    )
                )

    # At most one protagonist allowed
    if protagonist_count > 1:
        errors.append(
            BrainstormValidationError(
                field_path="entities",
                issue=f"Multiple protagonists defined ({protagonist_count} at indices {protagonist_indices}). Only one character can be the protagonist.",
                available=[],
                provided=str(protagonist_count),
            )
        )

    return errors


def _prefix_id(node_type: str, raw_id: str) -> str:
    """Prefix a raw ID with its node type for namespace isolation.

    This allows entities and dilemmas to have the same raw ID without collision.
    E.g., both can use "cipher_journal" -> "dilemma::cipher_journal"

    This function is idempotent - if the ID already has the correct prefix,
    it returns it unchanged. If it has a different prefix or multiple prefixes,
    the raw part is extracted and re-prefixed.

    Note: For entity IDs, use _prefix_entity_id() which uses category-based
    prefixes (character::, location::, etc.) instead of generic "entity::".

    Args:
        node_type: Node type prefix (dilemma, path, beat, etc.)
        raw_id: Raw ID from LLM output (may already be prefixed).

    Returns:
        Prefixed ID in format "type::raw_id".
    """
    # Strip any existing prefixes to get the raw ID
    # This handles: "the_detective", "dilemma::the_detective", etc.
    if "::" in raw_id:
        raw_id = raw_id.rsplit("::", 1)[-1]

    return f"{node_type}::{raw_id}"


def _prefix_entity_id(category: str, raw_id: str) -> str:
    """Prefix an entity ID with its category for semantic clarity.

    Entity IDs use category as prefix (character::pim, location::manor)
    rather than generic "entity::" prefix. This avoids LLM confusion with
    abbreviated prefixes like "char_" that resemble programming types.

    Args:
        category: Entity category (character, location, object, faction).
        raw_id: Raw entity ID from LLM output (may already be prefixed).

    Returns:
        Category-prefixed ID (e.g., "character::pim").

    Raises:
        ValueError: If category is not valid.
    """
    # Strip any existing prefix (handles legacy "entity::" or double prefixes)
    if "::" in raw_id:
        raw_id = raw_id.rsplit("::", 1)[-1]

    return format_entity_id(category, raw_id)


def _resolve_entity_ref(graph: Graph, entity_ref: str) -> str:
    """Resolve an entity reference to its full prefixed ID.

    Handles various input formats:
    - Raw ID: "pim" -> looks up entity, returns "character::pim"
    - Legacy format: "entity::pim" -> returns as-is if exists in graph
    - Category format: "character::pim" -> returns as-is

    For backwards compatibility, accepts both legacy "entity::" and new
    category-based prefixes. Returns the ID as it exists in the graph.

    Args:
        graph: Graph containing entity nodes.
        entity_ref: Entity reference (raw ID or scoped ID).

    Returns:
        Entity ID as it exists in the graph.

    Raises:
        ValueError: If entity not found in graph.
    """
    # If already has valid entity category prefix and exists, return as-is
    if is_entity_id(entity_ref) and graph.has_node(entity_ref):
        return entity_ref

    # Check if it's a legacy entity:: format that exists in graph
    if entity_ref.startswith("entity::") and graph.has_node(entity_ref):
        return entity_ref

    # Extract raw ID (handles "entity::pim" -> "pim")
    raw_id = strip_scope_prefix(entity_ref)

    # Try to find entity in graph by checking all category prefixes
    for category in ENTITY_CATEGORIES:
        candidate_id = f"{category}::{raw_id}"
        if graph.has_node(candidate_id):
            return candidate_id

    # Also try legacy entity:: prefix for backwards compatibility
    legacy_id = f"entity::{raw_id}"
    if graph.has_node(legacy_id):
        return legacy_id

    # Entity not found - return with placeholder prefix for error reporting
    # The validation layer will catch this
    raise ValueError(f"Entity '{entity_ref}' not found in graph")


def apply_brainstorm_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply BRAINSTORM stage output to graph.

    Creates entity nodes and dilemma nodes with their answers.
    Dilemmas are linked to their answers via has_answer edges.

    Node IDs are prefixed by type to avoid collisions:
    - character::name, location::name, etc. (entities use category prefix)
    - dilemma::raw_id
    - dilemma::dilemma_id::alt::answer_id (for answers)

    Args:
        graph: Graph to mutate.
        output: BRAINSTORM stage output (entities, dilemmas).

    Raises:
        MutationError: If entities or dilemmas are missing required id fields.
    """
    # Add entities (must be done first so dilemmas can reference them)
    for i, entity in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity, "entity_id", f"Entity at index {i}")
        category = entity.get("entity_category")
        if not category:
            raise MutationError(f"Entity at index {i} missing entity_category")
        if category not in ENTITY_CATEGORIES:
            raise MutationError(
                f"Entity at index {i} has invalid category '{category}'. "
                f"Must be one of: {', '.join(sorted(ENTITY_CATEGORIES))}"
            )
        entity_id = _prefix_entity_id(category, raw_id)
        raw_id = strip_scope_prefix(entity_id)
        node_data = {
            "type": "entity",
            "raw_id": raw_id,
            "category": category,  # Store category for easy access
            "entity_type": category,  # Backwards compat: some code still uses entity_type
            "concept": entity.get("concept"),
            "notes": entity.get("notes"),
            "disposition": "proposed",  # All entities start as proposed
        }
        # Remove None values
        node_data = _clean_dict(node_data)
        graph.create_node(entity_id, node_data)

    # Add dilemmas with answers
    for i, dilemma in enumerate(output.get("dilemmas", [])):
        raw_id = dilemma.get("dilemma_id")
        if not raw_id:
            raise MutationError(f"Dilemma at index {i} missing dilemma_id")
        dilemma_node_id = _prefix_id("dilemma", raw_id)
        raw_id = strip_scope_prefix(dilemma_node_id)

        # Resolve entity references in central_entity_ids list
        raw_central_entities = dilemma.get("central_entity_ids", [])
        prefixed_central_entities = []
        for eid in raw_central_entities:
            try:
                prefixed_central_entities.append(_resolve_entity_ref(graph, eid))
            except ValueError:
                # Entity not found - keep raw ID for error reporting later
                prefixed_central_entities.append(eid)

        # Create dilemma node
        dilemma_data = {
            "type": "dilemma",
            "raw_id": raw_id,
            "question": dilemma.get("question"),
            "central_entity_ids": prefixed_central_entities,
            "why_it_matters": dilemma.get("why_it_matters"),
        }
        dilemma_data = _clean_dict(dilemma_data)
        graph.create_node(dilemma_node_id, dilemma_data)

        # Create answer nodes and edges
        for j, answer in enumerate(dilemma.get("answers", [])):
            answer_local_id = answer.get("answer_id")
            if not answer_local_id:
                raise MutationError(f"Answer at index {j} in dilemma '{raw_id}' missing answer_id")
            answer_local_id = strip_scope_prefix(_prefix_id("answer", answer_local_id))
            # Answer ID format: dilemma::dilemma_raw_id::alt::answer_local_id
            answer_node_id = f"{dilemma_node_id}::alt::{answer_local_id}"
            answer_data = {
                "type": "answer",
                "raw_id": answer_local_id,
                "description": answer.get("description"),
                "is_default_path": answer.get("is_default_path", False),
            }
            answer_data = _clean_dict(answer_data)
            graph.create_node(answer_node_id, answer_data)
            graph.add_edge("has_answer", dilemma_node_id, answer_node_id)


def _cross_type_hint(
    raw_id: str,
    expected_type: str,
    valid_entity_ids: set[str],
    valid_dilemma_ids: set[str],
    seed_path_ids: set[str],
    entity_types: dict[str, str],
) -> str:
    """Return type-aware message when ID exists as different type.

    When an ID isn't found in the expected set, checks whether it exists
    in a different type's set and provides a helpful cross-type hint.
    This prevents misleading "not in BRAINSTORM" feedback when the ID
    actually IS in brainstorm but as a different type (e.g., a faction
    entity used as a dilemma_id).

    Args:
        raw_id: The normalized ID that wasn't found.
        expected_type: What type was expected ("entity", "dilemma", "path").
        valid_entity_ids: Set of valid entity raw IDs from brainstorm.
        valid_dilemma_ids: Set of valid dilemma raw IDs from brainstorm.
        seed_path_ids: Set of path raw IDs from SEED output.
        entity_types: Mapping of entity raw_id to entity_type (character, faction, etc.).

    Returns:
        Descriptive error message indicating the type mismatch or fallback.
    """
    if expected_type == "dilemma" and raw_id in valid_entity_ids:
        etype = entity_types.get(raw_id, "entity")
        return (
            f"'{raw_id}' is an entity ({etype}), not a dilemma. "
            "Dilemma IDs follow the pattern subject_X_or_Y"
        )
    if expected_type == "dilemma" and raw_id in seed_path_ids:
        return f"'{raw_id}' is a path ID, not a dilemma. Dilemma IDs are longer binary questions"
    if expected_type == "entity" and raw_id in valid_dilemma_ids:
        return f"'{raw_id}' is a dilemma ID, not an entity"
    if expected_type == "entity" and raw_id in seed_path_ids:
        return f"'{raw_id}' is a path ID, not an entity"
    if expected_type == "path" and raw_id in valid_entity_ids:
        etype = entity_types.get(raw_id, "entity")
        return f"'{raw_id}' is an entity ({etype}), not a path"
    if expected_type == "path" and raw_id in valid_dilemma_ids:
        return f"'{raw_id}' is a dilemma ID, not a path"
    if expected_type == "path":
        return f"Path '{raw_id}' not defined in SEED paths"
    return f"'{raw_id}' not in BRAINSTORM"


def _validate_id(
    raw_id: str | None,
    expected_scope: str,
    valid_ids: set[str],
    field_path: str,
    errors: list[SeedValidationError],
    sorted_available: list[str],
    category: SeedErrorCategory = SeedErrorCategory.SEMANTIC,
    *,
    cross_type_sets: tuple[set[str], set[str], set[str], dict[str, str]] | None = None,
) -> str | None:
    """Validate and normalize an ID, appending errors if invalid.

    Encapsulates the repeated normalize → scope-check → existence-check pattern.

    Args:
        raw_id: The ID from LLM output (may be None, scoped, or bare).
        expected_scope: Expected scope type (e.g., "entity", "dilemma", "path").
        valid_ids: Set of valid raw IDs for the expected type.
        field_path: Field path for error reporting.
        errors: Error list to append to.
        sorted_available: Pre-sorted valid IDs for error messages.
        category: Error category to set on created errors.
        cross_type_sets: Optional tuple of (valid_entity_ids, valid_dilemma_ids,
            seed_path_ids, entity_types) for cross-type hint messages.
            If None, uses generic "not in BRAINSTORM/SEED" messages.

    Returns:
        Normalized ID if valid, None if invalid or missing.
    """
    if not raw_id:
        return None

    normalized_id, scope_error = _normalize_id(raw_id, expected_scope)
    if scope_error:
        errors.append(
            SeedValidationError(
                field_path=field_path,
                issue=scope_error,
                available=sorted_available,
                provided=raw_id,
                category=SeedErrorCategory.INNER,
            )
        )
        return None

    if normalized_id not in valid_ids:
        if cross_type_sets:
            entity_ids, dilemma_ids, path_ids, entity_types = cross_type_sets
            issue = _cross_type_hint(
                normalized_id, expected_scope, entity_ids, dilemma_ids, path_ids, entity_types
            )
        elif expected_scope == "path":
            issue = f"Path '{normalized_id}' not defined in SEED paths"
        else:
            display_type = {"dilemma": "Dilemma", "path": "Path"}.get(
                expected_scope, expected_scope.title()
            )
            issue = f"{display_type} '{normalized_id}' not in BRAINSTORM"
        errors.append(
            SeedValidationError(
                field_path=field_path,
                issue=issue,
                available=sorted_available,
                provided=raw_id,
                category=category,
            )
        )
        return None

    return normalized_id


def validate_seed_mutations(graph: Graph, output: dict[str, Any]) -> list[SeedValidationError]:
    """Validate SEED output references against BRAINSTORM data in graph.

    Checks semantic validity (not just structural):
    1. Entity IDs in decisions exist in graph
    2. Dilemma IDs in decisions exist in graph
    3. Path dilemma_ids reference valid dilemmas
    4. Path answer_ids exist for their dilemma
    5. Beat entity references exist
    6. Beat path references exist (within SEED output)
    7. Consequence path references exist (within SEED output)
    8. Dilemma impacts reference valid dilemmas
    9. All BRAINSTORM entities have decisions (completeness)
    10. All BRAINSTORM dilemmas have decisions (completeness)
    11. All dilemmas have at least one path (completeness)
    12. Each beat references its path's parent dilemma in dilemma_impacts
    13. Each path has at least one beat with effect="commits" for its dilemma

    Args:
        graph: Graph containing BRAINSTORM data (entities, dilemmas, answers).
        output: SEED stage output to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[SeedValidationError] = []

    # Build lookup sets from graph (BRAINSTORM data)
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    # Raw IDs (unprefixed) for validation - what LLM uses
    # Type annotation ensures mypy knows these are str sets (filter guarantees non-None)
    valid_entity_ids: set[str] = {n["raw_id"] for n in entity_nodes.values() if n.get("raw_id")}
    valid_dilemma_ids: set[str] = {n["raw_id"] for n in dilemma_nodes.values() if n.get("raw_id")}

    # Entity type lookup for informative error messages
    entity_types: dict[str, str] = {
        n["raw_id"]: n.get("entity_type", "entity")
        for n in entity_nodes.values()
        if n.get("raw_id")
    }

    # Build answer lookup: dilemma_raw_id -> set of answer_raw_ids
    # Use O(edges) algorithm instead of O(dilemmas * edges)
    answers_by_dilemma: dict[str, set[str]] = {}
    dilemma_id_to_raw: dict[str, str] = {
        did: ddata["raw_id"] for did, ddata in dilemma_nodes.items() if ddata.get("raw_id")
    }
    for edge in graph.get_edges(edge_type="has_answer"):
        from_node_id = edge.get("from", "")
        raw_dilemma_id = dilemma_id_to_raw.get(from_node_id)
        if not raw_dilemma_id:
            continue
        answer_node = graph.get_node(edge.get("to", ""))
        if answer_node and answer_node.get("raw_id"):
            answers_by_dilemma.setdefault(raw_dilemma_id, set()).add(answer_node["raw_id"])

    # Extract path IDs from SEED output (for internal references)
    # Normalize IDs to handle both scoped (path::foo) and bare (foo) formats
    seed_path_ids: set[str] = {
        _normalize_id(p["path_id"], "path")[0] for p in output.get("paths", []) if p.get("path_id")
    }

    # Pre-sort available IDs once (used in error messages)
    sorted_entity_ids = sorted(valid_entity_ids)
    sorted_dilemma_ids = sorted(valid_dilemma_ids)
    sorted_path_ids = sorted(seed_path_ids)

    # Cross-type sets for helpful error messages (entity used as dilemma, etc.)
    cross_type_sets = (valid_entity_ids, valid_dilemma_ids, seed_path_ids, entity_types)

    # 0. Check for duplicate IDs in output (prevents LLM from outputting same item twice)
    for field_path, id_field, scope, display_name in [
        ("entities", "entity_id", "entity", "entity"),
        ("dilemmas", "dilemma_id", "dilemma", "dilemma"),
        ("paths", "path_id", "path", "path"),
        ("consequences", "consequence_id", "consequence", "consequence"),
        ("initial_beats", "beat_id", "beat", "beat"),
    ]:
        id_counts: Counter[str] = Counter()
        for decision in output.get(field_path, []):
            raw_id = decision.get(id_field)
            if raw_id:
                normalized_id, _ = _normalize_id(raw_id, scope)
                id_counts[normalized_id] += 1
        for dup_id, count in id_counts.items():
            if count > 1:
                errors.append(
                    SeedValidationError(
                        field_path=field_path,
                        issue=f"Duplicate {display_name}_id '{dup_id}' appears {count} times - each {display_name} should have exactly one decision",
                        available=[],
                        provided=dup_id,
                        category=SeedErrorCategory.INNER,
                    )
                )

    # 1. Validate entity decisions
    for i, decision in enumerate(output.get("entities", [])):
        _validate_id(
            decision.get("entity_id"),
            "entity",
            valid_entity_ids,
            f"entities.{i}.entity_id",
            errors,
            sorted_entity_ids,
        )

    # 2. Validate dilemma decisions
    for i, decision in enumerate(output.get("dilemmas", [])):
        _validate_id(
            decision.get("dilemma_id"),
            "dilemma",
            valid_dilemma_ids,
            f"dilemmas.{i}.dilemma_id",
            errors,
            sorted_dilemma_ids,
        )

    # 3-4. Validate paths
    for i, path in enumerate(output.get("paths", [])):
        raw_path_dilemma_id = path.get("dilemma_id")
        raw_answer_id = path.get("answer_id")

        path_dilemma_id = _validate_id(
            raw_path_dilemma_id,
            "dilemma",
            valid_dilemma_ids,
            f"paths.{i}.dilemma_id",
            errors,
            sorted_dilemma_ids,
        )

        # Check answer (report even if dilemma was also invalid, so LLM gets all feedback)
        if raw_path_dilemma_id and raw_answer_id:
            # Need the normalized dilemma ID even if invalid for the answer check
            norm_did = (
                _normalize_id(raw_path_dilemma_id, "dilemma")[0] if raw_path_dilemma_id else None
            )
            if norm_did:
                valid_answers = answers_by_dilemma.get(norm_did, set())
                if raw_answer_id not in valid_answers:
                    errors.append(
                        SeedValidationError(
                            field_path=f"paths.{i}.answer_id",
                            issue=f"Answer '{raw_answer_id}' not in dilemma '{norm_did}'",
                            available=sorted(valid_answers) if path_dilemma_id else [],
                            provided=raw_answer_id,
                            category=SeedErrorCategory.SEMANTIC,
                        )
                    )

        # 4b. Validate pov_character references a valid retained entity
        pov_char = path.get("pov_character")
        if pov_char:
            normalized_pov, scope_error = _normalize_id(pov_char, "entity")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"paths.{i}.pov_character",
                        issue=scope_error,
                        available=sorted_entity_ids,
                        provided=pov_char,
                        category=SeedErrorCategory.INNER,
                    )
                )
            elif normalized_pov not in valid_entity_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"paths.{i}.pov_character",
                        issue=f"POV character '{normalized_pov}' not found in entities",
                        available=sorted_entity_ids,
                        provided=pov_char,
                        category=SeedErrorCategory.SEMANTIC,
                    )
                )
            else:
                # Validate pov_character is a character entity (not location/object/faction)
                pov_entity_type = entity_types.get(normalized_pov, "")
                if pov_entity_type != "character":
                    # Get list of character entities for helpful error message
                    character_ids = sorted(
                        eid for eid, etype in entity_types.items() if etype == "character"
                    )
                    errors.append(
                        SeedValidationError(
                            field_path=f"paths.{i}.pov_character",
                            issue=f"POV character must be a character entity, not '{pov_entity_type}'",
                            available=character_ids,
                            provided=pov_char,
                            category=SeedErrorCategory.SEMANTIC,
                        )
                    )

    # 5-8. Validate beats (single pass over initial_beats)
    # Build cut-entity set for disposition checks
    cut_entity_ids: set[str] = {
        _normalize_id(d["entity_id"], "entity")[0]
        for d in output.get("entities", [])
        if d.get("disposition") == "cut" and d.get("entity_id")
    }

    for i, beat in enumerate(output.get("initial_beats", [])):
        # 5a. Entity references (entities, location, location_alternatives)
        for raw_entity_id in beat.get("entities", []):
            _validate_id(
                raw_entity_id,
                "entity",
                valid_entity_ids,
                f"initial_beats.{i}.entities",
                errors,
                sorted_entity_ids,
                cross_type_sets=cross_type_sets,
            )

        _validate_id(
            beat.get("location"),
            "entity",
            valid_entity_ids,
            f"initial_beats.{i}.location",
            errors,
            sorted_entity_ids,
            cross_type_sets=cross_type_sets,
        )

        for raw_loc_alt in beat.get("location_alternatives", []):
            _validate_id(
                raw_loc_alt,
                "entity",
                valid_entity_ids,
                f"initial_beats.{i}.location_alternatives",
                errors,
                sorted_entity_ids,
                cross_type_sets=cross_type_sets,
            )

        # 5b. Cut-entity check
        entity_references = [
            *[(eid, "entities") for eid in beat.get("entities", [])],
            (beat.get("location"), "location"),
            *[(alt, "location_alternatives") for alt in beat.get("location_alternatives", [])],
        ]
        for raw_id, field_name in entity_references:
            if not raw_id:
                continue
            entity_id, _ = _normalize_id(raw_id, "entity")
            if entity_id in cut_entity_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"initial_beats.{i}.{field_name}",
                        issue=f"Entity '{entity_id}' has disposition 'cut' but is referenced in beat",
                        available=[],
                        provided=raw_id,
                        category=SeedErrorCategory.SEMANTIC,
                    )
                )

        # 6. Path references
        for raw_path_id in beat.get("paths", []):
            _validate_id(
                raw_path_id,
                "path",
                seed_path_ids,
                f"initial_beats.{i}.paths",
                errors,
                sorted_path_ids,
                cross_type_sets=cross_type_sets,
            )

        # 8. Dilemma impacts
        for j, impact in enumerate(beat.get("dilemma_impacts", [])):
            _validate_id(
                impact.get("dilemma_id"),
                "dilemma",
                valid_dilemma_ids,
                f"initial_beats.{i}.dilemma_impacts.{j}.dilemma_id",
                errors,
                sorted_dilemma_ids,
                cross_type_sets=cross_type_sets,
            )

    # 7. Validate consequence path references (internal to SEED)
    for i, consequence in enumerate(output.get("consequences", [])):
        _validate_id(
            consequence.get("path_id"),
            "path",
            seed_path_ids,
            f"consequences.{i}.path_id",
            errors,
            sorted_path_ids,
        )

    # 9 & 10. Check completeness: all BRAINSTORM entities and dilemmas should have decisions
    completeness_checks = [
        ("entities", "entity", "entity", valid_entity_ids),
        ("dilemmas", "dilemma", "dilemma", valid_dilemma_ids),
    ]
    for field_path, storage_type, display_type, valid_ids in completeness_checks:
        id_field = f"{storage_type}_id"
        # Normalize IDs from output to handle scoped IDs (entity::hero -> hero)
        # Only count IDs with correct scope (or unscoped) toward completeness
        decided_ids: set[str] = set()
        for decision in output.get(field_path, []):
            raw_id = decision.get(id_field)
            if raw_id:
                normalized_id, scope_error = _normalize_id(raw_id, storage_type)
                if not scope_error:
                    decided_ids.add(normalized_id)

        missing_ids = valid_ids - decided_ids
        for item_id in sorted(missing_ids):
            # For entities, include type (character/location/object/faction) for clarity
            if display_type == "entity":
                entity_type_name = entity_types.get(item_id, "entity")
                issue_msg = f"Missing decision for {entity_type_name} '{item_id}'"
            else:
                issue_msg = f"Missing decision for {display_type} '{item_id}'"
            errors.append(
                SeedValidationError(
                    field_path=field_path,
                    issue=issue_msg,
                    available=[],
                    provided="",
                    category=SeedErrorCategory.COMPLETENESS,
                )
            )

    # 11. Check completeness: all dilemmas should have at least one path
    dilemmas_with_paths: set[str] = set()
    path_dilemma_map: dict[str, str] = {}  # path_id -> dilemma_id (normalized)
    dilemma_path_counts: dict[str, int] = {}  # dilemma_id -> count of paths
    for path in output.get("paths", []):
        raw_dilemma_id = path.get("dilemma_id")
        raw_path_id = path.get("path_id")
        if raw_dilemma_id:
            normalized_did, scope_error = _normalize_id(raw_dilemma_id, "dilemma")
            if not scope_error:
                dilemmas_with_paths.add(normalized_did)
                dilemma_path_counts[normalized_did] = dilemma_path_counts.get(normalized_did, 0) + 1
                if raw_path_id:
                    normalized_pid, _ = _normalize_id(raw_path_id, "path")
                    path_dilemma_map[normalized_pid] = normalized_did

    dilemmas_without_paths = valid_dilemma_ids - dilemmas_with_paths
    for dilemma_id in sorted(dilemmas_without_paths):
        errors.append(
            SeedValidationError(
                field_path="paths",
                issue=(
                    f"Dilemma '{dilemma_id}' has no path. "
                    f"Create at least one path exploring this dilemma."
                ),
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            )
        )

    # 11b. Check each explored answer has a path
    # For each dilemma decision, verify path count matches explored count
    for dilemma_decision in output.get("dilemmas", []):
        raw_did = dilemma_decision.get("dilemma_id")
        if not raw_did:
            continue
        normalized_did, scope_error = _normalize_id(raw_did, "dilemma")
        if scope_error:
            continue

        explored = dilemma_decision.get("explored", [])
        path_count = dilemma_path_counts.get(normalized_did, 0)

        if len(explored) > path_count:
            missing_count = len(explored) - path_count
            errors.append(
                SeedValidationError(
                    field_path="paths",
                    issue=(
                        f"Dilemma '{normalized_did}' has {len(explored)} explored answers "
                        f"but only {path_count} path(s). "
                        f"Create {missing_count} more path(s) - one for EACH explored answer."
                    ),
                    available=explored,
                    provided=str(path_count),
                    category=SeedErrorCategory.COMPLETENESS,
                )
            )

    # 11c. Check each path's answer_id is in its dilemma's explored list
    # This catches data integrity issues where paths exist but explored is empty/mismatched
    for i, path in enumerate(output.get("paths", [])):
        raw_did = path.get("dilemma_id")
        raw_answer_id = path.get("answer_id")
        if not raw_did or not raw_answer_id:
            continue

        normalized_did, _ = _normalize_id(raw_did, "dilemma")

        # Find the corresponding dilemma decision
        for dilemma_decision in output.get("dilemmas", []):
            decision_did = strip_scope_prefix(dilemma_decision.get("dilemma_id", ""))
            if decision_did == normalized_did:
                explored = dilemma_decision.get("explored", [])
                if raw_answer_id not in explored:
                    errors.append(
                        SeedValidationError(
                            field_path=f"paths.{i}.answer_id",
                            issue=(
                                f"Path answer '{raw_answer_id}' is not in dilemma "
                                f"'{normalized_did}' explored list: {explored}"
                            ),
                            available=explored,
                            provided=raw_answer_id,
                            category=SeedErrorCategory.CROSS_REFERENCE,
                        )
                    )
                break

    # 11d. Check default (canonical) answer is in explored, not unexplored
    # LLMs sometimes invert the buckets, putting the default answer in unexplored.
    # This guardrail catches the inversion early so the serialize loop can retry.
    for dilemma_decision in output.get("dilemmas", []):
        raw_did = dilemma_decision.get("dilemma_id")
        if not raw_did:
            continue
        normalized_did, scope_error = _normalize_id(raw_did, "dilemma")
        if scope_error:
            continue

        explored = dilemma_decision.get("explored", [])
        unexplored = dilemma_decision.get("unexplored", [])
        if not unexplored:
            continue  # Nothing in unexplored, no inversion possible

        # Look up which answer is the default from the graph
        prefixed_did = f"dilemma::{normalized_did}"
        dilemma_node = graph.get_node(prefixed_did)
        if not dilemma_node:
            continue

        # Find the default answer among the dilemma's alternatives
        alt_edges = graph.get_edges(from_id=prefixed_did, edge_type="has_answer")
        for edge in alt_edges:
            alt_node = graph.get_node(edge["to"])
            if alt_node and alt_node.get("is_default_path"):
                default_answer_id = alt_node.get("raw_id", "")
                if default_answer_id in unexplored and default_answer_id not in explored:
                    errors.append(
                        SeedValidationError(
                            field_path="dilemmas",
                            issue=(
                                f"Dilemma '{normalized_did}': default answer "
                                f"'{default_answer_id}' is in unexplored but MUST be "
                                f"in explored. The default/canonical answer is always "
                                f"explored. Move it from unexplored to explored."
                            ),
                            available=explored,
                            provided=default_answer_id,
                            category=SeedErrorCategory.CROSS_REFERENCE,
                        )
                    )
                break  # Only one default per dilemma

    # NOTE: Arc count validation removed - now handled by runtime pruning (over-generate-and-select)
    # See seed_pruning.py for the new approach: LLM generates freely, runtime selects best dilemmas

    # 12. Check beats reference their path's parent dilemma
    # 13. Check each path has at least one commits beat for its dilemma
    paths_with_commits: set[str] = set()  # path_ids that have a commits beat
    for i, beat in enumerate(output.get("initial_beats", [])):
        beat_path_ids: list[str] = []
        for raw_path_id in beat.get("paths", []):
            normalized_pid, _ = _normalize_id(raw_path_id, "path")
            beat_path_ids.append(normalized_pid)

        # Collect normalized dilemma_ids referenced in this beat's impacts
        beat_impact_dilemmas: set[str] = set()
        beat_impact_commits_dilemmas: set[str] = set()
        for impact in beat.get("dilemma_impacts", []):
            raw_did = impact.get("dilemma_id")
            if raw_did:
                normalized_did, _ = _normalize_id(raw_did, "dilemma")
                beat_impact_dilemmas.add(normalized_did)
                if impact.get("effect") == "commits":
                    beat_impact_commits_dilemmas.add(normalized_did)

        # For each path this beat belongs to, check it references the parent dilemma
        for path_id in beat_path_ids:
            expected_dilemma = path_dilemma_map.get(path_id)
            if not expected_dilemma:
                continue  # Path not found in output; already caught by check 6

            if expected_dilemma not in beat_impact_dilemmas:
                beat_id = beat.get("beat_id", f"beat_{i}")
                errors.append(
                    SeedValidationError(
                        field_path=f"initial_beats.{i}.dilemma_impacts",
                        issue=(
                            f"Beat '{beat_id}' belongs to path '{path_id}' "
                            f"but does not reference its parent dilemma "
                            f"'{expected_dilemma}' in dilemma_impacts. "
                            f"Each beat must impact its own path's dilemma."
                        ),
                        available=[expected_dilemma],
                        provided=", ".join(sorted(beat_impact_dilemmas)) or "(none)",
                        category=SeedErrorCategory.SEMANTIC,
                    )
                )

            # Track commits beats per path
            if expected_dilemma in beat_impact_commits_dilemmas:
                paths_with_commits.add(path_id)

    # Report paths missing commits beats
    for path_id in sorted(path_dilemma_map.keys()):
        if path_id not in paths_with_commits:
            expected_dilemma = path_dilemma_map[path_id]
            errors.append(
                SeedValidationError(
                    field_path=f"paths.{path_id}",
                    issue=(
                        f"Path '{path_id}' has no beat with effect='commits' "
                        f"for its parent dilemma '{expected_dilemma}'. "
                        f"Each path must have at least one beat that commits "
                        f"its dilemma resolution."
                    ),
                    available=[expected_dilemma],
                    provided="",
                    category=SeedErrorCategory.COMPLETENESS,
                )
            )

    return errors


def apply_seed_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply SEED stage output to graph.

    First validates all cross-references, then applies mutations.

    Updates entity dispositions, creates paths from explored dilemmas,
    creates consequences, and creates initial beats.

    Node IDs are prefixed by type to match BRAINSTORM's namespacing:
    - path::raw_id
    - consequence::raw_id
    - beat::raw_id

    References to entities/dilemmas from LLM output are prefixed for lookup.

    Args:
        graph: Graph to mutate.
        output: SEED stage output (SeedOutput fields).

    Raises:
        SeedMutationError: If semantic validation fails (with feedback for retry).
        MutationError: If required id fields are missing.
    """
    # Migration: backfill empty explored arrays from path answer_ids
    # This fixes legacy data where LLM serialized paths but left explored empty
    _backfill_explored_from_paths(output)

    # Validate cross-references first
    errors = validate_seed_mutations(graph, output)
    if errors:
        raise SeedMutationError(errors)

    # Update entity dispositions and names
    for i, entity_decision in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity_decision, "entity_id", f"Entity decision at index {i}")
        try:
            entity_id = _resolve_entity_ref(graph, raw_id)
        except ValueError:
            # Entity not found - validation should have caught this
            continue

        # Build update dict with disposition and optional name
        update_data: dict[str, Any] = {
            "disposition": entity_decision.get("disposition", "retained"),
        }

        # Apply SEED name only if entity doesn't already have one from BRAINSTORM.
        # This preserves BRAINSTORM names (which emerged naturally during discussion)
        # while allowing SEED to generate names for entities that need them.
        if name := entity_decision.get("name"):
            existing_node = graph.get_node(entity_id)
            if existing_node and not existing_node.get("name"):
                update_data["name"] = name

        graph.update_node(entity_id, **update_data)

    # Update dilemma exploration decisions
    for i, dilemma_decision in enumerate(output.get("dilemmas", [])):
        raw_id = _require_field(dilemma_decision, "dilemma_id", f"Dilemma decision at index {i}")
        dilemma_node_id = _prefix_id("dilemma", raw_id)
        if graph.has_node(dilemma_node_id):
            explored = dilemma_decision.get("explored", [])
            graph.update_node(
                dilemma_node_id,
                explored=explored,
                unexplored=dilemma_decision.get("unexplored", []),
            )

    # Create paths from explored dilemmas (must be created before consequences)
    for i, path in enumerate(output.get("paths", [])):
        raw_id = _require_field(path, "path_id", f"Path at index {i}")
        path_node_id = _prefix_id("path", raw_id)

        # Store prefixed dilemma reference
        raw_dilemma_id = path.get("dilemma_id")
        prefixed_dilemma_id = _prefix_id("dilemma", raw_dilemma_id) if raw_dilemma_id else None

        # Prefix unexplored answers from the same dilemma
        raw_unexplored = path.get("unexplored_answer_ids", [])
        prefixed_unexplored = []
        if prefixed_dilemma_id:
            for unexplored_answer_id in raw_unexplored:
                # Format: dilemma::dilemma_id::alt::answer_id
                full_unexplored_id = f"{prefixed_dilemma_id}::alt::{unexplored_answer_id}"
                prefixed_unexplored.append(full_unexplored_id)

        # Look up answer's is_default_path to determine if path is canonical (spine)
        is_canonical = False
        if prefixed_dilemma_id and "answer_id" in path:
            answer_local_id = path["answer_id"]
            full_answer_id = f"{prefixed_dilemma_id}::alt::{answer_local_id}"
            answer_node = graph.get_node(full_answer_id)
            if answer_node is not None:
                is_canonical = answer_node.get("is_default_path", False)

        # Build path node data
        path_data = {
            "type": "path",
            "raw_id": raw_id,
            "name": path.get("name"),
            "dilemma_id": prefixed_dilemma_id,
            "answer_id": path.get("answer_id"),
            "unexplored_answer_ids": prefixed_unexplored,
            "path_importance": path.get("path_importance"),
            "description": path.get("description"),
            "consequence_ids": path.get("consequence_ids", []),
            "is_canonical": is_canonical,  # True if exploring default answer (for spine arc)
        }
        path_data = _clean_dict(path_data)
        graph.create_node(path_node_id, path_data)

        # Link path to the answer it explores
        if "answer_id" in path and prefixed_dilemma_id:
            answer_local_id = path["answer_id"]
            # Answer ID format: dilemma::dilemma_id::alt::answer_id
            full_answer_id = f"{prefixed_dilemma_id}::alt::{answer_local_id}"
            graph.add_edge("explores", path_node_id, full_answer_id)

    # Create consequences (after paths so edges can be created)
    for i, consequence in enumerate(output.get("consequences", [])):
        raw_id = _require_field(consequence, "consequence_id", f"Consequence at index {i}")
        consequence_id = _prefix_id("consequence", raw_id)

        # Prefix path reference
        raw_path_id = consequence.get("path_id")
        prefixed_path_id = _prefix_id("path", raw_path_id) if raw_path_id else None

        consequence_data = {
            "type": "consequence",
            "raw_id": raw_id,
            "path_id": prefixed_path_id,
            "description": consequence.get("description"),
            "narrative_effects": consequence.get("narrative_effects", []),
        }
        consequence_data = _clean_dict(consequence_data)
        graph.create_node(consequence_id, consequence_data)

        # Link consequence to its path (path must exist)
        if prefixed_path_id and graph.has_node(prefixed_path_id):
            graph.add_edge("has_consequence", prefixed_path_id, consequence_id)

    # Create initial beats
    for i, beat in enumerate(output.get("initial_beats", [])):
        raw_id = _require_field(beat, "beat_id", f"Beat at index {i}")
        beat_id = _prefix_id("beat", raw_id)

        # Resolve entity references (entities now use category-based IDs like character::pim)
        raw_entities = beat.get("entities", [])
        prefixed_entities = []
        for eid in raw_entities:
            with contextlib.suppress(ValueError):
                prefixed_entities.append(_resolve_entity_ref(graph, eid))

        # Resolve location reference (location is an entity, typically location::X)
        raw_location = beat.get("location")
        prefixed_location = None
        if raw_location:
            with contextlib.suppress(ValueError):
                prefixed_location = _resolve_entity_ref(graph, raw_location)

        # Resolve location_alternatives (also entity IDs)
        raw_location_alts = beat.get("location_alternatives", [])
        prefixed_location_alts = []
        for eid in raw_location_alts:
            with contextlib.suppress(ValueError):
                prefixed_location_alts.append(_resolve_entity_ref(graph, eid))

        # Prefix dilemma_id in dilemma_impacts
        raw_impacts = beat.get("dilemma_impacts", [])
        prefixed_impacts = []
        for impact in raw_impacts:
            prefixed_impact = dict(impact)
            if "dilemma_id" in impact:
                prefixed_impact["dilemma_id"] = _prefix_id("dilemma", impact["dilemma_id"])
            prefixed_impacts.append(prefixed_impact)

        beat_data = {
            "type": "beat",
            "raw_id": raw_id,
            "summary": beat.get("summary"),
            "dilemma_impacts": prefixed_impacts,
            "entities": prefixed_entities,
            "location": prefixed_location,
            "location_alternatives": prefixed_location_alts,
        }
        beat_data = _clean_dict(beat_data)
        graph.create_node(beat_id, beat_data)

        # Link beat to paths it belongs to
        for raw_path_id in beat.get("paths", []):
            prefixed_path_id = _prefix_id("path", raw_path_id)
            graph.add_edge("belongs_to", beat_id, prefixed_path_id)

    # Store convergence sketch as metadata (upsert allows re-running SEED)
    if "convergence_sketch" in output:
        sketch = output["convergence_sketch"]
        graph.upsert_node(
            "convergence_sketch",
            {
                "type": "convergence_sketch",
                "convergence_points": sketch.get("convergence_points", []),
                "residue_notes": sketch.get("residue_notes", []),
            },
        )


def format_semantic_errors_as_content(errors: list[SeedValidationError]) -> str:
    """Format semantic errors as content-focused feedback for outer loop retry.

    This function formats errors in a way that helps the LLM understand what
    went wrong conceptually, not just structurally. It groups errors by type
    and provides actionable guidance.

    Args:
        errors: List of SeedValidationError objects from validate_seed_mutations.

    Returns:
        Content-focused feedback string suitable for appending to conversation.

    Example output:
        I found some issues with the summary that need correction:

        **Missing items** - these need decisions:
          - hollow_key
          - ancient_scroll

        **Invalid references** - these don't exist in BRAINSTORM:
          - 'ghost' was referenced but isn't defined

        Please reconsider the summary, ensuring you only reference
        entities and dilemmas from the BRAINSTORM phase.
    """
    if not errors:
        return ""

    by_category = categorize_errors(errors)
    lines: list[str] = ["I found some issues with the summary that need correction:"]

    # Completeness errors (missing decisions and missing paths)
    completeness_errors = by_category.get(SeedErrorCategory.COMPLETENESS, [])
    if completeness_errors:
        # Note: Issue text now uses "has no path" (updated terminology)
        decision_errors = [e for e in completeness_errors if "has no path" not in e.issue]
        path_errors = [e for e in completeness_errors if "has no path" in e.issue]

        if decision_errors:
            lines.append("")
            lines.append("**Missing items** - these need decisions:")
            for e in decision_errors[:_MAX_ERRORS_DISPLAY]:
                # Extract item ID from issue message (e.g., "Missing decision for entity 'X'")
                match = re.search(r"'([^']+)'", e.issue)
                if match:
                    lines.append(f"  - {match.group(1)}")
                else:
                    lines.append(f"  - {e.field_path}: {e.issue}")
            if len(decision_errors) > _MAX_ERRORS_DISPLAY:
                lines.append(f"  ... and {len(decision_errors) - _MAX_ERRORS_DISPLAY} more")

        if path_errors:
            lines.append("")
            lines.append("**Missing paths** - each dilemma must have at least one path:")
            for e in path_errors[:_MAX_ERRORS_DISPLAY]:
                match = re.search(r"'([^']+)'", e.issue)
                if match:
                    lines.append(f"  - dilemma '{match.group(1)}' has no path")
                else:
                    lines.append(f"  - {e.field_path}: {e.issue}")
            if len(path_errors) > _MAX_ERRORS_DISPLAY:
                lines.append(f"  ... and {len(path_errors) - _MAX_ERRORS_DISPLAY} more")

    # Semantic errors (invalid references)
    semantic_errors = by_category.get(SeedErrorCategory.SEMANTIC, [])
    if semantic_errors:
        lines.append("")
        lines.append("**Invalid references** - fix these fields:")
        for e in semantic_errors[:_MAX_ERRORS_DISPLAY]:
            if e.provided:
                lines.append(f"  - {e.field_path}: '{e.provided}' is not valid")
                # Add suggestion if available
                suggestion = _format_error_available(e.provided, e.available)
                if suggestion:
                    lines.append(f"    {suggestion}")
            else:
                lines.append(f"  - {e.field_path}: {e.issue}")
        if len(semantic_errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(semantic_errors) - _MAX_ERRORS_DISPLAY} more")

    # Cross-reference errors (bucket misplacement, answer not in explored)
    cross_ref_errors = by_category.get(SeedErrorCategory.CROSS_REFERENCE, [])
    if cross_ref_errors:
        lines.append("")
        lines.append("**Bucket misplacement** - these answers are in the wrong list:")
        for e in cross_ref_errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.issue}")
        if len(cross_ref_errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(cross_ref_errors) - _MAX_ERRORS_DISPLAY} more")

    # Inner/structural errors (rarely should make it to outer loop, but handle gracefully)
    inner_errors = by_category.get(SeedErrorCategory.INNER, [])
    if inner_errors:
        lines.append("")
        lines.append("**Structural issues**:")
        for e in inner_errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.field_path}: {e.issue}")
        if len(inner_errors) > _MAX_ERRORS_DISPLAY:
            lines.append(f"  ... and {len(inner_errors) - _MAX_ERRORS_DISPLAY} more")

    lines.append("")
    lines.append(
        "Please reconsider the summary, ensuring you only reference "
        "entities and dilemmas from the BRAINSTORM phase."
    )

    return "\n".join(lines)
