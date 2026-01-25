"""Stage mutation appliers.

Each stage produces structured output that the runtime applies as graph
mutations. This module contains the logic for each stage's mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

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

    def strip_scope(s: str) -> str:
        """Remove scope prefix if present (e.g., 'entity::hero' -> 'hero')."""
        return s.split("::")[-1] if "::" in s else s

    invalid_raw_lower = strip_scope(invalid_id).lower()
    matcher = SequenceMatcher(a=invalid_raw_lower)
    scored = []
    for aid in available:
        matcher.set_seq2(strip_scope(aid).lower())
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
    COMPLETENESS = auto()  # Missing entity/tension decisions
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
        field_path: Path to the invalid field (e.g., "tensions.0.central_entity_ids").
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
        field_path: Path to the invalid field (e.g., "threads.0.tension_id").
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
        field_path: Path to the invalid field (e.g., "arc.threads.0").
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
        context: Description for error message (e.g., "entity", "tension").

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

    Accepts both scoped (`entity::hero`) and unscoped (`hero`) IDs.
    If scoped, validates the scope matches expected_scope.

    Args:
        provided_id: The ID from LLM output (may or may not have scope prefix).
        expected_scope: Expected scope type (e.g., "entity", "tension", "thread").

    Returns:
        Tuple of (normalized_id, error_message).
        - If valid: (raw_id, None)
        - If wrong scope: (provided_id, error message describing mismatch)

    Examples:
        >>> _normalize_id("entity::hero", "entity")
        ('hero', None)
        >>> _normalize_id("hero", "entity")
        ('hero', None)
        >>> _normalize_id("tension::hero", "entity")
        ('tension::hero', "Wrong scope prefix: expected 'entity::', got 'tension::'")
    """
    if "::" in provided_id:
        scope, raw_id = provided_id.split("::", 1)
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


# Registry of stages with mutation handlers
_MUTATION_STAGES = frozenset({"dream", "brainstorm", "seed", "grow"})


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
    }

    # Remove None values for cleaner storage
    vision_data = _clean_dict(vision_data)

    # Use upsert to allow re-running DREAM stage (replaces existing vision)
    graph.upsert_node("vision", vision_data)


def validate_brainstorm_mutations(output: dict[str, Any]) -> list[BrainstormValidationError]:
    """Validate BRAINSTORM output internal consistency.

    Checks that the output is self-consistent (no graph needed):
    1. All central_entity_ids in tensions exist in entities list
    2. All alternative IDs within a tension are unique
    3. Each tension has exactly one is_default_path=true alternative

    Args:
        output: BRAINSTORM stage output (entities, tensions).

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
            entity_ids.add(entity_id)
    sorted_entity_ids = sorted(entity_ids)

    # Validate each tension
    for i, tension in enumerate(output.get("tensions", [])):
        tension_id = tension.get("tension_id", f"<index {i}>")

        # 1. Check central_entity_ids reference valid entities
        for eid in tension.get("central_entity_ids", []):
            if eid not in entity_ids:
                errors.append(
                    BrainstormValidationError(
                        field_path=f"tensions.{i}.central_entity_ids",
                        issue=f"Entity '{eid}' not in entities list",
                        available=sorted_entity_ids,
                        provided=eid,
                    )
                )

        # 2. Check alternative IDs are unique within this tension
        alts = tension.get("alternatives", [])
        alt_ids = [a.get("alternative_id") for a in alts if a.get("alternative_id")]
        alt_id_counts = Counter(alt_ids)
        for alt_id, count in alt_id_counts.items():
            if count > 1:
                errors.append(
                    BrainstormValidationError(
                        field_path=f"tensions.{i}.alternatives",
                        issue=f"Duplicate alternative_id '{alt_id}' appears {count} times in tension '{tension_id}'",
                        available=[],
                        provided=alt_id,
                    )
                )

        # 3. Check exactly one alternative has is_default_path=True
        # (missing or False both count as non-default - Pydantic validation ensures
        # the field exists for valid BrainstormOutput, this handles edge cases)
        default_count = sum(1 for a in alts if a.get("is_default_path"))
        if default_count != 1:
            issue = (
                f"No alternative has is_default_path=true in tension '{tension_id}'"
                if default_count == 0
                else f"Multiple alternatives have is_default_path=true in tension '{tension_id}'"
            )
            errors.append(
                BrainstormValidationError(
                    field_path=f"tensions.{i}.alternatives",
                    issue=issue,
                    available=[],
                    provided=f"found {default_count} defaults",
                )
            )

    return errors


def _prefix_id(node_type: str, raw_id: str) -> str:
    """Prefix a raw ID with its node type for namespace isolation.

    This allows entities and tensions to have the same raw ID without collision.
    E.g., both can use "cipher_journal" -> "entity::cipher_journal", "tension::cipher_journal"

    This function is idempotent - if the ID already has the correct prefix,
    it returns it unchanged. If it has a different prefix or multiple prefixes,
    the raw part is extracted and re-prefixed.

    Args:
        node_type: Node type prefix (entity, tension, thread, etc.)
        raw_id: Raw ID from LLM output (may already be prefixed).

    Returns:
        Prefixed ID in format "type::raw_id".
    """
    # Strip any existing prefixes to get the raw ID
    # This handles: "the_detective", "entity::the_detective", "entity::entity::the_detective"
    if "::" in raw_id:
        raw_id = raw_id.rsplit("::", 1)[-1]

    return f"{node_type}::{raw_id}"


def apply_brainstorm_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply BRAINSTORM stage output to graph.

    Creates entity nodes and tension nodes with their alternatives.
    Tensions are linked to their alternatives via has_alternative edges.

    Node IDs are prefixed by type to avoid collisions:
    - entity::raw_id
    - tension::raw_id
    - tension::tension_id::alt::alt_id (for alternatives)

    Args:
        graph: Graph to mutate.
        output: BRAINSTORM stage output (entities, tensions).

    Raises:
        MutationError: If entities or tensions are missing required id fields.
    """
    # Add entities
    for i, entity in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity, "entity_id", f"Entity at index {i}")
        entity_id = _prefix_id("entity", raw_id)
        node_data = {
            "type": "entity",
            "raw_id": raw_id,  # Store original ID for reference
            "entity_type": entity.get("entity_category"),  # character, location, object, faction
            "concept": entity.get("concept"),
            "notes": entity.get("notes"),
            "disposition": "proposed",  # All entities start as proposed
        }
        # Remove None values
        node_data = _clean_dict(node_data)
        graph.create_node(entity_id, node_data)

    # Add tensions with alternatives
    for i, tension in enumerate(output.get("tensions", [])):
        raw_id = _require_field(tension, "tension_id", f"Tension at index {i}")
        tension_id = _prefix_id("tension", raw_id)

        # Prefix entity references in central_entity_ids list
        raw_central_entities = tension.get("central_entity_ids", [])
        prefixed_central_entities = [_prefix_id("entity", eid) for eid in raw_central_entities]

        # Create tension node
        tension_data = {
            "type": "tension",
            "raw_id": raw_id,  # Store original ID for reference
            "question": tension.get("question"),
            "central_entity_ids": prefixed_central_entities,
            "why_it_matters": tension.get("why_it_matters"),
        }
        tension_data = _clean_dict(tension_data)
        graph.create_node(tension_id, tension_data)

        # Create alternative nodes and edges
        for j, alt in enumerate(tension.get("alternatives", [])):
            alt_local_id = _require_field(
                alt, "alternative_id", f"Alternative at index {j} in tension '{raw_id}'"
            )
            # Alternative ID format: tension::tension_raw_id::alt::alt_local_id
            alt_id = f"{tension_id}::alt::{alt_local_id}"
            alt_data = {
                "type": "alternative",
                "raw_id": alt_local_id,
                "description": alt.get("description"),
                "is_default_path": alt.get("is_default_path", False),
            }
            alt_data = _clean_dict(alt_data)
            graph.create_node(alt_id, alt_data)
            graph.add_edge("has_alternative", tension_id, alt_id)


def _cross_type_hint(
    raw_id: str,
    expected_type: str,
    valid_entity_ids: set[str],
    valid_tension_ids: set[str],
    seed_thread_ids: set[str],
    entity_types: dict[str, str],
) -> str:
    """Return type-aware message when ID exists as different type.

    When an ID isn't found in the expected set, checks whether it exists
    in a different type's set and provides a helpful cross-type hint.
    This prevents misleading "not in BRAINSTORM" feedback when the ID
    actually IS in brainstorm but as a different type (e.g., a faction
    entity used as a tension_id).

    Args:
        raw_id: The normalized ID that wasn't found.
        expected_type: What type was expected ("entity", "tension", "thread").
        valid_entity_ids: Set of valid entity raw IDs from brainstorm.
        valid_tension_ids: Set of valid tension raw IDs from brainstorm.
        seed_thread_ids: Set of thread raw IDs from SEED output.
        entity_types: Mapping of entity raw_id to entity_type (character, faction, etc.).

    Returns:
        Descriptive error message indicating the type mismatch or fallback.
    """
    if expected_type == "tension" and raw_id in valid_entity_ids:
        etype = entity_types.get(raw_id, "entity")
        return (
            f"'{raw_id}' is an entity ({etype}), not a tension. "
            "Tension IDs follow the pattern subject_X_or_Y"
        )
    if expected_type == "tension" and raw_id in seed_thread_ids:
        return f"'{raw_id}' is a thread ID, not a tension. Tension IDs are longer binary questions"
    if expected_type == "entity" and raw_id in valid_tension_ids:
        return f"'{raw_id}' is a tension ID, not an entity"
    if expected_type == "entity" and raw_id in seed_thread_ids:
        return f"'{raw_id}' is a thread ID, not an entity"
    if expected_type == "thread" and raw_id in valid_entity_ids:
        etype = entity_types.get(raw_id, "entity")
        return f"'{raw_id}' is an entity ({etype}), not a thread"
    if expected_type == "thread" and raw_id in valid_tension_ids:
        return f"'{raw_id}' is a tension ID, not a thread"
    if expected_type == "thread":
        return f"Thread '{raw_id}' not defined in SEED threads"
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
        expected_scope: Expected scope type (e.g., "entity", "tension", "thread").
        valid_ids: Set of valid raw IDs for the expected type.
        field_path: Field path for error reporting.
        errors: Error list to append to.
        sorted_available: Pre-sorted valid IDs for error messages.
        category: Error category to set on created errors.
        cross_type_sets: Optional tuple of (valid_entity_ids, valid_tension_ids,
            seed_thread_ids, entity_types) for cross-type hint messages.
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
            entity_ids, tension_ids, thread_ids, entity_types = cross_type_sets
            issue = _cross_type_hint(
                normalized_id, expected_scope, entity_ids, tension_ids, thread_ids, entity_types
            )
        elif expected_scope == "thread":
            issue = f"Thread '{normalized_id}' not defined in SEED threads"
        else:
            issue = f"{expected_scope.title()} '{normalized_id}' not in BRAINSTORM"
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
    2. Tension IDs in decisions exist in graph
    3. Thread tension_ids reference valid tensions
    4. Thread alternative_ids exist for their tension
    5. Beat entity references exist
    6. Beat thread references exist (within SEED output)
    7. Consequence thread references exist (within SEED output)
    8. Tension impacts reference valid tensions
    9. All BRAINSTORM entities have decisions (completeness)
    10. All BRAINSTORM tensions have decisions (completeness)
    11. All tensions have at least one thread (completeness)
    12. Each beat references its thread's parent tension in tension_impacts
    13. Each thread has at least one beat with effect="commits" for its tension

    Args:
        graph: Graph containing BRAINSTORM data (entities, tensions, alternatives).
        output: SEED stage output to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[SeedValidationError] = []

    # Build lookup sets from graph (BRAINSTORM data)
    entity_nodes = graph.get_nodes_by_type("entity")
    tension_nodes = graph.get_nodes_by_type("tension")

    # Raw IDs (unprefixed) for validation - what LLM uses
    # Type annotation ensures mypy knows these are str sets (filter guarantees non-None)
    valid_entity_ids: set[str] = {n["raw_id"] for n in entity_nodes.values() if n.get("raw_id")}
    valid_tension_ids: set[str] = {n["raw_id"] for n in tension_nodes.values() if n.get("raw_id")}

    # Entity type lookup for informative error messages
    entity_types: dict[str, str] = {
        n["raw_id"]: n.get("entity_type", "entity")
        for n in entity_nodes.values()
        if n.get("raw_id")
    }

    # Build alternative lookup: tension_raw_id -> set of alt_raw_ids
    # Use O(edges) algorithm instead of O(tensions * edges)
    alt_by_tension: dict[str, set[str]] = {}
    tension_id_to_raw: dict[str, str] = {
        tid: tdata["raw_id"] for tid, tdata in tension_nodes.items() if tdata.get("raw_id")
    }
    for edge in graph.get_edges(edge_type="has_alternative"):
        from_node_id = edge.get("from", "")
        raw_tension_id = tension_id_to_raw.get(from_node_id)
        if not raw_tension_id:
            continue
        alt_node = graph.get_node(edge.get("to", ""))
        if alt_node and alt_node.get("raw_id"):
            alt_by_tension.setdefault(raw_tension_id, set()).add(alt_node["raw_id"])

    # Extract thread IDs from SEED output (for internal references)
    # Normalize IDs to handle both scoped (thread::foo) and bare (foo) formats
    seed_thread_ids: set[str] = {
        _normalize_id(t["thread_id"], "thread")[0]
        for t in output.get("threads", [])
        if t.get("thread_id")
    }

    # Pre-sort available IDs once (used in error messages)
    sorted_entity_ids = sorted(valid_entity_ids)
    sorted_tension_ids = sorted(valid_tension_ids)
    sorted_thread_ids = sorted(seed_thread_ids)

    # Cross-type sets for helpful error messages (entity used as tension, etc.)
    cross_type_sets = (valid_entity_ids, valid_tension_ids, seed_thread_ids, entity_types)

    # 0. Check for duplicate IDs in output (prevents LLM from outputting same item twice)
    for field_path, id_field, scope in [
        ("entities", "entity_id", "entity"),
        ("tensions", "tension_id", "tension"),
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
                        issue=f"Duplicate {id_field} '{dup_id}' appears {count} times - each {scope} should have exactly one decision",
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

    # 2. Validate tension decisions
    for i, decision in enumerate(output.get("tensions", [])):
        _validate_id(
            decision.get("tension_id"),
            "tension",
            valid_tension_ids,
            f"tensions.{i}.tension_id",
            errors,
            sorted_tension_ids,
        )

    # 3-4. Validate threads
    for i, thread in enumerate(output.get("threads", [])):
        raw_thread_tension_id = thread.get("tension_id")
        raw_alt_id = thread.get("alternative_id")

        thread_tension_id = _validate_id(
            raw_thread_tension_id,
            "tension",
            valid_tension_ids,
            f"threads.{i}.tension_id",
            errors,
            sorted_tension_ids,
        )

        # Check alternative (report even if tension was also invalid, so LLM gets all feedback)
        if raw_thread_tension_id and raw_alt_id:
            # Need the normalized tension ID even if invalid for the alt check
            norm_tid = (
                _normalize_id(raw_thread_tension_id, "tension")[0]
                if raw_thread_tension_id
                else None
            )
            if norm_tid:
                valid_alts = alt_by_tension.get(norm_tid, set())
                if raw_alt_id not in valid_alts:
                    errors.append(
                        SeedValidationError(
                            field_path=f"threads.{i}.alternative_id",
                            issue=f"Alternative '{raw_alt_id}' not in tension '{norm_tid}'",
                            available=sorted(valid_alts) if thread_tension_id else [],
                            provided=raw_alt_id,
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

        # 6. Thread references
        for raw_thread_id in beat.get("threads", []):
            _validate_id(
                raw_thread_id,
                "thread",
                seed_thread_ids,
                f"initial_beats.{i}.threads",
                errors,
                sorted_thread_ids,
                cross_type_sets=cross_type_sets,
            )

        # 8. Tension impacts
        for j, impact in enumerate(beat.get("tension_impacts", [])):
            _validate_id(
                impact.get("tension_id"),
                "tension",
                valid_tension_ids,
                f"initial_beats.{i}.tension_impacts.{j}.tension_id",
                errors,
                sorted_tension_ids,
                cross_type_sets=cross_type_sets,
            )

    # 7. Validate consequence thread references (internal to SEED)
    for i, consequence in enumerate(output.get("consequences", [])):
        _validate_id(
            consequence.get("thread_id"),
            "thread",
            seed_thread_ids,
            f"consequences.{i}.thread_id",
            errors,
            sorted_thread_ids,
        )

    # 9 & 10. Check completeness: all BRAINSTORM entities and tensions should have decisions
    completeness_checks = [
        ("entities", "entity", valid_entity_ids),
        ("tensions", "tension", valid_tension_ids),
    ]
    for field_path, item_type, valid_ids in completeness_checks:
        id_field = f"{item_type}_id"
        # Normalize IDs from output to handle scoped IDs (entity::hero -> hero)
        # Only count IDs with correct scope (or unscoped) toward completeness
        decided_ids: set[str] = set()
        for decision in output.get(field_path, []):
            raw_id = decision.get(id_field)
            if raw_id:
                normalized_id, scope_error = _normalize_id(raw_id, item_type)
                if not scope_error:
                    decided_ids.add(normalized_id)

        missing_ids = valid_ids - decided_ids
        for item_id in sorted(missing_ids):
            # For entities, include type (character/location/object/faction) for clarity
            if item_type == "entity":
                entity_type_name = entity_types.get(item_id, "entity")
                issue_msg = f"Missing decision for {entity_type_name} '{item_id}'"
            else:
                issue_msg = f"Missing decision for {item_type} '{item_id}'"
            errors.append(
                SeedValidationError(
                    field_path=field_path,
                    issue=issue_msg,
                    available=[],
                    provided="",
                    category=SeedErrorCategory.COMPLETENESS,
                )
            )

    # 11. Check completeness: all tensions should have at least one thread
    tensions_with_threads: set[str] = set()
    thread_tension_map: dict[str, str] = {}  # thread_id -> tension_id (normalized)
    for thread in output.get("threads", []):
        raw_tension_id = thread.get("tension_id")
        raw_thread_id = thread.get("thread_id")
        if raw_tension_id:
            normalized_tid, scope_error = _normalize_id(raw_tension_id, "tension")
            if not scope_error:
                tensions_with_threads.add(normalized_tid)
                if raw_thread_id:
                    normalized_thid, _ = _normalize_id(raw_thread_id, "thread")
                    thread_tension_map[normalized_thid] = normalized_tid

    tensions_without_threads = valid_tension_ids - tensions_with_threads
    for tension_id in sorted(tensions_without_threads):
        errors.append(
            SeedValidationError(
                field_path="threads",
                issue=(
                    f"Tension '{tension_id}' has no thread. "
                    f"Create at least one thread exploring this tension."
                ),
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            )
        )

    # 12. Check beats reference their thread's parent tension
    # 13. Check each thread has at least one commits beat for its tension
    threads_with_commits: set[str] = set()  # thread_ids that have a commits beat
    for i, beat in enumerate(output.get("initial_beats", [])):
        beat_thread_ids: list[str] = []
        for raw_thread_id in beat.get("threads", []):
            normalized_thid, _ = _normalize_id(raw_thread_id, "thread")
            beat_thread_ids.append(normalized_thid)

        # Collect normalized tension_ids referenced in this beat's impacts
        beat_impact_tensions: set[str] = set()
        beat_impact_commits_tensions: set[str] = set()
        for impact in beat.get("tension_impacts", []):
            raw_tid = impact.get("tension_id")
            if raw_tid:
                normalized_tid, _ = _normalize_id(raw_tid, "tension")
                beat_impact_tensions.add(normalized_tid)
                if impact.get("effect") == "commits":
                    beat_impact_commits_tensions.add(normalized_tid)

        # For each thread this beat belongs to, check it references the parent tension
        for thread_id in beat_thread_ids:
            expected_tension = thread_tension_map.get(thread_id)
            if not expected_tension:
                continue  # Thread not found in output; already caught by check 6

            if expected_tension not in beat_impact_tensions:
                beat_id = beat.get("beat_id", f"beat_{i}")
                errors.append(
                    SeedValidationError(
                        field_path=f"initial_beats.{i}.tension_impacts",
                        issue=(
                            f"Beat '{beat_id}' belongs to thread '{thread_id}' "
                            f"but does not reference its parent tension "
                            f"'{expected_tension}' in tension_impacts. "
                            f"Each beat must impact its own thread's tension."
                        ),
                        available=[expected_tension],
                        provided=", ".join(sorted(beat_impact_tensions)) or "(none)",
                        category=SeedErrorCategory.SEMANTIC,
                    )
                )

            # Track commits beats per thread
            if expected_tension in beat_impact_commits_tensions:
                threads_with_commits.add(thread_id)

    # Report threads missing commits beats
    for thread_id in sorted(thread_tension_map.keys()):
        if thread_id not in threads_with_commits:
            expected_tension = thread_tension_map[thread_id]
            errors.append(
                SeedValidationError(
                    field_path=f"threads.{thread_id}.commits",
                    issue=(
                        f"Thread '{thread_id}' has no beat with effect='commits' "
                        f"for its parent tension '{expected_tension}'. "
                        f"Each thread must have at least one beat that commits "
                        f"its tension resolution."
                    ),
                    available=[expected_tension],
                    provided="",
                    category=SeedErrorCategory.COMPLETENESS,
                )
            )

    return errors


def apply_seed_mutations(graph: Graph, output: dict[str, Any]) -> None:
    """Apply SEED stage output to graph.

    First validates all cross-references, then applies mutations.

    Updates entity dispositions, creates threads from explored tensions,
    creates consequences, and creates initial beats.

    Node IDs are prefixed by type to match BRAINSTORM's namespacing:
    - thread::raw_id
    - consequence::raw_id
    - beat::raw_id

    References to entities/tensions from LLM output are prefixed for lookup.

    Args:
        graph: Graph to mutate.
        output: SEED stage output (SeedOutput fields).

    Raises:
        SeedMutationError: If semantic validation fails (with feedback for retry).
        MutationError: If required id fields are missing.
    """
    # Validate cross-references first
    errors = validate_seed_mutations(graph, output)
    if errors:
        raise SeedMutationError(errors)

    # Update entity dispositions
    for i, entity_decision in enumerate(output.get("entities", [])):
        raw_id = _require_field(entity_decision, "entity_id", f"Entity decision at index {i}")
        entity_id = _prefix_id("entity", raw_id)
        if graph.has_node(entity_id):
            graph.update_node(
                entity_id,
                disposition=entity_decision.get("disposition", "retained"),
            )

    # Update tension exploration decisions
    for i, tension_decision in enumerate(output.get("tensions", [])):
        raw_id = _require_field(tension_decision, "tension_id", f"Tension decision at index {i}")
        tension_id = _prefix_id("tension", raw_id)
        if graph.has_node(tension_id):
            graph.update_node(
                tension_id,
                explored=tension_decision.get("explored", []),
                implicit=tension_decision.get("implicit", []),
            )

    # Create threads from explored tensions (must be created before consequences)
    for i, thread in enumerate(output.get("threads", [])):
        raw_id = _require_field(thread, "thread_id", f"Thread at index {i}")
        thread_id = _prefix_id("thread", raw_id)

        # Store prefixed tension reference
        raw_tension_id = thread.get("tension_id")
        prefixed_tension_id = _prefix_id("tension", raw_tension_id) if raw_tension_id else None

        # Prefix unexplored alternatives from the same tension
        raw_unexplored = thread.get("unexplored_alternative_ids", [])
        prefixed_unexplored = []
        if prefixed_tension_id:
            for unexplored_alt_id in raw_unexplored:
                # Format: tension::tension_id::alt::alt_id
                full_unexplored_id = f"{prefixed_tension_id}::alt::{unexplored_alt_id}"
                prefixed_unexplored.append(full_unexplored_id)

        thread_data = {
            "type": "thread",
            "raw_id": raw_id,
            "name": thread.get("name"),
            "tension_id": prefixed_tension_id,
            "alternative_id": thread.get("alternative_id"),  # Local alt ID, not prefixed
            "unexplored_alternative_ids": prefixed_unexplored,
            "thread_importance": thread.get("thread_importance"),
            "description": thread.get("description"),
            "consequence_ids": thread.get("consequence_ids", []),
        }
        thread_data = _clean_dict(thread_data)
        graph.create_node(thread_id, thread_data)

        # Link thread to the alternative it explores
        if "alternative_id" in thread and prefixed_tension_id:
            alt_local_id = thread["alternative_id"]
            # Alternative ID format: tension::tension_id::alt::alt_id
            full_alt_id = f"{prefixed_tension_id}::alt::{alt_local_id}"
            graph.add_edge("explores", thread_id, full_alt_id)

    # Create consequences (after threads so edges can be created)
    for i, consequence in enumerate(output.get("consequences", [])):
        raw_id = _require_field(consequence, "consequence_id", f"Consequence at index {i}")
        consequence_id = _prefix_id("consequence", raw_id)

        # Prefix thread reference
        raw_thread_id = consequence.get("thread_id")
        prefixed_thread_id = _prefix_id("thread", raw_thread_id) if raw_thread_id else None

        consequence_data = {
            "type": "consequence",
            "raw_id": raw_id,
            "thread_id": prefixed_thread_id,
            "description": consequence.get("description"),
            "narrative_effects": consequence.get("narrative_effects", []),
        }
        consequence_data = _clean_dict(consequence_data)
        graph.create_node(consequence_id, consequence_data)

        # Link consequence to its thread (thread must exist)
        if prefixed_thread_id and graph.has_node(prefixed_thread_id):
            graph.add_edge("has_consequence", prefixed_thread_id, consequence_id)

    # Create initial beats
    for i, beat in enumerate(output.get("initial_beats", [])):
        raw_id = _require_field(beat, "beat_id", f"Beat at index {i}")
        beat_id = _prefix_id("beat", raw_id)

        # Prefix entity references
        raw_entities = beat.get("entities", [])
        prefixed_entities = [_prefix_id("entity", eid) for eid in raw_entities]

        # Prefix location reference (location is an entity)
        raw_location = beat.get("location")
        prefixed_location = _prefix_id("entity", raw_location) if raw_location else None

        # Prefix location_alternatives (also entity IDs)
        raw_location_alts = beat.get("location_alternatives", [])
        prefixed_location_alts = [_prefix_id("entity", eid) for eid in raw_location_alts]

        # Prefix tension_id in tension_impacts
        raw_impacts = beat.get("tension_impacts", [])
        prefixed_impacts = []
        for impact in raw_impacts:
            prefixed_impact = dict(impact)
            if "tension_id" in impact:
                prefixed_impact["tension_id"] = _prefix_id("tension", impact["tension_id"])
            prefixed_impacts.append(prefixed_impact)

        beat_data = {
            "type": "beat",
            "raw_id": raw_id,
            "summary": beat.get("summary"),
            "tension_impacts": prefixed_impacts,
            "entities": prefixed_entities,
            "location": prefixed_location,
            "location_alternatives": prefixed_location_alts,
        }
        beat_data = _clean_dict(beat_data)
        graph.create_node(beat_id, beat_data)

        # Link beat to threads it belongs to
        for raw_thread_id in beat.get("threads", []):
            prefixed_thread_id = _prefix_id("thread", raw_thread_id)
            graph.add_edge("belongs_to", beat_id, prefixed_thread_id)

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
        entities and tensions from the BRAINSTORM phase.
    """
    if not errors:
        return ""

    by_category = categorize_errors(errors)
    lines: list[str] = ["I found some issues with the summary that need correction:"]

    # Completeness errors (missing decisions and missing threads)
    completeness_errors = by_category.get(SeedErrorCategory.COMPLETENESS, [])
    if completeness_errors:
        decision_errors = [e for e in completeness_errors if "has no thread" not in e.issue]
        thread_errors = [e for e in completeness_errors if "has no thread" in e.issue]

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

        if thread_errors:
            lines.append("")
            lines.append("**Missing threads** - each tension must have at least one thread:")
            for e in thread_errors[:_MAX_ERRORS_DISPLAY]:
                match = re.search(r"'([^']+)'", e.issue)
                if match:
                    lines.append(f"  - tension '{match.group(1)}' has no thread")
                else:
                    lines.append(f"  - {e.field_path}: {e.issue}")
            if len(thread_errors) > _MAX_ERRORS_DISPLAY:
                lines.append(f"  ... and {len(thread_errors) - _MAX_ERRORS_DISPLAY} more")

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
        "entities and tensions from the BRAINSTORM phase."
    )

    return "\n".join(lines)
