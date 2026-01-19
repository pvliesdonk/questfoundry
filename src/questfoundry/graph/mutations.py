"""Stage mutation appliers.

Each stage produces structured output that the runtime applies as graph
mutations. This module contains the logic for each stage's mutations.

See docs/architecture/graph-storage.md for design details.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Display limits for error messages
_MAX_ERRORS_DISPLAY = 8
_MAX_AVAILABLE_DISPLAY = 5

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
        """Format errors for exception message."""
        lines = ["BRAINSTORM has invalid internal references:"]
        for e in self.errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.field_path}: {e.issue}")
            if e.available:
                avail = e.available[:_MAX_AVAILABLE_DISPLAY]
                suffix = "..." if len(e.available) > _MAX_AVAILABLE_DISPLAY else ""
                lines.append(f"    Available: {avail}{suffix}")
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
    """

    field_path: str
    issue: str
    available: list[str] = field(default_factory=list)
    provided: str = ""


def categorize_error(error: SeedValidationError) -> SeedErrorCategory:
    """Categorize a SEED validation error for targeted retry strategy.

    Error categories determine how to recover:
    - SEMANTIC: Invalid ID reference → retry with valid ID list
    - COMPLETENESS: Missing decisions → retry with manifest counts
    - INNER: Everything else → retry with Pydantic feedback

    Uses module-level pattern constants for testability and maintainability.
    See _PATTERN_SEMANTIC_BRAINSTORM, _PATTERN_SEMANTIC_SEED, _PATTERN_COMPLETENESS.

    Args:
        error: SeedValidationError to categorize.

    Returns:
        SeedErrorCategory indicating the error type.
    """
    issue = error.issue.lower()

    # Semantic errors: invalid ID references (phantom IDs)
    if _PATTERN_SEMANTIC_BRAINSTORM in issue or _PATTERN_SEMANTIC_SEED in issue:
        return SeedErrorCategory.SEMANTIC

    # Completeness errors: missing decisions
    if _PATTERN_COMPLETENESS in issue:
        return SeedErrorCategory.COMPLETENESS

    # Default to INNER (schema/structural errors)
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
        """Format errors for exception message."""
        lines = ["SEED has invalid cross-references:"]
        for e in self.errors[:_MAX_ERRORS_DISPLAY]:
            lines.append(f"  - {e.field_path}: {e.issue}")
            if e.available:
                avail = e.available[:_MAX_AVAILABLE_DISPLAY]
                suffix = "..." if len(e.available) > _MAX_AVAILABLE_DISPLAY else ""
                lines.append(f"    Available: {avail}{suffix}")
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

    Args:
        node_type: Node type prefix (entity, tension, thread, etc.)
        raw_id: Raw ID from LLM output.

    Returns:
        Prefixed ID in format "type::raw_id".
    """
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
    seed_thread_ids: set[str] = {
        t["thread_id"] for t in output.get("threads", []) if t.get("thread_id")
    }

    # Pre-sort available IDs once (used in error messages)
    sorted_entity_ids = sorted(valid_entity_ids)
    sorted_tension_ids = sorted(valid_tension_ids)
    sorted_thread_ids = sorted(seed_thread_ids)

    # 1. Validate entity decisions
    for i, decision in enumerate(output.get("entities", [])):
        raw_entity_id = decision.get("entity_id")
        if raw_entity_id:
            entity_id, scope_error = _normalize_id(raw_entity_id, "entity")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"entities.{i}.entity_id",
                        issue=scope_error,
                        available=sorted_entity_ids,
                        provided=raw_entity_id,
                    )
                )
            elif entity_id not in valid_entity_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"entities.{i}.entity_id",
                        issue=f"Entity '{entity_id}' not in BRAINSTORM",
                        available=sorted_entity_ids,
                        provided=raw_entity_id,
                    )
                )

    # 2. Validate tension decisions
    for i, decision in enumerate(output.get("tensions", [])):
        raw_tension_id = decision.get("tension_id")
        if raw_tension_id:
            tension_id, scope_error = _normalize_id(raw_tension_id, "tension")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"tensions.{i}.tension_id",
                        issue=scope_error,
                        available=sorted_tension_ids,
                        provided=raw_tension_id,
                    )
                )
            elif tension_id not in valid_tension_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"tensions.{i}.tension_id",
                        issue=f"Tension '{tension_id}' not in BRAINSTORM",
                        available=sorted_tension_ids,
                        provided=raw_tension_id,
                    )
                )

    # 3-4. Validate threads
    for i, thread in enumerate(output.get("threads", [])):
        raw_thread_tension_id = thread.get("tension_id")
        raw_alt_id = thread.get("alternative_id")

        # Normalize tension_id
        thread_tension_id: str | None = None
        tension_valid = False
        if raw_thread_tension_id:
            thread_tension_id, scope_error = _normalize_id(raw_thread_tension_id, "tension")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"threads.{i}.tension_id",
                        issue=scope_error,
                        available=sorted_tension_ids,
                        provided=raw_thread_tension_id,
                    )
                )
            elif thread_tension_id not in valid_tension_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"threads.{i}.tension_id",
                        issue=f"Tension '{thread_tension_id}' not in BRAINSTORM",
                        available=sorted_tension_ids,
                        provided=raw_thread_tension_id,
                    )
                )
            else:
                tension_valid = True

        # Check alternative (report even if tension was also invalid, so LLM gets all feedback)
        # Alternative IDs are not scoped (they're local to their tension)
        if thread_tension_id and raw_alt_id:
            valid_alts = alt_by_tension.get(thread_tension_id, set())
            if raw_alt_id not in valid_alts:
                errors.append(
                    SeedValidationError(
                        field_path=f"threads.{i}.alternative_id",
                        issue=f"Alternative '{raw_alt_id}' not in tension '{thread_tension_id}'",
                        available=sorted(valid_alts) if tension_valid else [],
                        provided=raw_alt_id,
                    )
                )

    # 5. Validate beat entity references
    for i, beat in enumerate(output.get("initial_beats", [])):
        for raw_entity_id in beat.get("entities", []):
            if raw_entity_id:
                entity_id, scope_error = _normalize_id(raw_entity_id, "entity")
                if scope_error:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.entities",
                            issue=scope_error,
                            available=sorted_entity_ids,
                            provided=raw_entity_id,
                        )
                    )
                elif entity_id not in valid_entity_ids:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.entities",
                            issue=f"Entity '{entity_id}' not in BRAINSTORM",
                            available=sorted_entity_ids,
                            provided=raw_entity_id,
                        )
                    )

        # Location is also an entity
        raw_location = beat.get("location")
        if raw_location:
            location, scope_error = _normalize_id(raw_location, "entity")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"initial_beats.{i}.location",
                        issue=scope_error,
                        available=sorted_entity_ids,
                        provided=raw_location,
                    )
                )
            elif location not in valid_entity_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"initial_beats.{i}.location",
                        issue=f"Location '{location}' not in BRAINSTORM entities",
                        available=sorted_entity_ids,
                        provided=raw_location,
                    )
                )

        # Location alternatives are also entities
        for raw_loc_alt in beat.get("location_alternatives", []):
            if raw_loc_alt:
                loc_alt, scope_error = _normalize_id(raw_loc_alt, "entity")
                if scope_error:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.location_alternatives",
                            issue=scope_error,
                            available=sorted_entity_ids,
                            provided=raw_loc_alt,
                        )
                    )
                elif loc_alt not in valid_entity_ids:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.location_alternatives",
                            issue=f"Location alternative '{loc_alt}' not in BRAINSTORM entities",
                            available=sorted_entity_ids,
                            provided=raw_loc_alt,
                        )
                    )

    # 6. Validate beat thread references (internal to SEED)
    for i, beat in enumerate(output.get("initial_beats", [])):
        for raw_thread_id in beat.get("threads", []):
            if raw_thread_id:
                thread_id, scope_error = _normalize_id(raw_thread_id, "thread")
                if scope_error:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.threads",
                            issue=scope_error,
                            available=sorted_thread_ids,
                            provided=raw_thread_id,
                        )
                    )
                elif thread_id not in seed_thread_ids:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.threads",
                            issue=f"Thread '{thread_id}' not defined in SEED threads",
                            available=sorted_thread_ids,
                            provided=raw_thread_id,
                        )
                    )

    # 7. Validate consequence thread references (internal to SEED)
    for i, consequence in enumerate(output.get("consequences", [])):
        raw_thread_id = consequence.get("thread_id")
        if raw_thread_id:
            thread_id, scope_error = _normalize_id(raw_thread_id, "thread")
            if scope_error:
                errors.append(
                    SeedValidationError(
                        field_path=f"consequences.{i}.thread_id",
                        issue=scope_error,
                        available=sorted_thread_ids,
                        provided=raw_thread_id,
                    )
                )
            elif thread_id not in seed_thread_ids:
                errors.append(
                    SeedValidationError(
                        field_path=f"consequences.{i}.thread_id",
                        issue=f"Thread '{thread_id}' not defined in SEED threads",
                        available=sorted_thread_ids,
                        provided=raw_thread_id,
                    )
                )

    # 8. Validate tension_impacts in beats
    for i, beat in enumerate(output.get("initial_beats", [])):
        for j, impact in enumerate(beat.get("tension_impacts", [])):
            raw_tension_id = impact.get("tension_id")
            if raw_tension_id:
                tension_id, scope_error = _normalize_id(raw_tension_id, "tension")
                if scope_error:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.tension_impacts.{j}.tension_id",
                            issue=scope_error,
                            available=sorted_tension_ids,
                            provided=raw_tension_id,
                        )
                    )
                elif tension_id not in valid_tension_ids:
                    errors.append(
                        SeedValidationError(
                            field_path=f"initial_beats.{i}.tension_impacts.{j}.tension_id",
                            issue=f"Tension '{tension_id}' not in BRAINSTORM",
                            available=sorted_tension_ids,
                            provided=raw_tension_id,
                        )
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
