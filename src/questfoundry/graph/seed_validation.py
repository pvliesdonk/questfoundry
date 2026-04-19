"""SEED Stage Output Contract validator.

Validates the graph satisfies every rule in
docs/design/procedures/seed.md §Stage Output Contract.

Called at SEED exit (from apply_seed_mutations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class SeedContractError(ValueError):
    """Raised when SEED's Stage Output Contract is violated."""


_VALID_DILEMMA_ROLES = frozenset({"hard", "soft"})
_VALID_RESIDUE_WEIGHTS = frozenset({"heavy", "light", "cosmetic"})
_VALID_ENDING_SALIENCES = frozenset({"high", "low", "none"})
_VALID_ORDERING_RELATIONSHIPS = frozenset({"wraps", "concurrent", "serial"})
_VALID_DISPOSITIONS = frozenset({"retained", "cut"})
_FORBIDDEN_NODE_TYPES = frozenset(
    {"passage", "state_flag", "intersection_group", "transition_beat", "choice"}
)
_MAX_ARC_COUNT = 16


def validate_seed_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies SEED's Stage Output Contract.

    Args:
        graph: Graph expected to contain SEED output.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []
    _check_upstream_contract(graph, errors)
    _check_entities(graph, errors)
    _check_paths_and_consequences(graph, errors)
    _check_beats(graph, errors)
    return errors


def _check_entities(graph: Graph, errors: list[str]) -> None:
    """Phase 1 entity-triage checks (R-1.1, R-1.2, R-1.4)."""
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    retained_location_count = 0
    for entity_id, entity in sorted(entity_nodes.items()):
        disposition = entity.get("disposition")
        if disposition is None:
            errors.append(
                f"R-1.1: entity {entity_id!r} has no disposition (must be 'retained' or 'cut')"
            )
            continue
        if disposition not in _VALID_DISPOSITIONS:
            errors.append(
                f"R-1.1: entity {entity_id!r} has invalid disposition {disposition!r}; "
                f"must be one of {sorted(_VALID_DISPOSITIONS)}"
            )
        if disposition == "retained" and entity.get("category") == "location":
            retained_location_count += 1

    # R-1.2: anchored_to from surviving dilemmas must not point to cut entities.
    for edge in sorted(
        graph.get_edges(edge_type="anchored_to"),
        key=lambda e: (e["from"], e["to"]),
    ):
        dilemma = dilemma_nodes.get(edge["from"])
        if dilemma is None:
            continue
        entity_data = entity_nodes.get(edge["to"])
        if entity_data is None:
            continue
        if entity_data.get("disposition") == "cut":
            errors.append(
                f"R-1.2: dilemma {edge['from']!r} is anchored to cut entity "
                f"{edge['to']!r}; re-anchor or cut the dilemma first"
            )

    if retained_location_count < 2:
        errors.append(
            f"R-1.4: SEED must retain ≥2 location entities, found {retained_location_count}"
        )


def _check_paths_and_consequences(graph: Graph, errors: list[str]) -> None:
    """Phase 3 path structure (R-3.1, R-3.2, R-3.3, R-3.4)."""
    path_nodes = graph.get_nodes_by_type("path")
    answer_nodes = graph.get_nodes_by_type("answer")
    consequence_nodes = graph.get_nodes_by_type("consequence")

    # R-3.1 + R-3.2: each explored answer has exactly one Path via `explores`.
    explores_edges = graph.get_edges(edge_type="explores")
    path_by_answer: dict[str, list[str]] = {}
    for edge in sorted(explores_edges, key=lambda e: (e["from"], e["to"])):
        path_by_answer.setdefault(edge["to"], []).append(edge["from"])

    for answer_id, answer in sorted(answer_nodes.items()):
        if not answer.get("explored"):
            continue
        paths = path_by_answer.get(answer_id, [])
        if len(paths) == 0:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has no path (expected exactly one)"
            )
        elif len(paths) > 1:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has {len(paths)} paths; "
                f"expected exactly one: {sorted(paths)}"
            )

    for path_id in sorted(path_nodes.keys()):
        if not path_id.startswith("path::"):
            errors.append(f"R-3.2: path id {path_id!r} missing 'path::' prefix")

    # R-3.3: every Path has ≥1 has_consequence edge.
    has_consequence_edges = graph.get_edges(edge_type="has_consequence")
    consequences_per_path: dict[str, list[str]] = {}
    for edge in has_consequence_edges:
        consequences_per_path.setdefault(edge["from"], []).append(edge["to"])

    for path_id in sorted(path_nodes.keys()):
        if not consequences_per_path.get(path_id):
            errors.append(f"R-3.3: path {path_id!r} has no has_consequence edge")

    # R-3.4: every Consequence has non-empty description + ≥1 ripple.
    for conseq_id, conseq in sorted(consequence_nodes.items()):
        if not conseq.get("description"):
            errors.append(f"R-3.4: consequence {conseq_id!r} has empty description")
        ripples = conseq.get("ripples", [])
        if not ripples:
            errors.append(f"R-3.4: consequence {conseq_id!r} has no ripples")


def _check_beats(graph: Graph, errors: list[str]) -> None:
    """Phase 3 beat structural rules (R-3.13, R-3.14, R-3.15)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beats_with_path: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beats_with_path.setdefault(edge["from"], []).append(edge["to"])

    for beat_id, beat in sorted(beat_nodes.items()):
        role = beat.get("role")
        is_structural = role in {"setup", "epilogue"}

        # R-3.13 + R-3.15: every beat has non-empty summary + entities.
        if not beat.get("summary"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty summary")
        if not beat.get("entities"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty entities list")

        # R-3.14: structural beats must have zero belongs_to + zero commits impact.
        if is_structural:
            paths = beats_with_path.get(beat_id, [])
            if paths:
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must have zero belongs_to "
                    f"edges, found {len(paths)}"
                )
            if any(impact.get("effect") == "commits" for impact in beat.get("dilemma_impacts", [])):
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must not contain "
                    f"dilemma_impacts.effect: commits"
                )


def _check_upstream_contract(graph: Graph, errors: list[str]) -> None:
    """Delegate to BRAINSTORM validator (with downstream-node types allowed).

    At SEED time, the graph legitimately contains beat/path/consequence nodes
    that BRAINSTORM's R-3.8 would forbid — skip_forbidden_types=True relaxes
    that one check while preserving all other upstream invariants.
    """
    # Inline import avoids module-load circular dependencies.
    from questfoundry.graph.brainstorm_validation import validate_brainstorm_output

    upstream = validate_brainstorm_output(graph, skip_forbidden_types=True)
    for e in upstream:
        errors.append(f"Output-0: BRAINSTORM contract violated post-SEED — {e}")
