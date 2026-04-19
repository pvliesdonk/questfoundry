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
