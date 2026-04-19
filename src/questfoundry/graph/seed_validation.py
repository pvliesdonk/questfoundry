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
    return errors


def _check_upstream_contract(graph: Graph, errors: list[str]) -> None:
    """Verify BRAINSTORM foundation hasn't been corrupted (minus forbidden-types check).

    At SEED time, we don't fail on beat nodes (which SEED creates), but we do
    verify that entities and dilemmas remain valid per BRAINSTORM's contract.
    """
    # Inline imports avoid module-load circular dependencies.
    from questfoundry.graph.brainstorm_validation import (
        _check_dilemmas,
        _check_entities,
    )
    from questfoundry.graph.dream_validation import validate_vision_node

    # Check DREAM foundation first.
    vision_errors = validate_vision_node(graph)
    for e in vision_errors:
        errors.append(f"Output-0: DREAM contract violated post-SEED — {e}")

    # Check entity and dilemma structure (but skip R-3.8 forbidden types).
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    # R-1.1: minimum floors — at least one entity and one dilemma
    if not entity_nodes:
        errors.append(
            "Output-0: BRAINSTORM contract violated post-SEED — R-1.1: BRAINSTORM must produce at least one entity"
        )
    if not dilemma_nodes:
        errors.append(
            "Output-0: BRAINSTORM contract violated post-SEED — R-1.1: BRAINSTORM must produce at least one dilemma"
        )

    if entity_nodes:
        entity_errors: list[str] = []
        _check_entities(entity_nodes, entity_errors)
        for e in entity_errors:
            errors.append(f"Output-0: BRAINSTORM contract violated post-SEED — {e}")

    # Gather edges for dilemma checks.
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    anchored_to_edges = graph.get_edges(edge_type="anchored_to")

    dilemma_errors: list[str] = []
    _check_dilemmas(dilemma_nodes, has_answer_edges, anchored_to_edges, graph, dilemma_errors)
    for e in dilemma_errors:
        errors.append(f"Output-0: BRAINSTORM contract violated post-SEED — {e}")
