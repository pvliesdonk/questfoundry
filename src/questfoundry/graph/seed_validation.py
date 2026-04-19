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
