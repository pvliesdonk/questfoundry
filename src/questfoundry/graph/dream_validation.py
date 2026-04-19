"""DREAM Stage Output Contract validator.

Validates the graph's vision node satisfies every rule in
docs/design/procedures/dream.md §Stage Output Contract.

Called at DREAM exit (from apply_dream_mutations) and at BRAINSTORM
entry (from pipeline/stages/brainstorm.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class DreamContractError(ValueError):
    """Raised when DREAM's Stage Output Contract is violated."""


_ALLOWED_POV_STYLES = frozenset(
    {
        "first_person",
        "second_person",
        "third_person_limited",
        "third_person_omniscient",
    }
)

_REQUIRED_NON_EMPTY_FIELDS = ("genre", "tone", "themes", "audience", "scope")


def validate_dream_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies DREAM's Stage Output Contract.

    Args:
        graph: Graph expected to contain a vision node after DREAM.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []

    # R-1.7: exactly one vision node
    vision_nodes = graph.get_nodes_by_type("vision")
    if len(vision_nodes) == 0:
        errors.append("R-1.7: no vision node found in graph")
        return errors
    if len(vision_nodes) > 1:
        node_ids = sorted(vision_nodes.keys())
        errors.append(
            f"R-1.7: expected exactly one vision node, found {len(vision_nodes)}: {node_ids}"
        )

    vision_id, vision = next(iter(vision_nodes.items()))

    # R-1.8: required fields must be present and non-empty
    for field in _REQUIRED_NON_EMPTY_FIELDS:
        value = vision.get(field)
        if value is None:
            errors.append(f"R-1.8: vision.{field} is missing")
        elif (isinstance(value, str) and not value.strip()) or (
            isinstance(value, list) and len(value) == 0
        ):
            errors.append(f"R-1.8: vision.{field} is empty")
        elif field == "scope" and isinstance(value, dict) and not value.get("story_size"):
            errors.append("R-1.8: vision.scope.story_size is empty")

    # R-1.9: pov_style, when present, must be one of the allowed values
    pov_style = vision.get("pov_style")
    if pov_style is not None and pov_style not in _ALLOWED_POV_STYLES:
        errors.append(
            f"R-1.9: vision.pov_style must be one of {sorted(_ALLOWED_POV_STYLES)}, "
            f"got {pov_style!r}"
        )

    # R-1.10: vision node must have no edges (incoming or outgoing)
    vision_edges_out = graph.get_edges(from_id=vision_id)
    vision_edges_in = graph.get_edges(to_id=vision_id)
    if vision_edges_out or vision_edges_in:
        errors.append(
            f"R-1.10: vision node {vision_id!r} must have no edges; "
            f"found {len(vision_edges_out)} outgoing and {len(vision_edges_in)} incoming"
        )

    # Output-5: no node types other than 'vision' may exist
    all_node_ids = graph._store.all_node_ids()
    forbidden_node_ids = [nid for nid in all_node_ids if not nid.startswith("vision")]
    if forbidden_node_ids:
        forbidden_types = sorted(
            {nid.split("::")[0] if "::" in nid else nid for nid in forbidden_node_ids}
        )
        errors.append(
            f"Output-5: only vision nodes are allowed after DREAM; "
            f"found other node type(s): {forbidden_types}"
        )

    # Output-6: human approval must be recorded as True
    if not vision.get("human_approved"):
        errors.append(
            "Output-6: DREAM vision missing recorded human approval "
            "(vision.human_approved must be True)"
        )

    return errors
