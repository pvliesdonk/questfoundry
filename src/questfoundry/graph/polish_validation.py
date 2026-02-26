"""POLISH entry contract validation.

Validates that GROW's output meets POLISH's input requirements before
the POLISH stage begins. Failures indicate issues in GROW or SEED
that must be fixed there, not in POLISH.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def validate_grow_output(graph: Graph) -> list[str]:
    """Verify GROW's output meets POLISH's input contract.

    Args:
        graph: Graph containing GROW stage output.

    Returns:
        List of error strings. Empty means valid.
    """
    errors: list[str] = []

    # 1. Beat nodes exist with summaries and dilemma_impacts
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        errors.append("No beat nodes found in graph")
    else:
        for beat_id, beat_data in beat_nodes.items():
            if not beat_data.get("summary"):
                errors.append(f"Beat {beat_id} missing summary")
            # Accept both plural "dilemma_impacts" and legacy singular "dilemma_impact"
            # from older GROW outputs that used the singular key.
            if "dilemma_impacts" not in beat_data and "dilemma_impact" not in beat_data:
                errors.append(f"Beat {beat_id} missing dilemma_impacts")

    # 2. No cycles in predecessor DAG
    if beat_nodes:
        _check_predecessor_cycles(graph, beat_nodes, errors)

    # 3. Every beat has exactly one belongs_to edge
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beats_with_path: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes:
            beats_with_path.setdefault(from_id, []).append(to_id)

    for beat_id in beat_nodes:
        paths = beats_with_path.get(beat_id, [])
        if len(paths) == 0:
            errors.append(f"Beat {beat_id} has no belongs_to edge (no path assignment)")
        elif len(paths) > 1:
            errors.append(
                f"Beat {beat_id} has {len(paths)} belongs_to edges (must have exactly 1): {paths}"
            )

    # 4. State flag nodes exist for explored dilemmas
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    state_flag_nodes = graph.get_nodes_by_type("state_flag")
    explored_dilemmas = {
        did
        for did, ddata in dilemma_nodes.items()
        # GROW may omit status for dilemmas it fully explored; treat None as "explored".
        if ddata.get("status") in ("explored", None)
    }
    dilemmas_with_flags: set[str] = set()
    for _flag_id, flag_data in state_flag_nodes.items():
        dilemma_ref = flag_data.get("dilemma_id") or flag_data.get("dilemma")
        if dilemma_ref:
            dilemmas_with_flags.add(dilemma_ref)

    for dilemma_id in explored_dilemmas:
        if dilemma_id not in dilemmas_with_flags:
            errors.append(f"Explored dilemma {dilemma_id} has no state flag nodes")

    # 5. Dilemma nodes have dilemma_role set
    for dilemma_id, dilemma_data in dilemma_nodes.items():
        if not dilemma_data.get("dilemma_role"):
            errors.append(f"Dilemma {dilemma_id} missing dilemma_role (hard/soft)")

    # 6. Intersection groups reference beats from different paths only
    intersection_groups = graph.get_nodes_by_type("intersection_group")
    for group_id, group_data in intersection_groups.items():
        _check_intersection_group_paths(
            graph, group_id, group_data, beat_nodes, beats_with_path, errors
        )

    return errors


def _check_predecessor_cycles(
    graph: Graph,
    beat_nodes: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    """Check for cycles in predecessor edges using Kahn's algorithm."""
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # Build adjacency: predecessor edges mean "from depends on to"
    # i.e., edge from A to B means "A's predecessor is B" (B comes before A)
    in_degree: dict[str, int] = dict.fromkeys(beat_nodes, 0)
    adj: dict[str, list[str]] = {bid: [] for bid in beat_nodes}

    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            in_degree[from_id] += 1
            adj[to_id].append(from_id)

    queue = [bid for bid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(beat_nodes):
        cycle_members = [bid for bid, deg in in_degree.items() if deg > 0]
        errors.append(
            f"Cycle detected in predecessor DAG among {len(cycle_members)} beats: "
            f"{', '.join(sorted(cycle_members)[:5])}" + ("..." if len(cycle_members) > 5 else "")
        )


def _check_intersection_group_paths(
    graph: Graph,  # noqa: ARG001
    group_id: str,
    group_data: dict[str, Any],
    beat_nodes: dict[str, dict[str, Any]],
    beats_with_path: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Check that an intersection group's beats come from different paths."""
    # Intersection group nodes store beat IDs in node_ids field
    node_ids = group_data.get("node_ids", [])
    if not node_ids:
        errors.append(f"Intersection group {group_id} has empty node_ids")
        return

    paths_seen: set[str] = set()
    for beat_id in node_ids:
        if beat_id not in beat_nodes:
            continue
        beat_paths = beats_with_path.get(beat_id, [])
        for path_id in beat_paths:
            if path_id in paths_seen:
                errors.append(
                    f"Intersection group {group_id} has multiple beats from the same path {path_id}"
                )
                return
            paths_seen.add(path_id)
