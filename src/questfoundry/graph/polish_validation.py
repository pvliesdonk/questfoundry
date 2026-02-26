"""POLISH validation functions.

Entry contract validation (validate_grow_output) checks GROW's output
before POLISH begins. Exit validation (validate_polish_output) checks
the passage graph after Phase 6 application.
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


# ---------------------------------------------------------------------------
# Phase 7: Passage graph validation (exit contract)
# ---------------------------------------------------------------------------


def validate_polish_output(graph: Graph) -> list[str]:
    """Validate the passage graph produced by POLISH Phases 4-6.

    Checks structural completeness, variant integrity, choice integrity,
    and feasibility constraints.

    Args:
        graph: Graph with passage layer applied.

    Returns:
        List of error strings. Empty means valid.
    """
    errors: list[str] = []

    beat_nodes = graph.get_nodes_by_type("beat")
    passage_nodes = graph.get_nodes_by_type("passage")

    if not passage_nodes:
        errors.append("No passage nodes found — Phase 6 may not have run")
        return errors

    # Build beat → passage mapping from grouped_in edges
    grouped_in_edges = graph.get_edges(edge_type="grouped_in")
    beat_to_passages: dict[str, list[str]] = {}
    passage_beats: dict[str, list[str]] = {pid: [] for pid in passage_nodes}

    for edge in grouped_in_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in passage_nodes:
            beat_to_passages.setdefault(from_id, []).append(to_id)
            passage_beats[to_id].append(from_id)

    _check_beat_grouping(beat_nodes, beat_to_passages, errors)
    _check_passage_beats(passage_nodes, passage_beats, errors)
    _check_start_passage(passage_beats, beat_nodes, graph, errors)
    _check_passage_reachability(passage_nodes, graph, errors)
    _check_variant_integrity(passage_nodes, graph, errors)
    _check_choice_integrity(graph, errors)
    _check_residue_ordering(graph, errors)

    return errors


def _check_beat_grouping(
    beat_nodes: dict[str, dict[str, Any]],
    beat_to_passages: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Every beat must be grouped into exactly one passage."""
    for beat_id in beat_nodes:
        passages = beat_to_passages.get(beat_id, [])
        if len(passages) == 0:
            # Skip micro/residue/sidetrack beats — they may not exist yet at validation time
            role = beat_nodes[beat_id].get("role", "")
            if role not in ("micro_beat", "residue_beat", "sidetrack_beat"):
                errors.append(f"Beat {beat_id} not grouped into any passage")
        elif len(passages) > 1:
            errors.append(f"Beat {beat_id} grouped into {len(passages)} passages: {passages}")


def _check_passage_beats(
    passage_nodes: dict[str, dict[str, Any]],
    passage_beats: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Every non-variant passage must have at least one beat."""
    for passage_id, pdata in passage_nodes.items():
        # Skip variant and diamond alt passages — they may not have direct beats
        if pdata.get("is_variant") or pdata.get("is_diamond_alt"):
            continue
        beats = passage_beats.get(passage_id, [])
        if not beats:
            errors.append(f"Passage {passage_id} has no beats (no grouped_in edges)")


def _check_start_passage(
    passage_beats: dict[str, list[str]],
    beat_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """Exactly one start passage must exist (containing the earliest beat)."""
    # Find root beats (no predecessor edges pointing TO them)
    # Exclude synthetic beats (micro_beat, residue_beat, sidetrack_beat) —
    # they're added fresh by Phase 6 with no predecessor edges
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    _synthetic_roles = {"micro_beat", "residue_beat", "sidetrack_beat"}
    beats_with_parents: set[str] = set()
    for edge in predecessor_edges:
        if edge["from"] in beat_nodes:
            beats_with_parents.add(edge["from"])

    root_beats = {
        bid
        for bid in beat_nodes
        if bid not in beats_with_parents and beat_nodes[bid].get("role", "") not in _synthetic_roles
    }

    # Find which passages contain root beats
    start_passages: set[str] = set()
    for passage_id, beats in passage_beats.items():
        if any(bid in root_beats for bid in beats):
            start_passages.add(passage_id)

    if len(start_passages) == 0:
        errors.append("No start passage found (no passage contains a root beat)")
    elif len(start_passages) > 1:
        errors.append(f"Multiple start passages found: {sorted(start_passages)}")


def _check_passage_reachability(
    passage_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """All passages should be reachable from start via choice edges."""
    choice_edges = graph.get_edges(edge_type="choice")
    precedes_edges = graph.get_edges(edge_type="precedes")

    if not choice_edges and len(passage_nodes) > 1:
        # No choices at all — can't check reachability
        return

    # Build adjacency from choice + precedes edges
    adj: dict[str, set[str]] = {pid: set() for pid in passage_nodes}
    for edge in choice_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in passage_nodes and to_id in passage_nodes:
            adj[from_id].add(to_id)
    for edge in precedes_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in passage_nodes and to_id in passage_nodes:
            adj[from_id].add(to_id)

    # BFS from all passages with no incoming edges
    incoming: dict[str, int] = dict.fromkeys(passage_nodes, 0)
    for _pid, neighbors in adj.items():
        for n in neighbors:
            incoming[n] = incoming.get(n, 0) + 1

    starts = {pid for pid, count in incoming.items() if count == 0}
    if not starts:
        # No clear start — skip check
        return

    from collections import deque

    visited: set[str] = set()
    queue: deque[str] = deque(starts)
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    unreachable = set(passage_nodes.keys()) - visited
    # Filter out variant passages — they're reachable through their base
    unreachable_real = {
        pid
        for pid in unreachable
        if not passage_nodes[pid].get("is_variant") and not passage_nodes[pid].get("is_diamond_alt")
    }

    if unreachable_real:
        errors.append(
            f"{len(unreachable_real)} passage(s) unreachable from start: "
            f"{sorted(unreachable_real)[:5]}"
        )


def _check_variant_integrity(
    passage_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """Every variant passage must have a variant_of edge to a base passage."""
    variant_of_edges = graph.get_edges(edge_type="variant_of")
    variants_with_base: set[str] = set()
    for edge in variant_of_edges:
        variants_with_base.add(edge["from"])

    for passage_id, pdata in passage_nodes.items():
        if pdata.get("is_variant") and passage_id not in variants_with_base:
            errors.append(f"Variant passage {passage_id} has no variant_of edge to a base passage")


def _check_choice_integrity(
    graph: Graph,
    errors: list[str],
) -> None:
    """Check choice edge integrity: unique labels per source passage."""
    choice_edges = graph.get_edges(edge_type="choice")

    # Group choices by source passage
    choices_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in choice_edges:
        from_id = edge["from"]
        choices_by_source.setdefault(from_id, []).append(edge)

    from collections import Counter

    for source_id, choices in choices_by_source.items():
        # Check for duplicate labels
        labels: list[str] = [c["label"] for c in choices if c.get("label")]
        counts = Counter(labels)
        duplicate_labels = sorted(label for label, count in counts.items() if count > 1)
        if duplicate_labels:
            errors.append(f"Passage {source_id} has duplicate choice labels: {duplicate_labels}")


def _check_residue_ordering(
    graph: Graph,
    errors: list[str],
) -> None:
    """Residue passages must precede their target shared passages."""
    precedes_edges = graph.get_edges(edge_type="precedes")
    passage_nodes = graph.get_nodes_by_type("passage")

    # Check that residue passages have precedes edges
    passages_with_precedes = {e["from"] for e in precedes_edges}
    for passage_id, pdata in passage_nodes.items():
        if pdata.get("is_residue") and passage_id not in passages_with_precedes:
            errors.append(f"Residue passage {passage_id} has no precedes edge to a target passage")
