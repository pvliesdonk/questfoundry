"""Shared graph algorithms for pipeline stages.

Pure functions that operate on the graph without modifying it. These
replace Arc-dependent computations and can be used by both GROW
(for validation) and POLISH (for plan computation).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import normalize_scoped_id
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


def compute_active_flags_at_beat(graph: Graph, beat_id: str) -> set[frozenset[str]]:
    """Compute all valid state flag combinations at a beat position.

    Returns a set of frozensets, where each frozenset is one possible
    combination of active ``state_flag::*`` node IDs at this beat's
    position in the DAG.

    When a player reaches beat B, they have necessarily traversed some
    subset of commit beats. Each commit beat has ``grants`` edges to
    ``state_flag::*`` nodes. This function resolves those real node IDs
    (not synthetic dilemma:path strings) so that downstream consumers
    (prose feasibility, overlays) can cross-reference against entity
    overlay ``when`` fields, which also use ``state_flag::*`` IDs.

    Algorithm:
        1. Build predecessor adjacency and reverse-BFS from beat_id
           to find all ancestor beats (including beat_id itself).
        2. Filter ancestors to commit beats (effect="commits").
        3. Resolve each commit beat's ``grants`` edges to state_flag node IDs.
        4. Group by dilemma and compute Cartesian product (mutual exclusivity).

    Args:
        graph: Graph containing beat DAG with predecessor edges.
        beat_id: Beat node ID to compute flags for.

    Returns:
        Set of frozensets. Each frozenset contains the active state
        flag IDs for one valid traversal path. Returns ``{frozenset()}``
        if no commit beats are ancestors (empty flag set).

    Raises:
        ValueError: If beat_id is not a valid beat node.
    """
    beat_node = graph.get_node(beat_id)
    if beat_node is None or beat_node.get("type") != "beat":
        msg = f"Node {beat_id!r} is not a valid beat node"
        raise ValueError(msg)

    # Step 1: Build predecessor adjacency (child → parents)
    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # predecessor edge: from=child, to=parent (child depends on parent)
    parents: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            parents[from_id].append(to_id)

    # Step 2: Reverse BFS to find all ancestors of beat_id
    ancestors: set[str] = set()
    queue = list(parents.get(beat_id, []))
    while queue:
        current = queue.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        queue.extend(parents.get(current, []))

    # Step 3: Find commit beats among ancestors and resolve their state flags.
    #
    # Each commit beat has a ``grants`` edge to a ``state_flag::*`` node.
    # We use those real node IDs (not synthetic dilemma:path strings) so that
    # downstream consumers (prose feasibility, overlays) can cross-reference
    # state flags against entity overlay ``when`` fields, which also use
    # ``state_flag::*`` IDs.
    #
    # Group by dilemma so the Cartesian product respects mutual exclusivity:
    # a player can only be on one path per dilemma.

    # Build beat → grants targets (state flag IDs) from grants edges.
    beat_grants: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="grants"):
        beat_grants.setdefault(edge["from"], []).append(edge["to"])

    # Include beat_id itself as a candidate (it may be a commit beat)
    candidates = ancestors | {beat_id}

    # dilemma_id → list of state_flag node IDs (one per path of that dilemma)
    dilemma_flags: dict[str, list[str]] = {}

    for candidate_id in candidates:
        candidate_data = beat_nodes[candidate_id]
        impacts = candidate_data.get("dilemma_impacts", [])
        for impact in impacts:
            if impact.get("effect") != "commits":
                continue
            dilemma_id = impact.get("dilemma_id", "")
            if not dilemma_id:
                continue
            # Resolve to actual state_flag node IDs via grants edges.
            flag_ids = beat_grants.get(candidate_id, [])
            if not flag_ids:
                log.warning(
                    "commit_beat_no_grants",
                    beat_id=candidate_id,
                    dilemma_id=dilemma_id,
                )
                continue
            if len(flag_ids) > 1:
                log.warning(
                    "commit_beat_multiple_grants",
                    beat_id=candidate_id,
                    dilemma_id=dilemma_id,
                    flag_ids=flag_ids,
                )
            for flag_id in flag_ids:
                dilemma_flags.setdefault(dilemma_id, []).append(flag_id)

    if not dilemma_flags:
        return {frozenset()}

    # Step 4: Deduplicate flags per dilemma and compute Cartesian product
    per_dilemma_options: list[list[str]] = []
    for _dilemma_id, flags in sorted(dilemma_flags.items()):
        unique_flags = sorted(set(flags))
        per_dilemma_options.append(unique_flags)

    result: set[frozenset[str]] = set()
    for combo in product(*per_dilemma_options):
        result.add(frozenset(combo))

    return result


def arc_key_for_paths(
    path_nodes: dict[str, dict[str, Any]],
    path_ids: list[str],
) -> str:
    """Build a canonical arc key from path node IDs.

    Arc keys are sorted path ``raw_id`` values joined by ``"+"``.
    """
    raw_ids = sorted(str(path_nodes.get(pid, {}).get("raw_id", pid)) for pid in path_ids)
    return "+".join(raw_ids)


def compute_arc_traversals(graph: Graph) -> dict[str, list[str]]:
    """Compute arc traversals by walking the beat DAG.

    Each arc is a specific combination of path choices (one per dilemma).
    The traversal walks the DAG from the root beat following successor
    edges.  At each beat:
    - **One successor** → follow it.
    - **Multiple successors** → fork.  Follow successors whose path
      membership overlaps the arc's selected paths, or that have no
      ``belongs_to`` (zero-membership transition/gap beats).
      Successors on other paths are pruned.

    See Story Graph Ontology Part 3 "Total Order Per Arc".

    Args:
        graph: Graph containing dilemma, path, and beat nodes with
            ``belongs_to`` and ``predecessor`` edges.

    Returns:
        Dict mapping arc key to topologically sorted list of beat IDs.
        Returns empty dict if no dilemmas or paths exist.
    """
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    if not dilemma_nodes or not path_nodes:
        return {}

    # Build dilemma → paths mapping
    dilemma_paths: dict[str, list[str]] = defaultdict(list)
    for path_id, path_data in path_nodes.items():
        dilemma_id = path_data.get("dilemma_id")
        if dilemma_id:
            prefixed = normalize_scoped_id(dilemma_id, "dilemma")
            if prefixed in dilemma_nodes:
                dilemma_paths[prefixed].append(path_id)

    if not dilemma_paths:
        return {}

    for paths in dilemma_paths.values():
        paths.sort()

    sorted_dilemmas = sorted(dilemma_paths.keys())
    path_lists = [dilemma_paths[did] for did in sorted_dilemmas]

    # Build beat → path set
    beat_nodes = graph.get_nodes_by_type("beat")
    beat_path_set: dict[str, set[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        bid = edge["from"]
        pid = edge["to"]
        if bid in beat_nodes and pid in path_nodes:
            beat_path_set.setdefault(bid, set()).add(pid)

    # Build successor adjacency from predecessor edges
    successors_all: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    predecessors_all: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in graph.get_edges(edge_type="predecessor"):
        child = edge["from"]
        parent = edge["to"]
        if child in beat_nodes and parent in beat_nodes:
            successors_all[parent].append(child)
            predecessors_all[child].append(parent)

    roots = [bid for bid in beat_nodes if not predecessors_all[bid]]

    # For each arc, walk the DAG from root
    result: dict[str, list[str]] = {}
    for combo in product(*path_lists):
        path_combo = list(combo)
        arc_path_set = set(path_combo)
        path_raw_ids = sorted(path_nodes[pid].get("raw_id", pid) for pid in path_combo)
        arc_key = "+".join(path_raw_ids)

        # BFS walk: at each beat, follow successors that match this arc.
        # One successor → follow. Multiple → follow those on arc's paths
        # or with zero belongs_to. Prune the rest.
        reachable: set[str] = set()
        queue = list(roots)
        visited: set[str] = set()

        while queue:
            bid = queue.pop(0)
            if bid in visited:
                continue
            visited.add(bid)
            reachable.add(bid)

            for succ in successors_all[bid]:
                if succ in visited:
                    continue
                succ_paths = beat_path_set.get(succ)
                if not succ_paths:
                    # Zero-membership (transition/gap) → always follow
                    queue.append(succ)
                elif succ_paths & arc_path_set:
                    # Successor's paths overlap with arc → follow
                    queue.append(succ)
                # else: successor is on a different path → prune

        # Prune zero-membership beats (transition/gap) whose successors
        # are all outside reachable — bridges to paths not in this arc.
        to_remove = {
            bid
            for bid in reachable
            if not beat_path_set.get(bid)  # zero-membership only
            and successors_all[bid]
            and not any(s in reachable for s in successors_all[bid])
        }
        reachable -= to_remove

        sequence = _topological_sort_subset(reachable, successors_all)
        result[arc_key] = sequence

    return result


def compute_passage_traversals(graph: Graph) -> dict[str, list[str]]:
    """Compute passage traversal orders from graph structure.

    Builds on :func:`compute_arc_traversals` which provides beat sequences
    per arc.  Maps beats to passages via ``grouped_in`` edges
    (beat → passage), falling back to ``passage_from`` edges
    (passage → beat) for legacy graphs.

    Multiple beats in the same passage are deduplicated — each passage
    appears at most once per arc traversal, at its first occurrence.

    Args:
        graph: Graph with beat, path, passage nodes and ``grouped_in``
            (or ``passage_from``) edges.

    Returns:
        Dict mapping arc key (``"path_a+path_b"``) to ordered list of
        unique passage IDs.  Returns empty dict if no arc traversals can
        be computed (e.g. missing dilemma / path nodes).
    """
    arc_traversals = compute_arc_traversals(graph)
    if not arc_traversals:
        return {}

    # Build beat→passages lookup from grouped_in edges (primary)
    beat_to_passages: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get_edges(edge_type="grouped_in"):
        beat_to_passages[edge["from"]].append(edge["to"])

    # Fallback: passage_from edges (legacy, passage→beat)
    if not beat_to_passages:
        for edge in graph.get_edges(edge_type="passage_from"):
            beat_to_passages[edge["to"]].append(edge["from"])

    result: dict[str, list[str]] = {}
    for arc_key, beat_sequence in arc_traversals.items():
        seen: set[str] = set()
        passages: list[str] = []
        for beat_id in beat_sequence:
            for passage_id in beat_to_passages.get(beat_id, []):
                if passage_id not in seen:
                    seen.add(passage_id)
                    passages.append(passage_id)
        result[arc_key] = passages

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _topological_sort_subset(
    beat_set: set[str],
    successors_all: dict[str, list[str]],
) -> list[str]:
    """Topologically sort a subset of beats using successor edges.

    Uses Kahn's algorithm restricted to beats in ``beat_set``.
    Falls back to sorted order if cycles are detected.
    """
    if not beat_set:
        return []

    # Build in-degree map restricted to subset
    in_degree: dict[str, int] = dict.fromkeys(beat_set, 0)
    for parent in beat_set:
        for child in successors_all.get(parent, []):
            if child in beat_set:
                in_degree[child] = in_degree.get(child, 0) + 1

    queue_sorted = sorted(bid for bid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue_sorted:
        bid = queue_sorted.pop(0)
        result.append(bid)
        for child in successors_all.get(bid, []):
            if child in in_degree:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    # Insert in sorted position for determinism
                    import bisect

                    bisect.insort(queue_sorted, child)

    # Cycle fallback: if not all beats were emitted, append remaining sorted
    if len(result) < len(beat_set):
        remaining = sorted(beat_set - set(result))
        result.extend(remaining)

    return result
