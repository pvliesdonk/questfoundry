"""Shared graph algorithms for pipeline stages.

Pure functions that operate on the graph without modifying it. These
replace Arc-dependent computations and can be used by both GROW
(for validation) and POLISH (for plan computation).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def compute_active_flags_at_beat(graph: Graph, beat_id: str) -> set[frozenset[str]]:
    """Compute all valid state flag combinations at a beat position.

    Returns a set of frozensets, where each frozenset is one possible
    combination of active state flags at this beat's position in the DAG.

    A "state flag" here is a string identifier derived from a commit beat's
    dilemma/path membership. When a player reaches beat B, they have
    necessarily traversed some subset of commit beats. This function
    computes all valid subsets respecting mutual exclusivity (cannot
    have committed to both paths of the same dilemma).

    Algorithm:
        1. Build predecessor adjacency and reverse-BFS from beat_id
           to find all ancestor beats (including beat_id itself).
        2. Filter ancestors to commit beats (effect="commits").
        3. Group commit beats by dilemma, using belongs_to path as flag.
        4. Compute Cartesian product of per-dilemma flag options.

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

    # Step 3: Find commit beats among ancestors and group by dilemma
    # A commit beat has dilemma_impacts with effect="commits"
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        from_id = edge["from"]
        if from_id in beat_nodes:
            if from_id in beat_to_path:
                # Entry contract (validate_grow_output) enforces exactly-one belongs_to
                # per beat, so this should never happen on valid GROW output.
                msg = f"Beat {from_id!r} has multiple belongs_to edges"
                raise ValueError(msg)
            beat_to_path[from_id] = edge["to"]

    # Include beat_id itself as a candidate (it may be a commit beat)
    candidates = ancestors | {beat_id}

    # dilemma_id → list of state flag identifiers from different paths
    dilemma_flags: dict[str, list[str]] = {}

    for candidate_id in candidates:
        candidate_data = beat_nodes[candidate_id]
        impacts = candidate_data.get("dilemma_impacts", [])
        for impact in impacts:
            if impact.get("effect") == "commits":
                dilemma_id = impact.get("dilemma_id", "")
                path_id = beat_to_path.get(candidate_id, "")
                if dilemma_id and path_id:
                    # State flag: "{dilemma_id}:{path_id}"
                    flag = f"{dilemma_id}:{path_id}"
                    dilemma_flags.setdefault(dilemma_id, []).append(flag)

    if not dilemma_flags:
        return {frozenset()}

    # Step 4: Deduplicate flags per dilemma and compute Cartesian product
    # For each dilemma, the player can only be on one path, so each dilemma
    # contributes exactly one flag (or none if not yet committed).
    # We include "no flag for this dilemma" as an option only if the
    # beat is NOT downstream of a commit for that dilemma. But since
    # we only collected flags from ancestors that ARE commits, having
    # a flag means the dilemma IS committed. We just need unique flags.
    per_dilemma_options: list[list[str]] = []
    for _dilemma_id, flags in sorted(dilemma_flags.items()):
        unique_flags = sorted(set(flags))
        per_dilemma_options.append(unique_flags)

    # Cartesian product: one flag per committed dilemma
    result: set[frozenset[str]] = set()
    for combo in product(*per_dilemma_options):
        result.add(frozenset(combo))

    return result


def compute_arc_traversals(graph: Graph) -> dict[str, list[str]]:
    """Compute arc traversals from graph structure without stored Arc nodes.

    Replicates the logic of ``enumerate_arcs()`` in ``grow_algorithms.py``
    but as a pure read-only computation. For each Cartesian product
    combination of paths across dilemmas, collects all beats belonging
    to those paths and topologically sorts them.

    The arc key is the sorted path raw_ids joined by ``"+"``, matching
    the ``arc_id`` convention used by stored Arc nodes.

    Algorithm:
        1. Build dilemma → paths mapping from path node ``dilemma_id``.
        2. Build path → beat set mapping from ``belongs_to`` edges.
        3. Compute Cartesian product of paths (one per dilemma).
        4. For each combination, union the beat sets and topologically sort.

    Args:
        graph: Graph containing dilemma, path, and beat nodes with
            ``belongs_to`` and ``predecessor`` edges.

    Returns:
        Dict mapping arc key (``"path_a+path_b"``) to topologically sorted
        list of beat IDs. Returns empty dict if no dilemmas or paths exist.
    """
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    if not dilemma_nodes or not path_nodes:
        return {}

    # Step 1: Build dilemma → paths mapping
    dilemma_paths: dict[str, list[str]] = defaultdict(list)
    for path_id, path_data in path_nodes.items():
        dilemma_id = path_data.get("dilemma_id")
        if dilemma_id:
            # Normalize: ensure "dilemma::" prefix
            prefixed = (
                dilemma_id if dilemma_id.startswith("dilemma::") else f"dilemma::{dilemma_id}"
            )
            if prefixed in dilemma_nodes:
                dilemma_paths[prefixed].append(path_id)

    if not dilemma_paths or any(not pl for pl in dilemma_paths.values()):
        return {}

    # Sort paths within each dilemma for determinism
    for paths in dilemma_paths.values():
        paths.sort()

    sorted_dilemmas = sorted(dilemma_paths.keys())
    path_lists = [dilemma_paths[did] for did in sorted_dilemmas]

    # Step 2: Build path → beat set mapping via belongs_to edges
    path_beat_sets: dict[str, set[str]] = defaultdict(set)
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_nodes = graph.get_nodes_by_type("beat")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        if beat_id in beat_nodes and path_id in path_nodes:
            path_beat_sets[path_id].add(beat_id)

    # Step 3: Build predecessor adjacency for topological sort
    # successors_all: parent → [children] (prerequisite → dependents)
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    successors_all: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]  # dependent (child)
        to_id = edge["to"]  # prerequisite (parent)
        if from_id in beat_nodes and to_id in beat_nodes:
            successors_all[to_id].append(from_id)

    # Step 4: Cartesian product of paths, build traversals
    result: dict[str, list[str]] = {}
    for combo in product(*path_lists):
        path_combo = list(combo)
        path_raw_ids = sorted(path_nodes[pid].get("raw_id", pid) for pid in path_combo)
        arc_key = "+".join(path_raw_ids)

        # Collect beats from all selected paths
        beat_set: set[str] = set()
        for pid in path_combo:
            beat_set.update(path_beat_sets.get(pid, set()))

        # Topological sort within the beat subset
        sequence = _topological_sort_subset(beat_set, successors_all)
        result[arc_key] = sequence

    return result


def _topological_sort_subset(
    beat_set: set[str],
    successors_all: dict[str, list[str]],
) -> list[str]:
    """Topologically sort a subset of beats using Kahn's algorithm.

    Uses alphabetical tie-breaking for determinism. Falls back to
    sorted order if a cycle is detected.

    Args:
        beat_set: Set of beat IDs to sort.
        successors_all: Successor adjacency for all beats in the full graph.

    Returns:
        Topologically sorted list of beat IDs.
    """
    if not beat_set:
        return []

    # Build in-degree restricted to the subset
    in_degree: dict[str, int] = dict.fromkeys(beat_set, 0)
    for bid in beat_set:
        for succ in successors_all.get(bid, []):
            if succ in beat_set:
                in_degree[succ] = in_degree.get(succ, 0) + 1

    queue = sorted(bid for bid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        new_ready: list[str] = []
        for succ in successors_all.get(node, []):
            if succ in beat_set:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    new_ready.append(succ)
        # Insert in sorted order for determinism
        queue = sorted(queue + new_ready)

    if len(result) != len(beat_set):
        # Fallback for cycles: sorted order
        return sorted(beat_set)

    return result
