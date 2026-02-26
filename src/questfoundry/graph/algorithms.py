"""Shared graph algorithms for pipeline stages.

Pure functions that operate on the graph without modifying it. These
replace Arc-dependent computations and can be used by both GROW
(for validation) and POLISH (for plan computation).
"""

from __future__ import annotations

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
