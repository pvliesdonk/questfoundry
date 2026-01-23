"""Pure graph algorithms for the GROW stage.

These functions operate on Graph objects and return structured results.
Phase orchestration lives in pipeline/stages/grow.py.

Algorithm summary:
- validate_beat_dag: Kahn's algorithm for cycle detection in requires edges
- validate_commits_beats: Verify each explored tension has commits beat per thread
- topological_sort_beats: Stable topological sort with alphabetical tie-breaking
- enumerate_arcs: Cartesian product of threads across tensions
- compute_divergence_points: Find where arcs diverge from the spine
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from questfoundry.graph.mutations import GrowErrorCategory, GrowValidationError
from questfoundry.models.grow import Arc

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Maximum number of arcs before triggering COMBINATORIAL error.
# With 2 tensions x 2 threads each = 4 arcs. With 5 tensions x 2 threads = 32 arcs.
# Beyond 32, the combinatorial explosion makes the story unmanageable.
_MAX_ARC_COUNT = 32


@dataclass
class DivergenceInfo:
    """Information about where an arc diverges from the spine.

    Attributes:
        arc_id: The arc that diverges.
        diverges_from: The arc it diverges from (spine).
        diverges_at: The last shared beat before divergence.
    """

    arc_id: str
    diverges_from: str
    diverges_at: str | None = None


# ---------------------------------------------------------------------------
# Phase 1: DAG Validation
# ---------------------------------------------------------------------------


def validate_beat_dag(graph: Graph) -> list[GrowValidationError]:
    """Validate that beat requires edges form a DAG (no cycles).

    Uses Kahn's algorithm: if all nodes can be processed, there's no cycle.
    If nodes remain after processing, they're in a cycle.

    Args:
        graph: Graph containing beat nodes and requires edges.

    Returns:
        List of validation errors. Empty if DAG is valid.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    # Build adjacency list and in-degree count from requires edges
    # requires edge: from=dependent_beat, to=prerequisite_beat
    # So the DAG direction is: prerequisite → dependent
    in_degree: dict[str, int] = dict.fromkeys(beat_nodes, 0)
    successors: dict[str, list[str]] = {bid: [] for bid in beat_nodes}

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="requires")
    for edge in requires_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        # from requires to: to is prerequisite of from
        # In DAG terms: to → from (to comes before from)
        if from_id in beat_nodes and to_id in beat_nodes:
            in_degree[from_id] += 1
            successors[to_id].append(from_id)

    # Kahn's algorithm
    queue = deque(sorted(bid for bid, deg in in_degree.items() if deg == 0))
    processed = 0

    while queue:
        node = queue.popleft()
        processed += 1
        for successor in sorted(successors[node]):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if processed == len(beat_nodes):
        return []

    # Find nodes in cycles (those with remaining in-degree > 0)
    cycle_nodes = [bid for bid, deg in in_degree.items() if deg > 0]
    return [
        GrowValidationError(
            field_path="beat_dag",
            issue=f"Cycle detected involving {len(cycle_nodes)} beats: {', '.join(sorted(cycle_nodes)[:5])}",
            available=sorted(cycle_nodes),
            category=GrowErrorCategory.STRUCTURAL,
        )
    ]


def validate_commits_beats(graph: Graph) -> list[GrowValidationError]:
    """Validate that each explored tension has a commits beat per thread.

    For each tension that has threads exploring it, every thread must have
    at least one beat with a tension_impact of effect="commits" for that tension.

    Args:
        graph: Graph containing tension, thread, and beat nodes.

    Returns:
        List of validation errors for threads missing commits beats.
    """
    errors: list[GrowValidationError] = []

    # Get all tensions that have exploring threads
    tension_nodes = graph.get_nodes_by_type("tension")
    thread_nodes = graph.get_nodes_by_type("thread")

    # Build tension → threads mapping via explores edges
    tension_threads: dict[str, list[str]] = defaultdict(list)
    explores_edges = graph.get_edges(from_id=None, to_id=None, edge_type="explores")
    for edge in explores_edges:
        thread_id = edge["from"]
        tension_id = edge["to"]
        if thread_id in thread_nodes and tension_id in tension_nodes:
            tension_threads[tension_id].append(thread_id)

    # Build thread → beats mapping via belongs_to edges
    thread_beats: dict[str, list[str]] = defaultdict(list)
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        thread_id = edge["to"]
        thread_beats[thread_id].append(beat_id)

    # For each tension's threads, check for commits beats
    beat_nodes = graph.get_nodes_by_type("beat")
    for tension_id, threads in sorted(tension_threads.items()):
        tension_raw = tension_nodes[tension_id].get("raw_id", tension_id)
        for thread_id in sorted(threads):
            thread_raw = thread_nodes[thread_id].get("raw_id", thread_id)
            beats_in_thread = thread_beats.get(thread_id, [])

            has_commits = False
            for beat_id in beats_in_thread:
                beat_data = beat_nodes.get(beat_id, {})
                impacts = beat_data.get("tension_impacts", [])
                for impact in impacts:
                    if (
                        impact.get("tension_id") == tension_raw
                        and impact.get("effect") == "commits"
                    ):
                        has_commits = True
                        break
                if has_commits:
                    break

            if not has_commits:
                errors.append(
                    GrowValidationError(
                        field_path=f"thread.{thread_raw}.commits",
                        issue=(
                            f"Thread '{thread_raw}' has no commits beat for tension '{tension_raw}'"
                        ),
                        category=GrowErrorCategory.STRUCTURAL,
                    )
                )

    return errors


# ---------------------------------------------------------------------------
# Topological Sort
# ---------------------------------------------------------------------------


def topological_sort_beats(graph: Graph, beat_ids: list[str]) -> list[str]:
    """Topologically sort a subset of beats using requires edges.

    Uses Kahn's algorithm with alphabetical tie-breaking for determinism.
    Only considers requires edges between beats in the provided set.

    Args:
        graph: Graph containing requires edges.
        beat_ids: Subset of beat node IDs to sort.

    Returns:
        Sorted list of beat IDs (prerequisites first).

    Raises:
        ValueError: If a cycle is detected in the beat subset.
    """
    if not beat_ids:
        return []

    beat_set = set(beat_ids)

    # Build adjacency within the subset
    in_degree: dict[str, int] = dict.fromkeys(beat_set, 0)
    successors: dict[str, list[str]] = {bid: [] for bid in beat_set}

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="requires")
    for edge in requires_edges:
        from_id = edge["from"]  # dependent
        to_id = edge["to"]  # prerequisite
        if from_id in beat_set and to_id in beat_set:
            in_degree[from_id] += 1
            successors[to_id].append(from_id)

    # Kahn's with alphabetical tie-breaking (using sorted heap simulation)
    # Use a sorted list as a priority queue for determinism
    queue = sorted(bid for bid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)  # Take alphabetically first
        result.append(node)
        new_ready = []
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                new_ready.append(successor)
        # Insert new ready nodes maintaining sorted order
        queue = sorted(queue + new_ready)

    if len(result) != len(beat_set):
        remaining = beat_set - set(result)
        raise ValueError(f"Cycle detected in beat subset: {sorted(remaining)}")

    return result


# ---------------------------------------------------------------------------
# Phase 5: Arc Enumeration
# ---------------------------------------------------------------------------


def enumerate_arcs(graph: Graph) -> list[Arc]:
    """Enumerate all arcs from the Cartesian product of threads across tensions.

    For each tension, collects its threads. Takes the Cartesian product across
    all tensions to produce arc combinations. Each arc gets the beats that
    belong to ANY of its constituent threads, topologically sorted.

    The spine arc contains all canonical threads. Branch arcs contain at least
    one non-canonical thread.

    Args:
        graph: Graph containing tension, thread, and beat nodes.

    Returns:
        List of Arc models, spine first, then branches sorted by ID.

    Raises:
        None - returns empty list if no tensions/threads exist.
    """
    tension_nodes = graph.get_nodes_by_type("tension")
    thread_nodes = graph.get_nodes_by_type("thread")

    if not tension_nodes or not thread_nodes:
        return []

    # Build tension → threads mapping (sorted for determinism)
    tension_threads: dict[str, list[str]] = defaultdict(list)
    explores_edges = graph.get_edges(from_id=None, to_id=None, edge_type="explores")
    for edge in explores_edges:
        thread_id = edge["from"]
        tension_id = edge["to"]
        if thread_id in thread_nodes and tension_id in tension_nodes:
            tension_threads[tension_id].append(thread_id)

    # Sort threads within each tension for determinism
    for threads in tension_threads.values():
        threads.sort()

    # Get thread lists per tension (sorted by tension ID for determinism)
    sorted_tensions = sorted(tension_threads.keys())
    thread_lists = [tension_threads[tid] for tid in sorted_tensions]

    if not thread_lists or any(not tl for tl in thread_lists):
        return []

    # Build thread → beat set mapping via belongs_to
    thread_beat_sets: dict[str, set[str]] = defaultdict(set)
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        thread_id = edge["to"]
        thread_beat_sets[thread_id].add(beat_id)

    # Cartesian product of threads
    arcs: list[Arc] = []
    for combo in product(*thread_lists):
        thread_combo = list(combo)
        # Get raw_ids for arc naming (sorted alphabetically)
        thread_raw_ids = sorted(thread_nodes[tid].get("raw_id", tid) for tid in thread_combo)
        arc_id = "+".join(thread_raw_ids)

        # Collect beats: beats that belong to ANY thread in the combo
        beat_set: set[str] = set()
        for tid in thread_combo:
            beat_set.update(thread_beat_sets.get(tid, set()))

        # Topological sort of the beats
        try:
            sequence = topological_sort_beats(graph, list(beat_set))
        except ValueError:
            sequence = sorted(beat_set)  # Fallback for cycles (Phase 1 should catch)

        # Determine if spine (all canonical)
        is_spine = all(thread_nodes[tid].get("is_canonical", False) for tid in thread_combo)

        arcs.append(
            Arc(
                arc_id=arc_id,
                arc_type="spine" if is_spine else "branch",
                threads=thread_raw_ids,
                sequence=sequence,
            )
        )

    # Check combinatorial limit
    if len(arcs) > _MAX_ARC_COUNT:
        # This will be caught by the phase and raised as GrowMutationError
        raise ValueError(
            f"Arc count ({len(arcs)}) exceeds limit of {_MAX_ARC_COUNT}. "
            f"Reduce the number of tensions or threads."
        )

    # Sort: spine first, then branches by ID
    spine_arcs = [a for a in arcs if a.arc_type == "spine"]
    branch_arcs = sorted(
        (a for a in arcs if a.arc_type == "branch"),
        key=lambda a: a.arc_id,
    )
    return spine_arcs + branch_arcs


# ---------------------------------------------------------------------------
# Phase 6: Divergence Points
# ---------------------------------------------------------------------------


def compute_divergence_points(
    arcs: list[Arc],
    spine_arc_id: str | None = None,
) -> dict[str, DivergenceInfo]:
    """Find where branch arcs diverge from the spine arc.

    Walks the sequences of spine and each branch arc in parallel.
    The divergence point is the last shared beat before sequences differ.

    Args:
        arcs: List of Arc models (must include spine).
        spine_arc_id: ID of the spine arc. If None, detected from arc_type.

    Returns:
        Dict mapping branch arc_id → DivergenceInfo.
        Empty dict if no spine arc found or no branches.
    """
    # Find spine
    spine: Arc | None = None
    for arc in arcs:
        if spine_arc_id and arc.arc_id == spine_arc_id:
            spine = arc
            break
        if arc.arc_type == "spine":
            spine = arc
            break

    if spine is None:
        return {}

    result: dict[str, DivergenceInfo] = {}

    for arc in arcs:
        if arc.arc_type == "spine":
            continue

        # Walk sequences in parallel to find last shared beat
        last_shared: str | None = None
        for spine_beat, branch_beat in zip(spine.sequence, arc.sequence, strict=False):
            if spine_beat == branch_beat:
                last_shared = spine_beat
            else:
                break

        result[arc.arc_id] = DivergenceInfo(
            arc_id=arc.arc_id,
            diverges_from=spine.arc_id,
            diverges_at=last_shared,
        )

    return result


# ---------------------------------------------------------------------------
# Phase 7: Convergence Points
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceInfo:
    """Information about where a branch arc converges back to the spine.

    Attributes:
        arc_id: The branch arc that converges.
        converges_to: The arc it converges to (spine).
        converges_at: The first shared beat after divergence.
    """

    arc_id: str
    converges_to: str
    converges_at: str | None = None


def find_convergence_points(
    graph: Graph,  # noqa: ARG001 - available for future validation
    arcs: list[Arc],
    divergence_map: dict[str, DivergenceInfo] | None = None,
    spine_arc_id: str | None = None,
) -> dict[str, ConvergenceInfo]:
    """Find where branch arcs converge back to the spine.

    For each diverged branch arc, looks for shared beats that appear in
    both the spine and branch sequences AFTER the divergence point.
    The first such shared beat is the convergence point.

    Args:
        graph: Graph (reserved for future validation).
        arcs: List of Arc models.
        divergence_map: Pre-computed divergence info. If None, computed internally.
        spine_arc_id: ID of the spine arc. If None, detected from arc_type.

    Returns:
        Dict mapping branch arc_id to ConvergenceInfo.
        Empty dict if no convergence found.
    """
    # Find spine
    spine: Arc | None = None
    for arc in arcs:
        if spine_arc_id and arc.arc_id == spine_arc_id:
            spine = arc
            break
        if arc.arc_type == "spine":
            spine = arc
            break

    if spine is None:
        return {}

    # Compute divergence if not provided
    if divergence_map is None:
        divergence_map = compute_divergence_points(arcs, spine_arc_id)

    result: dict[str, ConvergenceInfo] = {}
    spine_seq_set = set(spine.sequence)

    for arc in arcs:
        if arc.arc_type == "spine":
            continue

        div_info = divergence_map.get(arc.arc_id)
        if not div_info:
            continue

        # Find beats in branch after divergence point
        diverge_at = div_info.diverges_at
        if diverge_at and diverge_at in arc.sequence:
            div_idx = arc.sequence.index(diverge_at)
            # Look at beats after the divergence point in the branch
            branch_after_div = arc.sequence[div_idx + 1 :]
        else:
            # No divergence point means they diverge from the start
            branch_after_div = arc.sequence

        # Find first shared beat after divergence
        converges_at: str | None = None
        for beat_id in branch_after_div:
            if beat_id in spine_seq_set:
                converges_at = beat_id
                break

        result[arc.arc_id] = ConvergenceInfo(
            arc_id=arc.arc_id,
            converges_to=spine.arc_id,
            converges_at=converges_at,
        )

    return result


# ---------------------------------------------------------------------------
# Phase 11: BFS Reachability
# ---------------------------------------------------------------------------


def bfs_reachable(graph: Graph, start_node_id: str, edge_types: list[str]) -> set[str]:
    """Find all nodes reachable from start_node_id via specified edge types.

    Standard BFS following edges of the given types. Follows edges where
    the current node is the 'from' endpoint.

    Args:
        graph: Graph to traverse.
        start_node_id: Node to start BFS from.
        edge_types: Edge types to follow.

    Returns:
        Set of reachable node IDs (includes start_node_id).
    """
    if not graph.has_node(start_node_id):
        return set()

    visited: set[str] = set()
    queue = deque([start_node_id])

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        for edge_type in edge_types:
            edges = graph.get_edges(from_id=node_id, to_id=None, edge_type=edge_type)
            for edge in edges:
                to_id = edge["to"]
                if to_id not in visited:
                    queue.append(to_id)

    return visited
