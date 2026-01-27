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

import contextlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import normalize_scoped_id
from questfoundry.graph.mutations import GrowErrorCategory, GrowValidationError
from questfoundry.models.grow import Arc

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Maximum number of arcs before triggering COMBINATORIAL error.
# With 2 tensions x 2 threads each = 4 arcs. With 5 tensions x 2 threads = 32 arcs.
# With 6 tensions x 2 threads = 64 arcs. Beyond 64 arcs, processing becomes very
# expensive and the story structure may be difficult to navigate.
_MAX_ARC_COUNT = 64


def build_tension_threads(graph: Graph) -> dict[str, list[str]]:
    """Build tension → threads mapping from thread node tension_id properties.

    Uses the tension_id property on thread nodes instead of explores edges,
    since explores edges point to alternatives (not tensions) in real SEED output.

    Args:
        graph: Graph containing tension and thread nodes.

    Returns:
        Dict mapping tension node ID → list of thread node IDs.
    """
    tension_nodes = graph.get_nodes_by_type("tension")
    thread_nodes = graph.get_nodes_by_type("thread")
    tension_threads: dict[str, list[str]] = defaultdict(list)
    for thread_id, thread_data in thread_nodes.items():
        tension_id = thread_data.get("tension_id")
        if tension_id:
            prefixed = normalize_scoped_id(tension_id, "tension")
            if prefixed in tension_nodes:
                tension_threads[prefixed].append(thread_id)
    return tension_threads


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

    # Build tension → threads mapping from thread node tension_id properties
    tension_threads = build_tension_threads(graph)

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
                    if impact.get("tension_id") == tension_id and impact.get("effect") == "commits":
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

    # Build tension → threads mapping from thread node tension_id properties
    tension_threads = build_tension_threads(graph)

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
                paths=thread_raw_ids,  # paths is the new field name (was threads)
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


# ---------------------------------------------------------------------------
# Phase 3: Knot Detection
# ---------------------------------------------------------------------------


@dataclass
class KnotCandidate:
    """A group of beats that share signals and could form a knot.

    Attributes:
        beat_ids: Beat node IDs that share signals.
        signal_type: What signal links them (location, entity).
        shared_value: The shared signal value.
    """

    beat_ids: list[str]
    signal_type: str
    shared_value: str


def build_knot_candidates(graph: Graph) -> list[KnotCandidate]:
    """Find beats that share signals and could form knots.

    Groups beats by shared locations/location_alternatives and shared entities.
    Only considers beats from different tensions (same tension = alternative).

    Args:
        graph: Graph with beat, thread, and tension nodes.

    Returns:
        List of KnotCandidate groups, prioritizing location overlap.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    # Build beat → tension mapping via belongs_to → explores
    beat_tensions = _build_beat_tensions(graph, beat_nodes)

    # Group by location overlap (highest priority)
    location_groups = _group_by_location(beat_nodes, beat_tensions)

    # Group by shared entity
    entity_groups = _group_by_entity(graph, beat_nodes, beat_tensions)

    return location_groups + entity_groups


def _build_beat_tensions(graph: Graph, beat_nodes: dict[str, Any]) -> dict[str, set[str]]:
    """Map each beat to its tension IDs (via thread → tension edges).

    Returns:
        Dict mapping beat_id → set of tension raw_ids.
    """
    # thread → tension mapping (from thread node tension_id properties)
    thread_tension: dict[str, str] = {}
    thread_nodes = graph.get_nodes_by_type("thread")
    tension_nodes = graph.get_nodes_by_type("tension")
    for thread_id, thread_data in thread_nodes.items():
        tension_id = thread_data.get("tension_id")
        if tension_id:
            prefixed = normalize_scoped_id(tension_id, "tension")
            if prefixed in tension_nodes:
                tension_raw = tension_nodes[prefixed].get("raw_id", prefixed)
                thread_tension[thread_id] = tension_raw

    # beat → tensions via belongs_to
    beat_tensions: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        thread_id = edge["to"]
        if beat_id in beat_nodes and thread_id in thread_tension:
            beat_tensions[beat_id].add(thread_tension[thread_id])

    return beat_tensions


def _group_by_location(
    beat_nodes: dict[str, Any],
    beat_tensions: dict[str, set[str]],
) -> list[KnotCandidate]:
    """Group beats by location overlap (primary location or alternatives).

    Two beats have location overlap if:
    - Beat A's location is in Beat B's location_alternatives, or vice versa
    - They share the same primary location
    """
    # Build location → beats mapping
    location_beats: dict[str, list[str]] = defaultdict(list)

    for beat_id, beat_data in beat_nodes.items():
        primary = beat_data.get("location")
        if primary:
            location_beats[primary].append(beat_id)
        alternatives = beat_data.get("location_alternatives", [])
        for alt in alternatives:
            location_beats[alt].append(beat_id)

    candidates: list[KnotCandidate] = []
    for location, beats in sorted(location_beats.items()):
        if len(beats) < 2:
            continue
        # Filter to beats from different tensions
        multi_tension = _filter_different_tensions(beats, beat_tensions)
        if len(multi_tension) >= 2:
            candidates.append(
                KnotCandidate(
                    beat_ids=sorted(multi_tension),
                    signal_type="location",
                    shared_value=location,
                )
            )

    return candidates


def _group_by_entity(
    graph: Graph,
    beat_nodes: dict[str, Any],
    beat_tensions: dict[str, set[str]],
) -> list[KnotCandidate]:
    """Group beats by shared entity references."""
    # Build entity → beats mapping from features edges
    entity_beats: dict[str, list[str]] = defaultdict(list)
    features_edges = graph.get_edges(from_id=None, to_id=None, edge_type="features")
    for edge in features_edges:
        beat_id = edge["from"]
        entity_id = edge["to"]
        if beat_id in beat_nodes:
            entity_beats[entity_id].append(beat_id)

    # Also check entity references in beat data.
    # beat_data["entities"] may contain raw IDs ("mentor") or prefixed ("entity::mentor").
    # Normalize to prefixed form to match features edges.
    for beat_id, beat_data in beat_nodes.items():
        entities = beat_data.get("entities", [])
        for entity_ref in entities:
            prefixed = entity_ref if entity_ref.startswith("entity::") else f"entity::{entity_ref}"
            entity_beats[prefixed].append(beat_id)

    candidates: list[KnotCandidate] = []
    seen_pairs: set[tuple[str, ...]] = set()

    for entity_id, beats in sorted(entity_beats.items()):
        unique_beats = sorted(set(beats))
        if len(unique_beats) < 2:
            continue
        multi_tension = _filter_different_tensions(unique_beats, beat_tensions)
        if len(multi_tension) >= 2:
            key = tuple(multi_tension)
            if key not in seen_pairs:
                seen_pairs.add(key)
                candidates.append(
                    KnotCandidate(
                        beat_ids=multi_tension,
                        signal_type="entity",
                        shared_value=entity_id,
                    )
                )

    return candidates


def _filter_different_tensions(
    beat_ids: list[str],
    beat_tensions: dict[str, set[str]],
) -> list[str]:
    """Filter to beats that span at least 2 different tensions.

    Returns all beats if the group spans multiple tensions,
    empty list otherwise.
    """
    all_tensions: set[str] = set()
    for bid in beat_ids:
        all_tensions.update(beat_tensions.get(bid, set()))
    if len(all_tensions) < 2:
        return []
    return sorted(beat_ids)


def check_knot_compatibility(
    graph: Graph,
    beat_ids: list[str],
) -> list[GrowValidationError]:
    """Check if beats can form a valid knot.

    Validates:
    - All beat IDs exist in the graph
    - Beats are from different tensions (not same tension)
    - No circular requires conflicts between the beats
    - At least 2 beats

    Args:
        graph: Graph with beat and thread nodes.
        beat_ids: Proposed knot beat IDs.

    Returns:
        List of validation errors. Empty if compatible.
    """
    errors: list[GrowValidationError] = []

    if len(beat_ids) < 2:
        errors.append(
            GrowValidationError(
                field_path="knot.beat_ids",
                issue="Knot requires at least 2 beats",
                category=GrowErrorCategory.STRUCTURAL,
            )
        )
        return errors

    beat_nodes = graph.get_nodes_by_type("beat")

    # Check all beats exist
    for bid in beat_ids:
        if bid not in beat_nodes:
            errors.append(
                GrowValidationError(
                    field_path=f"knot.beat_ids.{bid}",
                    issue=f"Beat '{bid}' not found in graph",
                    category=GrowErrorCategory.REFERENCE,
                )
            )

    if errors:
        return errors

    # Check beats are from different tensions
    beat_tensions = _build_beat_tensions(graph, beat_nodes)
    tension_sets = [beat_tensions.get(bid, set()) for bid in beat_ids]

    # Knots must span at least 2 different tensions
    all_tensions = set.union(*tension_sets) if tension_sets else set()

    if len(all_tensions) < 2:
        errors.append(
            GrowValidationError(
                field_path="knot.tensions",
                issue=(
                    f"Beats span only {len(all_tensions)} tension(s): {sorted(all_tensions)}. "
                    f"Knots must span at least 2 different tensions."
                ),
                category=GrowErrorCategory.STRUCTURAL,
            )
        )

    # Check no circular requires between the knot beats
    beat_set = set(beat_ids)
    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="requires")
    for edge in requires_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_set and to_id in beat_set:
            errors.append(
                GrowValidationError(
                    field_path="knot.requires",
                    issue=(
                        f"Beat '{from_id}' requires '{to_id}' — "
                        f"knot beats cannot have requires dependencies on each other"
                    ),
                    category=GrowErrorCategory.STRUCTURAL,
                )
            )

    return errors


def resolve_knot_location(graph: Graph, beat_ids: list[str]) -> str | None:
    """Find a shared location for the knot beats.

    Resolution priority:
    1. Shared primary location
    2. Primary location of one that appears in alternatives of another
    3. Shared alternative location
    4. None if no common location found

    Args:
        graph: Graph with beat nodes.
        beat_ids: Beat IDs in the proposed knot.

    Returns:
        Resolved location string, or None if no common location.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Collect primary and alternative locations per beat
    primaries: list[str | None] = []
    all_locations: list[set[str]] = []

    for bid in beat_ids:
        data = beat_nodes.get(bid, {})
        primary = data.get("location")
        primaries.append(primary)
        locs = set()
        if primary:
            locs.add(primary)
        for alt in data.get("location_alternatives", []):
            locs.add(alt)
        all_locations.append(locs)

    if not all_locations:
        return None

    # 1. Shared primary location
    non_none_primaries = [p for p in primaries if p is not None]
    if non_none_primaries and len(set(non_none_primaries)) == 1:
        return non_none_primaries[0]

    # 2. Primary of one in alternatives/primaries of all others
    for primary in non_none_primaries:
        if all(primary in locs for locs in all_locations):
            return primary

    # 3. Any shared location across all beats
    if all_locations:
        shared = set.intersection(*all_locations)
        if shared:
            return sorted(shared)[0]  # Deterministic: alphabetically first

    return None


def apply_knot_mark(
    graph: Graph,
    beat_ids: list[str],
    resolved_location: str | None,
) -> None:
    """Mark beats as belonging to a knot (multi-thread scene).

    Updates beat nodes with:
    - knot_group: list of other beat IDs in the knot
    - resolved_location: the location chosen for the combined scene

    Also adds additional belongs_to edges so each beat is assigned to
    all threads from all beats in the knot.

    Args:
        graph: Graph to mutate.
        beat_ids: Beat IDs to knot together.
        resolved_location: Resolved location, or None.
    """
    beat_set = set(beat_ids)

    # Collect all thread assignments across all knot beats
    all_thread_ids: set[str] = set()
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        if edge["from"] in beat_set:
            all_thread_ids.add(edge["to"])

    # Collect new edges to add (batch to avoid stale reads)
    new_edges: list[tuple[str, str]] = []
    for bid in beat_ids:
        current_threads = {e["to"] for e in belongs_to_edges if e["from"] == bid}
        for thread_id in all_thread_ids - current_threads:
            new_edges.append((bid, thread_id))

    # Update each beat's node data
    for bid in beat_ids:
        others = sorted(b for b in beat_ids if b != bid)
        update_data: dict[str, Any] = {"knot_group": others}
        if resolved_location:
            update_data["location"] = resolved_location
        graph.update_node(bid, **update_data)

    # Apply cross-thread edges
    for from_id, to_id in new_edges:
        graph.add_edge("belongs_to", from_id, to_id)


# ---------------------------------------------------------------------------
# Phase 4: Gap detection algorithms
# ---------------------------------------------------------------------------


@dataclass
class PacingIssue:
    """A sequence of 3+ consecutive beats with the same scene_type."""

    thread_id: str
    beat_ids: list[str]
    scene_type: str


def get_thread_beat_sequence(graph: Graph, thread_id: str) -> list[str]:
    """Get ordered beat sequence for a thread using topological sort on requires edges.

    Delegates to topological_sort_beats() for the sorting logic.

    Args:
        graph: Graph with beat nodes and requires edges.
        thread_id: Prefixed thread ID (e.g., "thread::mentor_trust_canonical").

    Returns:
        Ordered list of beat IDs in the thread.

    Raises:
        ValueError: If a cycle is detected among the thread's beats.
    """
    belongs_to_edges = graph.get_edges(from_id=None, to_id=thread_id, edge_type="belongs_to")
    thread_beats = [e["from"] for e in belongs_to_edges]

    if not thread_beats:
        return []

    return topological_sort_beats(graph, thread_beats)


def detect_pacing_issues(graph: Graph) -> list[PacingIssue]:
    """Detect pacing issues: 3+ consecutive beats with the same scene_type.

    Checks each thread's beat sequence for runs of 3 or more beats
    all tagged with the same scene_type (scene, sequel, or micro_beat).

    Args:
        graph: Graph with beat nodes that have scene_type data.

    Returns:
        List of PacingIssue objects describing problematic sequences.
    """
    issues: list[PacingIssue] = []
    thread_nodes = graph.get_nodes_by_type("thread")

    for tid in sorted(thread_nodes.keys()):
        sequence = get_thread_beat_sequence(graph, tid)
        if len(sequence) < 3:
            continue

        # Get scene_type for each beat
        beat_types: list[tuple[str, str]] = []
        for bid in sequence:
            node = graph.get_node(bid)
            if node:
                scene_type = node.get("scene_type", "")
                beat_types.append((bid, scene_type))

        # Find runs of 3+ same type
        run_start = 0
        while run_start < len(beat_types):
            current_type = beat_types[run_start][1]
            if not current_type:
                run_start += 1
                continue

            run_end = run_start + 1
            while run_end < len(beat_types) and beat_types[run_end][1] == current_type:
                run_end += 1

            run_length = run_end - run_start
            if run_length >= 3:
                issues.append(
                    PacingIssue(
                        thread_id=tid,
                        beat_ids=[bt[0] for bt in beat_types[run_start:run_end]],
                        scene_type=current_type,
                    )
                )

            run_start = run_end

    return issues


def insert_gap_beat(
    graph: Graph,
    thread_id: str,
    after_beat: str | None,
    before_beat: str | None,
    summary: str,
    scene_type: str,
) -> str:
    """Insert a new gap beat into the graph between existing beats.

    Creates a new beat node and adjusts requires edges to maintain ordering.
    The new beat is assigned to the specified thread.

    Args:
        graph: Graph to mutate.
        thread_id: Thread this beat belongs to (prefixed ID).
        after_beat: Beat that should come before the new beat (or None for start).
        before_beat: Beat that should come after the new beat (or None for end).
        summary: Summary text for the new beat.
        scene_type: Scene type tag for the new beat.

    Returns:
        The new beat's node ID.
    """
    # Generate unique beat ID using max existing gap index + 1
    existing_beats = graph.get_nodes_by_type("beat")
    max_gap_index = 0
    for bid, data in existing_beats.items():
        if data.get("is_gap_beat", False):
            parts = bid.split("gap_")
            if len(parts) > 1:
                with contextlib.suppress(ValueError):
                    max_gap_index = max(max_gap_index, int(parts[-1]))
    raw_id = f"gap_{max_gap_index + 1}"
    beat_id = f"beat::{raw_id}"

    # Create the beat node
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": raw_id,
            "summary": summary,
            "scene_type": scene_type,
            "threads": [thread_id.removeprefix("thread::")],
            "is_gap_beat": True,
        },
    )

    # Add belongs_to edge
    graph.add_edge("belongs_to", beat_id, thread_id)

    # Adjust requires edges for ordering.
    # Existing transitive requires (before_beat → after_beat) is kept as redundant
    # but harmless for topological sort correctness.
    if after_beat:
        graph.add_edge("requires", beat_id, after_beat)

    if before_beat:
        graph.add_edge("requires", before_beat, beat_id)

    return beat_id


# ---------------------------------------------------------------------------
# Phase 9: Choice derivation helpers
# ---------------------------------------------------------------------------


@dataclass
class PassageSuccessor:
    """A successor passage reachable from a given passage on a specific arc."""

    to_passage: str
    arc_id: str
    grants: list[str] = field(default_factory=list)


def find_passage_successors(graph: Graph) -> dict[str, list[PassageSuccessor]]:
    """Find unique successor passages for each passage across all arcs.

    For each arc's beat sequence, converts to passage sequence and records
    which passages follow which. Deduplicates successors (same target passage
    across multiple arcs is recorded once, keeping the first encountered in
    arc sort order).

    Returns:
        Mapping of passage_id -> list of unique PassageSuccessor objects.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    passage_nodes = graph.get_nodes_by_type("passage")

    if not arc_nodes or not passage_nodes:
        return {}

    # Build beat → passage mapping
    beat_to_passage: dict[str, str] = {}
    for p_id, p_data in passage_nodes.items():
        from_beat = p_data.get("from_beat", "")
        if from_beat:
            beat_to_passage[from_beat] = p_id

    # Collect grants edges: beat → codeword
    grants_edges = graph.get_edges(from_id=None, to_id=None, edge_type="grants")
    beat_grants: dict[str, list[str]] = {}
    for edge in grants_edges:
        beat_grants.setdefault(edge["from"], []).append(edge["to"])

    successors: dict[str, list[PassageSuccessor]] = {}
    seen_targets: dict[str, set[str]] = {}

    for arc_id, arc_data in sorted(arc_nodes.items()):
        sequence: list[str] = arc_data.get("sequence", [])
        if len(sequence) < 2:
            continue

        # Convert beat sequence to passage sequence, preserving original beat index.
        # Beats without passages are intentionally skipped - not all beats become
        # passages (Phase 8a selects which beats get interactive passages).
        passage_seq: list[tuple[str, int]] = []
        for beat_idx, beat_id in enumerate(sequence):
            if beat_id in beat_to_passage:
                passage_seq.append((beat_to_passage[beat_id], beat_idx))

        for i in range(len(passage_seq) - 1):
            p_id, beat_idx = passage_seq[i]
            next_p, _ = passage_seq[i + 1]

            if p_id not in successors:
                successors[p_id] = []
                seen_targets[p_id] = set()

            # Skip if we already recorded this successor target
            if next_p in seen_targets[p_id]:
                continue
            seen_targets[p_id].add(next_p)

            # Grants: codewords from beats AFTER this beat's position on this arc.
            # Includes beats without passages - codewords are granted by beat
            # traversal regardless of passage representation.
            arc_grants: list[str] = []
            for beat_id in sequence[beat_idx + 1 :]:
                arc_grants.extend(beat_grants.get(beat_id, []))

            successors[p_id].append(
                PassageSuccessor(
                    to_passage=next_p,
                    arc_id=arc_id,
                    grants=arc_grants,
                )
            )

    return successors
