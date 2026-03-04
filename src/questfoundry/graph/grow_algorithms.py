"""Pure graph algorithms for the GROW stage.

These functions operate on Graph objects and return structured results.
Phase orchestration lives in pipeline/stages/grow.py.

Algorithm summary:
- validate_beat_dag: Kahn's algorithm for cycle detection in requires edges
- validate_commits_beats: Verify each explored dilemma has commits beat per path
- topological_sort_beats: Stable topological sort with priority + alphabetical tie-breaking
- compute_shared_beats: Find beats common to all arc combinations
- enumerate_arcs: Cartesian product of paths across dilemmas
- compute_divergence_points: Find where arcs diverge from the spine
- build_arc_state_flags: Map arc nodes to their state flag signatures
- select_entities_for_arc: Deterministic entity selection for Phase 4f
"""

from __future__ import annotations

import contextlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.graph.mutations import GrowErrorCategory, GrowValidationError
from questfoundry.models.grow import Arc
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from questfoundry.graph.graph import Graph

# Maximum number of arcs before triggering COMBINATORIAL error.
# With 2 dilemmas x 2 paths each = 4 arcs. With 5 dilemmas x 2 paths = 32 arcs.
# With 6 dilemmas x 2 paths = 64 arcs. Beyond 64 arcs, processing becomes very
# expensive and the story structure may be difficult to navigate.
_MAX_ARC_COUNT = 64


def build_dilemma_paths(graph: Graph) -> dict[str, list[str]]:
    """Build dilemma → paths mapping from path node dilemma_id properties.

    Uses the dilemma_id property on path nodes instead of explores edges,
    since explores edges point to answers (not dilemmas) in real SEED output.

    Args:
        graph: Graph containing dilemma and path nodes.

    Returns:
        Dict mapping dilemma node ID → list of path node IDs.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_paths: dict[str, list[str]] = defaultdict(list)
    for path_id, path_data in path_nodes.items():
        dilemma_id = path_data.get("dilemma_id")
        if dilemma_id:
            prefixed = normalize_scoped_id(dilemma_id, "dilemma")
            if prefixed in dilemma_nodes:
                dilemma_paths[prefixed].append(path_id)
    return dilemma_paths


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

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="predecessor")
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
    """Validate that each explored dilemma has a commits beat per path.

    For each dilemma that has paths exploring it, every path must have
    at least one beat with a dilemma_impact of effect="commits" for that dilemma.

    Args:
        graph: Graph containing dilemma, path, and beat nodes.

    Returns:
        List of validation errors for paths missing commits beats.
    """
    errors: list[GrowValidationError] = []

    # Get all dilemmas that have exploring paths
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")

    # Build dilemma → paths mapping from path node dilemma_id properties
    dilemma_paths = build_dilemma_paths(graph)

    # Build path → beats mapping via belongs_to edges
    path_beats: dict[str, list[str]] = defaultdict(list)
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        path_beats[path_id].append(beat_id)

    # For each dilemma's paths, check for commits beats
    beat_nodes = graph.get_nodes_by_type("beat")
    for dilemma_id, paths in sorted(dilemma_paths.items()):
        dilemma_raw = dilemma_nodes[dilemma_id].get("raw_id", dilemma_id)
        for path_id in sorted(paths):
            path_raw = path_nodes[path_id].get("raw_id", path_id)
            beats_in_path = path_beats.get(path_id, [])

            has_commits = False
            for beat_id in beats_in_path:
                beat_data = beat_nodes.get(beat_id, {})
                impacts = beat_data.get("dilemma_impacts", [])
                for impact in impacts:
                    if impact.get("dilemma_id") == dilemma_id and impact.get("effect") == "commits":
                        has_commits = True
                        break
                if has_commits:
                    break

            if not has_commits:
                errors.append(
                    GrowValidationError(
                        field_path=f"path.{path_raw}.commits",
                        issue=(
                            f"Path '{path_raw}' has no commits beat for dilemma '{dilemma_raw}'"
                        ),
                        category=GrowErrorCategory.STRUCTURAL,
                    )
                )

    return errors


# ---------------------------------------------------------------------------
# Topological Sort
# ---------------------------------------------------------------------------


def topological_sort_beats(
    graph: Graph,
    beat_ids: list[str],
    *,
    priority_beats: set[str] | None = None,
    reference_positions: dict[str, int] | None = None,
) -> list[str]:
    """Topologically sort a subset of beats using requires edges.

    Uses Kahn's algorithm with deterministic tie-breaking.  Tie-breaking
    order (highest priority first):

    1. **Priority tier** — beats in *priority_beats* (shared) before others.
    2. **Reference position** — if *reference_positions* is provided, beats
       with a reference position sort by that position.  Beats without a
       reference position sort after all referenced beats.  This ensures
       cross-arc consistency for shared beats.
    3. **Dilemma round-robin** — beats from the dilemma that has emitted
       fewest beats so far sort first, interleaving dilemmas instead of
       clustering all of one dilemma's beats together.
    4. **Alphabetical** — deterministic within the same tier and count.

    Args:
        graph: Graph containing requires edges and beat nodes with
            optional ``dilemma_impacts`` metadata.
        beat_ids: Subset of beat node IDs to sort.
        priority_beats: Optional set of beat IDs that should sort before
            non-priority beats when topological constraints allow. When
            ``None``, falls back to purely alphabetical tie-breaking.
        reference_positions: Optional mapping of beat ID → position from
            a canonical (spine) sequence.  When provided, beats present
            in the reference sort by their reference position before
            falling back to round-robin, guaranteeing cross-arc ordering
            consistency for shared beats.

    Returns:
        Sorted list of beat IDs (prerequisites first).

    Raises:
        ValueError: If a cycle is detected in the beat subset.
    """
    if not beat_ids:
        return []

    beat_set = set(beat_ids)

    # Build beat → dilemma mapping from dilemma_impacts metadata.
    # A beat may impact multiple dilemmas; use the first one for
    # round-robin grouping (beats typically belong to one dilemma).
    beat_dilemma: dict[str, str] = {}
    for bid in beat_set:
        beat_data = graph.get_node(bid)
        if beat_data:
            impacts = beat_data.get("dilemma_impacts", [])
            if impacts:
                beat_dilemma[bid] = impacts[0].get("dilemma_id", "")

    dilemma_emission: dict[str, int] = {}
    _NO_REF = len(beat_ids) + 1  # Sentinel: sort after all referenced beats

    def _sort_key(bid: str) -> tuple[int, int, int, str]:
        priority = 0 if priority_beats and bid in priority_beats else 1
        ref_pos = reference_positions.get(bid, _NO_REF) if reference_positions else _NO_REF
        did = beat_dilemma.get(bid, "")
        rr = dilemma_emission.get(did, 0)
        return (priority, ref_pos, rr, bid)

    # Build adjacency within the subset
    in_degree: dict[str, int] = dict.fromkeys(beat_set, 0)
    successors: dict[str, list[str]] = {bid: [] for bid in beat_set}

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="predecessor")
    for edge in requires_edges:
        from_id = edge["from"]  # dependent
        to_id = edge["to"]  # prerequisite
        if from_id in beat_set and to_id in beat_set:
            in_degree[from_id] += 1
            successors[to_id].append(from_id)

    # Kahn's with priority + round-robin + alphabetical tie-breaking
    queue = sorted((bid for bid, deg in in_degree.items() if deg == 0), key=_sort_key)
    result: list[str] = []

    while queue:
        node = queue.pop(0)  # Take highest-priority first
        result.append(node)
        # Track emission count for round-robin
        did = beat_dilemma.get(node, "")
        dilemma_emission[did] = dilemma_emission.get(did, 0) + 1
        new_ready = []
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                new_ready.append(successor)
        # Re-sort queue: emission counts changed, so round-robin ranks update
        queue = sorted(queue + new_ready, key=_sort_key)

    if len(result) != len(beat_set):
        remaining = beat_set - set(result)
        raise ValueError(f"Cycle detected in beat subset: {sorted(remaining)}")

    return result


def compute_shared_beats(
    path_beat_sets: dict[str, set[str]],
    path_lists: list[list[str]],
) -> set[str]:
    """Find beats guaranteed to appear in every possible arc.

    A beat is "shared" if it appears on **every** path of the dilemma it
    belongs to.  Such beats cannot differentiate arcs because every arc
    must include them.  For single-path (partially-explored) dilemmas all
    beats qualify automatically.

    Args:
        path_beat_sets: Mapping from path ID to the set of beat IDs
            that belong to that path (via ``belongs_to`` edges).
        path_lists: Per-dilemma lists of path IDs (one list per dilemma,
            in the same order used by ``enumerate_arcs``).

    Returns:
        Set of beat IDs present in every possible arc combination.
        Empty set if *path_lists* is empty.
    """
    if not path_lists:
        return set()

    shared: set[str] = set()
    for paths_in_dilemma in path_lists:
        if not paths_in_dilemma:
            continue
        # Beats on EVERY path of this dilemma appear in all arcs
        beat_sets = [path_beat_sets.get(pid, set()) for pid in paths_in_dilemma]
        shared |= set.intersection(*beat_sets)
    return shared


# ---------------------------------------------------------------------------
# Phase 7b: Collapse linear beats
# ---------------------------------------------------------------------------


@dataclass
class CollapseResult:
    """Summary of linear-beat collapsing."""

    runs_collapsed: int
    beats_removed: int


def collapse_linear_beats(graph: Graph, *, min_run_length: int = 2) -> CollapseResult:
    """Collapse mandatory linear beat runs into a single combined beat.

    A beat is eligible for collapsing when it has exactly one predecessor and
    one successor in the forward (requires) graph, belongs to a single path,
    and is not an exempt narrative function (confront/resolve).

    Args:
        graph: Story graph with beat nodes.
        min_run_length: Minimum length of an eligible run to collapse.

    Returns:
        CollapseResult with counts of collapsed runs and removed beats.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return CollapseResult(runs_collapsed=0, beats_removed=0)

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="predecessor")
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")

    forward_predecessors: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    forward_successors: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in requires_edges:
        dependent = edge["from"]
        prereq = edge["to"]
        if dependent in beat_nodes and prereq in beat_nodes:
            forward_predecessors[dependent].append(prereq)
            forward_successors[prereq].append(dependent)

    beat_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        if beat_id in beat_nodes:
            beat_paths.setdefault(beat_id, []).append(path_id)

    def _is_exempt(beat_id: str) -> bool:
        narrative_function = beat_nodes.get(beat_id, {}).get("narrative_function")
        return narrative_function in {"confront", "resolve"}

    def _is_eligible(beat_id: str) -> bool:
        if _is_exempt(beat_id):
            return False
        if len(beat_paths.get(beat_id, [])) != 1:
            return False
        return (
            len(forward_predecessors.get(beat_id, [])) == 1
            and len(forward_successors.get(beat_id, [])) == 1
        )

    # Build path → beats mapping
    path_beats: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        if beat_id in beat_nodes:
            path_beats.setdefault(path_id, []).append(beat_id)

    runs: list[list[str]] = []
    for _path_id, beat_ids in sorted(path_beats.items()):
        try:
            ordered = topological_sort_beats(graph, beat_ids)
        except ValueError:
            ordered = sorted(beat_ids)

        current: list[str] = []
        for beat_id in ordered:
            if _is_eligible(beat_id):
                if not current:
                    current = [beat_id]
                else:
                    prev = current[-1]
                    if beat_id in forward_successors.get(
                        prev, []
                    ) and prev in forward_predecessors.get(beat_id, []):
                        current.append(beat_id)
                    else:
                        if len(current) >= min_run_length:
                            runs.append(current)
                        current = [beat_id]
            else:
                if len(current) >= min_run_length:
                    runs.append(current)
                current = []
        if len(current) >= min_run_length:
            runs.append(current)

    if not runs:
        return CollapseResult(runs_collapsed=0, beats_removed=0)

    removed_beats: set[str] = set()

    def _merge_beat_data(keep_id: str, remove_ids: list[str]) -> None:
        keep_data = beat_nodes.get(keep_id, {})
        summaries: list[str] = []
        if keep_data.get("summary"):
            summaries.append(str(keep_data.get("summary")))
        entities: list[str] = list(keep_data.get("entities", []))
        impacts: list[dict[str, Any]] = list(keep_data.get("dilemma_impacts", []))

        for rid in remove_ids:
            data = beat_nodes.get(rid, {})
            summary = data.get("summary")
            if summary:
                summaries.append(str(summary))
            entities.extend(data.get("entities", []))
            impacts.extend(data.get("dilemma_impacts", []))

        updates: dict[str, Any] = {}
        if summaries:
            updates["summary"] = " / ".join(summaries)
        if entities:
            updates["entities"] = list(dict.fromkeys(entities))
        if impacts:
            updates["dilemma_impacts"] = impacts
        if updates:
            graph.update_node(keep_id, **updates)

    def _ensure_edge(edge_type: str, from_id: str, to_id: str) -> None:
        if not graph.get_edges(from_id=from_id, to_id=to_id, edge_type=edge_type):
            graph.add_edge(edge_type, from_id, to_id)

    def _transfer_edges(keep_id: str, remove_ids: list[str]) -> None:
        for rid in remove_ids:
            for edge in graph.get_edges(from_id=rid, to_id=None, edge_type="grants"):
                _ensure_edge("grants", keep_id, edge["to"])
            for edge in graph.get_edges(from_id=rid, to_id=None, edge_type="belongs_to"):
                _ensure_edge("belongs_to", keep_id, edge["to"])

    for run in runs:
        keep_id = run[0]
        remove_ids = run[1:]
        if not remove_ids:
            continue

        _merge_beat_data(keep_id, remove_ids)
        _transfer_edges(keep_id, remove_ids)

        before_ids = [bid for bid in forward_predecessors.get(keep_id, []) if bid not in run]
        after_ids = [bid for bid in forward_successors.get(run[-1], []) if bid not in run]

        if before_ids:
            _ensure_edge("predecessor", keep_id, before_ids[0])
        if after_ids:
            _ensure_edge("predecessor", after_ids[0], keep_id)

        for rid in remove_ids:
            removed_beats.add(rid)
            graph.delete_node(rid, cascade=True)

    if removed_beats:
        arc_nodes = graph.get_nodes_by_type("arc")
        for arc_id, arc_data in arc_nodes.items():
            seq: list[str] = arc_data.get("sequence", [])
            if not seq:
                continue
            new_seq = [bid for bid in seq if bid not in removed_beats]
            if new_seq != seq:
                graph.update_node(arc_id, sequence=new_seq)

    return CollapseResult(runs_collapsed=len(runs), beats_removed=len(removed_beats))


def build_arc_state_flags(
    graph: Graph,
    arc_nodes: dict[str, dict[str, Any]],
    scope: Literal["ending", "routing", "all"] = "ending",
) -> dict[str, frozenset[str]]:
    """Build mapping from arc node ID to its state flag signature.

    Args:
        graph: The story graph.
        arc_nodes: Arc node data from ``graph.get_nodes_by_type("arc")``.
        scope: Which state flags to include:

            - ``"ending"`` — only state flags from ``ending_salience == "high"``
              dilemmas (for ``split_ending_families``).
            - ``"routing"`` — state flags from dilemmas with any active routing
              need (``ending_salience == "high"`` OR
              ``residue_weight == "heavy"``), for routing validation.
            - ``"all"`` — all state flags, unfiltered.

    Traces: arc → paths → consequences → state flags via graph edges,
    filtered by: path → dilemma → ending_salience / residue_weight.
    """
    # Build path → dilemma lookup, filtered by scope
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    included_paths: set[str] = set()
    for path_id, path_data in path_nodes.items():
        dilemma_id = path_data.get("dilemma_id")
        if not dilemma_id:
            continue
        if scope == "all":
            included_paths.add(path_id)
            continue
        dilemma_node_id = normalize_scoped_id(dilemma_id, "dilemma")
        dilemma_data = dilemma_nodes.get(dilemma_node_id, {})
        salience = dilemma_data.get("ending_salience", "low")
        weight = dilemma_data.get("residue_weight", "light")
        if (scope == "ending" and salience == "high") or (
            scope == "routing" and (salience == "high" or weight == "heavy")
        ):
            included_paths.add(path_id)

    # consequence → state flag (via derived_from edges: state_flag derived_from consequence)
    derived_from_edges = graph.get_edges(edge_type="derived_from")
    consequence_to_state_flag: dict[str, str] = {}
    for edge in derived_from_edges:
        consequence_to_state_flag[edge["to"]] = edge["from"]

    # path → consequences (via has_consequence edges)
    has_consequence_edges = graph.get_edges(edge_type="has_consequence")
    path_consequences: dict[str, list[str]] = {}
    for edge in has_consequence_edges:
        path_consequences.setdefault(edge["from"], []).append(edge["to"])

    result: dict[str, frozenset[str]] = {}
    for arc_id, data in arc_nodes.items():
        sfs: set[str] = set()
        for path_raw in data.get("paths", []):
            path_id = normalize_scoped_id(path_raw, "path")
            if path_id not in included_paths:
                continue
            for cons_id in path_consequences.get(path_id, []):
                if sf := consequence_to_state_flag.get(cons_id):
                    sfs.add(sf)
        result[arc_id] = frozenset(sfs)

    return result


# ---------------------------------------------------------------------------
# Phase 5: Arc Enumeration
# ---------------------------------------------------------------------------


def enumerate_arcs(graph: Graph, *, max_arc_count: int | None = None) -> list[Arc]:
    """Enumerate all arcs from the Cartesian product of paths across dilemmas.

    For each dilemma, collects its paths. Takes the Cartesian product across
    all dilemmas to produce arc combinations. Each arc gets the beats that
    belong to ANY of its constituent paths, topologically sorted.

    The spine arc contains all canonical paths. Branch arcs contain at least
    one non-canonical path.

    The arc count limit is **policy-aware**: only hard-policy dilemmas count
    toward the limit, since soft/flavor dilemmas converge and don't multiply
    endings.  The full Cartesian product is still enumerated so downstream
    phases (convergence metadata, residue) can operate on soft-variant arcs.

    Args:
        graph: Graph containing dilemma, path, and beat nodes.
        max_arc_count: Safety ceiling for effective (hard-only) arc count.
            Defaults to ``_MAX_ARC_COUNT`` (64) if not provided.

    Returns:
        List of Arc models, spine first, then branches sorted by ID.

    Raises:
        ValueError: If effective arc count exceeds the limit.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")

    if not dilemma_nodes or not path_nodes:
        return []

    # Build dilemma → paths mapping from path node dilemma_id properties
    dilemma_paths_map = build_dilemma_paths(graph)

    # Sort paths within each dilemma for determinism
    for paths in dilemma_paths_map.values():
        paths.sort()

    # Get path lists per dilemma (sorted by dilemma ID for determinism)
    sorted_dilemmas = sorted(dilemma_paths_map.keys())
    path_lists = [dilemma_paths_map[did] for did in sorted_dilemmas]

    if not path_lists or any(not pl for pl in path_lists):
        return []

    # Build path → beat set mapping via belongs_to
    path_beat_sets: dict[str, set[str]] = defaultdict(set)
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        path_beat_sets[path_id].add(beat_id)

    # Compute shared beats (present in every arc) for priority ordering.
    # Shared beats sort first so exclusive beats land at the end of sequences,
    # causing arcs to diverge naturally toward distinct endings.
    shared = compute_shared_beats(dict(path_beat_sets), path_lists)

    # Compute a global reference ordering from ALL beats across all paths.
    # This ensures every beat has a reference position, preventing
    # context-dependent round-robin from producing cross-arc inversions (#929).
    all_beats: set[str] = set()
    for beats in path_beat_sets.values():
        all_beats.update(beats)

    reference_positions: dict[str, int] | None = None
    try:
        global_sequence = topological_sort_beats(
            graph,
            list(all_beats),
            priority_beats=shared,
        )
        reference_positions = {bid: idx for idx, bid in enumerate(global_sequence)}
    except ValueError:
        pass  # Fallback: no reference if global beat set has cycles

    # Cartesian product of paths
    arcs: list[Arc] = []
    for combo in product(*path_lists):
        path_combo = list(combo)
        path_raw_ids = sorted(path_nodes[pid].get("raw_id", pid) for pid in path_combo)
        arc_id = "+".join(path_raw_ids)
        is_spine = all(path_nodes[pid].get("is_canonical", False) for pid in path_combo)

        beat_set: set[str] = set()
        for pid in path_combo:
            beat_set.update(path_beat_sets.get(pid, set()))

        try:
            sequence = topological_sort_beats(
                graph,
                list(beat_set),
                priority_beats=shared,
                reference_positions=reference_positions,
            )
        except ValueError:
            sequence = sorted(beat_set)  # Fallback for cycles

        arcs.append(
            Arc(
                arc_id=arc_id,
                arc_type="spine" if is_spine else "branch",
                paths=path_raw_ids,
                sequence=sequence,
            )
        )

    # Check combinatorial limit using EFFECTIVE arc count.
    # Only hard-policy dilemmas multiply endings; soft/flavor dilemmas converge
    # back and don't produce distinct endings.  The full Cartesian product is
    # still enumerated (downstream phases need soft-variant arcs for convergence
    # metadata and residue), but the limit check counts hard dilemmas only.
    limit = max_arc_count if max_arc_count is not None else _MAX_ARC_COUNT
    hard_dilemma_count = sum(
        1
        for did in sorted_dilemmas
        if dilemma_nodes.get(did, {}).get("dilemma_role", "soft") == "hard"
        and len(dilemma_paths_map.get(did, [])) >= 2
    )
    effective_arc_count = 2**hard_dilemma_count

    if effective_arc_count > limit:
        raise ValueError(
            f"Effective arc count ({effective_arc_count}) from "
            f"{hard_dilemma_count} hard dilemmas exceeds limit of {limit}. "
            f"Reduce the number of hard-policy dilemmas or paths."
        )

    log.debug(
        "arc_enumeration_complete",
        total_arcs=len(arcs),
        effective_arcs=effective_arc_count,
        hard_dilemmas=hard_dilemma_count,
        total_dilemmas=len(sorted_dilemmas),
        limit=limit,
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
class DilemmaPolicy:
    """Policy info for a single divergent dilemma within an arc.

    Attributes:
        dilemma_id: Scoped dilemma ID (e.g. ``dilemma::foo``).
        policy: Convergence policy (hard/soft).
        budget: Payoff budget for this dilemma.
        non_canon_path_id: The non-canonical path for this dilemma in this arc.
    """

    dilemma_id: str
    policy: str
    budget: int
    non_canon_path_id: str


@dataclass
class DilemmaConvergence:
    """Convergence result for a single dilemma within a branch arc.

    Attributes:
        dilemma_id: Scoped dilemma ID.
        policy: The dilemma's convergence policy.
        budget: The dilemma's payoff budget.
        converges_at: Beat where this dilemma converges (None for hard).
    """

    dilemma_id: str
    policy: str
    budget: int
    converges_at: str | None = None


@dataclass
class ConvergenceInfo:
    """Information about where a branch arc converges back to the spine.

    Attributes:
        arc_id: The branch arc that converges.
        converges_to: The arc it converges to (spine).
        converges_at: The beat where convergence occurs (None if no convergence).
            soft: boundary beat after last exclusive beat (if payoff_budget met).
            hard: always None (no convergence).
        dilemma_role: Effective policy applied to this arc.
        payoff_budget: Effective payoff budget applied to this arc.
        dilemma_convergences: Per-dilemma convergence details.
    """

    arc_id: str
    converges_to: str
    converges_at: str | None = None
    dilemma_role: str = "soft"
    payoff_budget: int = 2
    dilemma_convergences: list[DilemmaConvergence] = field(default_factory=list)


def _count_dilemma_explored_paths(graph: Graph) -> dict[str, int]:
    """Count total explored paths per dilemma across the whole graph.

    Returns a mapping from scoped dilemma ID to path count.
    Compute once and pass to ``_find_arc_dilemma_policies`` to avoid
    repeated full-graph scans.
    """
    counts: dict[str, int] = {}
    for path_data in graph.get_nodes_by_type("path").values():
        did = path_data.get("dilemma_id")
        if did:
            scoped = normalize_scoped_id(did, "dilemma")
            counts[scoped] = counts.get(scoped, 0) + 1
    return counts


def _find_arc_dilemma_policies(
    graph: Graph,
    arc: Arc,
    dilemma_path_counts: dict[str, int] | None = None,
) -> list[DilemmaPolicy]:
    """Collect per-dilemma policy info for divergent dilemmas only.

    Only considers dilemmas that have 2+ explored paths in the graph.
    Single-explored dilemmas contribute universal beats but should not
    influence convergence policy — the story doesn't diverge on them.

    Args:
        graph: Graph with dilemma and path nodes.
        arc: Arc to inspect.
        dilemma_path_counts: Pre-computed counts from
            ``_count_dilemma_explored_paths``. Computed on demand if None.

    Returns:
        List of DilemmaPolicy, one per divergent dilemma in this arc.
    """
    if dilemma_path_counts is None:
        dilemma_path_counts = _count_dilemma_explored_paths(graph)

    seen_dilemmas: set[str] = set()
    policies: list[DilemmaPolicy] = []
    for raw_path_id in arc.paths:
        path_node_id = normalize_scoped_id(raw_path_id, "path")
        path_node = graph.get_node(path_node_id)
        if not path_node or not (dilemma_id := path_node.get("dilemma_id")):
            continue
        scoped_did = normalize_scoped_id(dilemma_id, "dilemma")
        if scoped_did in seen_dilemmas:
            continue
        seen_dilemmas.add(scoped_did)
        # Skip single-explored dilemmas (universal beats, no actual divergence)
        if dilemma_path_counts.get(scoped_did, 0) < 2:
            continue
        dilemma_node = graph.get_node(scoped_did)
        if dilemma_node:
            policies.append(
                DilemmaPolicy(
                    dilemma_id=scoped_did,
                    policy=dilemma_node.get("dilemma_role", "soft"),
                    budget=dilemma_node.get("payoff_budget", 2),
                    non_canon_path_id=path_node_id,
                )
            )
    return policies


def _get_effective_policy(
    graph: Graph,
    arc: Arc,
    dilemma_path_counts: dict[str, int] | None = None,
) -> tuple[str, int]:
    """Combine dilemma roles for a (possibly multi-dilemma) arc.

    Combine rule per issue #743: hard dominates; payoff_budget = max across
    all dilemmas the arc diverges on.  Falls back to ("soft", 0) when no
    dilemma metadata is found (soft with zero budget = converge at first
    shared beat after divergence).
    """
    policies = _find_arc_dilemma_policies(graph, arc, dilemma_path_counts)
    if not policies:
        # No SEED dilemma metadata — default to soft with no budget constraint.
        return ("soft", 0)

    max_budget = max(dp.budget for dp in policies)
    if any(dp.policy == "hard" for dp in policies):
        return ("hard", max_budget)
    return ("soft", max_budget)


def _find_convergence_for_soft(
    branch_after_div: list[str],
    spine_seq_set: set[str],
    payoff_budget: int,
) -> str | None:
    """Find converges_at using backward scan for soft policy.

    Scans from the end of the branch sequence backward to find the true
    convergence boundary — the first shared beat that has NO later exclusive
    beats.  Then verifies the payoff_budget is met (enough exclusive beats
    before convergence).
    """
    if not branch_after_div:
        return None

    # Find the index of the last exclusive beat
    last_exclusive_idx: int | None = None
    for i in range(len(branch_after_div) - 1, -1, -1):
        if branch_after_div[i] not in spine_seq_set:
            last_exclusive_idx = i
            break

    if last_exclusive_idx is None:
        # All beats are shared — budget must still be satisfied
        if payoff_budget > 0:
            return None
        return branch_after_div[0] if branch_after_div else None

    # converges_at = the beat immediately after the last exclusive beat
    next_idx = last_exclusive_idx + 1
    if next_idx >= len(branch_after_div):
        # Last exclusive beat is at the very end — no convergence
        return None
    candidate = branch_after_div[next_idx]
    if candidate not in spine_seq_set:
        # Shouldn't happen since everything after last_exclusive should be shared,
        # but guard defensively.
        return None

    # Check payoff_budget: count exclusive beats before convergence
    exclusive_count = sum(1 for b in branch_after_div[:next_idx] if b not in spine_seq_set)
    if exclusive_count < payoff_budget:
        log.debug(
            "convergence_budget_not_met",
            exclusive_count=exclusive_count,
            payoff_budget=payoff_budget,
        )
        return None

    return candidate


def _build_beat_dilemma_map_for_convergence(
    graph: Graph,
) -> dict[str, set[str]]:
    """Map each beat to its prefixed dilemma IDs via belongs_to → path → dilemma.

    Similar to ``_build_beat_dilemmas`` but returns prefixed dilemma IDs
    (e.g. ``dilemma::foo``) for direct comparison with dilemma node keys.
    """
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    path_to_dilemma: dict[str, str] = {}
    for path_id, path_data in path_nodes.items():
        did = path_data.get("dilemma_id")
        if did:
            prefixed = normalize_scoped_id(did, "dilemma")
            if prefixed in dilemma_nodes:
                path_to_dilemma[path_id] = prefixed

    beat_dilemmas: dict[str, set[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        beat_id = edge["from"]
        path_id = edge["to"]
        if path_id in path_to_dilemma:
            beat_dilemmas.setdefault(beat_id, set()).add(path_to_dilemma[path_id])

    return beat_dilemmas


def _compute_per_dilemma_convergence(
    dilemma_policies: list[DilemmaPolicy],
    branch_after_div: list[str],
    spine_seq_set: set[str],
    beat_dilemma_map: dict[str, set[str]],
) -> list[DilemmaConvergence]:
    """Compute convergence point separately for each divergent dilemma.

    For each dilemma:
    - **hard**: ``converges_at = None`` (never converges).
    - **soft**: Filter beats to this dilemma + neutral, then backward scan.

    A beat is "relevant" to a dilemma if it has that dilemma association or
    has no dilemma association at all (neutral/shared beat).
    """
    results: list[DilemmaConvergence] = []

    for dp in dilemma_policies:
        if dp.policy == "hard":
            results.append(
                DilemmaConvergence(
                    dilemma_id=dp.dilemma_id,
                    policy=dp.policy,
                    budget=dp.budget,
                    converges_at=None,
                )
            )
            continue

        # Filter beats: keep if neutral (no dilemma assoc) or associated with this dilemma
        filtered_branch = [
            b
            for b in branch_after_div
            if not beat_dilemma_map.get(b) or dp.dilemma_id in beat_dilemma_map[b]
        ]
        filtered_spine = {
            b
            for b in spine_seq_set
            if not beat_dilemma_map.get(b) or dp.dilemma_id in beat_dilemma_map[b]
        }

        # soft (or legacy flavor): backward scan with budget
        converges_at = _find_convergence_for_soft(filtered_branch, filtered_spine, dp.budget)

        results.append(
            DilemmaConvergence(
                dilemma_id=dp.dilemma_id,
                policy=dp.policy,
                budget=dp.budget,
                converges_at=converges_at,
            )
        )

    return results


def find_convergence_points(
    graph: Graph,
    arcs: list[Arc],
    divergence_map: dict[str, DivergenceInfo] | None = None,
    spine_arc_id: str | None = None,
) -> dict[str, ConvergenceInfo]:
    """Find where branch arcs converge back to the spine.

    For multi-dilemma arcs, computes convergence from non-hard dilemma beats
    only.  Hard dilemmas never converge (by spec), but non-hard dilemmas in
    the same arc DO converge, and FILL should know about it.

    Policy-aware convergence (applied to non-hard beats):
    - **flavor**: First shared beat after divergence (immediate convergence).
    - **soft**: Last-exclusive-beat boundary via backward scan, respecting
      payoff_budget.
    - **hard** (all dilemmas hard): No convergence (converges_at = None).

    Args:
        graph: Graph with dilemma nodes containing dilemma_role/payoff_budget.
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
    beat_dilemma_map = _build_beat_dilemma_map_for_convergence(graph)
    dilemma_path_counts = _count_dilemma_explored_paths(graph)

    for arc in arcs:
        if arc.arc_type == "spine":
            continue

        div_info = divergence_map.get(arc.arc_id)
        if not div_info:
            continue

        # Get the full arc-level effective policy (for storage on the node)
        eff_policy, eff_budget = _get_effective_policy(graph, arc, dilemma_path_counts)

        # Find beats in branch after divergence point
        diverge_at = div_info.diverges_at
        if diverge_at and diverge_at in arc.sequence:
            div_idx = arc.sequence.index(diverge_at)
            branch_after_div = arc.sequence[div_idx + 1 :]
        else:
            branch_after_div = arc.sequence

        policies = _find_arc_dilemma_policies(graph, arc, dilemma_path_counts)

        converges_at: str | None = None
        per_dilemma: list[DilemmaConvergence] = []

        if not policies:
            # No dilemma metadata → use arc-level effective policy (backward compat)
            if eff_policy == "soft":
                converges_at = _find_convergence_for_soft(
                    branch_after_div, spine_seq_set, eff_budget
                )
            # else: hard with no metadata shouldn't happen, but safe
        else:
            # Compute per-dilemma convergence
            per_dilemma = _compute_per_dilemma_convergence(
                policies, branch_after_div, spine_seq_set, beat_dilemma_map
            )

            # Arc-level converges_at = earliest non-None per-dilemma convergence
            non_none = [dc for dc in per_dilemma if dc.converges_at]
            if non_none:

                def _beat_index(dc: DilemmaConvergence, seq: list[str] = branch_after_div) -> int:
                    at = dc.converges_at
                    return seq.index(at) if at and at in seq else len(seq)

                converges_at = min(non_none, key=_beat_index).converges_at

        result[arc.arc_id] = ConvergenceInfo(
            arc_id=arc.arc_id,
            converges_to=spine.arc_id,
            converges_at=converges_at,
            dilemma_role=eff_policy,
            payoff_budget=eff_budget,
            dilemma_convergences=per_dilemma,
        )

    return result


# ---------------------------------------------------------------------------
# Phase 3: Intersection Detection
# ---------------------------------------------------------------------------


@dataclass
class IntersectionCandidate:
    """A group of beats that share signals and could form an intersection.

    Attributes:
        beat_ids: Beat node IDs that share signals.
        signal_type: What signal links them (location, entity).
        shared_value: The shared signal value.
    """

    beat_ids: list[str]
    signal_type: str
    shared_value: str


def format_intersection_candidates(
    candidates: list[IntersectionCandidate],
    beat_nodes: dict[str, Any],
    beat_dilemmas: dict[str, set[str]],
    graph: Graph | None = None,
) -> str:
    """Format intersection candidates as numbered groups for the LLM prompt.

    Each group shows the shared signal, involved dilemmas, and compact
    beat details. Beats appearing in multiple candidate groups are
    included in all of them.

    When ``graph`` is provided, dilemma context (question, stakes) is
    included per group and beat summaries are enriched with
    narrative_function.

    Args:
        candidates: Pre-screened cross-dilemma candidate groups.
        beat_nodes: Beat node data keyed by beat ID.
        beat_dilemmas: Mapping of beat_id to set of dilemma IDs.
        graph: Optional graph for dilemma/entity context enrichment.

    Returns:
        Formatted string with numbered candidate groups, or empty string
        if no candidates.
    """
    from questfoundry.graph.context_compact import truncate_summary

    if not candidates:
        return ""

    sections: list[str] = []
    for i, candidate in enumerate(candidates, 1):
        signal_label = candidate.signal_type
        header = f'### Candidate Group {i} (shared {signal_label}: "{candidate.shared_value}")'

        # Collect dilemmas represented in this group
        group_dilemmas: set[str] = set()
        for bid in candidate.beat_ids:
            group_dilemmas.update(beat_dilemmas.get(bid, set()))

        # Dilemma context block (enriched when graph is available)
        dilemma_lines: list[str] = []
        for did in sorted(group_dilemmas):
            if graph:
                dnode = graph.get_node(did) or graph.get_node(f"dilemma::{did}")
                if dnode:
                    question = dnode.get("question", "")
                    stakes = dnode.get("why_it_matters", "")
                    label = f'- {did}: "{question}"'
                    if stakes:
                        label += f" (Stakes: {truncate_summary(stakes, 100)})"
                    dilemma_lines.append(label)
                    continue
            dilemma_lines.append(f"- {did}")

        dilemma_block = "Dilemmas:\n" + "\n".join(dilemma_lines)

        # Format each beat compactly
        beat_lines: list[str] = []
        for bid in candidate.beat_ids:
            data = beat_nodes.get(bid, {})
            dilemma_ids = sorted(beat_dilemmas.get(bid, set()))
            dilemma_tag = dilemma_ids[0] if dilemma_ids else "unknown"
            summary = truncate_summary(data.get("summary", ""), 80)
            location = data.get("location", "unspecified")
            narrative_fn = data.get("narrative_function", "")

            fn_tag = f", {narrative_fn}" if narrative_fn else ""
            beat_lines.append(f'- {bid} [{dilemma_tag}{fn_tag}]: "{summary}" (loc: {location})')

        sections.append("\n".join([header, dilemma_block, "", *beat_lines]))

    return "\n\n".join(sections)


def build_intersection_candidates(graph: Graph) -> list[IntersectionCandidate]:
    """Find beats that share signals and could form intersections.

    Groups beats by shared locations/flexibility edges and shared entities.
    Only considers beats from different dilemmas (same dilemma = alternative).

    Args:
        graph: Graph with beat, path, and dilemma nodes.

    Returns:
        List of IntersectionCandidate groups, prioritizing location overlap.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    # Build beat → dilemma mapping via belongs_to → path → dilemma
    beat_dilemmas = _build_beat_dilemmas(graph, beat_nodes)

    # Filter out gap beats: they are path-local by construction (their
    # predecessors exist on a single path only) and are therefore never eligible as
    # intersection beats. See spec §GROW Intersections — Intersection eligibility constraint.
    beat_nodes = {bid: d for bid, d in beat_nodes.items() if not d.get("is_gap_beat")}
    beat_dilemmas = {bid: ds for bid, ds in beat_dilemmas.items() if bid in beat_nodes}
    if not beat_nodes:
        return []

    # Group by location overlap (highest priority)
    location_groups = _group_by_location(graph, beat_nodes, beat_dilemmas)

    # Group by shared entity
    entity_groups = _group_by_entity(graph, beat_nodes, beat_dilemmas)

    return location_groups + entity_groups


def _build_beat_dilemmas(graph: Graph, beat_nodes: dict[str, Any]) -> dict[str, set[str]]:
    """Map each beat to its dilemma IDs (via path → dilemma edges).

    Returns:
        Dict mapping beat_id → set of dilemma raw_ids.
    """
    # path → dilemma mapping (from path node dilemma_id properties)
    path_dilemma: dict[str, str] = {}
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    for path_id, path_data in path_nodes.items():
        dilemma_id = path_data.get("dilemma_id")
        if dilemma_id:
            prefixed = normalize_scoped_id(dilemma_id, "dilemma")
            if prefixed in dilemma_nodes:
                dilemma_raw = dilemma_nodes[prefixed].get("raw_id", prefixed)
                path_dilemma[path_id] = dilemma_raw

    # beat → dilemmas via belongs_to
    beat_dilemmas: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        if beat_id in beat_nodes and path_id in path_dilemma:
            beat_dilemmas[beat_id].add(path_dilemma[path_id])

    return beat_dilemmas


def _get_hard_policy_beats(
    graph: Graph,
    beat_ids: list[str],
    beat_dilemma_map: dict[str, set[str]],
) -> set[str]:
    """Identify beats belonging to hard-policy dilemmas.

    A beat is hard-policy if any of its dilemmas has
    ``dilemma_role == "hard"``.

    Returns:
        Set of beat IDs from hard-policy dilemmas.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    # raw_id → prefixed ID lookup
    raw_to_prefixed: dict[str, str] = {}
    for did, ddata in dilemma_nodes.items():
        raw = ddata.get("raw_id", did)
        raw_to_prefixed[raw] = did

    hard_beats: set[str] = set()
    for bid in beat_ids:
        for raw in beat_dilemma_map.get(bid, set()):
            prefixed = raw_to_prefixed.get(raw)
            dnode = dilemma_nodes.get(prefixed) if prefixed else None
            if dnode and dnode.get("dilemma_role") == "hard":
                hard_beats.add(bid)
                break
    return hard_beats


def _limit_one_beat_per_dilemma(
    beats: list[str],
    beat_dilemmas: dict[str, set[str]],
) -> list[str]:
    """Return at most one beat per dilemma from `beats`.

    Iterates beats in sorted order (prefers earlier-numbered beats like _beat_01).
    A beat is included only if none of its dilemmas overlap with dilemmas already
    claimed by previously included beats. All of the beat's dilemmas are marked
    as seen on inclusion, preventing later beats from sharing any of them.

    Args:
        beats: Candidate beat IDs (need not be sorted; sorted internally).
        beat_dilemmas: Mapping from beat_id to the set of dilemma IDs it belongs to.

    Returns:
        Subset of `beats` with at most one beat per dilemma, in sorted order.
    """
    seen_dilemmas: set[str] = set()
    limited: list[str] = []
    for bid in sorted(beats):
        dilemmas = beat_dilemmas.get(bid, set())
        if not dilemmas:
            continue
        if dilemmas.isdisjoint(seen_dilemmas):
            seen_dilemmas.update(dilemmas)
            limited.append(bid)
    return limited


def _group_by_location(
    graph: Graph,
    beat_nodes: dict[str, Any],
    beat_dilemmas: dict[str, set[str]],
) -> list[IntersectionCandidate]:
    """Group beats by location overlap (primary location or flexibility edges).

    Two beats have location overlap if:
    - Beat A's location matches a flexibility target of Beat B, or vice versa
    - They share the same primary location
    """
    # Build location → beats mapping
    location_beats: dict[str, list[str]] = defaultdict(list)

    for beat_id, beat_data in beat_nodes.items():
        primary = beat_data.get("location")
        if primary:
            location_beats[primary].append(beat_id)
        # Read flexibility edges (Doc 3) instead of location_alternatives property.
        # NOTE: No role filter applied — currently only role="location" flexibility
        # edges exist (written by mutations.py). If other roles are added (e.g.,
        # role="entity"), this query must filter by role="location".
        for edge in graph.get_edges(from_id=beat_id, edge_type="flexibility"):
            location_beats[edge["to"]].append(beat_id)

    candidates: list[IntersectionCandidate] = []
    for location, beats in sorted(location_beats.items()):
        if len(beats) < 2:
            continue
        # Filter to beats from different dilemmas
        multi_dilemma = _filter_different_dilemmas(beats, beat_dilemmas)
        if len(multi_dilemma) >= 2:
            limited = _limit_one_beat_per_dilemma(multi_dilemma, beat_dilemmas)
            if len(limited) >= 2:
                candidates.append(
                    IntersectionCandidate(
                        beat_ids=limited,
                        signal_type="location",
                        shared_value=location,
                    )
                )

    return candidates


def _group_by_entity(
    graph: Graph,
    beat_nodes: dict[str, Any],
    beat_dilemmas: dict[str, set[str]],
) -> list[IntersectionCandidate]:
    """Group beats by shared entity references."""
    # Build entity → beats mapping from features edges
    # Normalize to raw IDs (without prefix) since entities may have different
    # prefix formats (character::X, location::Y, or legacy entity::X)
    entity_beats: dict[str, list[str]] = defaultdict(list)
    features_edges = graph.get_edges(from_id=None, to_id=None, edge_type="features")
    for edge in features_edges:
        beat_id = edge["from"]
        entity_id = edge["to"]
        if beat_id in beat_nodes:
            raw_entity_id = strip_scope_prefix(entity_id)
            entity_beats[raw_entity_id].append(beat_id)

    # Also check entity references in beat data.
    # beat_data["entities"] may contain raw IDs ("mentor") or prefixed IDs.
    # Normalize to raw ID form to match across different prefix formats.
    for beat_id, beat_data in beat_nodes.items():
        entities = beat_data.get("entities", [])
        for entity_ref in entities:
            raw_entity_id = strip_scope_prefix(entity_ref)
            entity_beats[raw_entity_id].append(beat_id)

    candidates: list[IntersectionCandidate] = []
    seen_pairs: set[tuple[str, ...]] = set()

    for entity_id, beats in sorted(entity_beats.items()):
        unique_beats = sorted(set(beats))
        if len(unique_beats) < 2:
            continue
        multi_dilemma = _filter_different_dilemmas(unique_beats, beat_dilemmas)
        if len(multi_dilemma) >= 2:
            limited = _limit_one_beat_per_dilemma(multi_dilemma, beat_dilemmas)
            if len(limited) < 2:
                continue
            key = tuple(limited)
            if key not in seen_pairs:
                seen_pairs.add(key)
                candidates.append(
                    IntersectionCandidate(
                        beat_ids=limited,
                        signal_type="entity",
                        shared_value=entity_id,
                    )
                )

    return candidates


def _filter_different_dilemmas(
    beat_ids: list[str],
    beat_dilemmas: dict[str, set[str]],
) -> list[str]:
    """Filter to beats that span at least 2 different dilemmas.

    Returns all beats if the group spans multiple dilemmas,
    empty list otherwise.
    """
    all_dilemmas: set[str] = set()
    for bid in beat_ids:
        all_dilemmas.update(beat_dilemmas.get(bid, set()))
    if len(all_dilemmas) < 2:
        return []
    return sorted(beat_ids)


# Maximum transitive closure depth for prerequisite lifting.
# Beyond this depth, the dependency chain is too deep to safely widen.
_MAX_LIFT_DEPTH = 3


def _try_lift_prerequisite(
    graph: Graph,
    prereq_id: str,
    target_paths: set[str],
    beat_paths: dict[str, set[str]],
    *,
    _depth: int = 0,
) -> bool:
    """Try to widen a prerequisite beat to cover all target paths.

    Adds ``belongs_to`` edges so the prerequisite (and its own
    prerequisites, transitively) spans all paths in the intersection.

    Args:
        graph: Graph to mutate if lift succeeds.
        prereq_id: The prerequisite beat to widen.
        target_paths: The set of paths the intersection spans.
        beat_paths: Mutable mapping of beat_id → set of path IDs.
        _depth: Current recursion depth (internal).

    Returns:
        True if the prerequisite was successfully lifted to cover
        all target_paths; False if lifting would be unsafe.
    """
    if _depth > _MAX_LIFT_DEPTH:
        return False

    current_paths = beat_paths.get(prereq_id, set())
    missing_paths = target_paths - current_paths

    if not missing_paths:
        return True  # Already covers all target paths

    # Check for cycles: if any target_path beat has a requires edge
    # TO this prereq through the intersection beats, lifting would
    # create a cycle. Simple check: does the prereq transitively
    # require any beat that already belongs to all target_paths?
    # (This is a conservative check — full cycle detection is expensive.)

    # First, transitively lift this prereq's own prerequisites
    for edge in graph.get_edges(from_id=prereq_id, to_id=None, edge_type="predecessor"):
        sub_prereq_id = edge["to"]
        sub_paths = beat_paths.get(sub_prereq_id, set())
        if not sub_paths >= target_paths and not _try_lift_prerequisite(
            graph, sub_prereq_id, target_paths, beat_paths, _depth=_depth + 1
        ):
            return False

    # All transitive prereqs lifted successfully — now lift this one
    for path_id in missing_paths:
        graph.add_edge("belongs_to", prereq_id, path_id)

    beat_paths[prereq_id] = current_paths | missing_paths

    log.debug(
        "prerequisite_lifted",
        prereq_id=prereq_id,
        added_paths=sorted(missing_paths),
        depth=_depth,
    )
    return True


def _try_split_beat(
    graph: Graph,
    beat_id: str,
    prereq_id: str,
    narrow_paths: set[str],
    wide_paths: set[str],
    beat_paths: dict[str, set[str]],
) -> str | None:
    """Split a beat into two variants for different path sets.

    Creates a new beat variant for the narrow paths (keeping the
    prerequisite), and narrows the original to the wide paths
    (without the prerequisite).

    Args:
        graph: Graph to mutate.
        beat_id: The intersection beat to split.
        prereq_id: The prerequisite that can't be lifted.
        narrow_paths: Paths where the prerequisite exists.
        wide_paths: Paths where the prerequisite doesn't exist.
        beat_paths: Mutable mapping of beat_id → set of path IDs.

    Returns:
        The variant beat ID if split succeeded, None if failed.
    """
    beat_data = graph.get_node(beat_id)
    if beat_data is None:
        return None

    # Use prereq ID in suffix to disambiguate multiple splits on the same beat.
    prereq_suffix = prereq_id.rsplit("::", 1)[-1] if "::" in prereq_id else prereq_id
    variant_id = f"{beat_id}_split_{prereq_suffix}"
    if graph.has_node(variant_id):
        # Fall back to generic suffix
        variant_id = f"{beat_id}_split"
        if graph.has_node(variant_id):
            return None  # Name collision — can't split

    # Create variant with same data but different ID
    raw_variant = variant_id.rsplit("::", 1)[-1] if "::" in variant_id else variant_id
    variant_data = {
        **beat_data,
        "raw_id": raw_variant,
        "split_from": beat_id,
    }
    graph.create_node(variant_id, variant_data)

    # Variant gets belongs_to edges for narrow_paths only
    for path_id in narrow_paths:
        graph.add_edge("belongs_to", variant_id, path_id)

    # Variant keeps the requires edge to the prereq
    graph.add_edge("predecessor", variant_id, prereq_id)

    # Remove the narrow_paths from the original beat's belongs_to
    # (The original beat keeps the wide_paths.)
    # Note: we can't remove edges from the graph directly, so we track
    # in beat_paths which paths the original beat effectively covers.
    # The actual belongs_to edges for narrow_paths remain but the
    # intersection will use the variant for those paths.
    beat_paths[variant_id] = narrow_paths
    beat_paths[beat_id] = wide_paths

    log.debug(
        "beat_split_for_prerequisite",
        original=beat_id,
        variant=variant_id,
        prereq=prereq_id,
        narrow_paths=sorted(narrow_paths),
        wide_paths=sorted(wide_paths),
    )
    return variant_id


def check_intersection_compatibility(
    graph: Graph,
    beat_ids: list[str],
    *,
    max_intersection_size: int = 3,
    allow_prerequisite_recovery: bool = False,
) -> list[GrowValidationError]:
    """Check if beats can form a valid intersection.

    Validates:
    - All beat IDs exist in the graph
    - Beats are from different dilemmas (not same dilemma)
    - No circular requires conflicts between the beats
    - At least 2 beats

    For conditional prerequisites (beat requires a prerequisite that doesn't
    span all intersection paths), the default strategy is to reject the
    intersection. Optional recovery strategies (lift/split) can be enabled via
    ``allow_prerequisite_recovery``.

    Args:
        graph: Graph with beat and path nodes.
        beat_ids: Proposed intersection beat IDs.
        max_intersection_size: Maximum allowed beats per intersection.
        allow_prerequisite_recovery: If True, attempt lift/split before rejecting.

    Returns:
        List of validation errors. Empty if compatible.
    """
    errors: list[GrowValidationError] = []

    if len(beat_ids) < 2:
        errors.append(
            GrowValidationError(
                field_path="intersection.beat_ids",
                issue="Intersection requires at least 2 beats",
                category=GrowErrorCategory.STRUCTURAL,
            )
        )
        return errors

    if len(beat_ids) > max_intersection_size:
        errors.append(
            GrowValidationError(
                field_path="intersection.beat_ids",
                issue=(
                    f"Intersection has {len(beat_ids)} beats; "
                    f"maximum allowed is {max_intersection_size}"
                ),
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
                    field_path=f"intersection.beat_ids.{bid}",
                    issue=f"Beat '{bid}' not found in graph",
                    category=GrowErrorCategory.REFERENCE,
                )
            )

    if errors:
        return errors

    # Check beats are from different dilemmas
    beat_dilemma_map = _build_beat_dilemmas(graph, beat_nodes)
    beat_primary_dilemma: dict[str, str] = {}
    for bid in beat_ids:
        dilemmas = beat_dilemma_map.get(bid, set())
        if len(dilemmas) != 1:
            errors.append(
                GrowValidationError(
                    field_path="intersection.dilemmas",
                    issue=(
                        f"Beat '{bid}' maps to {len(dilemmas)} dilemmas: {sorted(dilemmas)}. "
                        f"Each beat in an intersection must map to exactly 1 dilemma."
                    ),
                    category=GrowErrorCategory.STRUCTURAL,
                )
            )
            continue
        beat_primary_dilemma[bid] = next(iter(dilemmas))

    if errors:
        return errors

    # Intersections must span at least 2 different dilemmas and include
    # at most 1 beat per dilemma (otherwise exclusivity collapses).
    dilemma_to_beats: dict[str, list[str]] = defaultdict(list)
    for bid, did in beat_primary_dilemma.items():
        dilemma_to_beats[did].append(bid)

    if len(dilemma_to_beats) < 2:
        only = sorted(dilemma_to_beats.keys())
        errors.append(
            GrowValidationError(
                field_path="intersection.dilemmas",
                issue=(
                    f"Beats span only {len(only)} dilemma(s): {only}. "
                    f"Intersections must span at least 2 different dilemmas."
                ),
                category=GrowErrorCategory.STRUCTURAL,
            )
        )

    multi = {d: sorted(bs) for d, bs in dilemma_to_beats.items() if len(bs) > 1}
    if multi:
        errors.append(
            GrowValidationError(
                field_path="intersection.dilemmas",
                issue=(
                    "Intersection contains multiple beats from the same dilemma(s): "
                    f"{multi}. Intersections must include at most 1 beat per dilemma."
                ),
                category=GrowErrorCategory.STRUCTURAL,
            )
        )

    # Hard policy no longer forces topology isolation; intersections are allowed
    # and may be split later during Phase 10 (Routing).

    # Check predecessor edges originating from intersection beats.
    # Two checks in one pass: (1) no circular predecessor between intersection
    # beats, and (2) no conditional prerequisites (see below).
    beat_set = set(beat_ids)

    # Build beat→paths mapping for the conditional-prerequisites check.
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    beat_paths: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        beat_paths.setdefault(edge["from"], set()).add(edge["to"])

    # The union of all paths that the intersection would span
    union_paths: set[str] = set()
    for bid in beat_ids:
        union_paths.update(beat_paths.get(bid, set()))

    # Iterate only over outgoing predecessor edges from intersection beats
    # (targeted lookups instead of scanning all predecessor edges).
    for from_id in beat_set:
        for edge in graph.get_edges(from_id=from_id, to_id=None, edge_type="predecessor"):
            to_id = edge["to"]
            if to_id in beat_set:
                # Circular predecessor edge between intersection beats
                errors.append(
                    GrowValidationError(
                        field_path="intersection.predecessor",
                        issue=(
                            f"Beat '{from_id}' has predecessor edge to '{to_id}' — "
                            f"intersection beats cannot have ordering "
                            f"dependencies on each other"
                        ),
                        category=GrowErrorCategory.STRUCTURAL,
                    )
                )
            else:
                # No-conditional-prerequisites invariant: a shared beat
                # cannot depend on a beat that exists only on a strict
                # subset of its paths.  After intersection marking, every
                # beat in the group will belong to the union of all paths.
                # If a `predecessor` target is narrower, that edge would be
                # silently dropped in arcs missing the target's path,
                # producing inconsistent orderings and passage DAG cycles.
                #
                prereq_paths = beat_paths.get(to_id, set())
                if not prereq_paths >= union_paths:
                    if allow_prerequisite_recovery:
                        # Try lift first: widen prerequisite to cover union_paths
                        lifted = _try_lift_prerequisite(graph, to_id, union_paths, beat_paths)
                        if lifted:
                            continue

                        # Try split: create variant for narrow paths
                        narrow = prereq_paths & beat_paths.get(from_id, set())
                        wide = union_paths - narrow
                        if narrow and wide:
                            variant = _try_split_beat(
                                graph, from_id, to_id, narrow, wide, beat_paths
                            )
                            if variant is not None:
                                continue

                    missing = sorted(union_paths - prereq_paths)
                    errors.append(
                        GrowValidationError(
                            field_path="intersection.conditional_prerequisite",
                            issue=(
                                f"Beat '{from_id}' has predecessor edge to '{to_id}' which "
                                f"is only on paths {sorted(prereq_paths)}, "
                                f"but the intersection would span "
                                f"{sorted(union_paths)}. "
                                f"Missing paths: {missing}. "
                                + (
                                    "Lift and split strategies both failed."
                                    if allow_prerequisite_recovery
                                    else "Conditional prerequisites are not allowed."
                                )
                            ),
                            category=GrowErrorCategory.STRUCTURAL,
                        )
                    )

    return errors


def resolve_intersection_location(graph: Graph, beat_ids: list[str]) -> str | None:
    """Find a shared location for the intersection beats.

    Resolution priority:
    1. Shared primary location
    2. Primary location of one that appears in alternatives of another
    3. Shared alternative location
    4. None if no common location found

    Args:
        graph: Graph with beat nodes.
        beat_ids: Beat IDs in the proposed intersection.

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
        # Read flexibility edges (Doc 3) instead of location_alternatives property
        for edge in graph.get_edges(from_id=bid, edge_type="flexibility"):
            locs.add(edge["to"])
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


def apply_intersection_mark(
    graph: Graph,
    beat_ids: list[str],
    resolved_location: str | None,
    shared_entities: list[str] | None = None,
    rationale: str | None = None,
) -> None:
    """Mark beats as co-occurring in an intersection group (Doc 3, Part 4).

    Creates an ``intersection_group`` node and links each participating
    beat to it via ``intersection`` edges.  Each beat keeps its single
    ``belongs_to`` edge to its original path — no cross-path assignment.

    Args:
        graph: Graph to mutate.
        beat_ids: Beat IDs to group into intersection.
        resolved_location: Resolved location, or None.
        shared_entities: Entity IDs shared between the intersecting beats.
        rationale: One-sentence explanation of why these beats form a natural scene.
    """
    # Derive a stable group ID from the sorted beat IDs
    sorted_ids = sorted(beat_ids)
    raw_group_id = "--".join(strip_scope_prefix(b) for b in sorted_ids)
    group_node_id = f"intersection_group::{raw_group_id}"

    # Idempotency: skip if this group already exists (same beat pair proposed twice)
    if graph.get_node(group_node_id) is not None:
        return

    # Build group node data
    group_data: dict[str, Any] = {
        "type": "intersection_group",
        "raw_id": raw_group_id,
        "beat_ids": sorted_ids,
        "shared_entities": shared_entities or [],
        "rationale": rationale or "",
    }
    if resolved_location:
        group_data["resolved_location"] = resolved_location

    graph.create_node(group_node_id, group_data)

    # Link each beat to the group via intersection edges
    for bid in beat_ids:
        graph.add_edge("intersection", bid, group_node_id)
        # Update beat's location if a resolved location was determined
        if resolved_location:
            graph.update_node(bid, location=resolved_location)


# ---------------------------------------------------------------------------
# Phase 4: Gap detection algorithms
# ---------------------------------------------------------------------------


@dataclass
class PacingIssue:
    """A sequence of 3+ consecutive beats with the same scene_type."""

    path_id: str
    beat_ids: list[str]
    scene_type: str


def get_path_beat_sequence(graph: Graph, path_id: str) -> list[str]:
    """Get ordered beat sequence for a path using topological sort on requires edges.

    Delegates to topological_sort_beats() for the sorting logic.

    Args:
        graph: Graph with beat nodes and requires edges.
        path_id: Prefixed path ID (e.g., "path::mentor_trust_canonical").

    Returns:
        Ordered list of beat IDs in the path.

    Raises:
        ValueError: If a cycle is detected among the path's beats.
    """
    belongs_to_edges = graph.get_edges(from_id=None, to_id=path_id, edge_type="belongs_to")
    path_beats = [e["from"] for e in belongs_to_edges]

    if not path_beats:
        return []

    return topological_sort_beats(graph, path_beats)


def detect_pacing_issues(graph: Graph) -> list[PacingIssue]:
    """Detect pacing issues: 3+ consecutive beats with the same scene_type.

    Checks each path's beat sequence for runs of 3 or more beats
    all tagged with the same scene_type (scene, sequel, or micro_beat).

    Args:
        graph: Graph with beat nodes that have scene_type data.

    Returns:
        List of PacingIssue objects describing problematic sequences.
    """
    issues: list[PacingIssue] = []
    path_nodes = graph.get_nodes_by_type("path")

    for pid in sorted(path_nodes.keys()):
        sequence = get_path_beat_sequence(graph, pid)
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
                        path_id=pid,
                        beat_ids=[bt[0] for bt in beat_types[run_start:run_end]],
                        scene_type=current_type,
                    )
                )

            run_start = run_end

    return issues


def insert_gap_beat(
    graph: Graph,
    path_id: str,
    after_beat: str | None,
    before_beat: str | None,
    summary: str,
    scene_type: str,
    dilemma_impacts: list[dict[str, Any]] | None = None,
) -> str:
    """Insert a new gap beat into the graph between existing beats.

    Creates a new beat node and adjusts requires edges to maintain ordering.
    The new beat is assigned to the specified path.

    Gap beats inherit entities (union) and location from adjacent beats to
    provide context for FILL stage transitions. A transition_style field
    indicates whether the gap should be a smooth continuation or a hard cut.

    Args:
        graph: Graph to mutate.
        path_id: Path this beat belongs to (prefixed ID).
        after_beat: Beat that should come before the new beat (or None for start).
        before_beat: Beat that should come after the new beat (or None for end).
        summary: Summary text for the new beat.
        scene_type: Scene type tag for the new beat.
        dilemma_impacts: List of dilemma impact dicts (dilemma_id, effect, note).

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

    # Get adjacent beat nodes for inheritance
    after_node = graph.get_node(after_beat) if after_beat else None
    before_node = graph.get_node(before_beat) if before_beat else None

    # Inherit entities (union of both adjacent beats, deduplicated)
    entities: list[str] = []
    if after_node:
        after_ents = after_node.get("entities")
        if isinstance(after_ents, list):
            entities.extend(after_ents)
    if before_node:
        before_ents = before_node.get("entities")
        if isinstance(before_ents, list):
            entities.extend(before_ents)
    entities = list(dict.fromkeys(entities))  # Deduplicate preserving order

    # Inherit location (prefer shared location, fallback to either)
    after_loc = after_node.get("location") if after_node else None
    before_loc = before_node.get("location") if before_node else None
    location = after_loc if after_loc == before_loc else (after_loc or before_loc)

    # Infer transition style based on context
    transition_style = _infer_transition_style(after_node, before_node)

    # Create the beat node with enriched context
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": raw_id,
            "summary": summary,
            "scene_type": scene_type,
            "paths": [path_id.removeprefix("path::")],
            "is_gap_beat": True,
            # Enrichment fields for transition handling
            "entities": entities,
            "location": location,
            "transition_style": transition_style,
            "bridges_from": after_beat,
            "bridges_to": before_beat,
            "dilemma_impacts": dilemma_impacts or [],
        },
    )

    # Add belongs_to edge
    graph.add_edge("belongs_to", beat_id, path_id)

    # Adjust requires edges for ordering.
    # Existing transitive requires (before_beat → after_beat) is kept as redundant
    # but harmless for topological sort correctness.
    if after_beat:
        graph.add_edge("predecessor", beat_id, after_beat)

    if before_beat:
        graph.add_edge("predecessor", before_beat, beat_id)

    return beat_id


def _infer_transition_style(
    from_beat: dict[str, object] | None,
    to_beat: dict[str, object] | None,
) -> Literal["smooth", "cut"]:
    """Infer whether a gap transition should be smooth or a hard cut.

    Heuristics:
    - Same location + shared entities → smooth
    - Different locations → cut
    - Different scene types → cut
    - No shared entities but same location → smooth

    Args:
        from_beat: The beat before the gap (or None).
        to_beat: The beat after the gap (or None).

    Returns:
        "smooth" or "cut" based on context analysis.
    """
    if not from_beat or not to_beat:
        return "smooth"  # Default when context is missing

    from_loc = from_beat.get("location")
    to_loc = to_beat.get("location")

    # Different locations usually warrant a cut
    if from_loc and to_loc and from_loc != to_loc:
        return "cut"

    # Scene type changes often need cuts
    if from_beat.get("scene_type") != to_beat.get("scene_type"):
        return "cut"

    # Same location with any shared entities → smooth
    from_ent_raw = from_beat.get("entities")
    to_ent_raw = to_beat.get("entities")
    from_entities: set[str] = set(from_ent_raw) if isinstance(from_ent_raw, list) else set()
    to_entities: set[str] = set(to_ent_raw) if isinstance(to_ent_raw, list) else set()
    if from_loc == to_loc and from_entities & to_entities:
        return "smooth"

    return "smooth"  # Default to smooth for continuity


ARC_TYPE_BY_ENTITY_TYPE: dict[str, str] = {
    "character": "transformation",
    "location": "atmosphere",
    "object": "significance",
    "faction": "relationship",
}
"""Deterministic arc-type mapping. Character arcs describe internal change,
object arcs describe meaning shifts, location arcs describe atmosphere shifts,
faction arcs describe relationship changes."""


def select_entities_for_arc(
    graph: Graph,
    path_id: str,
    beat_sequence: list[str],
) -> list[str]:
    """Select entities eligible for arc generation on a path.

    Selection rules:
    - Characters/factions with 2+ appearances on this path's beats
    - Characters/factions listed in the path's dilemma ``involves`` field
    - Objects/locations with 1+ appearance (they can carry thematic weight
      even in a single scene)

    Args:
        graph: Graph with entity and beat nodes.
        path_id: Prefixed path ID.
        beat_sequence: Ordered beat IDs for this path.

    Returns:
        Sorted list of entity IDs eligible for arc generation.
    """
    from collections import Counter

    appearance_count: Counter[str] = Counter()
    for beat_id in beat_sequence:
        beat = graph.get_node(beat_id)
        if beat is None:
            continue
        for eid in beat.get("entities", []):
            appearance_count[eid] += 1

    # Collect dilemma-involved entities for this path
    path_node = graph.get_node(path_id)
    dilemma_involved: set[str] = set()
    if path_node:
        dilemma_id = path_node.get("dilemma_id", "")
        dilemma_node = graph.get_node(dilemma_id)
        if dilemma_node:
            dilemma_involved = set(dilemma_node.get("involves", []))

    eligible: set[str] = set()
    for eid, count in appearance_count.items():
        entity_node = graph.get_node(eid)
        if entity_node is None:
            continue
        entity_type = entity_node.get("entity_type", "")
        if not entity_type:
            log.warning("entity_missing_type", entity_id=eid)
            continue
        if entity_type in ("object", "location"):
            # Objects/locations always eligible with 1+ appearance
            eligible.add(eid)
        elif count >= 2 or eid in dilemma_involved:
            # Characters/factions need 2+ appearances or dilemma involvement
            eligible.add(eid)

    return sorted(eligible)


# ---------------------------------------------------------------------------
# Cross-Path Beat Interleaving
# ---------------------------------------------------------------------------


def _get_path_beats_ordered(
    graph: Graph,
    path_id: str,
    path_beats_map: dict[str, list[str]],
) -> list[str]:
    """Return beats for a path in topological order.

    Args:
        graph: The story graph.
        path_id: Scoped path node ID.
        path_beats_map: Pre-computed mapping of path_id → list of beat IDs.

    Returns:
        Beat IDs in topological order (prerequisites first). Empty list if no beats.
    """
    beats = path_beats_map.get(path_id, [])
    if not beats:
        return []
    try:
        return topological_sort_beats(graph, beats)
    except ValueError:
        log.warning(
            "interleave_path_cycle_fallback",
            path_id=path_id,
            beats=beats,
        )
        return sorted(beats)  # Fallback to alphabetical on cycle (should not happen)


def _commits_beats_for_dilemma(
    beats: list[str],
    dilemma_id: str,
    beat_nodes: dict[str, Any],
) -> list[str]:
    """Return beat IDs that have effect='commits' for the given dilemma.

    Args:
        beats: Beat IDs to search within.
        dilemma_id: Scoped dilemma ID to match against impact records.
        beat_nodes: Pre-fetched beat node data.

    Returns:
        List of beat IDs with the commits effect for this dilemma.
    """
    result = []
    for bid in beats:
        data = beat_nodes.get(bid, {})
        for impact in data.get("dilemma_impacts", []):
            if impact.get("dilemma_id") == dilemma_id and impact.get("effect") == "commits":
                result.append(bid)
                break
    return result


def _would_create_cycle(
    new_from: str,
    new_to: str,
    successors: dict[str, set[str]],
    beat_set: set[str],
) -> bool:
    """Check if adding a predecessor edge (new_from requires new_to) creates a cycle.

    In our DAG: ``predecessor`` edge (X, Y) means X requires Y, i.e. Y comes before X.
    Adding predecessor(new_from, new_to) means new_to → new_from in topological order.
    A cycle would exist if new_to is already reachable from new_from via existing edges.

    Args:
        new_from: The beat that would require new_to.
        new_to: The beat that would become a prerequisite.
        successors: Current forward adjacency (prerequisite → dependents).
        beat_set: All beat IDs in the graph.

    Returns:
        True if adding this edge would create a cycle.
    """
    # A cycle exists if new_to is already reachable from new_from via successors.
    # Adding predecessor(new_from, new_to) means new_to executes before new_from.
    # If new_to is already reachable from new_from (new_from → ... → new_to),
    # then adding new_to → new_from closes a cycle.
    if new_from not in beat_set or new_to not in beat_set:
        return False
    visited: set[str] = set()
    queue = [new_from]
    while queue:
        node = queue.pop()
        if node == new_to:
            return True
        if node in visited:
            continue
        visited.add(node)
        queue.extend(successors.get(node, set()))
    return False


def _strip_temporal_hints(graph: Graph, beat_nodes: dict[str, Any]) -> int:
    """Remove temporal_hint from all beat nodes.

    Hints have served their purpose once the beat DAG encodes the ordering.
    Sets temporal_hint to None (JSON null) which satisfies the spec's
    "not carried forward" requirement and the SQL verification query.

    Returns:
        Count of beats that had a hint stripped.
    """
    stripped = 0
    for beat_id, data in beat_nodes.items():
        if data.get("temporal_hint") is not None:
            graph.update_node(beat_id, temporal_hint=None)
            stripped += 1
    return stripped


@dataclass
class TemporalHintConflict:
    """A temporal hint that would create a cycle in the beat ordering DAG.

    Produced by ``detect_temporal_hint_conflicts`` and consumed by the
    ``resolve_temporal_hints`` phase to ask the LLM which hints to drop.
    """

    beat_id: str
    hint_relative_to: str  # dilemma the hint references
    hint_position: str  # before_commit | after_commit | before_introduce | after_introduce
    from_beat: str  # proposed predecessor in the rejected edge
    to_beat: str  # proposed dependent in the rejected edge
    beat_summary: str  # beat summary for LLM context


@dataclass
class _HintEdge:
    from_beat: str
    to_beat: str
    beat_id: str
    relative_to: str
    position: str


def _iter_temporal_hint_edges(
    all_beats: list[str],
    beat_nodes: dict[str, dict[str, Any]],
    dilemma_a: str,
    dilemma_b: str,
    all_beats_a: list[str],
    ordered_a: list[list[str]],
    all_beats_b: list[str],
    ordered_b: list[list[str]],
    beat_id_to_dilemmas: dict[str, set[str]],
) -> Iterator[_HintEdge]:
    """Yield candidate hint edges for a concurrent dilemma pair.

    Shared iteration logic used by both ``detect_temporal_hint_conflicts``
    (simulation, no edge creation) and ``interleave_cross_path_beats``
    (actual edge application).  Does not perform cycle detection or
    duplicate checks — those are the caller's responsibility.

    Args:
        all_beats: Combined beat list for the dilemma pair (a + b).
        beat_nodes: All beat node data keyed by beat ID.
        dilemma_a: ID of the first dilemma.
        dilemma_b: ID of the second dilemma.
        all_beats_a: Flat beat list for dilemma_a.
        ordered_a: Per-path ordered beat sequences for dilemma_a.
        all_beats_b: Flat beat list for dilemma_b.
        ordered_b: Per-path ordered beat sequences for dilemma_b.
        beat_id_to_dilemmas: Mapping of beat ID → set of owning dilemma IDs.

    Yields:
        ``_HintEdge`` for each candidate (from_beat, to_beat) pair derived
        from temporal hints, before cycle or duplicate filtering.
    """
    for beat_id in all_beats:
        data = beat_nodes.get(beat_id, {})
        hint = data.get("temporal_hint")
        if not isinstance(hint, dict):
            continue
        relative_to = hint.get("relative_to", "")
        position = hint.get("position", "")
        if not relative_to or not position:
            continue

        beat_own_dilemmas = beat_id_to_dilemmas.get(beat_id, set())
        if relative_to in beat_own_dilemmas:
            continue  # same-dilemma guard

        if relative_to == dilemma_a:
            ref_all, ref_ordered, ref_dil = all_beats_a, ordered_a, dilemma_a
        elif relative_to == dilemma_b:
            ref_all, ref_ordered, ref_dil = all_beats_b, ordered_b, dilemma_b
        else:
            continue

        ref_commits = _commits_beats_for_dilemma(ref_all, ref_dil, beat_nodes)
        ref_first = [seq[0] for seq in ref_ordered if seq]
        is_before = position.startswith("before_")
        target_beats = ref_commits if "commit" in position else ref_first

        for target in sorted(target_beats):
            from_b, to_b = (target, beat_id) if is_before else (beat_id, target)
            yield _HintEdge(
                from_beat=from_b,
                to_beat=to_b,
                beat_id=beat_id,
                relative_to=relative_to,
                position=position,
            )


def detect_temporal_hint_conflicts(graph: Graph) -> list[TemporalHintConflict]:
    """Simulate temporal hint edge application and return hints that would create cycles.

    Replicates the full edge-building logic of ``interleave_cross_path_beats``
    (serial, wraps, and concurrent including heuristic commit-ordering edges)
    without committing any edges to the graph.  Hints that would be
    skipped as cycle-creating are returned as ``TemporalHintConflict`` objects
    for LLM resolution before interleave runs.

    Returns:
        List of conflicting hints.  Empty if all hints are consistent.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    dilemma_paths = build_dilemma_paths(graph)
    if len(dilemma_paths) < 2:
        return []

    path_beats_map: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to"):
        path_beats_map[edge["to"]].append(edge["from"])

    beat_id_to_dilemmas: dict[str, set[str]] = defaultdict(set)
    for dil_id, paths in dilemma_paths.items():
        for path_id in paths:
            for bid in path_beats_map.get(path_id, []):
                beat_id_to_dilemmas[bid].add(dil_id)

    # Start with existing predecessor edges (intra-path ordering already in the DAG).
    # Must use "predecessor" — "requires" edges do not exist in the beat DAG.
    existing: set[tuple[str, str]] = set()
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="predecessor"):
        existing.add((edge["from"], edge["to"]))

    beat_set = set(beat_nodes.keys())
    successors: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    for from_id, to_id in existing:
        if from_id in successors and to_id in successors:
            successors[to_id].add(from_id)

    # Build intersection-group index to match interleave's skip logic.
    beat_intersection_groups: defaultdict[str, set[str]] = defaultdict(set)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="intersection"):
        beat_intersection_groups[edge["from"]].add(edge["to"])

    def _is_valid_edge_candidate(from_b: str, to_b: str) -> bool:
        """Check pre-conditions for adding an edge, before cycle detection."""
        if from_b == to_b:
            return False
        if (from_b, to_b) in existing:
            return False
        if from_b not in beat_set or to_b not in beat_set:
            return False
        from_groups = beat_intersection_groups.get(from_b, set())
        to_groups = beat_intersection_groups.get(to_b, set())
        return not from_groups.intersection(to_groups)

    def _sim_add(from_b: str, to_b: str) -> bool:
        """Simulate adding a non-hint edge (serial/wraps/heuristic), updating state."""
        if not _is_valid_edge_candidate(from_b, to_b):
            return False
        if _would_create_cycle(from_b, to_b, successors, beat_set):
            return False
        existing.add((from_b, to_b))
        successors[to_b].add(from_b)
        return True

    conflicts: list[TemporalHintConflict] = []

    # Collect ALL relationship edges in the same order as interleave_cross_path_beats.
    relationship_edges: list[tuple[str, str, str]] = []
    for ordering in ("concurrent", "wraps", "serial"):
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type=ordering):
            a = edge["from"]
            b = edge["to"]
            if a in dilemma_paths and b in dilemma_paths:
                relationship_edges.append((a, b, ordering))

    for dilemma_a, dilemma_b, ordering in relationship_edges:
        paths_a = dilemma_paths.get(dilemma_a, [])
        paths_b = dilemma_paths.get(dilemma_b, [])
        if not paths_a or not paths_b:
            continue

        ordered_a = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_a]
        ordered_b = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_b]
        all_beats_a = [b for seq in ordered_a for b in seq]
        all_beats_b = [b for seq in ordered_b for b in seq]

        if not all_beats_a or not all_beats_b:
            continue

        if ordering == "serial":
            last_beats_a = {seq[-1] for seq in ordered_a if seq}
            first_beats_b = {seq[0] for seq in ordered_b if seq}
            for last_a in sorted(last_beats_a):
                for first_b in sorted(first_beats_b):
                    _sim_add(first_b, last_a)

        elif ordering == "wraps":
            first_beats_a = {seq[0] for seq in ordered_a if seq}
            first_beats_b = {seq[0] for seq in ordered_b if seq}
            last_beats_b = {seq[-1] for seq in ordered_b if seq}
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))
            for first_a in sorted(first_beats_a):
                for first_b in sorted(first_beats_b):
                    _sim_add(first_b, first_a)
            for last_b in sorted(last_beats_b):
                for commit_a in sorted(commits_a):
                    _sim_add(commit_a, last_b)

        elif ordering == "concurrent":
            # Temporal hints first (same order as interleave_cross_path_beats)
            for hint_edge in _iter_temporal_hint_edges(
                all_beats_a + all_beats_b,
                beat_nodes,
                dilemma_a,
                dilemma_b,
                all_beats_a,
                ordered_a,
                all_beats_b,
                ordered_b,
                beat_id_to_dilemmas,
            ):
                from_b, to_b = hint_edge.from_beat, hint_edge.to_beat
                if not _is_valid_edge_candidate(from_b, to_b):
                    continue
                if _would_create_cycle(from_b, to_b, successors, beat_set):
                    conflicts.append(
                        TemporalHintConflict(
                            beat_id=hint_edge.beat_id,
                            hint_relative_to=hint_edge.relative_to,
                            hint_position=hint_edge.position,
                            from_beat=from_b,
                            to_beat=to_b,
                            beat_summary=beat_nodes.get(hint_edge.beat_id, {}).get("summary", ""),
                        )
                    )
                else:
                    existing.add((from_b, to_b))
                    successors[to_b].add(from_b)

            # Heuristic commit-ordering edges — MUST match interleave_cross_path_beats
            # so later concurrent pairs see the same graph state.
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))
            commits_b = set(_commits_beats_for_dilemma(all_beats_b, dilemma_b, beat_nodes))
            if commits_a and commits_b:
                if dilemma_a < dilemma_b:
                    prereq_commits, dependent_commits = commits_a, commits_b
                else:
                    prereq_commits, dependent_commits = commits_b, commits_a
                for prereq in sorted(prereq_commits):
                    for dependent in sorted(dependent_commits):
                        _sim_add(dependent, prereq)

    return conflicts


# ---------------------------------------------------------------------------
# Conflict-graph-based detection (replaces detect_temporal_hint_conflicts)
# ---------------------------------------------------------------------------


@dataclass
class HintConflict:
    """A pair of hint-bearing beats that cannot both survive.

    When ``mandatory`` is True, ``beat_a`` must be dropped regardless of
    ``beat_b`` — its hint creates a cycle even in isolation against the
    base DAG (all non-hint edges).  When ``mandatory`` is False, this is a
    swap pair: either beat_a or beat_b may survive, but not both.

    Attributes:
        beat_a: First beat in the conflict pair.
        beat_b: Second beat (None for mandatory solo drops).
        mandatory: True if beat_a MUST be dropped (no swap available).
        default_drop: Heuristically-preferred drop (beat_a or beat_b).
    """

    beat_a: str
    beat_b: str | None
    mandatory: bool
    default_drop: str


@dataclass
class HintConflictResult:
    """Full conflict analysis produced by ``build_hint_conflict_graph``.

    Attributes:
        conflicts: All HintConflict objects (mandatory + swap pairs).
        mandatory_drops: Beat IDs that must be dropped (no swap available).
        swap_pairs: (beat_a, beat_b) pairs where LLM may choose which to drop.
        minimum_drop_set: Mechanical MDS: mandatory_drops + default choices for
            each swap pair.  This is applied when no LLM is available.
    """

    conflicts: list[HintConflict]
    mandatory_drops: set[str]
    swap_pairs: list[tuple[str, str]]
    minimum_drop_set: set[str]


def _hint_strength(position: str) -> int:
    """Return hint strength: 2=commit (strong), 1=introduce (weak)."""
    return 2 if "commit" in position else 1


def _is_canonical_beat(
    beat_id: str, path_beats_map: dict[str, list[str]], canonical_paths: set[str]
) -> bool:
    """Return True if the beat belongs to at least one canonical path."""
    for path_id, beats in path_beats_map.items():
        if beat_id in beats and path_id in canonical_paths:
            return True
    return False


class _HintResolutionContext(NamedTuple):
    """Pre-built indexes shared by build_hint_conflict_graph and verify_hints_acyclic."""

    dilemma_paths: dict[str, list[str]]
    path_beats_map: dict[str, list[str]]
    beat_id_to_dilemmas: dict[str, set[str]]
    beat_intersection_groups: defaultdict[str, set[str]]
    relationship_edges: list[tuple[str, str, str]]


def _build_hint_resolution_context(graph: Graph) -> _HintResolutionContext | None:
    """Build shared indexes needed for temporal-hint resolution.

    Returns ``None`` if the graph has fewer than two dilemmas (no cross-path
    ordering is possible and all hint functions should be no-ops).

    Args:
        graph: The story graph.

    Returns:
        A ``_HintResolutionContext`` with pre-built indexes, or ``None`` when
        there are fewer than two dilemmas.
    """
    dilemma_paths = build_dilemma_paths(graph)
    if len(dilemma_paths) < 2:
        return None

    path_beats_map: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to"):
        path_beats_map[edge["to"]].append(edge["from"])

    beat_id_to_dilemmas: dict[str, set[str]] = defaultdict(set)
    for dil_id, paths in dilemma_paths.items():
        for path_id in paths:
            for bid in path_beats_map.get(path_id, []):
                beat_id_to_dilemmas[bid].add(dil_id)

    beat_intersection_groups: defaultdict[str, set[str]] = defaultdict(set)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="intersection"):
        beat_intersection_groups[edge["from"]].add(edge["to"])

    relationship_edges: list[tuple[str, str, str]] = []
    for ordering in ("concurrent", "wraps", "serial"):
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type=ordering):
            a = edge["from"]
            b = edge["to"]
            if a in dilemma_paths and b in dilemma_paths:
                relationship_edges.append((a, b, ordering))

    return _HintResolutionContext(
        dilemma_paths=dilemma_paths,
        path_beats_map=path_beats_map,
        beat_id_to_dilemmas=beat_id_to_dilemmas,
        beat_intersection_groups=beat_intersection_groups,
        relationship_edges=relationship_edges,
    )


def _simulate_hints_sequential(
    hint_edges: list[_HintEdge],
    base_existing: set[tuple[str, str]],
    base_succ: dict[str, set[str]],
    beat_set: set[str],
    beat_intersection_groups: defaultdict[str, set[str]],
) -> list[_HintEdge]:
    """Simulate applying hint edges sequentially and return those that are rejected.

    Iterates ``hint_edges`` in order.  For each hint, if the edge is valid and
    would create a cycle against the accumulated DAG, the hint is added to the
    rejected list and NOT accumulated.  Otherwise, the edge is added to the
    working DAG so subsequent hints see its effect.

    This matches the sequential model used by ``verify_hints_acyclic``: the
    order of iteration matters because an earlier accepted hint may block a
    later one.

    Args:
        hint_edges: Ordered list of candidate hint edges to simulate.
        base_existing: Set of already-present (from_beat, to_beat) pairs
            (predecessor edges + any edges from non-hint simulation).
        base_succ: Adjacency map (prerequisite → set of dependents) for
            the base DAG.  Copied internally so the caller's map is unmodified.
        beat_set: All beat IDs present in the graph.
        beat_intersection_groups: Mapping of beat_id → set of intersection
            group IDs the beat belongs to.

    Returns:
        List of ``_HintEdge`` objects that would be rejected (would create a
        cycle when tested against the accumulated DAG at the point they are
        processed).
    """
    working_existing = set(base_existing)
    working_succ: dict[str, set[str]] = {bid: set(s) for bid, s in base_succ.items()}
    rejected: list[_HintEdge] = []

    for hint in hint_edges:
        from_b, to_b = hint.from_beat, hint.to_beat
        # Validity checks (mirror _build_hint_base_dag._valid logic)
        if from_b == to_b or from_b not in beat_set or to_b not in beat_set:
            continue
        if (from_b, to_b) in working_existing:
            continue
        if beat_intersection_groups.get(from_b, set()).intersection(
            beat_intersection_groups.get(to_b, set())
        ):
            continue
        if _would_create_cycle(from_b, to_b, working_succ, beat_set):
            rejected.append(hint)
        else:
            working_existing.add((from_b, to_b))
            working_succ[to_b].add(from_b)

    return rejected


def _build_hint_base_dag(
    graph: Graph,
    beat_nodes: dict[str, dict[str, object]],
    beat_set: set[str],
    beat_intersection_groups: defaultdict[str, set[str]],
    relationship_edges: list[tuple[str, str, str]],
    dilemma_paths: dict[str, list[str]],
    path_beats_map: dict[str, list[str]],
) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Build the base DAG for hint conflict detection/postcondition checking.

    Pre-loads ALL non-hint edges (predecessor + serial + wraps + concurrent
    commit-ordering) from ALL relationship pairs into the base DAG.  Hints are
    NOT included.  This is the shared DAG construction used by both
    ``build_hint_conflict_graph`` (detection) and ``verify_hints_acyclic``
    (postcondition), ensuring they produce consistent results.

    Args:
        graph: The story graph.
        beat_nodes: Mapping of beat_id → node data.
        beat_set: Set of all beat IDs.
        beat_intersection_groups: Mapping of beat_id → set of intersection group IDs.
        relationship_edges: List of (dilemma_a, dilemma_b, ordering) triples.
        dilemma_paths: Mapping of dilemma_id → list of path IDs.
        path_beats_map: Mapping of path_id → list of beat IDs.

    Returns:
        Tuple of (existing_edges set, successors adjacency dict).
    """
    existing: set[tuple[str, str]] = set()
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="predecessor"):
        existing.add((edge["from"], edge["to"]))
    succ: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    for from_id, to_id in existing:
        if from_id in succ and to_id in succ:
            succ[to_id].add(from_id)

    def _valid(from_b: str, to_b: str) -> bool:
        if from_b == to_b or (from_b, to_b) in existing:
            return False
        if from_b not in beat_set or to_b not in beat_set:
            return False
        return not beat_intersection_groups.get(from_b, set()).intersection(
            beat_intersection_groups.get(to_b, set())
        )

    def _sim(from_b: str, to_b: str) -> None:
        if not _valid(from_b, to_b):
            return
        if _would_create_cycle(from_b, to_b, succ, beat_set):
            return
        existing.add((from_b, to_b))
        succ[to_b].add(from_b)

    for dilemma_a, dilemma_b, ordering in relationship_edges:
        paths_a = dilemma_paths.get(dilemma_a, [])
        paths_b = dilemma_paths.get(dilemma_b, [])
        if not paths_a or not paths_b:
            continue
        ordered_a = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_a]
        ordered_b = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_b]
        all_beats_a = [b for seq in ordered_a for b in seq]
        all_beats_b = [b for seq in ordered_b for b in seq]
        if not all_beats_a or not all_beats_b:
            continue

        if ordering == "serial":
            for last_a in sorted({seq[-1] for seq in ordered_a if seq}):
                for first_b in sorted({seq[0] for seq in ordered_b if seq}):
                    _sim(first_b, last_a)
        elif ordering == "wraps":
            for first_a in sorted({seq[0] for seq in ordered_a if seq}):
                for first_b in sorted({seq[0] for seq in ordered_b if seq}):
                    _sim(first_b, first_a)
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))
            for last_b in sorted({seq[-1] for seq in ordered_b if seq}):
                for commit_a in sorted(commits_a):
                    _sim(commit_a, last_b)
        elif ordering == "concurrent":
            # Heuristic commit-ordering only (no hints applied here)
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))
            commits_b = set(_commits_beats_for_dilemma(all_beats_b, dilemma_b, beat_nodes))
            if commits_a and commits_b:
                if dilemma_a < dilemma_b:
                    prereq_commits, dependent_commits = commits_a, commits_b
                else:
                    prereq_commits, dependent_commits = commits_b, commits_a
                for prereq in sorted(prereq_commits):
                    for dependent in sorted(dependent_commits):
                        _sim(dependent, prereq)
    return existing, succ


def build_hint_conflict_graph(graph: Graph) -> HintConflictResult:
    """Build a conflict graph for temporal hints and compute the minimum drop set.

    Unlike ``detect_temporal_hint_conflicts`` (single-pass, cascade-blind),
    this function:

    1. Builds a *base DAG* by simulating all non-hint edges (serial, wraps,
       heuristic commit-ordering) without any hints.
    2. Tests each hint alone against the base DAG — hints that cycle alone
       are mandatory drops.
    3. Runs a greedy minimum-drop-set (MDS) loop using
       ``_simulate_hints_sequential`` to detect transitive multi-hint cycles
       that pairwise scanning misses:
       - Simulate all survivors sequentially; hints rejected by the simulation
         form the initial ``conflict_set``.
       - If non-empty: score each hint in ``conflict_set`` via ``_drop_score``,
         pick the weakest, re-simulate survivors minus that hint.  If the new
         conflict_set is empty → mandatory drop, done.  If it has exactly two
         hints that are mutually exclusive (binary irreducible) → swap pair.
         If it is smaller → mandatory drop, iterate.  Otherwise → all are
         mandatory drops (warn and stop).
    4. Applies a greedy heuristic to choose ``default_drop`` for each swap pair:
       prefer dropping introduce-hints over commit-hints, and branch beats over
       spine/canonical beats.
    5. Returns a ``HintConflictResult`` with mandatory_drops, swap_pairs, and
       minimum_drop_set.

    Args:
        graph: The story graph with beat nodes carrying temporal_hint values.

    Returns:
        HintConflictResult describing all conflicts.  ``conflicts`` is empty
        if all hints are consistent with the base DAG.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return HintConflictResult(
            conflicts=[], mandatory_drops=set(), swap_pairs=[], minimum_drop_set=set()
        )

    ctx = _build_hint_resolution_context(graph)
    if ctx is None:
        return HintConflictResult(
            conflicts=[], mandatory_drops=set(), swap_pairs=[], minimum_drop_set=set()
        )

    dilemma_paths = ctx.dilemma_paths
    path_beats_map = ctx.path_beats_map
    beat_id_to_dilemmas = ctx.beat_id_to_dilemmas
    beat_intersection_groups = ctx.beat_intersection_groups
    relationship_edges = ctx.relationship_edges

    # Determine canonical paths for heuristic scoring
    path_nodes = graph.get_nodes_by_type("path")
    canonical_paths: set[str] = {
        pid for pid, data in path_nodes.items() if data.get("is_canonical", False)
    }

    beat_set = set(beat_nodes.keys())

    # Build the full base DAG (all non-hint edges, no hints applied).
    # This is the same base DAG used by verify_hints_acyclic, ensuring
    # detection and postcondition share identical DAG construction.
    base_existing, base_succ = _build_hint_base_dag(
        graph,
        beat_nodes,
        beat_set,
        beat_intersection_groups,
        relationship_edges,
        dilemma_paths,
        path_beats_map,
    )

    # Collect all candidate hint edges across all concurrent pairs
    all_hint_edges: list[_HintEdge] = []
    for dilemma_a, dilemma_b, ordering in relationship_edges:
        if ordering != "concurrent":
            continue
        paths_a = dilemma_paths.get(dilemma_a, [])
        paths_b = dilemma_paths.get(dilemma_b, [])
        if not paths_a or not paths_b:
            continue
        ordered_a = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_a]
        ordered_b = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_b]
        all_beats_a = [b for seq in ordered_a for b in seq]
        all_beats_b = [b for seq in ordered_b for b in seq]
        if not all_beats_a or not all_beats_b:
            continue
        for hint_edge in _iter_temporal_hint_edges(
            all_beats_a + all_beats_b,
            beat_nodes,
            dilemma_a,
            dilemma_b,
            all_beats_a,
            ordered_a,
            all_beats_b,
            ordered_b,
            beat_id_to_dilemmas,
        ):
            all_hint_edges.append(hint_edge)

    if not all_hint_edges:
        return HintConflictResult(
            conflicts=[], mandatory_drops=set(), swap_pairs=[], minimum_drop_set=set()
        )

    # Deduplicate hint edges by beat_id — keep first occurrence per beat
    seen_beat_ids: set[str] = set()
    unique_hints: list[_HintEdge] = []
    for he in all_hint_edges:
        if he.beat_id not in seen_beat_ids:
            seen_beat_ids.add(he.beat_id)
            unique_hints.append(he)

    # base_existing and base_succ are already built above via _build_hint_base_dag()

    def _cycles_alone(hint: _HintEdge) -> bool:
        """Test whether a single hint cycles against the base DAG."""
        from_b, to_b = hint.from_beat, hint.to_beat
        if from_b == to_b or from_b not in beat_set or to_b not in beat_set:
            return False
        if (from_b, to_b) in base_existing:
            return False
        if beat_intersection_groups.get(from_b, set()).intersection(
            beat_intersection_groups.get(to_b, set())
        ):
            return False
        return _would_create_cycle(from_b, to_b, base_succ, beat_set)

    # Phase 1: identify mandatory solo drops
    mandatory_drop_ids: set[str] = set()
    survivors: list[_HintEdge] = []
    for hint in unique_hints:
        if _cycles_alone(hint):
            mandatory_drop_ids.add(hint.beat_id)
        else:
            survivors.append(hint)

    # Phase 2: greedy MDS loop — detect transitive multi-hint cycles.
    #
    # The old pairwise scan missed cycles that only appear when three or more
    # hints interact transitively (A→B, B→C, C→A: no two alone conflict, but
    # all three together cycle).  The sequential simulation in
    # _simulate_hints_sequential uses the same base DAG as _build_base_dag(),
    # so detection and postcondition check (verify_hints_acyclic) are consistent.
    hint_by_beat: dict[str, _HintEdge] = {h.beat_id: h for h in unique_hints}

    # Scoring heuristic — prefer dropping: introduce over commit; branch over canonical
    def _drop_score(beat_id: str, hint: _HintEdge) -> tuple[int, int, str]:
        """Lower score = preferred to drop (higher priority for dropping)."""
        strength = _hint_strength(hint.position)  # 2=commit, 1=introduce; lower = prefer drop
        is_canonical = _is_canonical_beat(beat_id, path_beats_map, canonical_paths)
        canonical_score = 1 if is_canonical else 0  # 0=branch (prefer drop), 1=canonical (keep)
        return (strength, canonical_score, beat_id)

    def _choose_default_drop(beat_a_id: str, beat_b_id: str) -> str:
        ha = hint_by_beat.get(beat_a_id)
        hb = hint_by_beat.get(beat_b_id)
        if ha is None or hb is None:
            raise ValueError(
                f"Could not find hint for beat in swap pair: "
                f"({beat_a_id!r}, {beat_b_id!r}). This is a bug in conflict detection."
            )
        score_a = _drop_score(beat_a_id, ha)
        score_b = _drop_score(beat_b_id, hb)
        # Lower score → prefer to drop
        return beat_a_id if score_a <= score_b else beat_b_id

    # Run the greedy MDS loop.  conflict_set starts as the hints rejected by
    # the sequential simulation of all survivors (those not already in
    # mandatory_drop_ids from Phase 1).
    swap_pairs_result: list[tuple[str, str]] = []

    def _sim_survivors(excluded_beat_ids: set[str]) -> list[_HintEdge]:
        """Simulate survivors minus the excluded beats; return rejected hints."""
        active = [h for h in survivors if h.beat_id not in excluded_beat_ids]
        return _simulate_hints_sequential(
            active, base_existing, base_succ, beat_set, beat_intersection_groups
        )

    greedy_excluded: set[str] = set()
    conflict_set = _sim_survivors(greedy_excluded)
    # Track accepted hint IDs from the last simulation (those not rejected).
    # Used to find swap-pair partners for rejected hints.
    conflict_beat_ids: set[str] = {h.beat_id for h in conflict_set}
    accepted_ids: set[str] = {
        h.beat_id
        for h in survivors
        if h.beat_id not in greedy_excluded and h.beat_id not in conflict_beat_ids
    }

    while conflict_set:
        # Score each conflicting hint; pick the weakest (lowest score = preferred drop)
        best_hint = min(conflict_set, key=lambda h: _drop_score(h.beat_id, hint_by_beat[h.beat_id]))
        candidate_id = best_hint.beat_id

        # Re-simulate without the candidate to see if the conflict set clears
        new_conflict_set = _sim_survivors(greedy_excluded | {candidate_id})

        if not new_conflict_set:
            # Dropping candidate resolves all conflicts.
            # Check if any accepted hint is mutually exclusive with the candidate:
            # if dropping the accepted hint instead also resolves all conflicts,
            # this is a swap pair (either can be dropped, not just the candidate).
            swap_partner: str | None = None
            for acc_id in sorted(accepted_ids):
                alt_conflict = _sim_survivors(greedy_excluded | {acc_id})
                if not alt_conflict:
                    swap_partner = acc_id
                    break
            if swap_partner is not None:
                swap_pairs_result.append((candidate_id, swap_partner))
                # Exclude the default_drop from the swap pair as resolved
                default = _choose_default_drop(candidate_id, swap_partner)
                greedy_excluded.add(default)
                conflict_set = []
                accepted_ids = set()
            else:
                # No alternative resolution → candidate is a true mandatory drop
                mandatory_drop_ids.add(candidate_id)
                greedy_excluded.add(candidate_id)
                conflict_set = new_conflict_set
                accepted_ids = set()
        elif len(new_conflict_set) == 2:
            # Binary residual: check if they are mutually exclusive (swap pair).
            h_a, h_b = new_conflict_set[0], new_conflict_set[1]
            # We only test dropping h_a (not h_b) because both hints passed
            # Phase 1 (solo-cycle check), meaning they only conflict as a pair.
            # By symmetry: if dropping h_a resolves the set, dropping h_b would
            # too — both are mutual excluders, so one test suffices.
            after_drop_a = _sim_survivors(greedy_excluded | {candidate_id, h_a.beat_id})
            if not after_drop_a:
                # Dropping either h_a or h_b resolves → true swap pair
                swap_pairs_result.append((h_a.beat_id, h_b.beat_id))
                mandatory_drop_ids.add(candidate_id)
                greedy_excluded.add(candidate_id)
                conflict_set = []
                accepted_ids = set()
            else:
                mandatory_drop_ids.add(candidate_id)
                greedy_excluded.add(candidate_id)
                conflict_set = new_conflict_set
                # Recompute accepted_ids so the next iteration can find swap partners
                # among the surviving non-conflicting hints (mirrors the elif len<len branch).
                new_conflict_ids = {h.beat_id for h in new_conflict_set}
                accepted_ids = {
                    h.beat_id
                    for h in survivors
                    if h.beat_id not in greedy_excluded and h.beat_id not in new_conflict_ids
                }
        elif len(new_conflict_set) < len(conflict_set):
            # Progress made but not fully resolved → mandatory drop, iterate
            mandatory_drop_ids.add(candidate_id)
            greedy_excluded.add(candidate_id)
            new_conflict_ids = {h.beat_id for h in new_conflict_set}
            accepted_ids = {
                h.beat_id
                for h in survivors
                if h.beat_id not in greedy_excluded and h.beat_id not in new_conflict_ids
            }
            conflict_set = new_conflict_set
        else:
            # Dropping the candidate made no progress; treat all as mandatory drops
            log.warning(
                "hint_conflict_greedy_irreducible",
                conflict_beat_ids=[h.beat_id for h in conflict_set],
                message=(
                    "Greedy MDS found an irreducible conflict set larger than 2; "
                    "marking all as mandatory drops."
                ),
            )
            for h in conflict_set:
                mandatory_drop_ids.add(h.beat_id)
            conflict_set = []
            accepted_ids = set()

    # Build final conflict list
    conflicts: list[HintConflict] = []
    mds: set[str] = set(mandatory_drop_ids)

    for bid in sorted(mandatory_drop_ids):
        conflicts.append(HintConflict(beat_a=bid, beat_b=None, mandatory=True, default_drop=bid))

    for beat_a_id, beat_b_id in swap_pairs_result:
        default = _choose_default_drop(beat_a_id, beat_b_id)
        conflicts.append(
            HintConflict(
                beat_a=beat_a_id,
                beat_b=beat_b_id,
                mandatory=False,
                default_drop=default,
            )
        )
        mds.add(default)

    return HintConflictResult(
        conflicts=conflicts,
        mandatory_drops=mandatory_drop_ids,
        swap_pairs=swap_pairs_result,
        minimum_drop_set=mds,
    )


def verify_hints_acyclic(
    graph: Graph,
    surviving_beat_ids: set[str],
) -> list[str]:
    """Re-run hint simulation with only surviving beats and return still-cyclic beat IDs.

    This is the postcondition check called after LLM resolution.  It re-runs
    the full interleave simulation (base DAG + surviving hints) and returns the
    beat IDs whose hints would still create a cycle.  An empty return value
    means all surviving hints are consistent.

    Args:
        graph: The story graph.  Dropped beats may already have
            ``temporal_hint=None`` (already stripped) or may still carry their
            hint value; either way only ``surviving_beat_ids`` are simulated.
        surviving_beat_ids: Beat IDs whose hints have NOT been dropped.

    Returns:
        List of beat IDs (from surviving_beat_ids) that still cycle.
        Empty list means the postcondition is satisfied.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    ctx = _build_hint_resolution_context(graph)
    if ctx is None:
        return []

    dilemma_paths = ctx.dilemma_paths
    path_beats_map = ctx.path_beats_map
    beat_id_to_dilemmas = ctx.beat_id_to_dilemmas
    beat_intersection_groups = ctx.beat_intersection_groups
    relationship_edges = ctx.relationship_edges

    beat_set = set(beat_nodes.keys())

    # Build the full base DAG (all non-hint edges) using the same function as
    # build_hint_conflict_graph.  This ensures detection and postcondition share
    # an identical DAG construction strategy.
    base_existing, base_succ = _build_hint_base_dag(
        graph,
        beat_nodes,
        beat_set,
        beat_intersection_groups,
        relationship_edges,
        dilemma_paths,
        path_beats_map,
    )

    # Collect all surviving hint edges across all concurrent pairs
    surviving_hint_edges: list[_HintEdge] = []
    seen_beat_ids: set[str] = set()
    for dilemma_a, dilemma_b, ordering in relationship_edges:
        if ordering != "concurrent":
            continue
        paths_a = dilemma_paths.get(dilemma_a, [])
        paths_b = dilemma_paths.get(dilemma_b, [])
        if not paths_a or not paths_b:
            continue
        ordered_a = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_a]
        ordered_b = [_get_path_beats_ordered(graph, p, path_beats_map) for p in paths_b]
        all_beats_a = [b for seq in ordered_a for b in seq]
        all_beats_b = [b for seq in ordered_b for b in seq]
        if not all_beats_a or not all_beats_b:
            continue
        for hint_edge in _iter_temporal_hint_edges(
            all_beats_a + all_beats_b,
            beat_nodes,
            dilemma_a,
            dilemma_b,
            all_beats_a,
            ordered_a,
            all_beats_b,
            ordered_b,
            beat_id_to_dilemmas,
        ):
            if hint_edge.beat_id not in surviving_beat_ids:
                continue
            # Deduplicate by beat_id (keep first occurrence, matching build_hint_conflict_graph)
            if hint_edge.beat_id not in seen_beat_ids:
                seen_beat_ids.add(hint_edge.beat_id)
                surviving_hint_edges.append(hint_edge)

    # Run the same sequential simulation as build_hint_conflict_graph uses.
    # Rejected hints are still-cyclic.
    rejected = _simulate_hints_sequential(
        surviving_hint_edges, base_existing, base_succ, beat_set, beat_intersection_groups
    )
    return [h.beat_id for h in rejected]


def strip_temporal_hints_by_id(graph: Graph, beat_ids: set[str]) -> int:
    """Set temporal_hint to None for the specified beat IDs.

    Used by ``resolve_temporal_hints`` phase to remove conflicting hints
    before ``interleave_beats`` processes the surviving set.

    Args:
        graph: Graph to mutate.
        beat_ids: Set of beat IDs whose temporal hints should be cleared.

    Returns:
        Count of beats that had a hint stripped.
    """
    stripped = 0
    for bid in sorted(beat_ids):
        node = graph.get_node(bid)
        if node is not None and node.get("temporal_hint") is not None:
            graph.update_node(bid, temporal_hint=None)
            stripped += 1
    return stripped


def interleave_cross_path_beats(graph: Graph) -> int:
    """Create predecessor edges between beats from different dilemma paths.

    Reads dilemma relationship edges (concurrent/wraps/serial) and applies
    cross-path ordering rules:

    - ``serial`` (A before B): Last beats of every A path must precede first
      beats of every B path.
    - ``wraps`` (A wraps B): A's intro beats precede B's intro beats; B's
      last beats precede A's commit beats.
    - ``concurrent``: Temporal hints on beats drive specific orderings; without
      hints, commit beats of one dilemma are ordered before commit beats of the
      other to ensure pacing.

    Hint-induced edges that would create a cycle raise RuntimeError (resolve_temporal_hints
    must prevent this).  Heuristic-induced cycles are soft-skipped (benign tiebreak conflicts).

    Args:
        graph: Graph containing beat, path, and dilemma nodes with relationship edges.

    Returns:
        Count of new ``predecessor`` edges created.
    """
    # --- Build indexes ---
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return 0

    dilemma_paths = build_dilemma_paths(graph)
    if len(dilemma_paths) < 2:
        _strip_temporal_hints(graph, beat_nodes)
        return 0

    # path_id → ordered list of beat IDs
    path_beats_map: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to"):
        path_beats_map[edge["to"]].append(edge["from"])

    # beat_id → set of dilemma_ids (for same-dilemma temporal hint guard).
    # A beat may belong to multiple dilemmas (e.g. intersection beats that
    # live on paths from different dilemmas), so we must track the full set.
    beat_id_to_dilemmas: dict[str, set[str]] = defaultdict(set)
    for dil_id, paths in dilemma_paths.items():
        for path_id in paths:
            for bid in path_beats_map.get(path_id, []):
                beat_id_to_dilemmas[bid].add(dil_id)

    beat_set = set(beat_nodes.keys())

    # Build intersection-group index (beat → set of group_ids) to avoid creating
    # predecessor edges between beats that are co-grouped in an intersection.
    # Such edges would create circular prerequisites on shared beats (#1124).
    # A beat can theoretically belong to multiple intersection groups (e.g., grouped
    # by both location and entity), so we track all groups per beat.
    beat_intersection_groups: defaultdict[str, set[str]] = defaultdict(set)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="intersection"):
        beat_intersection_groups[edge["from"]].add(edge["to"])

    # --- Collect dilemma relationship edges ---
    relationship_edges: list[tuple[str, str, str]] = []  # (dilemma_a, dilemma_b, ordering)
    for ordering in ("concurrent", "wraps", "serial"):
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type=ordering):
            a = edge["from"]
            b = edge["to"]
            if a in dilemma_paths and b in dilemma_paths:
                relationship_edges.append((a, b, ordering))

    if not relationship_edges:
        _strip_temporal_hints(graph, beat_nodes)
        return 0

    # Initialise the cycle-detection DAG from the same base as detection/postcondition.
    # _build_hint_base_dag pre-loads ALL non-hint heuristic edges from ALL pairs so
    # that a hint accepted by build_hint_conflict_graph cannot create a cycle here
    # due to a narrower incremental DAG (#1147).
    #
    # ``_base_edges`` contains real graph edges + simulated heuristic edges.
    # ``successors`` is derived from the full base and is used for cycle detection.
    # ``existing_predecessors`` tracks only edges actually written to the graph —
    # initialised from real graph edges so that simulated edges are still written
    # by _add_predecessor (avoiding silent drops of valid edges).
    _, successors = _build_hint_base_dag(
        graph,
        beat_nodes,
        beat_set,
        beat_intersection_groups,
        relationship_edges,
        dilemma_paths,
        path_beats_map,
    )
    # Seed existing_predecessors from real graph edges only (not simulated ones).
    existing_predecessors: set[tuple[str, str]] = {
        (edge["from"], edge["to"])
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type="predecessor")
    }

    created = 0

    def _add_predecessor(from_beat: str, to_beat: str, *, from_hint: bool = False) -> bool:
        """Add predecessor(from_beat, to_beat) if valid and not duplicate.

        Args:
            from_beat: Beat that requires to_beat.
            to_beat: Beat that becomes a prerequisite.
            from_hint: True when the edge is requested by a temporal hint (LLM
                output validated by resolve_temporal_hints).  Hint-induced cycles
                are hard errors — resolve_temporal_hints should have cleared them.
                Heuristic-induced cycles are soft-skipped (the heuristic is an
                arbitrary tiebreak, so a conflict with an existing hint edge is
                expected and benign).

        Returns:
            True if edge was added.
        """
        nonlocal created
        if from_beat == to_beat:
            return False
        if (from_beat, to_beat) in existing_predecessors:
            return False
        if from_beat not in beat_set or to_beat not in beat_set:
            return False
        # Skip edges between beats in the same intersection group —
        # such beats co-occur in a single scene and have no ordering (#1124).
        from_groups = beat_intersection_groups.get(from_beat, set())
        to_groups = beat_intersection_groups.get(to_beat, set())
        shared_groups = from_groups.intersection(to_groups)
        if shared_groups:
            log.debug(
                "interleave_skipped_same_intersection",
                from_beat=from_beat,
                to_beat=to_beat,
                groups=sorted(shared_groups),
            )
            return False
        if _would_create_cycle(from_beat, to_beat, successors, beat_set):
            if from_hint:
                # Hint-induced cycles must have been cleared by resolve_temporal_hints.
                # Reaching here means that phase failed its invariant.
                raise RuntimeError(
                    f"interleave_cross_path_beats: temporal hint on {from_beat!r} "
                    f"requests edge {from_beat!r} → {to_beat!r} which would create "
                    f"a cycle. resolve_temporal_hints should have prevented this."
                )
            # Heuristic-induced cycles are benign: the heuristic is an arbitrary
            # tiebreak and may conflict with existing hint-established ordering.
            log.debug(
                "interleave_heuristic_cycle_skipped",
                from_beat=from_beat,
                to_beat=to_beat,
            )
            return False
        graph.add_edge("predecessor", from_beat, to_beat)
        existing_predecessors.add((from_beat, to_beat))
        successors[to_beat].add(from_beat)
        created += 1
        return True

    # --- Process each relationship ---
    for dilemma_a, dilemma_b, ordering in relationship_edges:
        paths_a = dilemma_paths.get(dilemma_a, [])
        paths_b = dilemma_paths.get(dilemma_b, [])
        if not paths_a or not paths_b:
            continue

        # Ordered beats per path for both dilemmas
        ordered_a: list[list[str]] = [
            _get_path_beats_ordered(graph, p, path_beats_map) for p in paths_a
        ]
        ordered_b: list[list[str]] = [
            _get_path_beats_ordered(graph, p, path_beats_map) for p in paths_b
        ]

        # Collect all beats belonging exclusively to each dilemma's paths
        all_beats_a = [b for seq in ordered_a for b in seq]
        all_beats_b = [b for seq in ordered_b for b in seq]

        if not all_beats_a or not all_beats_b:
            continue

        if ordering == "serial":
            # All A beats before all B beats: last A beats → first B beats
            # "Last" per path is the final element in the ordered sequence
            last_beats_a = {seq[-1] for seq in ordered_a if seq}
            first_beats_b = {seq[0] for seq in ordered_b if seq}
            for last_a in sorted(last_beats_a):
                for first_b in sorted(first_beats_b):
                    _add_predecessor(first_b, last_a)

        elif ordering == "wraps":
            # A wraps B: A's first beats before B's first beats;
            #            B's last beats before A's commit beats
            first_beats_a = {seq[0] for seq in ordered_a if seq}
            first_beats_b = {seq[0] for seq in ordered_b if seq}
            last_beats_b = {seq[-1] for seq in ordered_b if seq}
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))

            # A's first intro before B's first intro
            for first_a in sorted(first_beats_a):
                for first_b in sorted(first_beats_b):
                    _add_predecessor(first_b, first_a)

            # B's last beat before A's commit beats
            for last_b in sorted(last_beats_b):
                for commit_a in sorted(commits_a):
                    _add_predecessor(commit_a, last_b)

        elif ordering == "concurrent":
            # Apply temporal hints first
            hints_applied = 0
            for hint_edge in _iter_temporal_hint_edges(
                all_beats_a + all_beats_b,
                beat_nodes,
                dilemma_a,
                dilemma_b,
                all_beats_a,
                ordered_a,
                all_beats_b,
                ordered_b,
                beat_id_to_dilemmas,
            ):
                if _add_predecessor(hint_edge.from_beat, hint_edge.to_beat, from_hint=True):
                    hints_applied += 1

            if hints_applied:
                log.debug(
                    "interleave_hints_applied",
                    dilemma_a=dilemma_a,
                    dilemma_b=dilemma_b,
                    count=hints_applied,
                )

            # Heuristic fallback for concurrent: commits of A before commits of B
            # (deterministic: use alphabetical dilemma ordering to pick direction)
            commits_a = set(_commits_beats_for_dilemma(all_beats_a, dilemma_a, beat_nodes))
            commits_b = set(_commits_beats_for_dilemma(all_beats_b, dilemma_b, beat_nodes))
            if commits_a and commits_b:
                # Alphabetically earlier dilemma's commits go first as a stable heuristic
                if dilemma_a < dilemma_b:
                    # A commits before B commits
                    for ca in sorted(commits_a):
                        for cb in sorted(commits_b):
                            _add_predecessor(cb, ca)
                else:
                    # B commits before A commits
                    for cb in sorted(commits_b):
                        for ca in sorted(commits_a):
                            _add_predecessor(ca, cb)

    log.info(
        "interleave_cross_path_beats_complete",
        edges_created=created,
        temporal_hints_stripped=_strip_temporal_hints(graph, beat_nodes),
    )
    return created
