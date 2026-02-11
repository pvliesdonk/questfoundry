"""Pure graph algorithms for the GROW stage.

These functions operate on Graph objects and return structured results.
Phase orchestration lives in pipeline/stages/grow.py.

Algorithm summary:
- validate_beat_dag: Kahn's algorithm for cycle detection in requires edges
- validate_commits_beats: Verify each explored dilemma has commits beat per path
- topological_sort_beats: Stable topological sort with alphabetical tie-breaking
- enumerate_arcs: Cartesian product of paths across dilemmas
- compute_divergence_points: Find where arcs diverge from the spine
- select_entities_for_arc: Deterministic entity selection for Phase 4f
"""

from __future__ import annotations

import contextlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.graph.mutations import GrowErrorCategory, GrowValidationError
from questfoundry.models.grow import Arc
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
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

    requires_edges = graph.get_edges(from_id=None, to_id=None, edge_type="requires")
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
            _ensure_edge("requires", keep_id, before_ids[0])
        if after_ids:
            _ensure_edge("requires", after_ids[0], keep_id)

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


# ---------------------------------------------------------------------------
# Phase 9d: Collapse linear passages
# ---------------------------------------------------------------------------


@dataclass
class PassageCollapseResult:
    """Summary of linear-passage collapsing."""

    chains_collapsed: int
    passages_removed: int


def collapse_linear_passages(
    graph: Graph,
    *,
    min_chain_length: int = 3,
    max_chain_length: int = 5,
) -> PassageCollapseResult:
    """Collapse linear passage chains into merged passages.

    A "linear chain" is a sequence of passages where each has exactly one
    outgoing choice leading to the next passage. This creates a merged
    passage with multiple source beats, providing FILL with richer context
    for writing continuous prose.

    Args:
        graph: Story graph with passage and choice nodes.
        min_chain_length: Minimum chain length to collapse (default 3).
        max_chain_length: Maximum beats per merged passage (default 5).

    Returns:
        PassageCollapseResult with counts of collapsed chains and removed passages.
    """
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return PassageCollapseResult(chains_collapsed=0, passages_removed=0)

    # Build passage adjacency (via choice edges)
    adjacency = _build_passage_choice_adjacency(graph)
    outgoing_count = _build_passage_outgoing_count(graph)
    exempt_passages = _build_collapse_exempt_passages(graph, passages)

    # Find all linear chains
    chains = _find_linear_chains(
        passages,
        adjacency,
        outgoing_count,
        exempt_passages,
        min_length=min_chain_length,
        max_length=max_chain_length,
    )

    if not chains:
        return PassageCollapseResult(chains_collapsed=0, passages_removed=0)

    removed_passages: set[str] = set()
    chains_collapsed = 0

    for chain in chains:
        if not _should_collapse_chain(graph, chain):
            continue

        merged_id = _create_merged_passage(graph, chain)
        if not merged_id:
            continue

        _redirect_edges_for_merge(graph, chain, merged_id)

        # Remove original passages (except they're now part of merged)
        for pid in chain:
            removed_passages.add(pid)
            graph.delete_node(pid, cascade=True)

        chains_collapsed += 1

    # Update arc sequences to remove collapsed passages
    if removed_passages:
        _update_arc_sequences(graph, removed_passages)

    return PassageCollapseResult(
        chains_collapsed=chains_collapsed,
        passages_removed=len(removed_passages),
    )


def _build_passage_choice_adjacency(graph: Graph) -> dict[str, list[str]]:
    """Build passage → successor passages mapping via choice edges."""
    adjacency: dict[str, list[str]] = {}
    choice_from_edges = graph.get_edges(edge_type="choice_from")
    choice_to_edges = graph.get_edges(edge_type="choice_to")

    # Build choice → (from_passage, to_passage) mapping
    choice_from: dict[str, str] = {}
    choice_to: dict[str, str] = {}
    for edge in choice_from_edges:
        choice_from[edge["from"]] = edge["to"]
    for edge in choice_to_edges:
        choice_to[edge["from"]] = edge["to"]

    # Build adjacency from choices
    for choice_id, from_passage in choice_from.items():
        to_passage = choice_to.get(choice_id)
        if to_passage:
            adjacency.setdefault(from_passage, []).append(to_passage)

    return adjacency


def _build_passage_outgoing_count(graph: Graph) -> dict[str, int]:
    """Count outgoing choices per passage."""
    outgoing: dict[str, int] = {}
    choice_from_edges = graph.get_edges(edge_type="choice_from")
    for edge in choice_from_edges:
        passage_id = edge["to"]
        outgoing[passage_id] = outgoing.get(passage_id, 0) + 1
    return outgoing


def mark_terminal_passages(graph: Graph) -> int:
    """Derive and persist is_ending on passages with no outgoing choices.

    A passage is terminal if no choice node has choice_from pointing to it.
    Must be called before collapse so that endings are exempt from merging.

    Returns:
        Count of passages marked as ending.
    """
    passages = graph.get_nodes_by_type("passage")
    choices = graph.get_nodes_by_type("choice")

    has_outgoing = {
        choice_data["choice_from"]
        for choice_data in choices.values()
        if choice_data.get("choice_from")
    }

    terminal = set(passages.keys()) - has_outgoing
    for pid in terminal:
        graph.update_node(pid, is_ending=True)
    return len(terminal)


def _build_collapse_exempt_passages(
    graph: Graph, passages: dict[str, dict[str, object]]
) -> set[str]:
    """Build set of passages exempt from collapse.

    Exempt passages:
    - Climax/resolution beats (narrative_function in {"confront", "resolve"})
    - Ending passages (is_ending=True)
    - Passages with transition_style="cut" in their beat
    """
    beats = graph.get_nodes_by_type("beat")
    exempt: set[str] = set()

    for pid, pdata in passages.items():
        # Check ending status
        if pdata.get("is_ending"):
            exempt.add(pid)
            continue

        # Check beat properties
        beat_id = pdata.get("from_beat")
        if not beat_id:
            continue
        beat = beats.get(str(beat_id), {})

        # Exempt confront/resolve beats
        if beat.get("narrative_function") in {"confront", "resolve"}:
            exempt.add(pid)

        # Exempt passages where beat has hard cut transition
        if beat.get("transition_style") == "cut":
            exempt.add(pid)

    return exempt


def _find_linear_chains(
    passages: dict[str, dict[str, object]],
    adjacency: dict[str, list[str]],
    outgoing_count: dict[str, int],
    exempt: set[str],
    min_length: int,
    max_length: int,
) -> list[list[str]]:
    """Find linear passage chains suitable for collapsing.

    A chain is a sequence where each passage has exactly one outgoing choice.
    """
    chains: list[list[str]] = []
    visited: set[str] = set()

    def _is_linear(pid: str) -> bool:
        return outgoing_count.get(pid, 0) == 1 and pid not in exempt

    # Start from passages that are linear but have non-linear predecessors
    # (chain starts) or no predecessors
    incoming: dict[str, list[str]] = {}
    for source, targets in adjacency.items():
        for target in targets:
            incoming.setdefault(target, []).append(source)

    for pid in passages:
        if pid in visited:
            continue
        if not _is_linear(pid):
            continue

        # Check if this is a chain start (no incoming or incoming is non-linear)
        preds = incoming.get(pid, [])
        is_start = not preds or all(not _is_linear(p) for p in preds)
        if not is_start:
            continue

        # Walk the chain
        chain: list[str] = [pid]
        visited.add(pid)
        current = pid

        while len(chain) < max_length:
            successors = adjacency.get(current, [])
            if len(successors) != 1:
                break
            next_p = successors[0]
            if next_p in visited or not _is_linear(next_p):
                break
            chain.append(next_p)
            visited.add(next_p)
            current = next_p

        if len(chain) >= min_length:
            chains.append(chain)

    return chains


def _should_collapse_chain(graph: Graph, chain: list[str]) -> bool:
    """Determine if a chain should be collapsed based on continuity.

    Criteria:
    - Scene continuity: Same or compatible locations
    - No hard cuts in gap beats within the chain
    """
    if len(chain) < 2:
        return False

    passages = {pid: graph.get_node(pid) or {} for pid in chain}
    beats = {}
    for pid, pdata in passages.items():
        beat_id = pdata.get("from_beat")
        if beat_id:
            beats[pid] = graph.get_node(str(beat_id)) or {}

    # Check for hard cuts
    for pid in chain[1:]:  # Skip first
        beat = beats.get(pid, {})
        if beat.get("transition_style") == "cut":
            return False

    # Check location continuity (allow if all same or if locations are None)
    locations = set()
    for pid in chain:
        beat = beats.get(pid, {})
        loc = beat.get("location")
        if loc:
            locations.add(loc)

    # More than one distinct location = don't collapse
    return len(locations) <= 1


def _create_merged_passage(graph: Graph, chain: list[str]) -> str | None:
    """Create a merged passage from a chain of passages.

    Returns the new passage ID, or None if creation fails.
    """
    if not chain:
        return None

    passages = [graph.get_node(pid) or {} for pid in chain]
    beat_ids = [str(p.get("from_beat", "")) for p in passages if p.get("from_beat")]

    if not beat_ids:
        return None

    # Primary beat is first non-gap beat, or just first beat
    beats = [graph.get_node(bid) or {} for bid in beat_ids]
    primary_idx = next(
        (i for i, b in enumerate(beats) if not b.get("is_gap_beat")),
        0,
    )
    primary_beat_id = beat_ids[primary_idx]
    primary_beat = beats[primary_idx]

    # Build transition points for gap beats
    transition_points: list[dict[str, object]] = []
    for i, (_bid, beat) in enumerate(zip(beat_ids[1:], beats[1:], strict=True), 1):
        if beat.get("is_gap_beat") or beat.get("bridges_from"):
            transition_points.append(
                {
                    "index": i,
                    "style": beat.get("transition_style", "smooth"),
                    "bridge_entities": beat.get("entities", []),
                    "note": beat.get("summary", ""),
                }
            )

    # Collect all entities from all beats
    all_entities: list[str] = []
    for beat in beats:
        all_entities.extend(beat.get("entities") or [])
    all_entities = list(dict.fromkeys(all_entities))  # Deduplicate

    # Generate merged passage ID
    primary_raw = primary_beat.get("raw_id", "merged")
    merged_raw = f"merged_{primary_raw}"
    merged_id = f"passage::{merged_raw}"

    # Ensure unique ID
    counter = 1
    while graph.get_node(merged_id):
        merged_id = f"passage::{merged_raw}_{counter}"
        counter += 1

    # Create merged passage node
    graph.create_node(
        merged_id,
        {
            "type": "passage",
            "raw_id": merged_id.removeprefix("passage::"),
            "from_beats": beat_ids,
            "primary_beat": primary_beat_id,
            "merged_from": chain,
            "transition_points": transition_points,
            "summary": primary_beat.get("summary", ""),
            "entities": all_entities,
            "prose": None,
        },
    )

    # Add passage_from edges for all source beats
    for bid in beat_ids:
        graph.add_edge("passage_from", merged_id, bid)

    log.info(
        "passage_collapsed",
        merged_id=merged_id,
        source_passages=chain,
        beats=beat_ids,
    )

    return merged_id


def _redirect_edges_for_merge(graph: Graph, chain: list[str], merged_id: str) -> None:
    """Redirect choice edges to point to/from the merged passage."""
    first_passage = chain[0]
    last_passage = chain[-1]

    # Find incoming choices (pointing to first passage)
    choice_to_edges = graph.get_edges(edge_type="choice_to", to_id=first_passage)
    for edge in choice_to_edges:
        choice_id = edge["from"]
        # Check if this choice is internal to the chain
        choice_from = graph.get_edges(edge_type="choice_from", from_id=choice_id)
        if choice_from:
            source = choice_from[0]["to"]
            if source in chain:
                # Internal choice - will be deleted
                continue
        # Redirect to merged passage
        graph.remove_edge("choice_to", edge["from"], edge["to"])
        graph.add_edge("choice_to", choice_id, merged_id)
        # Also update the choice node's to_passage attribute
        graph.update_node(choice_id, to_passage=merged_id)

    # Find outgoing choices (from last passage)
    choice_from_edges = graph.get_edges(edge_type="choice_from", to_id=last_passage)
    for edge in choice_from_edges:
        choice_id = edge["from"]
        # Check if this choice is internal to the chain
        choice_to = graph.get_edges(edge_type="choice_to", from_id=choice_id)
        if choice_to:
            target = choice_to[0]["to"]
            if target in chain:
                # Internal choice - will be deleted
                continue
        # Redirect from merged passage
        graph.remove_edge("choice_from", edge["from"], edge["to"])
        graph.add_edge("choice_from", choice_id, merged_id)
        # Also update the choice node's from_passage attribute
        graph.update_node(choice_id, from_passage=merged_id)

    # Delete internal choices (between chain passages)
    for i, pid in enumerate(chain[:-1]):
        next_pid = chain[i + 1]
        # Find choice connecting pid → next_pid
        choice_from_edges = graph.get_edges(edge_type="choice_from", to_id=pid)
        for edge in choice_from_edges:
            choice_id = edge["from"]
            choice_to = graph.get_edges(edge_type="choice_to", from_id=choice_id)
            if choice_to and choice_to[0]["to"] == next_pid:
                graph.delete_node(choice_id, cascade=True)


def _update_arc_sequences(graph: Graph, removed_passages: set[str]) -> None:  # noqa: ARG001
    """Update arc sequences after passage collapse.

    Note: Arc sequences contain beat IDs, not passage IDs.
    Collapsed passages don't affect arc sequences since beats remain intact.
    This function is a placeholder for future extensions.
    """
    # Arc sequences contain beat IDs, not passage IDs - no updates needed
    _ = graph  # Silence unused warning


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

    Args:
        graph: Graph containing dilemma, path, and beat nodes.
        max_arc_count: Safety ceiling for arc count. Defaults to
            ``_MAX_ARC_COUNT`` (64) if not provided.

    Returns:
        List of Arc models, spine first, then branches sorted by ID.

    Raises:
        ValueError: If arc count exceeds the limit.
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

    # Cartesian product of paths
    arcs: list[Arc] = []
    for combo in product(*path_lists):
        path_combo = list(combo)
        # Get raw_ids for arc naming (sorted alphabetically)
        path_raw_ids = sorted(path_nodes[pid].get("raw_id", pid) for pid in path_combo)
        arc_id = "+".join(path_raw_ids)

        # Collect beats: beats that belong to ANY path in the combo
        beat_set: set[str] = set()
        for pid in path_combo:
            beat_set.update(path_beat_sets.get(pid, set()))

        # Topological sort of the beats
        try:
            sequence = topological_sort_beats(graph, list(beat_set))
        except ValueError:
            sequence = sorted(beat_set)  # Fallback for cycles (Phase 1 should catch)

        # Determine if spine (all canonical)
        is_spine = all(path_nodes[pid].get("is_canonical", False) for pid in path_combo)

        arcs.append(
            Arc(
                arc_id=arc_id,
                arc_type="spine" if is_spine else "branch",
                paths=path_raw_ids,
                sequence=sequence,
            )
        )

    # Check combinatorial limit
    limit = max_arc_count if max_arc_count is not None else _MAX_ARC_COUNT
    if len(arcs) > limit:
        # This will be caught by the phase and raised as GrowMutationError
        raise ValueError(
            f"Arc count ({len(arcs)}) exceeds limit of {limit}. "
            f"Reduce the number of dilemmas or paths."
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
        converges_at: The beat where convergence occurs (None if no convergence).
            flavor: first shared beat after divergence.
            soft: first shared beat after last exclusive beat (if payoff_budget met).
            hard: always None.
        convergence_policy: Effective policy applied to this arc.
        payoff_budget: Effective payoff budget applied to this arc.
    """

    arc_id: str
    converges_to: str
    converges_at: str | None = None
    convergence_policy: str = "soft"
    payoff_budget: int = 2


def _find_arc_dilemma_policies(
    graph: Graph,
    arc: Arc,
) -> list[tuple[str, int]]:
    """Collect (convergence_policy, payoff_budget) for each dilemma an arc touches.

    Traverses arc.paths → path node → dilemma node to read policy metadata
    stored by SEED's post-prune analysis.
    """
    policies: list[tuple[str, int]] = []
    for raw_path_id in arc.paths:
        path_node_id = normalize_scoped_id(raw_path_id, "path")
        path_node = graph.get_node(path_node_id)
        if not path_node or not (dilemma_id := path_node.get("dilemma_id")):
            continue
        dilemma_node = graph.get_node(normalize_scoped_id(dilemma_id, "dilemma"))
        if dilemma_node:
            policies.append(
                (
                    dilemma_node.get("convergence_policy", "soft"),
                    dilemma_node.get("payoff_budget", 2),
                )
            )
    return policies


def _get_effective_policy(graph: Graph, arc: Arc) -> tuple[str, int]:
    """Combine convergence policies for a (possibly multi-dilemma) arc.

    Combine rule per issue #743: hard dominates; payoff_budget = max across
    all dilemmas the arc diverges on.  Falls back to ("flavor", 0) when no
    dilemma metadata is found (preserves pre-policy behavior).
    """
    policies = _find_arc_dilemma_policies(graph, arc)
    if not policies:
        # No SEED convergence metadata — preserve pre-policy behavior
        # (first shared beat, no budget constraint).
        return ("flavor", 0)

    max_budget = max(b for _, b in policies)
    if any(p == "hard" for p, _ in policies):
        return ("hard", max_budget)
    if any(p == "soft" for p, _ in policies):
        return ("soft", max_budget)
    return ("flavor", max_budget)


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
        graph: Graph with dilemma nodes containing convergence_policy/payoff_budget.
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
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    beat_dilemma_map = _build_beat_dilemma_map_for_convergence(graph)

    for arc in arcs:
        if arc.arc_type == "spine":
            continue

        div_info = divergence_map.get(arc.arc_id)
        if not div_info:
            continue

        # Get the full arc-level effective policy (for storage on the node)
        eff_policy, eff_budget = _get_effective_policy(graph, arc)

        # Find beats in branch after divergence point
        diverge_at = div_info.diverges_at
        if diverge_at and diverge_at in arc.sequence:
            div_idx = arc.sequence.index(diverge_at)
            branch_after_div = arc.sequence[div_idx + 1 :]
        else:
            branch_after_div = arc.sequence

        # Separate hard vs non-hard dilemma policies
        policies = _find_arc_dilemma_policies(graph, arc)
        non_hard = [(p, b) for p, b in policies if p != "hard"]

        converges_at: str | None = None
        if not policies:
            # No dilemma metadata → use arc-level effective policy (backward compat)
            if eff_policy == "flavor":
                for beat_id in branch_after_div:
                    if beat_id in spine_seq_set:
                        converges_at = beat_id
                        break
            elif eff_policy == "soft":
                converges_at = _find_convergence_for_soft(
                    branch_after_div, spine_seq_set, eff_budget
                )
            # else: hard with no metadata shouldn't happen, but safe
        elif not non_hard:
            # All policies are hard → no convergence
            pass
        else:
            # Compute effective non-hard policy
            non_hard_policy = "soft" if any(p == "soft" for p, _ in non_hard) else "flavor"
            non_hard_budget = max(b for _, b in non_hard)

            # Find hard dilemma IDs to filter beats
            hard_dilemma_ids: set[str] = set()
            for raw_path_id in arc.paths:
                path_node_id = normalize_scoped_id(raw_path_id, "path")
                path_node = graph.get_node(path_node_id)
                if not path_node:
                    continue
                did = path_node.get("dilemma_id")
                if not did:
                    continue
                prefixed = normalize_scoped_id(did, "dilemma")
                dnode = dilemma_nodes.get(prefixed)
                if dnode and dnode.get("convergence_policy") == "hard":
                    hard_dilemma_ids.add(prefixed)

            # Filter branch_after_div to non-hard dilemma beats only.
            # A beat is kept if it has no dilemma association (neutral) or
            # has ANY non-hard dilemma association.  Beats exclusively
            # owned by hard dilemmas are removed.
            if hard_dilemma_ids:
                hard_ids = hard_dilemma_ids  # bind for closure
                filtered_branch = [
                    b
                    for b in branch_after_div
                    if not (beat_dilemma_map.get(b) and beat_dilemma_map[b] <= hard_ids)
                ]
                filtered_spine = {
                    b
                    for b in spine_seq_set
                    if not (beat_dilemma_map.get(b) and beat_dilemma_map[b] <= hard_ids)
                }
            else:
                filtered_branch = branch_after_div
                filtered_spine = spine_seq_set

            # Apply policy-specific convergence logic on filtered beats
            if non_hard_policy == "flavor":
                for beat_id in filtered_branch:
                    if beat_id in filtered_spine:
                        converges_at = beat_id
                        break
            else:
                converges_at = _find_convergence_for_soft(
                    filtered_branch, filtered_spine, non_hard_budget
                )

        result[arc.arc_id] = ConvergenceInfo(
            arc_id=arc.arc_id,
            converges_to=spine.arc_id,
            converges_at=converges_at,
            convergence_policy=eff_policy,
            payoff_budget=eff_budget,
        )

    return result


# ---------------------------------------------------------------------------
# Phase 9 helpers: passage-arc membership and choice requires
# ---------------------------------------------------------------------------


def compute_passage_arc_membership(graph: Graph) -> dict[str, set[str]]:
    """Map each passage to the set of (prefixed) arc IDs whose sequences include it.

    Converts beat sequences to passage IDs via passage ``from_beat`` fields.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    passage_nodes = graph.get_nodes_by_type("passage")

    # Build beat → passage mapping
    beat_to_passage: dict[str, str] = {}
    for p_id, p_data in passage_nodes.items():
        from_beat = p_data.get("from_beat", "")
        if from_beat:
            beat_to_passage[from_beat] = p_id

    membership: dict[str, set[str]] = {}
    for arc_id, arc_data in arc_nodes.items():
        for beat_id in arc_data.get("sequence", []):
            passage_id = beat_to_passage.get(beat_id)
            if passage_id:
                membership.setdefault(passage_id, set()).add(arc_id)

    return membership


def compute_all_choice_requires(
    graph: Graph,
    passage_arcs: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Compute codeword ``requires`` for hard-policy branch entry passages.

    For each passage that appears exclusively in hard-policy branch arcs
    (not on the spine), collects codewords from **spine-exclusive** paths —
    paths on the spine that are NOT shared with the branch arc.  These
    codewords are earnable before the branch divergence point, making the
    gate satisfiable in a single playthrough.

    Returns:
        Mapping of ``passage_id`` → list of required codeword IDs.
        Empty dict if no spine arc exists or no hard-policy branches.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return {}

    # 1. Find spine arc and its paths
    spine_paths: set[str] = set()
    for arc_data in arc_nodes.values():
        if arc_data.get("arc_type") == "spine":
            spine_paths = {normalize_scoped_id(p, "path") for p in arc_data.get("paths", [])}
            break
    if not spine_paths:
        return {}

    # 2. Build consequence→codeword lookup (reverse of tracks edges)
    cons_to_codeword = {
        edge["to"]: edge["from"]
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
    }

    # 3. Build path→consequences lookup
    has_cons_edges = graph.get_edges(from_id=None, to_id=None, edge_type="has_consequence")
    path_consequences: dict[str, list[str]] = {}
    for edge in has_cons_edges:
        path_consequences.setdefault(edge["from"], []).append(edge["to"])

    # 4. For each passage exclusive to hard-policy branches, collect
    #    codewords from spine-exclusive paths.
    requires: dict[str, list[str]] = {}
    for passage_id, arc_ids in passage_arcs.items():
        # Skip passages reachable via the spine — no gating needed
        if any(arc_nodes.get(a, {}).get("arc_type") == "spine" for a in arc_ids):
            continue

        codewords: set[str] = set()
        for arc_id in sorted(arc_ids):
            arc_data = arc_nodes.get(arc_id, {})
            if arc_data.get("convergence_policy") != "hard":
                continue

            # Spine-exclusive = spine paths NOT shared with this branch
            branch_paths = {normalize_scoped_id(p, "path") for p in arc_data.get("paths", [])}
            spine_exclusive = spine_paths - branch_paths

            for path_id in sorted(spine_exclusive):
                for cons_id in path_consequences.get(path_id, []):
                    cw = cons_to_codeword.get(cons_id)
                    if cw:
                        codewords.add(cw)

        if codewords:
            requires[passage_id] = sorted(codewords)

    return requires


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

    Groups beats by shared locations/location_alternatives and shared entities.
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

    # Filter out beats from hard-policy dilemmas (topology isolation)
    hard_beats = _get_hard_policy_beats(graph, list(beat_nodes.keys()), beat_dilemmas)
    if hard_beats:
        beat_nodes = {bid: d for bid, d in beat_nodes.items() if bid not in hard_beats}
        beat_dilemmas = {bid: ds for bid, ds in beat_dilemmas.items() if bid not in hard_beats}
        if not beat_nodes:
            return []

    # Group by location overlap (highest priority)
    location_groups = _group_by_location(beat_nodes, beat_dilemmas)

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
    ``convergence_policy == "hard"``.

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
            if dnode and dnode.get("convergence_policy") == "hard":
                hard_beats.add(bid)
                break
    return hard_beats


def _group_by_location(
    beat_nodes: dict[str, Any],
    beat_dilemmas: dict[str, set[str]],
) -> list[IntersectionCandidate]:
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

    candidates: list[IntersectionCandidate] = []
    for location, beats in sorted(location_beats.items()):
        if len(beats) < 2:
            continue
        # Filter to beats from different dilemmas
        multi_dilemma = _filter_different_dilemmas(beats, beat_dilemmas)
        if len(multi_dilemma) >= 2:
            candidates.append(
                IntersectionCandidate(
                    beat_ids=sorted(multi_dilemma),
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
            key = tuple(multi_dilemma)
            if key not in seen_pairs:
                seen_pairs.add(key)
                candidates.append(
                    IntersectionCandidate(
                        beat_ids=multi_dilemma,
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
    for edge in graph.get_edges(from_id=prereq_id, to_id=None, edge_type="requires"):
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
    graph.add_edge("requires", variant_id, prereq_id)

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

    # Reject intersections involving hard-policy dilemmas (topology isolation)
    hard_beats = _get_hard_policy_beats(graph, beat_ids, beat_dilemma_map)
    if hard_beats:
        hard_dilemmas = sorted({d for bid in hard_beats for d in beat_dilemma_map.get(bid, set())})
        errors.append(
            GrowValidationError(
                field_path="intersection.hard_policy",
                issue=(
                    f"Beat(s) {sorted(hard_beats)} belong to hard-policy dilemma(s) "
                    f"{hard_dilemmas}. Hard-policy paths require topology isolation "
                    f"and cannot form intersections."
                ),
                category=GrowErrorCategory.STRUCTURAL,
            )
        )
        # Early return: hard-policy violations are structural and block all
        # downstream checks (requires edges, conditional prerequisites).
        return errors

    # Check requires edges originating from intersection beats.
    # Two checks in one pass: (1) no circular requires between intersection
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

    # Iterate only over outgoing requires edges from intersection beats
    # (targeted lookups instead of scanning all requires edges).
    for from_id in beat_set:
        for edge in graph.get_edges(from_id=from_id, to_id=None, edge_type="requires"):
            to_id = edge["to"]
            if to_id in beat_set:
                # Circular requires between intersection beats
                errors.append(
                    GrowValidationError(
                        field_path="intersection.requires",
                        issue=(
                            f"Beat '{from_id}' requires '{to_id}' — "
                            f"intersection beats cannot have requires "
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
                # If a `requires` target is narrower, that edge would be
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
                                f"Beat '{from_id}' requires '{to_id}' which "
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


def apply_intersection_mark(
    graph: Graph,
    beat_ids: list[str],
    resolved_location: str | None,
) -> None:
    """Mark beats as belonging to an intersection (multi-path scene).

    Updates beat nodes with:
    - intersection_group: list of other beat IDs in the intersection
    - resolved_location: the location chosen for the combined scene

    Also adds additional belongs_to edges so each beat is assigned to
    all paths from all beats in the intersection.

    Args:
        graph: Graph to mutate.
        beat_ids: Beat IDs to group into intersection.
        resolved_location: Resolved location, or None.
    """
    beat_set = set(beat_ids)

    # Collect all path assignments across all intersection beats
    all_path_ids: set[str] = set()
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        if edge["from"] in beat_set:
            all_path_ids.add(edge["to"])

    # Collect new edges to add (batch to avoid stale reads)
    new_edges: list[tuple[str, str]] = []
    for bid in beat_ids:
        current_paths = {e["to"] for e in belongs_to_edges if e["from"] == bid}
        for path_id in all_path_ids - current_paths:
            new_edges.append((bid, path_id))

    # Update each beat's node data
    for bid in beat_ids:
        others = sorted(b for b in beat_ids if b != bid)
        update_data: dict[str, Any] = {"intersection_group": others}
        if resolved_location:
            update_data["location"] = resolved_location
        graph.update_node(bid, **update_data)

    # Apply cross-path edges
    for from_id, to_id in new_edges:
        graph.add_edge("belongs_to", from_id, to_id)


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
        },
    )

    # Add belongs_to edge
    graph.add_edge("belongs_to", beat_id, path_id)

    # Adjust requires edges for ordering.
    # Existing transitive requires (before_beat → after_beat) is kept as redundant
    # but harmless for topological sort correctness.
    if after_beat:
        graph.add_edge("requires", beat_id, after_beat)

    if before_beat:
        graph.add_edge("requires", before_beat, beat_id)

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
            next_p, next_beat_idx = passage_seq[i + 1]

            if p_id not in successors:
                successors[p_id] = []
                seen_targets[p_id] = set()

            # Skip if we already recorded this successor target
            if next_p in seen_targets[p_id]:
                continue
            seen_targets[p_id].add(next_p)

            # Grants: codewords from beats between this passage's beat and the
            # next passage's beat (inclusive). This reflects state changes that
            # happen when the player takes this choice, without leaking future
            # arc codewords.
            arc_grants: list[str] = []
            for beat_id in sequence[beat_idx + 1 : next_beat_idx + 1]:
                arc_grants.extend(beat_grants.get(beat_id, []))

            successors[p_id].append(
                PassageSuccessor(
                    to_passage=next_p,
                    arc_id=arc_id,
                    grants=arc_grants,
                )
            )

    return successors


# ---------------------------------------------------------------------------
# Phase 4f: Entity arc selection
# ---------------------------------------------------------------------------

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
