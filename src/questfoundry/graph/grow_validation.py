"""Graph validation checks for GROW stage.

Beat-DAG validation checks that verify the story's structural integrity
at the beat/path/dilemma level. These are pure, deterministic functions
operating on the graph — no LLM calls.

Passage-layer checks (reachability, gates, routing, prose neutrality)
have moved to ``polish_validation.py`` and run during POLISH Phase 7.

Validation categories retained here:
- Beat-DAG: single start, dilemma resolution, DAG cycles, spine arc, role compliance
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from questfoundry.graph.context import get_primary_beat, normalize_scoped_id
from questfoundry.graph.fill_context import is_merged_passage
from questfoundry.graph.validation_types import ValidationCheck, ValidationReport

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Re-export types for backward compatibility — many modules import from here.
__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "build_exempt_passages",
    "build_outgoing_count",
    "build_passage_adjacency",
    "check_dilemma_role_compliance",
    "check_dilemmas_resolved",
    "check_passage_dag_cycles",
    "check_single_start",
    "check_spine_arc_exists",
    "compute_linear_threshold",
    "find_max_consecutive_linear",
    "find_start_passages",
    "passages_with_forward_incoming",
    "run_all_checks",
    "run_grow_checks",
    "walk_linear_stretches",
]


# ---------------------------------------------------------------------------
# Shared passage-graph utilities (used by GROW phases and POLISH validation)
# ---------------------------------------------------------------------------


def passages_with_forward_incoming(graph: Graph) -> set[str]:
    """Return passage IDs that have at least one non-return incoming choice.

    Excludes ``is_return`` choices (spoke→hub back-links) so that hub
    passages are still recognised as start passages when they have no
    other incoming edges.
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    has_incoming: set[str] = set()
    for choice_data in choice_nodes.values():
        if choice_data.get("is_return"):
            continue
        to_p = choice_data.get("to_passage")
        if to_p:
            has_incoming.add(to_p)
    return has_incoming


def build_passage_adjacency(graph: Graph) -> dict[str, list[str]]:
    """Build passage → successor passages adjacency list from choice nodes.

    Args:
        graph: The story graph.

    Returns:
        Dict mapping each passage ID to a list of successor passage IDs.
    """
    choices = graph.get_nodes_by_type("choice")
    adjacency: dict[str, list[str]] = {}
    for _cid, cdata in choices.items():
        from_p = cdata.get("from_passage", "")
        to_p = cdata.get("to_passage", "")
        if from_p and to_p:
            adjacency.setdefault(from_p, []).append(to_p)
    return adjacency


def build_outgoing_count(graph: Graph) -> dict[str, int]:
    """Count outgoing choices per passage from choice_from edges.

    Args:
        graph: The story graph.

    Returns:
        Dict mapping passage ID to number of outgoing choices.
    """
    # choice_from edges point choice → source_passage, so e["to"] = source passage.
    choice_from_edges = graph.get_edges(edge_type="choice_from")
    outgoing_count: dict[str, int] = {}
    for edge in choice_from_edges:
        source = edge["to"]
        outgoing_count[source] = outgoing_count.get(source, 0) + 1
    return outgoing_count


def build_exempt_passages(graph: Graph, passages: dict[str, dict[str, object]]) -> set[str]:
    """Build set of passages exempt from linearity checks.

    Exempt passages include:
    - Passages with confront/resolve narrative function (climax/resolution)
    - Merged passages (already collapsed from linear stretches)
    """
    beats = graph.get_nodes_by_type("beat")
    exempt_beats: set[str] = set()
    for bid, bdata in beats.items():
        if bdata.get("narrative_function") in {"confront", "resolve"}:
            exempt_beats.add(bid)

    exempt: set[str] = set()
    for pid, _pdata in passages.items():
        # Exempt confront/resolve passages
        beat = get_primary_beat(graph, pid)
        if beat and beat in exempt_beats:
            exempt.add(pid)
        # Exempt merged passages (they ARE the result of collapse)
        if is_merged_passage(graph, pid):
            exempt.add(pid)
    return exempt


def find_start_passages(graph: Graph, passages: dict[str, dict[str, object]]) -> list[str]:
    """Find passages with no forward incoming choice edges.

    Excludes ``is_return`` spoke→hub back-links.
    """
    has_incoming = passages_with_forward_incoming(graph)
    return [pid for pid in passages if pid not in has_incoming]


def walk_linear_stretches(
    starts: list[str],
    adjacency: dict[str, list[str]],
    outgoing_count: dict[str, int],
    exempt_passages: set[str],
    threshold: int,
) -> list[list[str]]:
    """BFS walk to find linear stretches exceeding threshold.

    Tracks per-path run context so convergence points are evaluated
    correctly from each incoming path.

    Args:
        starts: Start passage IDs for BFS.
        adjacency: Passage → successors mapping.
        outgoing_count: Passage → outgoing choice count.
        exempt_passages: Passages exempt from linearity (confront/resolve).
        threshold: Minimum run length to report. 0 = report all runs.

    Returns:
        List of passage ID lists, one per linear stretch exceeding threshold.
    """
    stretches: list[list[str]] = []
    # Track best known run reaching each node to prune redundant BFS paths
    best_run_at: dict[str, int] = {}

    for start in starts:
        queue: deque[tuple[str, list[str]]] = deque()
        is_linear = outgoing_count.get(start, 0) == 1 and start not in exempt_passages
        initial_run = [start] if is_linear else []
        queue.append((start, initial_run))

        while queue:
            current, run = queue.popleft()

            for successor in adjacency.get(current, []):
                is_succ_linear = (
                    outgoing_count.get(successor, 0) == 1 and successor not in exempt_passages
                )
                if is_succ_linear:
                    new_run = [*run, successor]
                    # Only continue if this path offers a longer run than previously seen
                    if len(new_run) <= best_run_at.get(successor, 0):
                        continue
                    best_run_at[successor] = len(new_run)
                    if (threshold > 0 and len(new_run) > threshold) or threshold == 0:
                        stretches.append(new_run)
                    queue.append((successor, new_run))
                else:
                    # Reset run at branching/exempt nodes; only visit if not yet seen
                    if successor not in best_run_at:
                        best_run_at[successor] = 0
                        queue.append((successor, []))

    return stretches


def find_max_consecutive_linear(graph: Graph) -> int:
    """Compute the longest consecutive single-outgoing passage stretch.

    Uses BFS from start passages, tracking per-path run lengths.
    Passages whose beat has ``narrative_function`` in {"confront", "resolve"}
    are exempt (linearity is narratively appropriate at climax/resolution).

    Args:
        graph: The story graph.

    Returns:
        Length of the longest linear stretch (0 if no passages or no linear runs).
    """
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return 0

    outgoing_count = build_outgoing_count(graph)
    adjacency = build_passage_adjacency(graph)
    exempt = build_exempt_passages(graph, passages)
    starts = find_start_passages(graph, passages)

    max_len = 0
    for _violation in walk_linear_stretches(starts, adjacency, outgoing_count, exempt, threshold=0):
        max_len = max(max_len, len(_violation))
    return max_len


def compute_linear_threshold(graph: Graph) -> int:
    """Scale max consecutive linear threshold with passage count.

    Larger stories naturally have longer linear stretches between branch
    points (each dilemma adds ~10 passages). The default of 2 is kept for
    small stories; larger stories get proportionally wider tolerance.
    """
    passage_count = len(graph.get_nodes_by_type("passage"))
    return max(2, passage_count // 20)


# ---------------------------------------------------------------------------
# Beat-DAG helpers (private)
# ---------------------------------------------------------------------------


def _get_spine_sequence(arc_nodes: dict[str, dict[str, object]]) -> set[str]:
    """Extract the spine arc's beat sequence as a set.

    Returns an empty set if no spine arc is found.
    """
    for data in arc_nodes.values():
        if data.get("arc_type") == "spine":
            seq = data.get("sequence", [])
            return set(seq) if isinstance(seq, list) else set()
    return set()


def _build_beat_dilemma_map(graph: Graph) -> dict[str, set[str]]:
    """Map each beat to its prefixed dilemma IDs via belongs_to → path → dilemma."""
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


# ---------------------------------------------------------------------------
# Beat-DAG check functions (stay in GROW)
# ---------------------------------------------------------------------------


def check_single_start(graph: Graph) -> ValidationCheck:
    """Verify exactly one start passage exists (no incoming choice_to edges).

    A start passage is one with no incoming choice_to edges. There must be
    exactly one for a well-formed story graph.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return ValidationCheck(
            name="single_start",
            severity="pass",
            message="No passages to check",
        )

    # Start passages = passages without forward incoming choice edges
    # (excludes is_return spoke→hub back-links)
    has_incoming = passages_with_forward_incoming(graph)
    start_passages = [pid for pid in passage_nodes if pid not in has_incoming]

    if len(start_passages) == 1:
        return ValidationCheck(
            name="single_start",
            severity="pass",
            message=f"Single start passage: {start_passages[0]}",
        )
    if len(start_passages) == 0:
        return ValidationCheck(
            name="single_start",
            severity="fail",
            message="No start passage found (all passages have incoming edges)",
        )
    return ValidationCheck(
        name="single_start",
        severity="fail",
        message=f"Multiple start passages found: {', '.join(sorted(start_passages))}",
    )


def check_dilemmas_resolved(graph: Graph) -> ValidationCheck:
    """Verify each explored dilemma has at least one commits beat per path.

    Re-checks after gap insertion phases (4b/4c) may have altered the graph.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")
    beat_nodes = graph.get_nodes_by_type("beat")

    if not dilemma_nodes or not path_nodes:
        return ValidationCheck(
            name="dilemmas_resolved",
            severity="pass",
            message="No dilemmas/paths to check",
        )

    # Build dilemma → paths mapping from path node dilemma_id properties
    from questfoundry.graph.grow_algorithms import build_dilemma_paths

    dilemma_paths = build_dilemma_paths(graph)

    # Build path → beats mapping
    path_beats: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        path_beats.setdefault(path_id, []).append(beat_id)

    # Check each dilemma's paths for commits beats
    unresolved: list[str] = []
    for dilemma_id, paths in sorted(dilemma_paths.items()):
        dilemma_raw = dilemma_nodes[dilemma_id].get("raw_id", dilemma_id)
        for path_id in paths:
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
                path_raw = path_nodes[path_id].get("raw_id", path_id)
                unresolved.append(f"{path_raw}/{dilemma_raw}")

    if not unresolved:
        return ValidationCheck(
            name="dilemmas_resolved",
            severity="pass",
            message=f"All {len(dilemma_paths)} dilemmas resolved",
        )
    return ValidationCheck(
        name="dilemmas_resolved",
        severity="fail",
        message=f"Unresolved dilemmas: {', '.join(unresolved[:5])}",
    )


def check_passage_dag_cycles(graph: Graph) -> ValidationCheck:
    """Verify passage→choice→passage directed edges form a DAG (no cycles).

    Uses topological sort via Kahn's algorithm on the passage graph derived
    from choice edges.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return ValidationCheck(
            name="passage_dag_cycles",
            severity="pass",
            message="No passages to check",
        )

    choice_nodes = graph.get_nodes_by_type("choice")
    if not choice_nodes:
        return ValidationCheck(
            name="passage_dag_cycles",
            severity="pass",
            message="No choices to check",
        )

    # Build directed graph: passage → passage via choices
    in_degree: dict[str, int] = dict.fromkeys(passage_nodes, 0)
    successors: dict[str, list[str]] = {pid: [] for pid in passage_nodes}

    for choice_data in choice_nodes.values():
        # Skip is_return edges (spoke→hub back-links) — they are intentional cycles
        if choice_data.get("is_return"):
            continue
        from_p = choice_data.get("from_passage")
        to_p = choice_data.get("to_passage")
        if from_p and to_p and from_p in passage_nodes and to_p in passage_nodes:
            in_degree[to_p] += 1
            successors[from_p].append(to_p)

    # Kahn's algorithm
    queue = deque(sorted(pid for pid, deg in in_degree.items() if deg == 0))
    processed = 0

    while queue:
        node = queue.popleft()
        processed += 1
        for successor in sorted(successors[node]):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if processed == len(passage_nodes):
        return ValidationCheck(
            name="passage_dag_cycles",
            severity="pass",
            message=f"Passage graph is acyclic ({len(passage_nodes)} passages)",
        )

    cycle_nodes = [pid for pid, deg in in_degree.items() if deg > 0]
    return ValidationCheck(
        name="passage_dag_cycles",
        severity="fail",
        message=f"Cycle detected involving {len(cycle_nodes)} passages: {', '.join(sorted(cycle_nodes)[:5])}",
    )


def check_spine_arc_exists(graph: Graph) -> ValidationCheck:
    """Verify that a spine arc exists (computed from the beat DAG).

    The spine arc contains all canonical paths and is required for
    pruning and reachability analysis. Its absence indicates that
    enumerate_arcs failed to find a complete path combination.
    """
    from questfoundry.graph.grow_algorithms import enumerate_arcs

    arcs = enumerate_arcs(graph)

    # No arcs at all is a degenerate case (empty story) — warn, not fail.
    if not arcs:
        return ValidationCheck(
            name="spine_arc_exists",
            severity="warn",
            message="No arcs exist — spine arc check skipped",
        )

    for arc in arcs:
        if arc.arc_type == "spine":
            return ValidationCheck(
                name="spine_arc_exists",
                severity="pass",
                message="Spine arc found",
            )

    return ValidationCheck(
        name="spine_arc_exists",
        severity="fail",
        message=(
            f"No spine arc among {len(arcs)} arcs. "
            f"Story has no complete canonical path through all dilemmas."
        ),
    )


def check_dilemma_role_compliance(graph: Graph) -> list[ValidationCheck]:
    """Verify branch arcs honor their declared dilemma_role per-dilemma.

    Uses computed arcs and divergence points (not stored arc nodes).

    For each branch arc, identifies which dilemma paths differ from the spine.
    For each differing dilemma, checks only THAT dilemma's beats against its policy:
    - ``hard``: no beats from this dilemma shared with spine after divergence
    - ``soft``: at least ``payoff_budget`` exclusive beats for this dilemma
    - ``flavor``: always passes (no structural constraint)

    This per-dilemma approach is necessary because combinatorial arcs contain
    beats from ALL dilemmas, and only the flipped dilemma's beats should be
    checked against its policy.
    """
    from questfoundry.graph.grow_algorithms import (
        compute_divergence_points,
        enumerate_arcs,
    )

    arcs = enumerate_arcs(graph)
    if not arcs:
        return [
            ValidationCheck(
                name="dilemma_role_compliance",
                severity="pass",
                message="No arcs to check",
            )
        ]

    # Find spine arc and compute divergence points
    spine = next((a for a in arcs if a.arc_type == "spine"), None)
    spine_seq_set = set(spine.sequence) if spine else set()
    spine_paths = set(spine.paths) if spine else set()

    spine_arc_id = spine.arc_id if spine else None
    divergence_map = compute_divergence_points(arcs, spine_arc_id)

    beat_dilemmas = _build_beat_dilemma_map(graph)

    # Build path (raw ID) → prefixed dilemma mapping
    path_nodes = graph.get_nodes_by_type("path")
    raw_path_to_dilemma: dict[str, str] = {}
    for pid, pdata in path_nodes.items():
        did = pdata.get("dilemma_id")
        if did:
            raw_id = pdata.get("raw_id") or pid
            raw_path_to_dilemma[raw_id] = normalize_scoped_id(did, "dilemma")

    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    violations: list[ValidationCheck] = []
    checked = 0

    for arc in arcs:
        if arc.arc_type == "spine":
            continue

        sequence = arc.sequence
        div_info = divergence_map.get(arc.arc_id)
        diverges_at = div_info.diverges_at if div_info else None
        if not sequence or not diverges_at:
            continue

        try:
            div_idx = sequence.index(diverges_at)
        except ValueError:
            continue
        branch_after_div = sequence[div_idx + 1 :]

        # Find which dilemmas differ from spine (symmetric difference of path sets)
        arc_paths = set(arc.paths)
        flipped_dilemmas: set[str] = set()
        for path_raw in arc_paths.symmetric_difference(spine_paths):
            if did := raw_path_to_dilemma.get(path_raw):
                flipped_dilemmas.add(did)

        # Check each flipped dilemma's beats against ITS policy
        for dilemma_id in sorted(flipped_dilemmas):
            dnode = dilemma_nodes.get(dilemma_id)
            if not dnode:
                continue
            policy = dnode.get("dilemma_role")
            if policy is None:
                continue

            checked += 1
            # Filter to beats belonging to this dilemma only
            dilemma_beats_after = [
                b for b in branch_after_div if dilemma_id in beat_dilemmas.get(b, set())
            ]
            shared = [b for b in dilemma_beats_after if b in spine_seq_set]
            exclusive = [b for b in dilemma_beats_after if b not in spine_seq_set]

            if policy == "hard" and shared:
                violations.append(
                    ValidationCheck(
                        name="dilemma_role_compliance",
                        severity="fail",
                        message=(
                            f"arc::{arc.arc_id}: hard policy violated for {dilemma_id} — "
                            f"{len(shared)} shared beat(s) after divergence"
                        ),
                    )
                )
            elif policy == "soft":
                budget = dnode.get("payoff_budget", 2)
                if len(exclusive) < budget:
                    violations.append(
                        ValidationCheck(
                            name="dilemma_role_compliance",
                            severity="warn",
                            message=(
                                f"arc::{arc.arc_id}: soft policy for {dilemma_id} — "
                                f"{len(exclusive)} exclusive beat(s), needs {budget}"
                            ),
                        )
                    )

    if violations:
        return violations
    return [
        ValidationCheck(
            name="dilemma_role_compliance",
            severity="pass",
            message=f"All {checked} dilemma-arc pair(s) comply with dilemma role and payoff budget"
            if checked
            else "No branch arcs with divergence metadata to check",
        )
    ]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_grow_checks(graph: Graph) -> ValidationReport:
    """Run beat-DAG validation checks only (GROW Phase 10).

    Returns a ValidationReport containing structural beat-DAG checks.
    """
    checks: list[ValidationCheck] = [
        check_single_start(graph),
        check_passage_dag_cycles(graph),
        check_spine_arc_exists(graph),
        check_dilemmas_resolved(graph),
    ]
    checks.extend(check_dilemma_role_compliance(graph))
    return ValidationReport(checks=checks)


def run_all_checks(graph: Graph) -> ValidationReport:
    """Run all validation checks (beat-DAG + passage-layer) and aggregate results.

    Combines GROW beat-DAG checks with POLISH passage-layer checks for
    backward compatibility. Individual stages should prefer
    ``run_grow_checks()`` or ``run_passage_checks()``.
    """
    from questfoundry.graph.polish_validation import run_passage_checks

    grow_report = run_grow_checks(graph)
    passage_report = run_passage_checks(graph)
    return ValidationReport(checks=grow_report.checks + passage_report.checks)


# ---------------------------------------------------------------------------
# Backward-compatible lazy re-exports (passage-layer checks moved to
# polish_validation). Uses __getattr__ to avoid circular import at load time.
# ---------------------------------------------------------------------------

_MOVED_TO_POLISH = {
    "check_all_endings_reachable",
    "check_all_passages_reachable",
    "check_arc_divergence",
    "check_state_flag_gate_coverage",
    "check_commits_timing",
    "check_forward_path_reachability",
    "check_gate_co_satisfiability",
    "check_gate_satisfiability",
    "check_max_consecutive_linear",
    "check_prose_neutrality",
    "check_routing_coverage",
}

# Legacy private name alias (tests import this directly).
_compute_linear_threshold = compute_linear_threshold


def __getattr__(name: str) -> object:
    """Lazy re-export of passage-layer checks moved to polish_validation."""
    if name in _MOVED_TO_POLISH:
        from questfoundry.graph import polish_validation

        return getattr(polish_validation, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
