"""Graph validation checks for GROW Phase 10.

Validates structural integrity and narrative pacing of the story graph
after all construction phases (1-9) have built it. These are pure,
deterministic functions operating on the graph — no LLM calls.

Validation categories:
- Structural: single start, reachability, DAG cycles, gate satisfiability
- Narrative: tension resolution, commits timing heuristics
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Narrative pacing thresholds for commits timing validation.
# Based on three-act structure: setup requires buildup before resolution.
MIN_BEATS_BEFORE_COMMITS = 3  # Minimum beats for narrative setup
MAX_COMMITS_POSITION_RATIO = 0.8  # Commits should allow 20% for denouement
MAX_BUILDUP_GAP_BEATS = 5  # Maximum beats between last buildup and commits


@dataclass
class ValidationCheck:
    """Result of a single validation check.

    Attributes:
        name: Identifier for the check.
        severity: "pass", "warn", or "fail".
        message: Human-readable description of the result.
    """

    name: str
    severity: Literal["pass", "warn", "fail"]
    message: str = ""


@dataclass
class ValidationReport:
    """Aggregated results of all Phase 10 checks.

    Attributes:
        checks: List of individual validation check results.
    """

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """True if any check has severity 'fail'."""
        return any(c.severity == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """True if any check has severity 'warn'."""
        return any(c.severity == "warn" for c in self.checks)

    @property
    def summary(self) -> str:
        """Human-readable summary of all checks."""
        fails = [c for c in self.checks if c.severity == "fail"]
        warns = [c for c in self.checks if c.severity == "warn"]
        passes = [c for c in self.checks if c.severity == "pass"]

        parts: list[str] = []
        if fails:
            parts.append(f"{len(fails)} failed")
        if warns:
            parts.append(f"{len(warns)} warnings")
        if passes:
            parts.append(f"{len(passes)} passed")
        return ", ".join(parts)


def _find_start_passage(graph: Graph) -> str | None:
    """Find the unique start passage (no incoming choice_to edges).

    Returns the passage ID if exactly one start exists, None otherwise.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return None
    # choice_to edges point TO destination passages
    choice_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_to")
    passages_with_incoming: set[str] = {edge["to"] for edge in choice_to_edges}
    start_passages = [pid for pid in passage_nodes if pid not in passages_with_incoming]
    return start_passages[0] if len(start_passages) == 1 else None


def _build_passage_successors(graph: Graph) -> dict[str, list[str]]:
    """Build passage->passage successor map from choice nodes.

    Each choice node has from_passage and to_passage fields defining
    the directed edge in the passage graph.
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    successors: dict[str, list[str]] = {}
    for choice_data in choice_nodes.values():
        from_p = choice_data.get("from_passage")
        to_p = choice_data.get("to_passage")
        if from_p and to_p:
            successors.setdefault(from_p, []).append(to_p)
    return successors


def _bfs_reachable(start: str, successors: dict[str, list[str]]) -> set[str]:
    """BFS to find all reachable passages from start."""
    reachable: set[str] = {start}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for next_p in successors.get(current, []):
            if next_p not in reachable:
                reachable.add(next_p)
                queue.append(next_p)
    return reachable


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

    # Find passages with incoming choice_to edges
    choice_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_to")
    passages_with_incoming: set[str] = {edge["to"] for edge in choice_to_edges}

    # Start passages = passages without incoming choice_to edges
    start_passages = [pid for pid in passage_nodes if pid not in passages_with_incoming]

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


def check_all_passages_reachable(graph: Graph) -> ValidationCheck:
    """Verify all passages are reachable from the start passage via choice edges.

    BFS from the start passage (no incoming choice_to edges) via choice_to edges.
    Reports any unreachable passages.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return ValidationCheck(
            name="all_passages_reachable",
            severity="pass",
            message="No passages to check",
        )

    start = _find_start_passage(graph)
    if start is None:
        return ValidationCheck(
            name="all_passages_reachable",
            severity="fail",
            message="Cannot check reachability: no unique start passage",
        )

    successors = _build_passage_successors(graph)
    reachable = _bfs_reachable(start, successors)

    unreachable = set(passage_nodes.keys()) - reachable
    if not unreachable:
        return ValidationCheck(
            name="all_passages_reachable",
            severity="pass",
            message=f"All {len(passage_nodes)} passages reachable from start",
        )
    return ValidationCheck(
        name="all_passages_reachable",
        severity="fail",
        message=f"{len(unreachable)} unreachable passages: {', '.join(sorted(unreachable)[:5])}",
    )


def check_all_endings_reachable(graph: Graph) -> ValidationCheck:
    """Verify at least one ending is reachable from the start passage.

    Endings are passages with no outgoing choice_from edges. At least one
    ending must be reachable for the story to be completable.

    Edge semantics:
    - choice_from: choice -> originating_passage (passage the choice leads FROM)
    - choice_to: choice -> destination_passage (passage the choice leads TO)
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return ValidationCheck(
            name="all_endings_reachable",
            severity="pass",
            message="No passages to check",
        )

    start = _find_start_passage(graph)
    if start is None:
        return ValidationCheck(
            name="all_endings_reachable",
            severity="fail",
            message="Cannot check endings: no unique start passage",
        )

    # Find endings: passages with no outgoing choices (choice_from -> passage)
    choice_from_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_from")
    passages_with_outgoing: set[str] = {edge["to"] for edge in choice_from_edges}
    endings = [pid for pid in passage_nodes if pid not in passages_with_outgoing]

    if not endings:
        return ValidationCheck(
            name="all_endings_reachable",
            severity="fail",
            message="No ending passages found (all passages have outgoing edges)",
        )

    successors = _build_passage_successors(graph)
    reachable = _bfs_reachable(start, successors)

    reachable_endings = [e for e in endings if e in reachable]
    if reachable_endings:
        return ValidationCheck(
            name="all_endings_reachable",
            severity="pass",
            message=f"{len(reachable_endings)}/{len(endings)} endings reachable",
        )
    return ValidationCheck(
        name="all_endings_reachable",
        severity="fail",
        message=f"No endings reachable from start (0/{len(endings)} reachable)",
    )


def check_tensions_resolved(graph: Graph) -> ValidationCheck:
    """Verify each explored tension has at least one commits beat per thread.

    Re-checks after gap insertion phases (4b/4c) may have altered the graph.
    """
    tension_nodes = graph.get_nodes_by_type("tension")
    thread_nodes = graph.get_nodes_by_type("thread")
    beat_nodes = graph.get_nodes_by_type("beat")

    if not tension_nodes or not thread_nodes:
        return ValidationCheck(
            name="tensions_resolved",
            severity="pass",
            message="No tensions/threads to check",
        )

    # Build tension → threads mapping from thread node tension_id properties
    from questfoundry.graph.grow_algorithms import build_tension_threads

    tension_threads = build_tension_threads(graph)

    # Build thread → beats mapping
    thread_beats: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        thread_id = edge["to"]
        thread_beats.setdefault(thread_id, []).append(beat_id)

    # Check each tension's threads for commits beats
    unresolved: list[str] = []
    for tension_id, threads in sorted(tension_threads.items()):
        tension_raw = tension_nodes[tension_id].get("raw_id", tension_id)
        for thread_id in threads:
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
                thread_raw = thread_nodes[thread_id].get("raw_id", thread_id)
                unresolved.append(f"{thread_raw}/{tension_raw}")

    if not unresolved:
        return ValidationCheck(
            name="tensions_resolved",
            severity="pass",
            message=f"All {len(tension_threads)} tensions resolved",
        )
    return ValidationCheck(
        name="tensions_resolved",
        severity="fail",
        message=f"Unresolved tensions: {', '.join(unresolved[:5])}",
    )


def check_gate_satisfiability(graph: Graph) -> ValidationCheck:
    """Verify all choice requires are satisfiable (required codewords exist globally).

    Collects all grantable codewords (union of all grants lists). For each
    choice with non-empty requires, verifies every required codeword is in
    the global grantable set.
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    if not choice_nodes:
        return ValidationCheck(
            name="gate_satisfiability",
            severity="pass",
            message="No choices to check",
        )

    # Collect all globally grantable codewords
    grantable: set[str] = set()
    for choice_data in choice_nodes.values():
        grants = choice_data.get("grants", [])
        grantable.update(grants)

    # Check each choice's requires
    unsatisfiable: list[str] = []
    for choice_id, choice_data in sorted(choice_nodes.items()):
        requires = choice_data.get("requires", [])
        for req in requires:
            if req not in grantable:
                unsatisfiable.append(f"{choice_id} requires '{req}'")

    if not unsatisfiable:
        return ValidationCheck(
            name="gate_satisfiability",
            severity="pass",
            message=f"All gates satisfiable ({len(grantable)} codewords grantable)",
        )
    return ValidationCheck(
        name="gate_satisfiability",
        severity="fail",
        message=f"Unsatisfiable gates: {', '.join(unsatisfiable[:5])}",
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


def check_commits_timing(graph: Graph) -> list[ValidationCheck]:
    """Check narrative pacing heuristics around commits beats.

    For each thread, checks:
    1. commits too early (<3 beats from arc start)
    2. No reveals/advances before commits (no buildup)
    3. commits too late (final 20% of arc)
    4. Large gap (>5 beats) after last reveals before commits

    Returns list of warning-level checks (timing issues are advisory, not blocking).
    """
    thread_nodes = graph.get_nodes_by_type("thread")
    beat_nodes = graph.get_nodes_by_type("beat")
    tension_nodes = graph.get_nodes_by_type("tension")

    if not thread_nodes or not beat_nodes:
        return []

    # Build thread → tension mapping from thread node tension_id properties
    thread_tension: dict[str, str] = {}
    for thread_id, thread_data in thread_nodes.items():
        tid = thread_data.get("tension_id")
        if tid:
            prefixed = tid if tid.startswith("tension::") else f"tension::{tid}"
            if prefixed in tension_nodes:
                tension_raw = tension_nodes[prefixed].get("raw_id", prefixed)
                thread_tension[thread_id] = tension_raw

    # Build thread → beats mapping (ordered by requires)
    thread_beats: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        thread_id = edge["to"]
        if beat_id in beat_nodes:
            thread_beats.setdefault(thread_id, []).append(beat_id)

    # Sort beats within each thread by topological order if possible.
    # Import here to avoid circular dependency (grow_algorithms imports Graph types).
    from questfoundry.graph.grow_algorithms import topological_sort_beats

    for thread_id in thread_beats:
        try:
            thread_beats[thread_id] = topological_sort_beats(graph, thread_beats[thread_id])
        except ValueError:
            thread_beats[thread_id] = sorted(thread_beats[thread_id])

    checks: list[ValidationCheck] = []

    for thread_id, beat_sequence in sorted(thread_beats.items()):
        if thread_id not in thread_tension:
            continue
        tension_raw = thread_tension[thread_id]
        thread_raw = thread_nodes[thread_id].get("raw_id", thread_id)

        # Find commits beat index and buildup beats
        commits_idx: int | None = None
        last_buildup_idx: int | None = None

        for idx, beat_id in enumerate(beat_sequence):
            beat_data = beat_nodes.get(beat_id, {})
            impacts = beat_data.get("tension_impacts", [])
            for impact in impacts:
                if impact.get("tension_id") != tension_raw:
                    continue
                effect = impact.get("effect", "")
                if effect == "commits":
                    commits_idx = idx
                elif effect in ("reveals", "advances"):
                    last_buildup_idx = idx

        if commits_idx is None:
            continue

        total_beats = len(beat_sequence)
        if total_beats < 2:
            continue

        # Check 1: commits too early
        if commits_idx < MIN_BEATS_BEFORE_COMMITS:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Thread '{thread_raw}': commits at beat {commits_idx + 1}/{total_beats} (too early, <{MIN_BEATS_BEFORE_COMMITS} beats)",
                )
            )

        # Check 2: No buildup before commits
        if last_buildup_idx is None:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Thread '{thread_raw}': no reveals/advances before commits",
                )
            )

        # Check 3: commits too late (final portion of arc)
        threshold = total_beats * MAX_COMMITS_POSITION_RATIO
        if commits_idx >= threshold:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Thread '{thread_raw}': commits at beat {commits_idx + 1}/{total_beats} (too late, >80%)",
                )
            )

        # Check 4: Large gap after last buildup
        if last_buildup_idx is not None and commits_idx - last_buildup_idx > MAX_BUILDUP_GAP_BEATS:
            gap = commits_idx - last_buildup_idx
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Thread '{thread_raw}': {gap} beat gap between last reveals and commits",
                )
            )

    return checks


def run_all_checks(graph: Graph) -> ValidationReport:
    """Run all Phase 10 validation checks and aggregate results.

    Returns a ValidationReport containing all structural and timing checks.
    """
    checks: list[ValidationCheck] = [
        check_single_start(graph),
        check_all_passages_reachable(graph),
        check_all_endings_reachable(graph),
        check_tensions_resolved(graph),
        check_gate_satisfiability(graph),
        check_passage_dag_cycles(graph),
    ]
    checks.extend(check_commits_timing(graph))
    return ValidationReport(checks=checks)
