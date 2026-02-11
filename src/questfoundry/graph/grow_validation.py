"""Graph validation checks for GROW Phase 10.

Validates structural integrity and narrative pacing of the story graph
after all construction phases (1-9) have built it. These are pure,
deterministic functions operating on the graph — no LLM calls.

Validation categories:
- Structural: single start, reachability, DAG cycles, gate satisfiability
- Narrative: dilemma resolution, commits timing heuristics
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from questfoundry.graph.context import ENTITY_CATEGORIES, normalize_scoped_id

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Default narrative pacing thresholds (used as minimums for scaling).
_DEFAULT_MIN_BEATS_BEFORE_COMMITS = 3
_DEFAULT_MAX_COMMITS_POSITION_RATIO = 0.8
_DEFAULT_MAX_BUILDUP_GAP_BEATS = 5


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


def _passages_with_forward_incoming(graph: Graph) -> set[str]:
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


def _find_start_passage(graph: Graph) -> str | None:
    """Find the unique start passage (no forward incoming choice edges).

    Returns the passage ID if exactly one start exists, None otherwise.
    Excludes ``is_return`` edges from spoke→hub back-links.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return None
    passages_with_incoming = _passages_with_forward_incoming(graph)
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

    # Start passages = passages without forward incoming choice edges
    # (excludes is_return spoke→hub back-links)
    passages_with_incoming = _passages_with_forward_incoming(graph)
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


def check_commits_timing(graph: Graph) -> list[ValidationCheck]:
    """Check narrative pacing heuristics around commits beats.

    The player walks the *arc* sequence (all beats from all paths in the arc),
    not individual path beats. Measuring commits position against path-local
    beats produces false positives on short branch paths that sit inside
    longer arcs.

    For each path, checks against its arc's beat sequence:
    1. commits too early (<3 beats from arc start)
    2. No reveals/advances before commits (no buildup)
    3. commits too late (final 20% of arc)
    4. Large gap (>5 beats) after last reveals before commits

    Returns list of warning-level checks (timing issues are advisory, not blocking).
    """
    path_nodes = graph.get_nodes_by_type("path")
    beat_nodes = graph.get_nodes_by_type("beat")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    arc_nodes = graph.get_nodes_by_type("arc")

    if not path_nodes or not beat_nodes:
        return []

    # No arcs = pre-GROW or incomplete graph; skip timing checks
    if not arc_nodes:
        return []

    # Build path → dilemma node ID mapping for beat impact comparison
    path_dilemma: dict[str, str] = {}
    for path_id, path_data in path_nodes.items():
        did = path_data.get("dilemma_id")
        if did:
            prefixed = normalize_scoped_id(did, "dilemma")
            if prefixed in dilemma_nodes:
                path_dilemma[path_id] = prefixed

    # Build path → arc sequence mapping (prefer spine arc)
    path_to_arc_seq: dict[str, list[str]] = {}
    for _arc_id, arc_data in sorted(arc_nodes.items()):
        seq = arc_data.get("sequence", [])
        is_spine = arc_data.get("arc_type") == "spine"
        for path_raw in arc_data.get("paths", []):
            path_id = normalize_scoped_id(path_raw, "path")
            if is_spine:
                path_to_arc_seq[path_id] = seq  # spine takes priority
            elif path_id not in path_to_arc_seq:
                path_to_arc_seq[path_id] = seq  # branch as fallback

    checks: list[ValidationCheck] = []

    for path_id in sorted(path_nodes):
        if path_id not in path_dilemma:
            continue
        dilemma_node_id = path_dilemma[path_id]
        path_raw = path_nodes[path_id].get("raw_id", path_id)

        # Get arc sequence for this path
        arc_seq = path_to_arc_seq.get(path_id)
        if not arc_seq or len(arc_seq) < 2:
            continue

        # Scan arc sequence for this dilemma's commits/buildup beats
        commits_idx: int | None = None
        last_buildup_idx: int | None = None

        for idx, beat_id in enumerate(arc_seq):
            beat_data = beat_nodes.get(beat_id, {})
            for impact in beat_data.get("dilemma_impacts", []):
                if impact.get("dilemma_id") != dilemma_node_id:
                    continue
                effect = impact.get("effect", "")
                if effect == "commits":
                    commits_idx = idx
                elif effect in ("reveals", "advances"):
                    last_buildup_idx = idx

        if commits_idx is None:
            continue

        total_beats = len(arc_seq)

        # Scale thresholds with arc length — larger arcs get slightly wider
        # tolerances since beats are spread across more dilemmas.
        min_beats = max(_DEFAULT_MIN_BEATS_BEFORE_COMMITS, total_beats // 10)
        max_gap = max(_DEFAULT_MAX_BUILDUP_GAP_BEATS, total_beats // 8)

        # Check 1: commits too early
        if commits_idx < min_beats:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Path '{path_raw}': commits at arc position {commits_idx + 1}/{total_beats} (too early, <{min_beats} beats of setup)",
                )
            )

        # Check 2: No buildup before commits
        if last_buildup_idx is None:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Path '{path_raw}': no reveals/advances before commits",
                )
            )

        # Check 3: commits too late (final portion of arc)
        threshold = total_beats * _DEFAULT_MAX_COMMITS_POSITION_RATIO
        if commits_idx >= threshold:
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Path '{path_raw}': commits at arc position {commits_idx + 1}/{total_beats} (too late, >80%)",
                )
            )

        # Check 4: Large gap after last buildup
        if last_buildup_idx is not None and commits_idx - last_buildup_idx > max_gap:
            gap = commits_idx - last_buildup_idx
            checks.append(
                ValidationCheck(
                    name="commits_timing",
                    severity="warn",
                    message=f"Path '{path_raw}': {gap} beat gap between last reveals and commits",
                )
            )

    return checks


def check_spine_arc_exists(graph: Graph) -> ValidationCheck:
    """Verify that a spine arc exists in the graph.

    The spine arc contains all canonical paths and is required for
    pruning and reachability analysis. Its absence indicates that
    enumerate_arcs failed to find a complete path combination.
    """
    arc_nodes = graph.get_nodes_by_type("arc")

    # No arcs at all is a degenerate case (empty story) — warn, not fail.
    if not arc_nodes:
        return ValidationCheck(
            name="spine_arc_exists",
            severity="warn",
            message="No arcs exist — spine arc check skipped",
        )

    for data in arc_nodes.values():
        if data.get("arc_type") == "spine":
            return ValidationCheck(
                name="spine_arc_exists",
                severity="pass",
                message="Spine arc found",
            )

    return ValidationCheck(
        name="spine_arc_exists",
        severity="fail",
        message=(
            f"No spine arc among {len(arc_nodes)} arcs. "
            f"Story has no complete canonical path through all dilemmas."
        ),
    )


def check_arc_divergence(
    graph: Graph,
    *,
    min_exclusive_beats: int = 2,
    max_shared_ratio: float = 0.9,
) -> ValidationCheck:
    """Warn when branch arcs are too similar to the spine arc.

    Low divergence can produce a linear-feeling story even when multiple
    dilemmas exist. This check compares each branch arc's beat sequence
    against the spine arc and flags cases with too few exclusive beats
    or extremely high overlap.

    Args:
        graph: Story graph.
        min_exclusive_beats: Minimum beats in a branch arc not in spine.
        max_shared_ratio: Maximum allowed fraction of branch beats shared
            with the spine arc before warning.

    Returns:
        ValidationCheck with severity "warn" when divergence is insufficient.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return ValidationCheck(
            name="arc_divergence",
            severity="pass",
            message="No arcs to check",
        )

    spine_id = None
    for arc_id, data in arc_nodes.items():
        if data.get("arc_type") == "spine":
            spine_id = arc_id
            break

    if not spine_id:
        return ValidationCheck(
            name="arc_divergence",
            severity="warn",
            message="No spine arc found; divergence check skipped",
        )

    spine_seq = arc_nodes[spine_id].get("sequence", [])
    if not spine_seq:
        return ValidationCheck(
            name="arc_divergence",
            severity="warn",
            message="Spine arc has no sequence; divergence check skipped",
        )

    spine_set = set(spine_seq)
    total_branches = 0
    low_divergence: list[tuple[str, int, float]] = []

    for arc_id, data in arc_nodes.items():
        if arc_id == spine_id:
            continue
        seq = data.get("sequence", [])
        if not seq:
            continue
        total_branches += 1
        exclusive = [beat for beat in seq if beat not in spine_set]
        exclusive_count = len(exclusive)
        shared_ratio = 1 - (exclusive_count / len(seq))
        if exclusive_count < min_exclusive_beats or shared_ratio >= max_shared_ratio:
            low_divergence.append((arc_id, exclusive_count, shared_ratio))

    if not total_branches:
        return ValidationCheck(
            name="arc_divergence",
            severity="pass",
            message="No branch arcs to check",
        )

    if low_divergence:
        worst = max(low_divergence, key=lambda item: item[2])
        return ValidationCheck(
            name="arc_divergence",
            severity="warn",
            message=(
                f"Low divergence in {len(low_divergence)}/{total_branches} branch arcs "
                f"(min_exclusive_beats={min_exclusive_beats}, max_shared_ratio={max_shared_ratio:.2f}). "
                f"Worst: {worst[0]} exclusive={worst[1]} shared_ratio={worst[2]:.2f}"
            ),
        )

    return ValidationCheck(
        name="arc_divergence",
        severity="pass",
        message="All branch arcs show sufficient divergence from spine",
    )


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
    exempt_passages = _build_exempt_passages(graph, passages)
    starts = _find_start_passages(graph, passages)

    max_len = 0
    for _violation in _walk_linear_stretches(
        starts, adjacency, outgoing_count, exempt_passages, threshold=0
    ):
        max_len = max(max_len, len(_violation))
    return max_len


def _build_exempt_passages(graph: Graph, passages: dict[str, dict[str, object]]) -> set[str]:
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
    for pid, pdata in passages.items():
        # Exempt confront/resolve passages
        if pdata.get("from_beat") in exempt_beats:
            exempt.add(pid)
        # Exempt merged passages (they ARE the result of collapse)
        from_beats = pdata.get("from_beats")
        if from_beats and isinstance(from_beats, list) and len(from_beats) > 1:
            exempt.add(pid)
    return exempt


def _find_start_passages(graph: Graph, passages: dict[str, dict[str, object]]) -> list[str]:
    """Find passages with no forward incoming choice edges.

    Excludes ``is_return`` spoke→hub back-links.
    """
    has_incoming = _passages_with_forward_incoming(graph)
    return [pid for pid in passages if pid not in has_incoming]


def _walk_linear_stretches(
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


def check_max_consecutive_linear(graph: Graph, max_run: int = 2) -> ValidationCheck:
    """Warn when too many consecutive single-outgoing passages form a linear stretch.

    Long linear stretches create a passive reading experience. This check walks
    the passage graph and flags any path with more than ``max_run`` consecutive
    passages that each have exactly one outgoing choice.

    Passages whose beat has ``narrative_function`` in {"confront", "resolve"} are
    exempt, since linearity is narratively appropriate for climax/resolution.

    Args:
        graph: The story graph to validate.
        max_run: Maximum allowed consecutive single-outgoing passages.

    Returns:
        A ValidationCheck with severity "warn" if a violation is found, "pass" otherwise.
    """
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return ValidationCheck(
            name="max_consecutive_linear",
            severity="pass",
            message="No passages to check",
        )

    outgoing_count = build_outgoing_count(graph)
    adjacency = build_passage_adjacency(graph)
    exempt_passages = _build_exempt_passages(graph, passages)
    starts = _find_start_passages(graph, passages)

    violations = _walk_linear_stretches(starts, adjacency, outgoing_count, exempt_passages, max_run)

    if violations:
        longest = max(violations, key=len)
        return ValidationCheck(
            name="max_consecutive_linear",
            severity="warn",
            message=(
                f"Found {len(violations)} linear stretch(es) exceeding {max_run} "
                f"consecutive single-outgoing passages. Longest: {len(longest)} "
                f"passages ({', '.join(longest[:5])}{'...' if len(longest) > 5 else ''})"
            ),
        )

    return ValidationCheck(
        name="max_consecutive_linear",
        severity="pass",
        message=f"No linear stretches exceed {max_run} consecutive passages",
    )


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


def check_convergence_policy_compliance(graph: Graph) -> list[ValidationCheck]:
    """Verify branch arcs honor their declared convergence_policy per-dilemma.

    For each branch arc, identifies which dilemma paths differ from the spine.
    For each differing dilemma, checks only THAT dilemma's beats against its policy:
    - ``hard``: no beats from this dilemma shared with spine after divergence
    - ``soft``: at least ``payoff_budget`` exclusive beats for this dilemma
    - ``flavor``: always passes (no structural constraint)

    This per-dilemma approach is necessary because combinatorial arcs contain
    beats from ALL dilemmas, and only the flipped dilemma's beats should be
    checked against its policy.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return [
            ValidationCheck(
                name="convergence_policy_compliance",
                severity="pass",
                message="No arcs to check",
            )
        ]

    spine_seq_set = _get_spine_sequence(arc_nodes)

    # Get spine path set for comparison
    spine_paths: set[str] = set()
    for _aid, adata in arc_nodes.items():
        if adata.get("arc_type") == "spine":
            spine_paths = set(adata.get("paths", []))
            break

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

    for arc_id, data in sorted(arc_nodes.items()):
        if data.get("arc_type") == "spine":
            continue

        sequence: list[str] = data.get("sequence", [])
        diverges_at = data.get("diverges_at")
        if not sequence or not diverges_at:
            continue

        try:
            div_idx = sequence.index(diverges_at)
        except ValueError:
            continue
        branch_after_div = sequence[div_idx + 1 :]

        # Find which dilemmas differ from spine (symmetric difference of path sets)
        arc_paths = set(data.get("paths", []))
        flipped_dilemmas: set[str] = set()
        for path_raw in arc_paths.symmetric_difference(spine_paths):
            if did := raw_path_to_dilemma.get(path_raw):
                flipped_dilemmas.add(did)

        # Check each flipped dilemma's beats against ITS policy
        for dilemma_id in sorted(flipped_dilemmas):
            dnode = dilemma_nodes.get(dilemma_id)
            if not dnode:
                continue
            policy = dnode.get("convergence_policy")
            if policy is None or policy == "flavor":
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
                        name="convergence_policy_compliance",
                        severity="fail",
                        message=(
                            f"{arc_id}: hard policy violated for {dilemma_id} — "
                            f"{len(shared)} shared beat(s) after divergence"
                        ),
                    )
                )
            elif policy == "soft":
                budget = dnode.get("payoff_budget", 2)
                if len(exclusive) < budget:
                    violations.append(
                        ValidationCheck(
                            name="convergence_policy_compliance",
                            severity="warn",
                            message=(
                                f"{arc_id}: soft policy for {dilemma_id} — "
                                f"{len(exclusive)} exclusive beat(s), needs {budget}"
                            ),
                        )
                    )

    if violations:
        return violations
    return [
        ValidationCheck(
            name="convergence_policy_compliance",
            severity="pass",
            message=f"All {checked} dilemma-arc pair(s) comply with convergence policy"
            if checked
            else "No branch arcs with convergence metadata to check",
        )
    ]


def check_codeword_gate_coverage(graph: Graph) -> ValidationCheck:
    """Check that every codeword is consumed by a gate or overlay condition.

    Implements the "Residue Must Be Read" invariant: checks that each
    codeword appears in at least one ``choice.requires`` gate or
    ``overlay.when`` condition.
    """
    codeword_nodes = graph.get_nodes_by_type("codeword")
    if not codeword_nodes:
        return ValidationCheck(
            name="codeword_gate_coverage",
            severity="pass",
            message="No codewords in graph",
        )

    choice_nodes = graph.get_nodes_by_type("choice")
    consumed: set[str] = set()
    for choice_data in choice_nodes.values():
        consumed.update(choice_data.get("requires") or [])

    # Overlays are embedded arrays on entity nodes, not separate typed nodes.
    for category in ENTITY_CATEGORIES:
        for entity_data in graph.get_nodes_by_type(category).values():
            for overlay in entity_data.get("overlays") or []:
                consumed.update(overlay.get("when") or [])

    unconsumed = sorted(set(codeword_nodes.keys()) - consumed)
    if not unconsumed:
        return ValidationCheck(
            name="codeword_gate_coverage",
            severity="pass",
            message=f"All {len(codeword_nodes)} codeword(s) consumed by gates or overlays",
        )
    return ValidationCheck(
        name="codeword_gate_coverage",
        severity="warn",
        message=(
            f"{len(unconsumed)} of {len(codeword_nodes)} codeword(s) not consumed "
            f"by any choice.requires or overlay.when: {', '.join(unconsumed[:5])}"
            f"{'...' if len(unconsumed) > 5 else ''}"
        ),
    )


def check_forward_path_reachability(graph: Graph) -> ValidationCheck:
    """Warn when a non-ending passage has only gated outgoing choices.

    Catches soft-lock risks where ``requires`` wiring accidentally gates
    ALL forward paths from a passage. Excludes ``is_return`` choices
    (spoke-to-hub return links) from the forward-path count.

    v1 simplification: does not check whether requires are already
    satisfiable (would require path simulation).
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    if not choice_nodes:
        return ValidationCheck(
            name="forward_path_reachability",
            severity="pass",
            message="No choices in graph",
        )

    # Build from_passage → list of choice data
    from_passage_choices: dict[str, list[dict[str, object]]] = {}
    for choice_data in choice_nodes.values():
        fp = choice_data.get("from_passage")
        if fp:
            from_passage_choices.setdefault(fp, []).append(choice_data)

    # Identify endings (no outgoing choices at all, or only return links)
    passages = graph.get_nodes_by_type("passage")
    soft_locked: list[str] = []

    for pid in sorted(passages):
        choices = from_passage_choices.get(pid, [])
        forward = [c for c in choices if not c.get("is_return")]
        if not forward:
            continue  # ending passage — no forward choices
        ungated = [c for c in forward if not c.get("requires")]
        if not ungated:
            soft_locked.append(pid)

    if not soft_locked:
        return ValidationCheck(
            name="forward_path_reachability",
            severity="pass",
            message="All non-ending passages have at least one ungated forward choice",
        )
    return ValidationCheck(
        name="forward_path_reachability",
        severity="warn",
        message=(
            f"{len(soft_locked)} passage(s) have only gated forward choices "
            f"(potential soft-lock): {', '.join(soft_locked[:5])}"
            f"{'...' if len(soft_locked) > 5 else ''}"
        ),
    )


def _compute_linear_threshold(graph: Graph) -> int:
    """Scale max consecutive linear threshold with passage count.

    Larger stories naturally have longer linear stretches between branch
    points (each dilemma adds ~10 passages). The default of 2 is kept for
    small stories; larger stories get proportionally wider tolerance.
    """
    passage_count = len(graph.get_nodes_by_type("passage"))
    return max(2, passage_count // 20)


def run_all_checks(graph: Graph) -> ValidationReport:
    """Run all Phase 10 validation checks and aggregate results.

    Returns a ValidationReport containing all structural and timing checks.
    """
    linear_threshold = _compute_linear_threshold(graph)
    checks: list[ValidationCheck] = [
        check_single_start(graph),
        check_all_passages_reachable(graph),
        check_all_endings_reachable(graph),
        check_spine_arc_exists(graph),
        check_arc_divergence(graph),
        check_dilemmas_resolved(graph),
        check_gate_satisfiability(graph),
        check_passage_dag_cycles(graph),
        check_max_consecutive_linear(graph, max_run=linear_threshold),
        check_codeword_gate_coverage(graph),
        check_forward_path_reachability(graph),
    ]
    checks.extend(check_commits_timing(graph))
    checks.extend(check_convergence_policy_compliance(graph))
    return ValidationReport(checks=checks)
