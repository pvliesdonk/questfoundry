"""POLISH validation functions.

Entry contract validation (validate_grow_output) checks GROW's output
before POLISH begins. Exit validation (validate_polish_output) checks
the passage graph after Phase 6 application.

Passage-layer checks (reachability, gates, routing, prose neutrality,
arc divergence, etc.) moved here from grow_validation.py. They run
during POLISH Phase 7 via ``run_passage_checks()``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import get_primary_beat, normalize_scoped_id
from questfoundry.graph.grow_validation import (
    build_exempt_passages,
    build_outgoing_count,
    build_passage_adjacency,
    find_start_passages,
    passages_with_forward_incoming,
    walk_linear_stretches,
)
from questfoundry.graph.validation_types import ValidationCheck, ValidationReport

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


# ---------------------------------------------------------------------------
# Entry contract validation (POLISH input — checks GROW output)
# ---------------------------------------------------------------------------


def validate_grow_output(graph: Graph) -> list[str]:
    """Verify GROW's output meets POLISH's input contract.

    Args:
        graph: Graph containing GROW stage output.

    Returns:
        List of error strings. Empty means valid.
    """
    errors: list[str] = []

    # 1. Beat nodes exist with summaries and dilemma_impacts
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        errors.append("No beat nodes found in graph")
    else:
        for beat_id, beat_data in beat_nodes.items():
            if not beat_data.get("summary"):
                errors.append(f"Beat {beat_id} missing summary")
            # Accept both plural "dilemma_impacts" and legacy singular "dilemma_impact"
            # from older GROW outputs that used the singular key.
            if "dilemma_impacts" not in beat_data and "dilemma_impact" not in beat_data:
                errors.append(f"Beat {beat_id} missing dilemma_impacts")

    # 2. No cycles in predecessor DAG
    if beat_nodes:
        _check_predecessor_cycles(graph, beat_nodes, errors)

    # 3. Every beat has exactly one belongs_to edge
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beats_with_path: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes:
            beats_with_path.setdefault(from_id, []).append(to_id)

    for beat_id in beat_nodes:
        paths = beats_with_path.get(beat_id, [])
        if len(paths) == 0:
            errors.append(f"Beat {beat_id} has no belongs_to edge (no path assignment)")
        elif len(paths) > 1:
            errors.append(
                f"Beat {beat_id} has {len(paths)} belongs_to edges (must have exactly 1): {paths}"
            )

    # 4. State flag nodes exist for explored dilemmas
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    state_flag_nodes = graph.get_nodes_by_type("state_flag")
    explored_dilemmas = {
        did
        for did, ddata in dilemma_nodes.items()
        # GROW may omit status for dilemmas it fully explored; treat None as "explored".
        if ddata.get("status") in ("explored", None)
    }
    dilemmas_with_flags: set[str] = set()
    for _flag_id, flag_data in state_flag_nodes.items():
        dilemma_ref = flag_data.get("dilemma_id") or flag_data.get("dilemma")
        if dilemma_ref:
            dilemmas_with_flags.add(dilemma_ref)

    for dilemma_id in explored_dilemmas:
        if dilemma_id not in dilemmas_with_flags:
            errors.append(f"Explored dilemma {dilemma_id} has no state flag nodes")

    # 5. Dilemma nodes have dilemma_role set
    for dilemma_id, dilemma_data in dilemma_nodes.items():
        if not dilemma_data.get("dilemma_role"):
            errors.append(f"Dilemma {dilemma_id} missing dilemma_role (hard/soft)")

    # 6. Intersection groups reference beats from different paths only
    intersection_groups = graph.get_nodes_by_type("intersection_group")
    for group_id, group_data in intersection_groups.items():
        _check_intersection_group_paths(
            graph, group_id, group_data, beat_nodes, beats_with_path, errors
        )

    return errors


def _check_predecessor_cycles(
    graph: Graph,
    beat_nodes: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    """Check for cycles in predecessor edges using Kahn's algorithm."""
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # Build adjacency: predecessor edges mean "from depends on to"
    # i.e., edge from A to B means "A's predecessor is B" (B comes before A)
    in_degree: dict[str, int] = dict.fromkeys(beat_nodes, 0)
    adj: dict[str, list[str]] = {bid: [] for bid in beat_nodes}

    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            in_degree[from_id] += 1
            adj[to_id].append(from_id)

    queue = [bid for bid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(beat_nodes):
        cycle_members = [bid for bid, deg in in_degree.items() if deg > 0]
        errors.append(
            f"Cycle detected in predecessor DAG among {len(cycle_members)} beats: "
            f"{', '.join(sorted(cycle_members)[:5])}" + ("..." if len(cycle_members) > 5 else "")
        )


def _check_intersection_group_paths(
    graph: Graph,  # noqa: ARG001
    group_id: str,
    group_data: dict[str, Any],
    beat_nodes: dict[str, dict[str, Any]],
    beats_with_path: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Check that an intersection group's beats come from different paths."""
    # Intersection group nodes store beat IDs in node_ids field
    node_ids = group_data.get("node_ids", [])
    if not node_ids:
        errors.append(f"Intersection group {group_id} has empty node_ids")
        return

    paths_seen: set[str] = set()
    for beat_id in node_ids:
        if beat_id not in beat_nodes:
            continue
        beat_paths = beats_with_path.get(beat_id, [])
        for path_id in beat_paths:
            if path_id in paths_seen:
                errors.append(
                    f"Intersection group {group_id} has multiple beats from the same path {path_id}"
                )
                return
            paths_seen.add(path_id)


# ---------------------------------------------------------------------------
# Phase 7: Passage graph validation (exit contract)
# ---------------------------------------------------------------------------


def validate_polish_output(graph: Graph) -> list[str]:
    """Validate the passage graph produced by POLISH Phases 4-6.

    Checks structural completeness, variant integrity, choice integrity,
    and feasibility constraints.

    Args:
        graph: Graph with passage layer applied.

    Returns:
        List of error strings. Empty means valid.
    """
    errors: list[str] = []

    beat_nodes = graph.get_nodes_by_type("beat")
    passage_nodes = graph.get_nodes_by_type("passage")

    if not passage_nodes:
        errors.append("No passage nodes found — Phase 6 may not have run")
        return errors

    # Build beat → passage mapping from grouped_in edges
    grouped_in_edges = graph.get_edges(edge_type="grouped_in")
    beat_to_passages: dict[str, list[str]] = {}
    passage_beats: dict[str, list[str]] = {pid: [] for pid in passage_nodes}

    for edge in grouped_in_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in passage_nodes:
            beat_to_passages.setdefault(from_id, []).append(to_id)
            passage_beats[to_id].append(from_id)

    _check_beat_grouping(beat_nodes, beat_to_passages, errors)
    _check_passage_beats(passage_nodes, passage_beats, errors)
    _check_start_passage(passage_beats, beat_nodes, graph, errors)
    _check_passage_reachability(passage_nodes, graph, errors)
    _check_variant_integrity(passage_nodes, graph, errors)
    _check_choice_integrity(graph, errors)
    _check_residue_ordering(graph, errors)

    return errors


def _check_beat_grouping(
    beat_nodes: dict[str, dict[str, Any]],
    beat_to_passages: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Every beat must be grouped into exactly one passage."""
    for beat_id in beat_nodes:
        passages = beat_to_passages.get(beat_id, [])
        if len(passages) == 0:
            # Skip micro/residue/sidetrack beats — they may not exist yet at validation time
            role = beat_nodes[beat_id].get("role", "")
            if role not in ("micro_beat", "residue_beat", "sidetrack_beat"):
                errors.append(f"Beat {beat_id} not grouped into any passage")
        elif len(passages) > 1:
            errors.append(f"Beat {beat_id} grouped into {len(passages)} passages: {passages}")


def _check_passage_beats(
    passage_nodes: dict[str, dict[str, Any]],
    passage_beats: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Every non-variant passage must have at least one beat."""
    for passage_id, pdata in passage_nodes.items():
        # Skip variant and diamond alt passages — they may not have direct beats
        if pdata.get("is_variant") or pdata.get("is_diamond_alt"):
            continue
        beats = passage_beats.get(passage_id, [])
        if not beats:
            errors.append(f"Passage {passage_id} has no beats (no grouped_in edges)")


def _check_start_passage(
    passage_beats: dict[str, list[str]],
    beat_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """Exactly one start passage must exist (containing the earliest beat)."""
    # Find root beats (no predecessor edges pointing TO them)
    # Exclude synthetic beats (micro_beat, residue_beat, sidetrack_beat) —
    # they're added fresh by Phase 6 with no predecessor edges
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    _synthetic_roles = {"micro_beat", "residue_beat", "sidetrack_beat"}
    beats_with_parents: set[str] = set()
    for edge in predecessor_edges:
        if edge["from"] in beat_nodes:
            beats_with_parents.add(edge["from"])

    root_beats = {
        bid
        for bid in beat_nodes
        if bid not in beats_with_parents and beat_nodes[bid].get("role", "") not in _synthetic_roles
    }

    # Find which passages contain root beats
    start_passages: set[str] = set()
    for passage_id, beats in passage_beats.items():
        if any(bid in root_beats for bid in beats):
            start_passages.add(passage_id)

    if len(start_passages) == 0:
        errors.append("No start passage found (no passage contains a root beat)")
    elif len(start_passages) > 1:
        errors.append(f"Multiple start passages found: {sorted(start_passages)}")


def _check_passage_reachability(
    passage_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """All passages should be reachable from start via choice edges."""
    choice_edges = graph.get_edges(edge_type="choice")
    precedes_edges = graph.get_edges(edge_type="precedes")

    if not choice_edges and len(passage_nodes) > 1:
        # No choices at all — can't check reachability
        return

    # Build adjacency from choice + precedes edges
    adj: dict[str, set[str]] = {pid: set() for pid in passage_nodes}
    for edge in choice_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in passage_nodes and to_id in passage_nodes:
            adj[from_id].add(to_id)
    for edge in precedes_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in passage_nodes and to_id in passage_nodes:
            adj[from_id].add(to_id)

    # BFS from all passages with no incoming edges
    incoming: dict[str, int] = dict.fromkeys(passage_nodes, 0)
    for _pid, neighbors in adj.items():
        for n in neighbors:
            incoming[n] = incoming.get(n, 0) + 1

    starts = {pid for pid, count in incoming.items() if count == 0}
    if not starts:
        # No clear start — skip check
        return

    visited: set[str] = set()
    queue: deque[str] = deque(starts)
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    unreachable = set(passage_nodes.keys()) - visited
    # Filter out variant passages — they're reachable through their base
    unreachable_real = {
        pid
        for pid in unreachable
        if not passage_nodes[pid].get("is_variant") and not passage_nodes[pid].get("is_diamond_alt")
    }

    if unreachable_real:
        errors.append(
            f"{len(unreachable_real)} passage(s) unreachable from start: "
            f"{sorted(unreachable_real)[:5]}"
        )


def _check_variant_integrity(
    passage_nodes: dict[str, dict[str, Any]],
    graph: Graph,
    errors: list[str],
) -> None:
    """Every variant passage must have a variant_of edge to a base passage."""
    variant_of_edges = graph.get_edges(edge_type="variant_of")
    variants_with_base: set[str] = set()
    for edge in variant_of_edges:
        variants_with_base.add(edge["from"])

    for passage_id, pdata in passage_nodes.items():
        if pdata.get("is_variant") and passage_id not in variants_with_base:
            errors.append(f"Variant passage {passage_id} has no variant_of edge to a base passage")


def _check_choice_integrity(
    graph: Graph,
    errors: list[str],
) -> None:
    """Check choice edge integrity: unique labels per source passage."""
    choice_edges = graph.get_edges(edge_type="choice")

    # Group choices by source passage
    choices_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in choice_edges:
        from_id = edge["from"]
        choices_by_source.setdefault(from_id, []).append(edge)

    from collections import Counter

    for source_id, choices in choices_by_source.items():
        # Check for duplicate labels
        labels: list[str] = [c["label"] for c in choices if c.get("label")]
        counts = Counter(labels)
        duplicate_labels = sorted(label for label, count in counts.items() if count > 1)
        if duplicate_labels:
            errors.append(f"Passage {source_id} has duplicate choice labels: {duplicate_labels}")


def _check_residue_ordering(
    graph: Graph,
    errors: list[str],
) -> None:
    """Residue passages must precede their target shared passages."""
    precedes_edges = graph.get_edges(edge_type="precedes")
    passage_nodes = graph.get_nodes_by_type("passage")

    # Check that residue passages have precedes edges
    passages_with_precedes = {e["from"] for e in precedes_edges}
    for passage_id, pdata in passage_nodes.items():
        if pdata.get("is_residue") and passage_id not in passages_with_precedes:
            errors.append(f"Residue passage {passage_id} has no precedes edge to a target passage")


# ---------------------------------------------------------------------------
# Passage-layer check functions (moved from grow_validation.py)
# ---------------------------------------------------------------------------

# Narrative pacing thresholds for commits_timing (copied from grow_validation).
_DEFAULT_MIN_BEATS_BEFORE_COMMITS = 3
_DEFAULT_MAX_COMMITS_POSITION_RATIO = 0.8
_DEFAULT_MAX_BUILDUP_GAP_BEATS = 5


def _find_start_passage_for_check(graph: Graph) -> str | None:
    """Find the unique start passage (no forward incoming choice edges).

    Returns the passage ID if exactly one start exists, None otherwise.
    Excludes ``is_return`` edges from spoke→hub back-links.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return None
    has_incoming = passages_with_forward_incoming(graph)
    start = [pid for pid in passage_nodes if pid not in has_incoming]
    return start[0] if len(start) == 1 else None


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

    start = _find_start_passage_for_check(graph)
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

    start = _find_start_passage_for_check(graph)
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


def check_gate_satisfiability(graph: Graph) -> ValidationCheck:
    """Verify all choice requires are satisfiable (required state flags exist globally).

    Collects all grantable state flags (union of all grants lists). For each
    choice with non-empty requires, verifies every required state flag is in
    the global grantable set.
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    if not choice_nodes:
        return ValidationCheck(
            name="gate_satisfiability",
            severity="pass",
            message="No choices to check",
        )

    # Collect all globally grantable state flags
    grantable: set[str] = set()
    for choice_data in choice_nodes.values():
        grants = choice_data.get("grants", [])
        grantable.update(grants)

    # Check each choice's requires_state_flags
    unsatisfiable: list[str] = []
    for choice_id, choice_data in sorted(choice_nodes.items()):
        requires = choice_data.get("requires_state_flags", [])
        for req in requires:
            if req not in grantable:
                unsatisfiable.append(f"{choice_id} requires_state_flags '{req}'")

    if not unsatisfiable:
        return ValidationCheck(
            name="gate_satisfiability",
            severity="pass",
            message=f"All gates satisfiable ({len(grantable)} state flags grantable)",
        )
    return ValidationCheck(
        name="gate_satisfiability",
        severity="fail",
        message=f"Unsatisfiable gates: {', '.join(unsatisfiable[:5])}",
    )


def check_gate_co_satisfiability(graph: Graph) -> ValidationCheck:
    """Verify all required state flags are co-reachable in a single playthrough.

    For each choice with non-empty requires, checks that at least one arc
    provides ALL required state flags.  A gate that requires state flags from
    mutually exclusive paths is paradoxical — the player can never satisfy it.
    """
    choice_nodes = graph.get_nodes_by_type("choice")
    if not choice_nodes:
        return ValidationCheck(
            name="gate_co_satisfiability",
            severity="pass",
            message="No choices to check",
        )

    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return ValidationCheck(
            name="gate_co_satisfiability",
            severity="pass",
            message="No arcs to check",
        )

    # Build consequence→state_flag lookup
    cons_to_state_flag = {
        edge["to"]: edge["from"]
        for edge in graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
    }

    # Build path→consequences lookup
    path_consequences: dict[str, list[str]] = {}
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="has_consequence"):
        path_consequences.setdefault(edge["from"], []).append(edge["to"])

    # Build earnable state flags per arc
    arc_state_flags: dict[str, set[str]] = {}
    for arc_id, arc_data in arc_nodes.items():
        sfs: set[str] = set()
        for raw_path in arc_data.get("paths", []):
            path_id = normalize_scoped_id(raw_path, "path")
            for cons_id in path_consequences.get(path_id, []):
                sf = cons_to_state_flag.get(cons_id)
                if sf:
                    sfs.add(sf)
        arc_state_flags[arc_id] = sfs

    # Check each gated choice
    paradoxical: list[str] = []
    for choice_id, choice_data in sorted(choice_nodes.items()):
        requires = set(choice_data.get("requires_state_flags", []))
        if not requires:
            continue

        # A gate is satisfiable if ANY arc provides all required state flags
        satisfiable = any(requires <= sfs for sfs in arc_state_flags.values())
        if not satisfiable:
            paradoxical.append(f"{choice_id} requires {sorted(requires)}")

    if not paradoxical:
        return ValidationCheck(
            name="gate_co_satisfiability",
            severity="pass",
            message="All gates co-satisfiable",
        )
    return ValidationCheck(
        name="gate_co_satisfiability",
        severity="fail",
        message=f"Paradoxical gates ({len(paradoxical)}): {', '.join(paradoxical[:3])}",
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
    exempt = build_exempt_passages(graph, passages)
    starts = find_start_passages(graph, passages)

    violations = walk_linear_stretches(starts, adjacency, outgoing_count, exempt, max_run)

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


def check_state_flag_gate_coverage(graph: Graph) -> ValidationCheck:
    """Check that every state flag is consumed by a gate or overlay condition.

    Implements the "Residue Must Be Read" invariant: checks that each
    state flag appears in at least one ``choice.requires_state_flags`` gate or
    ``overlay.when`` condition.
    """
    state_flag_nodes = graph.get_nodes_by_type("state_flag")
    if not state_flag_nodes:
        return ValidationCheck(
            name="state_flag_gate_coverage",
            severity="pass",
            message="No state flags in graph",
        )

    choice_nodes = graph.get_nodes_by_type("choice")
    consumed: set[str] = set()
    for choice_data in choice_nodes.values():
        consumed.update(choice_data.get("requires_state_flags") or [])

    # Overlays are embedded arrays on entity nodes (type="entity"),
    # not separate typed nodes.
    consumed.update(
        sf
        for entity_data in graph.get_nodes_by_type("entity").values()
        for overlay in entity_data.get("overlays") or []
        for sf in overlay.get("when") or []
    )

    unconsumed = sorted(set(state_flag_nodes.keys()) - consumed)
    if not unconsumed:
        return ValidationCheck(
            name="state_flag_gate_coverage",
            severity="pass",
            message=f"All {len(state_flag_nodes)} state flag(s) consumed by gates or overlays",
        )
    return ValidationCheck(
        name="state_flag_gate_coverage",
        severity="warn",
        message=(
            f"{len(unconsumed)} of {len(state_flag_nodes)} state flag(s) not consumed "
            f"by any choice.requires_state_flags or overlay.when: {', '.join(unconsumed[:5])}"
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
        forward = [c for c in choices if not c.get("is_return") and not c.get("is_routing")]
        if not forward:
            continue  # ending passage or routing-only — no forward choices
        ungated = [c for c in forward if not c.get("requires_state_flags")]
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


def check_routing_coverage(graph: Graph) -> list[ValidationCheck]:
    """Verify routing choice sets are collectively-exhaustive and mutually-exclusive.

    For each source passage that has ``is_routing=True`` choices, checks:
    - CE: every arc reaching the source has at least one satisfiable route
    - ME: at most one routing choice is satisfiable per arc (excluding fallback)

    Uses arc state flag signatures from ``build_arc_state_flags()`` with scope
    matched to how each routing set was produced:
    - ``ending_split`` passages use ``scope="ending"`` — strict CE required
    - All other passages use ``scope="routing"`` — fallback-lenient CE

    Returns:
        List of ValidationCheck results (one per routing set with issues,
        or a single pass check if all sets are valid).
    """
    from questfoundry.graph.grow_algorithms import build_arc_state_flags
    from questfoundry.graph.grow_routing import get_routing_applied_metadata

    choices = graph.get_nodes_by_type("choice")
    arc_nodes = graph.get_nodes_by_type("arc")

    if not arc_nodes:
        return [
            ValidationCheck(
                name="routing_coverage",
                severity="pass",
                message="No arcs to validate routing against",
            )
        ]

    # Build both state flag scopes; select per-passage based on routing metadata
    ending_split_passages, _residue_passages = get_routing_applied_metadata(graph)
    arc_state_flags_ending = build_arc_state_flags(graph, arc_nodes, scope="ending")
    arc_state_flags_routing = build_arc_state_flags(graph, arc_nodes, scope="routing")

    # Group routing choices by source passage; also detect fallback choices
    routing_sets: dict[str, list[dict[str, object]]] = {}
    # Source passages that also have a non-routing fallback choice
    has_fallback: set[str] = set()
    for _cid, cdata in choices.items():
        source = str(cdata.get("from_passage", ""))
        if cdata.get("is_routing"):
            routing_sets.setdefault(source, []).append(cdata)
        else:
            # A non-routing choice from a source that has routing choices
            # is a fallback — it covers arcs that don't match any variant
            if source:
                has_fallback.add(source)

    if not routing_sets:
        return [
            ValidationCheck(
                name="routing_coverage",
                severity="pass",
                message="No routing choice sets to validate",
            )
        ]

    # Build beat → covering arcs mapping
    beat_to_arcs: dict[str, list[str]] = {}
    for arc_id, adata in arc_nodes.items():
        for beat_id in adata.get("sequence", []):
            beat_to_arcs.setdefault(str(beat_id), []).append(arc_id)

    checks: list[ValidationCheck] = []

    for source_pid, routing_choices in sorted(routing_sets.items()):
        # Find arcs covering this passage's beat
        from_beat = get_primary_beat(graph, source_pid) or ""
        covering_arcs = beat_to_arcs.get(from_beat, [])
        if not covering_arcs:
            continue

        # Extract requires sets from routing choices
        route_requires: list[set[str]] = []
        for rc in routing_choices:
            reqs = rc.get("requires_state_flags", [])
            route_requires.append(set(reqs) if isinstance(reqs, list) else set())

        # Select state flag scope: ending splits use "ending" scope (exhaustive),
        # residue routing uses "routing" scope (best-effort, fallback OK).
        is_ending_split = source_pid in ending_split_passages
        arc_sf = arc_state_flags_ending if is_ending_split else arc_state_flags_routing

        # CE check: for each covering arc, at least one route is satisfiable.
        # Ending splits must be strictly exhaustive — no fallback exemption.
        # Residue routing allows fallback (arcs not matching any variant use
        # the original unmodified choice).
        source_has_fallback = (not is_ending_split) and (source_pid in has_fallback)
        ce_gaps: list[str] = []
        for arc_id in covering_arcs:
            arc_sfs = arc_sf.get(arc_id, frozenset())
            satisfiable = [
                i for i, reqs in enumerate(route_requires) if reqs and reqs.issubset(arc_sfs)
            ]
            if not satisfiable and not source_has_fallback:
                ce_gaps.append(arc_id)

        if ce_gaps:
            checks.append(
                ValidationCheck(
                    name="routing_coverage_ce",
                    severity="fail",
                    message=(
                        f"Routing set at {source_pid}: {len(ce_gaps)} arc(s) have "
                        f"no satisfiable route: {', '.join(ce_gaps[:3])}"
                    ),
                )
            )

        # ME check: at most one route satisfiable per arc
        me_violations: list[str] = []
        for arc_id in covering_arcs:
            arc_sfs = arc_sf.get(arc_id, frozenset())
            satisfiable = [
                i for i, reqs in enumerate(route_requires) if reqs and reqs.issubset(arc_sfs)
            ]
            if len(satisfiable) > 1:
                me_violations.append(arc_id)

        if me_violations:
            checks.append(
                ValidationCheck(
                    name="routing_coverage_me",
                    severity="warn",
                    message=(
                        f"Routing set at {source_pid}: {len(me_violations)} arc(s) "
                        f"satisfy multiple routes: {', '.join(me_violations[:3])}"
                    ),
                )
            )

    if not checks:
        checks.append(
            ValidationCheck(
                name="routing_coverage",
                severity="pass",
                message="All routing choice sets are CE+ME valid",
            )
        )
    return checks


def check_prose_neutrality(graph: Graph) -> list[ValidationCheck]:
    """Validate prose-layer contracts for shared (converged) passages.

    For each passage shared across arcs from different paths of the same
    dilemma, checks whether the dilemma's prose-layer settings require
    variant routing:

    - ``residue_weight: heavy`` or ``ending_salience: high`` without routing
      → fail (prose MUST vary but no mechanism exists)
    - ``residue_weight: light`` without routing → warn (prose MAY vary)
    - ``residue_weight: cosmetic`` / ``ending_salience: none`` → pass

    Returns one check per violation, or a single pass if all contracts hold.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    passage_nodes = graph.get_nodes_by_type("passage")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    if not arc_nodes or not passage_nodes or not dilemma_nodes:
        return [
            ValidationCheck(
                name="prose_neutrality",
                severity="pass",
                message="No arcs/passages/dilemmas to validate prose neutrality",
            )
        ]

    # Build beat → set of arc IDs covering it
    beat_arcs: dict[str, set[str]] = {}
    for arc_id, adata in arc_nodes.items():
        for beat_id in adata.get("sequence", []):
            beat_arcs.setdefault(str(beat_id), set()).add(arc_id)

    path_nodes = graph.get_nodes_by_type("path")

    # Build set of passages that have variant routing applied.
    # Primary source: the routing_applied metadata node written by
    # apply_routing_plan (S3).  Fall back to scanning residue_for on
    # variant passages for graphs that pre-date S3.
    from questfoundry.graph.grow_routing import (
        ROUTING_APPLIED_NODE_ID,
        get_routing_applied_metadata,
    )

    routing_node = graph.get_node(ROUTING_APPLIED_NODE_ID)
    ending_split_pids, residue_pids = get_routing_applied_metadata(graph)
    routed_passages: set[str] = ending_split_pids | residue_pids

    if routing_node is None:
        # Legacy / pre-S3 fallback: scan residue_for on variant passages
        for _pid, _pdata in passage_nodes.items():
            residue_for = _pdata.get("residue_for")
            if residue_for:
                routed_passages.add(str(residue_for))

    checks: list[ValidationCheck] = []

    for pid, pdata in passage_nodes.items():
        # Skip variant passages - they ARE the routing solution, not a problem
        if pdata.get("residue_for"):
            continue

        # Skip endings - they're handled by ending splits, not heavy residue
        if pdata.get("is_ending"):
            continue

        from_beat = get_primary_beat(graph, pid) or ""
        if not from_beat:
            continue

        covering_arcs = beat_arcs.get(from_beat, set())
        if len(covering_arcs) < 2:
            continue  # Not shared

        # Find dilemmas that actually diverge across covering arcs:
        # a dilemma diverges only when covering arcs chose *different* paths
        # for it (i.e., 2+ distinct path IDs for that dilemma).
        # NOTE: This assumes valid arcs contain at most one path per dilemma.
        dilemma_to_paths: dict[str, set[str]] = {}
        for arc_id in covering_arcs:
            for path_raw in arc_nodes.get(arc_id, {}).get("paths", []):
                p_id = normalize_scoped_id(path_raw, "path")
                p_data = path_nodes.get(p_id, {})
                raw_did = p_data.get("dilemma_id", "")
                if not raw_did:
                    continue
                dilemma_id = normalize_scoped_id(raw_did, "dilemma")
                dilemma_to_paths.setdefault(dilemma_id, set()).add(p_id)

        diverging_dilemmas: set[str] = {
            dilemma_id
            for dilemma_id, paths_for_dilemma in dilemma_to_paths.items()
            if len(paths_for_dilemma) > 1
        }

        has_routing = pid in routed_passages

        for dilemma_id in sorted(diverging_dilemmas):
            ddata = dilemma_nodes.get(dilemma_id, {})
            weight = ddata.get("residue_weight", "light")
            salience = ddata.get("ending_salience", "low")
            label = ddata.get("question", dilemma_id)

            if has_routing:
                continue  # Variant routing exists, prose contract met

            if weight == "heavy" or salience == "high":
                # Heavy/high dilemmas require variant routing. With deterministic
                # heavy-residue routing in place, missing variants is a failure.
                checks.append(
                    ValidationCheck(
                        name="prose_neutrality",
                        severity="fail",
                        message=(
                            f"Shared passage {pid} requires variant routing "
                            f"for dilemma '{label}' "
                            f"(residue_weight={weight}, ending_salience={salience})"
                        ),
                    )
                )
            elif weight == "light" and salience == "low":
                checks.append(
                    ValidationCheck(
                        name="prose_neutrality",
                        severity="warn",
                        message=(
                            f"Shared passage {pid} has no variant routing "
                            f"for dilemma '{label}' (residue_weight=light)"
                        ),
                    )
                )
            # cosmetic/none → pass, no check needed

    if not checks:
        checks.append(
            ValidationCheck(
                name="prose_neutrality",
                severity="pass",
                message="All shared passages satisfy prose-layer contracts",
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Passage-layer entry point
# ---------------------------------------------------------------------------


def run_passage_checks(graph: Graph) -> ValidationReport:
    """Run all passage-layer validation checks (POLISH Phase 7).

    Returns a ValidationReport containing passage-layer checks:
    reachability, gates, routing, prose neutrality, arc divergence, etc.
    """
    from questfoundry.graph.grow_validation import compute_linear_threshold

    linear_threshold = compute_linear_threshold(graph)
    checks: list[ValidationCheck] = [
        check_all_passages_reachable(graph),
        check_all_endings_reachable(graph),
        check_gate_satisfiability(graph),
        check_gate_co_satisfiability(graph),
        check_arc_divergence(graph),
        check_max_consecutive_linear(graph, max_run=linear_threshold),
        check_state_flag_gate_coverage(graph),
        check_forward_path_reachability(graph),
    ]
    checks.extend(check_commits_timing(graph))
    checks.extend(check_routing_coverage(graph))
    checks.extend(check_prose_neutrality(graph))
    return ValidationReport(checks=checks)
