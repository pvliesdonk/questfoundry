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
from typing import TYPE_CHECKING, Any

from questfoundry.graph.algorithms import compute_arc_traversals
from questfoundry.graph.context import get_primary_beat, normalize_scoped_id
from questfoundry.graph.fill_context import is_merged_passage
from questfoundry.graph.validation_types import ValidationCheck, ValidationReport

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Re-export types for backward compatibility — many modules import from here.
__all__ = [
    "GrowContractError",
    "ValidationCheck",
    "ValidationReport",
    "build_exempt_passages",
    "build_outgoing_count",
    "build_passage_adjacency",
    "check_dilemma_role_compliance",
    "check_dilemmas_resolved",
    "check_no_cross_dilemma_belongs_to",
    "check_no_dual_on_commit_beat",
    "check_no_pre_commit_intersections",
    "check_passage_dag_cycles",
    "check_single_root_beat",
    "check_single_start",
    "check_spine_arc_exists",
    "compute_linear_threshold",
    "find_max_consecutive_linear",
    "find_start_passages",
    "passages_with_forward_incoming",
    "run_all_checks",
    "run_grow_checks",
    "validate_grow_output",
    "walk_linear_stretches",
]


class GrowContractError(ValueError):
    """Raised when GROW's Stage Output Contract is violated."""


_FORBIDDEN_NODE_TYPES_GROW_EXIT = frozenset({"passage", "choice"})
_ACTION_PHRASE_PATTERNS = (
    "player_",
    "user_chose",
    "_chose_",
    "_chooses_",
    "chose_to_",
)


# ---------------------------------------------------------------------------
# GROW Stage Output Contract validator
# ---------------------------------------------------------------------------


def _check_transition_beats(graph: Graph, errors: list[str]) -> None:
    """Transition beat structure (R-5.1 zero belongs_to + zero dilemma_impacts)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_to_paths.setdefault(edge["from"], []).append(edge["to"])

    for beat_id, beat in sorted(beat_nodes.items()):
        if beat.get("role") != "transition_beat":
            continue
        if beat_to_paths.get(beat_id):
            errors.append(
                f"R-5.1: transition beat {beat_id!r} must have zero "
                f"belongs_to edges, found "
                f"{len(beat_to_paths.get(beat_id, []))}"
            )
        if beat.get("dilemma_impacts"):
            errors.append(f"R-5.1: transition beat {beat_id!r} must have zero dilemma_impacts")


def _check_entity_overlays(graph: Graph, errors: list[str]) -> None:
    """Entity overlay composition (R-6.5, R-6.6)."""
    entity_nodes = graph.get_nodes_by_type("entity")

    for entity_id, entity in sorted(entity_nodes.items()):
        overlays = entity.get("overlays", [])
        if not overlays:
            continue

        for idx, overlay in enumerate(overlays):
            if not isinstance(overlay, dict):
                errors.append(f"R-6.5: entity {entity_id!r} overlay[{idx}] is not a dict")
                continue
            when = overlay.get("when", [])
            details = overlay.get("details")
            if not when:
                errors.append(
                    f"R-6.5: entity {entity_id!r} overlay[{idx}] has empty "
                    "'when' (activation condition missing)"
                )
            if not details:
                errors.append(f"R-6.5: entity {entity_id!r} overlay[{idx}] has empty 'details'")

    # R-6.6: detect per-state entity duplicates (entity_id__state pattern).
    for entity_id in sorted(entity_nodes):
        if "__" in entity_id:
            base = entity_id.rsplit("__", 1)[0]
            if base in entity_nodes:
                errors.append(
                    f"R-6.6: entity {entity_id!r} appears to be a state-variant "
                    f"of {base!r}; overlays must be embedded, not separate nodes"
                )


def _check_state_flags(graph: Graph, errors: list[str]) -> None:
    """State flag derivation + naming (R-6.1, R-6.2, R-6.4)."""
    state_flag_nodes = graph.get_nodes_by_type("state_flag")
    consequence_nodes = graph.get_nodes_by_type("consequence")
    derived_from_edges = graph.get_edges(edge_type="derived_from")

    # R-6.1: every state_flag has ≥1 derived_from edge.
    flag_to_conseqs: dict[str, list[str]] = {}
    for edge in derived_from_edges:
        flag_to_conseqs.setdefault(edge["from"], []).append(edge["to"])

    for flag_id in sorted(state_flag_nodes.keys()):
        if not flag_to_conseqs.get(flag_id):
            errors.append(
                f"R-6.1: state_flag {flag_id!r} has no derived_from edge "
                "(ad-hoc creation forbidden)"
            )

    # R-6.2: state flag names express world state, not player actions.
    for flag_id, flag in sorted(state_flag_nodes.items()):
        name = flag.get("name", "")
        lowered = name.lower()
        for pattern in _ACTION_PHRASE_PATTERNS:
            if pattern in lowered:
                errors.append(
                    f"R-6.2: state_flag {flag_id!r} name {name!r} is "
                    f"action-phrased (contains {pattern!r}); must express "
                    "world state"
                )
                break

    # R-6.4: every Consequence produces at least one State Flag.
    derived_conseqs: set[str] = set()
    for edge in derived_from_edges:
        derived_conseqs.add(edge["to"])
    for conseq_id in sorted(consequence_nodes.keys()):
        if conseq_id not in derived_conseqs:
            errors.append(f"R-6.4: consequence {conseq_id!r} has no derived state_flag")


def _check_intersections(graph: Graph, errors: list[str]) -> None:
    """Intersection Group invariants (R-2.3 ≥2 different dilemmas, R-2.4 no same-dilemma pre-commit).

    Extends existing _check_intersection_group_paths.
    Accepts both intersection-edge membership and legacy beat_ids field.
    """
    group_nodes = graph.get_nodes_by_type("intersection_group")
    intersection_edges = graph.get_edges(edge_type="intersection")
    beat_nodes = graph.get_nodes_by_type("beat")
    path_nodes = graph.get_nodes_by_type("path")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")

    beat_to_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_to_paths.setdefault(edge["from"], []).append(edge["to"])
    path_dilemma = {pid: path.get("dilemma_id", "") for pid, path in path_nodes.items()}

    members_by_group: dict[str, list[str]] = {}
    for edge in intersection_edges:
        members_by_group.setdefault(edge["to"], []).append(edge["from"])
    for group_id, group in group_nodes.items():
        legacy_members = group.get("beat_ids", [])
        if legacy_members:
            members_by_group.setdefault(group_id, []).extend(legacy_members)

    for group_id in sorted(group_nodes.keys()):
        members = sorted(set(members_by_group.get(group_id, [])))
        if len(members) < 2:
            errors.append(
                f"R-2.3: intersection_group {group_id!r} has {len(members)} member(s); must be ≥2"
            )
            continue

        # R-2.3: ≥2 different dilemmas across members.
        beat_dilemmas: dict[str, set[str]] = {}
        for m in members:
            dilemmas: set[str] = set()
            for p in beat_to_paths.get(m, []):
                d = path_dilemma.get(p, "")
                if d:
                    dilemmas.add(d)
            beat_dilemmas[m] = dilemmas

        all_dilemmas: set[str] = set()
        for d_set in beat_dilemmas.values():
            all_dilemmas |= d_set
        if len(all_dilemmas) < 2:
            errors.append(
                f"R-2.3: intersection_group {group_id!r} contains beats from "
                f"only {len(all_dilemmas)} dilemma(s); need ≥2 "
                f"(members: {members})"
            )

        # R-2.4: no two pre-commit beats of the same dilemma.
        pre_commit_by_dilemma: dict[str, list[str]] = {}
        for m in members:
            beat = beat_nodes.get(m, {})
            paths = beat_to_paths.get(m, [])
            impacts = beat.get("dilemma_impacts", [])
            has_commits = any(i.get("effect") == "commits" for i in impacts)
            if len(paths) >= 2 and not has_commits:
                for d in beat_dilemmas.get(m, set()):
                    pre_commit_by_dilemma.setdefault(d, []).append(m)
        for d, beats in sorted(pre_commit_by_dilemma.items()):
            if len(beats) >= 2:
                errors.append(
                    f"R-2.4: intersection_group {group_id!r} contains "
                    f"{len(beats)} pre-commit beats of dilemma {d!r}: "
                    f"{sorted(beats)}"
                )


def _check_beat_dag(graph: Graph, errors: list[str]) -> None:
    """Y-fork postcondition (R-1.4).

    The last shared pre-commit beat of each dilemma must have one
    commit-beat successor per explored path of that dilemma.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # Build beat → belongs_to paths lookup.
    beat_to_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_to_paths.setdefault(edge["from"], []).append(edge["to"])

    # Build beat → successors lookup.
    # predecessor edge: from=B to=A means A is predecessor of B;
    # so successors-of-A are B (=edge["from"]).
    successors_by_beat: dict[str, list[str]] = {}
    for edge in predecessor_edges:
        successors_by_beat.setdefault(edge["to"], []).append(edge["from"])

    # Identify pre-commit beats: multi-belongs_to + no commits impact.
    pre_commit_beats: dict[str, list[str]] = {}
    for beat_id, beat in beat_nodes.items():
        paths = beat_to_paths.get(beat_id, [])
        impacts = beat.get("dilemma_impacts", [])
        has_commits = any(i.get("effect") == "commits" for i in impacts)
        if len(paths) >= 2 and not has_commits:
            pre_commit_beats[beat_id] = sorted(paths)

    # For each pre-commit beat that is a Y-fork tip (has at least one
    # commit-beat successor), verify its commit successors cover all paths.
    for beat_id, paths in sorted(pre_commit_beats.items()):
        successors = successors_by_beat.get(beat_id, [])
        commit_successor_paths: set[str] = set()
        for s in successors:
            s_beat = beat_nodes.get(s, {})
            s_paths = beat_to_paths.get(s, [])
            s_impacts = s_beat.get("dilemma_impacts", [])
            s_has_commits = any(i.get("effect") == "commits" for i in s_impacts)
            if s_has_commits and len(s_paths) == 1:
                commit_successor_paths.update(s_paths)

        # Only check if this beat is a Y-fork tip.
        if commit_successor_paths:
            missing = set(paths) - commit_successor_paths
            if missing:
                errors.append(
                    f"R-1.4: Y-fork postcondition — pre-commit beat "
                    f"{beat_id!r} has commit successors for paths "
                    f"{sorted(commit_successor_paths)} but is missing "
                    f"commit beats for path(s) {sorted(missing)}"
                )


def _check_upstream_contract(graph: Graph, errors: list[str]) -> None:
    """Delegate to SEED validator with skip_forbidden_types=True."""
    from questfoundry.graph.seed_validation import validate_seed_output

    upstream = validate_seed_output(graph, skip_forbidden_types=True)
    for e in upstream:
        errors.append(f"Output-0: SEED contract violated post-GROW — {e}")


def validate_grow_output(graph: Graph) -> list[str]:
    """Verify GROW's output meets POLISH's input contract.

    Args:
        graph: Graph containing GROW stage output.

    Returns:
        List of error strings. Empty means valid.
    """
    errors: list[str] = []
    _check_upstream_contract(graph, errors)
    _check_beat_dag(graph, errors)
    _check_intersections(graph, errors)
    _check_state_flags(graph, errors)
    _check_entity_overlays(graph, errors)
    _check_transition_beats(graph, errors)

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

    # 3. Every beat has at least one belongs_to edge.
    # Pre-commit beats may have multiple belongs_to edges (Y-shape: one edge per path
    # of their dilemma). Guard rails for dual-belongs_to correctness are enforced by
    # check_no_cross_dilemma_belongs_to and check_no_dual_on_commit_beat in
    # grow_validation.py.
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
            # Zero-belongs_to beats (transition beats, gap beats) are DAG
            # infrastructure — they don't belong to any dilemma's Y-shape.
            # See Story Graph Ontology Part 8 "Zero-belongs_to beats".
            beat_data = beat_nodes[beat_id]
            if beat_data.get("role") in ("transition_beat", "gap_beat"):
                continue
            errors.append(f"Beat {beat_id} has no belongs_to edge (no path assignment)")

    # 4. State flag nodes exist for explored dilemmas
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    explored_dilemmas = {
        did
        for did, ddata in dilemma_nodes.items()
        # GROW may omit status for dilemmas it fully explored; treat None as "explored".
        if ddata.get("status") in ("explored", None)
    }
    # Build consequence → state_flag map via derived_from edges (Story Graph Ontology, Part 9).
    # state_flag nodes have no dilemma_id field; the association is:
    #   state_flag --derived_from--> consequence <--has_consequence-- path.dilemma_id
    _flagged_consequences: set[str] = {
        edge["to"] for edge in graph.get_edges(edge_type="derived_from")
    }
    # Build consequence → dilemma lookup via has_consequence edges + path.dilemma_id
    _path_nodes = graph.get_nodes_by_type("path")
    _cons_to_dilemma: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="has_consequence"):
        path_data = _path_nodes.get(edge["from"], {})
        raw_did = path_data.get("dilemma_id", "")
        if raw_did:
            linked_did = normalize_scoped_id(raw_did, "dilemma")
            _cons_to_dilemma[edge["to"]] = linked_did

    dilemmas_with_flags: set[str] = set()
    for cons_id in _flagged_consequences:
        linked_dilemma = _cons_to_dilemma.get(cons_id)
        if linked_dilemma:
            dilemmas_with_flags.add(linked_dilemma)

    for dilemma_id in explored_dilemmas:
        if dilemma_id not in dilemmas_with_flags:
            errors.append(f"Explored dilemma {dilemma_id} has no state flag nodes")

    # 5. Dilemma nodes have dilemma_role set
    for dilemma_id, dilemma_data in dilemma_nodes.items():
        if not dilemma_data.get("dilemma_role"):
            errors.append(f"Dilemma {dilemma_id} missing dilemma_role (hard/soft)")

    # 6. Every computed arc traversal is complete (no dead ends)
    _check_arc_traversal_completeness(graph, beat_nodes, errors)

    # 7. Intersection groups reference beats from different paths only
    intersection_groups = graph.get_nodes_by_type("intersection_group")
    for group_id, group_data in intersection_groups.items():
        _check_intersection_group_paths(
            graph, group_id, group_data, beat_nodes, beats_with_path, errors
        )

    return errors


def _check_arc_traversal_completeness(
    graph: Graph,
    beat_nodes: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    """Verify every arc traversal terminates at a beat with no arc-internal successors.

    A "dead end" is a beat in the middle of an arc that has successors in the
    full beat DAG but none within the arc's own beat set. This indicates GROW
    produced an incomplete arc where beats belonging to the arc's paths were
    not connected forward.

    Args:
        graph: Graph containing the beat DAG.
        beat_nodes: Dict of beat node ID → data (pre-fetched).
        errors: Mutable list to append error strings to.
    """
    arc_traversals = compute_arc_traversals(graph)
    if not arc_traversals:
        return  # No dilemmas/paths — skip

    # Build full children adjacency from predecessor edges
    # predecessor edge: from=child, to=parent  →  children[parent].append(child)
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    children_all: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            children_all[to_id].append(from_id)

    for arc_key, beat_sequence in arc_traversals.items():
        arc_beat_set = set(beat_sequence)

        for beat_id in beat_sequence:
            beat_children = children_all.get(beat_id, [])
            children_in_arc = [c for c in beat_children if c in arc_beat_set]

            # A dead end: beat has successors globally but none within this arc
            if not children_in_arc and beat_children:
                errors.append(
                    f"Arc '{arc_key}' has dead-end beat {beat_id!r}: "
                    f"beat has successors {beat_children} outside the arc "
                    f"but no successors within the arc's beat set"
                )


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
    # Intersection group nodes store beat IDs in beat_ids field
    beat_ids = group_data.get("beat_ids", [])
    if not beat_ids:
        errors.append(f"Intersection group {group_id} has empty beat_ids")
        return

    paths_seen: set[str] = set()
    for beat_id in beat_ids:
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


def check_single_root_beat(graph: Graph) -> ValidationCheck:
    """Verify the beat DAG has exactly one root beat.

    A root beat is a beat with no prerequisites — it does not appear as
    ``from_id`` in any predecessor edge. The beat DAG must have exactly one
    root for POLISH to produce a single start passage.

    Synthetic beat roles (micro_beat, residue_beat, sidetrack_beat) are
    excluded since they are created by POLISH, not GROW.

    Args:
        graph: The story graph to validate.

    Returns:
        A ValidationCheck with severity "pass" if exactly one root beat
        exists, or "fail" if zero or multiple roots are found.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return ValidationCheck(
            name="single_root_beat",
            severity="pass",
            message="No beats to check",
        )

    # Collect all beats that appear as from_id in predecessor edges.
    # predecessor(from_id, to_id) means from_id requires to_id as a
    # prerequisite — so from_id comes AFTER to_id in the narrative.
    # A root beat is one that never appears as from_id (no prerequisites).
    beats_with_prereqs: set[str] = set()
    for edge in graph.get_edges(edge_type="predecessor"):
        if edge["from"] in beat_nodes:
            beats_with_prereqs.add(edge["from"])

    # Exclude synthetic beat roles that are created by POLISH, not GROW.
    # At GROW Phase 10 none exist, but run_all_checks() (used by qf inspect)
    # may run after POLISH when synthetic beats are present.
    synthetic_roles = {"micro_beat", "residue_beat", "sidetrack_beat"}
    root_beats = sorted(
        bid
        for bid in beat_nodes
        if bid not in beats_with_prereqs and beat_nodes[bid].get("role", "") not in synthetic_roles
    )

    if len(root_beats) == 1:
        return ValidationCheck(
            name="single_root_beat",
            severity="pass",
            message=f"Single root beat: {root_beats[0]}",
        )
    if len(root_beats) == 0:
        return ValidationCheck(
            name="single_root_beat",
            severity="fail",
            message="No root beat found (all beats have predecessors — possible cycle)",
        )
    return ValidationCheck(
        name="single_root_beat",
        severity="fail",
        message=f"Multiple root beats found ({len(root_beats)}): {', '.join(root_beats)}",
    )


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
# Y-shape guard rail checks (Story Graph Ontology §8)
# ---------------------------------------------------------------------------


def check_no_cross_dilemma_belongs_to(graph: Graph) -> ValidationCheck:
    """Guard rail 1: dual belongs_to must reference paths of the same dilemma.

    Cross-dilemma dual belongs_to is forbidden (Story Graph Ontology §8
    "Path Membership ≠ Scene Participation").  Transition beats and other
    zero-belongs_to DAG infrastructure have no belongs_to edges at all
    and are therefore not checked here.
    """
    path_nodes = graph.get_nodes_by_type("path")
    beat_nodes = graph.get_nodes_by_type("beat")

    # path → dilemma map (normalized)
    path_to_dilemma: dict[str, str] = {}
    for pid, pdata in path_nodes.items():
        did = pdata.get("dilemma_id")
        if did:
            path_to_dilemma[pid] = normalize_scoped_id(did, "dilemma")

    # beat → set of paths
    beat_paths: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            beat_paths.setdefault(e["from"], set()).add(e["to"])

    violations: list[str] = []
    for beat_id, paths in beat_paths.items():
        if len(paths) < 2:
            continue
        dilemmas = {path_to_dilemma.get(p) for p in paths}
        dilemmas.discard(None)  # paths without dilemma_id are caught by earlier checks
        if len(dilemmas) > 1:
            violations.append(
                f"{beat_id} -> {sorted(paths)} across dilemmas {sorted(d for d in dilemmas if d)}"
            )

    dual_count = sum(1 for p in beat_paths.values() if len(p) >= 2)
    if not violations:
        return ValidationCheck(
            name="no_cross_dilemma_belongs_to",
            severity="pass",
            message=f"All {dual_count} dual-belongs_to beats are same-dilemma",
        )
    return ValidationCheck(
        name="no_cross_dilemma_belongs_to",
        severity="fail",
        message=f"cross-dilemma dual belongs_to (guard rail 1): {'; '.join(violations[:3])}",
    )


def check_no_dual_on_commit_beat(graph: Graph) -> ValidationCheck:
    """Guard rail 2: commit beats must have a single belongs_to.

    A commit beat is the first beat exclusive to its path; dual membership
    on a commit beat is structurally impossible (Story Graph Ontology §8).
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    beat_paths: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            beat_paths.setdefault(e["from"], set()).add(e["to"])

    violations: list[str] = []
    for bid, pset in beat_paths.items():
        if len(pset) < 2:
            continue
        impacts = beat_nodes[bid].get("dilemma_impacts", [])
        if any(imp.get("effect") == "commits" for imp in impacts):
            violations.append(f"{bid} commits AND has {len(pset)} belongs_to edges: {sorted(pset)}")

    if not violations:
        return ValidationCheck(
            name="no_dual_on_commit_beat",
            severity="pass",
            message="No commit beats with dual belongs_to",
        )
    return ValidationCheck(
        name="no_dual_on_commit_beat",
        severity="fail",
        message=f"commit beat with dual belongs_to (guard rail 2): {'; '.join(violations[:3])}",
    )


def check_no_pre_commit_intersections(graph: Graph) -> ValidationCheck:
    """Guard rail 3: intersection groups must not contain two pre-commit
    beats with identical dual ``belongs_to`` path sets (same dilemma).

    Implementation note: under the binary Y-shape assumption (guard rail 1),
    two pre-commit beats belong to the same dilemma if and only if they have
    identical `belongs_to` path frozensets. This check enforces that
    constraint by checking for beat pairs that share the same path set
    within an intersection group.

    Rationale: two pre-commit beats of the same dilemma have identical dual
    ``belongs_to`` path sets and are sequentially ordered in the dilemma's
    pre-commit chain. An intersection group implies simultaneity, contradicting
    that chain ordering. Cross-dilemma pre-commit co-occurrence is not affected.
    See ``docs/design/story-graph-ontology.md`` Part 8 guard rail 3 for the ruling.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    group_nodes = graph.get_nodes_by_type("intersection_group")

    raw_beat_paths: dict[str, set[str]] = {}
    for e in graph.get_edges(edge_type="belongs_to"):
        if e["from"] in beat_nodes:
            raw_beat_paths.setdefault(e["from"], set()).add(e["to"])
    beat_paths: dict[str, frozenset[str]] = {b: frozenset(p) for b, p in raw_beat_paths.items()}

    violations: list[str] = []
    for gid, gdata in group_nodes.items():
        beat_ids = gdata.get("beat_ids", [])
        dual_by_pathset: dict[frozenset[str], list[str]] = {}
        for bid in beat_ids:
            pset = beat_paths.get(bid, frozenset())
            if len(pset) < 2:
                continue
            dual_by_pathset.setdefault(pset, []).append(bid)
        for pset, bids in dual_by_pathset.items():
            if len(bids) >= 2:
                violations.append(
                    f"{gid} contains {len(bids)} pre-commit beats sharing paths "
                    f"{sorted(pset)}: {', '.join(bids)}"
                )

    if not violations:
        return ValidationCheck(
            name="no_pre_commit_intersections",
            severity="pass",
            message="No intersection groups with pre-commit collisions",
        )
    return ValidationCheck(
        name="no_pre_commit_intersections",
        severity="fail",
        message=f"pre-commit beats grouped in intersection (guard rail 3): {'; '.join(violations[:3])}",
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_grow_checks(graph: Graph) -> ValidationReport:
    """Run beat-DAG validation checks only (GROW Phase 10).

    Returns a ValidationReport containing structural beat-DAG checks.
    The predecessor DAG cycle check runs first; if a cycle is found, the
    report contains a single fail check and no further checks are run.
    """
    from questfoundry.graph.invariants import (
        PipelineInvariantError,
        assert_predecessor_dag_acyclic,
    )

    try:
        assert_predecessor_dag_acyclic(graph, "validation")
    except PipelineInvariantError as exc:
        return ValidationReport(
            checks=[
                ValidationCheck(
                    name="predecessor_dag_acyclic",
                    severity="fail",
                    message=str(exc),
                )
            ]
        )

    checks: list[ValidationCheck] = [
        check_single_root_beat(graph),
        check_single_start(graph),
        check_passage_dag_cycles(graph),
        check_spine_arc_exists(graph),
        check_dilemmas_resolved(graph),
        check_no_cross_dilemma_belongs_to(graph),
        check_no_dual_on_commit_beat(graph),
        check_no_pre_commit_intersections(graph),
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
