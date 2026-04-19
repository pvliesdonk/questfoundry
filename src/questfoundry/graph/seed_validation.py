"""SEED Stage Output Contract validator.

Validates the graph satisfies every rule in
docs/design/procedures/seed.md §Stage Output Contract.

Called at SEED exit (from apply_seed_mutations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


class SeedContractError(ValueError):
    """Raised when SEED's Stage Output Contract is violated."""


_VALID_DILEMMA_ROLES = frozenset({"hard", "soft"})
_VALID_RESIDUE_WEIGHTS = frozenset({"heavy", "light", "cosmetic"})
_VALID_ENDING_SALIENCES = frozenset({"high", "low", "none"})
_VALID_ORDERING_RELATIONSHIPS = frozenset({"wraps", "concurrent", "serial"})
_VALID_DISPOSITIONS = frozenset({"retained", "cut"})
_FORBIDDEN_NODE_TYPES = frozenset(
    {"passage", "state_flag", "intersection_group", "transition_beat", "choice"}
)
_MAX_ARC_COUNT = 16


def validate_seed_output(graph: Graph) -> list[str]:
    """Verify the graph satisfies SEED's Stage Output Contract.

    Args:
        graph: Graph expected to contain SEED output.

    Returns:
        List of human-readable error strings. Empty means compliant.
        Pure read-only — never mutates the graph.
    """
    errors: list[str] = []
    _check_upstream_contract(graph, errors)
    _check_entities(graph, errors)
    _check_paths_and_consequences(graph, errors)
    _check_beats(graph, errors)
    _check_belongs_to_yshape(graph, errors)
    _check_convergence_and_ordering(graph, errors)
    _check_arc_count_and_approval(graph, errors)
    return errors


def _check_entities(graph: Graph, errors: list[str]) -> None:
    """Phase 1 entity-triage checks (R-1.1, R-1.2, R-1.4)."""
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    retained_location_count = 0
    for entity_id, entity in sorted(entity_nodes.items()):
        disposition = entity.get("disposition")
        if disposition is None:
            errors.append(
                f"R-1.1: entity {entity_id!r} has no disposition (must be 'retained' or 'cut')"
            )
            continue
        if disposition not in _VALID_DISPOSITIONS:
            errors.append(
                f"R-1.1: entity {entity_id!r} has invalid disposition {disposition!r}; "
                f"must be one of {sorted(_VALID_DISPOSITIONS)}"
            )
        if disposition == "retained" and entity.get("category") == "location":
            retained_location_count += 1

    # R-1.2: anchored_to from surviving dilemmas must not point to cut entities.
    for edge in sorted(
        graph.get_edges(edge_type="anchored_to"),
        key=lambda e: (e["from"], e["to"]),
    ):
        dilemma = dilemma_nodes.get(edge["from"])
        if dilemma is None:
            continue
        entity_data = entity_nodes.get(edge["to"])
        if entity_data is None:
            continue
        if entity_data.get("disposition") == "cut":
            errors.append(
                f"R-1.2: dilemma {edge['from']!r} is anchored to cut entity "
                f"{edge['to']!r}; re-anchor or cut the dilemma first"
            )

    if retained_location_count < 2:
        errors.append(
            f"R-1.4: SEED must retain ≥2 location entities, found {retained_location_count}"
        )


def _check_paths_and_consequences(graph: Graph, errors: list[str]) -> None:
    """Phase 3 path structure (R-3.1, R-3.2, R-3.3, R-3.4)."""
    path_nodes = graph.get_nodes_by_type("path")
    answer_nodes = graph.get_nodes_by_type("answer")
    consequence_nodes = graph.get_nodes_by_type("consequence")

    # R-3.1 + R-3.2: each explored answer has exactly one Path via `explores`.
    explores_edges = graph.get_edges(edge_type="explores")
    path_by_answer: dict[str, list[str]] = {}
    for edge in sorted(explores_edges, key=lambda e: (e["from"], e["to"])):
        path_by_answer.setdefault(edge["to"], []).append(edge["from"])

    for answer_id, answer in sorted(answer_nodes.items()):
        if not answer.get("explored"):
            continue
        paths = path_by_answer.get(answer_id, [])
        if len(paths) == 0:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has no path (expected exactly one)"
            )
        elif len(paths) > 1:
            errors.append(
                f"R-3.1: explored answer {answer_id!r} has {len(paths)} paths; "
                f"expected exactly one: {sorted(paths)}"
            )

    for path_id in sorted(path_nodes.keys()):
        if not path_id.startswith("path::"):
            errors.append(f"R-3.2: path id {path_id!r} missing 'path::' prefix")

    # R-3.3: every Path has ≥1 has_consequence edge.
    has_consequence_edges = graph.get_edges(edge_type="has_consequence")
    consequences_per_path: dict[str, list[str]] = {}
    for edge in has_consequence_edges:
        consequences_per_path.setdefault(edge["from"], []).append(edge["to"])

    for path_id in sorted(path_nodes.keys()):
        if not consequences_per_path.get(path_id):
            errors.append(f"R-3.3: path {path_id!r} has no has_consequence edge")

    # R-3.4: every Consequence has non-empty description + ≥1 ripple.
    for conseq_id, conseq in sorted(consequence_nodes.items()):
        if not conseq.get("description"):
            errors.append(f"R-3.4: consequence {conseq_id!r} has empty description")
        ripples = conseq.get("ripples", [])
        if not ripples:
            errors.append(f"R-3.4: consequence {conseq_id!r} has no ripples")


def _check_beats(graph: Graph, errors: list[str]) -> None:
    """Phase 3 beat structural rules (R-3.13, R-3.14, R-3.15)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beats_with_path: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beats_with_path.setdefault(edge["from"], []).append(edge["to"])

    for beat_id, beat in sorted(beat_nodes.items()):
        role = beat.get("role")
        is_structural = role in {"setup", "epilogue"}

        # R-3.13 + R-3.15: every beat has non-empty summary + entities.
        if not beat.get("summary"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty summary")
        if not beat.get("entities"):
            errors.append(f"R-3.13: beat {beat_id!r} has empty entities list")

        # R-3.14: structural beats must have zero belongs_to + zero commits impact.
        if is_structural:
            paths = beats_with_path.get(beat_id, [])
            if paths:
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must have zero belongs_to "
                    f"edges, found {len(paths)}"
                )
            if any(impact.get("effect") == "commits" for impact in beat.get("dilemma_impacts", [])):
                errors.append(
                    f"R-3.14: {role} beat {beat_id!r} must not contain "
                    f"dilemma_impacts.effect: commits"
                )


def _check_belongs_to_yshape(graph: Graph, errors: list[str]) -> None:
    """Y-shape guard rails and commit/post-commit counts (R-3.6 to R-3.12)."""
    beat_nodes = graph.get_nodes_by_type("beat")
    path_nodes = graph.get_nodes_by_type("path")
    answer_nodes = graph.get_nodes_by_type("answer")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_paths: dict[str, list[str]] = {}
    for edge in belongs_to_edges:
        beat_to_paths.setdefault(edge["from"], []).append(edge["to"])

    path_dilemma: dict[str, str] = {
        path_id: path.get("dilemma_id", "") for path_id, path in path_nodes.items()
    }

    commit_beats_per_path: dict[str, list[str]] = {}
    post_beats_per_path: dict[str, list[str]] = {}
    pre_commit_by_dilemma: dict[str, list[str]] = {}

    for beat_id in sorted(beat_nodes.keys()):
        beat = beat_nodes[beat_id]
        role = beat.get("role")
        if role in {"setup", "epilogue"}:
            continue

        paths = beat_to_paths.get(beat_id, [])
        impacts = beat.get("dilemma_impacts", [])
        has_commits_impact = any(imp.get("effect") == "commits" for imp in impacts)

        dilemmas_of_this_beat = {path_dilemma.get(p, "") for p in paths if p in path_nodes}
        dilemmas_of_this_beat.discard("")
        if len(dilemmas_of_this_beat) > 1:
            errors.append(
                f"R-3.9: beat {beat_id!r} has cross-dilemma belongs_to — "
                f"references paths of dilemmas {sorted(dilemmas_of_this_beat)}"
            )

        if has_commits_impact:
            if len(paths) != 1:
                errors.append(
                    f"R-3.7: commit beat {beat_id!r} must have exactly one "
                    f"belongs_to edge, found {len(paths)}"
                )
            for p in paths:
                commit_beats_per_path.setdefault(p, []).append(beat_id)
        elif len(paths) >= 2:
            # Pre-commit: R-3.6
            if len(dilemmas_of_this_beat) != 1:
                errors.append(
                    f"R-3.6: pre-commit beat {beat_id!r} belongs_to edges must "
                    f"reference paths of the same dilemma, got "
                    f"{sorted(dilemmas_of_this_beat)}"
                )
            for d in dilemmas_of_this_beat:
                pre_commit_by_dilemma.setdefault(d, []).append(beat_id)
        elif len(paths) == 1:
            post_beats_per_path.setdefault(paths[0], []).append(beat_id)

    # R-3.11: exactly one commit beat per path.
    for path_id in sorted(path_nodes.keys()):
        commits = commit_beats_per_path.get(path_id, [])
        if len(commits) != 1:
            errors.append(
                f"R-3.11: path {path_id!r} must have exactly one commit beat, "
                f"found {len(commits)}: {sorted(commits)}"
            )

    # R-3.12: 2-4 post-commit beats per path.
    for path_id in sorted(path_nodes.keys()):
        post = post_beats_per_path.get(path_id, [])
        if len(post) < 2 or len(post) > 4:
            errors.append(
                f"R-3.12: path {path_id!r} must have 2-4 post-commit beats, found {len(post)}"
            )

    # R-3.10: every dilemma with 2 explored answers has ≥1 pre-commit beat.
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    for dilemma_id in sorted(dilemma_nodes.keys()):
        explored_answers = [
            edge["to"]
            for edge in has_answer_edges
            if edge["from"] == dilemma_id and answer_nodes.get(edge["to"], {}).get("explored")
        ]
        if len(explored_answers) >= 2 and not pre_commit_by_dilemma.get(dilemma_id):
            errors.append(
                f"R-3.10: dilemma {dilemma_id!r} has {len(explored_answers)} "
                f"explored answers but no pre-commit beats (beats with multiple "
                f"belongs_to edges) -- Y-shape fork missing"
            )


def _check_convergence_and_ordering(graph: Graph, errors: list[str]) -> None:
    """Phase 7 dilemma analysis + Phase 8 ordering (R-7.1 to R-7.3, R-8.3, R-8.4)."""
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    for dilemma_id, dilemma in sorted(dilemma_nodes.items()):
        role = dilemma.get("dilemma_role")
        weight = dilemma.get("residue_weight")
        salience = dilemma.get("ending_salience")
        if role is None:
            errors.append(f"R-7.1: dilemma {dilemma_id!r} missing dilemma_role")
        elif role not in _VALID_DILEMMA_ROLES:
            errors.append(
                f"R-7.1: dilemma {dilemma_id!r} has invalid dilemma_role {role!r}; "
                f"must be one of {sorted(_VALID_DILEMMA_ROLES)}"
            )
        if weight is None:
            errors.append(f"R-7.2: dilemma {dilemma_id!r} missing residue_weight")
        elif weight not in _VALID_RESIDUE_WEIGHTS:
            errors.append(
                f"R-7.2: dilemma {dilemma_id!r} has invalid residue_weight {weight!r}; "
                f"must be one of {sorted(_VALID_RESIDUE_WEIGHTS)}"
            )
        if salience is None:
            errors.append(f"R-7.3: dilemma {dilemma_id!r} missing ending_salience")
        elif salience not in _VALID_ENDING_SALIENCES:
            errors.append(
                f"R-7.3: dilemma {dilemma_id!r} has invalid ending_salience "
                f"{salience!r}; must be one of {sorted(_VALID_ENDING_SALIENCES)}"
            )

    ordering_nodes = graph.get_nodes_by_type("ordering")
    for ord_id, ord_node in sorted(ordering_nodes.items()):
        rel = ord_node.get("relationship")
        if rel not in _VALID_ORDERING_RELATIONSHIPS:
            errors.append(f"R-8.1: ordering {ord_id!r} has invalid relationship {rel!r}")
        a, b = ord_node.get("dilemma_a"), ord_node.get("dilemma_b")
        if rel == "concurrent" and a and b and a > b:
            errors.append(
                f"R-8.3: concurrent ordering {ord_id!r} must have lex-smaller "
                f"dilemma as dilemma_a (got {a!r} > {b!r})"
            )

    shared_entity_edges = graph.get_edges(edge_type="shared_entity")
    if shared_entity_edges:
        errors.append(
            f"R-8.4: shared_entity edges are forbidden (derived from anchored_to, "
            f"not declared); found {len(shared_entity_edges)}"
        )


def _check_arc_count_and_approval(graph: Graph, errors: list[str]) -> None:
    """Phase 5 arc-count guardrail (R-5.1), Phase 6 approval (R-6.4),
    and Stage Output Contract item 16 (forbidden node types)."""
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    answer_nodes = graph.get_nodes_by_type("answer")

    # R-5.1: arc count = 2 ^ (# dilemmas with both answers explored).
    fully_explored_dilemmas = 0
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    answers_by_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_by_dilemma.setdefault(edge["from"], []).append(edge["to"])
    for dilemma_id in dilemma_nodes:
        ans_ids = answers_by_dilemma.get(dilemma_id, [])
        explored = [a_id for a_id in ans_ids if answer_nodes.get(a_id, {}).get("explored")]
        if len(explored) >= 2:
            fully_explored_dilemmas += 1
    arc_count = 2**fully_explored_dilemmas if fully_explored_dilemmas else 1
    if arc_count > _MAX_ARC_COUNT:
        errors.append(
            f"R-5.1: arc count {arc_count} exceeds maximum {_MAX_ARC_COUNT} "
            f"({fully_explored_dilemmas} fully explored dilemmas)"
        )

    # R-6.4: path freeze human approval recorded.
    freeze = graph.get_node("seed_freeze")
    if freeze is None:
        errors.append(
            "R-6.4: SEED Path Freeze approval is not recorded "
            "(expected seed_freeze node with human_approved: True)"
        )
    elif not freeze.get("human_approved"):
        errors.append(
            "R-6.4: SEED Path Freeze is not approved (seed_freeze.human_approved is not True)"
        )

    # Output-16: no forbidden node types.
    for node_type in sorted(_FORBIDDEN_NODE_TYPES):
        forbidden = graph.get_nodes_by_type(node_type)
        if forbidden:
            errors.append(
                f"Output-16: SEED must not create {node_type!r} nodes; "
                f"found {len(forbidden)}: {sorted(forbidden.keys())[:3]}"
            )


def _check_upstream_contract(graph: Graph, errors: list[str]) -> None:
    """Delegate to BRAINSTORM validator (with downstream-node types allowed).

    At SEED time, the graph legitimately contains beat/path/consequence nodes
    that BRAINSTORM's R-3.8 would forbid — skip_forbidden_types=True relaxes
    that one check while preserving all other upstream invariants.
    """
    # Inline import avoids module-load circular dependencies.
    from questfoundry.graph.brainstorm_validation import validate_brainstorm_output

    upstream = validate_brainstorm_output(graph, skip_forbidden_types=True)
    for e in upstream:
        errors.append(f"Output-0: BRAINSTORM contract violated post-SEED — {e}")
