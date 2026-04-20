"""Tests for POLISH entry contract validation."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import validate_grow_output


def _make_valid_grow_graph() -> Graph:
    """Create a minimal valid GROW output graph for testing.

    Layered over the tightened DREAM + BRAINSTORM + SEED upstream contract so
    `validate_grow_output` passes cleanly on the baseline.  Existing tests
    mutate this baseline to exercise specific failure modes.

    Structure:
    - DREAM vision node.
    - BRAINSTORM: 2 character + 2 location entities with name/category/
      disposition, 1 hard dilemma `courage_or_caution` with why_it_matters,
      2 answers (1 canonical) with descriptions, anchored_to an entity.
    - SEED: 2 paths (brave = canonical, cautious = alt), each with a
      consequence (with ripples), Y-shape beats (shared pre-commit `intro`,
      per-path commit beats, 2 post-commit beats per path), seed_freeze node.
    - GROW: predecessor edges, state flags per consequence.
    """
    graph = Graph.empty()

    # DREAM
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["duty and doubt"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )

    # BRAINSTORM — entities (2 characters + 2 locations, disposition retained)
    for entity_id, cat, name in [
        ("character::hero", "character", "Hero"),
        ("character::foe", "character", "Foe"),
        ("location::crossroads", "location", "Crossroads"),
        ("location::field", "location", "Field"),
    ]:
        graph.create_node(
            entity_id,
            {
                "type": "entity",
                "raw_id": entity_id.split("::", 1)[-1],
                "name": name,
                "category": cat,
                "concept": "x",
                "disposition": "retained",
            },
        )

    # BRAINSTORM — dilemma
    graph.create_node(
        "dilemma::courage_or_caution",
        {
            "type": "dilemma",
            "raw_id": "courage_or_caution",
            "question": "Fight or flee?",
            "why_it_matters": "the hero's response decides whether the village stands",
            "dilemma_role": "hard",
            "residue_weight": "light",
            "ending_salience": "high",
            "status": "explored",
        },
    )
    graph.add_edge("anchored_to", "dilemma::courage_or_caution", "character::hero")

    # BRAINSTORM — answers
    for ans, is_canon, desc in [
        ("brave", True, "Fight bravely"),
        ("cautious", False, "Flee to regroup"),
    ]:
        ans_id = f"dilemma::courage_or_caution::alt::{ans}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": desc,
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        graph.add_edge("has_answer", "dilemma::courage_or_caution", ans_id)

    # SEED — paths + consequences
    for ans, is_canon in [("brave", True), ("cautious", False)]:
        path_id = f"path::{ans}"
        graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": ans,
                "label": f"The {ans.title()} Path",
                "dilemma_id": "dilemma::courage_or_caution",
                "is_canonical": is_canon,
            },
        )
        graph.add_edge("explores", path_id, f"dilemma::courage_or_caution::alt::{ans}")
        conseq_id = f"consequence::{ans}_outcome"
        graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"{ans}_outcome",
                "description": f"The hero acts {ans}ly",
                "ripples": [f"village reacts to {ans} choice"],
            },
        )
        graph.add_edge("has_consequence", path_id, conseq_id)

    # SEED — Y-shape beats: shared pre-commit `intro`, per-path commit +
    # 2 post-commit beats each.
    graph.create_node(
        "beat::intro",
        {
            "type": "beat",
            "raw_id": "intro",
            "summary": "The hero arrives at the crossroads",
            "entities": ["character::hero", "location::crossroads"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::courage_or_caution", "effect": "reveals"}],
        },
    )
    graph.add_edge("belongs_to", "beat::intro", "path::brave")
    graph.add_edge("belongs_to", "beat::intro", "path::cautious")

    # Commit beats — `fight` on the brave path, `flee` on the cautious path.
    # Named this way because many existing tests mutate `beat::fight`.
    for ans, commit_raw in [("brave", "fight"), ("cautious", "flee")]:
        commit_id = f"beat::{commit_raw}"
        graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": commit_raw,
                "summary": f"The hero {commit_raw}s at the crossroads",
                "entities": ["character::hero", "character::foe"],
                "dilemma_impacts": [
                    {"dilemma_id": "dilemma::courage_or_caution", "effect": "commits"}
                ],
            },
        )
        graph.add_edge("belongs_to", commit_id, f"path::{ans}")
        graph.add_edge("predecessor", commit_id, "beat::intro")
        for i in range(1, 3):
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit {i} on {ans}",
                    "entities": ["character::hero"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, f"path::{ans}")
            prev = commit_id if i == 1 else f"beat::post_{ans}_{i - 1:02d}"
            graph.add_edge("predecessor", post_id, prev)

    # SEED — Path Freeze approval
    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})

    # GROW — the canonical fight beat stays as an alias for test compatibility:
    # older tests remove `beat::fight` or mutate it.  We keep commit_brave as the
    # real commit beat; tests that explicitly reference `beat::fight` build it
    # themselves via mutations (see existing fixtures).  Per-path state flags
    # derived from each consequence.
    for ans in ["brave", "cautious"]:
        flag_id = f"state_flag::{ans}_committed"
        graph.create_node(
            flag_id,
            {
                "type": "state_flag",
                "raw_id": f"{ans}_committed",
                "derived_from": f"consequence::{ans}_outcome",
                "flag_type": "granted",
            },
        )
        graph.add_edge("derived_from", flag_id, f"consequence::{ans}_outcome")

    return graph


class TestValidateGrowOutput:
    """Tests for validate_grow_output."""

    def test_valid_graph_passes(self) -> None:
        """A properly constructed GROW graph passes validation."""
        graph = _make_valid_grow_graph()
        errors = validate_grow_output(graph)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_no_beats_fails(self) -> None:
        """Empty graph fails validation."""
        graph = Graph.empty()
        errors = validate_grow_output(graph)
        assert any("No beat nodes" in e for e in errors)

    def test_beat_missing_summary(self) -> None:
        """Beat without summary fails validation."""
        graph = _make_valid_grow_graph()
        # Remove summary from one beat by updating with empty summary
        graph.update_node("beat::intro", summary="")

        errors = validate_grow_output(graph)
        assert any("beat::intro" in e and "summary" in e for e in errors)

    def test_beat_missing_dilemma_impacts(self) -> None:
        """Beat without dilemma_impacts fails validation."""
        graph = _make_valid_grow_graph()
        # Get current node, remove dilemma_impacts, recreate
        node = graph.get_node("beat::intro")
        assert node is not None
        graph.delete_node("beat::intro", cascade=True)
        node.pop("dilemma_impacts", None)
        graph.create_node("beat::intro", node)
        # Re-add edges
        graph.add_edge("belongs_to", "beat::intro", "path::brave")
        graph.add_edge("predecessor", "beat::fight", "beat::intro")

        errors = validate_grow_output(graph)
        assert any("beat::intro" in e and "dilemma_impacts" in e for e in errors)

    def test_beat_missing_belongs_to(self) -> None:
        """Beat without belongs_to edge fails validation."""
        graph = Graph.empty()
        graph.create_node(
            "beat::orphan",
            {
                "type": "beat",
                "raw_id": "orphan",
                "summary": "An orphan beat",
                "dilemma_impacts": [],
            },
        )

        errors = validate_grow_output(graph)
        assert any("beat::orphan" in e and "belongs_to" in e for e in errors)

    def test_beat_multiple_belongs_to(self) -> None:
        """Beat with multiple belongs_to edges is legal under Y-shape (pre-commit beats share paths)."""
        graph = _make_valid_grow_graph()
        # Create second path and add a second belongs_to edge (simulates a pre-commit beat)
        graph.create_node(
            "path::coward",
            {"type": "path", "raw_id": "coward", "label": "Coward Path"},
        )
        graph.add_edge("belongs_to", "beat::intro", "path::coward")

        errors = validate_grow_output(graph)
        multi_errors = [e for e in errors if "beat::intro" in e and "belongs_to" in e]
        assert not multi_errors, (
            f"Multiple belongs_to should be legal under Y-shape, got errors: {multi_errors}"
        )

    def test_dilemma_missing_role(self) -> None:
        """Dilemma without dilemma_role fails validation."""
        graph = _make_valid_grow_graph()
        graph.update_node("dilemma::courage_or_caution", dilemma_role=None)

        errors = validate_grow_output(graph)
        assert any("dilemma_role" in e for e in errors)

    def test_explored_dilemma_missing_state_flags(self) -> None:
        """Explored dilemma without state flags fails validation."""
        graph = _make_valid_grow_graph()
        # Remove both per-path state flag nodes (cascade strips derived_from edges).
        graph.delete_node("state_flag::brave_committed", cascade=True)
        graph.delete_node("state_flag::cautious_committed", cascade=True)

        errors = validate_grow_output(graph)
        assert any("state flag" in e.lower() for e in errors)

    def test_predecessor_cycle_detected(self) -> None:
        """Cycle in predecessor DAG fails validation."""
        graph = _make_valid_grow_graph()
        # Create a cycle: intro -> fight -> intro
        graph.add_edge("predecessor", "beat::intro", "beat::fight")

        errors = validate_grow_output(graph)
        assert any("cycle" in e.lower() for e in errors)

    def test_intersection_group_same_path_fails(self) -> None:
        """Intersection group with beats from the same path fails."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "beat_ids": ["beat::intro", "beat::fight"],
            },
        )

        errors = validate_grow_output(graph)
        assert any("same path" in e.lower() for e in errors)

    def test_intersection_group_empty_beat_ids_fails(self) -> None:
        """Intersection group with empty beat_ids fails validation."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig_empty",
            {
                "type": "intersection_group",
                "raw_id": "ig_empty",
                "beat_ids": [],
            },
        )

        errors = validate_grow_output(graph)
        assert any("ig_empty" in e and "empty beat_ids" in e for e in errors)

    def test_intersection_group_missing_beat_ids_fails(self) -> None:
        """Intersection group with no beat_ids field (treats as empty) fails validation."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig_missing",
            {
                "type": "intersection_group",
                "raw_id": "ig_missing",
                # beat_ids field intentionally absent
            },
        )

        errors = validate_grow_output(graph)
        assert any("ig_missing" in e and "empty beat_ids" in e for e in errors)

    def test_state_flag_found_via_edge_traversal(self) -> None:
        """State flags are found by traversing derived_from edges, not dilemma_id field.

        Verifies the fix for #1171: validation must traverse
        state_flag --derived_from--> consequence <--has_consequence-- path.dilemma_id
        rather than reading a (non-existent) dilemma_id field on state_flag nodes.
        """
        graph = _make_valid_grow_graph()
        errors = validate_grow_output(graph)
        # No state-flag related errors — edge traversal finds the flag
        flag_errors = [e for e in errors if "state flag" in e.lower()]
        assert flag_errors == [], f"Unexpected state-flag errors: {flag_errors}"

    def test_explored_dilemma_no_consequence_chain_reported_missing(self) -> None:
        """Explored dilemma with state_flag but no has_consequence edge is reported missing.

        If every path→consequence→state_flag chain for a dilemma is broken,
        the dilemma should be reported as having no state flags.  (A single
        broken path is not enough; the sibling path still provides the flag.)
        """
        graph = _make_valid_grow_graph()
        # Remove the has_consequence edge on BOTH paths — breaks every chain.
        graph.remove_edge("has_consequence", "path::brave", "consequence::brave_outcome")
        graph.remove_edge("has_consequence", "path::cautious", "consequence::cautious_outcome")

        errors = validate_grow_output(graph)
        assert any("courage_or_caution" in e and "state flag" in e.lower() for e in errors), (
            f"Expected missing state-flag error for courage_or_caution, got: {errors}"
        )

    def test_unexplored_dilemma_not_checked_for_state_flags(self) -> None:
        """Dilemmas with status='unexplored' are not checked for state flags."""
        graph = _make_valid_grow_graph()
        # Add an unexplored dilemma with no state flags at all
        graph.create_node(
            "dilemma::dormant",
            {
                "type": "dilemma",
                "raw_id": "dormant",
                "dilemma_role": "soft",
                "status": "unexplored",
            },
        )

        errors = validate_grow_output(graph)
        # No error about the unexplored dilemma missing state flags
        flag_errors = [e for e in errors if "dormant" in e and "state flag" in e.lower()]
        assert flag_errors == [], (
            f"Unexplored dilemma should not trigger state-flag check: {flag_errors}"
        )

    # Note: the old `test_intersection_group_different_paths_passes` has
    # moved to `tests/unit/test_grow_validation_contract.py` under R-2.3,
    # which tests intersection-group validity more precisely (beats must
    # come from different *dilemmas*, not just different paths of the
    # same dilemma).  The baseline here is single-dilemma so it cannot
    # legitimately exercise that case.


class TestArcTraversalCompleteness:
    """Tests for Issue #1160: arc traversal completeness check in validate_grow_output."""

    def _make_two_path_graph(self) -> Graph:
        """Create a graph with two paths (two dilemmas) for arc traversal testing."""
        graph = Graph.empty()

        # Two dilemmas, each with paths and consequences (spec-conformant)
        for label in ("choice_a", "choice_b"):
            graph.create_node(
                f"dilemma::{label}",
                {"type": "dilemma", "raw_id": label, "dilemma_role": "hard", "status": "explored"},
            )
            graph.create_node(
                f"consequence::{label}_outcome",
                {"type": "consequence", "raw_id": f"{label}_outcome"},
            )
            graph.create_node(
                f"path::{label}_path",
                {"type": "path", "raw_id": f"{label}_path", "dilemma_id": f"dilemma::{label}"},
            )
            graph.add_edge(
                "has_consequence", f"path::{label}_path", f"consequence::{label}_outcome"
            )
            graph.create_node(
                f"state_flag::{label}_flag",
                {
                    "type": "state_flag",
                    "raw_id": f"{label}_flag",
                    "derived_from": f"consequence::{label}_outcome",
                    "flag_type": "granted",
                },
            )
            graph.add_edge(
                "derived_from",
                f"state_flag::{label}_flag",
                f"consequence::{label}_outcome",
            )

        for path_label in ("brave", "cautious"):
            graph.create_node(
                f"path::{path_label}",
                {"type": "path", "raw_id": path_label, "dilemma_id": "dilemma::choice_a"},
            )

        return graph

    def test_complete_arc_traversal_passes(self) -> None:
        """A well-formed graph with complete arc traversals passes validation (#1160)."""
        graph = _make_valid_grow_graph()
        errors = validate_grow_output(graph)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_arc_with_dead_end_beat_fails(self) -> None:
        """Arc traversal with dead-end beat raises PolishEntryError (#1160).

        Structure: path p has beats b0 → b1 → b2. Beat b2 has a child b_other
        that belongs to a different path, making b2 a dead end within the arc
        for path p (b2 has successors outside the arc but none inside).
        """
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard", "status": "explored"},
        )
        graph.create_node(
            "consequence::d1_outcome",
            {"type": "consequence", "raw_id": "d1_outcome"},
        )
        graph.create_node(
            "path::p_brave",
            {"type": "path", "raw_id": "p_brave", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p_cautious",
            {"type": "path", "raw_id": "p_cautious", "dilemma_id": "dilemma::d1"},
        )
        graph.add_edge("has_consequence", "path::p_brave", "consequence::d1_outcome")
        graph.create_node(
            "state_flag::d1_flag",
            {
                "type": "state_flag",
                "raw_id": "d1_flag",
                "derived_from": "consequence::d1_outcome",
                "flag_type": "granted",
            },
        )
        graph.add_edge("derived_from", "state_flag::d1_flag", "consequence::d1_outcome")

        # b0 and b1 belong to brave; b2 belongs to cautious
        # predecessor chain: b2 → b1 → b0 (all belong to brave)
        # BUT b1 has an extra child b_other that leads outside brave's arc
        for bid in ("b0", "b1", "b2"):
            graph.create_node(
                f"beat::{bid}",
                {"type": "beat", "raw_id": bid, "summary": f"Beat {bid}", "dilemma_impacts": []},
            )
        graph.create_node(
            "beat::b_other",
            {"type": "beat", "raw_id": "b_other", "summary": "Other", "dilemma_impacts": []},
        )

        # b0, b1, b2 all on brave path
        graph.add_edge("belongs_to", "beat::b0", "path::p_brave")
        graph.add_edge("belongs_to", "beat::b1", "path::p_brave")
        graph.add_edge("belongs_to", "beat::b2", "path::p_brave")
        # b_other on cautious path
        graph.add_edge("belongs_to", "beat::b_other", "path::p_cautious")

        # Chain: b1 → b0, b2 → b1, b_other → b1
        # b1 has two children: b2 (in brave arc) and b_other (not in brave arc)
        # This is fine. Let's construct a dead end:
        # b0 → b1, b1 → b2, but b2 also → b_other (b2 has a child outside the arc)
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b_other", "beat::b2")
        # Now: arc for brave = [b0, b1, b2]; b2 has child b_other outside brave arc
        # → b2 is a dead end within brave arc

        errors = validate_grow_output(graph)
        dead_end_errors = [e for e in errors if "dead-end" in e]
        assert dead_end_errors, f"Expected dead-end error, got: {errors}"
        assert "b2" in dead_end_errors[0] or "b_other" in dead_end_errors[0]
