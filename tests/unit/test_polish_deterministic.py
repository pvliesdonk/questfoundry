"""Tests for POLISH Phase 4 deterministic plan computation.

Tests 4a (beat grouping), 4b (feasibility audit),
4c (choice edge derivation), and 4d (false branch candidates).
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.models.polish import PassageSpec
from questfoundry.pipeline.stages.polish._helpers import _PRE_PLAN_WARNINGS_NODE
from questfoundry.pipeline.stages.polish.deterministic import (
    PolishPlan,
    _drain_pre_plan_warnings,
    compute_beat_grouping,
    compute_choice_edges,
    compute_prose_feasibility,
    find_false_branch_candidates,
)
from questfoundry.pipeline.stages.polish.llm_phases import _upsert_pre_plan_warnings


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with defaults."""
    data = {
        "type": "beat",
        "raw_id": beat_id.split("::")[-1],
        "summary": summary,
        "dilemma_impacts": [],
        "entities": [],
        "scene_type": "scene",
    }
    data.update(kwargs)
    graph.create_node(beat_id, data)


def _add_predecessor(graph: Graph, child: str, parent: str) -> None:
    graph.add_edge("predecessor", child, parent)


def _add_belongs_to(graph: Graph, beat_id: str, path_id: str) -> None:
    graph.add_edge("belongs_to", beat_id, path_id)


class TestComputeBeatGrouping:
    """Tests for Phase 4a: beat grouping."""

    def test_singleton_beats(self) -> None:
        """Ungrouped beats become singleton passages."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "First beat")
        _make_beat(graph, "beat::b", "Second beat")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        # No predecessor edges — no linear chain

        specs = compute_beat_grouping(graph)
        assert len(specs) == 2
        assert all(s.grouping_type == "singleton" for s in specs)

    def test_collapse_grouping(self) -> None:
        """Sequential beats in a linear run collapse into one passage.

        Under R-4a.3, multi-beat runs are tagged ``grouping_type="collapse"``
        so Phase 5f's transition-guidance generator continues to fire; single-
        beat passages remain ``"singleton"``.  The collapse rule itself is
        DAG-topology driven — path membership is not consulted.
        """
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Search the study")
        _make_beat(graph, "beat::b", "Find the letter")
        _make_beat(graph, "beat::c", "Read the letter")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_belongs_to(graph, "beat::c", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")

        specs = compute_beat_grouping(graph)
        # All three beats form a maximal linear run → exactly one passage.
        assert len(specs) == 1
        assert set(specs[0].beat_ids) == {"beat::a", "beat::b", "beat::c"}
        assert specs[0].grouping_type == "collapse"

    def test_singleton_beat_keeps_singleton_grouping_type(self) -> None:
        """Single-beat passages stay tagged ``singleton`` so Phase 5f's
        transition-guidance gate does not fire on them (no transition needed
        within a one-beat passage)."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::solo", "Lone beat")
        _add_belongs_to(graph, "beat::solo", "path::p1")

        specs = compute_beat_grouping(graph)
        assert len(specs) == 1
        assert specs[0].grouping_type == "singleton"

    # DELETED: test_intersection_grouping
    # Removed as part of cluster #1311 (maximal-linear-collapse, R-4a.3).
    # Intersection-driven grouping is gone — compute_beat_grouping no longer
    # consumes intersection_group nodes.  Under the new rule, beats from
    # different paths with no predecessor relationship each become their own
    # singleton passage; intersection_group nodes are GROW-internal and are
    # ignored by POLISH.  The two beats in the old fixture (no predecessor edge
    # linking them) now correctly produce two separate singleton passages.
    # See docs/design/procedures/polish.md §R-4a.3, R-4a.4.

    def test_intersection_groups_are_ignored(self) -> None:
        """R-4a.4: intersection_group nodes are GROW-internal and must not affect
        POLISH's passage grouping.  Even a well-formed intersection group yields
        no intersection-typed passage — the DAG topology alone drives grouping."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        _make_beat(graph, "beat::a", "Path A beat")
        _add_belongs_to(graph, "beat::a", "path::pa")

        # Well-formed intersection group (correct beat_ids field) — still ignored.
        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "beat_ids": ["beat::a"],
            },
        )

        specs = compute_beat_grouping(graph)
        # Under R-4a.4 no spec is of intersection type; the beat becomes a
        # singleton passage based on its DAG topology.
        assert all(s.grouping_type != "intersection" for s in specs)
        assert any("beat::a" in s.beat_ids for s in specs)

    def test_cross_path_linear_beats_collapse(self) -> None:
        """R-4a.3: two linearly-adjacent beats collapse into one passage even
        when they belong to different paths.  Path membership is not consulted
        under the new maximal-linear-collapse rule."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        assert len(specs) == 1
        assert set(specs[0].beat_ids) == {"beat::a", "beat::b"}

    def test_all_beats_assigned(self) -> None:
        """Every beat is assigned to exactly one passage."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for i in range(5):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
            _add_belongs_to(graph, f"beat::b{i}", "path::p1")
        for i in range(1, 5):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        specs = compute_beat_grouping(graph)
        all_beats = set()
        for spec in specs:
            for bid in spec.beat_ids:
                assert bid not in all_beats, f"{bid} assigned to multiple passages"
                all_beats.add(bid)

        expected = {f"beat::b{i}" for i in range(5)}
        assert all_beats == expected

    def test_entities_merged(self) -> None:
        """Passage carries the order-preserving union of its member beats' entities.

        Phase 4b's prose-feasibility audit reads ``spec.entities`` to compute
        entity overlap between a passage and the structural flags active there
        (see ``compute_prose_feasibility``).  If this isn't merged across the
        whole run, flags get classified as irrelevant and no variant/residue
        specs are generated.
        """
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "B", entities=["entity::hero", "entity::mentor"])
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        # Maximal-linear-collapse groups both beats into one passage.
        assert len(specs) == 1
        assert set(specs[0].entities) == {"entity::hero", "entity::mentor"}

    def test_empty_graph(self) -> None:
        """Empty graph produces no specs."""
        graph = Graph.empty()
        specs = compute_beat_grouping(graph)
        assert specs == []

    # DELETED: test_zero_belongs_to_beats_do_not_collapse
    # Removed as part of cluster #1311 (maximal-linear-collapse, R-4a.3).
    # The new R-4a.3 rule collapses beats purely by DAG topology — the
    # belongs_to set is NOT consulted for collapse eligibility.  Two adjacent
    # zero-belongs_to beats in a linear DAG WILL collapse into one passage
    # under the new rule.  The old assertion (they must remain separate) is
    # directly contradicted by the new spec.
    # The B2 ruling from issue #1237 that this test locked in is superseded
    # by the cluster #1311 spec update.


class TestComputeProseFeasibility:
    """Tests for Phase 4b: prose feasibility audit."""

    def test_clean_passage(self) -> None:
        """Passage with no relevant flags is clean (no annotations)."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Simple beat")
        _add_belongs_to(graph, "beat::a", "path::p1")

        specs = [PassageSpec(passage_id="p1", beat_ids=["beat::a"], summary="test")]
        result = compute_prose_feasibility(graph, specs)

        assert result["annotations"] == {}
        assert result["variant_specs"] == []
        assert result["residue_specs"] == []

    def test_annotated_passage(self) -> None:
        """Passage with narratively irrelevant flags gets annotations."""
        graph = Graph.empty()
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft", "residue_weight": "light"},
        )

        # Commit beat as ancestor, with grants edge to a state_flag node
        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::target", "Target beat", entities=["entity::hero"])
        _add_belongs_to(graph, "beat::commit", "path::brave")
        _add_belongs_to(graph, "beat::target", "path::brave")
        _add_predecessor(graph, "beat::target", "beat::commit")
        graph.create_node(
            "state_flag::brave_committed",
            {"type": "state_flag", "raw_id": "brave_committed"},
        )
        graph.add_edge("grants", "beat::commit", "state_flag::brave_committed")

        # Overlay for a DIFFERENT entity than the passage's entity (embedded on entity node)
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "overlays": [
                    {
                        "when": ["state_flag::brave_committed"],
                        "details": {"description": "NPC changes"},
                    }
                ],
            },
        )

        specs = [
            PassageSpec(
                passage_id="passage::test",
                beat_ids=["beat::target"],
                summary="test",
                entities=["entity::hero"],
            )
        ]
        result = compute_prose_feasibility(graph, specs)

        # Flag is structurally relevant but entity::npc doesn't overlap entity::hero
        assert "passage::test" in result["annotations"]

    def test_empty_specs(self) -> None:
        """Empty specs produce empty results."""
        graph = Graph.empty()
        result = compute_prose_feasibility(graph, [])
        assert result["variant_specs"] == []
        assert result["residue_specs"] == []


def _setup_dilemma(graph: Graph, dilemma_id: str, path_ids: list[str]) -> None:
    """Helper to create a dilemma node and set dilemma_id on path nodes.

    Stores the prefixed dilemma_id (e.g. "dilemma::d1") on path nodes to
    match production data shape from mutations.py:apply_seed_output().
    """
    graph.create_node(dilemma_id, {"type": "dilemma", "raw_id": dilemma_id.split("::")[-1]})
    for pid in path_ids:
        graph.update_node(pid, dilemma_id=dilemma_id)


class TestComputeChoiceEdges:
    """Tests for Phase 4c: choice edge derivation."""

    def test_simple_divergence(self) -> None:
        """Commit beat with children on different same-dilemma paths produces choices."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _setup_dilemma(graph, "dilemma::d1", ["path::pa", "path::pb"])

        _make_beat(
            graph,
            "beat::start",
            "Start",
            entities=["entity::hero"],
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::a", "Path A", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "Path B", entities=["entity::hero"])

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        _add_predecessor(graph, "beat::a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        specs = [
            PassageSpec(passage_id="p_start", beat_ids=["beat::start"], summary="start"),
            PassageSpec(passage_id="p_a", beat_ids=["beat::a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 2
        from_passages = {c.from_passage for c in choices}
        to_passages = {c.to_passage for c in choices}
        assert from_passages == {"p_start"}
        assert to_passages == {"p_a", "p_b"}

    def test_no_divergence_emits_continue_choice(self) -> None:
        """Linear cross-passage transition emits a `Continue` choice (R-4c.7)."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = [
            PassageSpec(passage_id="p_a", beat_ids=["beat::a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 1
        assert choices[0].from_passage == "p_a"
        assert choices[0].to_passage == "p_b"
        assert choices[0].label == "Continue"
        assert choices[0].requires == []
        assert choices[0].grants == []

    def test_within_passage_transitions_emit_no_choice(self) -> None:
        """Two beats grouped into a single passage do NOT generate a choice
        (R-4c.7: within-passage transitions belong to FILL prose).
        """
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = [
            PassageSpec(
                passage_id="p_collapsed",
                beat_ids=["beat::a", "beat::b"],
                summary="collapsed",
            ),
        ]

        choices = compute_choice_edges(graph, specs)
        assert choices == []

    def test_grants_from_commit(self) -> None:
        """Choice leading to a commit beat populates grants."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _setup_dilemma(graph, "dilemma::d1", ["path::pa", "path::pb"])

        _make_beat(
            graph,
            "beat::start",
            "Start",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::commit_a",
            "Commit A",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::b", "Path B")

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::commit_a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        _add_predecessor(graph, "beat::commit_a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        # state_flag nodes and grants edges for commit beats
        graph.create_node(
            "state_flag::pa_committed", {"type": "state_flag", "raw_id": "pa_committed"}
        )
        graph.add_edge("grants", "beat::commit_a", "state_flag::pa_committed")

        specs = [
            PassageSpec(passage_id="p_start", beat_ids=["beat::start"], summary="start"),
            PassageSpec(passage_id="p_a", beat_ids=["beat::commit_a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
        ]

        choices = compute_choice_edges(graph, specs)
        # The choice to p_a should have grants for the commit
        choice_to_a = [c for c in choices if c.to_passage == "p_a"]
        assert len(choice_to_a) == 1
        assert len(choice_to_a[0].grants) > 0


class TestContinueChoiceEdges:
    """Tests for R-4c.7: cross-passage non-fork transitions emit Continue edges."""

    def test_three_beat_linear_path_emits_two_continue_choices(self) -> None:
        """A 3-beat linear path with one beat per passage produces 2 Continue choices."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for raw in ("a", "b", "c"):
            _make_beat(graph, f"beat::{raw}", raw.upper())
            _add_belongs_to(graph, f"beat::{raw}", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")

        specs = [
            PassageSpec(passage_id="p_a", beat_ids=["beat::a"], summary="a"),
            PassageSpec(passage_id="p_b", beat_ids=["beat::b"], summary="b"),
            PassageSpec(passage_id="p_c", beat_ids=["beat::c"], summary="c"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 2
        for choice in choices:
            assert choice.label == "Continue"
            assert choice.requires == []
            assert choice.grants == []
        pairs = {(c.from_passage, c.to_passage) for c in choices}
        assert pairs == {("p_a", "p_b"), ("p_b", "p_c")}

    def test_fork_plus_linear_emits_fork_choices_and_continue(self) -> None:
        """A Y-fork followed by linear post-commit paths produces fork choices
        plus Continue choices on each post-commit segment.
        """
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _setup_dilemma(graph, "dilemma::d1", ["path::pa", "path::pb"])

        _make_beat(graph, "beat::start", "Start")
        _make_beat(
            graph,
            "beat::commit_a",
            "Commit A",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::commit_b",
            "Commit B",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::after_a", "After A")
        _make_beat(graph, "beat::after_b", "After B")

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::start", "path::pb")
        _add_belongs_to(graph, "beat::commit_a", "path::pa")
        _add_belongs_to(graph, "beat::commit_b", "path::pb")
        _add_belongs_to(graph, "beat::after_a", "path::pa")
        _add_belongs_to(graph, "beat::after_b", "path::pb")

        _add_predecessor(graph, "beat::commit_a", "beat::start")
        _add_predecessor(graph, "beat::commit_b", "beat::start")
        _add_predecessor(graph, "beat::after_a", "beat::commit_a")
        _add_predecessor(graph, "beat::after_b", "beat::commit_b")

        graph.create_node(
            "state_flag::pa_committed", {"type": "state_flag", "raw_id": "pa_committed"}
        )
        graph.create_node(
            "state_flag::pb_committed", {"type": "state_flag", "raw_id": "pb_committed"}
        )
        graph.add_edge("grants", "beat::commit_a", "state_flag::pa_committed")
        graph.add_edge("grants", "beat::commit_b", "state_flag::pb_committed")

        specs = [
            PassageSpec(passage_id="p_start", beat_ids=["beat::start"], summary="start"),
            PassageSpec(passage_id="p_commit_a", beat_ids=["beat::commit_a"], summary="ca"),
            PassageSpec(passage_id="p_commit_b", beat_ids=["beat::commit_b"], summary="cb"),
            PassageSpec(passage_id="p_after_a", beat_ids=["beat::after_a"], summary="aa"),
            PassageSpec(passage_id="p_after_b", beat_ids=["beat::after_b"], summary="ab"),
        ]

        choices = compute_choice_edges(graph, specs)
        fork = [c for c in choices if c.label != "Continue"]
        cont = [c for c in choices if c.label == "Continue"]

        # Two fork choices from start
        assert len(fork) == 2
        assert {c.to_passage for c in fork} == {"p_commit_a", "p_commit_b"}
        assert all(c.from_passage == "p_start" for c in fork)

        # Two Continue choices on the post-commit linear segments
        assert len(cont) == 2
        assert {(c.from_passage, c.to_passage) for c in cont} == {
            ("p_commit_a", "p_after_a"),
            ("p_commit_b", "p_after_b"),
        }


class TestChoiceEdgesIntersectionMultiBeat:
    """Tests for #1185/#1188: intersection passages with multiple diverging beats.

    Scenario: An intersection passage contains beats from multiple paths.
    Two of those beats independently diverge to the same target passage.
    Before the fix: two duplicate ChoiceSpec entries for that (from, to) pair.
    After the fix: exactly one ChoiceSpec, with grants merged.
    """

    def test_deduplicates_same_target_from_multiple_beats(self) -> None:
        """Two commit beats in the same intersection passage that both diverge to
        single_A must produce exactly one ChoiceSpec(intersection_0 → single_A)."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})
        graph.create_node("path::p3", {"type": "path", "raw_id": "p3"})
        _setup_dilemma(graph, "dilemma::d1", ["path::p1", "path::p2", "path::p3"])

        # Three beats, one per path — all in the intersection passage.
        # b_p1 and b_p2 commit d1 (divergence beats); b_p3 does not commit.
        _make_beat(
            graph,
            "beat::b_p1",
            "P1 beat in intersection",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::b_p2",
            "P2 beat in intersection",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::b_p3", "P3 beat in intersection")
        _add_belongs_to(graph, "beat::b_p1", "path::p1")
        _add_belongs_to(graph, "beat::b_p2", "path::p2")
        _add_belongs_to(graph, "beat::b_p3", "path::p3")

        # Children: b_p1 diverges → child_A (p1) and child_C (p3)
        #           b_p2 diverges → child_A (same p1 target!) and child_B (p2)
        _make_beat(graph, "beat::child_A", "Child on p1")
        _make_beat(graph, "beat::child_B", "Child on p2")
        _make_beat(graph, "beat::child_C", "Child on p3")
        _add_belongs_to(graph, "beat::child_A", "path::p1")
        _add_belongs_to(graph, "beat::child_B", "path::p2")
        _add_belongs_to(graph, "beat::child_C", "path::p3")

        _add_predecessor(graph, "beat::child_A", "beat::b_p1")
        _add_predecessor(graph, "beat::child_C", "beat::b_p1")
        _add_predecessor(graph, "beat::child_A", "beat::b_p2")
        _add_predecessor(graph, "beat::child_B", "beat::b_p2")

        specs = [
            PassageSpec(
                passage_id="passage::intersection_0",
                beat_ids=["beat::b_p1", "beat::b_p2", "beat::b_p3"],
                summary="intersection",
                grouping_type="intersection",
            ),
            PassageSpec(passage_id="passage::single_A", beat_ids=["beat::child_A"], summary="A"),
            PassageSpec(passage_id="passage::single_B", beat_ids=["beat::child_B"], summary="B"),
            PassageSpec(passage_id="passage::single_C", beat_ids=["beat::child_C"], summary="C"),
        ]

        choices = compute_choice_edges(graph, specs)

        # Exactly one ChoiceSpec per unique (from_passage, to_passage) pair
        from_to_pairs = [(c.from_passage, c.to_passage) for c in choices]
        assert len(from_to_pairs) == len(set(from_to_pairs)), (
            f"Duplicate (from, to) pairs: {from_to_pairs}"
        )

        # Exactly one choice from intersection_0 → single_A (not two)
        to_A = [c for c in choices if c.to_passage == "passage::single_A"]
        assert len(to_A) == 1, f"Expected 1 choice to single_A, got {len(to_A)}"

        # All three target passages reachable from intersection_0
        to_passages = {c.to_passage for c in choices if c.from_passage == "passage::intersection_0"}
        assert to_passages == {"passage::single_A", "passage::single_B", "passage::single_C"}

    def test_grants_merged_on_deduplication(self) -> None:
        """When two commit beats in the same passage diverge to the same target
        via different child beats (each with different dilemma_impacts), grants
        from both children are merged (union) in the single deduplicated ChoiceSpec."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})
        graph.create_node("path::p3", {"type": "path", "raw_id": "p3"})
        _setup_dilemma(graph, "dilemma::d1", ["path::p1", "path::p2", "path::p3"])

        # Two divergence beats in the intersection passage — both commit d1
        _make_beat(
            graph,
            "beat::b_p1",
            "P1 beat in intersection",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::b_p2",
            "P2 beat in intersection",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::b_p1", "path::p1")
        _add_belongs_to(graph, "beat::b_p2", "path::p2")

        # b_p1 → child_X1 (p1, commits d1) and child_C (p3)
        _make_beat(
            graph,
            "beat::child_X1",
            "X1 on p1",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::child_C", "C on p3")
        _add_belongs_to(graph, "beat::child_X1", "path::p1")
        _add_belongs_to(graph, "beat::child_C", "path::p3")
        _add_predecessor(graph, "beat::child_X1", "beat::b_p1")
        _add_predecessor(graph, "beat::child_C", "beat::b_p1")

        # b_p2 → child_X2 (p1, commits d2) and child_B (p2)
        _make_beat(
            graph,
            "beat::child_X2",
            "X2 on p1",
            dilemma_impacts=[{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        )
        _make_beat(graph, "beat::child_B", "B on p2")
        _add_belongs_to(graph, "beat::child_X2", "path::p1")
        _add_belongs_to(graph, "beat::child_B", "path::p2")
        _add_predecessor(graph, "beat::child_X2", "beat::b_p2")
        _add_predecessor(graph, "beat::child_B", "beat::b_p2")

        # state_flag nodes and grants edges for commit beats
        # child_X1 commits d1 on p1; child_X2 commits d2 on p1 — use distinct IDs
        graph.create_node(
            "state_flag::p1_d1_committed", {"type": "state_flag", "raw_id": "p1_d1_committed"}
        )
        graph.add_edge("grants", "beat::child_X1", "state_flag::p1_d1_committed")
        graph.create_node(
            "state_flag::p1_d2_committed", {"type": "state_flag", "raw_id": "p1_d2_committed"}
        )
        graph.add_edge("grants", "beat::child_X2", "state_flag::p1_d2_committed")

        # Both child_X1 and child_X2 land in the SAME target passage (passage::X)
        specs = [
            PassageSpec(
                passage_id="passage::inter",
                beat_ids=["beat::b_p1", "beat::b_p2"],
                summary="intersection",
                grouping_type="intersection",
            ),
            PassageSpec(
                passage_id="passage::X",
                beat_ids=["beat::child_X1", "beat::child_X2"],
                summary="X",
            ),
            PassageSpec(passage_id="passage::B", beat_ids=["beat::child_B"], summary="B"),
            PassageSpec(passage_id="passage::C", beat_ids=["beat::child_C"], summary="C"),
        ]

        choices = compute_choice_edges(graph, specs)

        # Exactly one choice to passage::X (deduplicated from two divergence beats)
        to_X = [c for c in choices if c.to_passage == "passage::X"]
        assert len(to_X) == 1

        # Grants: child_X1 contributed state_flag::p1_d1_committed; child_X2 contributed state_flag::p1_d2_committed
        # Merged union should contain both
        assert len(to_X[0].grants) == 2
        assert set(to_X[0].grants) == {"state_flag::p1_d1_committed", "state_flag::p1_d2_committed"}


class TestChoiceEdgesGapBeatChild:
    """Tests for #1187/#1188: topological child selection avoids skipping gap beats.

    Scenario: A divergence beat has both a gap beat and the gap's successor
    as direct children on the same path (due to transitive interleave edges).
    Before the fix: sorted()[0] picks the alphabetically first, which may be
    the gap's successor, orphaning the gap beat's passage.
    After the fix: _topo_first picks the topologically earliest (the gap beat).
    """

    def test_gap_beat_chosen_over_successor(self) -> None:
        """Divergence beat with both gap and after_gap as direct children
        on path p1 must target gap's passage, not after_gap's passage."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})
        _setup_dilemma(graph, "dilemma::d1", ["path::p1", "path::p2"])

        # Divergence beat (commit beat on p1)
        _make_beat(
            graph,
            "beat::div",
            "Divergence",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::div", "path::p1")

        # Gap beat on p1 — the true next beat after div
        _make_beat(graph, "beat::gap_1", "Gap beat")
        _add_belongs_to(graph, "beat::gap_1", "path::p1")

        # Beat after the gap on p1
        _make_beat(graph, "beat::after_gap", "After gap")
        _add_belongs_to(graph, "beat::after_gap", "path::p1")

        # DAG: gap_1 → after_gap (gap_1 is predecessor of after_gap)
        _add_predecessor(graph, "beat::after_gap", "beat::gap_1")

        # div → gap_1 (real next beat)
        _add_predecessor(graph, "beat::gap_1", "beat::div")

        # div → after_gap directly (transitive edge from interleave — the bug scenario)
        _add_predecessor(graph, "beat::after_gap", "beat::div")

        # Beat on p2 to make div a real divergence point
        _make_beat(graph, "beat::p2_child", "P2 child")
        _add_belongs_to(graph, "beat::p2_child", "path::p2")
        _add_predecessor(graph, "beat::p2_child", "beat::div")

        specs = [
            PassageSpec(passage_id="passage::div", beat_ids=["beat::div"], summary="div"),
            PassageSpec(passage_id="passage::gap", beat_ids=["beat::gap_1"], summary="gap"),
            PassageSpec(
                passage_id="passage::after_gap", beat_ids=["beat::after_gap"], summary="after_gap"
            ),
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::p2_child"], summary="p2"),
        ]

        choices = compute_choice_edges(graph, specs)

        # The choice from div's passage on path p1 must target gap's passage,
        # not after_gap's passage (gap_1 is topologically earlier than after_gap).
        from_div = [c for c in choices if c.from_passage == "passage::div"]
        to_passages = {c.to_passage for c in from_div}

        # gap passage must be reachable
        assert "passage::gap" in to_passages, (
            f"Expected passage::gap to be a choice target, got: {to_passages}"
        )

        # after_gap passage should NOT be a direct choice from div (it's behind gap)
        assert "passage::after_gap" not in to_passages, (
            f"passage::after_gap should not be a direct choice target when gap is present, "
            f"got: {to_passages}"
        )

    def test_linear_children_not_affected(self) -> None:
        """When children are in a simple linear chain (no transitive shortcut),
        the result is unchanged — the head of the chain is selected."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})
        _setup_dilemma(graph, "dilemma::d1", ["path::p1", "path::p2"])

        _make_beat(
            graph,
            "beat::div",
            "Divergence",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::div", "path::p1")

        # Only one direct child on p1 (no transitive shortcut)
        _make_beat(graph, "beat::next", "Next on p1")
        _add_belongs_to(graph, "beat::next", "path::p1")
        _add_predecessor(graph, "beat::next", "beat::div")

        _make_beat(graph, "beat::p2_child", "P2 child")
        _add_belongs_to(graph, "beat::p2_child", "path::p2")
        _add_predecessor(graph, "beat::p2_child", "beat::div")

        specs = [
            PassageSpec(passage_id="passage::div", beat_ids=["beat::div"], summary="div"),
            PassageSpec(passage_id="passage::next", beat_ids=["beat::next"], summary="next"),
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::p2_child"], summary="p2"),
        ]

        choices = compute_choice_edges(graph, specs)
        from_div = [c for c in choices if c.from_passage == "passage::div"]
        to_passages = {c.to_passage for c in from_div}
        assert "passage::next" in to_passages
        assert "passage::p2" in to_passages
        assert len(from_div) == 2


class TestChoiceEdgesMultiDilemma:
    """Tests for #1197/#1198: multi-dilemma interleaved DAGs.

    Real GROW output has multiple concurrent dilemmas with interleave
    predecessor edges between their paths. These interleave edges are
    temporal ordering, NOT player choices. compute_choice_edges() must
    only create choices at commit beats for the committing dilemma's paths.
    """

    def test_non_commit_beat_with_cross_dilemma_children_produces_zero_choices(
        self,
    ) -> None:
        """A non-commit beat on dilemma_1::path_a with interleave-ordered
        children on dilemma_2::path_x and dilemma_3::path_y must produce
        no choice edges — interleave ordering is not a player choice."""
        graph = Graph.empty()
        # Dilemma 1 with 2 paths
        graph.create_node("path::d1_a", {"type": "path", "raw_id": "d1_a"})
        graph.create_node("path::d1_b", {"type": "path", "raw_id": "d1_b"})
        _setup_dilemma(graph, "dilemma::d1", ["path::d1_a", "path::d1_b"])
        # Dilemma 2 with 1 path
        graph.create_node("path::d2_a", {"type": "path", "raw_id": "d2_a"})
        _setup_dilemma(graph, "dilemma::d2", ["path::d2_a"])
        # Dilemma 3 with 1 path
        graph.create_node("path::d3_a", {"type": "path", "raw_id": "d3_a"})
        _setup_dilemma(graph, "dilemma::d3", ["path::d3_a"])

        # Non-commit beat on d1_a — has interleave children on other dilemmas
        _make_beat(graph, "beat::advance", "Advance d1")
        _add_belongs_to(graph, "beat::advance", "path::d1_a")

        # Interleave-ordered children on different dilemmas
        _make_beat(graph, "beat::d2_child", "D2 beat")
        _add_belongs_to(graph, "beat::d2_child", "path::d2_a")
        _make_beat(graph, "beat::d3_child", "D3 beat")
        _add_belongs_to(graph, "beat::d3_child", "path::d3_a")
        # Also a same-path child (linear continuation)
        _make_beat(graph, "beat::d1_next", "D1 next")
        _add_belongs_to(graph, "beat::d1_next", "path::d1_a")

        _add_predecessor(graph, "beat::d2_child", "beat::advance")
        _add_predecessor(graph, "beat::d3_child", "beat::advance")
        _add_predecessor(graph, "beat::d1_next", "beat::advance")

        specs = [
            PassageSpec(passage_id="passage::adv", beat_ids=["beat::advance"], summary="adv"),
            PassageSpec(passage_id="passage::d2", beat_ids=["beat::d2_child"], summary="d2"),
            PassageSpec(passage_id="passage::d3", beat_ids=["beat::d3_child"], summary="d3"),
            PassageSpec(passage_id="passage::d1n", beat_ids=["beat::d1_next"], summary="d1n"),
        ]

        choices = compute_choice_edges(graph, specs)
        assert len(choices) == 0, (
            f"Non-commit beat should produce zero choices, got {len(choices)}: "
            f"{[(c.from_passage, c.to_passage) for c in choices]}"
        )

    def test_commit_beat_creates_choices_only_for_own_dilemma(self) -> None:
        """A commit beat on dilemma_1::path_a with children on
        dilemma_1::path_b (same dilemma) AND dilemma_2::path_x (different
        dilemma) must produce exactly 1 choice edge (to path_b's passage)."""
        graph = Graph.empty()
        # Dilemma 1 with 2 paths
        graph.create_node("path::d1_a", {"type": "path", "raw_id": "d1_a"})
        graph.create_node("path::d1_b", {"type": "path", "raw_id": "d1_b"})
        _setup_dilemma(graph, "dilemma::d1", ["path::d1_a", "path::d1_b"])
        # Dilemma 2 with 1 path
        graph.create_node("path::d2_x", {"type": "path", "raw_id": "d2_x"})
        _setup_dilemma(graph, "dilemma::d2", ["path::d2_x"])

        # Commit beat on d1_a — commits dilemma d1
        _make_beat(
            graph,
            "beat::commit_d1",
            "Commit on d1",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::commit_d1", "path::d1_a")

        # Child on d1_b (same dilemma, different path → real choice)
        _make_beat(graph, "beat::d1b_child", "D1 path B beat")
        _add_belongs_to(graph, "beat::d1b_child", "path::d1_b")
        _add_predecessor(graph, "beat::d1b_child", "beat::commit_d1")

        # Child on d1_a (same path → no divergence for this path)
        _make_beat(graph, "beat::d1a_next", "D1 path A next")
        _add_belongs_to(graph, "beat::d1a_next", "path::d1_a")
        _add_predecessor(graph, "beat::d1a_next", "beat::commit_d1")

        # Child on d2_x (different dilemma → interleave, NOT a choice)
        _make_beat(graph, "beat::d2x_child", "D2 interleave beat")
        _add_belongs_to(graph, "beat::d2x_child", "path::d2_x")
        _add_predecessor(graph, "beat::d2x_child", "beat::commit_d1")

        specs = [
            PassageSpec(
                passage_id="passage::commit",
                beat_ids=["beat::commit_d1"],
                summary="commit",
            ),
            PassageSpec(
                passage_id="passage::d1a_next",
                beat_ids=["beat::d1a_next"],
                summary="d1a_next",
            ),
            PassageSpec(
                passage_id="passage::d1b",
                beat_ids=["beat::d1b_child"],
                summary="d1b",
            ),
            PassageSpec(
                passage_id="passage::d2x",
                beat_ids=["beat::d2x_child"],
                summary="d2x",
            ),
        ]

        choices = compute_choice_edges(graph, specs)

        # Exactly 2 choices: commit → d1a_next (same dilemma, path a) and
        # commit → d1b (same dilemma, path b). NOT commit → d2x.
        assert len(choices) == 2, (
            f"Expected 2 choices (d1 paths only), got {len(choices)}: "
            f"{[(c.from_passage, c.to_passage) for c in choices]}"
        )
        to_passages = {c.to_passage for c in choices}
        assert to_passages == {"passage::d1a_next", "passage::d1b"}, (
            f"Expected choices to d1 paths only, got {to_passages}"
        )

    def test_intersection_with_commit_beat_from_one_dilemma(self) -> None:
        """An intersection containing beats from 2 dilemmas where one beat
        commits its dilemma: choices only for that dilemma's paths."""
        graph = Graph.empty()
        # Dilemma 1 with 2 paths
        graph.create_node("path::d1_a", {"type": "path", "raw_id": "d1_a"})
        graph.create_node("path::d1_b", {"type": "path", "raw_id": "d1_b"})
        _setup_dilemma(graph, "dilemma::d1", ["path::d1_a", "path::d1_b"])
        # Dilemma 2 with 2 paths
        graph.create_node("path::d2_x", {"type": "path", "raw_id": "d2_x"})
        graph.create_node("path::d2_y", {"type": "path", "raw_id": "d2_y"})
        _setup_dilemma(graph, "dilemma::d2", ["path::d2_x", "path::d2_y"])

        # Intersection beat from d1_a — commits d1
        _make_beat(
            graph,
            "beat::inter_d1",
            "D1 intersection beat",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::inter_d1", "path::d1_a")

        # Intersection beat from d2_x — does NOT commit
        _make_beat(graph, "beat::inter_d2", "D2 intersection beat")
        _add_belongs_to(graph, "beat::inter_d2", "path::d2_x")

        # Children on d1's paths (real choices for d1)
        _make_beat(graph, "beat::d1a_next", "D1A next")
        _add_belongs_to(graph, "beat::d1a_next", "path::d1_a")
        _add_predecessor(graph, "beat::d1a_next", "beat::inter_d1")

        _make_beat(graph, "beat::d1b_next", "D1B next")
        _add_belongs_to(graph, "beat::d1b_next", "path::d1_b")
        _add_predecessor(graph, "beat::d1b_next", "beat::inter_d1")

        # Children on d2's paths (interleave from inter_d1, NOT choices)
        _make_beat(graph, "beat::d2x_next", "D2X next")
        _add_belongs_to(graph, "beat::d2x_next", "path::d2_x")
        _add_predecessor(graph, "beat::d2x_next", "beat::inter_d1")

        _make_beat(graph, "beat::d2y_next", "D2Y next")
        _add_belongs_to(graph, "beat::d2y_next", "path::d2_y")
        _add_predecessor(graph, "beat::d2y_next", "beat::inter_d1")

        specs = [
            PassageSpec(
                passage_id="passage::inter",
                beat_ids=["beat::inter_d1", "beat::inter_d2"],
                summary="intersection",
                grouping_type="intersection",
            ),
            PassageSpec(
                passage_id="passage::d1a",
                beat_ids=["beat::d1a_next"],
                summary="d1a",
            ),
            PassageSpec(
                passage_id="passage::d1b",
                beat_ids=["beat::d1b_next"],
                summary="d1b",
            ),
            PassageSpec(
                passage_id="passage::d2x",
                beat_ids=["beat::d2x_next"],
                summary="d2x",
            ),
            PassageSpec(
                passage_id="passage::d2y",
                beat_ids=["beat::d2y_next"],
                summary="d2y",
            ),
        ]

        choices = compute_choice_edges(graph, specs)

        # Only 2 choices: inter → d1a and inter → d1b (d1's commit)
        # NOT inter → d2x or inter → d2y (d2 has no commit here)
        assert len(choices) == 2, (
            f"Expected 2 choices (d1 commit only), got {len(choices)}: "
            f"{[(c.from_passage, c.to_passage) for c in choices]}"
        )
        to_passages = {c.to_passage for c in choices}
        assert to_passages == {"passage::d1a", "passage::d1b"}

    def test_realistic_fanout_bounded(self) -> None:
        """With 3 dilemmas (6 paths), interleave-style edges, and one commit
        per dilemma: total choices ≈ 2 per dilemma = 6, not combinatorial."""
        graph = Graph.empty()

        # 3 dilemmas x 2 paths each
        dilemmas = ["dilemma::d1", "dilemma::d2", "dilemma::d3"]
        all_paths: dict[str, list[str]] = {}
        for i, did in enumerate(dilemmas, 1):
            pa = f"path::d{i}_a"
            pb = f"path::d{i}_b"
            graph.create_node(pa, {"type": "path", "raw_id": f"d{i}_a"})
            graph.create_node(pb, {"type": "path", "raw_id": f"d{i}_b"})
            _setup_dilemma(graph, did, [pa, pb])
            all_paths[did] = [pa, pb]

        # One commit beat per dilemma (on path_a), with children on both paths
        specs_list: list[PassageSpec] = []
        for i, did in enumerate(dilemmas, 1):
            pa, pb = all_paths[did]
            commit_id = f"beat::commit_d{i}"
            _make_beat(
                graph,
                commit_id,
                f"Commit D{i}",
                dilemma_impacts=[{"dilemma_id": did, "effect": "commits"}],
            )
            _add_belongs_to(graph, commit_id, pa)

            # Child on path_a (same path continuation)
            ca = f"beat::d{i}_a_next"
            _make_beat(graph, ca, f"D{i} A next")
            _add_belongs_to(graph, ca, pa)
            _add_predecessor(graph, ca, commit_id)

            # Child on path_b (real choice)
            cb = f"beat::d{i}_b_next"
            _make_beat(graph, cb, f"D{i} B next")
            _add_belongs_to(graph, cb, pb)
            _add_predecessor(graph, cb, commit_id)

            # Cross-dilemma interleave children (NOT choices)
            for j, other_did in enumerate(dilemmas, 1):
                if other_did == did:
                    continue
                for suffix in ["a", "b"]:
                    other_path = f"path::d{j}_{suffix}"
                    interleave_id = f"beat::interleave_d{i}_to_d{j}_{suffix}"
                    _make_beat(graph, interleave_id, f"Interleave {i}→{j}{suffix}")
                    _add_belongs_to(graph, interleave_id, other_path)
                    _add_predecessor(graph, interleave_id, commit_id)

            specs_list.append(
                PassageSpec(
                    passage_id=f"passage::commit_d{i}",
                    beat_ids=[commit_id],
                    summary=f"commit d{i}",
                )
            )
            specs_list.append(
                PassageSpec(passage_id=f"passage::d{i}_a", beat_ids=[ca], summary=f"d{i} a")
            )
            specs_list.append(
                PassageSpec(passage_id=f"passage::d{i}_b", beat_ids=[cb], summary=f"d{i} b")
            )

        # Add passages for interleave beats
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    continue
                for suffix in ["a", "b"]:
                    bid = f"beat::interleave_d{i}_to_d{j}_{suffix}"
                    specs_list.append(
                        PassageSpec(
                            passage_id=f"passage::il_d{i}_d{j}_{suffix}",
                            beat_ids=[bid],
                            summary=f"il {i}→{j}{suffix}",
                        )
                    )

        choices = compute_choice_edges(graph, specs_list)

        # 3 dilemmas x 2 choices each = 6 total, NOT the combinatorial explosion
        assert len(choices) == 6, (
            f"Expected 6 choices (2 per dilemma), got {len(choices)}: "
            f"{[(c.from_passage, c.to_passage) for c in choices]}"
        )


class TestChoiceEdgesYShapeAdvancesChild:
    """Tests for #1254: Y-shape where commit beat is NOT the immediate child.

    SEED produces: shared_02 → beat_01 (advances) → beat_02 (commits) → beat_03
    The fork is at shared_02, but the commits effect is on beat_02 (grandchild).
    compute_choice_edges must detect the fork from path membership and walk
    deeper for grants.
    """

    def test_y_shape_with_advances_child_produces_choices(self) -> None:
        """Shared beat → advances child → commits grandchild must still produce choices."""
        graph = Graph.empty()

        graph.create_node("path::d1_a", {"type": "path", "raw_id": "d1_a"})
        graph.create_node("path::d1_b", {"type": "path", "raw_id": "d1_b"})
        _setup_dilemma(graph, "dilemma::d1", ["path::d1_a", "path::d1_b"])

        # Shared pre-commit beat (dual belongs_to)
        graph.create_node(
            "beat::shared_02",
            {
                "type": "beat",
                "raw_id": "shared_02",
                "summary": "Last shared",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "reveals"}],
            },
        )
        _add_belongs_to(graph, "beat::shared_02", "path::d1_a")
        _add_belongs_to(graph, "beat::shared_02", "path::d1_b")

        # First exclusive beats — advances, NOT commits
        graph.create_node(
            "beat::d1_a_01",
            {
                "type": "beat",
                "raw_id": "d1_a_01",
                "summary": "Path A setup",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        _add_belongs_to(graph, "beat::d1_a_01", "path::d1_a")
        graph.create_node(
            "beat::d1_b_01",
            {
                "type": "beat",
                "raw_id": "d1_b_01",
                "summary": "Path B setup",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        _add_belongs_to(graph, "beat::d1_b_01", "path::d1_b")

        # Commit beats — deeper in the chain
        graph.create_node(
            "beat::d1_a_02",
            {
                "type": "beat",
                "raw_id": "d1_a_02",
                "summary": "Path A commits",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        _add_belongs_to(graph, "beat::d1_a_02", "path::d1_a")
        graph.create_node(
            "beat::d1_b_02",
            {
                "type": "beat",
                "raw_id": "d1_b_02",
                "summary": "Path B commits",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        _add_belongs_to(graph, "beat::d1_b_02", "path::d1_b")

        # Predecessor chain: shared_02 → a_01 → a_02, shared_02 → b_01 → b_02
        graph.add_edge("predecessor", "beat::d1_a_01", "beat::shared_02")
        graph.add_edge("predecessor", "beat::d1_b_01", "beat::shared_02")
        graph.add_edge("predecessor", "beat::d1_a_02", "beat::d1_a_01")
        graph.add_edge("predecessor", "beat::d1_b_02", "beat::d1_b_01")

        # Passages: shared_02 in its own, each path beat in its own
        specs = [
            PassageSpec(
                passage_id="passage::shared",
                beat_ids=["beat::shared_02"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::a_setup",
                beat_ids=["beat::d1_a_01"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::b_setup",
                beat_ids=["beat::d1_b_01"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::a_commit",
                beat_ids=["beat::d1_a_02"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::b_commit",
                beat_ids=["beat::d1_b_02"],
                grouping_type="single",
                summary="auto-summary",
            ),
        ]

        choices = compute_choice_edges(graph, specs)

        # Filter out R-4c.7 Continue edges; this test asserts the fork shape.
        fork_choices = [c for c in choices if c.label != "Continue"]
        assert len(fork_choices) == 2, (
            f"Expected 2 fork choices, got {len(fork_choices)}: "
            f"{[(c.from_passage, c.to_passage) for c in fork_choices]}"
        )
        to_passages = {c.to_passage for c in fork_choices}
        assert "passage::a_setup" in to_passages
        assert "passage::b_setup" in to_passages
        # All fork choices should originate from the shared passage
        assert all(c.from_passage == "passage::shared" for c in fork_choices)

    def test_grants_found_via_deep_walk(self) -> None:
        """Grants should be populated even when commits beat is a grandchild."""
        graph = Graph.empty()

        graph.create_node("path::d1_a", {"type": "path", "raw_id": "d1_a"})
        graph.create_node("path::d1_b", {"type": "path", "raw_id": "d1_b"})
        _setup_dilemma(graph, "dilemma::d1", ["path::d1_a", "path::d1_b"])

        graph.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "summary": "Shared",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "reveals"}],
            },
        )
        _add_belongs_to(graph, "beat::shared", "path::d1_a")
        _add_belongs_to(graph, "beat::shared", "path::d1_b")

        graph.create_node(
            "beat::a_01",
            {
                "type": "beat",
                "raw_id": "a_01",
                "summary": "A advances",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        _add_belongs_to(graph, "beat::a_01", "path::d1_a")
        graph.create_node(
            "beat::a_02",
            {
                "type": "beat",
                "raw_id": "a_02",
                "summary": "A commits",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        _add_belongs_to(graph, "beat::a_02", "path::d1_a")

        graph.create_node(
            "beat::b_01",
            {
                "type": "beat",
                "raw_id": "b_01",
                "summary": "B advances",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        _add_belongs_to(graph, "beat::b_01", "path::d1_b")
        graph.create_node(
            "beat::b_02",
            {
                "type": "beat",
                "raw_id": "b_02",
                "summary": "B commits",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        _add_belongs_to(graph, "beat::b_02", "path::d1_b")

        graph.add_edge("predecessor", "beat::a_01", "beat::shared")
        graph.add_edge("predecessor", "beat::b_01", "beat::shared")
        graph.add_edge("predecessor", "beat::a_02", "beat::a_01")
        graph.add_edge("predecessor", "beat::b_02", "beat::b_01")

        # state_flag nodes and grants edges for commit beats
        graph.create_node(
            "state_flag::d1_a_committed", {"type": "state_flag", "raw_id": "d1_a_committed"}
        )
        graph.add_edge("grants", "beat::a_02", "state_flag::d1_a_committed")
        graph.create_node(
            "state_flag::d1_b_committed", {"type": "state_flag", "raw_id": "d1_b_committed"}
        )
        graph.add_edge("grants", "beat::b_02", "state_flag::d1_b_committed")

        specs = [
            PassageSpec(
                passage_id="passage::shared",
                beat_ids=["beat::shared"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::a",
                beat_ids=["beat::a_01"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::b",
                beat_ids=["beat::b_01"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::a_commit",
                beat_ids=["beat::a_02"],
                grouping_type="single",
                summary="auto-summary",
            ),
            PassageSpec(
                passage_id="passage::b_commit",
                beat_ids=["beat::b_02"],
                grouping_type="single",
                summary="auto-summary",
            ),
        ]

        choices = compute_choice_edges(graph, specs)

        # Filter to fork choices; R-4c.7 Continue edges are linear and have no grants.
        fork_choices = [c for c in choices if c.label != "Continue"]
        assert len(fork_choices) == 2
        # Each fork choice should have grants from the commits beat downstream
        for choice in fork_choices:
            assert len(choice.grants) > 0, (
                f"Choice {choice.from_passage}→{choice.to_passage} has no grants; "
                f"commits beat should have been found via deep walk"
            )


class TestFindFalseBranchCandidates:
    """Tests for Phase 4d: false branch identification."""

    def test_short_stretch_no_candidates(self) -> None:
        """Fewer than 3 passages → no candidates."""
        graph = Graph.empty()
        specs = [
            PassageSpec(passage_id="p1", beat_ids=["b1"], summary="a"),
            PassageSpec(passage_id="p2", beat_ids=["b2"], summary="b"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        assert len(candidates) == 0

    def test_linear_stretch_produces_candidate(self) -> None:
        """3+ passages in a row via beat DAG adjacency → at least one candidate."""
        graph = Graph.empty()
        # Create beat nodes and predecessor edges forming a linear chain
        for i in range(5):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        for i in range(1, 5):
            graph.add_edge("predecessor", f"beat::b{i}", f"beat::b{i - 1}")
        specs = [
            PassageSpec(passage_id=f"passage::p{i}", beat_ids=[f"beat::b{i}"], summary=f"P{i}")
            for i in range(5)
        ]
        candidates = find_false_branch_candidates(graph, specs)
        assert len(candidates) >= 1
        assert all(len(c.passage_ids) >= 3 for c in candidates)

    def test_out_of_order_spec_but_adjacent_via_dag(self) -> None:
        """Passages out-of-order in spec list but forming a linear chain via beat DAG
        are correctly identified as false branch candidates (#1161)."""
        graph = Graph.empty()
        # Beats form a linear chain: b0 → b1 → b2 → b3
        for i in range(4):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        for i in range(1, 4):
            graph.add_edge("predecessor", f"beat::b{i}", f"beat::b{i - 1}")

        # Specs listed in reverse order — spec list order is NOT topological
        specs = [
            PassageSpec(passage_id="passage::p3", beat_ids=["beat::b3"], summary="P3"),
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::b2"], summary="P2"),
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::b1"], summary="P1"),
            PassageSpec(passage_id="passage::p0", beat_ids=["beat::b0"], summary="P0"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # All four form a linear chain via beat DAG adjacency
        assert len(candidates) == 1
        assert len(candidates[0].passage_ids) == 4

    def test_non_adjacent_passages_not_identified(self) -> None:
        """Passages in spec order but NOT adjacent in the passage graph are
        NOT identified as false branch candidates (#1161)."""
        graph = Graph.empty()
        # Two separate passages with no beat DAG connection between them
        for i in range(4):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        # b0 → b1 form one chain; b2 → b3 form another (no connection between)
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

        specs = [
            PassageSpec(passage_id="passage::p0", beat_ids=["beat::b0"], summary="P0"),
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::b1"], summary="P1"),
            # gap — no connection from b1 to b2
            PassageSpec(passage_id="passage::p2", beat_ids=["beat::b2"], summary="P2"),
            PassageSpec(passage_id="passage::p3", beat_ids=["beat::b3"], summary="P3"),
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # Each sub-chain has only 2 passages — neither qualifies as 3+
        assert all(len(c.passage_ids) < 3 for c in candidates)


class TestPolishPlan:
    """Tests for the PolishPlan dataclass."""

    def test_empty_plan(self) -> None:
        plan = PolishPlan()
        assert len(plan.passage_specs) == 0
        assert len(plan.choice_specs) == 0
        assert len(plan.warnings) == 0

    def test_plan_with_specs(self) -> None:
        plan = PolishPlan(
            passage_specs=[PassageSpec(passage_id="p1", beat_ids=["b1"], summary="test")],
            warnings=["Something to review"],
        )
        assert len(plan.passage_specs) == 1
        assert len(plan.warnings) == 1


# ---------------------------------------------------------------------------
# Tests for Fix #1153: arc_traversals populated after phase_plan_application
# ---------------------------------------------------------------------------


class TestArcTraversalsAfterPlanApplication:
    """Phase 6 must populate arc_traversals on the polish_plan node."""

    def test_arc_traversals_non_empty_after_phase6(self) -> None:
        """Minimal Y-fork graph: phase 4 + phase 6 produce non-empty arc_traversals.

        The fixture was updated (cluster #1311 / Task 14) to include a Y-fork so
        that Phase 4c produces at least one choice edge and does not trigger the
        zero-choice PolishContractError halt (R-4c.2, Task 10).

        Graph: shared (pa+pb dual belongs_to) → commit_a (pa, commits d1)
                                               → commit_b (pb, commits d1)
        """
        import asyncio

        from questfoundry.pipeline.stages.polish.deterministic import (
            phase_plan_application,
            phase_plan_computation,
        )

        graph = Graph.empty()

        # One dilemma with two paths
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1", "status": "explored"})
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb", "dilemma_id": "dilemma::d1"})

        # State flags
        graph.create_node(
            "state_flag::d1_pa",
            {
                "type": "state_flag",
                "raw_id": "d1_pa",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pa",
            },
        )
        graph.create_node(
            "state_flag::d1_pb",
            {
                "type": "state_flag",
                "raw_id": "d1_pb",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pb",
            },
        )

        # Shared pre-commit beat (dual belongs_to — Y-shape §MEMORY)
        graph.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "summary": "Shared beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "reveals"}],
                "entities": [],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::shared", "path::pa")
        graph.add_edge("belongs_to", "beat::shared", "path::pb")

        # Commit beats — one per path (creates the Y-fork)
        for bid, pid, sfid in [
            ("beat::commit_a", "path::pa", "state_flag::d1_pa"),
            ("beat::commit_b", "path::pb", "state_flag::d1_pb"),
        ]:
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "raw_id": bid.split("::")[-1],
                    "summary": f"Commit on {pid}",
                    "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
                    "entities": [],
                    "scene_type": "scene",
                },
            )
            graph.add_edge("belongs_to", bid, pid)
            graph.add_edge("grants", bid, sfid)
            graph.add_edge("predecessor", bid, "beat::shared")

        # Run Phase 4 (plan computation) — should succeed now that Y-fork exists
        r4 = asyncio.run(phase_plan_computation(graph, None))  # type: ignore[arg-type]
        assert r4.status == "completed"

        # Run Phase 6 (plan application) — no Phase 5 needed for minimal graph
        r6 = asyncio.run(phase_plan_application(graph, None))  # type: ignore[arg-type]
        assert r6.status == "completed"

        # Verify arc_traversals is populated
        plan_nodes = graph.get_nodes_by_type("polish_plan")
        plan_data = plan_nodes.get("polish_plan::current", {})
        arc_traversals = plan_data.get("arc_traversals", {})

        assert arc_traversals, "arc_traversals must be non-empty after plan application"
        # Keys should follow arc_key format (e.g. "path_a+path_b")
        for key in arc_traversals:
            assert "+" in key or key  # arc keys join path IDs
        # Values should be lists of passage IDs
        for passages in arc_traversals.values():
            assert isinstance(passages, list)
            for pid in passages:
                assert pid.startswith("passage::")


# ---------------------------------------------------------------------------
# Tests for Fix #1152: ChoiceSpec.requires populated for convergence choices
# ---------------------------------------------------------------------------


class TestChoiceSpecRequires:
    """compute_choice_edges must populate requires for choices from intersection passages."""

    def _build_convergence_graph(self, graph: Graph) -> None:
        """Build a graph where an intersection commit beat diverges to two paths.

        Structure:
          start (pa) → merge (intersection, pa, commits d1) → c (pc, commits d1) ──┐
                                                             → d (pd, commits d1) ──┴→ (end)

        The intersection passage (containing beat::merge) diverges to paths pc
        and pd. beat::c and beat::d are the first commit beats on their paths,
        and merge itself has no commit ancestors — so compute_active_flags_at_beat
        returns a single-combo result for each child, allowing requires to be set.
        """
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1", "status": "explored"})
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pc", {"type": "path", "raw_id": "pc", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pd", {"type": "path", "raw_id": "pd", "dilemma_id": "dilemma::d1"})
        graph.create_node(
            "state_flag::d1_pa",
            {
                "type": "state_flag",
                "raw_id": "d1_pa",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pa",
            },
        )
        graph.create_node(
            "state_flag::d1_pc",
            {
                "type": "state_flag",
                "raw_id": "d1_pc",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pc",
            },
        )
        graph.create_node(
            "state_flag::d1_pd",
            {
                "type": "state_flag",
                "raw_id": "d1_pd",
                "dilemma_id": "dilemma::d1",
                "path_id": "path::pd",
            },
        )
        graph.add_edge("dilemma_path", "dilemma::d1", "path::pc")
        graph.add_edge("dilemma_path", "dilemma::d1", "path::pd")

        # Start beat on pa (no commits — clean ancestors for merge)
        _make_beat(graph, "beat::start", "Start")
        graph.add_edge("belongs_to", "beat::start", "path::pa")

        # Intersection beat — commits d1, creating the divergence point
        _make_beat(
            graph,
            "beat::merge",
            "Merge beat",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        graph.add_edge("belongs_to", "beat::merge", "path::pa")
        # grants edge: beat::merge activates state_flag::d1_pa (its own path)
        graph.add_edge("grants", "beat::merge", "state_flag::d1_pa")
        graph.create_node(
            "intersection_group::g1",
            {"type": "intersection_group", "raw_id": "g1", "beat_ids": ["beat::merge"]},
        )

        # Post-intersection beats on two different paths — the commit already
        # happened at merge; these are regular beats on d1's diverging paths.
        _make_beat(graph, "beat::c", "Path C beat")
        graph.add_edge("belongs_to", "beat::c", "path::pc")
        _make_beat(graph, "beat::d", "Path D beat")
        graph.add_edge("belongs_to", "beat::d", "path::pd")

        # Predecessor edges
        _add_predecessor(graph, "beat::merge", "beat::start")
        _add_predecessor(graph, "beat::c", "beat::merge")
        _add_predecessor(graph, "beat::d", "beat::merge")

    def test_divergence_choice_requires_empty_no_flags(self) -> None:
        """R-4c.4: Choices whose target beat has no active flags have empty requires.

        This covers the common divergence case (no state flags exist in the graph
        at all).  No flags → no soft flags → requires stays empty.
        """
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _setup_dilemma(graph, "dilemma::d1", ["path::pa", "path::pb"])

        _make_beat(
            graph,
            "beat::start",
            "Start",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::a", "Path A")
        _make_beat(graph, "beat::b", "Path B")

        graph.add_edge("belongs_to", "beat::start", "path::pa")
        graph.add_edge("belongs_to", "beat::a", "path::pa")
        graph.add_edge("belongs_to", "beat::b", "path::pb")

        _add_predecessor(graph, "beat::a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        specs = compute_beat_grouping(graph)
        choices = compute_choice_edges(graph, specs)

        # No state_flag nodes in graph → no active flags → requires is empty for all
        for c in choices:
            assert c.requires == [], f"Expected empty requires for {c.from_passage}"

    def test_post_convergence_soft_dilemma_choice_has_requires(self) -> None:
        """R-4c.3: Post-convergence soft-dilemma choices have requires set.

        Structure:
          beat::commit (d1, commits) → beat::pa (path::pa)
                                     → beat::pb (path::pb)

        A state_flag for d1 is active at beat::pa and beat::pb (soft dilemma).
        compute_choice_edges must set requires=[state_flag_id] on those choices.
        """
        graph = Graph.empty()

        # Soft dilemma
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft"},
        )
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb", "dilemma_id": "dilemma::d1"})

        # State flags for each path (active after commit)
        graph.create_node(
            "state_flag::sf_pa",
            {"type": "state_flag", "raw_id": "sf_pa", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "state_flag::sf_pb",
            {"type": "state_flag", "raw_id": "sf_pb", "dilemma_id": "dilemma::d1"},
        )

        # Commit beat (on path::pa — classic Case A divergence)
        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        graph.add_edge("belongs_to", "beat::commit", "path::pa")
        graph.add_edge("grants", "beat::commit", "state_flag::sf_pa")

        # Post-commit beats on each path
        _make_beat(graph, "beat::pa", "Path A beat")
        graph.add_edge("belongs_to", "beat::pa", "path::pa")

        _make_beat(graph, "beat::pb", "Path B beat")
        graph.add_edge("belongs_to", "beat::pb", "path::pb")
        graph.add_edge("grants", "beat::pb", "state_flag::sf_pb")

        _add_predecessor(graph, "beat::pa", "beat::commit")
        _add_predecessor(graph, "beat::pb", "beat::commit")

        specs = compute_beat_grouping(graph)
        choices = compute_choice_edges(graph, specs)

        assert len(choices) == 2, f"Expected 2 choices, got {len(choices)}: {choices}"
        for c in choices:
            # The target beat is a post-commit beat on a soft dilemma path.
            # compute_active_flags_at_beat returns the granted flags at that beat.
            # The soft filter should keep them → requires must be non-empty.
            assert c.requires != [], (
                f"R-4c.3 violation: choice {c.from_passage!r} → {c.to_passage!r} "
                f"targets a soft-dilemma post-commit beat but has empty requires"
            )

    def test_hard_dilemma_choice_has_empty_requires(self) -> None:
        """R-4c.4: Hard-dilemma choices have empty requires.

        Same divergence structure but dilemma_role == 'hard'.  State flags
        belonging to hard dilemmas must be filtered out, leaving requires=[].
        """
        graph = Graph.empty()

        # Hard dilemma
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard"},
        )
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa", "dilemma_id": "dilemma::d1"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb", "dilemma_id": "dilemma::d1"})

        # State flags belonging to the hard dilemma
        graph.create_node(
            "state_flag::sf_pa",
            {"type": "state_flag", "raw_id": "sf_pa", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "state_flag::sf_pb",
            {"type": "state_flag", "raw_id": "sf_pb", "dilemma_id": "dilemma::d1"},
        )

        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        graph.add_edge("belongs_to", "beat::commit", "path::pa")
        graph.add_edge("grants", "beat::commit", "state_flag::sf_pa")

        _make_beat(graph, "beat::pa", "Path A beat")
        graph.add_edge("belongs_to", "beat::pa", "path::pa")

        _make_beat(graph, "beat::pb", "Path B beat")
        graph.add_edge("belongs_to", "beat::pb", "path::pb")
        graph.add_edge("grants", "beat::pb", "state_flag::sf_pb")

        _add_predecessor(graph, "beat::pa", "beat::commit")
        _add_predecessor(graph, "beat::pb", "beat::commit")

        specs = compute_beat_grouping(graph)
        choices = compute_choice_edges(graph, specs)

        assert len(choices) == 2, f"Expected 2 choices, got {len(choices)}: {choices}"
        for c in choices:
            assert c.requires == [], (
                f"R-4c.4 violation: choice {c.from_passage!r} → {c.to_passage!r} "
                f"targets a hard-dilemma beat but has non-empty requires: {c.requires}"
            )


# ---------------------------------------------------------------------------
# Tests for Issue #1157: Ambiguous feasibility detection in Phase 4b
# ---------------------------------------------------------------------------


class TestAmbiguousFeasibilityDetection:
    """Phase 4b must detect ambiguous cases (mixed residue weights) and store them."""

    def _make_commit_beat(
        self,
        graph: Graph,
        beat_id: str,
        path_id: str,
        dilemma_id: str,
        state_flag_id: str | None = None,
    ) -> str:
        """Create a commit beat with an associated state_flag node and grants edge.

        Returns the state_flag_id used (derived from path_id if not provided).
        """
        graph.create_node(
            beat_id,
            {
                "type": "beat",
                "raw_id": beat_id.split("::")[-1],
                "summary": f"Commit on {path_id}",
                "dilemma_impacts": [{"dilemma_id": dilemma_id, "effect": "commits"}],
                "entities": [],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", beat_id, path_id)
        # Create state_flag node and grants edge so compute_active_flags_at_beat returns
        # real state_flag::* node IDs instead of synthetic dilemma::*:path::* strings.
        if state_flag_id is None:
            path_raw = path_id.split("::")[-1]
            state_flag_id = f"state_flag::{path_raw}_committed"
        if not graph.get_node(state_flag_id):
            graph.create_node(
                state_flag_id,
                {
                    "type": "state_flag",
                    "raw_id": state_flag_id.split("::")[-1],
                    "dilemma_id": dilemma_id,
                },
            )
        graph.add_edge("grants", beat_id, state_flag_id)
        return state_flag_id

    def test_mixed_weights_produces_ambiguous_spec(self) -> None:
        """Passage with one heavy flag and one light flag → ambiguous_specs, NOT in variant or residue."""
        graph = Graph.empty()

        # Two dilemmas — one heavy, one light
        graph.create_node(
            "dilemma::heavy",
            {
                "type": "dilemma",
                "raw_id": "heavy",
                "residue_weight": "heavy",
            },
        )
        graph.create_node(
            "dilemma::light",
            {
                "type": "dilemma",
                "raw_id": "light",
                "residue_weight": "light",
            },
        )

        # Two paths
        graph.create_node("path::ph", {"type": "path", "raw_id": "ph"})
        graph.create_node("path::pl", {"type": "path", "raw_id": "pl"})

        # Commit beats first so state_flag nodes exist before entity references them
        # Chain: commit_h → commit_l → target
        sf_h = self._make_commit_beat(graph, "beat::commit_h", "path::ph", "dilemma::heavy")
        sf_l = self._make_commit_beat(graph, "beat::commit_l", "path::pl", "dilemma::light")

        # Entity that appears in the passage — overlays use state_flag node IDs
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [sf_h], "details": {"mood": "grim"}},
                    {"when": [sf_l], "details": {"mood": "relieved"}},
                ],
            },
        )
        _add_predecessor(graph, "beat::commit_l", "beat::commit_h")

        # Target beat referencing entity::hero
        _make_beat(
            graph,
            "beat::target",
            "Target beat with hero",
            entities=["entity::hero"],
        )
        graph.add_edge("belongs_to", "beat::target", "path::ph")
        _add_predecessor(graph, "beat::target", "beat::commit_l")

        spec = PassageSpec(
            passage_id="passage::test_mixed",
            beat_ids=["beat::target"],
            summary="test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])

        # Mixed weights → goes to ambiguous, NOT variant or residue
        assert len(result["ambiguous_specs"]) == 1
        assert result["ambiguous_specs"][0].passage_id == "passage::test_mixed"
        assert len(result["variant_specs"]) == 0
        assert len(result["residue_specs"]) == 0

    def test_all_heavy_flags_not_ambiguous(self) -> None:
        """2+ flags all heavy → deterministic variant, NOT ambiguous."""
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "residue_weight": "heavy"},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "residue_weight": "hard"},
        )
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})

        # Create commit beats first so state_flag IDs are available for overlays
        sf_p1 = self._make_commit_beat(graph, "beat::c1", "path::p1", "dilemma::d1")
        sf_p2 = self._make_commit_beat(graph, "beat::c2", "path::p2", "dilemma::d2")

        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [sf_p1], "details": {"mood": "dark"}},
                    {"when": [sf_p2], "details": {"mood": "grim"}},
                ],
            },
        )

        # Chain: c2 → c1 → target so both flags are ancestors of beat::target
        _add_predecessor(graph, "beat::c1", "beat::c2")

        _make_beat(graph, "beat::target", "Target", entities=["entity::hero"])
        graph.add_edge("belongs_to", "beat::target", "path::p1")
        _add_predecessor(graph, "beat::target", "beat::c1")

        spec = PassageSpec(
            passage_id="passage::all_heavy",
            beat_ids=["beat::target"],
            summary="heavy test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])
        assert len(result["ambiguous_specs"]) == 0
        assert len(result["variant_specs"]) == 2  # one per heavy flag
        assert len(result["residue_specs"]) == 0

    def test_single_relevant_flag_always_deterministic(self) -> None:
        """Exactly 1 relevant flag → never ambiguous (deterministic assignment)."""
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "residue_weight": "heavy"},
        )
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        # Create commit beat first so state_flag ID is available for overlay
        sf_p1 = self._make_commit_beat(graph, "beat::c1", "path::p1", "dilemma::d1")

        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [sf_p1], "details": {"mood": "dark"}},
                ],
            },
        )

        _make_beat(graph, "beat::target", "Target", entities=["entity::hero"])
        graph.add_edge("belongs_to", "beat::target", "path::p1")
        _add_predecessor(graph, "beat::target", "beat::c1")

        spec = PassageSpec(
            passage_id="passage::single_flag",
            beat_ids=["beat::target"],
            summary="single flag test",
            entities=["entity::hero"],
        )

        result = compute_prose_feasibility(graph, [spec])
        assert len(result["ambiguous_specs"]) == 0
        assert len(result["variant_specs"]) == 1  # heavy → variant


# ---------------------------------------------------------------------------
# Tests for Issue #1158: transition_guidance stored on passage node in Phase 6
# ---------------------------------------------------------------------------


class TestTransitionGuidanceInGraph:
    """After Phase 6, passage nodes created from collapsed specs must have transition_guidance."""

    def test_transition_guidance_stored_on_passage_node(self) -> None:
        """Phase 6 stores transition_guidance from PassageSpec onto the passage graph node."""
        import asyncio

        from questfoundry.pipeline.stages.polish.deterministic import phase_plan_application

        graph = Graph.empty()

        # Build a minimal plan node directly (bypass phase 4) with a collapsed passage
        # that has transition_guidance set (simulating what phase 5f would produce).
        from questfoundry.models.polish import PassageSpec as _PassageSpec

        spec = _PassageSpec(
            passage_id="passage::collapse_0",
            beat_ids=["beat::a", "beat::b", "beat::c"],
            summary="Three beats collapsed",
            entities=[],
            grouping_type="collapse",
            transition_guidance=["Move from action to reflection.", "Time passes quietly."],
        )

        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "raw_id": "current",
                "passage_count": 1,
                "variant_count": 0,
                "residue_count": 0,
                "choice_count": 0,
                "candidate_count": 0,
                "warnings": [],
                "passage_specs": [spec.model_dump()],
                "variant_specs": [],
                "residue_specs": [],
                "choice_specs": [],
                "false_branch_candidates": [],
                "false_branch_specs": [],
                "feasibility_annotations": {},
                "ambiguous_specs": [],
                "arc_traversals": {},
            },
        )

        # Create beat nodes so grouped_in edges don't fail
        for bid in ["beat::a", "beat::b", "beat::c"]:
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "raw_id": bid.split("::")[-1],
                    "summary": f"Beat {bid}",
                    "dilemma_impacts": [],
                    "entities": [],
                    "scene_type": "scene",
                },
            )

        result = asyncio.run(phase_plan_application(graph, None))  # type: ignore[arg-type]
        assert result.status == "completed"

        passage_nodes = graph.get_nodes_by_type("passage")
        passage = passage_nodes.get("passage::collapse_0")
        assert passage is not None
        assert passage.get("transition_guidance") == [
            "Move from action to reflection.",
            "Time passes quietly.",
        ]


class TestPrePlanWarningAccumulator:
    """Tests for Issue #1159: Phase 1 warnings accumulated before PolishPlan exists."""

    def test_drain_pre_plan_warnings_drains_into_plan(self) -> None:
        """Warnings written to graph by Phase 1 are drained into plan.warnings (#1159)."""
        graph = Graph.empty()
        warnings = [
            "Section section_0: reordering rejected — beat set mismatch (expected 3, got 2)",
            "Section section_1: reordering rejected — hard constraint violation (commit before advance/reveal)",
        ]
        _upsert_pre_plan_warnings(graph, warnings)

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)

        assert plan.warnings == warnings

    def test_drain_clears_accumulator(self) -> None:
        """Draining pre-plan warnings clears the accumulator node (#1159)."""
        graph = Graph.empty()
        _upsert_pre_plan_warnings(graph, ["some warning"])

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert len(plan.warnings) == 1

        # Second drain should add nothing
        plan2 = PolishPlan()
        _drain_pre_plan_warnings(graph, plan2)
        assert len(plan2.warnings) == 0

    def test_drain_no_warnings_node_is_noop(self) -> None:
        """Draining when no pre-plan warnings node exists is a no-op (#1159)."""
        graph = Graph.empty()
        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)  # Should not raise
        assert plan.warnings == []

    def test_no_rejection_produces_empty_phase1_contribution(self) -> None:
        """When Phase 1 produces no rejections, plan.warnings is empty (#1159)."""
        graph = Graph.empty()
        # Do not call _upsert_pre_plan_warnings — simulates no rejections
        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert plan.warnings == []

    def test_upsert_creates_node_when_none_exists(self) -> None:
        """_upsert_pre_plan_warnings creates the node when it doesn't exist yet."""
        graph = Graph.empty()
        warnings = ["warning A", "warning B"]
        _upsert_pre_plan_warnings(graph, warnings)

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert plan.warnings == warnings

    def test_upsert_appends_to_existing_node(self) -> None:
        """_upsert_pre_plan_warnings with existing node replaces warnings list."""
        graph = Graph.empty()
        # First call creates the node
        _upsert_pre_plan_warnings(graph, ["first warning"])
        # Second call (simulating a second rejected section) overwrites with combined list
        _upsert_pre_plan_warnings(graph, ["first warning", "second warning"])

        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert plan.warnings == ["first warning", "second warning"]

    def test_upsert_empty_warnings_noop(self) -> None:
        """_upsert_pre_plan_warnings with an empty list is a no-op — no node is created."""
        graph = Graph.empty()
        _upsert_pre_plan_warnings(graph, [])

        assert graph.get_node(_PRE_PLAN_WARNINGS_NODE) is None
        plan = PolishPlan()
        _drain_pre_plan_warnings(graph, plan)
        assert plan.warnings == []


class TestFindFalseBranchCandidatesBranchingAndDisconnected:
    """Additional tests for find_false_branch_candidates covering branching passages
    and disconnected sub-graphs (#1161)."""

    def test_branching_passage_ends_run(self) -> None:
        """A passage with 2+ children in the passage graph ends the linear run.

        Scenario: p0 → p1, p0 → p2, then p1 → p3 → p4 → p5.
        p0 has two children so its run ends; p1..p5 form a 4-passage linear run.
        """
        graph = Graph.empty()
        # Beats: b0 is parent of b1 AND b2 (branch); b1 → b3 → b4 → b5 linear
        for i in range(6):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        # b1 comes after b0 (predecessor: child=b1, parent=b0)
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        # b2 also comes after b0 — creates 2-child branch at b0
        graph.add_edge("predecessor", "beat::b2", "beat::b0")
        # b3 → b4 → b5 come after b1
        graph.add_edge("predecessor", "beat::b3", "beat::b1")
        graph.add_edge("predecessor", "beat::b4", "beat::b3")
        graph.add_edge("predecessor", "beat::b5", "beat::b4")

        specs = [
            PassageSpec(passage_id=f"passage::p{i}", beat_ids=[f"beat::b{i}"], summary=f"P{i}")
            for i in range(6)
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # The b1→b3→b4→b5 arm forms a 4-passage linear run
        all_ids = [pid for c in candidates for pid in c.passage_ids]
        assert "passage::p1" in all_ids
        # p0 is a branching point — it should NOT appear alone as a linear run lead
        # (specifically p0 has 2 children so the walk breaks the run at p0)
        for c in candidates:
            assert len(c.passage_ids) >= 3

    def test_disconnected_passage_forms_linear_run(self) -> None:
        """Passages not reachable from any root (disconnected sub-graph) are
        still walked via the fallback loop and identified if 3+ consecutive."""
        graph = Graph.empty()
        # Two isolated chains:
        # chain A: b0 → b1 (2 passages — not a candidate)
        # chain B: b2 → b3 → b4 → b5 (4 passages — candidate)
        # No edge connecting A and B — chain B is unreachable from chain A's roots.
        for i in range(6):
            graph.create_node(
                f"beat::b{i}",
                {"type": "beat", "raw_id": f"b{i}", "summary": f"B{i}", "dilemma_impacts": []},
            )
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")
        graph.add_edge("predecessor", "beat::b4", "beat::b3")
        graph.add_edge("predecessor", "beat::b5", "beat::b4")

        specs = [
            PassageSpec(passage_id=f"passage::p{i}", beat_ids=[f"beat::b{i}"], summary=f"P{i}")
            for i in range(6)
        ]
        candidates = find_false_branch_candidates(graph, specs)
        # Chain B's 4 passages form one candidate
        assert len(candidates) == 1
        assert len(candidates[0].passage_ids) == 4


# ---------------------------------------------------------------------------
# Tests for Issue #1162: Overlay composition audit in Phase 4 (after 4b)
# ---------------------------------------------------------------------------


class TestAuditOverlayComposition:
    """_audit_overlay_composition must flag passages where any entity has 4+
    simultaneously active overlays under any reachable flag combination."""

    def _build_graph_with_overlays(
        self,
        overlay_when_lists: list[list[str]],
        commit_flags: list[tuple[str, str, str]],
    ) -> tuple[Graph, PassageSpec]:
        """Build a minimal graph with one entity having `len(overlay_when_lists)` overlays.

        Args:
            overlay_when_lists: Each element is the `when` list for one overlay.
            commit_flags: List of (path_id, dilemma_id, beat_id) tuples for commit beats
                that are ancestors of beat::target (chained in given order).

        Returns:
            (graph, spec) ready for _audit_overlay_composition.
        """

        graph = Graph.empty()

        overlays = [
            {"when": when, "details": {"key": f"val_{i}"}}
            for i, when in enumerate(overlay_when_lists)
        ]
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "overlays": overlays},
        )

        # Create paths + dilemmas
        for path_id, dilemma_id, _ in commit_flags:
            if path_id not in graph.get_nodes_by_type("path"):
                graph.create_node(path_id, {"type": "path", "raw_id": path_id.split("::")[-1]})
            if dilemma_id not in graph.get_nodes_by_type("dilemma"):
                graph.create_node(
                    dilemma_id, {"type": "dilemma", "raw_id": dilemma_id.split("::")[-1]}
                )

        # Create commit beats chained: last→...→first→target
        # Also create a state_flag node and grants edge for each commit beat.
        prev_beat = None
        for path_id, dilemma_id, beat_id in commit_flags:
            graph.create_node(
                beat_id,
                {
                    "type": "beat",
                    "raw_id": beat_id.split("::")[-1],
                    "summary": f"Commit on {path_id}",
                    "dilemma_impacts": [{"dilemma_id": dilemma_id, "effect": "commits"}],
                    "entities": [],
                    "scene_type": "scene",
                },
            )
            graph.add_edge("belongs_to", beat_id, path_id)
            # state_flag: named after path raw ID so callers can predict the ID
            path_raw = path_id.split("::")[-1]
            sf_id = f"state_flag::{path_raw}_committed"
            if not graph.get_node(sf_id):
                graph.create_node(
                    sf_id,
                    {
                        "type": "state_flag",
                        "raw_id": f"{path_raw}_committed",
                        "dilemma_id": dilemma_id,
                    },
                )
            graph.add_edge("grants", beat_id, sf_id)
            if prev_beat is not None:
                graph.add_edge("predecessor", beat_id, prev_beat)
            prev_beat = beat_id

        # Target beat containing entity::hero
        first_path = commit_flags[0][0] if commit_flags else "path::p1"
        if first_path not in graph.get_nodes_by_type("path"):
            graph.create_node(first_path, {"type": "path", "raw_id": first_path.split("::")[-1]})

        graph.create_node(
            "beat::target",
            {
                "type": "beat",
                "raw_id": "target",
                "summary": "Target beat",
                "dilemma_impacts": [],
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::target", first_path)
        if prev_beat is not None:
            graph.add_edge("predecessor", "beat::target", prev_beat)

        spec = PassageSpec(
            passage_id="passage::test",
            beat_ids=["beat::target"],
            summary="test passage",
            entities=["entity::hero"],
        )
        return graph, spec

    def test_four_overlays_all_coactive_flagged(self) -> None:
        """Entity with 4 overlays whose when-flags can all be simultaneously active
        → passage flagged as structural_split."""
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        # Four dilemmas, four paths, each overlay's when has exactly 1 flag
        # All four flags will be ancestors of beat::target → all 4 active in 1 combo
        commit_flags = [
            ("path::pa", "dilemma::da", "beat::ca"),
            ("path::pb", "dilemma::db", "beat::cb"),
            ("path::pc", "dilemma::dc", "beat::cc"),
            ("path::pd", "dilemma::dd", "beat::cd"),
        ]
        # Overlay when-flags use state_flag node IDs (state_flag::{path_raw}_committed)
        overlay_when_lists = [
            ["state_flag::pa_committed"],
            ["state_flag::pb_committed"],
            ["state_flag::pc_committed"],
            ["state_flag::pd_committed"],
        ]
        graph, spec = self._build_graph_with_overlays(overlay_when_lists, commit_flags)

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        assert len(feasibility["warnings"]) == 1
        assert "passage::test" in feasibility["warnings"][0]
        assert "structural split recommended" in feasibility["warnings"][0]

    def test_four_overlays_mutually_exclusive_not_flagged(self) -> None:
        """Entity with 4 overlays but each requires a different path of the SAME dilemma
        → at most 1 can be active at a time → passage NOT flagged."""
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        # One dilemma with four paths (mutually exclusive by belongs_to)
        # Only one commit beat → only one flag active at a time
        graph = Graph.empty()

        # Single dilemma, single commit beat for path::pa
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        graph.create_node("path::pc", {"type": "path", "raw_id": "pc"})
        graph.create_node("path::pd", {"type": "path", "raw_id": "pd"})

        # One commit beat on path::pa only, with state_flag node and grants edge
        graph.create_node(
            "beat::ca",
            {
                "type": "beat",
                "raw_id": "ca",
                "summary": "Commit a",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
                "entities": [],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::ca", "path::pa")
        graph.create_node(
            "state_flag::pa_committed",
            {"type": "state_flag", "raw_id": "pa_committed", "dilemma_id": "dilemma::d1"},
        )
        graph.add_edge("grants", "beat::ca", "state_flag::pa_committed")

        # Target beat
        graph.create_node(
            "beat::target",
            {
                "type": "beat",
                "raw_id": "target",
                "summary": "Target",
                "dilemma_impacts": [],
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::target", "path::pa")
        graph.add_edge("predecessor", "beat::target", "beat::ca")

        # Entity with 4 overlays, each requiring a different path of dilemma::d1.
        # Overlay when-flags use state_flag node IDs.
        # Only state_flag::pa_committed is active (beat::ca on path::pa was committed).
        # state_flag::pb/pc/pd_committed don't exist → 0 active overlays for those → not flagged.
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": ["state_flag::pa_committed"], "details": {}},
                    {"when": ["state_flag::pb_committed"], "details": {}},
                    {"when": ["state_flag::pc_committed"], "details": {}},
                    {"when": ["state_flag::pd_committed"], "details": {}},
                ],
            },
        )

        spec = PassageSpec(
            passage_id="passage::test",
            beat_ids=["beat::target"],
            summary="test",
            entities=["entity::hero"],
        )

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        # Only path::pa was committed → active flag combo is {state_flag::pa_committed}
        # Only 1 overlay matches that combo → not flagged
        assert feasibility["warnings"] == []

    def test_three_overlays_coactive_not_flagged(self) -> None:
        """Entity with exactly 3 simultaneously active overlays → NOT flagged (3 is manageable)."""
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        commit_flags = [
            ("path::pa", "dilemma::da", "beat::ca"),
            ("path::pb", "dilemma::db", "beat::cb"),
            ("path::pc", "dilemma::dc", "beat::cc"),
        ]
        # Overlay when-flags use state_flag node IDs (state_flag::{path_raw}_committed)
        overlay_when_lists = [
            ["state_flag::pa_committed"],
            ["state_flag::pb_committed"],
            ["state_flag::pc_committed"],
        ]
        graph, spec = self._build_graph_with_overlays(overlay_when_lists, commit_flags)

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        assert feasibility["warnings"] == []

    def test_already_structural_split_not_double_added(self) -> None:
        """If a passage is already flagged as structural_split in warnings, it is not added again."""
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        commit_flags = [
            ("path::pa", "dilemma::da", "beat::ca"),
            ("path::pb", "dilemma::db", "beat::cb"),
            ("path::pc", "dilemma::dc", "beat::cc"),
            ("path::pd", "dilemma::dd", "beat::cd"),
        ]
        # Overlay when-flags use state_flag node IDs (state_flag::{path_raw}_committed)
        overlay_when_lists = [
            ["state_flag::pa_committed"],
            ["state_flag::pb_committed"],
            ["state_flag::pc_committed"],
            ["state_flag::pd_committed"],
        ]
        graph, spec = self._build_graph_with_overlays(overlay_when_lists, commit_flags)

        # Pre-seed a structural_split warning for this passage (as Phase 4b would emit)
        existing_warning = (
            "Passage passage::test has 5 narratively relevant flags — structural split recommended"
        )
        feasibility: dict = {"warnings": [existing_warning]}
        _audit_overlay_composition(graph, [spec], feasibility)

        # Should still have exactly 1 warning (not double-added)
        assert len(feasibility["warnings"]) == 1
        assert feasibility["warnings"][0] == existing_warning

    def test_unconditional_overlays_always_active(self) -> None:
        """Overlays with empty ``when`` lists are always active regardless of flag combo.

        4 unconditional overlays (when=[]) on an entity in a passage → flagged
        as structural_split, even with no commit ancestors (empty flag combo).
        """
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        # Entity with 4 unconditional overlays (when: [])
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [], "details": {"key": "a"}},
                    {"when": [], "details": {"key": "b"}},
                    {"when": [], "details": {"key": "c"}},
                    {"when": [], "details": {"key": "d"}},
                ],
            },
        )

        # Single beat with no commit ancestors → empty flag combo
        graph.create_node(
            "beat::target",
            {
                "type": "beat",
                "raw_id": "target",
                "summary": "Target beat",
                "dilemma_impacts": [],
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::target", "path::p1")

        spec = PassageSpec(
            passage_id="passage::unconditional",
            beat_ids=["beat::target"],
            summary="unconditional overlay test",
            entities=["entity::hero"],
        )

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        assert len(feasibility["warnings"]) == 1
        assert "passage::unconditional" in feasibility["warnings"][0]
        assert "structural split recommended" in feasibility["warnings"][0]

    def test_entity_with_no_overlays_not_flagged(self) -> None:
        """Passage where the entity has no overlays is NOT flagged.

        Exercises the ``if not overlays: continue`` path.
        """
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        # Entity with no overlays at all
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero"},  # no "overlays" key
        )

        graph.create_node(
            "beat::target",
            {
                "type": "beat",
                "raw_id": "target",
                "summary": "Target beat",
                "dilemma_impacts": [],
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::target", "path::p1")

        spec = PassageSpec(
            passage_id="passage::no_overlays",
            beat_ids=["beat::target"],
            summary="no overlay test",
            entities=["entity::hero"],
        )

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        assert feasibility["warnings"] == []

    def test_entity_not_in_entity_nodes_skipped(self) -> None:
        """Passage referencing an entity ID not in the graph is NOT flagged.

        Exercises the ``if edata is None: continue`` path.
        """
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        # No entity node created for entity::ghost
        graph.create_node(
            "beat::target",
            {
                "type": "beat",
                "raw_id": "target",
                "summary": "Target beat",
                "dilemma_impacts": [],
                "entities": [],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::target", "path::p1")

        # PassageSpec references entity::ghost which does not exist in the graph
        spec = PassageSpec(
            passage_id="passage::ghost_entity",
            beat_ids=["beat::target"],
            summary="ghost entity test",
            entities=["entity::ghost"],
        )

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        assert feasibility["warnings"] == []

    def test_compute_active_flags_raises_warning_and_skips(self) -> None:
        """ValueError from compute_active_flags_at_beat → warning logged, beat skipped.

        Exercises the ``except ValueError`` path by referencing a beat_id that
        exists in the graph but is NOT a beat-type node. compute_active_flags_at_beat
        raises ValueError for non-beat nodes.
        """
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        # Entity with 4 unconditional overlays (normally would trigger threshold)
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [], "details": {"key": "a"}},
                    {"when": [], "details": {"key": "b"}},
                    {"when": [], "details": {"key": "c"}},
                    {"when": [], "details": {"key": "d"}},
                ],
            },
        )

        # A node that is NOT a beat type — compute_active_flags_at_beat raises ValueError
        graph.create_node(
            "beat::not_really_a_beat",
            {"type": "path", "raw_id": "not_really_a_beat"},  # wrong type
        )

        # A real beat so we have at least one valid combo (empty)
        graph.create_node(
            "beat::real",
            {
                "type": "beat",
                "raw_id": "real",
                "summary": "Real beat",
                "dilemma_impacts": [],
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::real", "path::p1")

        spec = PassageSpec(
            passage_id="passage::mixed_beats",
            beat_ids=["beat::not_really_a_beat", "beat::real"],
            summary="mixed beat types",
            entities=["entity::hero"],
        )

        feasibility: dict = {"warnings": []}
        # Should not raise; ValueError for the non-beat is caught and logged
        _audit_overlay_composition(graph, [spec], feasibility)

        # The real beat provides an empty flag combo → 4 unconditional overlays
        # are all active → passage IS flagged (the ValueError path was exercised
        # but the valid beat still produces a combo)
        assert any("structural split" in w for w in feasibility["warnings"])

    def test_passage_with_no_flag_combos_not_flagged(self) -> None:
        """Passage where all beats raise ValueError → no flag combos → NOT flagged.

        Exercises the ``if not all_flag_combos: continue`` path. When every
        beat_id in the spec raises ValueError, all_flag_combos stays empty
        and the passage is skipped entirely.
        """
        from questfoundry.pipeline.stages.polish.deterministic import _audit_overlay_composition

        graph = Graph.empty()

        # Entity with 4 unconditional overlays (would trigger threshold if evaluated)
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "overlays": [
                    {"when": [], "details": {"key": "a"}},
                    {"when": [], "details": {"key": "b"}},
                    {"when": [], "details": {"key": "c"}},
                    {"when": [], "details": {"key": "d"}},
                ],
            },
        )

        # Non-beat node in beat position → ValueError on every beat_id in spec
        graph.create_node(
            "beat::not_a_beat",
            {"type": "path", "raw_id": "not_a_beat"},  # wrong type
        )

        spec = PassageSpec(
            passage_id="passage::no_combos",
            beat_ids=["beat::not_a_beat"],
            summary="no combos test",
            entities=["entity::hero"],
        )

        feasibility: dict = {"warnings": []}
        _audit_overlay_composition(graph, [spec], feasibility)

        # all_flag_combos is empty (all beats raised ValueError) → passage skipped
        assert not any("structural split" in w for w in feasibility["warnings"])


# ---------------------------------------------------------------------------
# Y-shape collapse guard-rail tests (Task 3.3)
# ---------------------------------------------------------------------------


# DELETED: test_collapse_chain_does_not_join_shared_with_post_commit
# Removed as part of cluster #1311 (maximal-linear-collapse, R-4a.3).
# The test asserted Y-shape guard rail 2: a shared pre-commit beat (dual
# belongs_to) must not collapse into the same passage as its single-belongs_to
# commit successor.
#
# Under the new maximal-linear-collapse rule (R-4a.3), collapse decisions are
# made purely by DAG topology — the belongs_to set is NOT consulted.  In the
# `_build_y_shape_for_collapse` fixture, shared_setup → commit_a is a linear
# chain (commit_a has exactly one predecessor and shared_setup has exactly one
# successor).  The new algorithm correctly collapses them into one passage.
#
# The old guard rail (belongs_to-set mismatch prevents collapse) is superseded.
# The Y-shape fixture helper (`_build_y_shape_for_collapse`) and the test that
# used it have both been removed — no active test exercises this shape under
# the new rule.  If a new structural constraint is needed to keep shared/commit
# beats separate, it must be specified in docs/design/procedures/polish.md
# first, then
# implemented and tested here.
