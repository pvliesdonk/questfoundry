"""Tests for POLISH Phase 4 deterministic plan computation.

Tests 4a (beat grouping), 4b (feasibility audit),
4c (choice edge derivation), and 4d (false branch candidates).
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.models.polish import PassageSpec
from questfoundry.pipeline.stages.polish.deterministic import (
    PolishPlan,
    compute_beat_grouping,
    compute_choice_edges,
    compute_prose_feasibility,
    find_false_branch_candidates,
)


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
        """Sequential same-path beats collapse into one passage."""
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
        collapse_specs = [s for s in specs if s.grouping_type == "collapse"]
        assert len(collapse_specs) == 1
        assert len(collapse_specs[0].beat_ids) == 3

    def test_intersection_grouping(self) -> None:
        """Beats in intersection groups form intersection passages."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        # Create intersection group
        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "node_ids": ["beat::a", "beat::b"],
            },
        )

        specs = compute_beat_grouping(graph)
        intersection_specs = [s for s in specs if s.grouping_type == "intersection"]
        assert len(intersection_specs) == 1
        assert set(intersection_specs[0].beat_ids) == {"beat::a", "beat::b"}

    def test_different_paths_dont_collapse(self) -> None:
        """Sequential beats on different paths don't collapse."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        # Should be singletons, not collapsed
        assert all(s.grouping_type == "singleton" for s in specs)

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
        """Collapsed passage contains union of all beat entities."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "A", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "B", entities=["entity::hero", "entity::mentor"])
        _add_belongs_to(graph, "beat::a", "path::p1")
        _add_belongs_to(graph, "beat::b", "path::p1")
        _add_predecessor(graph, "beat::b", "beat::a")

        specs = compute_beat_grouping(graph)
        collapse_specs = [s for s in specs if s.grouping_type == "collapse"]
        assert len(collapse_specs) == 1
        assert set(collapse_specs[0].entities) == {"entity::hero", "entity::mentor"}

    def test_empty_graph(self) -> None:
        """Empty graph produces no specs."""
        graph = Graph.empty()
        specs = compute_beat_grouping(graph)
        assert specs == []


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

        # Commit beat as ancestor
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

        # Overlay for a DIFFERENT entity than the passage's entity (embedded on entity node)
        graph.create_node(
            "entity::npc",
            {
                "type": "entity",
                "raw_id": "npc",
                "overlays": [
                    {
                        "when": ["dilemma::d1:path::brave"],
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


class TestComputeChoiceEdges:
    """Tests for Phase 4c: choice edge derivation."""

    def test_simple_divergence(self) -> None:
        """Beat with children on different paths produces choices."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start", entities=["entity::hero"])
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

    def test_no_divergence(self) -> None:
        """Linear chain produces no choices."""
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
        assert len(choices) == 0

    def test_grants_from_commit(self) -> None:
        """Choice leading to a commit beat populates grants."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start")
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
        """3+ passages in a row → at least one candidate."""
        graph = Graph.empty()
        specs = [
            PassageSpec(passage_id=f"p{i}", beat_ids=[f"b{i}"], summary=f"P{i}") for i in range(5)
        ]
        candidates = find_false_branch_candidates(graph, specs)
        assert len(candidates) >= 1
        assert all(len(c.passage_ids) >= 3 for c in candidates)


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
