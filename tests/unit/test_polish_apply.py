"""Tests for POLISH Phase 6: atomic plan application.

Tests that the plan application creates correct nodes and edges
on the graph from passage specs, variant specs, residue specs,
choice specs, and false branch specs.
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.models.polish import (
    ChoiceSpec,
    FalseBranchSpec,
    PassageSpec,
    ResidueSpec,
    VariantSpec,
)
from questfoundry.pipeline.stages.polish.deterministic import (
    _apply_diamond,
    _apply_sidetrack,
    _create_choice_edge,
    _create_passage_node,
    _create_residue_beat_and_passage,
    _create_variant_passage,
)


def _make_beat(graph: Graph, beat_id: str, summary: str = "A beat") -> None:
    """Helper to create a beat node."""
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": beat_id.split("::")[-1],
            "summary": summary,
            "dilemma_impacts": [],
            "entities": [],
            "scene_type": "scene",
        },
    )


class TestCreatePassageNode:
    """Tests for _create_passage_node."""

    def test_creates_passage_with_grouped_in(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")

        spec = PassageSpec(
            passage_id="passage::test",
            beat_ids=["beat::a", "beat::b"],
            summary="Test passage",
            entities=["entity::hero"],
            grouping_type="collapse",
        )
        _create_passage_node(graph, spec)

        passages = graph.get_nodes_by_type("passage")
        assert "passage::test" in passages
        assert passages["passage::test"]["summary"] == "Test passage"
        assert passages["passage::test"]["grouping_type"] == "collapse"

        # Check grouped_in edges
        edges = graph.get_edges(edge_type="grouped_in")
        grouped_from = {e["from"] for e in edges}
        grouped_to = {e["to"] for e in edges}
        assert "beat::a" in grouped_from
        assert "beat::b" in grouped_from
        assert grouped_to == {"passage::test"}

    def test_singleton_passage(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::single")

        spec = PassageSpec(
            passage_id="passage::s1",
            beat_ids=["beat::single"],
            summary="Single beat",
        )
        _create_passage_node(graph, spec)

        passages = graph.get_nodes_by_type("passage")
        assert "passage::s1" in passages
        edges = graph.get_edges(edge_type="grouped_in")
        assert len(edges) == 1


class TestCreateVariantPassage:
    """Tests for _create_variant_passage."""

    def test_creates_variant_with_variant_of_edge(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a")

        # Create base passage first
        base_spec = PassageSpec(
            passage_id="passage::base",
            beat_ids=["beat::a"],
            summary="Base passage",
        )
        _create_passage_node(graph, base_spec)

        # Create variant
        vspec = VariantSpec(
            base_passage_id="passage::base",
            variant_id="passage::variant_0",
            requires=["flag1"],
            summary="Variant version",
        )
        _create_variant_passage(graph, vspec)

        passages = graph.get_nodes_by_type("passage")
        assert "passage::variant_0" in passages
        assert passages["passage::variant_0"]["is_variant"] is True
        assert passages["passage::variant_0"]["requires"] == ["flag1"]

        # Check variant_of edge
        edges = graph.get_edges(edge_type="variant_of")
        assert len(edges) == 1
        assert edges[0]["from"] == "passage::variant_0"
        assert edges[0]["to"] == "passage::base"


class TestCreateResidueBeatAndPassage:
    """Tests for _create_residue_beat_and_passage."""

    def test_creates_residue_beat_and_passage(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::target")
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})

        # Create target passage
        target_spec = PassageSpec(
            passage_id="passage::target",
            beat_ids=["beat::target"],
            summary="Target",
        )
        _create_passage_node(graph, target_spec)

        rspec = ResidueSpec(
            target_passage_id="passage::target",
            residue_id="residue::r1",
            flag="dilemma::d1:path::brave",
            path_id="path::brave",
            content_hint="You feel confident",
        )
        _create_residue_beat_and_passage(graph, rspec)

        # Check residue beat was created (residue_ prefix prevents ID collision)
        beat_nodes = graph.get_nodes_by_type("beat")
        residue_beat = beat_nodes.get("beat::residue_r1")
        assert residue_beat is not None
        assert residue_beat["role"] == "residue_beat"
        assert residue_beat["summary"] == "You feel confident"

        # Check residue passage was created
        passages = graph.get_nodes_by_type("passage")
        residue_passage = passages.get("passage::residue_r1")
        assert residue_passage is not None
        assert residue_passage["is_residue"] is True
        assert residue_passage["requires"] == ["dilemma::d1:path::brave"]

        # Check grouped_in edge
        grouped_edges = graph.get_edges(edge_type="grouped_in")
        residue_grouped = [e for e in grouped_edges if e["from"] == "beat::residue_r1"]
        assert len(residue_grouped) == 1
        assert residue_grouped[0]["to"] == "passage::residue_r1"

        # Check precedes edge
        precedes = graph.get_edges(edge_type="precedes")
        assert len(precedes) == 1
        assert precedes[0]["from"] == "passage::residue_r1"
        assert precedes[0]["to"] == "passage::target"

    def test_residue_without_content_hint_uses_default(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::target")

        target_spec = PassageSpec(
            passage_id="passage::target",
            beat_ids=["beat::target"],
            summary="Target",
        )
        _create_passage_node(graph, target_spec)

        rspec = ResidueSpec(
            target_passage_id="passage::target",
            residue_id="residue::r2",
            flag="flag1",
        )
        _create_residue_beat_and_passage(graph, rspec)

        beat_nodes = graph.get_nodes_by_type("beat")
        assert "beat::residue_r2" in beat_nodes
        assert "Residue moment for" in beat_nodes["beat::residue_r2"]["summary"]


class TestCreateChoiceEdge:
    """Tests for _create_choice_edge."""

    def test_creates_choice_edge_with_label(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")

        for pid, bid in [("passage::p1", "beat::a"), ("passage::p2", "beat::b")]:
            _create_passage_node(
                graph,
                PassageSpec(passage_id=pid, beat_ids=[bid], summary="s"),
            )

        cspec = ChoiceSpec(
            from_passage="passage::p1",
            to_passage="passage::p2",
            label="Go north",
            grants=["flag1"],
        )
        _create_choice_edge(graph, cspec)

        edges = graph.get_edges(edge_type="choice")
        assert len(edges) == 1
        assert edges[0]["from"] == "passage::p1"
        assert edges[0]["to"] == "passage::p2"

    def test_choice_edge_without_label(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")

        for pid, bid in [("passage::p1", "beat::a"), ("passage::p2", "beat::b")]:
            _create_passage_node(
                graph,
                PassageSpec(passage_id=pid, beat_ids=[bid], summary="s"),
            )

        cspec = ChoiceSpec(
            from_passage="passage::p1",
            to_passage="passage::p2",
        )
        _create_choice_edge(graph, cspec)

        edges = graph.get_edges(edge_type="choice")
        assert len(edges) == 1


class TestApplyDiamond:
    """Tests for _apply_diamond."""

    def test_diamond_creates_alternatives_and_choices(self) -> None:
        graph = Graph.empty()
        # Create 3 passages for the candidate stretch
        for i in range(3):
            _make_beat(graph, f"beat::b{i}")
            _create_passage_node(
                graph,
                PassageSpec(
                    passage_id=f"passage::p{i}",
                    beat_ids=[f"beat::b{i}"],
                    summary=f"Passage {i}",
                ),
            )

        fb_spec = FalseBranchSpec(
            candidate_passage_ids=["passage::p0", "passage::p1", "passage::p2"],
            branch_type="diamond",
            diamond_summary_a="Careful approach",
            diamond_summary_b="Bold approach",
        )

        beats, choices = _apply_diamond(graph, fb_spec)
        assert beats == 0
        assert choices == 4

        # Check alt passages exist
        passages = graph.get_nodes_by_type("passage")
        assert "passage::p1_alt_a" in passages
        assert "passage::p1_alt_b" in passages
        assert passages["passage::p1_alt_a"]["is_diamond_alt"] is True

        # Check choice edges exist
        choice_edges = graph.get_edges(edge_type="choice")
        assert len(choice_edges) == 4

    def test_diamond_too_few_passages(self) -> None:
        graph = Graph.empty()
        fb_spec = FalseBranchSpec(
            candidate_passage_ids=["p1", "p2"],
            branch_type="diamond",
        )
        beats, choices = _apply_diamond(graph, fb_spec)
        assert beats == 0
        assert choices == 0


class TestApplySidetrack:
    """Tests for _apply_sidetrack."""

    def test_sidetrack_creates_detour(self) -> None:
        graph = Graph.empty()
        for i in range(3):
            _make_beat(graph, f"beat::b{i}")
            _create_passage_node(
                graph,
                PassageSpec(
                    passage_id=f"passage::p{i}",
                    beat_ids=[f"beat::b{i}"],
                    summary=f"Passage {i}",
                ),
            )

        fb_spec = FalseBranchSpec(
            candidate_passage_ids=["passage::p0", "passage::p1", "passage::p2"],
            branch_type="sidetrack",
            sidetrack_summary="Meet a stranger",
            sidetrack_entities=["entity::stranger"],
            choice_label_enter="Approach",
            choice_label_return="Move on",
        )

        beats, choices = _apply_sidetrack(graph, fb_spec)
        assert beats == 1
        assert choices == 2

        # Check sidetrack beat exists
        beat_nodes = graph.get_nodes_by_type("beat")
        sidetrack_beats = {k: v for k, v in beat_nodes.items() if v.get("role") == "sidetrack_beat"}
        assert len(sidetrack_beats) == 1
        sb = next(iter(sidetrack_beats.values()))
        assert sb["summary"] == "Meet a stranger"
        assert sb["entities"] == ["entity::stranger"]

        # Check sidetrack passage exists
        passages = graph.get_nodes_by_type("passage")
        sidetrack_passages = {k: v for k, v in passages.items() if v.get("is_sidetrack")}
        assert len(sidetrack_passages) == 1

        # Check choice edges
        choice_edges = graph.get_edges(edge_type="choice")
        assert len(choice_edges) == 2

    def test_sidetrack_too_few_passages(self) -> None:
        graph = Graph.empty()
        fb_spec = FalseBranchSpec(
            candidate_passage_ids=["p1"],
            branch_type="sidetrack",
        )
        beats, choices = _apply_sidetrack(graph, fb_spec)
        assert beats == 0
        assert choices == 0


class TestPhase6Integration:
    """Integration tests combining multiple application steps."""

    def test_full_plan_application(self) -> None:
        """Apply a complete plan with passages, choices, and a variant."""
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::start", "Start")
        _make_beat(graph, "beat::a", "Path A")
        _make_beat(graph, "beat::b", "Path B")

        # Apply passage specs
        specs = [
            PassageSpec(
                passage_id="passage::start",
                beat_ids=["beat::start"],
                summary="Starting point",
            ),
            PassageSpec(
                passage_id="passage::a",
                beat_ids=["beat::a"],
                summary="Path A scene",
            ),
            PassageSpec(
                passage_id="passage::b",
                beat_ids=["beat::b"],
                summary="Path B scene",
            ),
        ]
        for spec in specs:
            _create_passage_node(graph, spec)

        # Apply choices
        choices = [
            ChoiceSpec(
                from_passage="passage::start",
                to_passage="passage::a",
                label="Take path A",
            ),
            ChoiceSpec(
                from_passage="passage::start",
                to_passage="passage::b",
                label="Take path B",
            ),
        ]
        for cspec in choices:
            _create_choice_edge(graph, cspec)

        # Apply variant
        vspec = VariantSpec(
            base_passage_id="passage::a",
            variant_id="passage::a_v1",
            requires=["flag1"],
            summary="Path A variant",
        )
        _create_variant_passage(graph, vspec)

        # Verify graph state
        passages = graph.get_nodes_by_type("passage")
        assert len(passages) == 4  # 3 base + 1 variant

        choice_edges = graph.get_edges(edge_type="choice")
        assert len(choice_edges) == 2

        grouped_edges = graph.get_edges(edge_type="grouped_in")
        assert len(grouped_edges) == 3  # 3 beats grouped into passages

        variant_edges = graph.get_edges(edge_type="variant_of")
        assert len(variant_edges) == 1

    def test_empty_plan_application(self) -> None:
        """Applying an empty plan creates nothing."""
        graph = Graph.empty()
        # No specs to apply — just verify no crash
        passages = graph.get_nodes_by_type("passage")
        assert len(passages) == 0
