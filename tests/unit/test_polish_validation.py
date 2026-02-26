"""Tests for POLISH Phase 7 validation (exit contract).

Tests validate_polish_output() which checks the passage graph
produced by Phases 4-6.
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import validate_polish_output
from questfoundry.models.polish import (
    ChoiceSpec,
    PassageSpec,
    PolishResult,
    ResidueSpec,
    VariantSpec,
)
from questfoundry.pipeline.stages.polish.deterministic import (
    _create_choice_edge,
    _create_passage_node,
    _create_residue_beat_and_passage,
    _create_variant_passage,
)


def _make_beat(graph: Graph, beat_id: str, summary: str = "A beat", **kwargs: object) -> None:
    """Helper to create a beat node."""
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


def _build_valid_graph() -> Graph:
    """Build a minimal valid passage graph for testing."""
    graph = Graph.empty()
    graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})

    _make_beat(graph, "beat::start", "Start")
    _make_beat(graph, "beat::end", "End")
    graph.add_edge("predecessor", "beat::end", "beat::start")

    # Create passages
    _create_passage_node(
        graph,
        PassageSpec(passage_id="passage::start", beat_ids=["beat::start"], summary="Start"),
    )
    _create_passage_node(
        graph,
        PassageSpec(passage_id="passage::end", beat_ids=["beat::end"], summary="End"),
    )

    # Add a choice edge
    _create_choice_edge(
        graph,
        ChoiceSpec(from_passage="passage::start", to_passage="passage::end", label="Continue"),
    )

    return graph


class TestValidatePolishOutputStructural:
    """Tests for structural completeness checks."""

    def test_valid_graph_passes(self) -> None:
        graph = _build_valid_graph()
        errors = validate_polish_output(graph)
        assert errors == []

    def test_no_passages_fails(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")
        errors = validate_polish_output(graph)
        assert any("No passage nodes" in e for e in errors)

    def test_beat_not_grouped_fails(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")
        _make_beat(graph, "beat::b", "B beat")

        # Only group beat::a
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::a"], summary="A"),
        )

        errors = validate_polish_output(graph)
        assert any("beat::b" in e and "not grouped" in e for e in errors)

    def test_beat_double_grouped_fails(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::a"], summary="P1"),
        )
        # Manually add a second grouped_in edge
        graph.create_node("passage::p2", {"type": "passage", "raw_id": "p2", "summary": "P2"})
        graph.add_edge("grouped_in", "beat::a", "passage::p2")

        errors = validate_polish_output(graph)
        assert any("beat::a" in e and "2 passages" in e for e in errors)

    def test_passage_without_beats_fails(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::p1", beat_ids=["beat::a"], summary="P1"),
        )
        # Create an empty passage (no beats)
        graph.create_node(
            "passage::empty",
            {"type": "passage", "raw_id": "empty", "summary": "Empty"},
        )

        errors = validate_polish_output(graph)
        assert any("passage::empty" in e and "no beats" in e for e in errors)

    def test_multiple_start_passages(self) -> None:
        graph = Graph.empty()
        # Two root beats (no predecessor edges)
        _make_beat(graph, "beat::root1", "Root 1")
        _make_beat(graph, "beat::root2", "Root 2")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::s1", beat_ids=["beat::root1"], summary="S1"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::s2", beat_ids=["beat::root2"], summary="S2"),
        )

        errors = validate_polish_output(graph)
        assert any("Multiple start passages" in e for e in errors)


class TestValidatePolishOutputVariants:
    """Tests for variant integrity checks."""

    def test_variant_without_edge_fails(self) -> None:
        graph = _build_valid_graph()

        # Create a variant passage without variant_of edge
        graph.create_node(
            "passage::bad_variant",
            {
                "type": "passage",
                "raw_id": "bad_variant",
                "is_variant": True,
                "summary": "Bad variant",
            },
        )

        errors = validate_polish_output(graph)
        assert any("bad_variant" in e and "variant_of" in e for e in errors)

    def test_proper_variant_passes(self) -> None:
        graph = _build_valid_graph()

        _create_variant_passage(
            graph,
            VariantSpec(
                base_passage_id="passage::start",
                variant_id="passage::v1",
                requires=["flag1"],
                summary="Variant",
            ),
        )

        errors = validate_polish_output(graph)
        assert not any("variant_of" in e for e in errors)


class TestValidatePolishOutputChoices:
    """Tests for choice integrity checks."""

    def test_duplicate_labels_detected(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::start", "Start")
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::start", beat_ids=["beat::start"], summary="Start"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::a", beat_ids=["beat::a"], summary="A"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::b", beat_ids=["beat::b"], summary="B"),
        )

        # Add two choices with the same label
        graph.add_edge("choice", "passage::start", "passage::a", label="Go")
        graph.add_edge("choice", "passage::start", "passage::b", label="Go")

        errors = validate_polish_output(graph)
        assert any("duplicate choice labels" in e for e in errors)

    def test_unique_labels_pass(self) -> None:
        graph = _build_valid_graph()
        errors = validate_polish_output(graph)
        assert not any("duplicate" in e for e in errors)


class TestValidatePolishOutputResidue:
    """Tests for residue ordering checks."""

    def test_residue_without_precedes_fails(self) -> None:
        graph = _build_valid_graph()

        # Create a residue passage without precedes edge
        graph.create_node(
            "passage::bad_residue",
            {
                "type": "passage",
                "raw_id": "bad_residue",
                "is_residue": True,
                "summary": "Bad residue",
            },
        )

        errors = validate_polish_output(graph)
        assert any("bad_residue" in e and "precedes" in e for e in errors)

    def test_proper_residue_passes(self) -> None:
        graph = _build_valid_graph()

        _create_residue_beat_and_passage(
            graph,
            ResidueSpec(
                target_passage_id="passage::start",
                residue_id="residue::r1",
                flag="flag1",
                content_hint="Mood",
            ),
        )

        errors = validate_polish_output(graph)
        assert not any("precedes" in e for e in errors)


class TestPolishResult:
    """Tests for the PolishResult model."""

    def test_default_values(self) -> None:
        result = PolishResult()
        assert result.passage_count == 0
        assert result.choice_count == 0
        assert result.variant_count == 0

    def test_with_values(self) -> None:
        result = PolishResult(
            passage_count=10,
            choice_count=5,
            variant_count=2,
            residue_count=3,
            sidetrack_count=1,
            false_branch_count=1,
        )
        assert result.passage_count == 10
        assert result.false_branch_count == 1
