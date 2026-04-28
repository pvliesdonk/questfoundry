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
    VariantSpec,
)
from questfoundry.pipeline.stages.polish.deterministic import (
    _create_choice_edge,
    _create_passage_node,
    _create_variant_passage,
)


def _make_beat(graph: Graph, beat_id: str, summary: str = "A beat", **kwargs: object) -> None:
    """Helper to create a beat node."""
    polish_created_roles = frozenset({"micro_beat", "residue_beat", "false_branch_beat"})
    data = {
        "type": "beat",
        "raw_id": beat_id.split("::")[-1],
        "summary": summary,
        "dilemma_impacts": [],
        "entities": [],
        "scene_type": "scene",
    }
    data.update(kwargs)
    # Add created_by attribution for POLISH-created beats (R-2.5)
    role = data.get("role", "")
    if role in polish_created_roles and "created_by" not in data:
        data["created_by"] = "POLISH"
    graph.create_node(beat_id, data)


def _build_valid_graph() -> Graph:
    """Build a minimal valid passage graph for testing.

    Graph shape: beat::start diverges into beat::end and beat::alt_end,
    forming a Y-fork.  This makes beat::start a genuine divergence point
    (out-degree 2), so the passage boundary between passage::start and
    the two successor passages is valid under R-4a.4 maximal-linear-collapse.
    """
    graph = Graph.empty()
    graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})

    _make_beat(graph, "beat::start", "Start")
    _make_beat(graph, "beat::end", "End")
    _make_beat(graph, "beat::alt_end", "Alt end")
    # beat::start → beat::end and beat::start → beat::alt_end (Y-fork)
    graph.add_edge("predecessor", "beat::end", "beat::start")
    graph.add_edge("predecessor", "beat::alt_end", "beat::start")

    # Create passages — each branch is its own passage; passage::start
    # ends at the divergence point which has out-degree 2 (valid boundary).
    _create_passage_node(
        graph,
        PassageSpec(passage_id="passage::start", beat_ids=["beat::start"], summary="Start"),
    )
    _create_passage_node(
        graph,
        PassageSpec(passage_id="passage::end", beat_ids=["beat::end"], summary="End"),
    )
    _create_passage_node(
        graph,
        PassageSpec(passage_id="passage::alt_end", beat_ids=["beat::alt_end"], summary="Alt end"),
    )

    # Add choice edges
    _create_choice_edge(
        graph,
        ChoiceSpec(from_passage="passage::start", to_passage="passage::end", label="Continue"),
    )
    _create_choice_edge(
        graph,
        ChoiceSpec(
            from_passage="passage::start", to_passage="passage::alt_end", label="Alt continue"
        ),
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


class TestResidueExemption:
    """Residue beats are no longer exempt from grouping checks."""

    def test_residue_beat_not_grouped_errors(self) -> None:
        """A residue_beat without a grouped_in edge should now produce an error."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")
        _make_beat(graph, "beat::residue", "Residue", role="residue_beat")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::pa", beat_ids=["beat::a"], summary="A"),
        )
        # beat::residue intentionally not grouped

        errors = validate_polish_output(graph)
        assert any("beat::residue" in e and "not grouped" in e for e in errors)

    def test_micro_beat_not_grouped_still_exempt(self) -> None:
        """micro_beat without grouped_in edge should NOT produce an error."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A beat")
        _make_beat(graph, "beat::micro", "Micro", role="micro_beat")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::pa", beat_ids=["beat::a"], summary="A"),
        )
        # beat::micro intentionally not grouped

        errors = validate_polish_output(graph)
        assert not any("beat::micro" in e and "not grouped" in e for e in errors)


class TestArcCompleteness:
    """Tests for _check_arc_completeness."""

    def test_repeated_passage_id_errors(self) -> None:
        """Arc with repeated passage ID should produce an error."""
        graph = _build_valid_graph()
        # Store a plan with a repeated passage
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "arc_traversals": {
                    "arc1": ["passage::start", "passage::start", "passage::end"],
                },
                "warnings": [],
            },
        )
        errors = validate_polish_output(graph)
        assert any("Arc arc1" in e and "repeated" in e for e in errors)

    def test_last_passage_with_outgoing_choices_errors(self) -> None:
        """Arc whose last passage has outgoing choices should produce an error."""
        graph = _build_valid_graph()
        # passage::start has an outgoing choice to passage::end — make it the last in arc
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "arc_traversals": {
                    "arc1": ["passage::start"],  # last passage has outgoing choices
                },
                "warnings": [],
            },
        )
        errors = validate_polish_output(graph)
        assert any("Arc arc1" in e and "outgoing choices" in e for e in errors)

    def test_valid_arc_passes(self) -> None:
        """Valid arc (no repeated passages, last passage is an ending) passes."""
        graph = _build_valid_graph()
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "arc_traversals": {
                    "arc1": ["passage::start", "passage::end"],
                },
                "warnings": [],
            },
        )
        errors = validate_polish_output(graph)
        assert not any("Arc arc1" in e for e in errors)


class TestNoOverlappingRequires:
    """Tests for _check_no_overlapping_requires."""

    def test_overlapping_requires_errors(self) -> None:
        """Two choices from same passage with overlapping requires should error."""
        graph = _build_valid_graph()
        # Add extra passages to route to
        _make_beat(graph, "beat::x", "X")
        _make_beat(graph, "beat::y", "Y")
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::x", beat_ids=["beat::x"], summary="X")
        )
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::y", beat_ids=["beat::y"], summary="Y")
        )
        graph.add_edge("choice", "passage::end", "passage::x", requires=["flagA", "flagB"])
        graph.add_edge("choice", "passage::end", "passage::y", requires=["flagB", "flagC"])

        errors = validate_polish_output(graph)
        assert any("overlapping requires" in e and "passage::end" in e for e in errors)

    def test_non_overlapping_requires_passes(self) -> None:
        """Two choices with non-overlapping requires should not error."""
        graph = _build_valid_graph()
        _make_beat(graph, "beat::x", "X")
        _make_beat(graph, "beat::y", "Y")
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::x", beat_ids=["beat::x"], summary="X")
        )
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::y", beat_ids=["beat::y"], summary="Y")
        )
        graph.add_edge("choice", "passage::end", "passage::x", requires=["flagA"])
        graph.add_edge("choice", "passage::end", "passage::y", requires=["flagB"])

        errors = validate_polish_output(graph)
        assert not any("overlapping requires" in e for e in errors)


class TestVariantRequiresNonEmpty:
    """Tests for _check_variant_requires_non_empty."""

    def test_variant_with_empty_requires_errors(self) -> None:
        """Variant passage with empty requires on the node itself should error."""
        graph = _build_valid_graph()
        # requires=[] (empty) — node-level gate is missing
        _create_variant_passage(
            graph,
            VariantSpec(
                base_passage_id="passage::start",
                variant_id="passage::v1",
                requires=[],
                summary="Variant",
            ),
        )
        _make_beat(graph, "beat::dest", "Dest")
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::dest", beat_ids=["beat::dest"], summary="Dest")
        )
        graph.add_edge("choice", "passage::v1", "passage::dest")

        errors = validate_polish_output(graph)
        assert any("passage::v1" in e and "cannot gate player entry" in e for e in errors)

    def test_variant_with_requires_passes(self) -> None:
        """Variant passage with non-empty requires on the node itself should pass."""
        graph = _build_valid_graph()
        _create_variant_passage(
            graph,
            VariantSpec(
                base_passage_id="passage::start",
                variant_id="passage::v1",
                requires=["flagA"],
                summary="Variant",
            ),
        )
        _make_beat(graph, "beat::dest", "Dest")
        _create_passage_node(
            graph, PassageSpec(passage_id="passage::dest", beat_ids=["beat::dest"], summary="Dest")
        )
        graph.add_edge("choice", "passage::v1", "passage::dest", requires=["flagA"])

        errors = validate_polish_output(graph)
        assert not any("cannot gate player entry" in e for e in errors)


class TestNoUnresolvedSplits:
    """Tests for _check_no_unresolved_splits."""

    def test_structural_split_warning_without_variant_errors(self) -> None:
        """A structural split warning without a variant passage should error."""
        graph = _build_valid_graph()
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "arc_traversals": {},
                "warnings": [
                    "Passage passage::start has 4 narratively relevant flags — structural split recommended"
                ],
            },
        )
        errors = validate_polish_output(graph)
        assert any("passage::start" in e and "structural split warning" in e for e in errors)

    def test_structural_split_warning_with_variant_passes(self) -> None:
        """A structural split warning resolved by a variant passage should pass."""
        graph = _build_valid_graph()
        graph.create_node(
            "polish_plan::current",
            {
                "type": "polish_plan",
                "arc_traversals": {},
                "warnings": [
                    "Passage passage::start has 4 narratively relevant flags — structural split recommended"
                ],
            },
        )
        # Create a variant for passage::start
        _create_variant_passage(
            graph,
            VariantSpec(
                base_passage_id="passage::start",
                variant_id="passage::v1",
                requires=["flagA"],
                summary="Variant",
            ),
        )

        errors = validate_polish_output(graph)
        assert not any("structural split warning" in e for e in errors)


class TestDivergencesHaveChoices:
    """Tests for _check_divergences_have_choices."""

    def test_divergence_beat_with_insufficient_choices_errors(self) -> None:
        """A divergence beat whose passage has only 1 outgoing choice should error."""
        graph = Graph.empty()

        # Two paths
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        # Beats: divergence beat with two children on different paths
        _make_beat(graph, "beat::div", "Divergence beat")
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")

        # belongs_to edges — children on different paths
        graph.add_edge("belongs_to", "beat::div", "path::pa")
        graph.add_edge("belongs_to", "beat::a", "path::pa")
        graph.add_edge("belongs_to", "beat::b", "path::pb")

        graph.add_edge("predecessor", "beat::a", "beat::div")
        graph.add_edge("predecessor", "beat::b", "beat::div")

        # Group all beats into passages
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::div", beat_ids=["beat::div"], summary="Div"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::a", beat_ids=["beat::a"], summary="A"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::b", beat_ids=["beat::b"], summary="B"),
        )

        # Only 1 outgoing choice from divergence passage — not enough
        _create_choice_edge(
            graph,
            ChoiceSpec(from_passage="passage::div", to_passage="passage::a", label="Go A"),
        )

        errors = validate_polish_output(graph)
        assert any(
            "beat::div" in e and "divergence point" in e and "passage::div" in e for e in errors
        )

    def test_divergence_beat_with_sufficient_choices_passes(self) -> None:
        """A divergence beat whose passage has 2+ outgoing choices should not error."""
        graph = Graph.empty()

        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})

        _make_beat(graph, "beat::div", "Divergence beat")
        _make_beat(graph, "beat::a", "Path A beat")
        _make_beat(graph, "beat::b", "Path B beat")

        graph.add_edge("belongs_to", "beat::div", "path::pa")
        graph.add_edge("belongs_to", "beat::a", "path::pa")
        graph.add_edge("belongs_to", "beat::b", "path::pb")

        graph.add_edge("predecessor", "beat::a", "beat::div")
        graph.add_edge("predecessor", "beat::b", "beat::div")

        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::div", beat_ids=["beat::div"], summary="Div"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::a", beat_ids=["beat::a"], summary="A"),
        )
        _create_passage_node(
            graph,
            PassageSpec(passage_id="passage::b", beat_ids=["beat::b"], summary="B"),
        )

        # 2 outgoing choices — sufficient
        _create_choice_edge(
            graph,
            ChoiceSpec(from_passage="passage::div", to_passage="passage::a", label="Go A"),
        )
        _create_choice_edge(
            graph,
            ChoiceSpec(from_passage="passage::div", to_passage="passage::b", label="Go B"),
        )

        errors = validate_polish_output(graph)
        assert not any("divergence point" in e for e in errors)
