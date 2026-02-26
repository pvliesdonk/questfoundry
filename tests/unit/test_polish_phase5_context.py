"""Tests for POLISH Phase 5 context builders."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_context import (
    format_choice_label_context,
    format_false_branch_context,
    format_residue_content_context,
    format_variant_summary_context,
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


class TestFormatChoiceLabelContext:
    def test_basic_context(self) -> None:
        graph = Graph.empty()
        choice_specs = [
            {"from_passage": "p1", "to_passage": "p2", "grants": ["flag1"]},
        ]
        passage_specs = [
            {"passage_id": "p1", "summary": "Start", "beat_ids": ["b1"]},
            {"passage_id": "p2", "summary": "End", "beat_ids": ["b2"]},
        ]

        ctx = format_choice_label_context(graph, choice_specs, passage_specs)
        assert "choice_count" in ctx
        assert ctx["choice_count"] == "1"
        assert "choice_details" in ctx
        assert "Start" in ctx["choice_details"]
        assert "End" in ctx["choice_details"]

    def test_empty_choices(self) -> None:
        graph = Graph.empty()
        ctx = format_choice_label_context(graph, [], [])
        assert ctx["choice_count"] == "0"
        assert ctx["choice_details"] == ""

    def test_story_context_from_dream(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dream_artifact::d1",
            {"type": "dream_artifact", "raw_id": "d1", "genre": "mystery", "tone": "dark"},
        )
        ctx = format_choice_label_context(graph, [], [])
        assert "mystery" in ctx["story_context"]
        assert "dark" in ctx["story_context"]


class TestFormatResidueContentContext:
    def test_basic_context(self) -> None:
        graph = Graph.empty()
        residue_specs = [
            {
                "target_passage_id": "p1",
                "residue_id": "r1",
                "flag": "flag1",
                "path_id": "path::brave",
            },
        ]
        passage_specs = [
            {"passage_id": "p1", "summary": "Target passage", "beat_ids": ["b1"]},
        ]

        ctx = format_residue_content_context(graph, residue_specs, passage_specs)
        assert ctx["residue_count"] == "1"
        assert "r1" in ctx["residue_details"]
        assert "Target passage" in ctx["residue_details"]

    def test_empty_residues(self) -> None:
        graph = Graph.empty()
        ctx = format_residue_content_context(graph, [], [])
        assert ctx["residue_count"] == "0"


class TestFormatFalseBranchContext:
    def test_basic_context(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "name": "Hero"},
        )

        candidates = [
            {
                "passage_ids": ["p1", "p2", "p3"],
                "context_summary": "Linear stretch",
            },
        ]
        passage_specs = [
            {"passage_id": "p1", "summary": "First", "beat_ids": ["b1"]},
            {"passage_id": "p2", "summary": "Second", "beat_ids": ["b2"]},
            {"passage_id": "p3", "summary": "Third", "beat_ids": ["b3"]},
        ]

        ctx = format_false_branch_context(graph, candidates, passage_specs)
        assert ctx["candidate_count"] == "1"
        assert "entity::hero" in ctx["valid_entity_ids"]
        assert "First" in ctx["candidate_details"]

    def test_empty_candidates(self) -> None:
        graph = Graph.empty()
        ctx = format_false_branch_context(graph, [], [])
        assert ctx["candidate_count"] == "0"


class TestFormatVariantSummaryContext:
    def test_basic_context(self) -> None:
        graph = Graph.empty()
        variant_specs = [
            {
                "variant_id": "v1",
                "base_passage_id": "p1",
                "requires": ["flag1"],
                "summary": "",
            },
        ]
        passage_specs = [
            {"passage_id": "p1", "summary": "Base passage", "beat_ids": ["b1"]},
        ]

        ctx = format_variant_summary_context(graph, variant_specs, passage_specs)
        assert ctx["variant_count"] == "1"
        assert "Base passage" in ctx["variant_details"]
        assert "flag1" in ctx["variant_details"]

    def test_empty_variants(self) -> None:
        graph = Graph.empty()
        ctx = format_variant_summary_context(graph, [], [])
        assert ctx["variant_count"] == "0"
