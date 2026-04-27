"""Tests for POLISH Phase 5 context builders."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_context import (
    format_ambiguous_feasibility_context,
    format_choice_label_context,
    format_false_branch_context,
    format_residue_content_context,
    format_transition_guidance_context,
    format_variant_summary_context,
)
from questfoundry.models.polish import AmbiguousFeasibilityCase


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
        # IDs in the choice line are backtick-wrapped per @prompt-engineer Rule 4.
        assert "From: `p1`" in ctx["choice_details"]
        assert "To: `p2`" in ctx["choice_details"]
        assert "grants: `flag1`" in ctx["choice_details"]
        # valid_passage_ids: backtick-wrapped, sorted, deduplicated.
        assert ctx["valid_passage_ids"] == "`p1`, `p2`"

    def test_empty_choices_falls_back_to_none(self) -> None:
        """Empty choice_specs MUST render as `(none)` for both
        `choice_details` and `valid_passage_ids` per the consistent
        empty-fallback pattern across polish_context render functions."""
        graph = Graph.empty()
        ctx = format_choice_label_context(graph, [], [])
        assert ctx["choice_count"] == "0"
        assert ctx["choice_details"] == "(none)"
        assert ctx["valid_passage_ids"] == "(none)"

    def test_valid_passage_ids_dedups_and_sorts(self) -> None:
        # Same passage appearing as both `from` and `to` across multiple choices
        # should be deduplicated; output is sorted for determinism.
        graph = Graph.empty()
        choice_specs = [
            {"from_passage": "p2", "to_passage": "p3"},
            {"from_passage": "p1", "to_passage": "p2"},
            {"from_passage": "p2", "to_passage": "p3"},
        ]
        ctx = format_choice_label_context(graph, choice_specs, [])
        assert ctx["valid_passage_ids"] == "`p1`, `p2`, `p3`"

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
        # IDs are backtick-wrapped per @prompt-engineer Rule 4.
        assert "`r1`" in ctx["residue_details"]
        assert "`flag1`" in ctx["residue_details"]
        assert "`path::brave`" in ctx["residue_details"]
        assert "`p1`" in ctx["residue_details"]
        assert "Target passage" in ctx["residue_details"]

    def test_empty_residues_falls_back_to_none(self) -> None:
        """Empty residue_specs MUST render as `(none)` per the consistent
        empty-fallback pattern across polish_context render functions."""
        graph = Graph.empty()
        ctx = format_residue_content_context(graph, [], [])
        assert ctx["residue_count"] == "0"
        assert ctx["residue_details"] == "(none)"


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
        # IDs backtick-wrapped per @prompt-engineer Rule 4.
        assert "`entity::hero`" in ctx["valid_entity_ids"]
        assert "`p1`" in ctx["candidate_details"]
        assert "`p2`" in ctx["candidate_details"]
        assert "`p3`" in ctx["candidate_details"]
        assert "First" in ctx["candidate_details"]

    def test_empty_candidates_falls_back_to_none(self) -> None:
        """Empty candidates / empty entity nodes MUST render `(none)` for both
        `candidate_details` and `valid_entity_ids` per the consistent
        empty-fallback pattern."""
        graph = Graph.empty()
        ctx = format_false_branch_context(graph, [], [])
        assert ctx["candidate_count"] == "0"
        assert ctx["candidate_details"] == "(none)"
        assert ctx["valid_entity_ids"] == "(none)"


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
        # IDs backtick-wrapped per @prompt-engineer Rule 4.
        assert "`v1`" in ctx["variant_details"]
        assert "`p1`" in ctx["variant_details"]
        assert "`flag1`" in ctx["variant_details"]

    def test_empty_variants_falls_back_to_none(self) -> None:
        """Empty variant_specs MUST render `(none)` for `variant_details`."""
        graph = Graph.empty()
        ctx = format_variant_summary_context(graph, [], [])
        assert ctx["variant_count"] == "0"
        assert ctx["variant_details"] == "(none)"

    def test_variant_with_no_requires_renders_none(self) -> None:
        """A variant with no `requires` flags MUST render `requires: (none)`
        in the per-variant line (consistent with the standalone fallback)."""
        graph = Graph.empty()
        ctx = format_variant_summary_context(
            graph,
            [
                {
                    "variant_id": "v1",
                    "base_passage_id": "p1",
                    "requires": [],
                    "summary": "",
                },
            ],
            [{"passage_id": "p1", "summary": "Base", "beat_ids": ["b1"]}],
        )
        assert "requires: (none)" in ctx["variant_details"]


class TestFormatAmbiguousFeasibilityContext:
    def test_passage_id_backtick_wrapped(self) -> None:
        r"""`case_details` lines wrap `passage_id` in backticks per CLAUDE.md
        §9 rule 1, consistent with the surrounding `flag=\`...\`` /
        `dilemma=\`...\`` pattern already in this function."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {"type": "dilemma", "raw_id": "trust", "question": "Trust?", "weight": "heavy"},
        )
        case = AmbiguousFeasibilityCase(
            passage_id="passage::p1",
            passage_summary="A tense scene",
            entities=["entity::hero"],
            flags=["dilemma::trust:path::brave"],
        )
        ctx = format_ambiguous_feasibility_context(
            graph, [case], [{"passage_id": "passage::p1", "summary": "scene"}]
        )
        assert "passage_id: `passage::p1`" in ctx["case_details"]

    def test_no_cases_renders_placeholder(self) -> None:
        """Empty input renders the existing `(no cases)` placeholder rather
        than an empty string."""
        graph = Graph.empty()
        ctx = format_ambiguous_feasibility_context(graph, [], [])
        assert ctx["case_details"] == "(no cases)"
        assert ctx["case_count"] == "0"


class TestFormatTransitionGuidanceContext:
    def test_passage_and_beat_ids_backtick_wrapped(self) -> None:
        """Both `passage_id` (per-passage header) and `bid` (per-beat line)
        are backtick-wrapped per @prompt-engineer Rule 4."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "First beat")
        _make_beat(graph, "beat::b", "Second beat")
        passage_specs = [
            {
                "passage_id": "passage::collapse_0",
                "beat_ids": ["beat::a", "beat::b"],
                "grouping_type": "collapse",
                "entities": ["entity::hero"],
            }
        ]
        ctx = format_transition_guidance_context(graph, passage_specs)
        assert "passage_id: `passage::collapse_0`" in ctx["collapsed_passage_details"]
        assert "`beat::a`" in ctx["collapsed_passage_details"]
        assert "`beat::b`" in ctx["collapsed_passage_details"]
        assert ctx["collapsed_count"] == "1"

    def test_no_collapsed_passages_renders_placeholder(self) -> None:
        """Empty / non-collapse-only input renders the existing `(none)`
        placeholder."""
        graph = Graph.empty()
        # All passages are single-beat or non-collapse — none qualify.
        ctx = format_transition_guidance_context(
            graph,
            [
                {
                    "passage_id": "passage::single",
                    "beat_ids": ["beat::a"],
                    "grouping_type": "collapse",
                    "entities": [],
                },
            ],
        )
        assert ctx["collapsed_passage_details"] == "(none)"
        assert ctx["collapsed_count"] == "0"
