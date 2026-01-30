"""Tests for FILL context formatting functions."""

from __future__ import annotations

import pytest

from questfoundry.graph.fill_context import (
    format_dream_vision,
    format_entity_states,
    format_grow_summary,
    format_lookahead_context,
    format_passage_context,
    format_passages_batch,
    format_scene_types_summary,
    format_shadow_states,
    format_sliding_window,
    format_voice_context,
    get_arc_passage_order,
    get_spine_arc_id,
)
from questfoundry.graph.graph import Graph


@pytest.fixture
def fill_graph() -> Graph:
    """Create a minimal GROW-completed graph for testing FILL context."""
    g = Graph.empty()

    # Dream node
    g.create_node(
        "vision",
        {
            "type": "vision",
            "raw_id": "vision",
            "genre": "dark fantasy",
            "tone": "atmospheric and tense",
            "themes": ["trust", "power", "sacrifice"],
            "style_notes": "A crumbling tower at the edge of civilization",
        },
    )

    # Entities
    g.create_node(
        "entity::kay",
        {
            "type": "entity",
            "raw_id": "kay",
            "entity_type": "character",
            "concept": "A young wanderer seeking answers",
            "overlays": [
                {
                    "when": ["codeword::betrayal_committed"],
                    "details": {"mood": "bitter", "trust": "broken"},
                }
            ],
        },
    )
    g.create_node(
        "entity::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "entity_type": "character",
            "concept": "An enigmatic guide with hidden motives",
        },
    )

    # Beats
    g.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "Kay enters the tower and meets the mentor",
            "scene_type": "scene",
            "entities": ["entity::kay", "entity::mentor"],
        },
    )
    g.create_node(
        "beat::explanation",
        {
            "type": "beat",
            "raw_id": "explanation",
            "summary": "Mentor explains the artifact's power",
            "scene_type": "scene",
            "path_agnostic_for": ["dilemma::mentor_trust"],
        },
    )
    g.create_node(
        "beat::aftermath",
        {
            "type": "beat",
            "raw_id": "aftermath",
            "summary": "Kay reflects on choices made",
            "scene_type": "sequel",
        },
    )
    g.create_node(
        "beat::branch_reveal",
        {
            "type": "beat",
            "raw_id": "branch_reveal",
            "summary": "Mentor's true nature is exposed",
            "scene_type": "scene",
        },
    )

    # Passages
    g.create_node(
        "passage::p_opening",
        {
            "type": "passage",
            "raw_id": "p_opening",
            "from_beat": "beat::opening",
            "summary": "Kay enters the tower and meets the mentor",
            "entities": ["entity::kay", "entity::mentor"],
            "prose": "The tower stairs wound upward into darkness.",
        },
    )
    g.create_node(
        "passage::p_explanation",
        {
            "type": "passage",
            "raw_id": "p_explanation",
            "from_beat": "beat::explanation",
            "summary": "Mentor explains the artifact's power",
            "entities": ["entity::kay", "entity::mentor"],
            "prose": "The artifact lay on the table between them.",
        },
    )
    g.create_node(
        "passage::p_aftermath",
        {
            "type": "passage",
            "raw_id": "p_aftermath",
            "from_beat": "beat::aftermath",
            "summary": "Kay reflects on choices made",
            "entities": ["entity::kay"],
        },
    )
    g.create_node(
        "passage::p_branch_reveal",
        {
            "type": "passage",
            "raw_id": "p_branch_reveal",
            "from_beat": "beat::branch_reveal",
            "summary": "Mentor's true nature is exposed",
            "entities": ["entity::mentor"],
        },
    )

    # Passage-from edges
    g.add_edge("passage_from", "passage::p_opening", "beat::opening")
    g.add_edge("passage_from", "passage::p_explanation", "beat::explanation")
    g.add_edge("passage_from", "passage::p_aftermath", "beat::aftermath")
    g.add_edge("passage_from", "passage::p_branch_reveal", "beat::branch_reveal")

    # Arcs
    g.create_node(
        "arc::spine_0_0",
        {
            "type": "arc",
            "raw_id": "spine_0_0",
            "arc_type": "spine",
            "paths": ["path::mentor_trust__protector"],
            "sequence": ["beat::opening", "beat::explanation", "beat::aftermath"],
        },
    )
    g.create_node(
        "arc::branch_1_0",
        {
            "type": "arc",
            "raw_id": "branch_1_0",
            "arc_type": "branch",
            "paths": ["path::mentor_trust__manipulator"],
            "sequence": [
                "beat::opening",
                "beat::explanation",
                "beat::branch_reveal",
                "beat::aftermath",
            ],
            "diverges_from": "arc::spine_0_0",
            "diverges_at": "beat::explanation",
            "converges_at": "beat::aftermath",
        },
    )

    return g


# ---------------------------------------------------------------------------
# get_spine_arc_id
# ---------------------------------------------------------------------------


class TestGetSpineArcId:
    def test_finds_spine(self, fill_graph: Graph) -> None:
        assert get_spine_arc_id(fill_graph) == "arc::spine_0_0"

    def test_no_arcs(self) -> None:
        g = Graph.empty()
        assert get_spine_arc_id(g) is None


# ---------------------------------------------------------------------------
# get_arc_passage_order
# ---------------------------------------------------------------------------


class TestGetArcPassageOrder:
    def test_spine_order(self, fill_graph: Graph) -> None:
        order = get_arc_passage_order(fill_graph, "arc::spine_0_0")
        assert order == [
            "passage::p_opening",
            "passage::p_explanation",
            "passage::p_aftermath",
        ]

    def test_branch_order(self, fill_graph: Graph) -> None:
        order = get_arc_passage_order(fill_graph, "arc::branch_1_0")
        assert order == [
            "passage::p_opening",
            "passage::p_explanation",
            "passage::p_branch_reveal",
            "passage::p_aftermath",
        ]

    def test_nonexistent_arc(self, fill_graph: Graph) -> None:
        assert get_arc_passage_order(fill_graph, "arc::nonexistent") == []


# ---------------------------------------------------------------------------
# format_voice_context
# ---------------------------------------------------------------------------


class TestFormatVoiceContext:
    def test_no_voice_node(self, fill_graph: Graph) -> None:
        assert format_voice_context(fill_graph) == ""

    def test_with_voice_node(self, fill_graph: Graph) -> None:
        fill_graph.create_node(
            "voice::main",
            {
                "type": "voice",
                "raw_id": "main",
                "pov": "third_limited",
                "tense": "past",
                "voice_register": "literary",
            },
        )
        result = format_voice_context(fill_graph)
        assert "pov: third_limited" in result
        assert "tense: past" in result
        assert "voice_register: literary" in result
        # Should not include graph metadata
        assert "type:" not in result
        assert "raw_id:" not in result


# ---------------------------------------------------------------------------
# format_passage_context
# ---------------------------------------------------------------------------


class TestFormatPassageContext:
    def test_passage_with_entities(self, fill_graph: Graph) -> None:
        result = format_passage_context(fill_graph, "passage::p_opening")
        assert "Kay enters the tower" in result
        assert "Scene Type" in result
        assert "kay" in result

    def test_nonexistent_passage(self, fill_graph: Graph) -> None:
        assert format_passage_context(fill_graph, "passage::none") == ""


# ---------------------------------------------------------------------------
# format_sliding_window
# ---------------------------------------------------------------------------


class TestFormatSlidingWindow:
    def test_first_passage_no_window(self, fill_graph: Graph) -> None:
        result = format_sliding_window(fill_graph, "arc::spine_0_0", 0)
        assert result == "(no previous passages)"

    def test_second_passage_has_window(self, fill_graph: Graph) -> None:
        result = format_sliding_window(fill_graph, "arc::spine_0_0", 1)
        assert "p_opening" in result
        assert "tower stairs" in result

    def test_window_size_limits(self, fill_graph: Graph) -> None:
        result = format_sliding_window(fill_graph, "arc::spine_0_0", 2, window_size=1)
        # Should only include the immediately preceding passage
        assert "p_explanation" in result
        assert "p_opening" not in result

    def test_no_prose_skipped(self, fill_graph: Graph) -> None:
        # p_aftermath has no prose — window should skip it
        result = format_sliding_window(fill_graph, "arc::spine_0_0", 2, window_size=3)
        assert "p_opening" in result
        assert "p_explanation" in result


# ---------------------------------------------------------------------------
# format_lookahead_context
# ---------------------------------------------------------------------------


class TestFormatLookaheadContext:
    def test_convergence_point(self, fill_graph: Graph) -> None:
        # p_aftermath is convergence point for branch_1_0
        result = format_lookahead_context(fill_graph, "passage::p_aftermath", "arc::spine_0_0")
        assert "Convergence" in result
        assert "branch_1_0" in result

    def test_no_lookahead_needed(self, fill_graph: Graph) -> None:
        result = format_lookahead_context(fill_graph, "passage::p_opening", "arc::spine_0_0")
        assert result == ""


# ---------------------------------------------------------------------------
# format_shadow_states
# ---------------------------------------------------------------------------


class TestFormatShadowStates:
    def test_shared_beat(self, fill_graph: Graph) -> None:
        # beat::explanation is path_agnostic_for mentor_trust
        result = format_shadow_states(fill_graph, "passage::p_explanation", "arc::spine_0_0")
        assert "shared beat" in result
        assert "Active state" in result
        assert "Shadow states" in result

    def test_non_shared_beat(self, fill_graph: Graph) -> None:
        result = format_shadow_states(fill_graph, "passage::p_opening", "arc::spine_0_0")
        assert result == ""


# ---------------------------------------------------------------------------
# format_entity_states
# ---------------------------------------------------------------------------


class TestFormatEntityStates:
    def test_passage_with_entities(self, fill_graph: Graph) -> None:
        result = format_entity_states(fill_graph, "passage::p_opening")
        assert "kay" in result
        assert "mentor" in result

    def test_entity_with_overlays(self, fill_graph: Graph) -> None:
        result = format_entity_states(fill_graph, "passage::p_opening")
        assert "betrayal_committed" in result
        assert "mood" in result

    def test_no_entities(self, fill_graph: Graph) -> None:
        # Create a passage with no entities
        fill_graph.create_node(
            "passage::empty",
            {"type": "passage", "raw_id": "empty", "from_beat": "", "summary": ""},
        )
        result = format_entity_states(fill_graph, "passage::empty")
        assert result == "(no entities)"


# ---------------------------------------------------------------------------
# format_scene_types_summary
# ---------------------------------------------------------------------------


class TestFormatSceneTypesSummary:
    def test_scene_type_counts(self, fill_graph: Graph) -> None:
        result = format_scene_types_summary(fill_graph)
        assert "4 beats total" in result
        assert "3 scene" in result
        assert "1 sequel" in result

    def test_empty_graph(self) -> None:
        g = Graph.empty()
        result = format_scene_types_summary(g)
        assert "(no beats with scene types)" in result


# ---------------------------------------------------------------------------
# format_grow_summary
# ---------------------------------------------------------------------------


class TestFormatGrowSummary:
    def test_summary_counts(self, fill_graph: Graph) -> None:
        result = format_grow_summary(fill_graph)
        assert "2" in result  # 2 arcs
        assert "1 spine" in result
        assert "1 branch" in result
        assert "4" in result  # 4 passages or beats


# ---------------------------------------------------------------------------
# format_dream_vision
# ---------------------------------------------------------------------------


class TestFormatDreamVision:
    def test_extracts_vision(self, fill_graph: Graph) -> None:
        result = format_dream_vision(fill_graph)
        assert "dark fantasy" in result
        assert "atmospheric" in result
        assert "trust" in result

    def test_no_dream(self) -> None:
        g = Graph.empty()
        assert format_dream_vision(g) == ""


# ---------------------------------------------------------------------------
# format_passages_batch
# ---------------------------------------------------------------------------


class TestFormatPassagesBatch:
    def test_batch_formatting(self, fill_graph: Graph) -> None:
        result = format_passages_batch(
            fill_graph,
            ["passage::p_opening", "passage::p_explanation"],
        )
        assert "p_opening" in result
        assert "p_explanation" in result
        assert "tower stairs" in result
        assert "scene_type: scene" in result

    def test_empty_batch(self, fill_graph: Graph) -> None:
        result = format_passages_batch(fill_graph, [])
        assert result == ""

    def test_passage_without_prose(self, fill_graph: Graph) -> None:
        result = format_passages_batch(fill_graph, ["passage::p_aftermath"])
        assert "(no prose)" in result
