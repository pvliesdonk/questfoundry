"""Tests for FILL context formatting functions."""

from __future__ import annotations

import pytest

from questfoundry.graph.fill_context import (
    _extract_top_bigrams,
    compute_arc_hints,
    compute_first_appearances,
    compute_is_ending,
    compute_lexical_diversity,
    compute_open_questions,
    derive_pacing,
    extract_used_imagery,
    format_atmospheric_detail,
    format_blueprint_context,
    format_dramatic_questions,
    format_dream_vision,
    format_ending_guidance,
    format_entity_arc_context,
    format_entity_states,
    format_entry_states,
    format_grow_summary,
    format_introduction_guidance,
    format_lookahead_context,
    format_merged_passage_context,
    format_narrative_context,
    format_passage_context,
    format_passages_batch,
    format_path_arc_context,
    format_scene_types_summary,
    format_shadow_states,
    format_sliding_window,
    format_used_imagery_blocklist,
    format_valid_characters,
    format_vocabulary_note,
    format_voice_context,
    get_arc_passage_order,
    get_spine_arc_id,
    is_merged_passage,
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

    def test_shadow_arcs_filtered_by_dilemma(self) -> None:
        """Beat agnostic for dilemma_A should not see arcs differing on dilemma_B."""
        g = Graph.empty()

        # Two dilemmas with two paths each
        g.create_node("dilemma::d_a", {"type": "dilemma", "raw_id": "d_a"})
        g.create_node("dilemma::d_b", {"type": "dilemma", "raw_id": "d_b"})
        g.create_node(
            "path::a1",
            {"type": "path", "raw_id": "a1", "dilemma_id": "d_a", "is_canonical": True},
        )
        g.create_node(
            "path::a2",
            {"type": "path", "raw_id": "a2", "dilemma_id": "d_a", "is_canonical": False},
        )
        g.create_node(
            "path::b1",
            {"type": "path", "raw_id": "b1", "dilemma_id": "d_b", "is_canonical": True},
        )
        g.create_node(
            "path::b2",
            {"type": "path", "raw_id": "b2", "dilemma_id": "d_b", "is_canonical": False},
        )

        # Shared beat: agnostic for d_a only
        g.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "summary": "A shared moment",
                "path_agnostic_for": ["dilemma::d_a"],
            },
        )
        g.create_node(
            "passage::p_shared",
            {"type": "passage", "raw_id": "p_shared", "from_beat": "beat::shared"},
        )
        g.add_edge("passage_from", "passage::p_shared", "beat::shared")

        # Arcs: spine (a1+b1), branch_a (a2+b1), branch_b (a1+b2), branch_ab (a2+b2)
        # All contain beat::shared
        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["a1", "b1"], "sequence": ["beat::shared"]},
        )
        g.create_node(
            "arc::branch_a",
            {
                "type": "arc",
                "raw_id": "branch_a",
                "paths": ["a2", "b1"],
                "sequence": ["beat::shared"],
            },
        )
        g.create_node(
            "arc::branch_b",
            {
                "type": "arc",
                "raw_id": "branch_b",
                "paths": ["a1", "b2"],
                "sequence": ["beat::shared"],
            },
        )
        g.create_node(
            "arc::branch_ab",
            {
                "type": "arc",
                "raw_id": "branch_ab",
                "paths": ["a2", "b2"],
                "sequence": ["beat::shared"],
            },
        )

        # From spine (a1+b1), agnostic for d_a:
        # branch_a (a2+b1) differs only on d_a → INCLUDE
        # branch_b (a1+b2) differs only on d_b → EXCLUDE (not agnostic for d_b)
        # branch_ab (a2+b2) differs on both → EXCLUDE (differs on d_b)
        result = format_shadow_states(g, "passage::p_shared", "arc::spine")
        assert "shared beat" in result
        assert "branch_a" in result
        assert "branch_b" not in result
        assert "branch_ab" not in result

    def test_empty_agnostic_for_returns_empty(self) -> None:
        """Beat with path_agnostic_for: [] should not show shadow arcs."""
        g = Graph.empty()
        g.create_node("dilemma::d_a", {"type": "dilemma", "raw_id": "d_a"})
        g.create_node(
            "path::a1",
            {"type": "path", "raw_id": "a1", "dilemma_id": "d_a", "is_canonical": True},
        )
        g.create_node(
            "path::a2",
            {"type": "path", "raw_id": "a2", "dilemma_id": "d_a", "is_canonical": False},
        )
        g.create_node(
            "beat::not_shared",
            {
                "type": "beat",
                "raw_id": "not_shared",
                "summary": "Not shared",
                "path_agnostic_for": [],
            },
        )
        g.create_node(
            "passage::p_not_shared",
            {"type": "passage", "raw_id": "p_not_shared", "from_beat": "beat::not_shared"},
        )
        g.add_edge("passage_from", "passage::p_not_shared", "beat::not_shared")
        g.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "paths": ["a1"],
                "sequence": ["beat::not_shared"],
            },
        )
        g.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "paths": ["a2"],
                "sequence": ["beat::not_shared"],
            },
        )
        result = format_shadow_states(g, "passage::p_not_shared", "arc::spine")
        assert result == ""

    def test_multi_dilemma_agnostic(self) -> None:
        """Beat agnostic for both dilemmas should see all shadow arcs."""
        g = Graph.empty()

        g.create_node("dilemma::d_a", {"type": "dilemma", "raw_id": "d_a"})
        g.create_node("dilemma::d_b", {"type": "dilemma", "raw_id": "d_b"})
        g.create_node(
            "path::a1",
            {"type": "path", "raw_id": "a1", "dilemma_id": "d_a", "is_canonical": True},
        )
        g.create_node(
            "path::a2",
            {"type": "path", "raw_id": "a2", "dilemma_id": "d_a", "is_canonical": False},
        )
        g.create_node(
            "path::b1",
            {"type": "path", "raw_id": "b1", "dilemma_id": "d_b", "is_canonical": True},
        )
        g.create_node(
            "path::b2",
            {"type": "path", "raw_id": "b2", "dilemma_id": "d_b", "is_canonical": False},
        )

        g.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "summary": "Shared everywhere",
                "path_agnostic_for": ["dilemma::d_a", "dilemma::d_b"],
            },
        )
        g.create_node(
            "passage::p_shared",
            {"type": "passage", "raw_id": "p_shared", "from_beat": "beat::shared"},
        )
        g.add_edge("passage_from", "passage::p_shared", "beat::shared")

        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["a1", "b1"], "sequence": ["beat::shared"]},
        )
        g.create_node(
            "arc::branch_a",
            {
                "type": "arc",
                "raw_id": "branch_a",
                "paths": ["a2", "b1"],
                "sequence": ["beat::shared"],
            },
        )
        g.create_node(
            "arc::branch_b",
            {
                "type": "arc",
                "raw_id": "branch_b",
                "paths": ["a1", "b2"],
                "sequence": ["beat::shared"],
            },
        )

        # Agnostic for both → all shadow arcs included
        result = format_shadow_states(g, "passage::p_shared", "arc::spine")
        assert "branch_a" in result
        assert "branch_b" in result


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
# format_pov_context
# ---------------------------------------------------------------------------


class TestFormatPovContext:
    def test_with_protagonist_and_pov_style(self) -> None:
        """Returns formatted context with protagonist and POV style."""
        from questfoundry.graph.fill_context import format_pov_context

        g = Graph.empty()
        g.create_node(
            "vision",
            {
                "type": "vision",
                "genre": "mystery",
                "pov_style": "first",
                "protagonist_defined": True,
            },
        )
        g.create_node(
            "character::detective",
            {
                "type": "entity",
                "raw_id": "detective",
                "category": "character",
                "concept": "A seasoned private eye",
                "is_protagonist": True,
            },
        )

        result = format_pov_context(g)

        assert "**Suggested POV:** first" in result
        assert "**Protagonist Defined:** Yes" in result
        assert "**Protagonist:** detective" in result
        assert "A seasoned private eye" in result

    def test_without_protagonist(self) -> None:
        """Returns context noting protagonist not marked when protagonist_defined but none found."""
        from questfoundry.graph.fill_context import format_pov_context

        g = Graph.empty()
        g.create_node(
            "vision",
            {
                "type": "vision",
                "genre": "fantasy",
                "protagonist_defined": True,  # Says there should be one
            },
        )
        # No entity with is_protagonist=True

        result = format_pov_context(g)

        assert "**Protagonist Defined:** Yes" in result
        assert "Not explicitly marked" in result

    def test_no_vision(self) -> None:
        """Returns empty string when no vision node."""
        from questfoundry.graph.fill_context import format_pov_context

        g = Graph.empty()

        result = format_pov_context(g)

        assert result == ""

    def test_minimal_vision(self) -> None:
        """Vision without POV fields returns empty string."""
        from questfoundry.graph.fill_context import format_pov_context

        g = Graph.empty()
        g.create_node(
            "vision",
            {"type": "vision", "genre": "fantasy"},  # No pov_style or protagonist_defined
        )

        result = format_pov_context(g)

        assert result == ""


# ---------------------------------------------------------------------------
# get_path_pov_character
# ---------------------------------------------------------------------------


class TestGetPathPovCharacter:
    def test_path_specific_override(self) -> None:
        """Returns path-specific pov_character when set."""
        from questfoundry.graph.fill_context import get_path_pov_character

        g = Graph.empty()
        g.create_node(
            "arc::spine",
            {"type": "arc", "arc_type": "spine", "paths": ["path::trust"]},
        )
        g.create_node(
            "path::trust",
            {"type": "path", "raw_id": "trust", "pov_character": "character::sidekick"},
        )

        result = get_path_pov_character(g, "arc::spine")

        assert result == "character::sidekick"

    def test_fallback_to_protagonist(self) -> None:
        """Falls back to global protagonist when no path override."""
        from questfoundry.graph.fill_context import get_path_pov_character

        g = Graph.empty()
        g.create_node(
            "arc::spine",
            {"type": "arc", "arc_type": "spine", "paths": ["path::trust"]},
        )
        g.create_node(
            "path::trust",
            {"type": "path", "raw_id": "trust"},  # No pov_character
        )
        g.create_node(
            "character::protagonist",
            {
                "type": "entity",
                "raw_id": "protagonist",
                "category": "character",
                "is_protagonist": True,
            },
        )

        result = get_path_pov_character(g, "arc::spine")

        assert result == "character::protagonist"

    def test_no_pov_character(self) -> None:
        """Returns None when no path override and no protagonist."""
        from questfoundry.graph.fill_context import get_path_pov_character

        g = Graph.empty()
        g.create_node(
            "arc::spine",
            {"type": "arc", "arc_type": "spine", "paths": ["path::trust"]},
        )
        g.create_node(
            "path::trust",
            {"type": "path", "raw_id": "trust"},
        )

        result = get_path_pov_character(g, "arc::spine")

        assert result is None

    def test_nonexistent_arc(self) -> None:
        """Returns None for nonexistent arc."""
        from questfoundry.graph.fill_context import get_path_pov_character

        g = Graph.empty()

        result = get_path_pov_character(g, "arc::nonexistent")

        assert result is None


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


# ---------------------------------------------------------------------------
# Dramatic Question Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dq_graph() -> Graph:
    """Graph with dilemmas and dilemma_impacts for dramatic question tests."""
    g = Graph.empty()

    # Dilemmas
    g.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Can the mentor be trusted?",
        },
    )
    g.create_node(
        "dilemma::artifact_cost",
        {
            "type": "dilemma",
            "raw_id": "artifact_cost",
            "question": "Is the artifact worth the cost?",
        },
    )

    # Beats with dilemma_impacts
    g.create_node(
        "beat::b1",
        {
            "type": "beat",
            "raw_id": "b1",
            "summary": "Kay meets the mentor",
            "scene_type": "scene",
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "advances",
                    "note": "First meeting establishes the question",
                },
            ],
        },
    )
    g.create_node(
        "beat::b2",
        {
            "type": "beat",
            "raw_id": "b2",
            "summary": "Mentor reveals the artifact",
            "scene_type": "scene",
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "reveals",
                    "note": "More about the mentor's past",
                },
                {
                    "dilemma_id": "dilemma::artifact_cost",
                    "effect": "advances",
                    "note": "Artifact's power is shown",
                },
            ],
        },
    )
    g.create_node(
        "beat::b3",
        {
            "type": "beat",
            "raw_id": "b3",
            "summary": "Kay discovers the mentor's secret",
            "scene_type": "scene",
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "complicates",
                    "note": "Secret changes everything",
                },
            ],
        },
    )
    g.create_node(
        "beat::b4",
        {
            "type": "beat",
            "raw_id": "b4",
            "summary": "Kay commits to trusting the mentor",
            "scene_type": "scene",
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "commits",
                    "note": "Trust is locked in",
                },
            ],
        },
    )
    g.create_node(
        "beat::b5",
        {
            "type": "beat",
            "raw_id": "b5",
            "summary": "Aftermath",
            "scene_type": "sequel",
            "dilemma_impacts": [],
        },
    )

    # Arc with beat sequence
    g.create_node(
        "arc::spine",
        {
            "type": "arc",
            "raw_id": "spine",
            "arc_type": "spine",
            "paths": ["path::trust"],
            "sequence": ["beat::b1", "beat::b2", "beat::b3", "beat::b4", "beat::b5"],
        },
    )

    return g


class TestComputeOpenQuestions:
    """Tests for compute_open_questions."""

    def test_empty_arc(self) -> None:
        """No questions when arc doesn't exist."""
        g = Graph.empty()
        result = compute_open_questions(g, "arc::nonexistent", "beat::b1")
        assert result == []

    def test_first_beat_no_prior(self, dq_graph: Graph) -> None:
        """First beat has no open questions (no prior beats)."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b1")
        assert result == []

    def test_single_advance(self, dq_graph: Graph) -> None:
        """After b1, mentor_trust should be open with 1 escalation."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b2")
        assert len(result) == 1
        assert result[0]["dilemma_id"] == "dilemma::mentor_trust"
        assert result[0]["escalations"] == 1
        assert result[0]["question"] == "Can the mentor be trusted?"

    def test_multiple_dilemmas(self, dq_graph: Graph) -> None:
        """After b2, both dilemmas should be open."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b3")
        assert len(result) == 2
        # mentor_trust has 2 escalations (advances + reveals), artifact_cost has 1
        mentor = next(q for q in result if q["dilemma_id"] == "dilemma::mentor_trust")
        artifact = next(q for q in result if q["dilemma_id"] == "dilemma::artifact_cost")
        assert mentor["escalations"] == 2
        assert artifact["escalations"] == 1

    def test_escalation_count(self, dq_graph: Graph) -> None:
        """After b3, mentor_trust should have 3 escalations."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b4")
        mentor = next(q for q in result if q["dilemma_id"] == "dilemma::mentor_trust")
        assert mentor["escalations"] == 3

    def test_commits_closes_question(self, dq_graph: Graph) -> None:
        """After b4 commits, mentor_trust should be closed."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b5")
        dilemma_ids = [q["dilemma_id"] for q in result]
        assert "dilemma::mentor_trust" not in dilemma_ids
        # artifact_cost should still be open
        assert "dilemma::artifact_cost" in dilemma_ids

    def test_action_here_advances(self, dq_graph: Graph) -> None:
        """Current beat's action is tracked."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b2")
        # b2 has reveals on mentor_trust
        assert result[0]["action_here"] == "reveals"

    def test_action_here_commits(self, dq_graph: Graph) -> None:
        """Commits action is tracked at current beat."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b4")
        mentor = next(q for q in result if q["dilemma_id"] == "dilemma::mentor_trust")
        assert mentor["action_here"] == "commits"

    def test_sorted_by_escalation(self, dq_graph: Graph) -> None:
        """Results are sorted by escalation count descending."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b3")
        assert result[0]["escalations"] > result[1]["escalations"]

    def test_complicates_counts_as_escalation(self, dq_graph: Graph) -> None:
        """Complicates effect counts as an escalation."""
        result = compute_open_questions(dq_graph, "arc::spine", "beat::b4")
        mentor = next(q for q in result if q["dilemma_id"] == "dilemma::mentor_trust")
        # advances(1) + reveals(1) + complicates(1) = 3
        assert mentor["escalations"] == 3


class TestFormatDramaticQuestions:
    """Tests for format_dramatic_questions."""

    def test_empty_when_no_questions(self, dq_graph: Graph) -> None:
        """Returns empty string when no open questions."""
        result = format_dramatic_questions(dq_graph, "arc::spine", "beat::b1")
        assert result == ""

    def test_formats_open_questions(self, dq_graph: Graph) -> None:
        """Formats open questions with escalation notes."""
        result = format_dramatic_questions(dq_graph, "arc::spine", "beat::b3")
        assert "Can the mentor be trusted?" in result
        assert "Is the artifact worth the cost?" in result
        assert "UNRESOLVED" in result

    def test_commits_note(self, dq_graph: Graph) -> None:
        """Shows resolving note when current beat commits."""
        result = format_dramatic_questions(dq_graph, "arc::spine", "beat::b4")
        assert "RESOLVING" in result

    def test_complicates_note(self, dq_graph: Graph) -> None:
        """Shows complicating note for complicates action."""
        result = format_dramatic_questions(dq_graph, "arc::spine", "beat::b3")
        # b3 complicates mentor_trust
        assert "complicating" in result

    def test_nonexistent_arc(self) -> None:
        """Returns empty for nonexistent arc."""
        g = Graph.empty()
        result = format_dramatic_questions(g, "arc::nope", "beat::b1")
        assert result == ""


# ---------------------------------------------------------------------------
# Pacing Derivation and Narrative Context Tests
# ---------------------------------------------------------------------------


class TestDerivePacing:
    """Tests for derive_pacing deterministic lookup."""

    def test_confront_scene_is_high_long(self) -> None:
        intensity, length = derive_pacing("confront", "scene")
        assert intensity == "high"
        assert length == "long"

    def test_introduce_micro_beat_is_low_short(self) -> None:
        intensity, length = derive_pacing("introduce", "micro_beat")
        assert intensity == "low"
        assert length == "short"

    def test_develop_sequel_is_low_medium(self) -> None:
        intensity, length = derive_pacing("develop", "sequel")
        assert intensity == "low"
        assert length == "medium"

    def test_resolve_scene_is_high_long(self) -> None:
        intensity, length = derive_pacing("resolve", "scene")
        assert intensity == "high"
        assert length == "long"

    def test_unknown_falls_back_to_medium(self) -> None:
        intensity, length = derive_pacing("unknown", "unknown")
        assert intensity == "medium"
        assert length == "medium"

    def test_all_15_combinations_defined(self) -> None:
        functions = ["introduce", "develop", "complicate", "confront", "resolve"]
        types = ["scene", "sequel", "micro_beat"]
        for func in functions:
            for stype in types:
                intensity, length = derive_pacing(func, stype)
                assert intensity in ("low", "medium", "high")
                assert length in ("short", "medium", "long")


class TestFormatNarrativeContext:
    """Tests for format_narrative_context."""

    def test_returns_empty_for_missing_passage(self) -> None:
        g = Graph.empty()
        assert format_narrative_context(g, "passage::nope") == ""

    def test_returns_empty_when_no_narrative_function(self, fill_graph: Graph) -> None:
        """Beats without narrative_function get empty context (graceful degradation)."""
        result = format_narrative_context(fill_graph, "passage::p_opening")
        assert result == ""

    def test_formats_with_narrative_function(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "scene_type": "scene",
                "narrative_function": "confront",
                "exit_mood": "bitter resolve",
            },
        )
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1"},
        )
        result = format_narrative_context(g, "passage::p1")
        assert "confront" in result
        assert "high" in result.lower()
        assert "bitter resolve" in result
        assert "Long" in result

    def test_formats_without_exit_mood(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "scene_type": "sequel",
                "narrative_function": "develop",
            },
        )
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1"},
        )
        result = format_narrative_context(g, "passage::p1")
        assert "develop" in result
        assert "Exit Mood" not in result


# ---------------------------------------------------------------------------
# Atmospheric Detail and Entry States Tests
# ---------------------------------------------------------------------------


class TestFormatAtmosphericDetail:
    """Tests for format_atmospheric_detail."""

    def test_returns_empty_for_missing_passage(self) -> None:
        g = Graph.empty()
        assert format_atmospheric_detail(g, "passage::nope") == ""

    def test_returns_empty_when_no_detail(self, fill_graph: Graph) -> None:
        """Beats without atmospheric_detail get empty context."""
        result = format_atmospheric_detail(fill_graph, "passage::p_opening")
        assert result == ""

    def test_formats_atmospheric_detail(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "atmospheric_detail": "Cold stone walls slick with condensation",
            },
        )
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1"},
        )
        result = format_atmospheric_detail(g, "passage::p1")
        assert "Cold stone walls" in result
        assert "sensory" in result.lower()

    def test_returns_empty_when_beat_missing(self) -> None:
        g = Graph.empty()
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::missing"},
        )
        assert format_atmospheric_detail(g, "passage::p1") == ""


class TestFormatEntryStates:
    """Tests for format_entry_states."""

    def test_returns_empty_for_missing_passage(self) -> None:
        g = Graph.empty()
        assert format_entry_states(g, "passage::nope", "arc::a1") == ""

    def test_returns_empty_when_no_entry_states(self) -> None:
        g = Graph.empty()
        g.create_node("beat::b1", {"type": "beat", "raw_id": "b1"})
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1"},
        )
        assert format_entry_states(g, "passage::p1", "arc::a1") == ""

    def test_formats_entry_states(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "entry_states": [
                    {"path_id": "path::trust", "mood": "cautious warmth"},
                    {"path_id": "path::betray", "mood": "defensive guilt"},
                ],
            },
        )
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::shared"},
        )
        g.create_node(
            "arc::a1",
            {"type": "arc", "paths": ["path::trust", "path::betray"]},
        )
        result = format_entry_states(g, "passage::p1", "arc::a1")
        assert "path::trust: cautious warmth <- ACTIVE" in result
        assert "path::betray: defensive guilt <- ACTIVE" in result

    def test_marks_active_paths(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::shared",
            {
                "type": "beat",
                "raw_id": "shared",
                "entry_states": [
                    {"path_id": "path::trust", "mood": "cautious warmth"},
                    {"path_id": "path::betray", "mood": "defensive guilt"},
                ],
            },
        )
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::shared"},
        )
        # Arc only includes trust path
        g.create_node(
            "arc::a1",
            {"type": "arc", "paths": ["path::trust"]},
        )
        result = format_entry_states(g, "passage::p1", "arc::a1")
        assert "path::trust: cautious warmth <- ACTIVE" in result
        # betray is NOT active in this arc
        assert "path::betray: defensive guilt" in result
        assert "path::betray: defensive guilt <- ACTIVE" not in result


class TestFormatPathArcContext:
    """Tests for format_path_arc_context."""

    def test_missing_passage(self) -> None:
        g = Graph.empty()
        result = format_path_arc_context(g, "passage::nonexistent", "arc::a1")
        assert result == ""

    def test_missing_arc(self) -> None:
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        result = format_path_arc_context(g, "passage::p1", "arc::nonexistent")
        assert result == ""

    def test_no_path_arcs_set(self) -> None:
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        g.create_node("arc::a1", {"type": "arc", "paths": ["path::trust"]})
        g.create_node("path::trust", {"type": "path", "raw_id": "trust"})
        result = format_path_arc_context(g, "passage::p1", "arc::a1")
        assert result == ""

    def test_formats_path_arcs(self) -> None:
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        g.create_node("arc::a1", {"type": "arc", "paths": ["path::trust", "path::betray"]})
        g.create_node(
            "path::trust",
            {
                "type": "path",
                "raw_id": "trust",
                "path_theme": "A slow surrender to vulnerability",
                "path_mood": "fragile hope",
            },
        )
        g.create_node(
            "path::betray",
            {
                "type": "path",
                "raw_id": "betray",
                "path_theme": "The cost of self-preservation",
                "path_mood": "bitter resolve",
            },
        )
        result = format_path_arc_context(g, "passage::p1", "arc::a1")
        assert "Path Arcs" in result
        assert "fragile hope" in result
        assert "A slow surrender to vulnerability" in result
        assert "bitter resolve" in result
        assert "The cost of self-preservation" in result

    def test_skips_paths_without_arcs(self) -> None:
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        g.create_node("arc::a1", {"type": "arc", "paths": ["path::trust", "path::betray"]})
        g.create_node(
            "path::trust",
            {
                "type": "path",
                "raw_id": "trust",
                "path_theme": "A slow surrender to vulnerability",
                "path_mood": "fragile hope",
            },
        )
        # betray has no path_theme or path_mood
        g.create_node("path::betray", {"type": "path", "raw_id": "betray"})
        result = format_path_arc_context(g, "passage::p1", "arc::a1")
        assert "fragile hope" in result
        assert "betray" not in result


# ---------------------------------------------------------------------------
# Lexical diversity helpers
# ---------------------------------------------------------------------------


class TestExtractTopBigrams:
    """Tests for _extract_top_bigrams helper."""

    def test_empty_input(self) -> None:
        assert _extract_top_bigrams([]) == []
        assert _extract_top_bigrams([""]) == []

    def test_no_repeated_bigrams(self) -> None:
        texts = ["The quick brown fox jumps over a lazy dog"]
        assert _extract_top_bigrams(texts) == []

    def test_extracts_repeated_bigrams(self) -> None:
        texts = [
            "stale air filled the room with stale air",
            "the stale air was thick",
        ]
        result = _extract_top_bigrams(texts)
        assert "stale air" in result

    def test_respects_min_count(self) -> None:
        texts = ["one two one two one two"]
        assert _extract_top_bigrams(texts, min_count=3) == ["one two"]
        assert _extract_top_bigrams(texts, min_count=4) == []

    def test_limits_to_n(self) -> None:
        # Repeat many different bigrams
        texts = ["a b a b c d c d e f e f g h g h"]
        result = _extract_top_bigrams(texts, n=2)
        assert len(result) <= 2

    def test_ordered_by_frequency(self) -> None:
        texts = ["x y x y x y a b a b"]
        result = _extract_top_bigrams(texts, n=2)
        assert result[0] == "x y"


class TestLexicalDiversity:
    """Tests for compute_lexical_diversity and format_vocabulary_note."""

    def test_empty_input(self) -> None:
        assert compute_lexical_diversity([]) == 1.0
        assert compute_lexical_diversity([""]) == 1.0

    def test_high_diversity(self) -> None:
        texts = ["The quick brown fox jumps", "over a lazy dog sleeping"]
        ratio = compute_lexical_diversity(texts)
        assert ratio > 0.8  # all unique words

    def test_low_diversity(self) -> None:
        texts = ["the the the the the", "the the the the the"]
        ratio = compute_lexical_diversity(texts)
        assert ratio < 0.2  # only one unique word

    def test_vocabulary_note_below_threshold(self) -> None:
        note = format_vocabulary_note(0.3)
        assert "VOCABULARY ALERT" in note
        assert "0.30" in note

    def test_vocabulary_note_above_threshold(self) -> None:
        assert format_vocabulary_note(0.5) == ""
        assert format_vocabulary_note(0.4) == ""

    def test_vocabulary_note_custom_threshold(self) -> None:
        assert format_vocabulary_note(0.5, threshold=0.6) != ""
        assert format_vocabulary_note(0.5, threshold=0.4) == ""

    def test_vocabulary_note_with_prose_shows_specific_phrases(self) -> None:
        prose = [
            "stale air filled the room with stale air",
            "the stale air was thick and stale air hung low",
        ]
        note = format_vocabulary_note(0.3, recent_prose=prose)
        assert "VOCABULARY ALERT" in note
        assert "stale air" in note
        assert "MUST NOT" in note

    def test_vocabulary_note_without_prose_shows_generic(self) -> None:
        note = format_vocabulary_note(0.3)
        assert "VOCABULARY ALERT" in note
        assert "MUST NOT" not in note
        assert "seek fresh" in note

    def test_vocabulary_note_with_prose_no_repeats_shows_generic(self) -> None:
        prose = ["every word here is unique and different"]
        note = format_vocabulary_note(0.3, recent_prose=prose)
        assert "VOCABULARY ALERT" in note
        assert "seek fresh" in note


# ---------------------------------------------------------------------------
# Ending detection and guidance
# ---------------------------------------------------------------------------


class TestIsEnding:
    """Tests for compute_is_ending."""

    def test_passage_with_outgoing_choice_is_not_ending(self, fill_graph: Graph) -> None:
        # Add a choice_from edge: passage::p_opening has an outgoing choice
        fill_graph.create_node(
            "choice::opening_to_explanation",
            {"type": "choice", "raw_id": "opening_to_explanation"},
        )
        fill_graph.add_edge("choice_from", "choice::opening_to_explanation", "passage::p_opening")
        assert compute_is_ending(fill_graph, "passage::p_opening") is False

    def test_passage_without_outgoing_choice_is_ending(self, fill_graph: Graph) -> None:
        # p_aftermath has no choice_from edges pointing to it
        assert compute_is_ending(fill_graph, "passage::p_aftermath") is True


class TestEndingGuidance:
    """Tests for format_ending_guidance."""

    def test_ending_returns_guidance(self) -> None:
        guidance = format_ending_guidance(True)
        assert "FINAL PASSAGE" in guidance
        assert "Close the emotional arc" in guidance
        assert "No new threads" in guidance
        assert "Do NOT write 'The End'" in guidance

    def test_non_ending_returns_empty(self) -> None:
        assert format_ending_guidance(False) == ""


# ---------------------------------------------------------------------------
# Echo prompt at convergence
# ---------------------------------------------------------------------------


class TestEchoPrompt:
    """Tests for thematic echo at convergence points."""

    def test_convergence_includes_echo(self) -> None:
        """At convergence, lookahead should include opening passage echo."""
        g = Graph()
        # Spine arc with two beats
        g.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": ["beat::b1", "beat::conv"],
            },
        )
        # Branch arc converging at beat::conv
        g.create_node(
            "arc::branch",
            {
                "type": "arc",
                "arc_type": "branch",
                "converges_at": "beat::conv",
                "sequence": ["beat::br1"],
            },
        )
        g.create_node("beat::b1", {"type": "beat", "summary": "opening"})
        g.create_node("beat::conv", {"type": "beat", "summary": "convergence"})
        g.create_node("beat::br1", {"type": "beat", "summary": "branch beat"})
        g.create_node(
            "passage::p1",
            {"type": "passage", "from_beat": "beat::b1", "prose": "The rain began to fall."},
        )
        g.create_node(
            "passage::conv",
            {"type": "passage", "from_beat": "beat::conv"},
        )
        g.add_edge("passage_from", "passage::p1", "beat::b1")
        g.add_edge("passage_from", "passage::conv", "beat::conv")

        result = format_lookahead_context(g, "passage::conv", "arc::spine")
        assert "Thematic Echo" in result
        assert "The rain began to fall." in result

    def test_no_echo_at_normal_passage(self) -> None:
        """Non-juncture passages should not have echo prompts."""
        g = Graph()
        g.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": ["beat::b1", "beat::b2"],
            },
        )
        g.create_node("beat::b1", {"type": "beat"})
        g.create_node("beat::b2", {"type": "beat"})
        g.create_node(
            "passage::p1",
            {"type": "passage", "from_beat": "beat::b1", "prose": "Opening."},
        )
        g.create_node("passage::p2", {"type": "passage", "from_beat": "beat::b2"})
        g.add_edge("passage_from", "passage::p1", "beat::b1")
        g.add_edge("passage_from", "passage::p2", "beat::b2")

        result = format_lookahead_context(g, "passage::p2", "arc::spine")
        assert "Thematic Echo" not in result


class TestFirstAppearances:
    """Tests for compute_first_appearances."""

    def test_first_passage_all_new(self) -> None:
        """All entities in the first passage are first appearances."""
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "entities": ["entity::a", "entity::b"]})
        g.create_node("passage::p2", {"type": "passage", "entities": ["entity::c"]})
        result = compute_first_appearances(g, "passage::p1", ["passage::p1", "passage::p2"])
        assert result == ["entity::a", "entity::b"]

    def test_already_seen_excluded(self) -> None:
        """Entities seen in earlier passages are not first appearances."""
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "entities": ["entity::a", "entity::b"]})
        g.create_node("passage::p2", {"type": "passage", "entities": ["entity::a", "entity::c"]})
        result = compute_first_appearances(g, "passage::p2", ["passage::p1", "passage::p2"])
        assert result == ["entity::c"]

    def test_no_entities_returns_empty(self) -> None:
        """Passage with no entities returns empty list."""
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "entities": []})
        result = compute_first_appearances(g, "passage::p1", ["passage::p1"])
        assert result == []

    def test_passage_not_in_arc_returns_empty(self) -> None:
        """Passage not in the arc list returns empty."""
        g = Graph.empty()
        g.create_node("passage::p1", {"type": "passage", "entities": ["entity::a"]})
        result = compute_first_appearances(g, "passage::p1", ["passage::other"])
        assert result == []

    def test_missing_passage_node_returns_empty(self) -> None:
        """Non-existent passage returns empty."""
        g = Graph.empty()
        result = compute_first_appearances(g, "passage::missing", ["passage::missing"])
        assert result == []


class TestIntroductionGuidance:
    """Tests for format_introduction_guidance."""

    def test_empty_names_returns_empty(self) -> None:
        result = format_introduction_guidance([])
        assert result == ""

    def test_single_name_returns_guidance(self) -> None:
        result = format_introduction_guidance(["butler"])
        assert "Character Introduction" in result
        assert "**butler**" in result
        assert "sensory detail" in result

    def test_two_names_uses_and(self) -> None:
        result = format_introduction_guidance(["butler", "detective"])
        assert "**butler** and **detective**" in result

    def test_three_names_uses_oxford_comma(self) -> None:
        result = format_introduction_guidance(["butler", "detective", "maid"])
        assert "**butler**, **detective**, and **maid**" in result

    def test_arc_hints_adds_arc_framing(self) -> None:
        result = format_introduction_guidance(
            ["butler"],
            arc_hints={"butler": "revelation"},
        )
        assert "Arc-aware introduction" in result
        assert "facade" in result

    def test_arc_hints_transformation(self) -> None:
        result = format_introduction_guidance(
            ["mentor"],
            arc_hints={"mentor": "transformation"},
        )
        assert "starting state" in result

    def test_arc_hints_significance(self) -> None:
        result = format_introduction_guidance(
            ["letter"],
            arc_hints={"letter": "significance"},
        )
        assert "ordinary" in result

    def test_arc_hints_none_no_change(self) -> None:
        without = format_introduction_guidance(["butler"])
        with_none = format_introduction_guidance(["butler"], arc_hints=None)
        assert without == with_none

    def test_arc_hints_empty_dict_no_change(self) -> None:
        without = format_introduction_guidance(["butler"])
        with_empty = format_introduction_guidance(["butler"], arc_hints={})
        assert without == with_empty


# ---------------------------------------------------------------------------
# format_entity_arc_context
# ---------------------------------------------------------------------------


class TestFormatEntityArcContext:
    def _make_graph(self) -> Graph:
        """Build a minimal graph with entity arcs on a path node."""
        g = Graph.empty()

        # Entities
        g.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character"},
        )
        g.create_node(
            "entity::letter",
            {"type": "entity", "raw_id": "letter", "entity_type": "object"},
        )

        # Dilemma (needed for get_path_beat_sequence)
        g.create_node(
            "dilemma::trust",
            {"type": "dilemma", "raw_id": "trust", "question": "Can you trust?"},
        )

        # Path with entity_arcs
        g.create_node(
            "path::trust__yes",
            {
                "type": "path",
                "raw_id": "trust__yes",
                "dilemma_id": "dilemma::trust",
                "entity_arcs": [
                    {
                        "entity_id": "entity::mentor",
                        "arc_line": "trusted ally → doubts → revealed as spy",
                        "pivot_beat": "beat::b2",
                        "arc_type": "transformation",
                    },
                    {
                        "entity_id": "entity::letter",
                        "arc_line": "mundane note → proof of betrayal",
                        "pivot_beat": "beat::b2",
                        "arc_type": "significance",
                    },
                ],
            },
        )

        # Beats
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "Meet mentor",
                "paths": ["path::trust__yes"],
                "entities": ["entity::mentor"],
            },
        )
        g.create_node(
            "beat::b2",
            {
                "type": "beat",
                "raw_id": "b2",
                "summary": "Doubts surface",
                "paths": ["path::trust__yes"],
                "requires": ["beat::b1"],
                "entities": ["entity::mentor", "entity::letter"],
            },
        )
        g.create_node(
            "beat::b3",
            {
                "type": "beat",
                "raw_id": "b3",
                "summary": "Revelation",
                "paths": ["path::trust__yes"],
                "requires": ["beat::b2"],
                "entities": ["entity::mentor"],
            },
        )

        # belongs_to edges (beat → path)
        g.add_edge("belongs_to", "beat::b1", "path::trust__yes")
        g.add_edge("belongs_to", "beat::b2", "path::trust__yes")
        g.add_edge("belongs_to", "beat::b3", "path::trust__yes")

        # Passages
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "from_beat": "beat::b1",
                "entities": ["entity::mentor"],
            },
        )
        g.create_node(
            "passage::p2",
            {
                "type": "passage",
                "raw_id": "p2",
                "from_beat": "beat::b2",
                "entities": ["entity::mentor", "entity::letter"],
            },
        )
        g.create_node(
            "passage::p3",
            {
                "type": "passage",
                "raw_id": "p3",
                "from_beat": "beat::b3",
                "entities": ["entity::mentor"],
            },
        )

        # Arc
        g.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::trust__yes"],
                "sequence": ["beat::b1", "beat::b2", "beat::b3"],
            },
        )

        return g

    def test_pre_pivot_position(self) -> None:
        g = self._make_graph()
        result = format_entity_arc_context(g, "passage::p1", "arc::spine")
        assert "Entity Arc Context" in result
        assert "mentor" in result
        assert "transformation" in result
        assert "before pivot" in result

    def test_at_pivot_position(self) -> None:
        g = self._make_graph()
        result = format_entity_arc_context(g, "passage::p2", "arc::spine")
        assert "AT PIVOT" in result
        # Both mentor and letter should appear (both in passage::p2 entities)
        assert "mentor" in result
        assert "letter" in result

    def test_post_pivot_position(self) -> None:
        g = self._make_graph()
        result = format_entity_arc_context(g, "passage::p3", "arc::spine")
        assert "past pivot" in result
        assert "mentor" in result
        # letter not in passage::p3 entities, so should not appear
        assert "letter" not in result

    def test_no_arcs_returns_empty(self) -> None:
        g = self._make_graph()
        # Remove entity_arcs from path
        g.update_node("path::trust__yes", entity_arcs=[])
        result = format_entity_arc_context(g, "passage::p1", "arc::spine")
        assert result == ""

    def test_no_passage_returns_empty(self) -> None:
        g = self._make_graph()
        result = format_entity_arc_context(g, "passage::nonexistent", "arc::spine")
        assert result == ""

    def test_entity_not_in_passage_filtered(self) -> None:
        """Entity with arc but not present in passage should be excluded."""
        g = self._make_graph()
        # passage::p1 only has entity::mentor, not entity::letter
        result = format_entity_arc_context(g, "passage::p1", "arc::spine")
        assert "letter" not in result
        assert "mentor" in result


# ---------------------------------------------------------------------------
# compute_arc_hints
# ---------------------------------------------------------------------------


class TestComputeArcHints:
    def test_returns_hints_for_entities_with_arcs(self) -> None:
        g = Graph.empty()
        g.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character"},
        )
        g.create_node(
            "path::trust__yes",
            {
                "type": "path",
                "raw_id": "trust__yes",
                "entity_arcs": [
                    {"entity_id": "entity::mentor", "arc_type": "transformation"},
                ],
            },
        )
        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["path::trust__yes"]},
        )
        hints = compute_arc_hints(g, ["entity::mentor"], "arc::spine")
        assert hints == {"mentor": "transformation"}

    def test_returns_empty_for_no_arcs(self) -> None:
        g = Graph.empty()
        g.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character"},
        )
        g.create_node(
            "path::trust__yes",
            {"type": "path", "raw_id": "trust__yes"},
        )
        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["path::trust__yes"]},
        )
        hints = compute_arc_hints(g, ["entity::mentor"], "arc::spine")
        assert hints == {}

    def test_empty_entity_ids_returns_empty(self) -> None:
        g = Graph.empty()
        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["path::trust__yes"]},
        )
        hints = compute_arc_hints(g, [], "arc::spine")
        assert hints == {}

    def test_entity_without_arc_not_in_hints(self) -> None:
        g = Graph.empty()
        g.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character"},
        )
        g.create_node(
            "entity::bystander",
            {"type": "entity", "raw_id": "bystander", "entity_type": "character"},
        )
        g.create_node(
            "path::trust__yes",
            {
                "type": "path",
                "raw_id": "trust__yes",
                "entity_arcs": [
                    {"entity_id": "entity::mentor", "arc_type": "transformation"},
                ],
            },
        )
        g.create_node(
            "arc::spine",
            {"type": "arc", "raw_id": "spine", "paths": ["path::trust__yes"]},
        )
        hints = compute_arc_hints(g, ["entity::mentor", "entity::bystander"], "arc::spine")
        assert "mentor" in hints
        assert "bystander" not in hints


# ---------------------------------------------------------------------------
# Merged Passage Context Tests
# ---------------------------------------------------------------------------


class TestIsMergedPassage:
    """Tests for is_merged_passage helper."""

    def test_single_beat_passage_is_not_merged(self) -> None:
        passage = {"type": "passage", "from_beat": "beat::b1"}
        assert is_merged_passage(passage) is False

    def test_passage_with_from_beats_is_merged(self) -> None:
        passage = {
            "type": "passage",
            "from_beats": ["beat::b1", "beat::b2", "beat::b3"],
            "primary_beat": "beat::b1",
        }
        assert is_merged_passage(passage) is True

    def test_empty_from_beats_is_not_merged(self) -> None:
        passage = {"type": "passage", "from_beats": []}
        assert is_merged_passage(passage) is False

    def test_single_from_beats_is_not_merged(self) -> None:
        passage = {"type": "passage", "from_beats": ["beat::b1"]}
        assert is_merged_passage(passage) is False

    def test_none_from_beats_is_not_merged(self) -> None:
        passage = {"type": "passage", "from_beats": None}
        assert is_merged_passage(passage) is False


class TestFormatMergedPassageContext:
    """Tests for format_merged_passage_context."""

    def _make_merged_passage_graph(self) -> Graph:
        """Create a graph with a merged passage."""
        g = Graph.empty()

        # Entities
        g.create_node(
            "entity::pim",
            {
                "type": "entity",
                "raw_id": "pim",
                "entity_type": "character",
                "concept": "A curious child",
            },
        )
        g.create_node(
            "entity::manor",
            {
                "type": "entity",
                "raw_id": "manor",
                "entity_type": "location",
                "concept": "An old manor house",
            },
        )

        # Beats
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "Pim searches the study",
                "scene_type": "scene",
                "entities": ["entity::pim"],
                "location": "entity::manor",
            },
        )
        g.create_node(
            "beat::gap_1",
            {
                "type": "beat",
                "raw_id": "gap_1",
                "summary": "Transition",
                "scene_type": "micro_beat",
                "is_gap_beat": True,
                "transition_style": "smooth",
                "entities": ["entity::pim"],
                "location": "entity::manor",
            },
        )
        g.create_node(
            "beat::b2",
            {
                "type": "beat",
                "raw_id": "b2",
                "summary": "Pim finds the letter",
                "scene_type": "scene",
                "entities": ["entity::pim"],
                "location": "entity::manor",
            },
        )

        # Merged passage
        g.create_node(
            "passage::merged_b1",
            {
                "type": "passage",
                "raw_id": "merged_b1",
                "from_beats": ["beat::b1", "beat::gap_1", "beat::b2"],
                "primary_beat": "beat::b1",
                "merged_from": ["passage::b1", "passage::gap_1", "passage::b2"],
                "transition_points": [
                    {"index": 1, "style": "smooth", "note": "Continue in same scene"},
                    {"index": 2, "style": "smooth", "note": "Discovery moment"},
                ],
                "entities": ["entity::pim"],
            },
        )

        return g

    def test_returns_empty_for_nonexistent_passage(self) -> None:
        g = Graph.empty()
        result = format_merged_passage_context(g, "passage::nonexistent")
        assert result == ""

    def test_falls_back_for_non_merged_passage(self) -> None:
        g = Graph.empty()
        g.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "Simple beat", "scene_type": "scene"},
        )
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "from_beat": "beat::b1",
                "summary": "Simple passage",
            },
        )
        result = format_merged_passage_context(g, "passage::p1")
        # Should use standard format (not merged)
        assert "Merged Passage Context" not in result
        assert "Simple" in result

    def test_formats_merged_passage_header(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "## Merged Passage Context" in result

    def test_includes_primary_summary(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Primary Summary:**" in result
        assert "Pim searches the study" in result

    def test_includes_beat_sequence(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Beat Sequence:**" in result
        assert "[beat::b1] Pim searches the study" in result
        assert "[gap] (smooth transition)" in result
        assert "[beat::b2] Pim finds the letter" in result

    def test_includes_transition_guidance(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Transition Guidance:**" in result
        assert "Smooth" in result
        assert "Continue in same scene" in result

    def test_includes_writing_instruction(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Writing Instruction:**" in result
        assert "continuous prose" in result
        assert "one cohesive scene" in result

    def test_includes_entities(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Entities:**" in result
        assert "pim" in result

    def test_includes_location_when_shared(self) -> None:
        g = self._make_merged_passage_graph()
        result = format_merged_passage_context(g, "passage::merged_b1")
        assert "**Location:**" in result
        assert "manor" in result
        assert "unchanged throughout" in result


class TestFormatValidCharacters:
    """Tests for format_valid_characters function."""

    def test_empty_graph_returns_error_message(self) -> None:
        """No character entities returns directive error message."""
        g = Graph.empty()
        result = format_valid_characters(g)
        assert "ERROR:" in result
        assert "No character entities found" in result
        assert "Do NOT invent" in result

    def test_single_character_without_protagonist(self) -> None:
        """Single character listed correctly."""
        g = Graph.empty()
        g.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "category": "character",
                "concept": "A brave adventurer",
            },
        )
        result = format_valid_characters(g)
        assert "- hero: A brave adventurer" in result
        assert "PROTAGONIST" not in result
        assert "Protagonist:" not in result

    def test_character_marked_as_protagonist(self) -> None:
        """Protagonist gets marked and header added."""
        g = Graph.empty()
        g.create_node(
            "entity::kay",
            {
                "type": "entity",
                "raw_id": "kay",
                "category": "character",
                "concept": "The chosen one",
                "is_protagonist": True,
            },
        )
        result = format_valid_characters(g)
        assert "Protagonist: **kay**" in result
        assert "- kay (PROTAGONIST): The chosen one" in result

    def test_multiple_characters_with_protagonist(self) -> None:
        """Multiple characters with one protagonist."""
        g = Graph.empty()
        g.create_node(
            "entity::alice",
            {
                "type": "entity",
                "raw_id": "alice",
                "category": "character",
                "concept": "The hero",
                "is_protagonist": True,
            },
        )
        g.create_node(
            "entity::bob",
            {
                "type": "entity",
                "raw_id": "bob",
                "category": "character",
                "concept": "The sidekick",
            },
        )
        result = format_valid_characters(g)
        assert "Protagonist: **alice**" in result
        assert "- alice (PROTAGONIST): The hero" in result
        assert "- bob: The sidekick" in result

    def test_character_with_empty_concept(self) -> None:
        """Empty concept doesn't produce trailing colon."""
        g = Graph.empty()
        g.create_node(
            "entity::mystery",
            {
                "type": "entity",
                "raw_id": "mystery",
                "category": "character",
                "concept": "",
            },
        )
        result = format_valid_characters(g)
        assert "- mystery" in result
        assert "- mystery:" not in result  # No trailing colon

    def test_non_character_entities_excluded(self) -> None:
        """Location and item entities are not included."""
        g = Graph.empty()
        g.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "category": "character",
                "concept": "Main character",
            },
        )
        g.create_node(
            "entity::castle",
            {
                "type": "entity",
                "raw_id": "castle",
                "category": "location",
                "concept": "A dark castle",
            },
        )
        g.create_node(
            "entity::sword",
            {
                "type": "entity",
                "raw_id": "sword",
                "category": "item",
                "concept": "A magic sword",
            },
        )
        result = format_valid_characters(g)
        assert "hero" in result
        assert "castle" not in result
        assert "sword" not in result

    def test_fallback_to_stripped_id_when_raw_id_missing(self) -> None:
        """Falls back to scope-stripped entity ID if raw_id missing."""
        g = Graph.empty()
        g.create_node(
            "entity::fallback_char",
            {
                "type": "entity",
                "category": "character",
                "concept": "Test character",
                # raw_id intentionally missing
            },
        )
        result = format_valid_characters(g)
        assert "fallback_char" in result


class TestGetPendingSpokeLabels:
    """Tests for get_pending_spoke_labels function."""

    def test_returns_empty_for_non_hub_passage(self) -> None:
        """Returns empty list for passages without spoke choices."""
        from questfoundry.graph.fill_context import get_pending_spoke_labels

        g = Graph.empty()
        g.create_node("passage::normal", {"type": "passage", "raw_id": "normal"})
        result = get_pending_spoke_labels(g, "passage::normal")
        assert result == []

    def test_returns_pending_spokes_without_labels(self) -> None:
        """Returns spokes that don't have labels set."""
        from questfoundry.graph.fill_context import get_pending_spoke_labels

        g = Graph.empty()
        # Hub passage
        g.create_node("passage::hub", {"type": "passage", "raw_id": "hub"})
        # Spoke passage
        g.create_node(
            "passage::spoke_hub_0",
            {
                "type": "passage",
                "raw_id": "spoke_hub_0",
                "summary": "Examine the mysterious artifact",
            },
        )
        # Choice without label (pending)
        g.create_node(
            "choice::hub__spoke_0",
            {
                "type": "choice",
                # no label field
                "label_style": "evocative",
            },
        )
        g.add_edge("choice_from", "choice::hub__spoke_0", "passage::hub")
        g.add_edge("choice_to", "choice::hub__spoke_0", "passage::spoke_hub_0")

        result = get_pending_spoke_labels(g, "passage::hub")
        assert len(result) == 1
        assert result[0]["choice_id"] == "choice::hub__spoke_0"
        assert result[0]["spoke_summary"] == "Examine the mysterious artifact"
        assert result[0]["label_style"] == "evocative"

    def test_excludes_spokes_with_labels(self) -> None:
        """Excludes spokes that already have labels."""
        from questfoundry.graph.fill_context import get_pending_spoke_labels

        g = Graph.empty()
        g.create_node("passage::hub", {"type": "passage", "raw_id": "hub"})
        g.create_node(
            "passage::spoke_hub_0",
            {"type": "passage", "raw_id": "spoke_hub_0", "summary": "Some spoke"},
        )
        # Choice WITH label (not pending)
        g.create_node(
            "choice::hub__spoke_0",
            {"type": "choice", "label": "Examine the clock"},
        )
        g.add_edge("choice_from", "choice::hub__spoke_0", "passage::hub")
        g.add_edge("choice_to", "choice::hub__spoke_0", "passage::spoke_hub_0")

        result = get_pending_spoke_labels(g, "passage::hub")
        assert result == []

    def test_excludes_non_spoke_choices(self) -> None:
        """Excludes choices to non-spoke passages."""
        from questfoundry.graph.fill_context import get_pending_spoke_labels

        g = Graph.empty()
        g.create_node("passage::hub", {"type": "passage", "raw_id": "hub"})
        # Normal passage (not a spoke - raw_id doesn't start with spoke_)
        g.create_node(
            "passage::next_scene",
            {"type": "passage", "raw_id": "next_scene", "summary": "Continue story"},
        )
        g.create_node(
            "choice::hub__next",
            {"type": "choice"},  # No label
        )
        g.add_edge("choice_from", "choice::hub__next", "passage::hub")
        g.add_edge("choice_to", "choice::hub__next", "passage::next_scene")

        result = get_pending_spoke_labels(g, "passage::hub")
        assert result == []


class TestFormatSpokeContext:
    """Tests for format_spoke_context function."""

    def test_returns_empty_for_non_hub(self) -> None:
        """Returns empty string for passages without pending spokes."""
        from questfoundry.graph.fill_context import format_spoke_context

        g = Graph.empty()
        g.create_node("passage::normal", {"type": "passage", "raw_id": "normal"})
        result = format_spoke_context(g, "passage::normal")
        assert result == ""

    def test_formats_spoke_context_for_hub(self) -> None:
        """Formats context string for hub with pending spokes."""
        from questfoundry.graph.fill_context import format_spoke_context

        g = Graph.empty()
        g.create_node("passage::hub", {"type": "passage", "raw_id": "hub"})
        g.create_node(
            "passage::spoke_hub_0",
            {"type": "passage", "raw_id": "spoke_hub_0", "summary": "Examine the map"},
        )
        g.create_node(
            "choice::hub__spoke_0",
            {"type": "choice", "label_style": "functional"},
        )
        g.add_edge("choice_from", "choice::hub__spoke_0", "passage::hub")
        g.add_edge("choice_to", "choice::hub__spoke_0", "passage::spoke_hub_0")

        result = format_spoke_context(g, "passage::hub")
        assert "Exploration Options" in result
        assert "Examine the map" in result
        assert "functional" in result
        assert "spoke_labels" in result


# ---------------------------------------------------------------------------
# extract_used_imagery
# ---------------------------------------------------------------------------


class TestExtractUsedImagery:
    def test_empty_input(self) -> None:
        assert extract_used_imagery([]) == []

    def test_no_repetition(self) -> None:
        result = extract_used_imagery(["The sun rose over the hills."])
        assert result == []

    def test_bigram_extraction(self) -> None:
        texts = [
            "The amber glow filled the room with amber glow.",
            "She saw the amber glow through the window.",
            "Once more the amber glow crept in.",
        ]
        result = extract_used_imagery(texts, min_bigram_count=2)
        assert any("amber glow" in item for item in result)

    def test_repeated_word_extraction(self) -> None:
        texts = [
            "The shadow crept through the corridor of shadows.",
            "Another shadow appeared in the distant shadow.",
            "Shadows danced where shadows fell before.",
        ]
        result = extract_used_imagery(texts, min_word_count=3, min_word_length=5)
        # "shadow" or "shadows" should appear
        assert len(result) > 0

    def test_top_n_limit(self) -> None:
        texts = [f"word{i} word{i} word{i} word{i}" for i in range(20)]
        result = extract_used_imagery(texts, top_n=5)
        assert len(result) <= 5

    def test_combined_bigrams_and_words(self) -> None:
        texts = [
            "The weight of choice pressed heavily. Weight of choice again.",
            "Weight of choice once more. The ancient stones whispered ancient stones.",
            "Ancient stones echoed. Weight of choice eternal.",
        ]
        result = extract_used_imagery(texts, min_bigram_count=2, min_word_count=2)
        assert len(result) > 0


class TestFormatBlueprintContext:
    def test_none_returns_fallback(self) -> None:
        result = format_blueprint_context(None)
        assert "no blueprint available" in result

    def test_empty_dict_returns_fallback(self) -> None:
        result = format_blueprint_context({})
        assert "no blueprint available" in result

    def test_full_blueprint(self) -> None:
        bp = {
            "sensory_palette": ["rain on cobblestones", "woodsmoke", "distant bells"],
            "character_gestures": ["fidgets with ring", "avoids eye contact"],
            "opening_move": "sensory_image",
            "craft_constraint": "Begin each paragraph with a different sense",
            "emotional_arc_word": "dread",
        }
        result = format_blueprint_context(bp)
        assert "Sensory Palette" in result
        assert "rain on cobblestones" in result
        assert "woodsmoke" in result
        assert "distant bells" in result
        assert "Character Gestures" in result
        assert "fidgets with ring" in result
        assert "Opening Move" in result
        assert "sensory_image" in result
        assert "Craft Constraint" in result
        assert "Begin each paragraph" in result
        assert "Emotional Arc Word" in result
        assert "dread" in result

    def test_palette_only(self) -> None:
        bp = {"sensory_palette": ["fog", "salt air", "creaking wood"]}
        result = format_blueprint_context(bp)
        assert "Sensory Palette" in result
        assert "fog" in result
        assert "Character Gestures" not in result
        assert "Opening Move" not in result

    def test_empty_lists_produces_empty_blueprint(self) -> None:
        bp = {"sensory_palette": [], "character_gestures": []}
        result = format_blueprint_context(bp)
        assert result == "(empty blueprint)"

    def test_no_craft_constraint_omits_section(self) -> None:
        bp = {
            "sensory_palette": ["firelight"],
            "opening_move": "dialogue",
            "craft_constraint": "",
            "emotional_arc_word": "hope",
        }
        result = format_blueprint_context(bp)
        assert "Craft Constraint" not in result
        assert "Opening Move" in result


class TestFormatUsedImageryBlocklist:
    def test_empty_blocklist(self) -> None:
        result = format_used_imagery_blocklist([])
        assert "no repeated imagery" in result
        assert "full creative freedom" in result

    def test_single_item(self) -> None:
        result = format_used_imagery_blocklist(["amber glow"])
        assert "DO NOT reuse" in result
        assert '"amber glow"' in result
        assert "fresh sensory material" in result

    def test_multiple_items(self) -> None:
        items = ["amber glow", "weight of choice", "ancient stones"]
        result = format_used_imagery_blocklist(items)
        for item in items:
            assert f'"{item}"' in result
        assert "DO NOT reuse" in result
