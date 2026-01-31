"""Tests for FILL context formatting functions."""

from __future__ import annotations

import pytest

from questfoundry.graph.fill_context import (
    compute_open_questions,
    derive_pacing,
    format_dramatic_questions,
    format_dream_vision,
    format_entity_states,
    format_grow_summary,
    format_lookahead_context,
    format_narrative_context,
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
