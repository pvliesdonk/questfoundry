"""Tests for LLM-phase fallback branches in _phase_resolve_temporal_hints.

Covers the three fallback paths that have no integration-level test coverage:
1. LLM call raises GrowStageError → mechanical defaults applied for all swap pairs.
2. LLM returns invalid group_id → warning logged, fallback to default_drop for that pair.
3. LLM returns invalid drop_beat_id → warning logged, fallback to default_drop for that pair.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.models.grow import (
    ConflictGroupResolution,
    TemporalResolutionOutput,
    TransitionBridge,
    TransitionGapsOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_dilemma_swap_graph() -> Graph:
    """Build a minimal two-dilemma graph where two beats form a swap pair.

    Uses before_introduce hints (weaker) so neither cycles alone against the
    base DAG, producing a swap pair P1 = (beat::mt_intro, beat::aq_intro).

    Dilemma IDs: artifact_quest (aq) < mentor_trust (mt) alphabetically.
    Heuristic base DAG: aq_commit ≺ mt_commit.
    Within-path: aq_intro ≺ aq_commit, mt_intro ≺ mt_commit.

    H1 on mt_intro: before_introduce artifact_quest → mt_intro ≺ aq_intro.
      Alone: no path aq_intro → mt_intro in base DAG, so no solo cycle.
    H2 on aq_intro: before_introduce mentor_trust → aq_intro ≺ mt_intro.
      Alone: no path mt_intro → aq_intro in base DAG, so no solo cycle.
    Together: cycle → swap pair.
    """
    graph = Graph.empty()
    for dil, path_prefix in (
        ("artifact_quest", "aq"),
        ("mentor_trust", "mt"),
    ):
        graph.create_node(
            f"dilemma::{dil}",
            {"type": "dilemma", "raw_id": dil},
        )
        graph.create_node(
            f"path::{path_prefix}_path",
            {
                "type": "path",
                "raw_id": f"{path_prefix}_path",
                "dilemma_id": f"dilemma::{dil}",
                "is_canonical": True,
            },
        )
        graph.create_node(
            f"beat::{path_prefix}_intro",
            {
                "type": "beat",
                "raw_id": f"{path_prefix}_intro",
                "summary": f"{dil} intro.",
                "dilemma_impacts": [{"dilemma_id": f"dilemma::{dil}", "effect": "advances"}],
            },
        )
        graph.create_node(
            f"beat::{path_prefix}_commit",
            {
                "type": "beat",
                "raw_id": f"{path_prefix}_commit",
                "summary": f"{dil} commit.",
                "dilemma_impacts": [{"dilemma_id": f"dilemma::{dil}", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", f"beat::{path_prefix}_intro", f"path::{path_prefix}_path")
        graph.add_edge("belongs_to", f"beat::{path_prefix}_commit", f"path::{path_prefix}_path")
        graph.add_edge("predecessor", f"beat::{path_prefix}_commit", f"beat::{path_prefix}_intro")

    graph.add_edge("concurrent", "dilemma::artifact_quest", "dilemma::mentor_trust")

    # Add before_introduce hints so both hints are only problematic together
    graph.update_node(
        "beat::mt_intro",
        temporal_hint={
            "relative_to": "dilemma::artifact_quest",
            "position": "before_introduce",
        },
    )
    graph.update_node(
        "beat::aq_intro",
        temporal_hint={
            "relative_to": "dilemma::mentor_trust",
            "position": "before_introduce",
        },
    )
    return graph


def _make_grow_stage_instance() -> Any:
    """Create a GrowStage instance with minimal attributes set for testing.

    Returns a GrowStage with _grow_llm_call patched as an AsyncMock so tests
    can control what the LLM returns without making real API calls.
    """
    from questfoundry.pipeline.stages.grow import GrowStage

    stage = GrowStage()
    stage._provider_name = "mock/test"
    stage._serialize_model = None
    stage._serialize_provider_name = None
    stage._callbacks = None
    return stage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseResolveTemporalHintsFallbacks:
    """Tests for the three fallback paths in _phase_resolve_temporal_hints."""

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_mechanical_defaults(self) -> None:
        """When _grow_llm_call raises GrowStageError, mechanical defaults are applied.

        The swap pair's default_drop (computed heuristically) should be stripped.
        No TemporalHintResolutionInvariantError should be raised.
        """
        from questfoundry.pipeline.stages.grow._helpers import GrowStageError

        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        with patch.object(
            stage,
            "_grow_llm_call",
            new=AsyncMock(side_effect=GrowStageError("LLM unavailable")),
        ):
            result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"

        # The mechanical default should have been stripped — one of the two beats
        # should have temporal_hint=None, and the other should still have a hint.
        beat_nodes = graph.get_nodes_by_type("beat")
        mt_hint = beat_nodes.get("beat::mt_intro", {}).get("temporal_hint")
        aq_hint = beat_nodes.get("beat::aq_intro", {}).get("temporal_hint")
        # Exactly one of the two should be stripped (None)
        hints = [mt_hint, aq_hint]
        assert hints.count(None) == 1, (
            f"Expected exactly one hint stripped as mechanical default fallback; "
            f"mt_hint={mt_hint!r}, aq_hint={aq_hint!r}"
        )

    @pytest.mark.asyncio
    async def test_invalid_group_id_falls_back_to_mechanical_default(self) -> None:
        """When LLM returns an unknown group_id, that resolution is skipped.

        The missing-resolution path fills in the mechanical default instead.
        """
        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        bad_resolution = TemporalResolutionOutput(
            resolutions=[
                # Use a group_id that doesn't exist ("P99" instead of "P1")
                ConflictGroupResolution(
                    group_id="P99",
                    drop_beat_id="beat::mt_intro",
                    reason="wrong group id",
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            new=AsyncMock(return_value=(bad_resolution, 1, 50)),
        ):
            result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"

        # Since P99 was invalid and P1 was missing from resolutions, the
        # missing-resolution fallback should have applied the mechanical default.
        beat_nodes = graph.get_nodes_by_type("beat")
        mt_hint = beat_nodes.get("beat::mt_intro", {}).get("temporal_hint")
        aq_hint = beat_nodes.get("beat::aq_intro", {}).get("temporal_hint")
        hints = [mt_hint, aq_hint]
        assert hints.count(None) == 1, (
            f"Expected mechanical default applied for unresolved swap pair; "
            f"mt_hint={mt_hint!r}, aq_hint={aq_hint!r}"
        )

    @pytest.mark.asyncio
    async def test_invalid_drop_beat_id_falls_back_to_mechanical_default(self) -> None:
        """When LLM returns a drop_beat_id that is not one of the two swap options,
        the resolution is rejected and the mechanical default_drop is used instead.
        """
        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        bad_resolution = TemporalResolutionOutput(
            resolutions=[
                # P1 exists but drop_beat_id is a completely different beat
                ConflictGroupResolution(
                    group_id="P1",
                    drop_beat_id="beat::aq_commit",
                    reason="wrong beat id",
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            new=AsyncMock(return_value=(bad_resolution, 1, 50)),
        ):
            result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"

        # The invalid drop_beat_id should have been ignored; mechanical default applied.
        beat_nodes = graph.get_nodes_by_type("beat")
        mt_hint = beat_nodes.get("beat::mt_intro", {}).get("temporal_hint")
        aq_hint = beat_nodes.get("beat::aq_intro", {}).get("temporal_hint")
        hints = [mt_hint, aq_hint]
        assert hints.count(None) == 1, (
            f"Expected mechanical default applied after invalid drop_beat_id; "
            f"mt_hint={mt_hint!r}, aq_hint={aq_hint!r}"
        )

    @pytest.mark.asyncio
    async def test_no_conflicts_returns_early_no_llm_call(self) -> None:
        """When no temporal hint conflicts exist, phase returns immediately.

        This is the early return path — no LLM call is made, function returns
        with status='completed' and detail about no conflicts.
        """
        # Build a graph with no temporal hints
        graph = Graph.empty()
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1", "is_canonical": True},
        )
        graph.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "b1",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"
        assert "No temporal hint conflicts" in result.detail

    @pytest.mark.asyncio
    async def test_happy_path_valid_llm_response_and_postcondition_pass(self) -> None:
        """Happy path: swap pairs exist, LLM provides valid response, postcondition passes.

        This is the primary success path — LLM is called, resolutions are applied,
        postcondition check succeeds, phase returns completed status.
        """
        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        # LLM returns a valid resolution for P1
        valid_resolution = TemporalResolutionOutput(
            resolutions=[
                ConflictGroupResolution(
                    group_id="P1",
                    drop_beat_id="beat::mt_intro",
                    reason="Drop mentor_trust hint to resolve conflict",
                ),
            ]
        )

        with (
            patch.object(
                stage,
                "_grow_llm_call",
                new=AsyncMock(return_value=(valid_resolution, 1, 100)),
            ),
            patch(
                "questfoundry.graph.grow_algorithms.verify_hints_acyclic",
                return_value=[],
            ),
        ):
            result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"

        # Verify that mt_intro hint was stripped, aq_intro hint remains
        beat_nodes = graph.get_nodes_by_type("beat")
        mt_hint = beat_nodes.get("beat::mt_intro", {}).get("temporal_hint")
        aq_hint = beat_nodes.get("beat::aq_intro", {}).get("temporal_hint")
        assert mt_hint is None, "Expected mt_intro hint to be dropped"
        assert aq_hint is not None, "Expected aq_intro hint to remain"

    @pytest.mark.asyncio
    async def test_postcondition_fails_raises_invariant_error(self) -> None:
        """When postcondition fails (surviving hints still cycle), raises TemporalHintResolutionInvariantError.

        This is the error path — LLM resolution is applied, but postcondition
        check detects that hints still cycle. The phase must fail loudly.
        """
        from questfoundry.graph.errors import TemporalHintResolutionInvariantError

        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        valid_resolution = TemporalResolutionOutput(
            resolutions=[
                ConflictGroupResolution(
                    group_id="P1",
                    drop_beat_id="beat::mt_intro",
                    reason="Drop mentor_trust hint",
                ),
            ]
        )

        with (
            patch.object(
                stage,
                "_grow_llm_call",
                new=AsyncMock(return_value=(valid_resolution, 1, 100)),
            ),
            patch(
                "questfoundry.graph.grow_algorithms.verify_hints_acyclic",
                return_value=["beat::aq_intro"],
            ),
            pytest.raises(TemporalHintResolutionInvariantError) as exc_info,
        ):
            await stage._phase_resolve_temporal_hints(graph, mock_model)

        error = exc_info.value
        assert "beat::aq_intro" in str(error)
        assert "still cycle" in str(error)

    @pytest.mark.asyncio
    async def test_missing_resolution_in_llm_response_applies_mechanical_default(
        self,
    ) -> None:
        """When LLM response omits a swap pair, mechanical default is applied for that pair.

        The missing-resolution path applies default_drop when LLM doesn't provide
        a resolution for a swap pair that exists.
        """
        graph = _make_two_dilemma_swap_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        # LLM returns empty response (no resolutions provided)
        empty_response = TemporalResolutionOutput(resolutions=[])

        with (
            patch.object(
                stage,
                "_grow_llm_call",
                new=AsyncMock(return_value=(empty_response, 1, 100)),
            ),
            patch(
                "questfoundry.graph.grow_algorithms.verify_hints_acyclic",
                return_value=[],
            ),
        ):
            result = await stage._phase_resolve_temporal_hints(graph, mock_model)

        assert result.phase == "resolve_temporal_hints"
        assert result.status == "completed"

        # At least one hint should be dropped (mechanical default applied)
        beat_nodes = graph.get_nodes_by_type("beat")
        mt_hint = beat_nodes.get("beat::mt_intro", {}).get("temporal_hint")
        aq_hint = beat_nodes.get("beat::aq_intro", {}).get("temporal_hint")
        hints = [mt_hint, aq_hint]
        assert hints.count(None) >= 1, "Expected mechanical default applied for missing resolution"


class TestTemporalHintErrors:
    """Tests for TemporalHintResolutionInvariantError."""

    def test_temporal_hint_error_formatting(self) -> None:
        """TemporalHintResolutionInvariantError formats message correctly.

        Tests __init__ and __str__ to ensure error message includes
        still_cyclic beat IDs and dropped beat IDs.
        """
        from questfoundry.graph.errors import TemporalHintResolutionInvariantError

        still_cyclic = ["beat::b1", "beat::b2"]
        dropped = {"beat::b3", "beat::b4"}

        error = TemporalHintResolutionInvariantError(still_cyclic, dropped)

        msg = str(error)
        assert "2 hint(s) still cycle" in msg
        assert "`beat::b1`" in msg
        assert "`beat::b2`" in msg
        assert "dropped" in msg.lower()

    def test_temporal_hint_error_stores_attributes(self) -> None:
        """TemporalHintResolutionInvariantError stores still_cyclic and dropped attributes."""
        from questfoundry.graph.errors import TemporalHintResolutionInvariantError

        still_cyclic = ["beat::b1"]
        dropped = {"beat::b2"}

        error = TemporalHintResolutionInvariantError(still_cyclic, dropped)

        assert error.still_cyclic == still_cyclic
        assert error.dropped == dropped


class TestBuildHintConflictGraph:
    """Tests for build_hint_conflict_graph function."""

    def test_empty_graph_returns_empty_result(self) -> None:
        """build_hint_conflict_graph returns empty result for empty graph."""
        from questfoundry.graph.grow_algorithms import build_hint_conflict_graph

        graph = Graph.empty()
        result = build_hint_conflict_graph(graph)

        assert result.conflicts == []
        assert result.mandatory_drops == set()
        assert result.swap_pairs == []
        assert result.minimum_drop_set == set()

    def test_graph_with_no_hints_returns_empty_result(self) -> None:
        """build_hint_conflict_graph returns empty result when no temporal hints present."""
        from questfoundry.graph.grow_algorithms import build_hint_conflict_graph

        graph = Graph.empty()
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1", "is_canonical": True},
        )
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "b1"})
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        result = build_hint_conflict_graph(graph)

        assert result.conflicts == []
        assert result.mandatory_drops == set()
        assert result.swap_pairs == []

    def test_build_conflict_graph_detects_swap_pairs(self) -> None:
        """build_hint_conflict_graph correctly identifies swap pairs in two-dilemma graph."""
        from questfoundry.graph.grow_algorithms import build_hint_conflict_graph

        graph = _make_two_dilemma_swap_graph()
        result = build_hint_conflict_graph(graph)

        # Should detect swap pair P1
        assert len(result.swap_pairs) == 1
        beat_a, beat_b = result.swap_pairs[0]
        assert {beat_a, beat_b} == {"beat::mt_intro", "beat::aq_intro"}


class TestVerifyHintsAcyclic:
    """Tests for verify_hints_acyclic postcondition check."""

    def test_empty_graph_returns_empty_list(self) -> None:
        """verify_hints_acyclic returns empty list for empty graph."""
        from questfoundry.graph.grow_algorithms import verify_hints_acyclic

        graph = Graph.empty()
        result = verify_hints_acyclic(graph, set())

        assert result == []

    def test_graph_with_no_surviving_hints_returns_empty_list(self) -> None:
        """verify_hints_acyclic returns empty list when no surviving hints."""
        from questfoundry.graph.grow_algorithms import verify_hints_acyclic

        graph = Graph.empty()
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1", "is_canonical": True},
        )
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "b1"})
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        result = verify_hints_acyclic(graph, set())

        assert result == []

    def test_surviving_hints_without_cycles(self) -> None:
        """verify_hints_acyclic returns empty list when surviving hints don't cycle."""
        from questfoundry.graph.grow_algorithms import verify_hints_acyclic

        graph = _make_two_dilemma_swap_graph()
        # Both beats still have hints, so just test with empty surviving set
        result = verify_hints_acyclic(graph, set())

        assert result == []


# ---------------------------------------------------------------------------
# Helpers for _phase_transition_gaps tests
# ---------------------------------------------------------------------------


def _make_cross_dilemma_hard_transition_graph() -> Graph:
    """Build a minimal graph with one cross-dilemma hard transition.

    Two dilemmas, each with one path and one beat. The beats have no shared
    entities or location. A predecessor edge connects them cross-dilemma,
    producing exactly one hard transition when detect_cross_dilemma_hard_transitions
    is called.

    Beat layout:
      - beat::aq_beat: dilemma artifact_quest, location "library", entities ["book"]
      - beat::mt_beat: dilemma mentor_trust, location "forest", entities ["sword"]
    Predecessor edge: mt_beat → aq_beat (later → earlier, i.e. mt_beat comes after aq_beat).
    Expected transition: (beat::aq_beat, beat::mt_beat).
    """
    graph = Graph.empty()

    for dil, path_prefix in (
        ("artifact_quest", "aq"),
        ("mentor_trust", "mt"),
    ):
        graph.create_node(
            f"dilemma::{dil}",
            {"type": "dilemma", "raw_id": dil},
        )
        graph.create_node(
            f"path::{path_prefix}_path",
            {
                "type": "path",
                "raw_id": f"{path_prefix}_path",
                "dilemma_id": f"dilemma::{dil}",
                "is_canonical": True,
            },
        )

    graph.create_node(
        "beat::aq_beat",
        {
            "type": "beat",
            "raw_id": "aq_beat",
            "summary": "Protagonist finds the artifact in the library.",
            "entities": ["entity::book"],
            "location": "library",
            "dilemma_impacts": [{"dilemma_id": "dilemma::artifact_quest", "effect": "advances"}],
        },
    )
    graph.create_node(
        "beat::mt_beat",
        {
            "type": "beat",
            "raw_id": "mt_beat",
            "summary": "Mentor challenges the hero in the forest.",
            "entities": ["entity::sword"],
            "location": "forest",
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
        },
    )

    graph.add_edge("belongs_to", "beat::aq_beat", "path::aq_path")
    graph.add_edge("belongs_to", "beat::mt_beat", "path::mt_path")

    # Cross-dilemma predecessor: mt_beat comes after aq_beat
    graph.add_edge("predecessor", "beat::mt_beat", "beat::aq_beat")

    return graph


# ---------------------------------------------------------------------------
# Tests for _phase_transition_gaps
# ---------------------------------------------------------------------------


class TestPhaseTransitionGaps:
    """Tests for _phase_transition_gaps insertion logic."""

    @pytest.mark.asyncio
    async def test_transition_gaps_inserts_bridge_beat(self) -> None:
        """Happy path: bridge beat is inserted and predecessor edges are rewired.

        Mock _grow_llm_call returns a TransitionGapsOutput with one bridge whose
        transition_id matches the expected 'earlier|later' format. The phase must:
        - Create a new beat::transition_aq_beat_mt_beat node with role='transition_beat'
        - Remove the original predecessor edge (mt_beat → aq_beat)
        - Add two new predecessor edges:
            transition → aq_beat (transition comes after aq_beat)
            mt_beat → transition (mt_beat comes after transition)
        - Return status='completed' with 'Inserted 1' in detail
        """
        graph = _make_cross_dilemma_hard_transition_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        # Expected transition_id mirrors the format built in _phase_transition_gaps:
        # "{earlier}|{later}" = "beat::aq_beat|beat::mt_beat"
        bridge_output = TransitionGapsOutput(
            bridges=[
                TransitionBridge(
                    transition_id="beat::aq_beat|beat::mt_beat",
                    summary="The hero moves from the library to the forest, feeling the weight of destiny.",
                    entities=[],
                    location="crossroads",
                )
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            new=AsyncMock(return_value=(bridge_output, 1, 80)),
        ):
            result = await stage._phase_transition_gaps(graph, mock_model)

        # Phase result
        assert result.phase == "transition_gaps"
        assert result.status == "completed"
        assert "Inserted 1" in result.detail

        # New transition beat node created
        beat_nodes = graph.get_nodes_by_type("beat")
        assert "beat::transition_aq_beat_mt_beat" in beat_nodes, (
            f"Expected beat::transition_aq_beat_mt_beat; got {list(beat_nodes.keys())}"
        )
        transition_node = beat_nodes["beat::transition_aq_beat_mt_beat"]
        assert transition_node.get("role") == "transition_beat"

        # Old predecessor edge is gone
        old_edges = graph.get_edges(
            from_id="beat::mt_beat", to_id="beat::aq_beat", edge_type="predecessor"
        )
        assert old_edges == [], "Original predecessor edge should have been removed"

        # Two new predecessor edges exist
        transition_to_earlier = graph.get_edges(
            from_id="beat::transition_aq_beat_mt_beat",
            to_id="beat::aq_beat",
            edge_type="predecessor",
        )
        assert len(transition_to_earlier) == 1, "Expected predecessor(transition_beat → aq_beat)"

        later_to_transition = graph.get_edges(
            from_id="beat::mt_beat",
            to_id="beat::transition_aq_beat_mt_beat",
            edge_type="predecessor",
        )
        assert len(later_to_transition) == 1, "Expected predecessor(mt_beat → transition_beat)"

    @pytest.mark.asyncio
    async def test_transition_gaps_warns_on_zero_matches(self) -> None:
        """When LLM returns a bridge with a wrong transition_id, nothing is inserted.

        Mock _grow_llm_call returns a bridge whose transition_id does not match any
        key in transition_map. The phase must:
        - Not create any new beat nodes
        - Return status='completed' with 'Inserted 0' in detail
        (The implementation also logs 'phase4g_no_bridges_matched', but we verify
        the observable graph state rather than structlog internals.)
        """
        graph = _make_cross_dilemma_hard_transition_graph()
        stage = _make_grow_stage_instance()
        mock_model = MagicMock()

        # transition_id that deliberately doesn't match
        bad_bridge_output = TransitionGapsOutput(
            bridges=[
                TransitionBridge(
                    transition_id="beat::nonexistent|beat::also_missing",
                    summary="A bridge to nowhere.",
                    entities=[],
                    location="",
                )
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            new=AsyncMock(return_value=(bad_bridge_output, 1, 40)),
        ):
            result = await stage._phase_transition_gaps(graph, mock_model)

        # Phase result
        assert result.phase == "transition_gaps"
        assert result.status == "completed"
        assert "Inserted 0" in result.detail

        # No new beat nodes were created (graph still has only the original two beats)
        beat_nodes = graph.get_nodes_by_type("beat")
        assert "beat::transition_aq_beat_mt_beat" not in beat_nodes, (
            "No transition beat should have been created for a mismatched transition_id"
        )
        # Only the original two beat nodes should exist
        assert set(beat_nodes.keys()) == {"beat::aq_beat", "beat::mt_beat"}
