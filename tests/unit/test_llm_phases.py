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
from questfoundry.models.grow import ConflictGroupResolution, TemporalResolutionOutput

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
