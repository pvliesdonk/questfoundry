"""Tests for POLISH LLM phase helpers — pacing detection and Y-shape compatibility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.polish.llm_phases import (
    _detect_duplicate_labels_in_passage,
    _detect_pacing_flags,
)
from questfoundry.pipeline.stages.polish.stage import PolishStage


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with defaults."""
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


def _build_y_shape_for_pacing() -> tuple[Graph, dict, list]:
    """Build a minimal Y-shape: shared_setup (dual) → commit_a (single) → post_a (single).

    Returns:
        (graph, beat_nodes, predecessor_edges) ready for _detect_pacing_flags.
    """
    graph = Graph.empty()
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})

    _make_beat(graph, "beat::shared_setup", "Shared setup beat.")
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__a")
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__b")

    _make_beat(graph, "beat::commit_a", "Commit A beat.", scene_type="scene")
    graph.add_edge("belongs_to", "beat::commit_a", "path::trust__a")
    graph.add_edge("predecessor", "beat::commit_a", "beat::shared_setup")

    _make_beat(graph, "beat::post_a", "Post-commit A beat.", scene_type="sequel")
    graph.add_edge("belongs_to", "beat::post_a", "path::trust__a")
    graph.add_edge("predecessor", "beat::post_a", "beat::commit_a")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    return graph, beat_nodes, predecessor_edges


def test_pacing_issues_does_not_raise_on_dual_belongs_to() -> None:
    """Pacing-issue scan works on Y-shape graphs (dual belongs_to for pre-commit beats)."""
    graph, beat_nodes, predecessor_edges = _build_y_shape_for_pacing()
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)
    # Not asserting issue count — just that we didn't raise.
    assert isinstance(result, list)


def test_pacing_flags_consecutive_scenes_on_single_path() -> None:
    """3+ consecutive scene beats on a single path are flagged."""
    graph = Graph.empty()
    graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
    for i in range(1, 5):
        _make_beat(graph, f"beat::s{i}", f"Scene {i}", scene_type="scene")
        graph.add_edge("belongs_to", f"beat::s{i}", "path::p1")
    for i in range(2, 5):
        graph.add_edge("predecessor", f"beat::s{i}", f"beat::s{i - 1}")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)

    consecutive = [f for f in result if f["issue_type"] == "consecutive_scene"]
    assert len(consecutive) >= 1


def test_pacing_path_id_reported_for_dual_belongs_to_beat() -> None:
    """When a flag involves a dual-membership beat, path_id is a non-empty string."""
    graph = Graph.empty()
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})

    # 3 scene beats: shared (dual), commit (single), post (single)
    for beat_id, paths in [
        ("beat::s1", ["path::trust__a", "path::trust__b"]),
        ("beat::s2", ["path::trust__a"]),
        ("beat::s3", ["path::trust__a"]),
    ]:
        _make_beat(graph, beat_id, f"Scene {beat_id}", scene_type="scene")
        for path in paths:
            graph.add_edge("belongs_to", beat_id, path)
    graph.add_edge("predecessor", "beat::s2", "beat::s1")
    graph.add_edge("predecessor", "beat::s3", "beat::s2")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)

    consecutive = [f for f in result if f["issue_type"] == "consecutive_scene"]
    assert len(consecutive) >= 1
    assert all(isinstance(f["path_id"], str) for f in consecutive)
    assert all(f["path_id"] != "" for f in consecutive)


# ---------------------------------------------------------------------------
# Phase 5a: R-5.2 duplicate-label detection
# ---------------------------------------------------------------------------


def test_detect_duplicate_labels_empty_returns_empty() -> None:
    """No choice specs → no collisions."""
    assert _detect_duplicate_labels_in_passage([]) == []


def test_detect_duplicate_labels_distinct_labels_no_collision() -> None:
    """Distinct labels from the same passage produce no collisions."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Go north"},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "Go south"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_case_insensitive_collision() -> None:
    """Case-insensitive collision within a passage is reported."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Forward"},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "forward"},
    ]
    collisions = _detect_duplicate_labels_in_passage(specs)
    assert len(collisions) == 1
    assert collisions[0]["from_passage"] == "passage::p1"
    assert collisions[0]["label"] == "forward"
    assert collisions[0]["targets"] == ["passage::a", "passage::b"]


def test_detect_duplicate_labels_empty_label_skipped() -> None:
    """A choice spec with empty/missing label is not counted."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": ""},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "Go"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_different_passages_no_collision() -> None:
    """Same label across different passages is fine (R-5.2 is per-passage)."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Continue"},
        {"from_passage": "passage::p2", "to_passage": "passage::b", "label": "Continue"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_multiple_collisions_sorted() -> None:
    """Multiple source passages with collisions return deterministically sorted results."""
    specs = [
        {"from_passage": "passage::p2", "to_passage": "passage::x", "label": "Go"},
        {"from_passage": "passage::p2", "to_passage": "passage::y", "label": "go"},
        {"from_passage": "passage::p1", "to_passage": "passage::z", "label": "Wait"},
        {"from_passage": "passage::p1", "to_passage": "passage::w", "label": "wait"},
    ]
    collisions = _detect_duplicate_labels_in_passage(specs)
    assert [c["from_passage"] for c in collisions] == ["passage::p1", "passage::p2"]


# ---------------------------------------------------------------------------
# Phase 1a Narrative Gap Insertion (migrated from GROW Phase 4b per PR #1366)
# ---------------------------------------------------------------------------


class TestPolishPhase1aNarrativeGaps:
    """POLISH Phase 1a — Narrative Gap Insertion.

    Migrated from ``tests/unit/test_grow_stage.py::TestPhase4bNarrativeGaps``
    when the implementation moved per the structural-vs-narrative migration
    (issue #1368). Behavior is identical; only the host stage changed.
    """

    @pytest.mark.asyncio
    async def test_phase_1a_inserts_gap_beats(self) -> None:
        """Phase 1a inserts gap beats from LLM proposals."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = PolishStage()

        phase_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::mentor_trust_canonical",
                    after_beat="beat::mentor_meet",
                    before_beat="beat::mentor_commits_canonical",
                    summary="Hero reflects on mentor's words",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_1a_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1" in result.detail

        beat_nodes = graph.get_nodes_by_type("beat")
        gap_beats = [bid for bid in beat_nodes if "gap" in bid]
        assert len(gap_beats) == 1

    @pytest.mark.asyncio
    async def test_phase_1a_skips_invalid_path(self) -> None:
        """Phase 1a skips gap proposals with invalid path IDs."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = PolishStage()

        phase_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::nonexistent",
                    after_beat="beat::opening",
                    before_beat="beat::mentor_meet",
                    summary="Invalid path gap",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_1a_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "0" in result.detail

    @pytest.mark.asyncio
    async def test_phase_1a_no_paths(self) -> None:
        """Phase 1a returns skipped when no paths exist."""
        graph = Graph.empty()
        stage = PolishStage()
        mock_model = MagicMock()

        result = await stage._phase_1a_narrative_gaps(graph, mock_model)

        assert result.status == "skipped"
        assert "No paths" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_1a_single_beat_paths_skipped(self) -> None:
        """Phase 1a skips paths with only 1 beat (no sequence to gap-check)."""
        graph = Graph.empty()
        graph.create_node("path::short", {"type": "path", "raw_id": "short"})
        graph.create_node("beat::only", {"type": "beat", "summary": "Lone beat"})
        graph.add_edge("belongs_to", "beat::only", "path::short")

        stage = PolishStage()
        mock_model = MagicMock()

        result = await stage._phase_1a_narrative_gaps(graph, mock_model)

        assert result.status == "skipped"
        assert "No paths with 2+ beats" in result.detail


# ---------------------------------------------------------------------------
# Phase 5e Atmospheric Annotation (migrated from GROW Phase 4d per PR #1370)
# ---------------------------------------------------------------------------


class TestPolishPhase5eAtmospheric:
    """POLISH Phase 5e — Atmospheric Annotation.

    Migrated from GROW Phase 4d when the implementation moved per the
    structural-vs-narrative migration (issue #1368).
    """

    @pytest.mark.asyncio
    async def test_phase_5e_no_beats_returns_skipped(self) -> None:
        """No beats → status='skipped' with no LLM call."""
        graph = Graph.empty()
        stage = PolishStage()
        result = await stage._phase_5e_atmospheric(graph, MagicMock())
        assert result.status == "skipped"
        assert "No beats" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_5e_applies_atmospheric_to_all_beats(self) -> None:
        """Happy path: every beat receives atmospheric_detail."""
        from questfoundry.models.grow import AtmosphericDetail, Phase4dOutput

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for bid in ("beat::a", "beat::b"):
            _make_beat(graph, bid, f"Summary of {bid}")
            graph.add_edge("belongs_to", bid, "path::p1")

        phase_output = Phase4dOutput(
            details=[
                AtmosphericDetail(
                    beat_id="beat::a",
                    atmospheric_detail="dim torchlight; cold stone underfoot",
                ),
                AtmosphericDetail(
                    beat_id="beat::b",
                    atmospheric_detail="howling wind through cracked windows",
                ),
            ]
        )

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return phase_output, 1, 250

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        result = await stage._phase_5e_atmospheric(graph, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        beat_a = graph.get_node("beat::a")
        beat_b = graph.get_node("beat::b")
        assert beat_a is not None and beat_b is not None
        assert beat_a.get("atmospheric_detail") == "dim torchlight; cold stone underfoot"
        assert beat_b.get("atmospheric_detail") == "howling wind through cracked windows"

    @pytest.mark.asyncio
    async def test_phase_5e_partial_coverage_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R-5e.1: partial coverage (LLM details only some beats) emits WARNING."""
        from questfoundry.models.grow import AtmosphericDetail, Phase4dOutput

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for bid in ("beat::a", "beat::b", "beat::c"):
            _make_beat(graph, bid, f"Summary of {bid}")
            graph.add_edge("belongs_to", bid, "path::p1")

        # LLM details only beat::a — the other two should trigger partial-coverage WARNING
        phase_output = Phase4dOutput(
            details=[
                AtmosphericDetail(beat_id="beat::a", atmospheric_detail="cold mist on stone"),
            ]
        )

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return phase_output, 1, 100

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        with caplog.at_level("WARNING"):
            result = await stage._phase_5e_atmospheric(graph, MagicMock())

        assert result.status == "completed"
        assert "1/3" in result.detail
        # Confirm partial-coverage warning was emitted
        assert any("atmospheric_partial_coverage" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_phase_5e_invalid_beat_id_skipped(self) -> None:
        """Invalid beat IDs in LLM output are skipped (logged at INFO)."""
        from questfoundry.models.grow import AtmosphericDetail, Phase4dOutput

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::real", "Real beat")
        graph.add_edge("belongs_to", "beat::real", "path::p1")

        phase_output = Phase4dOutput(
            details=[
                AtmosphericDetail(
                    beat_id="beat::ghost",
                    atmospheric_detail="phantom whispers in the empty corridor",
                ),
                AtmosphericDetail(
                    beat_id="beat::real",
                    atmospheric_detail="real detail of the chamber",
                ),
            ]
        )

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return phase_output, 1, 100

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        result = await stage._phase_5e_atmospheric(graph, MagicMock())

        assert result.status == "completed"
        beat_real = graph.get_node("beat::real")
        assert beat_real is not None
        assert beat_real.get("atmospheric_detail") == "real detail of the chamber"
        assert graph.get_node("beat::ghost") is None  # phantom never created


# ---------------------------------------------------------------------------
# Phase 5f Path Thematic Annotation (migrated from GROW Phase 4e per PR #1370)
# ---------------------------------------------------------------------------


class TestPolishPhase5fPathThematic:
    """POLISH Phase 5f — Path Thematic Annotation."""

    @pytest.mark.asyncio
    async def test_phase_5f_no_paths_returns_skipped(self) -> None:
        """No paths → status='skipped' with no LLM call."""
        graph = Graph.empty()
        stage = PolishStage()
        result = await stage._phase_5f_path_thematic(graph, MagicMock())
        assert result.status == "skipped"
        assert "No paths" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_5f_skips_short_paths(self) -> None:
        """R-5f.1: paths with <2 beats are skipped (no narrative arc to summarize)."""
        graph = Graph.empty()
        graph.create_node("path::short", {"type": "path", "raw_id": "short"})
        _make_beat(graph, "beat::lone", "Only beat")
        graph.add_edge("belongs_to", "beat::lone", "path::short")

        stage = PolishStage()
        # No need to mock LLM — phase short-circuits before any call
        result = await stage._phase_5f_path_thematic(graph, MagicMock())
        assert result.status == "completed"
        assert "0/0" in result.detail  # no paths qualified, none annotated
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_5f_annotates_multi_beat_path(self) -> None:
        """Happy path: a 2+ beat path receives path_theme + path_mood."""
        from questfoundry.models.grow import PathMiniArc

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Beat A")
        _make_beat(graph, "beat::b", "Beat B")
        graph.add_edge("belongs_to", "beat::a", "path::p1")
        graph.add_edge("belongs_to", "beat::b", "path::p1")
        graph.add_edge("predecessor", "beat::b", "beat::a")

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return (
                PathMiniArc(
                    path_id="path::p1",
                    path_theme="trust transforms into reluctant alliance",
                    path_mood="cautious-then-warming",
                ),
                1,
                100,
            )

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        result = await stage._phase_5f_path_thematic(graph, MagicMock())

        assert result.status == "completed"
        assert "1/1" in result.detail
        path = graph.get_node("path::p1")
        assert path is not None
        assert path.get("path_theme") == "trust transforms into reluctant alliance"
        assert path.get("path_mood") == "cautious-then-warming"

    @pytest.mark.asyncio
    async def test_phase_5f_per_path_llm_failure_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R-5f.3: per-path LLM failure logs WARNING and leaves fields unpopulated."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Beat A")
        _make_beat(graph, "beat::b", "Beat B")
        graph.add_edge("belongs_to", "beat::a", "path::p1")
        graph.add_edge("belongs_to", "beat::b", "path::p1")
        graph.add_edge("predecessor", "beat::b", "beat::a")

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            raise RuntimeError("LLM unavailable")

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        with caplog.at_level("WARNING"):
            result = await stage._phase_5f_path_thematic(graph, MagicMock())

        assert result.status == "completed"
        assert "0/1" in result.detail  # 0 annotated of 1 qualifying path
        # Path fields stay unpopulated
        path = graph.get_node("path::p1")
        assert path is not None
        assert path.get("path_theme") is None
        assert path.get("path_mood") is None
        # Per-path failure WARNING was emitted
        assert any("path_thematic_llm_failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_phase_5f_includes_atmospheric_in_context(self) -> None:
        """Per CLAUDE.md §Context Enrichment, the 5f LLM call should receive
        atmospheric_detail (populated by the preceding 5e phase) for each beat."""
        from questfoundry.models.grow import PathMiniArc

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "Beat A", atmospheric_detail="dim torchlight on stone")
        _make_beat(graph, "beat::b", "Beat B", atmospheric_detail="howling wind in the hall")
        graph.add_edge("belongs_to", "beat::a", "path::p1")
        graph.add_edge("belongs_to", "beat::b", "path::p1")
        graph.add_edge("predecessor", "beat::b", "beat::a")

        captured_contexts: list[dict] = []

        async def mock_call(
            *,
            model: object,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
        ) -> tuple:
            captured_contexts.append(context)
            return (
                PathMiniArc(
                    path_id="path::p1",
                    path_theme="resilience grows from torchlit isolation",
                    path_mood="hushed-yet-resolute",
                ),
                1,
                100,
            )

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        await stage._phase_5f_path_thematic(graph, MagicMock())

        assert len(captured_contexts) == 1
        beat_seq = captured_contexts[0]["beat_sequence"]
        # atmospheric_detail must appear in the beat sequence for each beat
        assert "dim torchlight on stone" in beat_seq
        assert "howling wind in the hall" in beat_seq


# ---------------------------------------------------------------------------
# Phase 2 R-2.7 gap-beat tagging (PR C of #1368)
# ---------------------------------------------------------------------------


class TestPolishPhase2GapBeatTagging:
    """Spec R-2.7: only correction beats (3+ consecutive-run breakers) carry
    ``is_gap_beat: True``; regular pacing-flag micro-beats do not.

    The flag distinguishes their origin so FILL renders the two kinds
    differently — a correction beat is a structural pacing intrusion
    (rendered as ``[gap]``), while a pacing-flag micro-beat carries actual
    narrative content that should appear in prose.
    """

    @pytest.mark.asyncio
    async def test_correction_beat_for_consecutive_run_carries_is_gap_beat_true(self) -> None:
        """Beat inserted after a consecutive-run flag → is_gap_beat=True."""
        from questfoundry.models.polish import MicroBeatProposal, Phase2Output

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        # 3 consecutive scenes — triggers consecutive_scene flag
        for i in range(1, 4):
            _make_beat(graph, f"beat::s{i}", f"Scene {i}", scene_type="scene")
            graph.add_edge("belongs_to", f"beat::s{i}", "path::p1")
        for i in range(2, 4):
            graph.add_edge("predecessor", f"beat::s{i}", f"beat::s{i - 1}")

        # LLM proposes a sequel after beat::s2 (the middle of the run)
        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return (
                Phase2Output(
                    micro_beats=[
                        MicroBeatProposal(
                            after_beat_id="beat::s2",
                            summary="Quiet pause to break the action chain",
                            entity_ids=[],
                        ),
                    ]
                ),
                1,
                100,
            )

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        result = await stage._phase_2_pacing(graph, MagicMock())

        assert result.status == "completed"
        # Find the inserted micro-beat and assert R-2.7 fields
        beats = graph.get_nodes_by_type("beat")
        micro_beats = [b for bid, b in beats.items() if "micro_" in bid]
        assert len(micro_beats) == 1
        mb = micro_beats[0]
        assert mb.get("is_gap_beat") is True, "consecutive-run correction must carry is_gap_beat"
        assert mb.get("role") == "micro_beat"
        assert mb.get("dilemma_impacts") == []
        assert mb.get("created_by") == "POLISH"

    @pytest.mark.asyncio
    async def test_micro_beat_for_post_commit_no_sequel_does_not_carry_is_gap_beat(self) -> None:
        """Beat inserted for the ``no_sequel_after_commit`` flag must NOT
        carry is_gap_beat (R-2.7 distinguishes correction vs pacing-flag origin)."""
        from questfoundry.models.polish import MicroBeatProposal, Phase2Output

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        # Setup: a commit beat with no sequel after — triggers no_sequel_after_commit
        _make_beat(
            graph,
            "beat::commit",
            "Commit beat",
            scene_type="scene",
            dilemma_impacts=[{"effect": "commits", "dilemma_id": "d1"}],
        )
        graph.add_edge("belongs_to", "beat::commit", "path::p1")
        # Followed directly by another scene (no sequel)
        _make_beat(graph, "beat::after", "After commit", scene_type="scene")
        graph.add_edge("belongs_to", "beat::after", "path::p1")
        graph.add_edge("predecessor", "beat::after", "beat::commit")

        async def mock_call(*_args: object, **_kwargs: object) -> tuple:
            return (
                Phase2Output(
                    micro_beats=[
                        MicroBeatProposal(
                            after_beat_id="beat::commit",
                            summary="Brief reflection on the choice just made",
                            entity_ids=[],
                        ),
                    ]
                ),
                1,
                100,
            )

        stage = PolishStage()
        stage._polish_llm_call = mock_call  # type: ignore[method-assign]
        result = await stage._phase_2_pacing(graph, MagicMock())

        assert result.status == "completed"
        beats = graph.get_nodes_by_type("beat")
        micro_beats = [b for bid, b in beats.items() if "micro_" in bid]
        assert len(micro_beats) == 1
        mb = micro_beats[0]
        # R-2.7: pacing-flag micro-beats (not consecutive-run correction) must
        # NOT carry is_gap_beat — FILL needs the summary to render normally.
        assert mb.get("is_gap_beat") is None or mb.get("is_gap_beat") is False, (
            "no_sequel_after_commit micro-beat must NOT carry is_gap_beat"
        )
        assert mb.get("role") == "micro_beat"
