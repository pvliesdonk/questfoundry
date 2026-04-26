"""Tests for FILL stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from questfoundry.graph.graph import Graph
from questfoundry.models.fill import (
    EntityUpdate,
    FillExtractOutput,
    FillPassageOutput,
    FillPhase0Output,
    FillPhase1Output,
    FillPhase2Output,
    FillPhaseResult,
    ReviewFlag,
    VoiceDocument,
)
from questfoundry.pipeline.gates import AutoApprovePhaseGate, PhaseGateHook
from questfoundry.pipeline.stages.fill import (
    FillStage,
    FillStageError,
    create_fill_stage,
    fill_stage,
)


@pytest.fixture
def grow_graph(tmp_path: Path) -> Graph:
    """Create a minimal GROW-completed graph."""
    g = Graph.empty()
    g.set_last_stage("polish")

    # Minimal passage for counting
    g.create_node(
        "passage::p1",
        {"type": "passage", "raw_id": "p1", "from_beat": "", "summary": "test"},
    )

    g.save(tmp_path / "graph.db")
    return g


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock LangChain model."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _bypass_seam_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass FILL's new POLISH-output entry validator (#1347) and the
    FILL-output exit validator (#1348) for all tests in this file. The
    test fixtures use minimal POLISH graphs that don't satisfy the full
    contracts on either side; the contract-chaining integration is
    exercised in test_contract_chaining.py instead.
    """
    from questfoundry.graph import (
        fill_output_validation as _fov,
    )
    from questfoundry.graph import (
        polish_validation as _pv,
    )

    monkeypatch.setattr(_pv, "validate_polish_output", lambda _g: [])
    monkeypatch.setattr(_fov, "validate_fill_output", lambda _g: [])


class TestFillStageInit:
    def test_default_gate(self) -> None:
        stage = FillStage()
        assert isinstance(stage.gate, AutoApprovePhaseGate)

    def test_custom_gate(self) -> None:
        gate = MagicMock(spec=PhaseGateHook)
        stage = FillStage(gate=gate)
        assert stage.gate is gate

    def test_name(self) -> None:
        assert FillStage.name == "fill"

    def test_project_path(self) -> None:
        stage = FillStage(project_path=Path("/tmp/test"))
        assert stage.project_path == Path("/tmp/test")


def _mock_implemented_phases(stage: FillStage) -> None:
    """Replace implemented phases with mocks for execute-level testing."""

    async def _fake_phase_0(
        graph: Graph,
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        graph.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                "pov": "third_person_limited",
                "tense": "past",
                "voice_register": "literary",
                "sentence_rhythm": "varied",
                "tone_words": ["atmospheric"],
            },
        )
        return FillPhaseResult(phase="voice", status="completed", llm_calls=1, tokens_used=500)

    async def _fake_phase_1a(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="expand", status="completed", llm_calls=1, tokens_used=300)

    async def _fake_phase_1(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="generate", status="completed", llm_calls=2, tokens_used=1000)

    async def _fake_phase_1c(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="quality_gate", status="completed", llm_calls=0, tokens_used=0)

    async def _fake_phase_2(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="review", status="completed", llm_calls=1, tokens_used=200)

    async def _fake_phase_3(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="revision", status="completed", llm_calls=0, tokens_used=0)

    async def _fake_phase_2_cycle2(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(
            phase="review_cycle2", status="completed", llm_calls=0, tokens_used=0
        )

    async def _fake_phase_3_final(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(
            phase="revision_final", status="completed", llm_calls=0, tokens_used=0
        )

    async def _fake_phase_4(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(
            phase="arc_validation", status="completed", llm_calls=0, tokens_used=0
        )

    stage._phase_0_voice = _fake_phase_0  # type: ignore[method-assign]
    stage._phase_1a_expand = _fake_phase_1a  # type: ignore[method-assign]
    stage._phase_1_generate = _fake_phase_1  # type: ignore[method-assign]
    stage._phase_1c_mechanical_gate = _fake_phase_1c  # type: ignore[method-assign]
    stage._phase_2_review = _fake_phase_2  # type: ignore[method-assign]
    stage._phase_3_revision = _fake_phase_3  # type: ignore[method-assign]
    stage._phase_2_review_cycle2 = _fake_phase_2_cycle2  # type: ignore[method-assign]
    stage._phase_3_revision_final = _fake_phase_3_final  # type: ignore[method-assign]
    stage._phase_4_arc_validation = _fake_phase_4  # type: ignore[method-assign]


class TestFillStageExecute:
    @pytest.mark.asyncio
    async def test_requires_project_path(self, mock_model: MagicMock) -> None:
        stage = FillStage()
        with pytest.raises(FillStageError, match="project_path is required"):
            await stage.execute(mock_model, "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("last_stage", ["dream", "brainstorm", "seed"])
    async def test_rejects_pre_grow_stages(
        self, mock_model: MagicMock, tmp_path: Path, last_stage: str
    ) -> None:
        g = Graph.empty()
        g.set_last_stage(last_stage)
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="FILL requires completed POLISH"):
            await stage.execute(mock_model, "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("last_stage", ["polish", "fill", "dress", "ship"])
    async def test_accepts_grow_and_later_stages(
        self, mock_model: MagicMock, tmp_path: Path, last_stage: str
    ) -> None:
        """FILL should accept re-runs when last_stage is polish or any later stage."""
        # Create a POLISH-completed graph as the pre-FILL snapshot
        pre_fill = Graph.empty()
        pre_fill.set_last_stage("polish")
        pre_fill.create_node("arc::spine", {"type": "arc", "raw_id": "spine", "arc_type": "spine"})

        # Save the pre-FILL checkpoint (needed for re-runs from fill/dress/ship)
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        pre_fill.save(snapshot_dir / "pre-fill.db")

        # The main graph has progressed past GROW
        g = Graph.empty()
        g.set_last_stage(last_stage)
        g.create_node(
            "voice::voice",
            {"type": "voice", "raw_id": "voice", "pov": "third_person_limited", "tense": "past"},
        )
        g.create_node("arc::spine", {"type": "arc", "raw_id": "spine", "arc_type": "spine"})
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        # Should not raise FillStageError about POLISH requirement
        # (will proceed to phases — we just verify the gate passes)
        try:
            await stage.execute(mock_model, "")
        except FillStageError as e:
            assert "FILL requires completed POLISH" not in str(e)

    @pytest.mark.asyncio
    async def test_rerun_rewinds_fill_mutations(
        self, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        """Re-running FILL should rewind fill mutations, removing stale nodes."""
        # Main graph: simulate GROW completed, then FILL ran and created voice node
        g = Graph.empty()
        g.set_last_stage("polish")
        g.create_node("arc::spine", {"type": "arc", "raw_id": "spine", "arc_type": "spine"})
        g.save(tmp_path / "graph.db")

        # Reload as SQLite-backed (auto-migration)
        g = Graph.load(tmp_path)

        # Simulate a previous FILL run by creating the voice node within fill mutation context
        with g.mutation_context(stage="fill", phase="voice"):
            g.create_node(
                "voice::voice",
                {"type": "voice", "raw_id": "voice", "pov": "first_person", "tense": "present"},
            )
        g.set_last_stage("fill")
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        await stage.execute(mock_model, "", project_path=tmp_path)

        # The voice node from the stale fill run should have been rewound
        # (snapshot restored before phases run)
        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "fill"

    @pytest.mark.asyncio
    async def test_runs_all_phases(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        result_dict, llm_calls, tokens = await stage.execute(mock_model, "")

        # Result is passage prose coverage stats (artifact extraction removed)
        assert "total_passages" in result_dict
        assert "passages_with_prose" in result_dict

        # Sum of all phase LLM calls (1 + 1 + 2 + 0 + 1 + 0 + 0 = 5)
        assert llm_calls == 5
        assert tokens == 2000

    @pytest.mark.asyncio
    async def test_sets_last_stage(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        await stage.execute(mock_model, "")

        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "fill"

    @pytest.mark.asyncio
    async def test_resume_from_phase(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)

        # First run to create checkpoints
        await stage.execute(mock_model, "")

        # Resume from review phase — only review + revision run
        result_dict, llm_calls, _ = await stage.execute(
            mock_model, "", resume_from="review", project_path=tmp_path
        )

        # Should return passage coverage stats
        assert "total_passages" in result_dict

        # Only review (1 call) + revision (0 calls) = 1 LLM call
        assert llm_calls == 1

    @pytest.mark.asyncio
    async def test_resume_invalid_phase(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="Unknown phase"):
            await stage.execute(mock_model, "", resume_from="nonexistent", project_path=tmp_path)

    @pytest.mark.asyncio
    async def test_gate_reject_rolls_back(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        gate = MagicMock()
        gate.on_phase_complete = AsyncMock(return_value="reject")

        stage = FillStage(project_path=tmp_path, gate=gate)
        _mock_implemented_phases(stage)
        _, llm_calls, _ = await stage.execute(mock_model, "")

        # Should stop after first phase (voice) is rejected — only 1 LLM call
        assert llm_calls == 1

        # Verify rollback was persisted — last_stage should remain "polish"
        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "polish"

    @pytest.mark.asyncio
    async def test_phase_failure_stops_execution(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=tmp_path)
        stage._phase_0_voice = AsyncMock(  # type: ignore[method-assign]
            return_value=FillPhaseResult(phase="voice", status="failed", detail="test error")
        )
        _, llm_calls, _ = await stage.execute(mock_model, "")

        # Should stop after voice phase fails — 0 LLM calls (failed immediately)
        assert llm_calls == 0

        # last_stage should remain "polish" (not promoted to "fill")
        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "polish"

    @pytest.mark.asyncio
    async def test_progress_callback(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        progress_calls: list[tuple[str, str, str | None]] = []

        def on_progress(phase: str, status: str, detail: str | None) -> None:
            progress_calls.append((phase, status, detail))

        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        await stage.execute(mock_model, "", on_phase_progress=on_progress)

        assert len(progress_calls) == 9
        assert progress_calls[0][0] == "voice"
        assert progress_calls[1][0] == "expand"

    @pytest.mark.asyncio
    async def test_project_path_override(
        self,
        mock_model: MagicMock,
        grow_graph: Graph,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        stage = FillStage(project_path=Path("/nonexistent"))
        _mock_implemented_phases(stage)
        # Override with tmp_path
        result_dict, _, _ = await stage.execute(mock_model, "", project_path=tmp_path)
        assert "total_passages" in result_dict


class TestPhaseOrder:
    def test_nine_phases(self) -> None:
        stage = FillStage()
        phases = stage._phase_order()
        assert len(phases) == 9

    def test_phase_names(self) -> None:
        stage = FillStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "voice",
            "expand",
            "generate",
            "quality_gate",
            "review",
            "revision",
            "review_cycle2",
            "revision_final",
            "arc_validation",
        ]


def _make_voice_output() -> FillPhase0Output:
    """Create a valid FillPhase0Output for mocking."""
    return FillPhase0Output(
        voice=VoiceDocument(
            pov="third_person_limited",
            pov_character="kay",
            tense="past",
            voice_register="literary",
            sentence_rhythm="varied",
            tone_words=["atmospheric", "tense"],
            avoid_words=["suddenly"],
            avoid_patterns=["adverb-heavy dialogue tags"],
        ),
        story_title="The Hollow Crown",
    )


class TestPhase0Voice:
    @pytest.mark.asyncio
    async def test_creates_voice_node(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dream::vision",
            {"type": "dream", "raw_id": "vision", "genre": "dark fantasy"},
        )
        stage = FillStage()

        with (
            patch.object(
                stage,
                "_phase_0a_voice_research",
                new_callable=AsyncMock,
                return_value=("", 0, 0),
            ),
            patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(_make_voice_output(), 1, 500),
            ),
        ):
            result = await stage._phase_0_voice(graph, MagicMock())

        assert result.phase == "voice"
        assert result.status == "completed"
        assert result.llm_calls == 1
        assert result.tokens_used == 500

        # Voice node should be created in graph with story_title
        voice_node = graph.get_node("voice::voice")
        assert voice_node is not None
        assert voice_node["pov"] == "third_person_limited"
        assert voice_node["tense"] == "past"
        assert voice_node["voice_register"] == "literary"
        assert voice_node["story_title"] == "The Hollow Crown"

        # No phantom vision::main node should exist
        assert graph.get_node("vision::main") is None

    @pytest.mark.asyncio
    async def test_passes_context_to_llm(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dream::vision",
            {"type": "dream", "raw_id": "vision", "genre": "dark fantasy"},
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "test", "scene_type": "scene"},
        )
        stage = FillStage()

        mock_llm_call = AsyncMock(return_value=(_make_voice_output(), 1, 500))
        with (
            patch.object(
                stage,
                "_phase_0a_voice_research",
                new_callable=AsyncMock,
                return_value=("Use third_person_limited past for fantasy.", 2, 300),
            ),
            patch.object(stage, "_fill_llm_call", mock_llm_call),
        ):
            await stage._phase_0_voice(graph, MagicMock())

        # Verify context was passed with expected keys
        call_args = mock_llm_call.call_args
        context = call_args[0][2]  # Third positional arg
        assert "dream_vision" in context
        assert "grow_summary" in context
        assert "scene_types_summary" in context
        assert "research_notes" in context
        assert "Use third_person_limited past for fantasy." in context["research_notes"]

    @pytest.mark.asyncio
    async def test_includes_research_metrics(self) -> None:
        """Voice phase includes research LLM calls and tokens in totals."""
        graph = Graph.empty()
        graph.create_node(
            "dream::vision",
            {"type": "dream", "raw_id": "vision", "genre": "dark fantasy"},
        )
        stage = FillStage()

        with (
            patch.object(
                stage,
                "_phase_0a_voice_research",
                new_callable=AsyncMock,
                return_value=("Research notes here.", 3, 800),
            ),
            patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(_make_voice_output(), 1, 500),
            ),
        ):
            result = await stage._phase_0_voice(graph, MagicMock())

        assert result.llm_calls == 4  # 3 research + 1 voice
        assert result.tokens_used == 1300  # 800 research + 500 voice

    @pytest.mark.asyncio
    async def test_research_failure_graceful(self) -> None:
        """Voice phase continues when research fails."""
        graph = Graph.empty()
        graph.create_node(
            "dream::vision",
            {"type": "dream", "raw_id": "vision", "genre": "dark fantasy"},
        )
        stage = FillStage()

        with (
            patch.object(
                stage,
                "_phase_0a_voice_research",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Corpus unavailable"),
            ),
            patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(_make_voice_output(), 1, 500),
            ),
        ):
            result = await stage._phase_0_voice(graph, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert result.tokens_used == 500

    @pytest.mark.asyncio
    async def test_llm_failure_raises(self) -> None:
        graph = Graph.empty()
        stage = FillStage()

        with (
            patch.object(
                stage,
                "_phase_0a_voice_research",
                new_callable=AsyncMock,
                return_value=("", 0, 0),
            ),
            patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                side_effect=FillStageError("LLM call failed"),
            ),
            pytest.raises(FillStageError, match="LLM call failed"),
        ):
            await stage._phase_0_voice(graph, MagicMock())


class TestPhase0aVoiceResearch:
    """Tests for the voice research sub-phase."""

    @pytest.mark.asyncio
    async def test_returns_summary_when_tools_available(self) -> None:
        """Research returns brief from discuss + summarize when corpus tools exist."""
        graph = Graph.empty()
        graph.create_node(
            "dream::vision",
            {"type": "dream", "raw_id": "vision", "genre": "dark fantasy"},
        )
        stage = FillStage()

        mock_tools = [MagicMock(), MagicMock()]
        mock_messages = [MagicMock()]

        with (
            patch(
                "questfoundry.tools.langchain_tools.get_corpus_tools",
                return_value=mock_tools,
            ),
            patch(
                "questfoundry.agents.run_discuss_phase",
                new_callable=AsyncMock,
                return_value=(mock_messages, 3, 600),
            ),
            patch(
                "questfoundry.agents.summarize_discussion",
                new_callable=AsyncMock,
                return_value=("Use third_person_limited past for dark fantasy.", 200),
            ),
        ):
            brief, calls, tokens = await stage._phase_0a_voice_research(graph, MagicMock())

        assert brief == "Use third_person_limited past for dark fantasy."
        assert calls == 4  # 3 discuss + 1 summarize
        assert tokens == 800  # 600 + 200

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_tools(self) -> None:
        """Research returns empty string when corpus tools unavailable."""
        graph = Graph.empty()
        stage = FillStage()

        with patch(
            "questfoundry.tools.langchain_tools.get_corpus_tools",
            return_value=[],
        ):
            brief, calls, tokens = await stage._phase_0a_voice_research(graph, MagicMock())

        assert brief == ""
        assert calls == 0
        assert tokens == 0


def _make_prose_graph() -> Graph:
    """Create a graph with voice node and passages for Phase 1 testing."""
    g = Graph.empty()
    g.create_node(
        "voice::voice",
        {
            "type": "voice",
            "raw_id": "voice",
            "pov": "third_person_limited",
            "tense": "past",
            "voice_register": "literary",
            # R-1.7 stamp — Phase 1a's entry assertion requires this. Tests
            # call Phase 1a directly; in real runs, the stamp is applied by
            # execute() after the Phase 0 gate decision.
            "approved_at": "2026-04-23T12:00:00+00:00",
            "approval_mode": "no_interactive",
        },
    )
    g.create_node(
        "beat::b1",
        {"type": "beat", "raw_id": "b1", "summary": "Kay enters", "scene_type": "scene"},
    )
    g.create_node(
        "beat::b2",
        {"type": "beat", "raw_id": "b2", "summary": "Kay explores", "scene_type": "sequel"},
    )
    g.create_node(
        "passage::p1",
        {
            "type": "passage",
            "raw_id": "p1",
            "from_beat": "beat::b1",
            "summary": "Kay enters the tower",
        },
    )
    g.create_node(
        "passage::p2",
        {
            "type": "passage",
            "raw_id": "p2",
            "from_beat": "beat::b2",
            "summary": "Kay explores the hall",
        },
    )
    g.add_edge("passage_from", "passage::p1", "beat::b1")
    g.add_edge("passage_from", "passage::p2", "beat::b2")

    # Dilemma + path structure for computed arcs
    g.create_node(
        "dilemma::d1",
        {"type": "dilemma", "raw_id": "d1"},
    )
    g.create_node(
        "path::main",
        {
            "type": "path",
            "raw_id": "main",
            "dilemma_id": "dilemma::d1",
            "is_canonical": True,
        },
    )
    g.add_edge("belongs_to", "beat::b1", "path::main")
    g.add_edge("belongs_to", "beat::b2", "path::main")
    g.add_edge("predecessor", "beat::b2", "beat::b1")
    g.add_edge("grouped_in", "beat::b1", "passage::p1")
    g.add_edge("grouped_in", "beat::b2", "passage::p2")
    return g


def _make_passage_output(passage_id: str, prose: str = "Generated prose.") -> FillPhase1Output:
    return FillPhase1Output(passage=FillPassageOutput(passage_id=passage_id, prose=prose))


class TestPhase1Generate:
    @pytest.mark.asyncio
    async def test_generates_prose_for_passages(self) -> None:
        graph = _make_prose_graph()
        stage = FillStage()

        call_count = 0

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal call_count
            call_count += 1
            pid = context["passage_id"]
            return _make_passage_output(pid, f"Prose for {pid}."), 1, 100

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert result.phase == "generate"
        assert result.status == "completed"
        assert result.llm_calls == 2
        assert call_count == 2

        # Check prose was stored in graph
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Prose for p1."

    @pytest.mark.asyncio
    async def test_applies_entity_updates(self) -> None:
        graph = _make_prose_graph()
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "concept": "A wanderer"},
        )
        stage = FillStage()

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            pid = context["passage_id"]
            if pid == "p1":
                return (
                    FillPhase1Output(
                        passage=FillPassageOutput(
                            passage_id=pid,
                            prose="Kay's scarred hands gripped the rail.",
                            entity_updates=[
                                EntityUpdate(
                                    entity_id="kay",
                                    field="appearance",
                                    value="scarred hands",
                                )
                            ],
                        )
                    ),
                    1,
                    100,
                )
            return _make_passage_output(pid), 1, 100

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_1_generate(graph, MagicMock())

        entity = graph.get_node("entity::kay")
        assert entity is not None
        assert entity.get("appearance") == "scarred hands"

    @pytest.mark.asyncio
    async def test_no_passages_returns_completed(self) -> None:
        graph = Graph.empty()
        stage = FillStage()
        result = await stage._phase_1_generate(graph, MagicMock())
        assert result.status == "completed"
        assert result.llm_calls == 0


class TestGenerationOrder:
    def test_spine_first(self) -> None:
        graph = _make_prose_graph()
        stage = FillStage()
        order = stage._get_generation_order(graph)
        assert len(order) == 2
        assert order[0] == ("passage::p1", "main")
        assert order[1] == ("passage::p2", "main")

    def test_skips_already_filled(self) -> None:
        graph = _make_prose_graph()
        # Add a non-canonical path to create a branch arc
        graph.create_node(
            "path::alt",
            {
                "type": "path",
                "raw_id": "alt",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )
        # Both beats also belong to the alt path (shared beats)
        graph.add_edge("belongs_to", "beat::b1", "path::alt")
        graph.add_edge("belongs_to", "beat::b2", "path::alt")
        # Mark p1 as already having prose
        graph.update_node("passage::p1", prose="Already filled.")
        stage = FillStage()
        order = stage._get_generation_order(graph)

        # Shared passages appear exactly once — deduped by seen set
        # p1 filled from spine, p2 unfilled from spine, both skipped in branch
        passage_ids = [pid for pid, _ in order]
        assert passage_ids.count("passage::p1") == 1
        assert passage_ids.count("passage::p2") == 1
        assert len(order) == 2


class TestPhase1aExpand:
    @pytest.mark.asyncio
    async def test_creates_blueprints_on_passages(self) -> None:
        graph = _make_prose_graph()
        stage = FillStage()

        from questfoundry.models.fill import BatchedExpandOutput, ExpandBlueprint

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return (
                BatchedExpandOutput(
                    blueprints=[
                        ExpandBlueprint(
                            passage_id="p1",
                            sensory_palette=["sight: torchlight", "sound: drip", "smell: damp"],
                            opening_move="sensory_image",
                            emotional_arc_word="dread",
                        ),
                        ExpandBlueprint(
                            passage_id="p2",
                            sensory_palette=["sight: dust", "sound: echo", "touch: cold stone"],
                            opening_move="action",
                            emotional_arc_word="resolve",
                        ),
                    ]
                ),
                1,
                400,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_1a_expand(graph, MagicMock())

        assert result.phase == "expand"
        assert result.status == "completed"
        assert result.llm_calls == 1

        # Blueprints should be stored on passage nodes
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1.get("blueprint") is not None
        assert p1["blueprint"]["opening_move"] == "sensory_image"

        p2 = graph.get_node("passage::p2")
        assert p2 is not None
        assert p2.get("blueprint") is not None
        assert p2["blueprint"]["emotional_arc_word"] == "resolve"

    @pytest.mark.asyncio
    async def test_no_passages_returns_completed(self) -> None:
        graph = Graph.empty()
        # R-1.7 stamp required by Phase 1a entry assertion.
        graph.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                "approved_at": "2026-04-23T12:00:00+00:00",
                "approval_mode": "no_interactive",
            },
        )
        stage = FillStage()
        result = await stage._phase_1a_expand(graph, MagicMock())
        assert result.status == "completed"
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_blueprint_context_in_generate(self) -> None:
        """Phase 1 generate includes blueprint_context when blueprint exists."""
        graph = _make_prose_graph()
        # Add a blueprint to p1
        graph.update_node(
            "passage::p1",
            blueprint={
                "passage_id": "p1",
                "sensory_palette": ["sight: torchlight", "sound: drip", "smell: damp"],
                "character_gestures": [],
                "opening_move": "sensory_image",
                "craft_constraint": "",
                "emotional_arc_word": "dread",
            },
        )
        stage = FillStage()

        captured_contexts: list[dict] = []

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            captured_contexts.append(context)
            pid = context["passage_id"]
            return _make_passage_output(pid, f"Prose for {pid}."), 1, 100

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_1_generate(graph, MagicMock())

        # p1 has blueprint — should have blueprint_context, empty atmospheric_detail
        p1_ctx = next(c for c in captured_contexts if c["passage_id"] == "p1")
        assert "sensory_image" in p1_ctx["blueprint_context"]
        assert p1_ctx["atmospheric_detail"] == ""

        # p2 has no blueprint — should have fallback blueprint_context
        p2_ctx = next(c for c in captured_contexts if c["passage_id"] == "p2")
        assert "no blueprint available" in p2_ctx["blueprint_context"]


def _make_reviewed_graph() -> Graph:
    """Create a graph with filled passages for Phase 2/3 testing."""
    g = _make_prose_graph()
    g.update_node("passage::p1", prose="Kay entered the tower.")
    g.update_node("passage::p2", prose="The hall stretched before Kay.")
    return g


class TestPhase2Review:
    @pytest.mark.asyncio
    async def test_review_finds_flags(self) -> None:
        graph = _make_reviewed_graph()
        stage = FillStage()

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return (
                FillPhase2Output(
                    flags=[
                        ReviewFlag(
                            passage_id="p1",
                            issue="Voice drift detected",
                            issue_type="voice_drift",
                        )
                    ]
                ),
                1,
                200,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_2_review(graph, MagicMock())

        assert result.phase == "review"
        assert result.status == "completed"
        assert "1 issues" in (result.detail or "")

        # Flag should be stored on passage node
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert len(p1.get("review_flags", [])) == 1

    @pytest.mark.asyncio
    async def test_no_passages_to_review(self) -> None:
        graph = Graph.empty()
        stage = FillStage()
        result = await stage._phase_2_review(graph, MagicMock())
        assert result.status == "completed"
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_empty_flags_is_fine(self) -> None:
        graph = _make_reviewed_graph()
        stage = FillStage()

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return FillPhase2Output(flags=[]), 1, 200

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_2_review(graph, MagicMock())

        assert result.status == "completed"
        assert "0 issues" in (result.detail or "")


class TestPhase3Revision:
    @pytest.mark.asyncio
    async def test_revises_flagged_passage(self) -> None:
        graph = _make_reviewed_graph()
        # Add review flags manually
        graph.update_node(
            "passage::p1",
            review_flags=[
                {"passage_id": "p1", "issue": "Voice drift", "issue_type": "voice_drift"}
            ],
        )
        stage = FillStage()

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(passage_id="p1", prose="Revised prose for Kay.")
                ),
                1,
                300,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_3_revision(graph, MagicMock())

        assert result.phase == "revision"
        assert result.status == "completed"
        assert "1 of 1 flags addressed" in (result.detail or "")

        # Prose should be updated
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Revised prose for Kay."

        # Review flags should be cleared
        assert p1.get("review_flags") == []

    @pytest.mark.asyncio
    async def test_no_flags_returns_completed(self) -> None:
        graph = _make_reviewed_graph()
        stage = FillStage()
        result = await stage._phase_3_revision(graph, MagicMock())
        assert result.status == "completed"
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_revision_context_and_hints_carry_valid_entity_ids(self) -> None:
        """The revision call MUST inject `valid_entity_ids` into the prompt
        context AND pass it as `extra_repair_hints` so the constraint survives
        context drift on retry. Closes the FILL §fill_phase3_revision audit
        finding (CLAUDE.md §6 Valid ID Injection)."""
        graph = _make_reviewed_graph()
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "concept": "A wanderer"},
        )
        graph.update_node(
            "passage::p1",
            review_flags=[
                {"passage_id": "p1", "issue": "Voice drift", "issue_type": "voice_drift"}
            ],
        )
        stage = FillStage()

        captured_context: dict[str, Any] = {}
        captured_hints: list[str] | None = None

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            *,
            creative: bool = False,  # noqa: ARG001
            extra_repair_hints: list[str] | None = None,
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal captured_context, captured_hints
            captured_context = context
            captured_hints = extra_repair_hints
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(passage_id="p1", prose="Revised prose.")
                ),
                1,
                300,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_3_revision(graph, MagicMock())

        # Context contains the prompt-injection variable used by
        # fill_phase3_revision.yaml's `## Valid Entity IDs` section.
        assert "valid_entity_ids" in captured_context
        valid_ids_text = captured_context["valid_entity_ids"]
        assert "kay" in valid_ids_text  # raw_id of the entity in _make_reviewed_graph
        assert "`kay`" in valid_ids_text  # backtick-wrapped per CLAUDE.md §9 rule 1

        # Same constraint also flows as a retry hint so it survives context drift.
        assert captured_hints is not None
        assert any("Valid entity IDs" in h and "`kay`" in h for h in captured_hints)

    @pytest.mark.asyncio
    async def test_multiple_flags_batched(self) -> None:
        """Multiple flags on same passage should be batched into one LLM call."""
        graph = _make_reviewed_graph()
        graph.update_node(
            "passage::p1",
            review_flags=[
                {"passage_id": "p1", "issue": "Voice drift", "issue_type": "voice_drift"},
                {"passage_id": "p1", "issue": "Flat prose", "issue_type": "flat_prose"},
            ],
        )
        stage = FillStage()

        call_count = 0
        captured_issues: str = ""

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal call_count, captured_issues
            call_count += 1
            captured_issues = context.get("issues_list", "")
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(
                        passage_id="p1", prose="Revised prose addressing all issues."
                    )
                ),
                1,
                300,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_3_revision(graph, MagicMock())

        assert result.status == "completed"
        # Both flags should be addressed
        assert "2 of 2 flags addressed" in (result.detail or "")

        # LLM called once (batched) not twice (per-flag)
        assert call_count == 1

        # issues_list should contain both flags
        assert "voice_drift" in captured_issues
        assert "flat_prose" in captured_issues

        # Prose updated
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Revised prose addressing all issues."

        # Flags should be cleared
        assert p1.get("review_flags") == []

    @pytest.mark.asyncio
    async def test_partial_revision_failure(self) -> None:
        """Failed revision (empty prose) should not clear flags for that passage."""
        graph = _make_reviewed_graph()
        graph.update_node(
            "passage::p1",
            review_flags=[
                {"passage_id": "p1", "issue": "Voice drift", "issue_type": "voice_drift"}
            ],
        )
        graph.update_node(
            "passage::p2",
            review_flags=[{"passage_id": "p2", "issue": "Pacing issue", "issue_type": "pacing"}],
        )
        stage = FillStage()

        call_count = 0

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal call_count
            call_count += 1
            pid = context.get("passage_id", "")
            if pid == "p1":
                # p1 revision succeeds
                return (
                    FillPhase1Output(
                        passage=FillPassageOutput(passage_id="p1", prose="Fixed prose.")
                    ),
                    1,
                    300,
                )
            # p2 revision fails (empty prose)
            return (
                FillPhase1Output(passage=FillPassageOutput(passage_id="p2", prose="")),
                1,
                300,
            )

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_3_revision(graph, MagicMock())

        assert result.status == "completed"
        # Only 1 of 2 flags addressed
        assert "1 of 2 flags addressed" in (result.detail or "")

        # p1 revised and flags cleared
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Fixed prose."
        assert p1.get("review_flags") == []

        # p2 unchanged with flags still present
        p2 = graph.get_node("passage::p2")
        assert p2 is not None
        assert p2["prose"] == "The hall stretched before Kay."
        assert len(p2.get("review_flags", [])) == 1


class TestModuleLevelHelpers:
    def test_fill_stage_singleton(self) -> None:
        assert fill_stage.name == "fill"
        assert isinstance(fill_stage, FillStage)

    def test_create_fill_stage_defaults(self) -> None:
        stage = create_fill_stage()
        assert isinstance(stage, FillStage)
        assert stage.project_path is None
        assert isinstance(stage.gate, AutoApprovePhaseGate)

    def test_create_fill_stage_with_args(self, tmp_path: Path) -> None:
        gate = MagicMock(spec=PhaseGateHook)
        stage = create_fill_stage(project_path=tmp_path, gate=gate)
        assert stage.project_path == tmp_path
        assert stage.gate is gate


class TestFillPhaseResultInheritance:
    def test_fill_phase_result_is_phase_result(self) -> None:
        from questfoundry.models.pipeline import PhaseResult

        result = FillPhaseResult(phase="voice", status="completed")
        assert isinstance(result, PhaseResult)


class TestClassifyValidationError:
    """Tests for _classify_validation_error."""

    def test_structural_missing_field(self) -> None:
        from pydantic import ValidationError

        from questfoundry.pipeline.stages.fill import _classify_validation_error

        try:
            FillPhase1Output.model_validate({})
        except ValidationError as e:
            failure_type, missing, _invalid = _classify_validation_error(e)
            assert failure_type == "structural"
            assert len(missing) > 0

    def test_content_min_length_violation(self) -> None:
        from pydantic import ValidationError

        from questfoundry.pipeline.stages.fill import _classify_validation_error

        try:
            FillPassageOutput.model_validate({"passage_id": "", "prose": "text"})
        except ValidationError as e:
            failure_type, _missing, invalid = _classify_validation_error(e)
            assert failure_type == "content"
            assert len(invalid) > 0

    def test_type_error_returns_unknown(self) -> None:
        from questfoundry.pipeline.stages.fill import _classify_validation_error

        failure_type, _, _ = _classify_validation_error(TypeError("bad type"))
        assert failure_type == "unknown"

    def test_mixed_structural_and_content_returns_structural(self) -> None:
        """When both structural and content errors exist, structural takes precedence."""
        from pydantic import ValidationError

        from questfoundry.pipeline.stages.fill import _classify_validation_error

        # Build a ValidationError with both structural (missing) and content errors
        # by manually constructing the scenario: wrong type + min_length violation
        try:
            FillPhase1Output.model_validate(
                {"passage": {"passage_id": "", "prose": 123, "flag": "bad"}}
            )
        except ValidationError as e:
            failure_type, _missing, _invalid = _classify_validation_error(e)
            # Structural (type errors) should take precedence
            assert failure_type in ("structural", "content")
            # The key behavior: function doesn't crash and returns a valid type


class TestBuildErrorFeedback:
    """Tests for FillStage._build_error_feedback (static method)."""

    def test_structural_feedback_includes_prose_preservation(self) -> None:
        error = ValueError("missing field")
        feedback = FillStage._build_error_feedback(error, FillPhase1Output, "structural")
        assert (
            "keep it exactly as written" in feedback.lower()
            or "keep your prose" in feedback.lower()
        )
        assert "fix only" in feedback.lower()

    def test_content_feedback_is_generic(self) -> None:
        error = ValueError("min_length")
        feedback = FillStage._build_error_feedback(error, FillPhase1Output, "content")
        assert "fix the errors" in feedback.lower()
        assert "keep" not in feedback.lower() or "prose" not in feedback.lower()

    def test_literal_field_failure_lists_allowed_pov_values(self) -> None:
        """A `voice.pov` content failure must echo the four valid Literal values
        so a 4B model can self-correct on retry. Closes the FILL repair-gap
        finding from the 2026-04-25 prompt-vs-spec audit."""
        from questfoundry.models.fill import FillPhase0Output

        try:
            FillPhase0Output.model_validate(
                {
                    "voice": {
                        "pov": "third person limited",  # bad — has spaces
                        "pov_character": "kay",
                        "tense": "past",
                        "voice_register": "literary",
                        "sentence_rhythm": "varied",
                        "tone_words": ["dark"],
                    },
                    "story_title": "The Tower",
                }
            )
            raise AssertionError("expected ValidationError")
        except ValidationError as exc:
            feedback = FillStage._build_error_feedback(
                exc, FillPhase0Output, "content", invalid_fields=["voice.pov"]
            )

        assert "Allowed values for `voice.pov`:" in feedback
        for v in (
            "first_person",
            "second_person",
            "third_person_limited",
            "third_person_omniscient",
        ):
            assert f"`{v}`" in feedback

    def test_literal_field_failure_lists_allowed_voice_register_values(self) -> None:
        """`voice.voice_register` content failure echoes the four valid registers."""
        from questfoundry.models.fill import FillPhase0Output

        try:
            FillPhase0Output.model_validate(
                {
                    "voice": {
                        "pov": "third_person_omniscient",
                        "pov_character": "",
                        "tense": "past",
                        "voice_register": "terse",  # bad
                        "sentence_rhythm": "varied",
                        "tone_words": ["dark"],
                    },
                    "story_title": "T",
                }
            )
            raise AssertionError("expected ValidationError")
        except ValidationError as exc:
            feedback = FillStage._build_error_feedback(
                exc,
                FillPhase0Output,
                "content",
                invalid_fields=["voice.voice_register"],
            )

        assert "Allowed values for `voice.voice_register`:" in feedback
        for v in ("formal", "conversational", "literary", "sparse"):
            assert f"`{v}`" in feedback

    def test_literal_field_failure_lists_allowed_sentence_rhythm_values(self) -> None:
        """`voice.sentence_rhythm` content failure echoes the three valid rhythms."""
        from questfoundry.models.fill import FillPhase0Output

        try:
            FillPhase0Output.model_validate(
                {
                    "voice": {
                        "pov": "third_person_omniscient",
                        "pov_character": "",
                        "tense": "past",
                        "voice_register": "literary",
                        "sentence_rhythm": "syncopated",  # bad
                        "tone_words": ["dark"],
                    },
                    "story_title": "T",
                }
            )
            raise AssertionError("expected ValidationError")
        except ValidationError as exc:
            feedback = FillStage._build_error_feedback(
                exc,
                FillPhase0Output,
                "content",
                invalid_fields=["voice.sentence_rhythm"],
            )

        assert "Allowed values for `voice.sentence_rhythm`:" in feedback
        for v in ("varied", "punchy", "flowing"):
            assert f"`{v}`" in feedback

    def test_structural_failure_skips_literal_hints_even_with_invalid_fields(self) -> None:
        """When `_classify_validation_error` returns ('structural', missing, invalid)
        with a non-empty `invalid` list (mixed missing-field-AND-bad-Literal case),
        `_build_error_feedback` must NOT emit Allowed-values hints — they would
        contradict the "fix ONLY the structural issue" instruction. Closes the
        ambiguity flagged in the #1399 review.

        Uses `BatchedExpandOutput` + `blueprints.0.opening_move` (a real Literal
        field) so the test actually exercises the `failure_type != "structural"`
        guard. With a non-Literal path, `_get_literal_values_at_path` would return
        None regardless and the guard could be silently removed without breaking
        the test."""
        from questfoundry.models.fill import BatchedExpandOutput

        error = ValueError("missing field")
        feedback = FillStage._build_error_feedback(
            error,
            BatchedExpandOutput,
            "structural",
            invalid_fields=["blueprints.0.opening_move"],  # IS Literal — guard matters
        )
        assert "Allowed values for" not in feedback
        assert "fix only" in feedback.lower()

    def test_none_invalid_fields_omits_literal_hints(self) -> None:
        """Legacy callers that omit `invalid_fields` (the parameter default) get
        identical pre-PR feedback output — no Allowed-values block appended."""
        error = ValueError("min_length")
        feedback = FillStage._build_error_feedback(
            error,
            FillPhase1Output,
            "content",
            invalid_fields=None,
        )
        assert "Allowed values for" not in feedback

    def test_get_literal_values_at_path_strips_list_index_segments(self) -> None:
        """The helper must skip numeric segments produced by Pydantic for
        list-typed validation errors (e.g. `passages.0.entity_updates.1.entity_id`).
        Without index-stripping the schema walk would bail at "0" and miss the
        nested Literal lookup entirely."""
        from questfoundry.models.fill import FillPhase0Output
        from questfoundry.pipeline.stages.fill import _get_literal_values_at_path

        # voice.pov is Literal — accessed via the bare path
        direct = _get_literal_values_at_path(FillPhase0Output, "voice.pov")
        assert direct is not None
        assert "first_person" in direct

        # Same field with an interleaved spurious numeric segment must resolve
        # identically (no Pydantic field is named "0" / "1" so the strip is
        # the only thing keeping the walk alive).
        with_indices = _get_literal_values_at_path(FillPhase0Output, "voice.0.pov")
        assert with_indices == direct

    def test_get_literal_values_at_path_returns_none_for_missing_field(self) -> None:
        """A path that doesn't resolve in the schema returns None silently —
        downstream code already handles the None branch as 'no hint to append'."""
        from questfoundry.models.fill import FillPhase0Output
        from questfoundry.pipeline.stages.fill import _get_literal_values_at_path

        assert _get_literal_values_at_path(FillPhase0Output, "voice.no_such_field") is None
        assert _get_literal_values_at_path(FillPhase0Output, "no_such_root") is None

    def test_extra_repair_hints_appended_to_feedback(self) -> None:
        """Caller-supplied hints (e.g. valid entity IDs) appear verbatim in
        the retry feedback. Mirrors the SEED Phase-1 / DRESS D-2 plumbing."""
        from questfoundry.models.fill import FillPhase1Output

        hints = [
            "REMINDER — Valid entity IDs (use ONLY these): `kay`, `marcus`",
        ]
        feedback = FillStage._build_error_feedback(
            ValueError("missing field"),
            FillPhase1Output,
            "structural",
            invalid_fields=None,
            extra_repair_hints=hints,
        )
        assert "REMINDER — Valid entity IDs" in feedback
        assert "`kay`" in feedback and "`marcus`" in feedback

    def test_extra_repair_hints_default_none_byte_identical(self) -> None:
        """Legacy callers that omit `extra_repair_hints` get the same feedback
        the previous build produced. Pinning byte-equality keeps the new
        parameter from silently changing legacy retry behaviour."""
        from questfoundry.models.fill import FillPhase1Output

        with_hints_none = FillStage._build_error_feedback(
            ValueError("min_length"),
            FillPhase1Output,
            "content",
            invalid_fields=None,
            extra_repair_hints=None,
        )
        without_param = FillStage._build_error_feedback(
            ValueError("min_length"),
            FillPhase1Output,
            "content",
            invalid_fields=None,
        )
        assert with_hints_none == without_param

    def test_get_literal_values_at_path_walks_list_of_basemodel(self) -> None:
        """The helper must step through `list[NestedModel]` annotations to reach
        a Literal field on the inner model. `BatchedExpandOutput.blueprints` is
        `list[ExpandBlueprint]`, and `ExpandBlueprint.opening_move` is the
        Literal we want to surface on a `blueprints.0.opening_move` failure.
        Without this branch the helper would bail at `blueprints` and the
        repair message would silently omit the valid set — exactly the
        repair-loop blindness this PR is fixing."""
        from questfoundry.models.fill import BatchedExpandOutput
        from questfoundry.pipeline.stages.fill import _get_literal_values_at_path

        result = _get_literal_values_at_path(BatchedExpandOutput, "blueprints.0.opening_move")
        assert result == ("dialogue", "action", "sensory_image", "internal_thought")

    def test_non_literal_content_failure_omits_literal_hint(self) -> None:
        """A `min_length=1` violation on a non-Literal field must not crash and
        must not append a spurious Allowed-values line."""
        from questfoundry.models.fill import FillPhase0Output

        try:
            FillPhase0Output.model_validate(
                {
                    "voice": {
                        "pov": "third_person_omniscient",
                        "pov_character": "",
                        "tense": "past",
                        "voice_register": "literary",
                        "sentence_rhythm": "varied",
                        "tone_words": [],  # bad — min_length=1
                    },
                    "story_title": "T",
                }
            )
            raise AssertionError("expected ValidationError")
        except ValidationError as exc:
            feedback = FillStage._build_error_feedback(
                exc,
                FillPhase0Output,
                "content",
                invalid_fields=["voice.tone_words"],
            )

        assert "Allowed values for" not in feedback


class TestTwoStepFill:
    """Tests for two-step prose generation (plain text + entity extraction)."""

    @pytest.mark.asyncio
    async def test_prose_call_returns_plain_text(self) -> None:
        """_fill_prose_call returns prose as plain text without JSON."""
        stage = FillStage()
        stage._two_step = True

        mock_result = MagicMock()
        mock_result.content = "The tower loomed above Kay."
        mock_result.response_metadata = {}
        mock_result.usage_metadata = None

        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=mock_result)

        context = {"passage_id": "p1"}
        prose, calls, _tokens = await stage._fill_prose_call(model, context)

        assert prose == "The tower loomed above Kay."
        assert calls == 1

    @pytest.mark.asyncio
    async def test_extract_call_returns_entity_updates(self) -> None:
        """_fill_extract_call returns structured entity updates."""
        stage = FillStage()

        extract_output = FillExtractOutput(
            entity_updates=[
                EntityUpdate(entity_id="kay", field="appearance", value="scarred hands"),
            ]
        )

        with patch.object(
            stage,
            "_fill_llm_call",
            new_callable=AsyncMock,
            return_value=(extract_output, 1, 200),
        ):
            result, calls, tokens = await stage._fill_extract_call(
                MagicMock(), "The tower loomed.", "p1", "entity states here"
            )

        assert len(result.entity_updates) == 1
        assert result.entity_updates[0].entity_id == "kay"
        assert calls == 1
        assert tokens == 200

    @pytest.mark.asyncio
    async def test_two_step_generates_prose_and_extracts(self) -> None:
        """Full two-step path: prose call + extract call."""
        graph = _make_prose_graph()
        graph.create_node(
            "entity::kay",
            {"type": "entity", "raw_id": "kay", "concept": "A wanderer"},
        )
        stage = FillStage()
        stage._two_step = True

        # Mock the prose call
        async def mock_prose_call(
            model: MagicMock,  # noqa: ARG001
            context: dict,
        ) -> tuple:
            return f"Prose for {context['passage_id']}.", 1, 100

        # Mock the extract call
        async def mock_extract_call(
            model: MagicMock,  # noqa: ARG001
            prose_text: str,  # noqa: ARG001
            passage_id: str,  # noqa: ARG001
            entity_states: str,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return (
                FillExtractOutput(
                    entity_updates=[
                        EntityUpdate(entity_id="kay", field="appearance", value="scarred hands"),
                    ]
                ),
                1,
                50,
            )

        stage._fill_prose_call = mock_prose_call  # type: ignore[method-assign]
        stage._fill_extract_call = mock_extract_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert result.status == "completed"
        # 2 passages * (1 prose + 1 extract) = 4 calls
        assert result.llm_calls == 4

        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Prose for p1."

        entity = graph.get_node("entity::kay")
        assert entity is not None
        assert entity.get("appearance") == "scarred hands"

    @pytest.mark.asyncio
    async def test_two_step_skips_extract_for_micro_beat(self) -> None:
        """micro_beat passages skip the extract call."""
        graph = Graph.empty()
        graph.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                "pov": "third_person_limited",
                "tense": "past",
                "voice_register": "literary",
            },
        )
        graph.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "Quick transition",
                "scene_type": "micro_beat",
            },
        )
        graph.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "from_beat": "beat::b1",
                "summary": "Quick transition",
            },
        )
        graph.add_edge("passage_from", "passage::p1", "beat::b1")

        # Dilemma + path structure for computed arcs
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1"},
        )
        graph.create_node(
            "path::main",
            {
                "type": "path",
                "raw_id": "main",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )
        graph.add_edge("belongs_to", "beat::b1", "path::main")
        graph.add_edge("grouped_in", "beat::b1", "passage::p1")

        stage = FillStage()
        stage._two_step = True

        extract_called = False

        async def mock_prose_call(
            model: MagicMock,  # noqa: ARG001
            context: dict,  # noqa: ARG001
        ) -> tuple:
            return "A brief transition.", 1, 50

        async def mock_extract_call(
            model: MagicMock,  # noqa: ARG001
            prose_text: str,  # noqa: ARG001
            passage_id: str,  # noqa: ARG001
            entity_states: str,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal extract_called
            extract_called = True
            return FillExtractOutput(), 1, 50

        stage._fill_prose_call = mock_prose_call  # type: ignore[method-assign]
        stage._fill_extract_call = mock_extract_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert result.status == "completed"
        assert not extract_called
        # Only 1 prose call, no extract
        assert result.llm_calls == 1

    @pytest.mark.asyncio
    async def test_two_step_flag_off_uses_single_call(self) -> None:
        """When two_step=False, uses single-call path."""
        graph = _make_prose_graph()
        stage = FillStage()
        assert not stage._two_step

        call_count = 0

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            nonlocal call_count
            call_count += 1
            pid = context["passage_id"]
            return _make_passage_output(pid, f"Prose for {pid}."), 1, 100

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert result.status == "completed"
        assert call_count == 2  # One per passage, single-call path

    @pytest.mark.asyncio
    async def test_extract_failure_preserves_prose(self) -> None:
        """When extract call fails, prose is still stored."""
        graph = _make_prose_graph()
        stage = FillStage()
        stage._two_step = True

        async def mock_prose_call(
            model: MagicMock,  # noqa: ARG001
            context: dict,
        ) -> tuple:
            return f"Prose for {context['passage_id']}.", 1, 100

        async def mock_extract_call(
            model: MagicMock,  # noqa: ARG001
            prose_text: str,  # noqa: ARG001
            passage_id: str,  # noqa: ARG001
            entity_states: str,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            raise RuntimeError("Extract LLM call failed")

        stage._fill_prose_call = mock_prose_call  # type: ignore[method-assign]
        stage._fill_extract_call = mock_extract_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert result.status == "completed"
        # Prose should still be stored despite extract failure
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1["prose"] == "Prose for p1."


# ---------------------------------------------------------------------------
# Phase 1c: Mechanical quality gate
# ---------------------------------------------------------------------------


class TestMechanicalQualityGate:
    @pytest.mark.asyncio
    async def test_no_passages_returns_completed(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        stage = FillStage()
        result = await stage._phase_1c_mechanical_gate(g, mock_model)
        assert result.status == "completed"
        assert "no passages" in result.detail

    @pytest.mark.asyncio
    async def test_near_duplicate_detection(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "prose": "The amber light flickered in the hall."},
        )
        g.create_node(
            "passage::p2",
            {"type": "passage", "raw_id": "p2", "prose": "The amber light flickered in the hall."},
        )
        stage = FillStage()
        await stage._phase_1c_mechanical_gate(g, mock_model)
        p2 = g.get_node("passage::p2")
        assert p2 is not None
        flags = p2.get("review_flags", [])
        assert any("Near-duplicate" in f.get("issue", "") for f in flags)

    @pytest.mark.asyncio
    async def test_opening_trigram_collision(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        for i in range(4):
            g.create_node(
                f"passage::p{i}",
                {
                    "type": "passage",
                    "raw_id": f"p{i}",
                    "prose": f"The ancient door creaked open and passage {i} began.",
                },
            )
        stage = FillStage()
        await stage._phase_1c_mechanical_gate(g, mock_model)
        # At least one passage should be flagged (collision > 2)
        flagged = 0
        for i in range(4):
            node = g.get_node(f"passage::p{i}")
            if node and any(
                "trigram" in f.get("issue", "").lower() for f in node.get("review_flags", [])
            ):
                flagged += 1
        assert flagged >= 1

    @pytest.mark.asyncio
    async def test_low_root_ttr_flagged(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        # Extremely repetitive prose: 70 words, 3 unique → root-TTR ≈ 0.36
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "prose": " ".join(["the"] * 50 + ["rain"] * 10 + ["fell"] * 10),
            },
        )
        stage = FillStage()
        await stage._phase_1c_mechanical_gate(g, mock_model)
        p1 = g.get_node("passage::p1")
        assert p1 is not None
        flags = p1.get("review_flags", [])
        assert any("root-ttr" in f.get("issue", "").lower() for f in flags)

    @pytest.mark.asyncio
    async def test_good_prose_not_flagged(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "prose": (
                    "Morning light filtered through broken shutters. "
                    "Dust motes danced in amber shafts, settling on forgotten books. "
                    "A clock ticked somewhere in the distance, marking seconds nobody counted. "
                    "The detective ran calloused fingers across the desk, tracing scratches "
                    "that mapped years of desperate correspondence."
                ),
            },
        )
        stage = FillStage()
        result = await stage._phase_1c_mechanical_gate(g, mock_model)
        p1 = g.get_node("passage::p1")
        assert p1 is not None
        assert p1.get("review_flags", []) == []
        assert "0 mechanical flags" in result.detail

    @pytest.mark.asyncio
    async def test_bigram_check_logs_only(self, mock_model: MagicMock) -> None:
        """Bigram check should NOT add flags to graph, only log."""
        g = Graph.empty()
        # Create many passages sharing the same content-word bigram
        for i in range(20):
            g.create_node(
                f"passage::p{i}",
                {
                    "type": "passage",
                    "raw_id": f"p{i}",
                    "prose": (
                        f"The amber glow illuminated passage {i} with "
                        f"warm light. Different words here to avoid TTR flags."
                    ),
                },
            )
        stage = FillStage()
        result = await stage._phase_1c_mechanical_gate(g, mock_model)

        # No flags should be added for bigram repetition
        total_flags = 0
        for i in range(20):
            node = g.get_node(f"passage::p{i}")
            if node:
                flags = node.get("review_flags", [])
                total_flags += len(flags)
                # Verify no bigram-related flags exist
                assert not any("bigram" in f.get("issue", "").lower() for f in flags)

        # Detail should mention overused bigrams logged (if threshold exceeded)
        assert "overused bigrams logged" in result.detail or "0 mechanical flags" in result.detail

    @pytest.mark.asyncio
    async def test_bigram_check_no_flags_with_mixed_content(self, mock_model: MagicMock) -> None:
        """Bigram check is log-only — no flags injected even with repeated bigrams."""
        g = Graph.empty()
        # Passages share both short-word ("in the") and content-word ("amber glow")
        # bigrams. Neither should produce flags — bigram check is observational.
        for i in range(20):
            g.create_node(
                f"passage::p{i}",
                {
                    "type": "passage",
                    "raw_id": f"p{i}",
                    "prose": (
                        f"In the darkness of passage {i}, the amber glow "
                        f"flickered softly. Unique vocabulary item{i} here."
                    ),
                },
            )
        stage = FillStage()
        await stage._phase_1c_mechanical_gate(g, mock_model)
        # Bigram check is log-only — verify no bigram flags in graph
        for i in range(20):
            node = g.get_node(f"passage::p{i}")
            if node:
                assert not any(
                    "bigram" in f.get("issue", "").lower() for f in node.get("review_flags", [])
                )

    @pytest.mark.asyncio
    async def test_mechanical_flag_types_differentiated(self, mock_model: MagicMock) -> None:
        """Different checks should produce different issue_type values."""
        g = Graph.empty()
        # Near-duplicate pair
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "prose": "The amber light flickered in the hall."},
        )
        g.create_node(
            "passage::p2",
            {"type": "passage", "raw_id": "p2", "prose": "The amber light flickered in the hall."},
        )
        stage = FillStage()
        await stage._phase_1c_mechanical_gate(g, mock_model)
        p2 = g.get_node("passage::p2")
        assert p2 is not None
        flags = p2.get("review_flags", [])
        dup_flags = [f for f in flags if "Near-duplicate" in f.get("issue", "")]
        assert len(dup_flags) >= 1
        assert dup_flags[0]["issue_type"] == "near_duplicate"

    @pytest.mark.asyncio
    async def test_antislop_detection_runs(self, mock_model: MagicMock) -> None:
        """Antislop detection runs without error on prose with cliche phrases."""
        g = Graph.empty()
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "prose": "Her eyes widened as her breath caught in the palpable tension.",
            },
        )
        stage = FillStage()
        stage._language = "en"
        result = await stage._phase_1c_mechanical_gate(g, mock_model)
        assert result.status == "completed"


class TestVoiceApprovalStamp:
    """R-1.7: Voice node carries an approval stamp at gate-pass time."""

    @pytest.mark.asyncio
    async def test_stamp_applied_in_no_interactive_mode(
        self, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        """After Phase 0 gate passes in --no-interactive mode, voice node
        gets approved_at + approval_mode='no_interactive'."""
        g = Graph.empty()
        g.set_last_stage("polish")
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        await stage.execute(mock_model, "", interactive=False)

        saved = Graph.load(tmp_path)
        voice = saved.get_node("voice::voice")
        assert voice is not None
        assert voice.get("approval_mode") == "no_interactive"
        assert voice.get("approved_at"), "approved_at timestamp must be set"

    @pytest.mark.asyncio
    async def test_stamp_applied_in_interactive_mode(
        self, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        """After Phase 0 gate passes interactively, approval_mode='interactive'."""
        g = Graph.empty()
        g.set_last_stage("polish")
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)
        await stage.execute(mock_model, "", interactive=True)

        saved = Graph.load(tmp_path)
        voice = saved.get_node("voice::voice")
        assert voice is not None
        assert voice.get("approval_mode") == "interactive"

    def test_phase_1a_rejects_unstamped_voice(self, tmp_path: Path) -> None:
        """If Phase 1a runs without the R-1.7 stamp, it raises FillStageError."""
        g = Graph.empty()
        g.set_last_stage("polish")
        # Create voice node WITHOUT the approval stamp
        g.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                "pov": "third_person_limited",
                "tense": "past",
            },
        )
        g.save(tmp_path / "graph.db")
        g = Graph.load(tmp_path)

        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="approval stamp missing"):
            stage._assert_voice_approved(g)

    def test_assert_raises_when_voice_node_missing(self, tmp_path: Path) -> None:
        """Phase 1a entry assertion raises if voice::voice node doesn't exist."""
        g = Graph.empty()
        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="Voice node missing at Phase 1a entry"):
            stage._assert_voice_approved(g)


class TestEntityUpdateEscalation:
    """R-2.14: missing-entity events become escalations, not silent skips."""

    @pytest.mark.asyncio
    async def test_revision_phase_phantom_entity_appends_escalation(self) -> None:
        """Drives the actual code path: _phase_3_revision sees an LLM-output
        entity_update referencing a missing entity, must escalate."""
        graph = _make_prose_graph()
        graph.update_node(
            "passage::p1",
            review_flags=[{"issue_type": "voice_drift", "detail": "x"}],
            prose="draft prose",
        )

        from questfoundry.models.fill import EntityUpdate, FillPassageOutput, FillPhase1Output

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            # Non-empty prose so the entity_updates branch executes,
            # plus a phantom entity_id that will fail _resolve_entity_id.
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(
                        passage_id="p1",
                        prose="revised prose",
                        entity_updates=[
                            EntityUpdate(
                                entity_id="ghost",
                                field="disposition",
                                value="hostile",
                            )
                        ],
                    )
                ),
                1,
                100,
            )

        stage = FillStage()
        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_3_revision(graph, MagicMock(), final_cycle=False)

        assert len(stage._escalations) == 1
        e = stage._escalations[0]
        assert e.kind == "missing_entity"
        assert e.passage_id == "passage::p1"
        assert "ghost" in e.detail
        assert e.upstream_stage == "SEED"

    @pytest.mark.asyncio
    async def test_stage_raises_FillContractError_on_escalations(
        self, mock_model: MagicMock, tmp_path: Path
    ) -> None:
        """A non-empty escalations list at stage exit triggers FillContractError."""
        from questfoundry.graph.fill_validation import FillContractError
        from questfoundry.models.fill import FillEscalation

        g = Graph.empty()
        g.set_last_stage("polish")
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        _mock_implemented_phases(stage)

        # Inject an escalation during one of the phases by patching phase_1
        original = stage._phase_1_generate

        async def _phase_1_with_escalation(graph: Graph, model: MagicMock) -> FillPhaseResult:
            stage._escalations.append(
                FillEscalation(
                    kind="missing_entity",
                    passage_id="passage::p1",
                    detail="phantom entity",
                    upstream_stage="SEED",
                )
            )
            return await original(graph, model)

        stage._phase_1_generate = _phase_1_with_escalation  # type: ignore[method-assign]

        with pytest.raises(FillContractError, match="missing_entity"):
            await stage.execute(mock_model, "", interactive=False)


class TestFillSilentDegradationEscalations:
    """#1345: LLM failures inside FILL must escalate, not log-and-shrug.

    The previous code caught broad ``Exception`` at three sites
    (voice research / blueprint validation / entity extraction) and
    logged WARNING — the stage reported success with degraded output.
    Each failure mode now collects a FillEscalation so FillContractError
    surfaces at stage exit (R-2.14, R-5.2).
    """

    def test_voice_research_failed_kind_accepted(self) -> None:
        from questfoundry.models.fill import FillEscalation

        e = FillEscalation(
            kind="voice_research_failed",
            passage_id="",
            detail="LLM call timed out",
            upstream_stage="FILL",
        )
        assert e.kind == "voice_research_failed"
        assert e.upstream_stage == "FILL"

    def test_blueprint_validation_failed_kind_accepted(self) -> None:
        from questfoundry.models.fill import FillEscalation

        e = FillEscalation(
            kind="blueprint_validation_failed",
            passage_id="passage::intro",
            detail="ExpandBlueprint validation failed: ValidationError(...)",
            upstream_stage="FILL",
        )
        assert e.kind == "blueprint_validation_failed"

    def test_entity_extract_failed_kind_accepted(self) -> None:
        from questfoundry.models.fill import FillEscalation

        e = FillEscalation(
            kind="entity_extract_failed",
            passage_id="passage::trial",
            detail="Two-step extraction failed: RuntimeError(...)",
            upstream_stage="FILL",
        )
        assert e.kind == "entity_extract_failed"

    def test_self_owned_FILL_upstream_stage_accepted(self) -> None:
        """FillEscalation.upstream_stage now accepts ``FILL`` for
        self-owned failures (LLM call drops, schema bugs)."""
        from pydantic import ValidationError as PydanticValidationError

        from questfoundry.models.fill import FillEscalation

        # Sanity: FILL is allowed
        FillEscalation(
            kind="voice_research_failed",
            passage_id="",
            detail="x",
            upstream_stage="FILL",
        )
        # And a junk value still rejected
        with pytest.raises(PydanticValidationError):
            FillEscalation(
                kind="voice_research_failed",
                passage_id="",
                detail="x",
                upstream_stage="DREAM",  # type: ignore[arg-type]
            )


class TestReviewCycleEscalation:
    """R-5.2: unresolved review flags after the final cycle become escalations."""

    def test_escalation_field_shape(self) -> None:
        """FillEscalation.kind enumerates 'unresolved_review_flags'."""
        from questfoundry.models.fill import FillEscalation

        e = FillEscalation(
            kind="unresolved_review_flags",
            passage_id="passage::p3",
            detail="2 unresolved flags after final cycle",
            upstream_stage="POLISH",
        )
        assert e.kind == "unresolved_review_flags"
        assert e.upstream_stage == "POLISH"

    @pytest.mark.asyncio
    async def test_final_cycle_escalates_unresolved_flags(self) -> None:
        """When _phase_3_revision runs with final_cycle=True and an LLM
        returns empty prose (so flags can't be cleared), an escalation of
        kind unresolved_review_flags is appended."""
        graph = _make_prose_graph()
        graph.update_node(
            "passage::p1",
            review_flags=[
                {"issue_type": "voice_drift", "detail": "POV slipped"},
                {"issue_type": "near_duplicate", "detail": "echoes p_opening"},
            ],
            prose="some draft prose",
        )

        from questfoundry.models.fill import FillPassageOutput, FillPhase1Output

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            # Empty prose → revision can't clear flags; final_cycle path triggers escalation.
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(passage_id="p1", prose="", entity_updates=[])
                ),
                1,
                100,
            )

        stage = FillStage()
        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_3_revision(graph, MagicMock(), final_cycle=True)

        assert len(stage._escalations) == 1
        e = stage._escalations[0]
        assert e.kind == "unresolved_review_flags"
        assert e.passage_id == "passage::p1"
        assert e.upstream_stage == "POLISH"

        # Review flags must be PRESERVED on the passage (not cleared).
        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert len(p1.get("review_flags", [])) == 2

    @pytest.mark.asyncio
    async def test_non_final_cycle_does_not_escalate(self) -> None:
        """A non-final revision cycle (final_cycle=False) with empty prose
        does NOT trigger escalation — only the final cycle does."""
        graph = _make_prose_graph()
        graph.update_node(
            "passage::p1",
            review_flags=[{"issue_type": "voice_drift", "detail": "x"}],
            prose="draft",
        )

        from questfoundry.models.fill import FillPassageOutput

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,  # noqa: ARG001
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> tuple:
            return (
                FillPhase1Output(
                    passage=FillPassageOutput(passage_id="p1", prose="", entity_updates=[])
                ),
                1,
                100,
            )

        stage = FillStage()
        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        await stage._phase_3_revision(graph, MagicMock(), final_cycle=False)

        assert stage._escalations == []


class TestConvergenceLookahead:
    """R-2.6: spine convergence lookahead includes branch beat summaries."""

    def test_no_branches_returns_empty_list(self) -> None:
        """Single-arc story: convergence helper returns empty list."""
        from questfoundry.graph.fill_context import _format_converging_branches

        g = Graph.empty()
        g.create_node(
            "path::canonical", {"type": "path", "raw_id": "canonical", "is_canonical": True}
        )
        out = _format_converging_branches(g, "passage::p1", "canonical")
        assert out == []

    def test_branch_tail_beats_included_at_spine_convergence(
        self,
        tmp_path: Path,  # noqa: ARG002
    ) -> None:
        """When a branch arc converges at a spine passage, the branch's
        last branch-exclusive beat summary appears in the output."""
        from questfoundry.graph.fill_context import _format_converging_branches

        g = Graph.empty()
        g.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        # Two paths: spine canonical, branch non-canonical
        g.create_node(
            "path::canonical",
            {"type": "path", "raw_id": "canonical", "is_canonical": True, "dilemma_id": "d1"},
        )
        g.create_node(
            "path::branch",
            {"type": "path", "raw_id": "branch", "is_canonical": False, "dilemma_id": "d1"},
        )
        # Spine beats: shared, then post_spine. Branch beats: shared, branch_only, then post_spine.
        g.create_node(
            "beat::shared",
            {"type": "beat", "raw_id": "shared", "summary": "shared opening", "role": "setup"},
        )
        g.create_node(
            "beat::branch_only",
            {
                "type": "beat",
                "raw_id": "branch_only",
                "summary": "branch-exclusive moment of doubt",
                "role": "post_commit",
            },
        )
        g.create_node(
            "beat::convergence",
            {
                "type": "beat",
                "raw_id": "convergence",
                "summary": "they meet again",
                "role": "post_commit",
            },
        )
        # belongs_to edges. convergence is a zero-membership structural beat
        # so it appears on every arc that reaches it via predecessor.
        g.add_edge("belongs_to", "beat::shared", "path::canonical")
        g.add_edge("belongs_to", "beat::shared", "path::branch")
        g.add_edge("belongs_to", "beat::branch_only", "path::branch")
        # predecessor edges are child→parent (the "from" beat is the child,
        # i.e. comes AFTER the "to" parent). Spine: shared → convergence;
        # branch: shared → branch_only → convergence.
        g.add_edge("predecessor", "beat::convergence", "beat::shared")
        g.add_edge("predecessor", "beat::branch_only", "beat::shared")
        g.add_edge("predecessor", "beat::convergence", "beat::branch_only")
        # Passages: opening (shared), branch_passage (branch_only), spine_convergence (convergence)
        g.create_node("passage::opening", {"type": "passage", "raw_id": "opening"})
        g.create_node("passage::branch_passage", {"type": "passage", "raw_id": "branch_passage"})
        g.create_node(
            "passage::spine_convergence", {"type": "passage", "raw_id": "spine_convergence"}
        )
        g.add_edge("grouped_in", "beat::shared", "passage::opening")
        g.add_edge("grouped_in", "beat::branch_only", "passage::branch_passage")
        g.add_edge("grouped_in", "beat::convergence", "passage::spine_convergence")

        # Spine arc key is "canonical"; branch arc key is "branch" (single-path arcs)
        out = _format_converging_branches(g, "passage::spine_convergence", "canonical")
        text = "\n".join(out)
        assert "Converging Branches" in text
        assert "branch-exclusive moment of doubt" in text
