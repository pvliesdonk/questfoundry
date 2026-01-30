"""Tests for FILL stage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.models.fill import (
    EntityUpdate,
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
    g.set_last_stage("grow")

    # Minimal passage for counting
    g.create_node(
        "passage::p1",
        {"type": "passage", "raw_id": "p1", "from_beat": "", "summary": "test"},
    )

    g.save(tmp_path / "graph.json")
    return g


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock LangChain model."""
    return MagicMock()


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
                "pov": "third_limited",
                "tense": "past",
                "voice_register": "literary",
                "sentence_rhythm": "varied",
                "tone_words": ["atmospheric"],
            },
        )
        return FillPhaseResult(phase="voice", status="completed", llm_calls=1, tokens_used=500)

    async def _fake_phase_1(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="generate", status="completed", llm_calls=2, tokens_used=1000)

    async def _fake_phase_2(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="review", status="completed", llm_calls=1, tokens_used=200)

    async def _fake_phase_3(
        graph: Graph,  # noqa: ARG001
        model: MagicMock,  # noqa: ARG001
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="revision", status="completed", llm_calls=0, tokens_used=0)

    stage._phase_0_voice = _fake_phase_0  # type: ignore[method-assign]
    stage._phase_1_generate = _fake_phase_1  # type: ignore[method-assign]
    stage._phase_2_review = _fake_phase_2  # type: ignore[method-assign]
    stage._phase_3_revision = _fake_phase_3  # type: ignore[method-assign]


class TestFillStageExecute:
    @pytest.mark.asyncio
    async def test_requires_project_path(self, mock_model: MagicMock) -> None:
        stage = FillStage()
        with pytest.raises(FillStageError, match="project_path is required"):
            await stage.execute(mock_model, "")

    @pytest.mark.asyncio
    async def test_requires_grow_completed(self, mock_model: MagicMock, tmp_path: Path) -> None:
        g = Graph.empty()
        g.set_last_stage("seed")
        g.save(tmp_path / "graph.json")

        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="FILL requires completed GROW"):
            await stage.execute(mock_model, "")

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

        # Result is artifact data (from extract_fill_artifact), not FillResult telemetry
        assert "voice_document" in result_dict
        assert "passages" in result_dict
        assert "review_summary" in result_dict

        # Sum of all phase LLM calls (1 + 2 + 1 + 0 = 4)
        assert llm_calls == 4
        assert tokens == 1700

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

        # Should return artifact data
        assert "voice_document" in result_dict

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

        # Verify rollback was persisted — last_stage should remain "grow"
        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "grow"

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

        # last_stage should remain "grow" (not promoted to "fill")
        saved = Graph.load(tmp_path)
        assert saved.get_last_stage() == "grow"

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

        assert len(progress_calls) == 4
        assert progress_calls[0][0] == "voice"

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
        assert "voice_document" in result_dict


class TestPhaseOrder:
    def test_four_phases(self) -> None:
        stage = FillStage()
        phases = stage._phase_order()
        assert len(phases) == 4

    def test_phase_names(self) -> None:
        stage = FillStage()
        names = [name for _, name in stage._phase_order()]
        assert names == ["voice", "generate", "review", "revision"]


class TestCheckpointing:
    def test_checkpoint_path(self) -> None:
        stage = FillStage()
        path = stage._get_checkpoint_path(Path("/proj"), "voice")
        assert path == Path("/proj/snapshots/fill-pre-voice.json")

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        stage = FillStage()
        g = Graph.empty()
        g.set_last_stage("grow")

        stage._save_checkpoint(g, tmp_path, "voice")
        loaded = stage._load_checkpoint(tmp_path, "voice")
        assert loaded.get_last_stage() == "grow"

    def test_load_missing_checkpoint(self, tmp_path: Path) -> None:
        stage = FillStage()
        with pytest.raises(FillStageError, match="No checkpoint found"):
            stage._load_checkpoint(tmp_path, "nonexistent")


def _make_voice_output() -> FillPhase0Output:
    """Create a valid FillPhase0Output for mocking."""
    return FillPhase0Output(
        voice=VoiceDocument(
            pov="third_limited",
            tense="past",
            voice_register="literary",
            sentence_rhythm="varied",
            tone_words=["atmospheric", "tense"],
            avoid_words=["suddenly"],
            avoid_patterns=["adverb-heavy dialogue tags"],
        )
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

        with patch.object(
            stage,
            "_fill_llm_call",
            new_callable=AsyncMock,
            return_value=(_make_voice_output(), 1, 500),
        ):
            result = await stage._phase_0_voice(graph, MagicMock())

        assert result.phase == "voice"
        assert result.status == "completed"
        assert result.llm_calls == 1
        assert result.tokens_used == 500

        # Voice node should be created in graph
        voice_node = graph.get_node("voice::voice")
        assert voice_node is not None
        assert voice_node["pov"] == "third_limited"
        assert voice_node["tense"] == "past"
        assert voice_node["voice_register"] == "literary"

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
        with patch.object(stage, "_fill_llm_call", mock_llm_call):
            await stage._phase_0_voice(graph, MagicMock())

        # Verify context was passed with expected keys
        call_args = mock_llm_call.call_args
        context = call_args[0][2]  # Third positional arg
        assert "dream_vision" in context
        assert "grow_summary" in context
        assert "scene_types_summary" in context

    @pytest.mark.asyncio
    async def test_llm_failure_raises(self) -> None:
        graph = Graph.empty()
        stage = FillStage()

        with (
            patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                side_effect=FillStageError("LLM call failed"),
            ),
            pytest.raises(FillStageError, match="LLM call failed"),
        ):
            await stage._phase_0_voice(graph, MagicMock())


def _make_prose_graph() -> Graph:
    """Create a graph with voice node and passages for Phase 1 testing."""
    g = Graph.empty()
    g.create_node(
        "voice::voice",
        {
            "type": "voice",
            "raw_id": "voice",
            "pov": "third_limited",
            "tense": "past",
            "voice_register": "literary",
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
    g.create_node(
        "arc::spine_0_0",
        {
            "type": "arc",
            "raw_id": "spine_0_0",
            "arc_type": "spine",
            "paths": ["path::main"],
            "sequence": ["beat::b1", "beat::b2"],
        },
    )
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
    async def test_handles_incompatible_states_flag(self) -> None:
        graph = _make_prose_graph()
        stage = FillStage()

        async def mock_llm_call(
            model: MagicMock,  # noqa: ARG001
            template_name: str,  # noqa: ARG001
            context: dict,
            output_schema: type,  # noqa: ARG001
            max_retries: int = 3,  # noqa: ARG001
        ) -> tuple:
            pid = context["passage_id"]
            if pid == "p1":
                return (
                    FillPhase1Output(
                        passage=FillPassageOutput(
                            passage_id=pid,
                            flag="incompatible_states",
                            flag_reason="States too divergent",
                        )
                    ),
                    1,
                    100,
                )
            return _make_passage_output(pid), 1, 100

        stage._fill_llm_call = mock_llm_call  # type: ignore[method-assign]
        result = await stage._phase_1_generate(graph, MagicMock())

        assert "1 filled, 1 flagged" in (result.detail or "")

        p1 = graph.get_node("passage::p1")
        assert p1 is not None
        assert p1.get("flag") == "incompatible_states"
        assert p1.get("prose", "") == ""

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
        assert order[0] == ("passage::p1", "arc::spine_0_0")
        assert order[1] == ("passage::p2", "arc::spine_0_0")

    def test_skips_already_filled(self) -> None:
        graph = _make_prose_graph()
        # Add a branch arc that shares p1
        graph.create_node(
            "arc::branch_1_0",
            {
                "type": "arc",
                "raw_id": "branch_1_0",
                "arc_type": "branch",
                "paths": ["path::alt"],
                "sequence": ["beat::b1", "beat::b2"],
            },
        )
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

    def test_incompatible_flagged_regenerated_in_branch(self) -> None:
        graph = _make_prose_graph()
        # Add branch arc sharing beats with spine
        graph.create_node(
            "arc::branch_1_0",
            {
                "type": "arc",
                "raw_id": "branch_1_0",
                "arc_type": "branch",
                "paths": ["path::alt"],
                "sequence": ["beat::b1", "beat::b2"],
            },
        )
        # p1 has prose but is flagged incompatible — should re-generate in branch
        graph.update_node("passage::p1", prose="Filled.", flag="incompatible_states")
        stage = FillStage()
        order = stage._get_generation_order(graph)

        passage_ids = [pid for pid, _ in order]
        # p1 appears twice: spine + branch (flagged incompatible)
        assert passage_ids.count("passage::p1") == 2
        # p2 appears once (deduped, no flag)
        assert passage_ids.count("passage::p2") == 1


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
        assert "1 passages revised" in (result.detail or "")

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
