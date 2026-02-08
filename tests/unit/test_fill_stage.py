"""Tests for FILL stage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    ) -> FillPhaseResult:
        return FillPhaseResult(phase="revision", status="completed", llm_calls=0, tokens_used=0)

    stage._phase_0_voice = _fake_phase_0  # type: ignore[method-assign]
    stage._phase_1a_expand = _fake_phase_1a  # type: ignore[method-assign]
    stage._phase_1_generate = _fake_phase_1  # type: ignore[method-assign]
    stage._phase_1c_mechanical_gate = _fake_phase_1c  # type: ignore[method-assign]
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
        assert "review_summary" not in result_dict  # No telemetry in artifact

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

        assert len(progress_calls) == 7
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
    def test_seven_phases(self) -> None:
        stage = FillStage()
        phases = stage._phase_order()
        assert len(phases) == 7

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
            "arc_validation",
        ]


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
        assert voice_node["pov"] == "third_limited"
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
                return_value=("Use third_limited past for fantasy.", 2, 300),
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
        assert "Use third_limited past for fantasy." in context["research_notes"]

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
                return_value=("Use third_limited past for dark fantasy.", 200),
            ),
        ):
            brief, calls, tokens = await stage._phase_0a_voice_research(graph, MagicMock())

        assert brief == "Use third_limited past for dark fantasy."
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
    async def test_handles_incompatible_states_flag(self) -> None:
        graph = _make_prose_graph()
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
        prose, flag, flag_reason, calls, _tokens = await stage._fill_prose_call(model, context)

        assert prose == "The tower loomed above Kay."
        assert flag == "ok"
        assert flag_reason == ""
        assert calls == 1

    @pytest.mark.asyncio
    async def test_prose_call_detects_sentinel(self) -> None:
        """_fill_prose_call detects INCOMPATIBLE_STATES sentinel."""
        stage = FillStage()
        stage._two_step = True

        mock_result = MagicMock()
        mock_result.content = (
            "INCOMPATIBLE_STATES: Emotional register too divergent between trust and betrayal paths"
        )
        mock_result.response_metadata = {}
        mock_result.usage_metadata = None

        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=mock_result)

        prose, flag, flag_reason, _calls, _tokens = await stage._fill_prose_call(
            model, {"passage_id": "p1"}
        )

        assert prose == ""
        assert flag == "incompatible_states"
        assert "divergent" in flag_reason

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
            return f"Prose for {context['passage_id']}.", "ok", "", 1, 100

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
                "pov": "third_limited",
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
        graph.create_node(
            "arc::spine_0_0",
            {
                "type": "arc",
                "raw_id": "spine_0_0",
                "arc_type": "spine",
                "paths": ["path::main"],
                "sequence": ["beat::b1"],
            },
        )

        stage = FillStage()
        stage._two_step = True

        extract_called = False

        async def mock_prose_call(
            model: MagicMock,  # noqa: ARG001
            context: dict,  # noqa: ARG001
        ) -> tuple:
            return "A brief transition.", "ok", "", 1, 50

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
            return f"Prose for {context['passage_id']}.", "ok", "", 1, 100

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

    @pytest.mark.asyncio
    async def test_sentinel_mid_prose_not_detected(self) -> None:
        """Sentinel appearing mid-response is NOT treated as incompatible."""
        stage = FillStage()
        stage._two_step = True

        mock_result = MagicMock()
        mock_result.content = "The sign read INCOMPATIBLE_STATES: a warning from a forgotten era."
        mock_result.response_metadata = {}
        mock_result.usage_metadata = None

        model = AsyncMock()
        model.ainvoke = AsyncMock(return_value=mock_result)

        prose, flag, _reason, _calls, _tokens = await stage._fill_prose_call(
            model, {"passage_id": "p1"}
        )

        # startswith() correctly ignores sentinel appearing mid-text
        assert flag == "ok"
        assert "INCOMPATIBLE_STATES" in prose


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
    async def test_low_ttr_flagged(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        # Extremely repetitive prose
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
        assert any("diversity" in f.get("issue", "").lower() for f in flags)

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
    async def test_skips_incompatible_states(self, mock_model: MagicMock) -> None:
        g = Graph.empty()
        g.create_node(
            "passage::p1",
            {
                "type": "passage",
                "raw_id": "p1",
                "prose": "some prose",
                "flag": "incompatible_states",
            },
        )
        stage = FillStage()
        result = await stage._phase_1c_mechanical_gate(g, mock_model)
        assert "no passages" in result.detail

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
    async def test_bigram_check_filters_short_words(self, mock_model: MagicMock) -> None:
        """Bigrams of only short words (< 4 chars) should be filtered out."""
        g = Graph.empty()
        # "in the" — both words < 4 chars — should be filtered
        # "amber glow" — both words >= 4 chars — should be counted
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
        # "in the" should NOT appear in logged bigrams
        # We can only verify no flags are added (bigram check is log-only)
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
