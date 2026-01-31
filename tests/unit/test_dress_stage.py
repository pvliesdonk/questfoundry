"""Tests for DRESS stage skeleton, Phase 0, Phase 1, and Phase 2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from pathlib import Path

from questfoundry.graph.graph import Graph
from questfoundry.models.dress import (
    ArtDirection,
    CodexEntry,
    DressPhase0Output,
    DressPhase1Output,
    DressPhase2Output,
    EntityVisualWithId,
    IllustrationBrief,
)
from questfoundry.pipeline.stages.dress import (
    DressStage,
    DressStageError,
    compute_structural_score,
    create_dress_stage,
    dress_stage,
    map_score_to_priority,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dress_graph(tmp_path: Path) -> Graph:
    """Graph with FILL completed, entities, and passages for DRESS."""
    g = Graph()
    g.set_last_stage("fill")
    g.create_node(
        "vision::main",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": "brooding",
            "themes": ["betrayal", "redemption"],
            "scope": {"story_size": "short"},
        },
    )
    g.create_node(
        "entity::protagonist",
        {
            "type": "entity",
            "raw_id": "protagonist",
            "entity_type": "character",
            "concept": "A young scholar seeking forbidden knowledge",
        },
    )
    g.create_node(
        "entity::aldric",
        {
            "type": "entity",
            "raw_id": "aldric",
            "entity_type": "character",
            "concept": "A former court advisor with hidden motives",
        },
    )
    g.create_node(
        "passage::opening",
        {
            "type": "passage",
            "raw_id": "opening",
            "prose": "The wind howled...",
        },
    )
    g.save(tmp_path / "graph.json")
    return g


@pytest.fixture()
def mock_phase0_output() -> DressPhase0Output:
    """Valid Phase 0 output for mocking serialize_to_artifact."""
    return DressPhase0Output(
        art_direction=ArtDirection(
            style="watercolor",
            medium="traditional watercolor on textured paper",
            palette=["deep indigo", "burnt sienna"],
            composition_notes="Wide shots for locations",
            negative_defaults="photorealism, anime",
            aspect_ratio="16:9",
        ),
        entity_visuals=[
            EntityVisualWithId(
                entity_id="protagonist",
                description="Young woman with short dark hair",
                distinguishing_features=["dark hair", "ink-stained fingers"],
                color_associations=["indigo"],
                reference_prompt_fragment="young woman, short dark hair, ink-stained fingers",
            ),
            EntityVisualWithId(
                entity_id="aldric",
                description="Tall man in faded court robes",
                distinguishing_features=["silver temples", "scarred hand"],
                color_associations=["silver", "grey"],
                reference_prompt_fragment="tall man, silver temples, faded court robes",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Stage basics
# ---------------------------------------------------------------------------


class TestDressStageInit:
    def test_singleton_exists(self) -> None:
        assert dress_stage.name == "dress"

    def test_factory(self, tmp_path: Path) -> None:
        stage = create_dress_stage(project_path=tmp_path)
        assert stage.project_path == tmp_path

    @pytest.mark.asyncio()
    async def test_requires_project_path(self) -> None:
        stage = DressStage()
        with pytest.raises(DressStageError, match="project_path is required"):
            await stage.execute(MagicMock(), "test")


class TestDressStagePrerequisites:
    @pytest.mark.asyncio()
    async def test_rejects_without_fill(self, tmp_path: Path) -> None:
        g = Graph()
        g.set_last_stage("grow")
        g.save(tmp_path / "graph.json")

        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="FILL"):
            await stage.execute(MagicMock(), "test")

    @pytest.mark.asyncio()
    async def test_accepts_dress_rerun(self, tmp_path: Path) -> None:
        """DRESS can re-run on a graph already at dress stage."""
        g = Graph()
        g.set_last_stage("dress")
        g.save(tmp_path / "graph.json")

        stage = DressStage(project_path=tmp_path)
        # Prerequisite check passes but Phase 0 fails (no entities)
        with pytest.raises(DressStageError, match="No entities"):
            await stage.execute(MagicMock(), "test")


class TestDressStageResume:
    @pytest.mark.asyncio()
    async def test_invalid_phase_raises(self, tmp_path: Path) -> None:
        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="Unknown phase"):
            await stage.execute(MagicMock(), "test", resume_from="nonexistent")

    @pytest.mark.asyncio()
    async def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="No checkpoint"):
            await stage.execute(MagicMock(), "test", resume_from="briefs")


# ---------------------------------------------------------------------------
# Phase 0: Art Direction
# ---------------------------------------------------------------------------


class TestPhase0ArtDirection:
    @pytest.mark.asyncio()
    async def test_phase0_creates_nodes(
        self,
        tmp_path: Path,
        dress_graph: Graph,  # noqa: ARG002
        mock_phase0_output: DressPhase0Output,
    ) -> None:
        """Phase 0 creates art_direction and entity_visual nodes."""
        mock_messages = [AIMessage(content="Let's use watercolor style.")]
        mock_brief = "Art direction: watercolor, dark palette"

        stage = DressStage(project_path=tmp_path)

        mock_brief_output = DressPhase1Output(
            brief=IllustrationBrief(
                priority=2,
                category="scene",
                subject="test",
                composition="Wide",
                mood="test",
                caption="test",
            ),
            llm_adjustment=0,
        )
        mock_codex_out = DressPhase2Output(
            entries=[CodexEntry(rank=1, visible_when=[], content="Base knowledge.")]
        )

        with (
            patch(
                "questfoundry.pipeline.stages.dress.run_discuss_phase",
                new_callable=AsyncMock,
                return_value=(mock_messages, 2, 500),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.summarize_discussion",
                new_callable=AsyncMock,
                return_value=(mock_brief, 200),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.serialize_to_artifact",
                new_callable=AsyncMock,
                return_value=(mock_phase0_output, 300),
            ),
            patch.object(
                stage,
                "_dress_llm_call",
                new_callable=AsyncMock,
                # Order is deterministic: dict insertion order (Python 3.7+).
                # Phase 1: 1 passage, Phase 2: protagonist then aldric.
                side_effect=[
                    (mock_brief_output, 1, 50),  # Phase 1: opening passage
                    (mock_codex_out, 1, 50),  # Phase 2: protagonist
                    (mock_codex_out, 1, 50),  # Phase 2: aldric
                ],
            ),
            # Phases 0-2 succeed, Phase 3 raises NotImplementedError
            pytest.raises(NotImplementedError, match="PR 6"),
        ):
            await stage.execute(MagicMock(), "Establish art direction")

        # Verify graph was updated (checkpoint before review has all results)
        checkpoint = tmp_path / "snapshots" / "dress-pre-review.json"
        assert checkpoint.exists()
        graph = Graph.load_from_file(checkpoint)
        assert graph.get_node("art_direction::main") is not None
        assert graph.get_node("entity_visual::protagonist") is not None
        assert graph.get_node("entity_visual::aldric") is not None

    @pytest.mark.asyncio()
    async def test_phase0_counts_metrics(
        self,
        tmp_path: Path,
        dress_graph: Graph,  # noqa: ARG002
        mock_phase0_output: DressPhase0Output,
    ) -> None:
        """Phase 0 correctly accumulates LLM calls and tokens."""
        stage = DressStage(project_path=tmp_path)

        # Use reject gate to stop after Phase 0 and get metrics back
        # (reject rolls back graph but still returns metrics)
        reject_gate = AsyncMock()
        reject_gate.on_phase_complete = AsyncMock(return_value="reject")

        with (
            patch(
                "questfoundry.pipeline.stages.dress.run_discuss_phase",
                new_callable=AsyncMock,
                return_value=([AIMessage(content="ok")], 3, 600),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.summarize_discussion",
                new_callable=AsyncMock,
                return_value=("brief", 150),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.serialize_to_artifact",
                new_callable=AsyncMock,
                return_value=(mock_phase0_output, 250),
            ),
        ):
            stage.gate = reject_gate

            _artifact, llm_calls, tokens = await stage.execute(MagicMock(), "test")

        # discuss(3) + summarize(1) + serialize(1) = 5
        assert llm_calls == 5
        # 600 + 150 + 250 = 1000
        assert tokens == 1000

    @pytest.mark.asyncio()
    async def test_phase0_passes_custom_prompt(
        self,
        tmp_path: Path,
        dress_graph: Graph,  # noqa: ARG002
        mock_phase0_output: DressPhase0Output,
    ) -> None:
        """The user prompt is forwarded to discuss phase."""
        captured_kwargs: dict[str, Any] = {}

        async def capture_discuss(**kwargs: Any) -> tuple:
            captured_kwargs.update(kwargs)
            return ([AIMessage(content="ok")], 1, 100)

        stage = DressStage(project_path=tmp_path)
        reject_gate = AsyncMock()
        reject_gate.on_phase_complete = AsyncMock(return_value="reject")
        stage.gate = reject_gate

        with (
            patch(
                "questfoundry.pipeline.stages.dress.run_discuss_phase",
                side_effect=capture_discuss,
            ),
            patch(
                "questfoundry.pipeline.stages.dress.summarize_discussion",
                new_callable=AsyncMock,
                return_value=("brief", 100),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.serialize_to_artifact",
                new_callable=AsyncMock,
                return_value=(mock_phase0_output, 100),
            ),
        ):
            await stage.execute(MagicMock(), "Make it look like Studio Ghibli")

        assert captured_kwargs["user_prompt"] == "Make it look like Studio Ghibli"
        assert captured_kwargs["stage_name"] == "dress"


# ---------------------------------------------------------------------------
# Phase stubs (3-4 remain)
# ---------------------------------------------------------------------------


class TestPhaseStubs:
    @pytest.mark.asyncio()
    async def test_phase3_not_implemented(self) -> None:
        stage = DressStage()
        with pytest.raises(NotImplementedError, match="PR 6"):
            await stage._phase_3_review(Graph(), MagicMock())

    @pytest.mark.asyncio()
    async def test_phase4_not_implemented(self) -> None:
        stage = DressStage()
        with pytest.raises(NotImplementedError, match="PR 6"):
            await stage._phase_4_generate(Graph(), MagicMock())


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_checkpoint_saved(self, tmp_path: Path) -> None:
        stage = DressStage()
        g = Graph()
        stage._save_checkpoint(g, tmp_path, "art_direction")

        path = tmp_path / "snapshots" / "dress-pre-art_direction.json"
        assert path.exists()

    def test_checkpoint_loaded(self, tmp_path: Path) -> None:
        stage = DressStage()
        g = Graph()
        g.create_node("test::node", {"type": "test"})
        stage._save_checkpoint(g, tmp_path, "briefs")

        loaded = stage._load_checkpoint(tmp_path, "briefs")
        assert loaded.get_node("test::node") is not None


# ---------------------------------------------------------------------------
# Artifact extraction
# ---------------------------------------------------------------------------


class TestExtractArtifact:
    def test_extracts_dress_nodes(self) -> None:
        g = Graph()
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "entity_visual::hero",
            {"type": "entity_visual", "description": "tall"},
        )

        stage = DressStage()
        artifact = stage._extract_artifact(g)

        assert artifact["art_direction"]["style"] == "ink"
        assert "entity_visual::hero" in artifact["entity_visuals"]

    def test_empty_graph(self) -> None:
        stage = DressStage()
        artifact = stage._extract_artifact(Graph())
        assert artifact["art_direction"] == {}
        assert artifact["entity_visuals"] == {}


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


@pytest.fixture()
def scored_graph() -> Graph:
    """Graph with arcs, beats, and passages for priority scoring tests."""
    g = Graph()
    # Spine arc with 3 beats
    g.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "sequence": ["beat::opening", "beat::climax", "beat::ending"],
        },
    )
    g.create_node("beat::opening", {"type": "beat", "scene_type": "establishing"})
    g.create_node("beat::climax", {"type": "beat", "scene_type": "climax"})
    g.create_node("beat::ending", {"type": "beat", "scene_type": "resolution"})
    # Branch arc
    g.create_node(
        "arc::branch1",
        {"type": "arc", "arc_type": "branch", "sequence": ["beat::side"]},
    )
    g.create_node("beat::side", {"type": "beat", "scene_type": "transition"})
    # Passages
    g.create_node(
        "passage::opening",
        {
            "type": "passage",
            "from_beat": "beat::opening",
            "prose": "The story begins.",
            "entities": ["entity::castle"],
        },
    )
    g.create_node(
        "passage::climax",
        {"type": "passage", "from_beat": "beat::climax", "prose": "The battle rages."},
    )
    g.create_node(
        "passage::ending",
        {"type": "passage", "from_beat": "beat::ending", "prose": "Peace returns."},
    )
    g.create_node(
        "passage::side",
        {"type": "passage", "from_beat": "beat::side", "prose": "Meanwhile..."},
    )
    # Location entity
    g.create_node("entity::castle", {"type": "entity", "entity_type": "location"})
    return g


class TestPriorityScoring:
    def test_spine_opening_with_location(self, scored_graph: Graph) -> None:
        """Spine opening + location = high score."""
        score = compute_structural_score(scored_graph, "passage::opening")
        # spine(+3) + opening(+2) + location(+1) = 6
        assert score == 6

    def test_spine_climax(self, scored_graph: Graph) -> None:
        """Spine climax = high score."""
        score = compute_structural_score(scored_graph, "passage::climax")
        # spine(+3) + climax(+2) = 5
        assert score == 5

    def test_spine_ending(self, scored_graph: Graph) -> None:
        score = compute_structural_score(scored_graph, "passage::ending")
        # spine(+3) + ending(+2) = 5
        assert score == 5

    def test_branch_transition(self, scored_graph: Graph) -> None:
        """Branch transition = moderate score (single-beat arc is both opening and ending)."""
        score = compute_structural_score(scored_graph, "passage::side")
        # transition(-1) + opening(+2) + ending(+2) = 3  (sole beat in branch arc)
        assert score == 3

    def test_nonexistent_passage(self, scored_graph: Graph) -> None:
        assert compute_structural_score(scored_graph, "passage::nope") == 0


class TestMapScoreToPriority:
    def test_high_score_must_have(self) -> None:
        assert map_score_to_priority(5) == 1
        assert map_score_to_priority(8) == 1

    def test_medium_score_important(self) -> None:
        assert map_score_to_priority(3) == 2
        assert map_score_to_priority(4) == 2

    def test_low_score_nice_to_have(self) -> None:
        assert map_score_to_priority(1) == 3
        assert map_score_to_priority(2) == 3

    def test_zero_or_negative_skip(self) -> None:
        assert map_score_to_priority(0) == 0
        assert map_score_to_priority(-1) == 0


# ---------------------------------------------------------------------------
# Phase 1: Illustration Briefs
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_brief_output() -> DressPhase1Output:
    return DressPhase1Output(
        brief=IllustrationBrief(
            priority=2,
            category="scene",
            subject="Scholar arrives at the ancient bridge",
            entities=["protagonist"],
            composition="Wide establishing shot",
            mood="foreboding",
            caption="The bridge loomed through the mist.",
        ),
        llm_adjustment=1,  # Enough to push base_score=0 to priority 3
    )


class TestPhase1Briefs:
    @pytest.mark.asyncio()
    async def test_creates_brief_nodes(self, mock_brief_output: DressPhase1Output) -> None:
        """Phase 1 creates illustration_brief nodes for passages with prose."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "palette": ["grey"]},
        )
        g.create_node(
            "passage::opening",
            {"type": "passage", "raw_id": "opening", "prose": "The wind howled."},
        )

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(mock_brief_output, 1, 100),
        ):
            result = await stage._phase_1_briefs(g, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert g.get_node("illustration_brief::opening") is not None

    @pytest.mark.asyncio()
    async def test_skips_passages_without_prose(self) -> None:
        """Passages without prose should be skipped."""
        g = Graph()
        g.create_node("passage::empty", {"type": "passage", "raw_id": "empty"})

        stage = DressStage()
        result = await stage._phase_1_briefs(g, MagicMock())

        assert result.status == "completed"
        assert "skipped" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio()
    async def test_no_passages(self) -> None:
        g = Graph()
        stage = DressStage()
        result = await stage._phase_1_briefs(g, MagicMock())
        assert result.detail == "no passages"

    @pytest.mark.asyncio()
    async def test_low_priority_skipped(self) -> None:
        """Brief with very low combined score is skipped."""
        g = Graph()
        g.create_node(
            "passage::boring", {"type": "passage", "raw_id": "boring", "prose": "Nothing happened."}
        )

        low_output = DressPhase1Output(
            brief=IllustrationBrief(
                priority=3,
                category="scene",
                subject="Nothing",
                composition="Static",
                mood="flat",
                caption="...",
            ),
            llm_adjustment=-2,
        )

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(low_output, 1, 50),
        ):
            result = await stage._phase_1_briefs(g, MagicMock())

        # base_score=0, llm_adj=-2 → total=-2 → priority=0 → skipped
        assert g.get_node("illustration_brief::boring") is None
        assert "skipped" in result.detail


# ---------------------------------------------------------------------------
# Phase 2: Codex Entries
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_codex_output() -> DressPhase2Output:
    return DressPhase2Output(
        entries=[
            CodexEntry(rank=1, visible_when=[], content="A young scholar of the old academy."),
            CodexEntry(
                rank=2,
                visible_when=["met_aldric"],
                content="The scholar secretly studies forbidden texts.",
            ),
        ]
    )


class TestPhase2Codex:
    @pytest.mark.asyncio()
    async def test_creates_codex_nodes(self, mock_codex_output: DressPhase2Output) -> None:
        """Phase 2 creates codex nodes for each entity."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {
                "type": "entity",
                "raw_id": "protagonist",
                "entity_type": "character",
                "concept": "Scholar",
            },
        )
        g.create_node(
            "codeword::met_aldric",
            {"type": "codeword", "raw_id": "met_aldric", "trigger": "Meets aldric"},
        )

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(mock_codex_output, 1, 150),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert g.get_node("codex::protagonist_rank1") is not None
        assert g.get_node("codex::protagonist_rank2") is not None

    @pytest.mark.asyncio()
    async def test_no_entities(self) -> None:
        g = Graph()
        stage = DressStage()
        result = await stage._phase_2_codex(g, MagicMock())
        assert result.detail == "no entities"

    @pytest.mark.asyncio()
    async def test_logs_validation_warnings(self, mock_codex_output: DressPhase2Output) -> None:
        """Codex validation warnings are logged but don't fail the phase."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        # No codewords defined — met_aldric in visible_when will trigger warning

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(mock_codex_output, 1, 150),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        assert g.get_node("codex::protagonist_rank1") is not None
