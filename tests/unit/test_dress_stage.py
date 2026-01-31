"""Tests for DRESS stage skeleton and Phase 0 (art direction)."""

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
    DressPhase0Output,
    EntityVisualWithId,
)
from questfoundry.pipeline.stages.dress import (
    DressStage,
    DressStageError,
    create_dress_stage,
    dress_stage,
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
    async def test_accepts_fill_stage(self, tmp_path: Path, dress_graph: Graph) -> None:  # noqa: ARG002
        """Verify prerequisite passes for fill (actual execution tested below)."""
        assert dress_graph.get_last_stage() == "fill"

    @pytest.mark.asyncio()
    async def test_accepts_dress_rerun(self, tmp_path: Path) -> None:
        """DRESS can re-run on a graph already at dress stage."""
        g = Graph()
        g.set_last_stage("dress")
        g.save(tmp_path / "graph.json")

        stage = DressStage(project_path=tmp_path)
        # Should not raise on prerequisite check
        # (will fail on Phase 0 since graph has no entities, but that's fine)
        with pytest.raises(Exception):  # noqa: B017
            await stage.execute(MagicMock(), "test")


class TestDressStageResume:
    @pytest.mark.asyncio()
    async def test_invalid_phase_raises(self, tmp_path: Path, dress_graph: Graph) -> None:  # noqa: ARG002
        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="Unknown phase"):
            await stage.execute(MagicMock(), "test", resume_from="nonexistent")

    @pytest.mark.asyncio()
    async def test_missing_checkpoint_raises(self, tmp_path: Path, dress_graph: Graph) -> None:  # noqa: ARG002
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
            # Phase 0 succeeds, Phase 1 raises NotImplementedError
            pytest.raises(NotImplementedError, match="PR 5"),
        ):
            await stage.execute(MagicMock(), "Establish art direction")

        # Verify graph was updated (checkpoint before Phase 1 has Phase 0 results)
        checkpoint = tmp_path / "snapshots" / "dress-pre-briefs.json"
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
# Phase stubs
# ---------------------------------------------------------------------------


class TestPhaseStubs:
    @pytest.mark.asyncio()
    async def test_phase1_not_implemented(self) -> None:
        stage = DressStage()
        with pytest.raises(NotImplementedError, match="PR 5"):
            await stage._phase_1_briefs(Graph(), MagicMock())

    @pytest.mark.asyncio()
    async def test_phase2_not_implemented(self) -> None:
        stage = DressStage()
        with pytest.raises(NotImplementedError, match="PR 5"):
            await stage._phase_2_codex(Graph(), MagicMock())

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
