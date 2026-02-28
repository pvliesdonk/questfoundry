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
    BatchedBriefItem,
    BatchedBriefOutput,
    BatchedCodexItem,
    BatchedCodexOutput,
    CodexEntry,
    DressPhase0Output,
    DressPhase1Output,
    DressPhaseResult,
    EntityVisualWithId,
    IllustrationBrief,
)
from questfoundry.pipeline.stages.dress import (
    DressStage,
    DressStageError,
    _apply_image_budget,
    _create_cover_brief,
    _filter_by_priority,
    assemble_image_prompt,
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
    g.save(tmp_path / "graph.db")
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

    @pytest.mark.asyncio()
    async def test_execute_sets_project_path_from_param(self, tmp_path: Path) -> None:
        """Singleton stage should persist a resolved project_path for all phases."""
        g = Graph()
        g.set_last_stage("fill")
        g.save(tmp_path / "graph.db")

        stage = DressStage()

        async def _phase_noop(_graph: Graph, _model: Any) -> DressPhaseResult:
            assert stage.project_path == tmp_path
            return DressPhaseResult(phase="noop", status="completed", detail="ok")

        with patch.object(stage, "_phase_order", return_value=[(_phase_noop, "noop")]):
            await stage.execute(MagicMock(), "test", project_path=tmp_path)


class TestDressStagePrerequisites:
    @pytest.mark.asyncio()
    async def test_rejects_without_fill(self, tmp_path: Path) -> None:
        g = Graph()
        g.set_last_stage("grow")
        g.save(tmp_path / "graph.db")

        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="FILL"):
            await stage.execute(MagicMock(), "test")

    @pytest.mark.asyncio()
    async def test_accepts_dress_rerun(self, tmp_path: Path) -> None:
        """DRESS can re-run on a graph already at dress stage."""
        # Pre-DRESS snapshot (FILL-completed state)
        pre_dress = Graph()
        pre_dress.set_last_stage("fill")
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        pre_dress.save(snapshot_dir / "pre-dress.db")

        g = Graph()
        g.set_last_stage("dress")
        g.save(tmp_path / "graph.db")

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
    async def test_resume_requires_fill_completed(self, tmp_path: Path) -> None:
        g = Graph()
        g.set_last_stage("seed")
        g.save(tmp_path / "graph.db")

        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="DRESS requires completed FILL"):
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
                entities=[],
                composition="Wide",
                mood="test",
                style_overrides="",
                negative="",
                caption="test",
            ),
            llm_adjustment=0,
        )
        mock_codex_out = BatchedCodexOutput(
            entities=[
                BatchedCodexItem(
                    entity_id="entity::protagonist",
                    entries=[CodexEntry(title="Test", rank=1, visible_when=[], content="Base.")],
                ),
                BatchedCodexItem(
                    entity_id="entity::aldric",
                    entries=[CodexEntry(title="Test", rank=1, visible_when=[], content="Base.")],
                ),
            ]
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
                # Phase 1: 1 passage (per-passage call).
                # Phase 2: 1 batch with both entities (batch size 4).
                side_effect=[
                    (mock_brief_output, 1, 50),  # Phase 1: opening passage
                    (mock_codex_out, 1, 50),  # Phase 2: batch of 2 entities
                ],
            ),
        ):
            await stage.execute(MagicMock(), "Establish art direction")

        # Verify graph was updated (final graph has all phase results)
        graph = Graph.load(tmp_path)
        assert graph.get_node("art_direction::main") is not None
        assert graph.get_node("entity_visual::protagonist") is not None
        assert graph.get_node("entity_visual::aldric") is not None
        assert graph.get_last_stage() == "dress"

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
# Phase 3: Review Gate
# ---------------------------------------------------------------------------


class TestPhase3Review:
    @pytest.mark.asyncio()
    async def test_selects_all_briefs(self) -> None:
        """Auto-approve mode selects all briefs sorted by priority."""
        g = Graph()
        g.create_node(
            "illustration_brief::opening",
            {"type": "illustration_brief", "priority": 2, "subject": "Opening scene"},
        )
        g.create_node(
            "illustration_brief::climax",
            {"type": "illustration_brief", "priority": 1, "subject": "Climax"},
        )

        stage = DressStage()
        result = await stage._phase_3_review(g, MagicMock())

        assert result.status == "completed"
        assert "2 of 2" in result.detail

        selection = g.get_node("dress_meta::selection")
        assert selection is not None
        # Should be sorted by priority (1 first)
        assert selection["selected_briefs"][0] == "illustration_brief::climax"

    @pytest.mark.asyncio()
    async def test_no_briefs(self) -> None:
        g = Graph()
        stage = DressStage()
        result = await stage._phase_3_review(g, MagicMock())
        assert result.detail == "no briefs to review"


# ---------------------------------------------------------------------------
# Phase 4: Image Generation
# ---------------------------------------------------------------------------


class TestPhase4Generate:
    @pytest.mark.asyncio()
    async def test_generates_images(self, tmp_path: Path) -> None:
        """Phase 4 generates images and creates illustration nodes."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "aspect_ratio": "16:9"},
        )
        g.create_node(
            "illustration_brief::opening",
            {
                "type": "illustration_brief",
                "subject": "Opening scene",
                "composition": "Wide shot",
                "mood": "foreboding",
                "caption": "The wind howled.",
                "category": "scene",
                "entities": [],
            },
        )
        g.create_node("passage::opening", {"type": "passage"})
        g.add_edge("targets", "illustration_brief::opening", "passage::opening")
        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": ["illustration_brief::opening"],
                "total_briefs": 1,
            },
        )

        mock_result = MagicMock()
        mock_result.image_data = b"fake_png_data"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "high"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(project_path=tmp_path, image_provider="openai/gpt-image-1")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert result.status == "completed"
        assert "1 images generated" in result.detail
        assert g.get_node("illustration::opening") is not None

    @pytest.mark.asyncio()
    async def test_writes_graph_after_each_image(self, tmp_path: Path) -> None:
        """Phase 4 writes illustration nodes to graph.db incrementally."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "aspect_ratio": "16:9"},
        )

        for pid in ("a", "b"):
            g.create_node(
                f"illustration_brief::{pid}",
                {
                    "type": "illustration_brief",
                    "subject": f"Scene {pid}",
                    "composition": "Wide shot",
                    "mood": "foreboding",
                    "caption": "Caption",
                    "category": "scene",
                    "entities": [],
                },
            )
            g.create_node(f"passage::{pid}", {"type": "passage"})
            g.add_edge("targets", f"illustration_brief::{pid}", f"passage::{pid}")

        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": ["illustration_brief::a", "illustration_brief::b"],
                "total_briefs": 2,
            },
        )

        mock_result_a = MagicMock()
        mock_result_a.image_data = b"fake_png_a"
        mock_result_a.content_type = "image/png"
        mock_result_a.provider_metadata = {"quality": "high"}

        mock_result_b = MagicMock()
        mock_result_b.image_data = b"fake_png_b"
        mock_result_b.content_type = "image/png"
        mock_result_b.provider_metadata = {"quality": "high"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(side_effect=[mock_result_a, mock_result_b])

        stage = DressStage(project_path=tmp_path, image_provider="openai/test")

        with (
            patch(
                "questfoundry.pipeline.stages.dress.create_image_provider",
                return_value=mock_provider,
            ),
            patch.object(g, "save", wraps=g.save) as save_mock,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert result.status == "completed"
        assert save_mock.call_count == 2
        for call in save_mock.call_args_list:
            assert call.args[0] == tmp_path / "graph.db"

    @pytest.mark.asyncio()
    async def test_no_provider_skips(self) -> None:
        """Phase 4 skips gracefully when no image provider configured."""
        g = Graph()
        stage = DressStage()
        result = await stage._phase_4_generate(g, MagicMock())
        assert "no image provider" in result.detail

    @pytest.mark.asyncio()
    async def test_no_selection_skips(self) -> None:
        stage = DressStage(image_provider="openai/gpt-image-1")
        g = Graph()
        result = await stage._phase_4_generate(g, MagicMock())
        assert "no selection metadata" in result.detail

    @pytest.mark.asyncio()
    async def test_provider_error_continues(self, tmp_path: Path) -> None:
        """ImageProviderError on one brief doesn't stop others."""
        from questfoundry.providers.image import ImageProviderError

        g = Graph()
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "illustration_brief::fail",
            {"type": "illustration_brief", "subject": "Fail", "entities": []},
        )
        g.upsert_node(
            "dress_meta::selection",
            {"type": "dress_meta", "selected_briefs": ["illustration_brief::fail"]},
        )

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(
            side_effect=ImageProviderError("openai", "content policy")
        )

        stage = DressStage(project_path=tmp_path, image_provider="openai/test")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert "0 images generated, 1 failed" in result.detail


class TestParseAspectRatio:
    """Tests for _parse_aspect_ratio helper."""

    def test_clean_ratio(self) -> None:
        from questfoundry.pipeline.stages.dress import _parse_aspect_ratio

        assert _parse_aspect_ratio("16:9") == "16:9"

    def test_verbose_llm_output(self) -> None:
        from questfoundry.pipeline.stages.dress import _parse_aspect_ratio

        raw = "16:9 (story panels), 4:5 (character plates), 21:9 (chases)"
        assert _parse_aspect_ratio(raw) == "16:9"

    def test_unsupported_ratio_skipped(self) -> None:
        from questfoundry.pipeline.stages.dress import _parse_aspect_ratio

        # 4:5 is not supported, should skip to 9:16 which is
        assert _parse_aspect_ratio("4:5, 9:16") == "9:16"

    def test_no_valid_ratio_falls_back(self) -> None:
        from questfoundry.pipeline.stages.dress import _parse_aspect_ratio

        assert _parse_aspect_ratio("widescreen cinematic") == "16:9"

    def test_all_valid_ratios(self) -> None:
        from questfoundry.pipeline.stages.dress import _parse_aspect_ratio

        for ratio in ("1:1", "16:9", "9:16", "3:2", "2:3"):
            assert _parse_aspect_ratio(ratio) == ratio


class TestPhase4SkipExisting:
    """Tests for skip-existing-illustrations behavior."""

    def _make_graph_with_illustration(self) -> Graph:
        """Create a graph where one brief already has an illustration."""
        g = Graph()
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "illustration_brief::opening",
            {
                "type": "illustration_brief",
                "subject": "Opening scene",
                "composition": "Wide",
                "mood": "ominous",
                "entities": [],
            },
        )
        g.create_node("passage::opening", {"type": "passage"})
        g.add_edge("targets", "illustration_brief::opening", "passage::opening")
        # Existing illustration for this brief
        g.upsert_node(
            "illustration::opening",
            {"type": "illustration", "asset": "images/old.png", "quality": "low"},
        )
        g.add_edge("Depicts", "illustration::opening", "passage::opening")
        g.add_edge("from_brief", "illustration::opening", "illustration_brief::opening")
        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": ["illustration_brief::opening"],
            },
        )
        return g

    @pytest.mark.asyncio()
    async def test_skips_existing_illustrations(self, tmp_path: Path) -> None:
        g = self._make_graph_with_illustration()

        mock_provider = AsyncMock(spec=["generate"])
        stage = DressStage(project_path=tmp_path, image_provider="a1111/test")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert "already have illustrations" in result.detail
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio()
    async def test_force_regenerates_existing(self, tmp_path: Path) -> None:
        g = self._make_graph_with_illustration()

        mock_result = MagicMock()
        mock_result.image_data = b"new_png"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "high"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(project_path=tmp_path, image_provider="a1111/test")
        stage._force_regenerate = True

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert "1 images generated" in result.detail
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_partial_skip(self, tmp_path: Path) -> None:
        """One brief has illustration, another doesn't — only new one generated."""
        g = self._make_graph_with_illustration()

        # Add a second brief WITHOUT an illustration
        g.create_node(
            "illustration_brief::climax",
            {
                "type": "illustration_brief",
                "subject": "Climax scene",
                "composition": "Close-up",
                "mood": "intense",
                "entities": [],
            },
        )
        g.create_node("passage::climax", {"type": "passage"})
        g.add_edge("targets", "illustration_brief::climax", "passage::climax")
        selection = g.get_node("dress_meta::selection")
        selection["selected_briefs"].append("illustration_brief::climax")

        mock_result = MagicMock()
        mock_result.image_data = b"new_png"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "high"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(project_path=tmp_path, image_provider="a1111/test")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage._phase_4_generate(g, MagicMock())

        assert "1 images generated" in result.detail
        mock_provider.generate.assert_called_once()


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

    def test_extracts_codex_entries(self) -> None:
        g = Graph()
        g.create_node(
            "codex::hero_rank1",
            {"type": "codex_entry", "rank": 1, "content": "A tall warrior"},
        )
        g.create_node(
            "codex::hero_rank2",
            {"type": "codex_entry", "rank": 2, "content": "Scarred from battle"},
        )

        stage = DressStage()
        artifact = stage._extract_artifact(g)

        assert len(artifact["codex_entries"]) == 2
        assert "codex::hero_rank1" in artifact["codex_entries"]
        assert artifact["codex_entries"]["codex::hero_rank1"]["content"] == "A tall warrior"

    def test_empty_graph(self) -> None:
        stage = DressStage()
        artifact = stage._extract_artifact(Graph())
        assert artifact["art_direction"] == {}
        assert artifact["entity_visuals"] == {}
        assert artifact["codex_entries"] == {}


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


def _make_brief_output(passage_id: str = "opening", llm_adjustment: int = 1) -> BatchedBriefOutput:
    """Helper to build a BatchedBriefOutput for a single passage."""
    return BatchedBriefOutput(
        briefs=[
            BatchedBriefItem(
                passage_id=passage_id,
                brief=IllustrationBrief(
                    priority=2,
                    category="scene",
                    subject="Scholar arrives at the ancient bridge",
                    entities=["protagonist"],
                    composition="Wide establishing shot",
                    mood="foreboding",
                    style_overrides="",
                    negative="",
                    caption="The bridge loomed through the mist.",
                ),
                llm_adjustment=llm_adjustment,
            )
        ]
    )


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
            style_overrides="",
            negative="",
            caption="The bridge loomed through the mist.",
        ),
        llm_adjustment=1,
    )


class TestPhase1Briefs:
    @pytest.mark.asyncio()
    async def test_creates_brief_nodes(self) -> None:
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
            return_value=(_make_brief_output("opening"), 1, 100),
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
            "passage::boring",
            {"type": "passage", "raw_id": "boring", "prose": "Nothing happened."},
        )

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(_make_brief_output("boring", llm_adjustment=-2), 1, 50),
        ):
            result = await stage._phase_1_briefs(g, MagicMock())

        # base_score=0, llm_adj=-2 → total=-2 → priority=0 → skipped
        assert g.get_node("illustration_brief::boring") is None
        assert "skipped" in result.detail

    @pytest.mark.asyncio()
    async def test_batching_groups_passages(self) -> None:
        """Multiple passages are grouped into batches of 5."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "palette": ["grey"]},
        )
        # Create 7 passages — should produce 2 batches (5 + 2)
        passage_ids = []
        for i in range(7):
            pid = f"p{i}"
            g.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "prose": f"Prose for passage {i}."},
            )
            passage_ids.append(pid)

        calls: list[dict[str, Any]] = []

        async def _mock_llm_call(
            _model: Any,
            _template: str,
            context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple[BatchedBriefOutput, int, int]:
            calls.append(context)
            # Identify which passages are in this batch by matching known IDs
            batch_text = context["passages_batch"]
            batch_pids = [pid for pid in passage_ids if f"### {pid}\n" in batch_text]
            briefs = []
            for raw_id in batch_pids:
                briefs.append(
                    BatchedBriefItem(
                        passage_id=f"passage::{raw_id}",
                        brief=IllustrationBrief(
                            priority=2,
                            category="scene",
                            subject=f"Subject {raw_id}",
                            entities=[],
                            composition=f"Comp {raw_id}",
                            mood="neutral",
                            style_overrides="",
                            negative="",
                            caption=f"Caption {raw_id}",
                        ),
                        llm_adjustment=1,
                    )
                )
            return BatchedBriefOutput(briefs=briefs), 1, 100

        stage = DressStage()
        with patch.object(stage, "_dress_llm_call", side_effect=_mock_llm_call):
            result = await stage._phase_1_briefs(g, MagicMock())

        assert len(calls) == 2  # 5 + 2
        assert result.status == "completed"
        assert "7 briefs created" in result.detail

    @pytest.mark.asyncio()
    async def test_min_priority_filters_low_priority_passages(self) -> None:
        """Passages with structural priority > min_priority are skipped before LLM calls."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "palette": ["grey"]},
        )
        # High-priority passage: spine arc opening + climax = high score
        g.create_node("arc::spine", {"type": "arc", "arc_type": "spine", "sequence": ["beat::a"]})
        g.create_node("beat::a", {"type": "beat", "raw_id": "a", "scene_type": "climax"})
        g.create_node(
            "passage::important",
            {
                "type": "passage",
                "raw_id": "important",
                "prose": "Epic moment.",
                "from_beat": "beat::a",
            },
        )
        # Low-priority passage: no beat, no arc = score 0 → priority 0 (or 3 if score=1+)
        g.create_node(
            "passage::filler",
            {"type": "passage", "raw_id": "filler", "prose": "Nothing notable."},
        )

        stage = DressStage()
        stage._min_priority = 2  # Only generate briefs for priority 1-2

        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(_make_brief_output("important"), 1, 100),
        ):
            result = await stage._phase_1_briefs(g, MagicMock())

        assert result.status == "completed"
        assert g.get_node("illustration_brief::important") is not None
        # Filler passage has score 0, best_possible=map(2)=3, filtered (3 > 2)
        assert g.get_node("illustration_brief::filler") is None

    @pytest.mark.asyncio()
    async def test_min_priority_stores_config_in_graph(self) -> None:
        """Phase 1 stores dress_min_priority in graph metadata."""
        g = Graph()
        g.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "prose": "Some prose."},
        )

        stage = DressStage()
        stage._min_priority = 2

        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(_make_brief_output("p1"), 1, 50),
        ):
            await stage._phase_1_briefs(g, MagicMock())

        config = g.get_node("dress_meta::brief_config")
        assert config is not None
        assert config["min_priority"] == 2

    @pytest.mark.asyncio()
    async def test_min_priority_3_generates_all(self) -> None:
        """min_priority=3 does not filter any passages (current default behavior)."""
        g = Graph()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "palette": ["grey"]},
        )
        # Low-score passage (score=0 → best_possible=map(0+2)=3, passes threshold)
        g.create_node(
            "passage::low",
            {"type": "passage", "raw_id": "low", "prose": "Filler."},
        )

        calls: list[dict[str, Any]] = []

        async def _mock_llm_call(
            _model: Any,
            _template: str,
            context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple[BatchedBriefOutput, int, int]:
            calls.append(context)
            return _make_brief_output("low", llm_adjustment=1), 1, 50

        stage = DressStage()
        stage._min_priority = 3  # Generate all

        with patch.object(stage, "_dress_llm_call", side_effect=_mock_llm_call):
            result = await stage._phase_1_briefs(g, MagicMock())

        # With min_priority=3, even score-0 passages pass (best_possible=3)
        assert len(calls) == 1
        assert result.status == "completed"


class TestPriorityMismatchWarning:
    """Test priority mismatch detection in generate-images."""

    @pytest.mark.asyncio()
    async def test_warns_on_priority_mismatch(self, tmp_path: Path) -> None:
        """run_generate_only warns when min_priority > dress_min_priority."""
        g = Graph()
        g.set_last_stage("dress")
        g.upsert_node(
            "dress_meta::brief_config",
            {"type": "dress_meta", "min_priority": 2},
        )
        g.upsert_node(
            "dress_meta::selection",
            {"type": "dress_meta", "selected_briefs": [], "total_briefs": 0},
        )
        g.save(tmp_path / "graph.db")

        stage = DressStage(project_path=tmp_path)

        with patch.object(stage, "_phase_4_generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = DressPhaseResult(
                phase="generate", status="completed", detail="0 images"
            )
            result = await stage.run_generate_only(tmp_path, min_priority=3, image_budget=0)

        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Phase 2: Codex Entries
# ---------------------------------------------------------------------------


def _make_codex_output(entity_id: str) -> BatchedCodexOutput:
    """Helper to create a BatchedCodexOutput for a single entity."""
    return BatchedCodexOutput(
        entities=[
            BatchedCodexItem(
                entity_id=entity_id,
                entries=[
                    CodexEntry(
                        title="Aldric",
                        rank=1,
                        visible_when=[],
                        content="A young scholar of the old academy.",
                    ),
                    CodexEntry(
                        title="Aldric's Secret",
                        rank=2,
                        visible_when=["met_aldric"],
                        content="The scholar secretly studies forbidden texts.",
                    ),
                ],
            )
        ]
    )


class TestPhase2Codex:
    @pytest.mark.asyncio()
    async def test_creates_codex_nodes(self) -> None:
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
        mock_output = _make_codex_output("entity::protagonist")
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(mock_output, 1, 150),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert g.get_node("codex::protagonist_rank1") is not None
        assert g.get_node("codex::protagonist_rank2") is not None

    @pytest.mark.asyncio()
    async def test_skip_codex_flag(self) -> None:
        """--no-codex flag skips Phase 2 entirely."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        stage = DressStage()
        stage._skip_codex = True
        result = await stage._phase_2_codex(g, MagicMock())
        assert result.status == "skipped"
        assert result.detail == "--no-codex"
        assert result.llm_calls == 0

    @pytest.mark.asyncio()
    async def test_no_entities(self) -> None:
        g = Graph()
        stage = DressStage()
        result = await stage._phase_2_codex(g, MagicMock())
        assert result.detail == "no entities"

    @pytest.mark.asyncio()
    async def test_batching_groups_entities(self) -> None:
        """5 entities with batch size 4 produce 2 LLM calls."""
        g = Graph()
        for i in range(5):
            g.create_node(
                f"entity::e{i}",
                {"type": "entity", "raw_id": f"e{i}", "entity_type": "character"},
            )

        def _make_batch_output(chunk: list[str]) -> BatchedCodexOutput:
            return BatchedCodexOutput(
                entities=[
                    BatchedCodexItem(
                        entity_id=eid,
                        entries=[
                            CodexEntry(title=f"E{i}", rank=1, visible_when=[], content="Info.")
                        ],
                    )
                    for i, eid in enumerate(chunk)
                ]
            )

        calls: list[dict[str, Any]] = []

        async def _mock_llm_call(
            _model: Any,
            _template: str,
            _context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            calls.append(_context)
            # Parse entity IDs from batch context (each starts with "## Entity: <raw_id>")
            import re

            raw_ids = re.findall(r"## Entity: (\S+)", _context["entities_batch"])
            eids = [f"entity::{raw_id}" for raw_id in raw_ids]
            return (_make_batch_output(eids), 1, 100)

        stage = DressStage()
        with patch.object(stage, "_dress_llm_call", side_effect=_mock_llm_call):
            result = await stage._phase_2_codex(g, MagicMock())

        assert len(calls) == 2  # 4 + 1
        assert result.llm_calls == 2
        # All 5 entities should have codex entries
        for i in range(5):
            assert g.get_node(f"codex::e{i}_rank1") is not None

    @pytest.mark.asyncio()
    async def test_invalid_entity_id_skipped(self) -> None:
        """LLM returning an entity_id not in the batch is skipped gracefully."""
        g = Graph()
        g.create_node(
            "entity::real",
            {"type": "entity", "raw_id": "real", "entity_type": "character"},
        )

        wrong_output = BatchedCodexOutput(
            entities=[
                BatchedCodexItem(
                    entity_id="entity::hallucinated",
                    entries=[CodexEntry(title="X", rank=1, visible_when=[], content="X")],
                ),
                BatchedCodexItem(
                    entity_id="entity::real",
                    entries=[CodexEntry(title="Real", rank=1, visible_when=[], content="Info.")],
                ),
            ]
        )

        stage = DressStage()
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(wrong_output, 1, 100),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # Valid entity should have codex, hallucinated one should not
        assert g.get_node("codex::real_rank1") is not None
        codex_nodes = g.get_nodes_by_type("codex_entry")
        # Only 1 entry (for "real"), not 2
        assert len(codex_nodes) == 1

    @pytest.mark.asyncio()
    async def test_logs_validation_warnings(self) -> None:
        """Codex validation warnings are logged but don't fail the phase."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        # No codewords defined — met_aldric in visible_when will trigger warning

        stage = DressStage()
        mock_output = _make_codex_output("entity::protagonist")
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            return_value=(mock_output, 1, 150),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        assert g.get_node("codex::protagonist_rank1") is not None


# ---------------------------------------------------------------------------
# AssetManager
# ---------------------------------------------------------------------------


class TestAssetManager:
    def test_store_creates_file(self, tmp_path: Path) -> None:
        from questfoundry.artifacts.assets import AssetManager

        mgr = AssetManager(tmp_path)
        path = mgr.store(b"fake_png_data", "image/png")

        assert path.startswith("assets/")
        assert path.endswith(".png")
        assert (tmp_path / path).exists()
        assert (tmp_path / path).read_bytes() == b"fake_png_data"

    def test_deduplication(self, tmp_path: Path) -> None:
        from questfoundry.artifacts.assets import AssetManager

        mgr = AssetManager(tmp_path)
        path1 = mgr.store(b"same_data", "image/png")
        path2 = mgr.store(b"same_data", "image/png")

        assert path1 == path2

    def test_webp_extension(self, tmp_path: Path) -> None:
        from questfoundry.artifacts.assets import AssetManager

        mgr = AssetManager(tmp_path)
        path = mgr.store(b"webp_data", "image/webp")
        assert path.endswith(".webp")


# ---------------------------------------------------------------------------
# Image prompt assembly
# ---------------------------------------------------------------------------


class TestAssembleImagePrompt:
    def test_combines_art_direction_and_brief(self) -> None:
        g = Graph()
        g.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "watercolor",
                "medium": "traditional paper",
                "palette": ["deep indigo", "gold"],
                "negative_defaults": "photorealism",
            },
        )

        brief = {
            "subject": "Scholar at the bridge",
            "composition": "Wide shot",
            "mood": "foreboding",
            "negative": "modern elements",
            "entities": [],
        }

        positive, negative = assemble_image_prompt(g, brief)

        assert "Scholar at the bridge" in positive
        assert "watercolor" in positive
        assert "deep indigo" in positive
        assert negative is not None
        assert "modern elements" in negative
        assert "photorealism" in negative

    def test_includes_entity_visuals(self) -> None:
        g = Graph()
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "entity_visual::hero",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "tall warrior, scarred face",
            },
        )

        brief = {"subject": "Battle scene", "entities": ["hero"]}
        positive, _negative = assemble_image_prompt(g, brief)

        assert "tall warrior, scarred face" in positive

    def test_no_art_direction(self) -> None:
        g = Graph()
        brief = {"subject": "A simple scene", "entities": []}
        positive, negative = assemble_image_prompt(g, brief)

        assert "A simple scene" in positive
        assert negative is None


# ---------------------------------------------------------------------------
# Budget Control
# ---------------------------------------------------------------------------


class TestApplyImageBudget:
    def _make_graph_with_briefs(self, priorities: dict[str, int]) -> Graph:
        """Create a graph with briefs at given priorities."""
        g = Graph()
        for bid, priority in priorities.items():
            g.create_node(bid, {"type": "illustration_brief", "priority": priority})
        return g

    def test_budget_selects_highest_priority(self) -> None:
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 2,
                "illustration_brief::b": 1,
                "illustration_brief::c": 3,
            }
        )
        result = _apply_image_budget(
            g,
            ["illustration_brief::a", "illustration_brief::b", "illustration_brief::c"],
            budget=1,
        )
        assert result == ["illustration_brief::b"]

    def test_budget_two_from_mixed(self) -> None:
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 3,
                "illustration_brief::b": 1,
                "illustration_brief::c": 2,
            }
        )
        result = _apply_image_budget(
            g,
            ["illustration_brief::a", "illustration_brief::b", "illustration_brief::c"],
            budget=2,
        )
        assert result == ["illustration_brief::b", "illustration_brief::c"]

    def test_budget_exceeds_total(self) -> None:
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 1,
                "illustration_brief::b": 2,
            }
        )
        result = _apply_image_budget(
            g,
            ["illustration_brief::a", "illustration_brief::b"],
            budget=10,
        )
        assert len(result) == 2

    def test_budget_stable_ordering(self) -> None:
        """Same priority briefs are ordered by ID for stability."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::z": 1,
                "illustration_brief::a": 1,
                "illustration_brief::m": 1,
            }
        )
        result = _apply_image_budget(
            g,
            ["illustration_brief::z", "illustration_brief::a", "illustration_brief::m"],
            budget=2,
        )
        # Should be alphabetical by ID since priorities are equal
        assert result == ["illustration_brief::a", "illustration_brief::m"]

    def test_missing_brief_defaults_to_low_priority(self) -> None:
        """Briefs not found in graph are treated as priority 3."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::real": 1,
            }
        )
        result = _apply_image_budget(
            g,
            ["illustration_brief::real", "illustration_brief::missing"],
            budget=1,
        )
        assert result == ["illustration_brief::real"]


class TestFilterByPriority:
    """Tests for _filter_by_priority helper."""

    def _make_graph_with_briefs(self, priorities: dict[str, int]) -> Graph:
        """Create a graph with briefs at given priorities."""
        g = Graph()
        for bid, priority in priorities.items():
            g.create_node(bid, {"type": "illustration_brief", "priority": priority})
        return g

    def test_keeps_high_priority(self) -> None:
        """priority_threshold=2 keeps priority 1 and 2, excludes 3."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 1,
                "illustration_brief::b": 2,
                "illustration_brief::c": 3,
            }
        )
        result = _filter_by_priority(
            g,
            ["illustration_brief::a", "illustration_brief::b", "illustration_brief::c"],
            priority_threshold=2,
        )
        assert result == ["illustration_brief::a", "illustration_brief::b"]

    def test_must_have_only(self) -> None:
        """priority_threshold=1 keeps only priority 1."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 1,
                "illustration_brief::b": 2,
                "illustration_brief::c": 1,
            }
        )
        result = _filter_by_priority(
            g,
            ["illustration_brief::a", "illustration_brief::b", "illustration_brief::c"],
            priority_threshold=1,
        )
        assert result == ["illustration_brief::a", "illustration_brief::c"]

    def test_all_priorities(self) -> None:
        """priority_threshold=3 keeps everything (no filtering)."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::a": 1,
                "illustration_brief::b": 3,
            }
        )
        all_ids = ["illustration_brief::a", "illustration_brief::b"]
        result = _filter_by_priority(g, all_ids, priority_threshold=3)
        assert result == all_ids

    def test_missing_node_defaults_to_low_priority(self) -> None:
        """Briefs not in graph default to priority 3 (excluded at priority_threshold=2)."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::real": 1,
            }
        )
        result = _filter_by_priority(
            g,
            ["illustration_brief::real", "illustration_brief::missing"],
            priority_threshold=2,
        )
        assert result == ["illustration_brief::real"]

    def test_preserves_input_order(self) -> None:
        """Filtered list preserves original ordering."""
        g = self._make_graph_with_briefs(
            {
                "illustration_brief::z": 1,
                "illustration_brief::a": 2,
                "illustration_brief::m": 1,
            }
        )
        result = _filter_by_priority(
            g,
            ["illustration_brief::z", "illustration_brief::a", "illustration_brief::m"],
            priority_threshold=1,
        )
        assert result == ["illustration_brief::z", "illustration_brief::m"]


# ---------------------------------------------------------------------------
# Cover Brief
# ---------------------------------------------------------------------------


class TestCreateCoverBrief:
    def test_creates_cover_brief_from_vision(self) -> None:
        g = Graph()
        g.create_node(
            "vision",
            {"type": "vision", "genre": "space opera", "tone": ["urgent"], "themes": ["fate"]},
        )
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "watercolor", "medium": "digital painting"},
        )

        result = _create_cover_brief(g)

        assert result is True
        brief = g.get_node("illustration_brief::cover")
        assert brief is not None
        assert brief["priority"] == 1
        assert brief["category"] == "cover"
        assert "space opera" in brief["subject"]
        assert "fate" in brief["subject"]

    def test_no_synthetic_passage_node(self) -> None:
        g = Graph()
        g.create_node("vision", {"type": "vision", "genre": "mystery"})

        _create_cover_brief(g)

        assert g.get_node("passage::cover") is None

    def test_no_targets_edge(self) -> None:
        g = Graph()
        g.create_node("vision", {"type": "vision", "genre": "fantasy"})

        _create_cover_brief(g)

        edges = g.get_edges(from_id="illustration_brief::cover", edge_type="targets")
        assert len(edges) == 0

    def test_cover_category(self) -> None:
        g = Graph()
        g.create_node("vision", {"type": "vision", "genre": "fantasy"})

        _create_cover_brief(g)

        brief = g.get_node("illustration_brief::cover")
        assert brief is not None
        assert brief["category"] == "cover"

    def test_returns_false_without_vision(self) -> None:
        g = Graph()
        result = _create_cover_brief(g)
        assert result is False
        assert g.get_node("illustration_brief::cover") is None

    def test_works_without_art_direction(self) -> None:
        g = Graph()
        g.create_node("vision", {"type": "vision", "genre": "thriller", "tone": ["tense"]})

        result = _create_cover_brief(g)

        assert result is True
        brief = g.get_node("illustration_brief::cover")
        assert brief is not None
        assert "thriller" in brief["subject"]

    def test_reads_story_title_from_voice_node(self) -> None:
        g = Graph()
        g.create_node("vision", {"type": "vision", "genre": "fantasy", "tone": ["epic"]})
        g.create_node(
            "voice::voice", {"type": "voice", "raw_id": "voice", "story_title": "The Hollow Crown"}
        )

        _create_cover_brief(g)

        brief = g.get_node("illustration_brief::cover")
        assert brief is not None
        assert "The Hollow Crown" in brief["subject"]


# ---------------------------------------------------------------------------
# Standalone generate-images (run_generate_only)
# ---------------------------------------------------------------------------


class TestRunGenerateOnly:
    @pytest.mark.asyncio()
    async def test_generates_and_saves_graph(self, tmp_path: Path) -> None:
        """run_generate_only generates images and saves updated graph."""
        g = Graph()
        g.set_last_stage("dress")
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "illustration_brief::opening",
            {
                "type": "illustration_brief",
                "subject": "Opening scene",
                "composition": "Wide",
                "mood": "ominous",
                "caption": "The wind howled.",
                "category": "scene",
                "entities": [],
            },
        )
        g.create_node("passage::opening", {"type": "passage"})
        g.add_edge("targets", "illustration_brief::opening", "passage::opening")
        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": ["illustration_brief::opening"],
                "total_briefs": 1,
            },
        )
        g.save(tmp_path / "graph.db")

        mock_result = MagicMock()
        mock_result.image_data = b"fake_png"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "placeholder"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(image_provider="placeholder")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage.run_generate_only(tmp_path)

        assert result.status == "completed"
        assert "1 images generated" in result.detail

        # Verify graph was saved with illustration node
        reloaded = Graph.load(tmp_path)
        assert reloaded.get_node("illustration::opening") is not None

    @pytest.mark.asyncio()
    async def test_raises_without_selection(self, tmp_path: Path) -> None:
        """run_generate_only raises if no brief selection exists."""
        g = Graph()
        g.set_last_stage("dress")
        g.save(tmp_path / "graph.db")

        stage = DressStage(image_provider="placeholder")
        with pytest.raises(DressStageError, match="No brief selection"):
            await stage.run_generate_only(tmp_path)

    @pytest.mark.asyncio()
    async def test_respects_image_budget(self, tmp_path: Path) -> None:
        """run_generate_only respects the image budget setting."""
        g = Graph()
        g.set_last_stage("dress")
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        for name, priority in [("a", 2), ("b", 1), ("c", 3)]:
            bid = f"illustration_brief::{name}"
            pid = f"passage::{name}"
            g.create_node(
                bid,
                {
                    "type": "illustration_brief",
                    "priority": priority,
                    "subject": name,
                    "caption": name,
                    "category": "scene",
                    "entities": [],
                },
            )
            g.create_node(pid, {"type": "passage"})
            g.add_edge("targets", bid, pid)
        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": [
                    "illustration_brief::a",
                    "illustration_brief::b",
                    "illustration_brief::c",
                ],
            },
        )
        g.save(tmp_path / "graph.db")

        mock_result = MagicMock()
        mock_result.image_data = b"fake"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "placeholder"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(image_provider="placeholder")

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            result = await stage.run_generate_only(tmp_path, image_budget=1)

        assert "1 images generated" in result.detail

    @pytest.mark.asyncio()
    async def test_calls_progress_callback(self, tmp_path: Path) -> None:
        """run_generate_only calls the on_phase_progress callback."""
        g = Graph()
        g.set_last_stage("dress")
        g.upsert_node(
            "dress_meta::selection",
            {"type": "dress_meta", "selected_briefs": []},
        )
        g.save(tmp_path / "graph.db")

        stage = DressStage(image_provider="placeholder")
        progress_calls: list[tuple[str, str, str | None]] = []

        def _on_progress(phase: str, status: str, detail: str | None) -> None:
            progress_calls.append((phase, status, detail))

        await stage.run_generate_only(tmp_path, on_phase_progress=_on_progress)

        assert len(progress_calls) == 1
        assert progress_calls[0][0] == "generate"
        assert progress_calls[0][1] == "completed"

    @pytest.mark.asyncio()
    async def test_progress_reports_distill_and_render(self, tmp_path: Path) -> None:
        """run_generate_only emits in_progress for distilling and rendering."""
        g = Graph()
        g.set_last_stage("dress")
        g.create_node("art_direction::main", {"type": "art_direction", "style": "ink"})
        g.create_node(
            "illustration_brief::opening",
            {
                "type": "illustration_brief",
                "subject": "Opening scene",
                "composition": "Wide",
                "mood": "ominous",
                "caption": "The wind howled.",
                "category": "scene",
                "entities": [],
            },
        )
        g.create_node("passage::opening", {"type": "passage"})
        g.add_edge("targets", "illustration_brief::opening", "passage::opening")
        g.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": ["illustration_brief::opening"],
                "total_briefs": 1,
            },
        )
        g.save(tmp_path / "graph.db")

        mock_result = MagicMock()
        mock_result.image_data = b"fake_png"
        mock_result.content_type = "image/png"
        mock_result.provider_metadata = {"quality": "placeholder"}

        mock_provider = AsyncMock(spec=["generate"])
        mock_provider.generate = AsyncMock(return_value=mock_result)

        stage = DressStage(image_provider="placeholder")
        progress_calls: list[tuple[str, str, str | None]] = []

        def _on_progress(phase: str, status: str, detail: str | None) -> None:
            progress_calls.append((phase, status, detail))

        with patch(
            "questfoundry.pipeline.stages.dress.create_image_provider",
            return_value=mock_provider,
        ):
            await stage.run_generate_only(tmp_path, on_phase_progress=_on_progress)

        # Should have: distill in_progress, sample render in_progress, final completed
        in_progress = [c for c in progress_calls if c[1] == "in_progress"]
        assert len(in_progress) == 2
        assert "Distilling 1 prompts" in (in_progress[0][2] or "")
        assert "Rendering sample" in (in_progress[1][2] or "")
        assert progress_calls[-1][1] == "completed"
