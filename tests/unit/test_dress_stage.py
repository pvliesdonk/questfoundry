"""Tests for DRESS stage skeleton, Phase 0, Phase 1, and Phase 2."""

from __future__ import annotations

import re
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
    DressPhase2Output,
    DressPhaseResult,
    EntityVisualWithId,
    IllustrationBrief,
    SpoilerCheckResult,
    SpoilerLeak,
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


@pytest.fixture(autouse=True)
def _bypass_seam_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass DRESS's new FILL-output entry validator (#1347) and the
    DRESS-output exit validator (#1348) for all tests in this file.
    Test fixtures use minimal FILL graphs that don't satisfy the full
    contracts; the seam-validation integration is exercised in
    test_contract_chaining.py instead.
    """
    from questfoundry.graph import (
        dress_output_validation as _dov,
    )
    from questfoundry.graph import (
        fill_output_validation as _fov,
    )

    monkeypatch.setattr(_fov, "validate_fill_output", lambda _g: [])
    monkeypatch.setattr(_dov, "validate_dress_output", lambda _g: [])


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
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_dispatch_mock(
    *,
    brief_output: Any = None,
    codex_batch_output: BatchedCodexOutput | None = None,
    spoiler_result: SpoilerCheckResult | None = None,
    codex_retry_output: DressPhase2Output | None = None,
) -> Any:
    """Build a side_effect that dispatches by template name.

    Phase 2 now issues a per-entity spoiler-check call after every batch
    (R-3.6). Tests that previously enumerated a fixed call list as side
    effects break when the new call slots in. This helper routes by
    ``template_name`` so the test stays declarative.

    Default ``spoiler_result`` is "no leak" so existing tests keep
    passing without changes to their assertions.
    """
    safe = spoiler_result or SpoilerCheckResult(has_leak=False, leaks=[], reason="")

    async def _dispatch(
        _model: Any,
        template_name: str,
        _context: dict[str, Any],
        _schema: type,
        **_kwargs: Any,
    ) -> tuple:
        if template_name == "dress_codex_spoiler_check":
            return (safe, 1, 25)
        if template_name == "dress_codex_batch":
            assert codex_batch_output is not None, "codex_batch_output not provided"
            return (codex_batch_output, 1, 50)
        if template_name == "dress_codex":
            assert codex_retry_output is not None, "codex_retry_output not provided"
            return (codex_retry_output, 1, 50)
        if template_name in {"dress_brief", "dress_brief_batch"}:
            assert brief_output is not None, "brief_output not provided"
            return (brief_output, 1, 50)
        msg = f"Unexpected template_name in dispatch mock: {template_name}"
        raise AssertionError(msg)

    return _dispatch


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
                side_effect=_make_dispatch_mock(
                    brief_output=mock_brief_output,
                    codex_batch_output=mock_codex_out,
                ),
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
    async def test_phase0_halts_when_appearing_entity_lacks_visual(
        self,
        tmp_path: Path,
    ) -> None:
        """R-1.3 / R-1.4: Phase 0 raises DressStageError if any entity
        with appears edges ends up without an EntityVisual (or with an
        empty reference_prompt_fragment) after the LLM call."""
        from questfoundry.pipeline.stages.dress import DressStageError

        g = Graph()
        g.set_last_stage("fill")
        g.create_node(
            "vision::main",
            {
                "type": "vision",
                "genre": "dark fantasy",
                "tone": "brooding",
                "themes": ["betrayal"],
                "scope": {"story_size": "short"},
            },
        )
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        g.create_node(
            "entity::ghost",
            {"type": "entity", "raw_id": "ghost", "entity_type": "character"},
        )
        g.create_node("passage::opening", {"type": "passage", "raw_id": "opening"})
        # Both entities appear in the prose
        g.add_edge("appears", "entity::protagonist", "passage::opening")
        g.add_edge("appears", "entity::ghost", "passage::opening")
        g.save(tmp_path / "graph.db")

        # LLM provides a visual for protagonist only — ghost is missing
        partial_output = DressPhase0Output(
            art_direction=ArtDirection(
                style="ink",
                medium="brush ink on rice paper",
                palette=["midnight blue"],
                composition_notes="balanced framing",
                negative_defaults="cartoon",
                aspect_ratio="16:9",
            ),
            entity_visuals=[
                EntityVisualWithId(
                    entity_id="protagonist",
                    description="Young scholar",
                    distinguishing_features=["focused eyes"],
                    color_associations=["indigo"],
                    reference_prompt_fragment="young scholar, focused eyes",
                ),
            ],
        )

        stage = DressStage(project_path=tmp_path)
        # _unload_after_discuss / _unload_after_summarize are normally
        # bound by execute() (model lifecycle hooks). When invoking the
        # phase directly we bypass execute(), so set them to no-op
        # defaults to satisfy the attribute access. Brittle if those
        # attrs ever get renamed; rename them here too if so.
        stage._unload_after_discuss = None  # type: ignore[attr-defined]
        stage._unload_after_summarize = None  # type: ignore[attr-defined]
        with (
            patch(
                "questfoundry.pipeline.stages.dress.run_discuss_phase",
                new_callable=AsyncMock,
                return_value=([AIMessage(content="ok")], 1, 100),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.summarize_discussion",
                new_callable=AsyncMock,
                return_value=("brief", 50),
            ),
            patch(
                "questfoundry.pipeline.stages.dress.serialize_to_artifact",
                new_callable=AsyncMock,
                return_value=(partial_output, 100),
            ),
            pytest.raises(DressStageError, match="EntityVisual coverage"),
        ):
            await stage._phase_0_art_direction(g, MagicMock())

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
    async def test_selects_briefs_within_priority_cutoff(self) -> None:
        """Default cutoff (_min_priority=3) selects priorities 1, 2, 3 sorted."""
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
        # Sorted by priority (1 first)
        assert selection["selected_briefs"][0] == "illustration_brief::climax"

    @pytest.mark.asyncio()
    async def test_no_briefs(self) -> None:
        g = Graph()
        stage = DressStage()
        result = await stage._phase_3_review(g, MagicMock())
        assert result.detail == "no briefs to review"

    @pytest.mark.asyncio()
    async def test_priority_cutoff_excludes_low_priority_briefs(self) -> None:
        """R-2.1 + R-4.1: low-priority briefs exist (no pre-filter) but
        Gate 2 selection respects the min_priority budget."""
        g = Graph()
        g.create_node(
            "illustration_brief::high",
            {"type": "illustration_brief", "priority": 1, "subject": "high"},
        )
        g.create_node(
            "illustration_brief::low",
            {"type": "illustration_brief", "priority": 4, "subject": "low"},
        )

        stage = DressStage()
        stage._min_priority = 2  # tight budget — only priority 1+2 render
        result = await stage._phase_3_review(g, MagicMock())

        assert result.status == "completed"
        assert "1 of 2" in result.detail
        selection = g.get_node("dress_meta::selection")
        assert selection["selected_briefs"] == ["illustration_brief::high"]
        # The low-priority brief still EXISTS (no pre-filter at Phase 1) —
        # it's just not selected for rendering.
        assert g.get_node("illustration_brief::low") is not None

    @pytest.mark.asyncio()
    async def test_review_records_approval_stamp(self) -> None:
        """R-4.4: dress_meta::selection carries approved_at + approval_mode + budget."""
        g = Graph()
        g.create_node(
            "illustration_brief::a",
            {"type": "illustration_brief", "priority": 1, "subject": "a"},
        )

        stage = DressStage()
        # Default mode is non-interactive (False)
        await stage._phase_3_review(g, MagicMock())
        sel = g.get_node("dress_meta::selection")
        assert sel is not None
        assert sel.get("approved_at"), "approved_at timestamp must be set"
        # Selection runs without a human prompt in either mode today, so
        # we record "auto" rather than claiming the user picked a budget.
        assert sel.get("approval_mode") == "auto"
        assert sel.get("budget", {}).get("rule") == "priority_cutoff"
        assert sel.get("budget", {}).get("priority_cutoff") == stage._min_priority

    @pytest.mark.asyncio()
    async def test_review_records_auto_mode_even_when_interactive_flag_set(self) -> None:
        """Until an in-loop selection prompt exists, even ``--interactive``
        runs are auto-selections; the audit trail must reflect that.
        """
        g = Graph()
        g.create_node(
            "illustration_brief::a",
            {"type": "illustration_brief", "priority": 1, "subject": "a"},
        )
        stage = DressStage()
        stage._interactive = True
        await stage._phase_3_review(g, MagicMock())
        sel = g.get_node("dress_meta::selection")
        assert sel is not None
        assert sel.get("approval_mode") == "auto"


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


class TestBuildDressErrorFeedback:
    """Tests for `_build_dress_error_feedback`.

    Cluster D-2 plumbing: extra_repair_hints must reach the retry message
    so DRESS templates that depend on the initial system prompt's IDs and
    constraints can self-correct after the first attempt drifts.
    """

    def test_no_hints_legacy_output_byte_identical(self) -> None:
        """Callers that omit `extra_repair_hints` (the parameter default) get
        the same generic feedback DRESS produced before this PR. Pinning the
        EXACT string is the entire safety claim of the refactor — substring
        checks would let a stray `\\n` or reorder slip through silently."""
        from questfoundry.artifacts.validator import get_all_field_paths
        from questfoundry.models.dress import DressPhase0Output
        from questfoundry.pipeline.stages.dress import _build_dress_error_feedback

        error = ValueError("bad output")
        expected = ", ".join(get_all_field_paths(DressPhase0Output))
        # This is what the prior inline f-string produced verbatim:
        # f"Your response failed validation:\n{e}\n\n"
        # f"Expected fields: {', '.join(expected)}\n"
        # f"Please fix the errors and try again."
        legacy = (
            f"Your response failed validation:\n{error}\n\n"
            f"Expected fields: {expected}\n"
            f"Please fix the errors and try again."
        )
        feedback = _build_dress_error_feedback(error, DressPhase0Output, extra_repair_hints=None)
        assert feedback == legacy

    def test_hints_appear_verbatim_in_feedback(self) -> None:
        """A caller-supplied hint block appears literally in the retry
        feedback so the LLM sees the constraint or ID list re-stated in the
        same human-message it's correcting against. Closes the murder1 failure
        shape generalised to DRESS — see audit DRESS §dress_serialize."""
        from questfoundry.models.dress import DressPhase0Output
        from questfoundry.pipeline.stages.dress import _build_dress_error_feedback

        hints = [
            "REMINDER — Valid entity IDs (use ONLY these): kay, marcus, archive",
            "REMINDER — `priority` MUST be 1, 2, or 3.",
        ]
        feedback = _build_dress_error_feedback(
            ValueError("bad output"), DressPhase0Output, extra_repair_hints=hints
        )
        for hint in hints:
            assert hint in feedback

    def test_hints_block_separated_from_generic_message(self) -> None:
        """The hint block is positioned AFTER the generic Pydantic-error and
        field-list lines so the model reads the failure first and the
        corrective constraint last (closest to its retry context)."""
        from questfoundry.models.dress import DressPhase0Output
        from questfoundry.pipeline.stages.dress import _build_dress_error_feedback

        feedback = _build_dress_error_feedback(
            ValueError("bad output"),
            DressPhase0Output,
            extra_repair_hints=["REMINDER — Use only valid IDs."],
        )
        assert feedback.index("Please fix the errors") < feedback.index("REMINDER")

    def test_empty_hints_list_omits_reminder_block(self) -> None:
        """An empty (but non-None) list must not append a stray separator —
        same falsy-guard semantics as `if extra_repair_hints:`."""
        from questfoundry.models.dress import DressPhase0Output
        from questfoundry.pipeline.stages.dress import _build_dress_error_feedback

        feedback = _build_dress_error_feedback(
            ValueError("bad output"), DressPhase0Output, extra_repair_hints=[]
        )
        assert "REMINDER" not in feedback

    @pytest.mark.asyncio
    async def test_dress_llm_call_forwards_hints_into_retry_message(self) -> None:
        """End-to-end: `_dress_llm_call` must actually plumb `extra_repair_hints`
        through to the appended `HumanMessage` on retry. The unit tests above
        cover the helper in isolation; this test guards against a regression
        where someone drops the parameter forwarding inside `_dress_llm_call`
        — the integration seam Cluster D-2 actually fixes."""
        from langchain_core.messages import HumanMessage
        from pydantic import BaseModel, Field, ValidationError

        from questfoundry.pipeline.stages.dress import DressStage

        class _Toy(BaseModel):
            name: str = Field(min_length=1)

        # First attempt: raise ValidationError. Second: return a valid result.
        valid = _Toy(name="ok")
        call_count = {"n": 0}
        captured_messages: list[Any] = []

        async def _ainvoke(messages: list[Any], config: Any = None) -> _Toy:  # noqa: ARG001
            captured_messages.append(list(messages))  # snapshot the conversation
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValidationError.from_exception_data(
                    "Toy",
                    [
                        {
                            "type": "missing",
                            "loc": ("name",),
                            "msg": "Field required",
                            "input": {},
                        }
                    ],
                )
            return valid

        structured_model = MagicMock()
        structured_model.ainvoke = _ainvoke

        # Patch with_structured_output (used by _dress_llm_call line ~640) so
        # we don't need a real LangChain provider.
        stage = DressStage()
        with (
            patch(
                "questfoundry.pipeline.stages.dress.with_structured_output",
                return_value=structured_model,
            ),
            patch("questfoundry.prompts.loader.PromptLoader") as mock_loader,
        ):
            mock_template = MagicMock()
            mock_template.system = "system text"
            mock_template.user = None
            mock_loader.return_value.load.return_value = mock_template

            hints = ["REMINDER — must-be-present-in-retry"]
            result, _calls, _tokens = await stage._dress_llm_call(
                model=MagicMock(),
                template_name="dummy",
                context={},
                output_schema=_Toy,
                extra_repair_hints=hints,
            )

        assert result == valid
        assert call_count["n"] == 2
        # The retry conversation (second snapshot) must include the hint.
        retry_messages = captured_messages[1]
        retry_human_messages = [m for m in retry_messages if isinstance(m, HumanMessage)]
        assert retry_human_messages, "expected an appended HumanMessage on retry"
        retry_text = retry_human_messages[-1].content
        assert "REMINDER — must-be-present-in-retry" in retry_text


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
    # Beats
    g.create_node(
        "beat::opening", {"type": "beat", "raw_id": "opening", "scene_type": "establishing"}
    )
    g.create_node("beat::climax", {"type": "beat", "raw_id": "climax", "scene_type": "climax"})
    g.create_node("beat::ending", {"type": "beat", "raw_id": "ending", "scene_type": "resolution"})
    g.create_node("beat::side", {"type": "beat", "raw_id": "side", "scene_type": "transition"})
    # Predecessor edges for ordering
    g.add_edge("predecessor", "beat::climax", "beat::opening")
    g.add_edge("predecessor", "beat::ending", "beat::climax")
    # Computed-arc pattern: dilemma + paths + belongs_to
    g.create_node(
        "dilemma::d1",
        {"type": "dilemma", "raw_id": "d1", "paths": ["spine_path", "branch_path"]},
    )
    g.create_node(
        "path::spine_path",
        {"type": "path", "raw_id": "spine_path", "dilemma_id": "dilemma::d1", "is_canonical": True},
    )
    g.create_node(
        "path::branch_path",
        {
            "type": "path",
            "raw_id": "branch_path",
            "dilemma_id": "dilemma::d1",
            "is_canonical": False,
        },
    )
    # Spine beats belong to spine_path
    g.add_edge("belongs_to", "beat::opening", "path::spine_path")
    g.add_edge("belongs_to", "beat::climax", "path::spine_path")
    g.add_edge("belongs_to", "beat::ending", "path::spine_path")
    # Branch beat belongs to branch_path
    g.add_edge("belongs_to", "beat::side", "path::branch_path")
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
    # grouped_in edges (beat → passage)
    g.add_edge("grouped_in", "beat::opening", "passage::opening")
    g.add_edge("grouped_in", "beat::climax", "passage::climax")
    g.add_edge("grouped_in", "beat::ending", "passage::ending")
    g.add_edge("grouped_in", "beat::side", "passage::side")
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
    async def test_low_priority_brief_still_created(self) -> None:
        """R-2.1: every passage with prose gets a brief, even if the computed
        priority is 0. The brief is stored with a high sentinel priority so
        Gate 2 can surface it, but it won't be auto-rendered."""
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

        # base_score=0, llm_adj=-2 → total=-2 → priority would be 0;
        # spec-compliant behaviour: brief is created with sentinel priority=99
        # so it appears at Gate 2 but lies below any reasonable budget cutoff.
        assert result.status == "completed"
        brief = g.get_node("illustration_brief::boring")
        assert brief is not None
        assert brief.get("priority") == 99

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
        # Dilemma/path structure for computed arcs
        g.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "paths": ["sp"]},
        )
        g.create_node(
            "path::sp",
            {"type": "path", "raw_id": "sp", "dilemma_id": "dilemma::d1", "is_canonical": True},
        )
        # High-priority passage: spine arc opening + climax = high score
        g.create_node("beat::a", {"type": "beat", "raw_id": "a", "scene_type": "climax"})
        g.add_edge("belongs_to", "beat::a", "path::sp")
        g.create_node(
            "passage::important",
            {
                "type": "passage",
                "raw_id": "important",
                "prose": "Epic moment.",
                "from_beat": "beat::a",
            },
        )
        g.add_edge("grouped_in", "beat::a", "passage::important")
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
            "state_flag::met_aldric",
            {"type": "state_flag", "raw_id": "met_aldric", "trigger": "Meets aldric"},
        )

        stage = DressStage()
        mock_output = _make_codex_output("entity::protagonist")
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            side_effect=_make_dispatch_mock(codex_batch_output=mock_output),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # 1 batch call + 1 spoiler-check call per entity (R-3.6)
        assert result.llm_calls == 2
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

        batch_calls: list[dict[str, Any]] = []

        async def _mock_llm_call(
            _model: Any,
            _template: str,
            _context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            calls.append(_context)
            if _template == "dress_codex_spoiler_check":
                return (SpoilerCheckResult(has_leak=False, leaks=[], reason=""), 1, 25)
            # dress_codex_batch
            batch_calls.append(_context)
            # Parse entity IDs from batch context. The header now emits the
            # full prefixed form (`## Entity: entity::e0`) per #1473 — the LLM
            # mirrors what it sees, and so does this mock.
            eids = re.findall(r"## Entity: (\S+)", _context["entities_batch"])
            return (_make_batch_output(eids), 1, 100)

        stage = DressStage()
        with patch.object(stage, "_dress_llm_call", side_effect=_mock_llm_call):
            result = await stage._phase_2_codex(g, MagicMock())

        assert len(batch_calls) == 2  # 4 + 1 batches
        # 2 batch calls + 5 spoiler-check calls (one per entity)
        assert result.llm_calls == 7
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
            side_effect=_make_dispatch_mock(codex_batch_output=wrong_output),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # Valid entity should have codex, hallucinated one should not
        assert g.get_node("codex::real_rank1") is not None
        codex_nodes = g.get_nodes_by_type("codex_entry")
        # Only 1 entry (for "real"), not 2
        assert len(codex_nodes) == 1

    @pytest.mark.asyncio()
    async def test_state_flag_list_uses_prefixed_form(self) -> None:
        """The ``state_flags`` context key injected into the codex prompt MUST
        list IDs with the ``state_flag::`` prefix so the LLM mirrors that
        form back in ``visible_when`` (#1473). The validator strip-prefixes
        for comparison so the unprefixed form was tolerated, but emitting the
        prefixed form prevents small-model drift between this list and the
        per-entity ``Related State Flags`` block."""
        g = Graph()
        g.create_node(
            "entity::e0",
            {"type": "entity", "raw_id": "e0", "entity_type": "character"},
        )
        g.create_node(
            "state_flag::met_aldric",
            {
                "type": "state_flag",
                "raw_id": "met_aldric",
                "trigger": "Player has met Aldric",
            },
        )

        captured_contexts: list[dict[str, Any]] = []

        async def _mock_llm_call(
            _model: Any,
            _template: str,
            _context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            captured_contexts.append(_context)
            if _template == "dress_codex_spoiler_check":
                return (SpoilerCheckResult(has_leak=False, leaks=[], reason=""), 1, 25)
            return (
                BatchedCodexOutput(
                    entities=[
                        BatchedCodexItem(
                            entity_id="entity::e0",
                            entries=[
                                CodexEntry(title="E0", rank=1, visible_when=[], content="Info.")
                            ],
                        )
                    ]
                ),
                1,
                100,
            )

        stage = DressStage()
        with patch.object(stage, "_dress_llm_call", side_effect=_mock_llm_call):
            await stage._phase_2_codex(g, MagicMock())

        batch_contexts = [c for c in captured_contexts if "entities_batch" in c]
        assert batch_contexts, "expected at least one codex_batch call"
        sf_block = batch_contexts[0]["state_flags"]
        assert "state_flag::met_aldric" in sf_block
        # And the unprefixed form is NOT what we emit as the leading ID:
        assert "- `met_aldric`" not in sf_block

    @pytest.mark.asyncio()
    async def test_logs_validation_warnings(self) -> None:
        """Codex validation warnings are logged but don't fail the phase."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        # No state flags defined — met_aldric in visible_when will trigger warning

        stage = DressStage()
        mock_output = _make_codex_output("entity::protagonist")
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            side_effect=_make_dispatch_mock(codex_batch_output=mock_output),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        assert g.get_node("codex::protagonist_rank1") is not None


# ---------------------------------------------------------------------------
# Phase 2: spoiler-direction enforcement (R-3.6)
# ---------------------------------------------------------------------------


class TestPhase2CodexSpoilerEnforcement:
    """R-3.6: lower-tier entries must not leak higher-tier reveals.

    Detection is an LLM call after each batch; on leak, the entity is
    regenerated alone (max 2 retries); on retry exhaustion, fall back
    to a minimal rank-1-only codex with a WARNING.
    """

    @pytest.mark.asyncio()
    async def test_clean_first_attempt_no_retry(self) -> None:
        """No leak detected → no retry, original entries persist."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        stage = DressStage()
        original = _make_codex_output("entity::protagonist")
        with patch.object(
            stage,
            "_dress_llm_call",
            new_callable=AsyncMock,
            side_effect=_make_dispatch_mock(codex_batch_output=original),
        ):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # Original rank-1 content (from _make_codex_output) survived
        rank1 = g.get_node("codex::protagonist_rank1")
        assert rank1 is not None
        assert rank1["content"] == "A young scholar of the old academy."
        assert g.get_node("codex::protagonist_rank2") is not None

    @pytest.mark.asyncio()
    async def test_leak_triggers_single_retry_then_clean(self) -> None:
        """Leak → 1 retry → clean. Replaced content lands on graph."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {"type": "entity", "raw_id": "protagonist", "entity_type": "character"},
        )
        stage = DressStage()
        original = _make_codex_output("entity::protagonist")
        clean_retry = DressPhase2Output(
            entries=[
                CodexEntry(
                    title="Protagonist",
                    rank=1,
                    visible_when=[],
                    content="A figure encountered early on. Vague.",
                ),
                CodexEntry(
                    title="Protagonist's Secret",
                    rank=2,
                    visible_when=["met_aldric"],
                    content="Now revealed: the deeper truth.",
                ),
            ]
        )
        # First spoiler check leaks; after retry, second spoiler check is clean.
        leaks_then_clean = [
            SpoilerCheckResult(
                has_leak=True,
                leaks=[
                    SpoilerLeak(
                        lower_rank=1,
                        higher_rank=2,
                        leaked_content="rank-1 already reveals rank-2 content",
                    )
                ],
                reason="rank 1 leaked the secret",
            ),
            SpoilerCheckResult(has_leak=False, leaks=[], reason=""),
        ]

        async def _dispatch(
            _model: Any,
            template_name: str,
            _context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            if template_name == "dress_codex_batch":
                return (original, 1, 50)
            if template_name == "dress_codex":
                return (clean_retry, 1, 50)
            if template_name == "dress_codex_spoiler_check":
                return (leaks_then_clean.pop(0), 1, 25)
            msg = f"Unexpected template: {template_name}"
            raise AssertionError(msg)

        with patch.object(stage, "_dress_llm_call", side_effect=_dispatch):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # Retry content landed (replaced original)
        rank1 = g.get_node("codex::protagonist_rank1")
        assert rank1 is not None
        assert "Vague" in rank1["content"]

    @pytest.mark.asyncio()
    async def test_retry_exhausted_falls_back_to_rank1_only(self) -> None:
        """3 leaks (initial + 2 retries) → minimal rank-1-only fallback."""
        g = Graph()
        g.create_node(
            "entity::protagonist",
            {
                "type": "entity",
                "raw_id": "protagonist",
                "entity_type": "character",
                "name": "The Wandering Scholar",
            },
        )
        stage = DressStage()
        original = _make_codex_output("entity::protagonist")
        leaky_retry = DressPhase2Output(
            entries=[
                CodexEntry(
                    title="Still Leaky",
                    rank=1,
                    visible_when=[],
                    content="Still leaks the rank-2 secret.",
                ),
                CodexEntry(
                    title="Secret",
                    rank=2,
                    visible_when=["met_aldric"],
                    content="The secret.",
                ),
            ]
        )
        always_leak = SpoilerCheckResult(
            has_leak=True,
            leaks=[SpoilerLeak(lower_rank=1, higher_rank=2, leaked_content="leak")],
            reason="persistent leak",
        )

        async def _dispatch(
            _model: Any,
            template_name: str,
            _context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            if template_name == "dress_codex_batch":
                return (original, 1, 50)
            if template_name == "dress_codex":
                return (leaky_retry, 1, 50)
            if template_name == "dress_codex_spoiler_check":
                return (always_leak, 1, 25)
            msg = f"Unexpected template: {template_name}"
            raise AssertionError(msg)

        with patch.object(stage, "_dress_llm_call", side_effect=_dispatch):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"
        # Fallback wrote ONLY a rank-1 entry; rank-2 absent
        assert g.get_node("codex::protagonist_rank1") is not None
        assert g.get_node("codex::protagonist_rank2") is None
        rank1 = g.get_node("codex::protagonist_rank1")
        # Fallback uses entity name when present
        assert "The Wandering Scholar" in rank1["title"]
        assert rank1["visible_when"] == []

    @pytest.mark.asyncio()
    async def test_mixed_outcomes_within_one_batch(self) -> None:
        """One batch with three entities: clean, retry-then-clean, exhausted.

        Exercises the per-entity sequencing inside a single batch and
        confirms the three outcomes coexist without cross-contamination.
        """
        g = Graph()
        for raw_id in ("alpha", "beta", "gamma"):
            g.create_node(
                f"entity::{raw_id}",
                {
                    "type": "entity",
                    "raw_id": raw_id,
                    "entity_type": "character",
                    "name": raw_id.capitalize(),
                },
            )
        stage = DressStage()

        batch = BatchedCodexOutput(
            entities=[
                BatchedCodexItem(
                    entity_id=f"entity::{raw_id}",
                    entries=[
                        CodexEntry(
                            title=raw_id, rank=1, visible_when=[], content="Original rank 1."
                        ),
                        CodexEntry(
                            title=f"{raw_id} Secret",
                            rank=2,
                            visible_when=["state_flag::known"],
                            content="Original rank 2.",
                        ),
                    ],
                )
                for raw_id in ("alpha", "beta", "gamma")
            ]
        )
        clean_retry = DressPhase2Output(
            entries=[
                CodexEntry(
                    title="Beta",
                    rank=1,
                    visible_when=[],
                    content="Replaced rank 1 — vague.",
                ),
                CodexEntry(
                    title="Beta Secret",
                    rank=2,
                    visible_when=["state_flag::known"],
                    content="Replaced rank 2.",
                ),
            ]
        )
        leaky_retry = DressPhase2Output(
            entries=[
                CodexEntry(
                    title="Gamma",
                    rank=1,
                    visible_when=[],
                    content="Still leaks.",
                ),
                CodexEntry(
                    title="Gamma Secret",
                    rank=2,
                    visible_when=["state_flag::known"],
                    content="Persistent leak.",
                ),
            ]
        )
        clean = SpoilerCheckResult(has_leak=False, leaks=[], reason="")
        leak = SpoilerCheckResult(
            has_leak=True,
            leaks=[SpoilerLeak(lower_rank=1, higher_rank=2, leaked_content="leak")],
            reason="leak",
        )
        # alpha: clean
        # beta: leak → retry → clean (3 spoiler checks total: alpha clean, beta leak, beta clean,
        #   then gamma block follows)
        # gamma: leak → retry → leak → retry → leak (3 spoiler checks for gamma)
        # Sequence: alpha-check(clean), beta-check(leak), beta-recheck(clean),
        #           gamma-check(leak), gamma-recheck(leak), gamma-recheck(leak)
        spoiler_sequence = [clean, leak, clean, leak, leak, leak]

        async def _dispatch(
            _model: Any,
            template_name: str,
            context: dict[str, Any],
            _schema: type,
            **_kwargs: Any,
        ) -> tuple:
            if template_name == "dress_codex_batch":
                return (batch, 1, 50)
            if template_name == "dress_codex":
                # Pick the right per-entity retry output by inspecting the
                # injected entity_details (which contains the raw id).
                if "beta" in context.get("entity_details", ""):
                    return (clean_retry, 1, 50)
                return (leaky_retry, 1, 50)
            if template_name == "dress_codex_spoiler_check":
                return (spoiler_sequence.pop(0), 1, 25)
            msg = f"Unexpected template: {template_name}"
            raise AssertionError(msg)

        with patch.object(stage, "_dress_llm_call", side_effect=_dispatch):
            result = await stage._phase_2_codex(g, MagicMock())

        assert result.status == "completed"

        # alpha: original entries kept (rank 1 + rank 2)
        alpha_r1 = g.get_node("codex::alpha_rank1")
        assert alpha_r1 is not None
        assert alpha_r1["content"] == "Original rank 1."
        assert g.get_node("codex::alpha_rank2") is not None

        # beta: retry replaced both ranks
        beta_r1 = g.get_node("codex::beta_rank1")
        assert beta_r1 is not None
        assert "Replaced rank 1" in beta_r1["content"]
        assert g.get_node("codex::beta_rank2") is not None

        # gamma: rank-1-only fallback (rank 2 dropped)
        gamma_r1 = g.get_node("codex::gamma_rank1")
        assert gamma_r1 is not None
        assert "Further details are not yet known" in gamma_r1["content"]
        assert g.get_node("codex::gamma_rank2") is None
        # Sequence fully consumed: 6 spoiler checks issued
        assert spoiler_sequence == []


def test_format_entries_for_spoiler_check_orders_by_rank() -> None:
    from questfoundry.pipeline.stages.dress import _format_entries_for_spoiler_check

    entries = [
        {"rank": 2, "title": "Hidden", "visible_when": ["state_flag::met"], "content": "Deep."},
        {"rank": 1, "title": "Surface", "visible_when": [], "content": "Shallow."},
    ]
    rendered = _format_entries_for_spoiler_check(entries)
    # Rank 1 must appear before Rank 2 in the rendered block
    assert rendered.index("Rank 1") < rendered.index("Rank 2")
    assert "always visible" in rendered
    assert "gated by `met`" in rendered


def test_minimal_rank_one_codex_uses_entity_name() -> None:
    from questfoundry.pipeline.stages.dress import _minimal_rank_one_codex

    g = Graph()
    g.create_node(
        "entity::aldric",
        {"type": "entity", "raw_id": "aldric", "name": "Aldric the Scribe"},
    )
    fallback = _minimal_rank_one_codex(g, "entity::aldric")
    assert len(fallback) == 1
    assert fallback[0]["rank"] == 1
    assert fallback[0]["visible_when"] == []
    assert "Aldric the Scribe" in fallback[0]["title"]


def test_minimal_rank_one_codex_falls_back_to_raw_id() -> None:
    from questfoundry.pipeline.stages.dress import _minimal_rank_one_codex

    g = Graph()
    fallback = _minimal_rank_one_codex(g, "entity::missing")
    # No node, no name → use raw id as title
    assert fallback[0]["title"] == "missing"


def test_minimal_rank_one_codex_uses_entity_type_descriptor() -> None:
    """Fallback content is diegetic for non-character entities (R-3.4)."""
    from questfoundry.pipeline.stages.dress import _minimal_rank_one_codex

    g = Graph()
    g.create_node(
        "entity::cliff_pass",
        {"type": "entity", "raw_id": "cliff_pass", "name": "Cliff Pass", "entity_type": "location"},
    )
    g.create_node(
        "entity::sword",
        {"type": "entity", "raw_id": "sword", "name": "Old Sword", "entity_type": "object"},
    )
    g.create_node(
        "entity::guild",
        {"type": "entity", "raw_id": "guild", "name": "The Guild", "entity_type": "faction"},
    )
    g.create_node(
        "entity::weird",
        {"type": "entity", "raw_id": "weird", "name": "Weirdness", "entity_type": "concept"},
    )

    assert "a place encountered" in _minimal_rank_one_codex(g, "entity::cliff_pass")[0]["content"]
    assert "an object encountered" in _minimal_rank_one_codex(g, "entity::sword")[0]["content"]
    assert "a group encountered" in _minimal_rank_one_codex(g, "entity::guild")[0]["content"]
    # Unknown entity_type falls back to the generic descriptor
    assert "an element of the story" in _minimal_rank_one_codex(g, "entity::weird")[0]["content"]


def test_spoiler_leak_rejects_inverted_rank_ordering() -> None:
    """Pydantic must reject lower_rank ≥ higher_rank — direction matters in R-3.6."""
    from pydantic import ValidationError

    # lower == higher
    with pytest.raises(ValidationError, match=r"lower_rank.*must be strictly less than"):
        SpoilerLeak(lower_rank=2, higher_rank=2, leaked_content="x")

    # lower > higher
    with pytest.raises(ValidationError, match=r"lower_rank.*must be strictly less than"):
        SpoilerLeak(lower_rank=3, higher_rank=2, leaked_content="x")

    # Valid case still works
    leak = SpoilerLeak(lower_rank=1, higher_rank=2, leaked_content="x")
    assert leak.lower_rank == 1


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
