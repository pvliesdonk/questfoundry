"""Contract-chaining seam tests (#1346 / #1347 / #1348).

Confirms that each stage exit and adjacent stage entry actually invokes
the validate_<stage>_output() helper, halts loudly on contract violations
with the right exception type, and lets clean inputs through.

Per-stage tests bypass these validators with autouse fixtures so they can
focus on stage mechanics. THIS file is where the seams themselves are
tested — the contract-chaining audit's whole purpose was to prove that a
malformed upstream artifact cannot silently flow downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from questfoundry.graph.dress_output_validation import validate_dress_output
from questfoundry.graph.fill_output_validation import validate_fill_output
from questfoundry.graph.graph import Graph

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# validate_fill_output (helper, called at FILL exit + DRESS entry + SHIP entry)
# ---------------------------------------------------------------------------


class TestValidateFillOutput:
    def test_empty_graph_flags_voice_and_passages(self) -> None:
        errors = validate_fill_output(Graph.empty())
        assert any("Voice Document" in e for e in errors)
        assert any("at least one Passage" in e for e in errors)

    def test_voice_present_but_missing_required_field(self) -> None:
        g = Graph.empty()
        g.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                # tense omitted
                "pov": "third_person_limited",
                "voice_register": "literary",
                "sentence_rhythm": "varied",
                "tone_words": ["wry"],
            },
        )
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1", "prose": "x"})
        errors = validate_fill_output(g)
        assert any("missing required field(s): tense" in e for e in errors)

    def test_passage_with_blank_prose_flagged(self) -> None:
        g = Graph.empty()
        _add_minimal_voice(g)
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1", "prose": "   "})
        errors = validate_fill_output(g)
        assert any("without prose" in e for e in errors)

    def test_clean_graph_passes(self) -> None:
        g = Graph.empty()
        _add_minimal_voice(g)
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1", "prose": "Hello."})
        assert validate_fill_output(g) == []


# ---------------------------------------------------------------------------
# validate_dress_output (helper, called at DRESS exit + SHIP entry)
# ---------------------------------------------------------------------------


class TestValidateDressOutput:
    def test_no_art_direction_treated_as_dress_skipped(self) -> None:
        """DRESS Output-10: opt-out projects skip the contract entirely."""
        g = Graph.empty()
        # No art_direction node → DRESS skipped → empty errors
        assert validate_dress_output(g) == []

    def test_appearing_entity_without_visual_flagged(self) -> None:
        g = Graph.empty()
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "palette": ["x"]},
        )
        g.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        g.add_edge("appears", "entity::mentor", "passage::p1")
        # No EntityVisual for mentor
        errors = validate_dress_output(g)
        assert any("entity_visual::mentor" in e and "missing" in e for e in errors)

    def test_passage_without_brief_flagged(self) -> None:
        g = Graph.empty()
        g.create_node("art_direction::main", {"type": "art_direction"})
        g.create_node("passage::p1", {"type": "passage", "raw_id": "p1"})
        # No IllustrationBrief for p1
        errors = validate_dress_output(g)
        assert any("without an IllustrationBrief" in e for e in errors)

    def test_entity_without_codex_flagged(self) -> None:
        g = Graph.empty()
        g.create_node("art_direction::main", {"type": "art_direction"})
        g.create_node("entity::mentor", {"type": "entity", "raw_id": "mentor"})
        # No passages → Output-3 is vacuously satisfied (no passages →
        # no missing-brief check fires), so this assertion isolates the
        # Output-5 (missing CodexEntry) error.
        errors = validate_dress_output(g)
        assert any("without a CodexEntry" in e for e in errors)

    def test_rank1_with_visible_when_flagged(self) -> None:
        g = Graph.empty()
        g.create_node("art_direction::main", {"type": "art_direction"})
        g.create_node(
            "codex::leak",
            {"type": "codex_entry", "rank": 1, "visible_when": ["state_flag::met"]},
        )
        errors = validate_dress_output(g)
        assert any("rank-1" in e and "visible_when" in e for e in errors)


# ---------------------------------------------------------------------------
# Stage entry seams: each stage rejects malformed upstream artifacts loudly
# ---------------------------------------------------------------------------


class TestSeedEntryRejectsBadBrainstorm:
    """SEED entry calls validate_brainstorm_output (#1347)."""

    @pytest.mark.asyncio
    async def test_seed_raises_on_empty_brainstorm(self, tmp_path: Path) -> None:
        from pathlib import Path as _P

        from questfoundry.pipeline.stages.seed import SeedStage, SeedStageError

        g = Graph.empty()
        g.set_last_stage("brainstorm")
        g.save(tmp_path / "graph.db")

        stage = SeedStage(project_path=tmp_path)
        with pytest.raises(SeedStageError, match="BRAINSTORM output validation failed"):
            await stage.execute(model=MagicMock(), user_prompt="x", project_path=_P(tmp_path))


class TestGrowEntryRejectsBadSeed:
    """GROW entry calls validate_seed_output (#1347)."""

    @pytest.mark.asyncio
    async def test_grow_raises_on_empty_seed(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage, GrowStageError

        g = Graph.empty()
        g.set_last_stage("seed")  # but no paths/beats/entities → SEED contract violated
        g.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        with pytest.raises(GrowStageError, match="SEED output validation failed"):
            await stage.execute(model=MagicMock(), user_prompt="x")


class TestFillEntryRejectsBadPolish:
    """FILL entry calls validate_polish_output (#1347)."""

    @pytest.mark.asyncio
    async def test_fill_raises_on_empty_polish(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.fill import FillStage, FillStageError

        g = Graph.empty()
        g.set_last_stage("polish")  # but no passages → POLISH contract violated
        g.save(tmp_path / "graph.db")

        stage = FillStage(project_path=tmp_path)
        with pytest.raises(FillStageError, match="POLISH output validation failed"):
            await stage.execute(model=MagicMock(), user_prompt="x")


class TestDressEntryRejectsBadFill:
    """DRESS entry calls validate_fill_output (#1347)."""

    @pytest.mark.asyncio
    async def test_dress_raises_on_empty_fill(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.dress import DressStage, DressStageError

        g = Graph.empty()
        g.set_last_stage("fill")  # but no voice/passages → FILL contract violated
        g.save(tmp_path / "graph.db")

        stage = DressStage(project_path=tmp_path)
        with pytest.raises(DressStageError, match="FILL output validation failed"):
            await stage.execute(model=MagicMock(), user_prompt="x")


class TestShipEntryRejectsBadUpstream:
    """SHIP entry runs both validate_fill_output and validate_dress_output (#1347)."""

    def test_ship_raises_on_empty_fill(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.ship import ShipStage, ShipStageError

        project = tmp_path / "story"
        project.mkdir()
        g = Graph()
        g.save(project / "graph.db")

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="Pre-SHIP contract validation failed"):
            stage.execute()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_minimal_voice(g: Graph) -> None:
    """Voice Document satisfying validate_fill_output's required-field set."""
    g.create_node(
        "voice::voice",
        {
            "type": "voice",
            "raw_id": "voice",
            "pov": "third_person_limited",
            "tense": "past",
            "voice_register": "literary",
            "sentence_rhythm": "varied",
            "tone_words": ["wry"],
        },
    )
