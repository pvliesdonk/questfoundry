"""Tests for DRESS stage Pydantic models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from questfoundry.models.dress import (
    ArtDirection,
    CodexEntry,
    DressPhase0Output,
    DressPhase1Output,
    DressPhase2Output,
    DressPhaseResult,
    DressResult,
    EntityVisual,
    EntityVisualWithId,
    Illustration,
    IllustrationBrief,
)

# ---------------------------------------------------------------------------
# ArtDirection
# ---------------------------------------------------------------------------


class TestArtDirection:
    def test_valid_art_direction(self) -> None:
        ad = ArtDirection(
            style="watercolor",
            medium="traditional",
            palette=["indigo", "rust"],
            composition_notes="wide shots",
            negative_defaults="photorealistic",
        )
        assert ad.style == "watercolor"
        assert ad.aspect_ratio == "16:9"  # default

    def test_custom_aspect_ratio(self) -> None:
        ad = ArtDirection(
            style="ink",
            medium="digital",
            palette=["black"],
            composition_notes="close-up",
            negative_defaults="text",
            aspect_ratio="1:1",
        )
        assert ad.aspect_ratio == "1:1"

    @pytest.mark.parametrize(
        "field",
        ["style", "medium", "composition_notes", "negative_defaults"],
        ids=["style", "medium", "composition_notes", "negative_defaults"],
    )
    def test_empty_string_rejected(self, field: str) -> None:
        data = {
            "style": "watercolor",
            "medium": "traditional",
            "palette": ["indigo"],
            "composition_notes": "wide shots",
            "negative_defaults": "photorealistic",
        }
        data[field] = ""
        with pytest.raises(ValidationError):
            ArtDirection(**data)

    def test_empty_palette_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtDirection(
                style="watercolor",
                medium="traditional",
                palette=[],
                composition_notes="wide shots",
                negative_defaults="photorealistic",
            )


# ---------------------------------------------------------------------------
# Illustration
# ---------------------------------------------------------------------------


class TestIllustration:
    def test_valid_illustration(self) -> None:
        illust = Illustration(
            asset="assets/abc123.png",
            caption="The bridge where loyalties shatter",
            category="scene",
        )
        assert illust.asset == "assets/abc123.png"

    @pytest.mark.parametrize("field", ["asset", "caption"])
    def test_empty_string_rejected(self, field: str) -> None:
        data = {"asset": "assets/x.png", "caption": "caption", "category": "scene"}
        data[field] = ""
        with pytest.raises(ValidationError):
            Illustration(**data)

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Illustration(asset="assets/x.png", caption="caption", category="panorama")


# ---------------------------------------------------------------------------
# CodexEntry
# ---------------------------------------------------------------------------


class TestCodexEntry:
    def test_valid_base_entry(self) -> None:
        entry = CodexEntry(rank=1, content="A traveling scholar.")
        assert entry.visible_when == []
        assert entry.rank == 1

    def test_gated_entry(self) -> None:
        entry = CodexEntry(
            rank=2,
            visible_when=["met_aldric"],
            content="Claims to be a former court advisor.",
        )
        assert entry.visible_when == ["met_aldric"]

    def test_rank_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodexEntry(rank=0, content="content")

    def test_negative_rank_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodexEntry(rank=-1, content="content")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodexEntry(rank=1, content="")


# ---------------------------------------------------------------------------
# EntityVisual
# ---------------------------------------------------------------------------


class TestEntityVisual:
    def test_valid_entity_visual(self) -> None:
        ev = EntityVisual(
            description="Young woman, short dark hair",
            distinguishing_features=["jade pendant"],
            color_associations=["indigo"],
            reference_prompt_fragment="young woman, short dark hair, jade pendant",
        )
        assert len(ev.distinguishing_features) == 1

    def test_empty_color_associations_allowed(self) -> None:
        ev = EntityVisual(
            description="A bridge",
            distinguishing_features=["crumbling stone"],
            color_associations=[],
            reference_prompt_fragment="crumbling stone bridge",
        )
        assert ev.color_associations == []

    def test_empty_distinguishing_features_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityVisual(
                description="desc",
                distinguishing_features=[],
                color_associations=[],
                reference_prompt_fragment="fragment",
            )

    def test_empty_prompt_fragment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityVisual(
                description="desc",
                distinguishing_features=["feature"],
                color_associations=[],
                reference_prompt_fragment="",
            )


class TestEntityVisualWithId:
    def test_includes_entity_id(self) -> None:
        ev = EntityVisualWithId(
            entity_id="protagonist",
            description="Young woman",
            distinguishing_features=["jade pendant"],
            reference_prompt_fragment="young woman, jade pendant",
        )
        assert ev.entity_id == "protagonist"

    def test_empty_entity_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EntityVisualWithId(
                entity_id="",
                description="desc",
                distinguishing_features=["feature"],
                reference_prompt_fragment="fragment",
            )


# ---------------------------------------------------------------------------
# IllustrationBrief
# ---------------------------------------------------------------------------


class TestIllustrationBrief:
    def test_valid_brief(self) -> None:
        brief = IllustrationBrief(
            priority=1,
            category="scene",
            subject="Two figures on a bridge at twilight",
            entities=["entity::protagonist", "entity::aldric"],
            composition="wide shot, silhouetted against sunset",
            mood="tense, bittersweet",
            style_overrides="",
            negative="",
            caption="The bridge where loyalties shatter",
        )
        assert brief.priority == 1
        assert len(brief.entities) == 2
        assert brief.style_overrides == ""
        assert brief.negative == ""

    def test_priority_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            IllustrationBrief(
                priority=4,
                category="scene",
                subject="subject",
                entities=[],
                composition="comp",
                mood="mood",
                style_overrides="",
                negative="",
                caption="caption",
            )

    def test_priority_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IllustrationBrief(
                priority=0,
                category="scene",
                subject="subject",
                entities=[],
                composition="comp",
                mood="mood",
                style_overrides="",
                negative="",
                caption="caption",
            )

    def test_empty_entities_allowed(self) -> None:
        brief = IllustrationBrief(
            priority=3,
            category="vista",
            subject="An empty battlefield",
            entities=[],
            composition="panoramic",
            mood="somber",
            style_overrides="",
            negative="",
            caption="Where the silence speaks loudest",
        )
        assert brief.entities == []

    def test_missing_entities_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IllustrationBrief(
                priority=1,
                category="scene",
                subject="subject",
                composition="comp",
                mood="mood",
                style_overrides="",
                negative="",
                caption="caption",
            )

    def test_missing_caption_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IllustrationBrief(
                priority=1,
                category="scene",
                subject="subject",
                entities=[],
                composition="comp",
                mood="mood",
                style_overrides="",
                negative="",
            )


# ---------------------------------------------------------------------------
# Phase outputs
# ---------------------------------------------------------------------------


class TestDressPhase0Output:
    def test_valid_phase0(self) -> None:
        output = DressPhase0Output(
            art_direction=ArtDirection(
                style="ink wash",
                medium="sumi-e",
                palette=["indigo", "rust"],
                composition_notes="wide shots",
                negative_defaults="photorealistic",
            ),
            entity_visuals=[
                EntityVisualWithId(
                    entity_id="protagonist",
                    description="Young woman",
                    distinguishing_features=["jade pendant"],
                    reference_prompt_fragment="young woman, jade pendant",
                ),
            ],
        )
        assert len(output.entity_visuals) == 1

    def test_empty_entity_visuals_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DressPhase0Output(
                art_direction=ArtDirection(
                    style="ink",
                    medium="digital",
                    palette=["black"],
                    composition_notes="close-up",
                    negative_defaults="text",
                ),
                entity_visuals=[],
            )


class TestDressPhase1Output:
    def _make_brief(self, **overrides: Any) -> IllustrationBrief:
        defaults: dict[str, Any] = {
            "priority": 1,
            "category": "scene",
            "subject": "subject",
            "entities": [],
            "composition": "comp",
            "mood": "mood",
            "style_overrides": "",
            "negative": "",
            "caption": "caption",
        }
        defaults.update(overrides)
        return IllustrationBrief(**defaults)

    def test_valid_phase1(self) -> None:
        output = DressPhase1Output(
            brief=self._make_brief(),
            llm_adjustment=1,
        )
        assert output.llm_adjustment == 1

    def test_missing_adjustment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DressPhase1Output(brief=self._make_brief())

    def test_adjustment_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DressPhase1Output(
                brief=self._make_brief(),
                llm_adjustment=3,
            )


class TestDressPhase2Output:
    def test_valid_phase2(self) -> None:
        output = DressPhase2Output(
            entries=[
                CodexEntry(rank=1, content="A traveling scholar."),
                CodexEntry(rank=2, visible_when=["met_aldric"], content="Former court advisor."),
            ],
        )
        assert len(output.entries) == 2

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DressPhase2Output(entries=[])


# ---------------------------------------------------------------------------
# Stage result containers
# ---------------------------------------------------------------------------


class TestDressPhaseResult:
    def test_inherits_phase_result(self) -> None:
        result = DressPhaseResult(phase="art_direction", status="completed")
        assert result.phase == "art_direction"
        assert result.llm_calls == 0


class TestDressResult:
    def test_defaults(self) -> None:
        result = DressResult()
        assert result.art_direction_created is False
        assert result.entity_visuals_created == 0
        assert result.illustrations_generated == 0

    def test_populated(self) -> None:
        result = DressResult(
            art_direction_created=True,
            entity_visuals_created=5,
            briefs_created=20,
            codex_entries_created=15,
            illustrations_generated=8,
            illustrations_failed=2,
        )
        assert result.briefs_created == 20


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_art_direction_roundtrip(self) -> None:
        ad = ArtDirection(
            style="watercolor",
            medium="traditional",
            palette=["indigo"],
            composition_notes="wide shots",
            negative_defaults="photorealistic",
        )
        data = ad.model_dump()
        restored = ArtDirection.model_validate(data)
        assert restored == ad

    def test_codex_entry_roundtrip(self) -> None:
        entry = CodexEntry(rank=2, visible_when=["met_aldric"], content="Former court advisor.")
        data = entry.model_dump()
        restored = CodexEntry.model_validate(data)
        assert restored == entry

    def test_phase0_output_roundtrip(self) -> None:
        output = DressPhase0Output(
            art_direction=ArtDirection(
                style="ink",
                medium="sumi-e",
                palette=["black"],
                composition_notes="minimalist",
                negative_defaults="color",
            ),
            entity_visuals=[
                EntityVisualWithId(
                    entity_id="hero",
                    description="Tall warrior",
                    distinguishing_features=["scarred cheek"],
                    reference_prompt_fragment="tall warrior, scarred cheek",
                ),
            ],
        )
        data = output.model_dump()
        restored = DressPhase0Output.model_validate(data)
        assert restored == output
