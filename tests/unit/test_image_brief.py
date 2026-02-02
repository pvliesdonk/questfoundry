"""Tests for ImageBrief dataclass, flatten_brief_to_prompt, and build_image_brief."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.dress import assemble_image_prompt, build_image_brief
from questfoundry.providers.image import PromptDistiller
from questfoundry.providers.image_brief import ImageBrief, flatten_brief_to_prompt


class TestImageBrief:
    def test_construction_with_defaults(self) -> None:
        brief = ImageBrief(subject="A castle", composition="Wide", mood="epic")
        assert brief.subject == "A castle"
        assert brief.negative is None
        assert brief.style_overrides is None
        assert brief.entity_fragments == []
        assert brief.palette == []
        assert brief.aspect_ratio == "16:9"
        assert brief.category == "scene"

    def test_construction_with_all_fields(self) -> None:
        brief = ImageBrief(
            subject="Battle",
            composition="Close-up",
            mood="tense",
            negative="gore",
            style_overrides="darker palette",
            entity_fragments=["warrior", "dragon"],
            art_style="oil painting",
            art_medium="canvas",
            palette=["crimson", "gold"],
            negative_defaults="photorealism",
            aspect_ratio="1:1",
            category="portrait",
        )
        assert brief.entity_fragments == ["warrior", "dragon"]
        assert brief.art_style == "oil painting"
        assert brief.category == "portrait"

    def test_frozen(self) -> None:
        brief = ImageBrief(subject="X", composition="Y", mood="Z")
        with pytest.raises(AttributeError):
            brief.subject = "changed"  # type: ignore[misc]


class TestFlattenBriefToPrompt:
    def test_basic_flatten(self) -> None:
        brief = ImageBrief(
            subject="Scholar at the bridge",
            composition="Wide shot",
            mood="foreboding",
            negative="modern elements",
            art_style="watercolor",
            art_medium="traditional paper",
            palette=["deep indigo", "gold"],
            negative_defaults="photorealism",
        )
        positive, negative = flatten_brief_to_prompt(brief)

        assert "Scholar at the bridge" in positive
        assert "watercolor" in positive
        assert "deep indigo" in positive
        assert negative is not None
        assert "modern elements" in negative
        assert "photorealism" in negative

    def test_includes_entity_fragments(self) -> None:
        brief = ImageBrief(
            subject="Battle scene",
            composition="",
            mood="",
            entity_fragments=["tall warrior, scarred face", "black dragon"],
            art_style="ink",
        )
        positive, _ = flatten_brief_to_prompt(brief)

        assert "tall warrior, scarred face" in positive
        assert "black dragon" in positive

    def test_no_art_direction(self) -> None:
        brief = ImageBrief(subject="A simple scene", composition="", mood="")
        positive, negative = flatten_brief_to_prompt(brief)

        assert "A simple scene" in positive
        assert negative is None

    def test_includes_style_overrides(self) -> None:
        brief = ImageBrief(
            subject="Storm",
            composition="",
            mood="",
            style_overrides="darker and grittier",
            art_style="watercolor",
        )
        positive, _ = flatten_brief_to_prompt(brief)

        assert "darker and grittier" in positive

    def test_empty_style_overrides_not_included(self) -> None:
        brief = ImageBrief(
            subject="Storm",
            composition="",
            mood="",
            style_overrides="",
        )
        positive, _ = flatten_brief_to_prompt(brief)
        # No trailing comma from empty style_overrides
        assert not positive.endswith(",")


class TestBuildImageBrief:
    def test_from_graph(self) -> None:
        g = Graph()
        g.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "watercolor",
                "medium": "traditional paper",
                "palette": ["deep indigo", "gold"],
                "negative_defaults": "photorealism",
                "aspect_ratio": "16:9",
            },
        )
        g.create_node(
            "entity_visual::hero",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "tall warrior, scarred face",
            },
        )

        brief_data = {
            "subject": "Battle scene",
            "composition": "Wide shot",
            "mood": "epic",
            "negative": "modern elements",
            "style_overrides": "darker palette",
            "entities": ["hero"],
            "category": "scene",
        }

        result = build_image_brief(g, brief_data)

        assert isinstance(result, ImageBrief)
        assert result.subject == "Battle scene"
        assert result.entity_fragments == ["hero: tall warrior, scarred face"]
        assert result.art_style == "watercolor"
        assert result.palette == ["deep indigo", "gold"]
        assert result.negative_defaults == "photorealism"
        assert result.style_overrides == "darker palette"
        assert result.aspect_ratio == "16:9"

    def test_missing_art_direction(self) -> None:
        g = Graph()
        brief_data = {"subject": "Simple", "entities": []}

        result = build_image_brief(g, brief_data)

        assert result.art_style is None
        assert result.art_medium is None
        assert result.palette == []
        assert result.aspect_ratio == "16:9"  # default

    def test_scoped_entity_ids(self) -> None:
        g = Graph()
        g.create_node("art_direction::main", {"type": "art_direction"})
        g.create_node(
            "entity_visual::hero",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "tall warrior",
            },
        )

        brief_data = {"subject": "Fight", "entities": ["entity::hero"]}
        result = build_image_brief(g, brief_data)

        assert result.entity_fragments == ["hero: tall warrior"]

    def test_empty_strings_become_none(self) -> None:
        g = Graph()
        brief_data = {
            "subject": "Scene",
            "negative": "",
            "style_overrides": "",
            "entities": [],
        }
        result = build_image_brief(g, brief_data)
        assert result.negative is None
        assert result.style_overrides is None


class TestAssembleImagePromptBackwardCompat:
    """Verify the wrapper still produces identical output."""

    def test_matches_flatten(self) -> None:
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

        wrapper_result = assemble_image_prompt(g, brief)
        direct_result = flatten_brief_to_prompt(build_image_brief(g, brief))

        assert wrapper_result == direct_result


class TestPromptDistillerProtocol:
    def test_conforming_class(self) -> None:
        class MyDistiller:
            async def distill_prompt(
                self,
                brief: ImageBrief,  # noqa: ARG002
            ) -> tuple[str, str | None]:
                return "tags", None

        assert isinstance(MyDistiller(), PromptDistiller)

    def test_non_conforming_class(self) -> None:
        class NotADistiller:
            async def generate(self, prompt: str) -> None:
                pass

        assert not isinstance(NotADistiller(), PromptDistiller)
