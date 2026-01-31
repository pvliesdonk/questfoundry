"""Tests for DRESS stage context formatters."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph


@pytest.fixture()
def dress_graph() -> Graph:
    """Graph with entities, passages, and codewords for DRESS testing."""
    g = Graph()
    g.create_node(
        "vision::main",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": "brooding",
            "themes": ["betrayal", "redemption"],
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
        "entity::bridge",
        {
            "type": "entity",
            "raw_id": "bridge",
            "entity_type": "location",
            "concept": "Ancient stone bridge spanning a chasm",
        },
    )
    g.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "Scholar arrives at bridge",
            "scene_type": "establishing",
        },
    )
    g.create_node(
        "passage::opening",
        {
            "type": "passage",
            "raw_id": "opening",
            "from_beat": "beat::opening",
            "prose": "The wind howled across the ancient stone bridge...",
            "entities": ["entity::protagonist", "entity::bridge"],
        },
    )
    g.create_node(
        "codeword::met_aldric",
        {
            "type": "codeword",
            "raw_id": "met_aldric",
            "trigger": "Player meets aldric at the bridge",
        },
    )
    return g


class TestFormatVisionAndEntities:
    def test_includes_vision(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_vision_and_entities

        result = format_vision_and_entities(dress_graph)
        assert "Creative Vision" in result
        assert "dark fantasy" in result

    def test_includes_entities_by_type(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_vision_and_entities

        result = format_vision_and_entities(dress_graph)
        assert "Characters" in result
        assert "entity::protagonist" in result
        assert "Locations" in result
        assert "entity::bridge" in result

    def test_empty_graph(self) -> None:
        from questfoundry.graph.dress_context import format_vision_and_entities

        assert format_vision_and_entities(Graph()) == ""


class TestFormatArtDirectionContext:
    def test_formats_art_direction(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_art_direction_context

        dress_graph.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "watercolor",
                "medium": "traditional",
                "palette": ["indigo", "rust"],
            },
        )
        result = format_art_direction_context(dress_graph)
        assert "watercolor" in result
        assert "indigo" in result

    def test_no_art_direction(self) -> None:
        from questfoundry.graph.dress_context import format_art_direction_context

        assert format_art_direction_context(Graph()) == ""


class TestFormatPassageForBrief:
    def test_includes_prose_and_metadata(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passage_for_brief

        result = format_passage_for_brief(dress_graph, "passage::opening")
        assert "wind howled" in result
        assert "establishing" in result
        assert "entity::protagonist" in result

    def test_nonexistent_passage(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passage_for_brief

        assert format_passage_for_brief(dress_graph, "passage::nonexistent") == ""


class TestFormatEntityForCodex:
    def test_includes_entity_details(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        result = format_entity_for_codex(dress_graph, "entity::aldric")
        assert "aldric" in result
        assert "character" in result
        assert "court advisor" in result

    def test_includes_related_codewords(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        result = format_entity_for_codex(dress_graph, "entity::aldric")
        assert "met_aldric" in result

    def test_nonexistent_entity(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        assert format_entity_for_codex(dress_graph, "entity::nonexistent") == ""


class TestFormatEntityVisualsForPassage:
    def test_includes_visual_fragments(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_visuals_for_passage

        dress_graph.create_node(
            "entity_visual::protagonist",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "young woman, short dark hair",
            },
        )
        result = format_entity_visuals_for_passage(dress_graph, "passage::opening")
        assert "young woman, short dark hair" in result
        assert "protagonist" in result

    def test_no_visuals(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_visuals_for_passage

        assert format_entity_visuals_for_passage(dress_graph, "passage::opening") == ""
