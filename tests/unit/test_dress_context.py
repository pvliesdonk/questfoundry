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
        "character::protagonist",
        {
            "type": "entity",
            "raw_id": "protagonist",
            "category": "character",
            "entity_type": "character",
            "concept": "A young scholar seeking forbidden knowledge",
        },
    )
    g.create_node(
        "character::aldric",
        {
            "type": "entity",
            "raw_id": "aldric",
            "category": "character",
            "entity_type": "character",
            "concept": "A former court advisor with hidden motives",
        },
    )
    g.create_node(
        "location::bridge",
        {
            "type": "entity",
            "raw_id": "bridge",
            "category": "location",
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
            "entities": ["character::protagonist", "location::bridge"],
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
        assert "character::protagonist" in result
        assert "Locations" in result
        assert "location::bridge" in result

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
        assert "character::protagonist" in result

    def test_nonexistent_passage(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passage_for_brief

        assert format_passage_for_brief(dress_graph, "passage::nonexistent") == ""


class TestFormatEntityForCodex:
    def test_includes_entity_details(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        result = format_entity_for_codex(dress_graph, "character::aldric")
        assert "aldric" in result
        assert "character" in result
        assert "court advisor" in result

    def test_includes_related_codewords(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        result = format_entity_for_codex(dress_graph, "character::aldric")
        assert "met_aldric" in result

    def test_nonexistent_entity(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_for_codex

        assert format_entity_for_codex(dress_graph, "character::nonexistent") == ""


class TestFormatEntitiesBatchForCodex:
    def test_formats_multiple_entities(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entities_batch_for_codex

        result = format_entities_batch_for_codex(
            dress_graph, ["character::protagonist", "character::aldric"]
        )
        assert "protagonist" in result
        assert "aldric" in result
        assert "court advisor" in result

    def test_skips_missing_entities(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entities_batch_for_codex

        result = format_entities_batch_for_codex(
            dress_graph, ["character::protagonist", "character::nonexistent"]
        )
        assert "protagonist" in result
        assert "nonexistent" not in result

    def test_empty_list(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entities_batch_for_codex

        assert format_entities_batch_for_codex(dress_graph, []) == ""


class TestGetPassageEntityIds:
    def test_falls_back_to_entities_field(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import get_passage_entity_ids

        ids = get_passage_entity_ids(dress_graph, "passage::opening")
        assert "character::protagonist" in ids
        assert "location::bridge" in ids

    def test_prefers_appears_edges(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import get_passage_entity_ids

        # Add appears edges (entity → passage)
        dress_graph.add_edge("appears", "character::protagonist", "passage::opening")
        # Deliberately omit location::bridge edge
        ids = get_passage_entity_ids(dress_graph, "passage::opening")
        # Should only return entities with appears edges, ignoring entities field
        assert set(ids) == {"character::protagonist"}

    def test_nonexistent_passage(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import get_passage_entity_ids

        assert get_passage_entity_ids(dress_graph, "passage::nope") == []


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

    def test_uses_appears_edges(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_visuals_for_passage

        dress_graph.create_node(
            "entity_visual::protagonist",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "young woman, short dark hair",
            },
        )
        dress_graph.create_node(
            "entity_visual::bridge",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "crumbling stone arch",
            },
        )
        # Only protagonist has an appears edge
        dress_graph.add_edge("appears", "character::protagonist", "passage::opening")
        result = format_entity_visuals_for_passage(dress_graph, "passage::opening")
        assert "protagonist" in result
        assert "bridge" not in result

    def test_no_visuals(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_entity_visuals_for_passage

        assert format_entity_visuals_for_passage(dress_graph, "passage::opening") == ""


# ---------------------------------------------------------------------------
# Batch context formatters
# ---------------------------------------------------------------------------


class TestFormatPassagesBatchForBriefs:
    def test_formats_multiple_passages(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passages_batch_for_briefs

        passage_ids = ["passage::opening"]
        base_scores = {"passage::opening": 3}
        result = format_passages_batch_for_briefs(dress_graph, passage_ids, base_scores)
        assert "opening" in result
        assert "Structural base score: 3" in result

    def test_empty_list(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passages_batch_for_briefs

        result = format_passages_batch_for_briefs(dress_graph, [], {})
        assert result == ""

    def test_missing_passage_graceful(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_passages_batch_for_briefs

        result = format_passages_batch_for_briefs(
            dress_graph, ["passage::nonexistent"], {"passage::nonexistent": 0}
        )
        # Should still produce a section header even if passage context is empty
        assert "nonexistent" in result


class TestFormatAllEntityVisuals:
    def test_deduplicates_across_passages(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_all_entity_visuals

        # Add a second passage with same entity
        dress_graph.create_node(
            "passage::p2",
            {
                "type": "passage",
                "raw_id": "p2",
                "prose": "More prose.",
                "entities": ["character::protagonist"],
            },
        )
        dress_graph.create_node(
            "entity_visual::protagonist",
            {
                "type": "entity_visual",
                "reference_prompt_fragment": "young scholar, dark hair",
            },
        )

        result = format_all_entity_visuals(dress_graph, ["passage::opening", "passage::p2"])
        # Should appear only once despite being in both passages
        assert result.count("protagonist") == 1
        assert "young scholar" in result

    def test_no_visuals(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_context import format_all_entity_visuals

        result = format_all_entity_visuals(dress_graph, ["passage::opening"])
        assert result == ""
