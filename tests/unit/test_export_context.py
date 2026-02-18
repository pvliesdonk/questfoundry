"""Tests for ExportContext builder."""

from __future__ import annotations

import pytest

from questfoundry.export.context import build_export_context
from questfoundry.graph.graph import Graph


def _minimal_graph() -> Graph:
    """Build a minimal graph with passages and choices."""
    g = Graph()
    g.create_node(
        "passage::intro",
        {
            "type": "passage",
            "raw_id": "intro",
            "prose": "You stand at the gates.",
        },
    )
    g.create_node(
        "passage::choice_a",
        {
            "type": "passage",
            "raw_id": "choice_a",
            "prose": "You enter the castle.",
        },
    )
    g.create_node(
        "passage::choice_b",
        {
            "type": "passage",
            "raw_id": "choice_b",
            "prose": "You flee into the forest.",
        },
    )
    g.create_node(
        "choice::intro_to_a",
        {
            "type": "choice",
            "from_passage": "passage::intro",
            "to_passage": "passage::choice_a",
            "label": "Enter the castle",
            "requires_codewords": [],
            "grants": ["codeword::entered_castle"],
        },
    )
    g.create_node(
        "choice::intro_to_b",
        {
            "type": "choice",
            "from_passage": "passage::intro",
            "to_passage": "passage::choice_b",
            "label": "Flee to the forest",
            "requires_codewords": [],
            "grants": [],
        },
    )
    # Edges for choice connectivity
    g.add_edge("choice_from", "choice::intro_to_a", "passage::intro")
    g.add_edge("choice_to", "choice::intro_to_a", "passage::choice_a")
    g.add_edge("choice_from", "choice::intro_to_b", "passage::intro")
    g.add_edge("choice_to", "choice::intro_to_b", "passage::choice_b")
    return g


def _graph_with_entities(g: Graph) -> Graph:
    """Add entity and codeword nodes to a graph."""
    g.create_node(
        "entity::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_type": "character",
            "concept": "The protagonist",
            "overlays": [{"when": ["codeword::entered_castle"], "details": {"mood": "brave"}}],
        },
    )
    g.create_node(
        "codeword::entered_castle",
        {
            "type": "codeword",
            "raw_id": "entered_castle",
            "codeword_type": "granted",
            "tracks": "consequence::entered",
        },
    )
    return g


def _graph_with_dress(g: Graph) -> Graph:
    """Add DRESS stage nodes to a graph."""
    g.create_node(
        "art_direction::main",
        {
            "type": "art_direction",
            "style": "watercolor",
            "medium": "digital painting",
            "palette": ["blue", "gold"],
        },
    )
    g.create_node(
        "illustration::intro_scene",
        {
            "type": "illustration",
            "asset": "assets/abc123.png",
            "caption": "The gates loom before you.",
            "category": "scene",
        },
    )
    g.add_edge("Depicts", "illustration::intro_scene", "passage::intro")
    g.create_node(
        "codex::hero_rank1",
        {
            "type": "codex_entry",
            "rank": 1,
            "visible_when": [],
            "content": "A brave adventurer.",
        },
    )
    g.create_node(
        "entity::hero",
        {
            "type": "entity",
            "raw_id": "hero",
            "entity_type": "character",
            "concept": "The protagonist",
        },
    )
    g.add_edge("HasEntry", "codex::hero_rank1", "entity::hero")
    return g


class TestBuildExportContext:
    def test_minimal_graph(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test-story")

        assert ctx.title == "test-story"
        assert len(ctx.passages) == 3
        assert len(ctx.choices) == 2

    def test_start_passage_detected(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        start_passages = [p for p in ctx.passages if p.is_start]
        assert len(start_passages) == 1
        assert start_passages[0].id == "passage::intro"

    def test_start_passage_ignores_return_links(self) -> None:
        """Return links (spoke→hub) should not prevent start passage detection."""
        g = _minimal_graph()
        g.create_node(
            "passage::spoke_0",
            {
                "type": "passage",
                "raw_id": "spoke_0",
                "prose": "You look around.",
            },
        )
        g.create_node(
            "choice::intro_to_spoke_0",
            {
                "type": "choice",
                "from_passage": "passage::intro",
                "to_passage": "passage::spoke_0",
                "label": "Look around",
                "requires_codewords": [],
                "grants": [],
            },
        )
        g.add_edge("choice_from", "choice::intro_to_spoke_0", "passage::intro")
        g.add_edge("choice_to", "choice::intro_to_spoke_0", "passage::spoke_0")
        g.create_node(
            "choice::spoke_0_return",
            {
                "type": "choice",
                "from_passage": "passage::spoke_0",
                "to_passage": "passage::intro",
                "label": "Return",
                "is_return": True,
                "requires_codewords": [],
                "grants": [],
            },
        )
        g.add_edge("choice_from", "choice::spoke_0_return", "passage::spoke_0")
        g.add_edge("choice_to", "choice::spoke_0_return", "passage::intro")

        ctx = build_export_context(g, "test")
        start_passages = [p for p in ctx.passages if p.is_start]
        assert len(start_passages) == 1
        assert start_passages[0].id == "passage::intro"

    def test_ending_passages_detected(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        endings = [p for p in ctx.passages if p.is_ending]
        assert len(endings) == 2
        ending_ids = {p.id for p in endings}
        assert ending_ids == {"passage::choice_a", "passage::choice_b"}

    def test_choice_data(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        castle_choice = next(c for c in ctx.choices if c.label == "Enter the castle")
        assert castle_choice.from_passage == "passage::intro"
        assert castle_choice.to_passage == "passage::choice_a"
        assert castle_choice.grants == ["codeword::entered_castle"]

    def test_entities_and_codewords(self) -> None:
        g = _graph_with_entities(_minimal_graph())
        ctx = build_export_context(g, "test")

        assert len(ctx.entities) == 1
        assert ctx.entities[0].id == "entity::hero"
        assert ctx.entities[0].entity_type == "character"

        assert len(ctx.codewords) == 1
        assert ctx.codewords[0].id == "codeword::entered_castle"
        assert ctx.codewords[0].tracks == "consequence::entered"

    def test_dress_nodes_present(self) -> None:
        g = _graph_with_dress(_minimal_graph())
        ctx = build_export_context(g, "test")

        assert ctx.art_direction is not None
        assert ctx.art_direction["style"] == "watercolor"

        assert len(ctx.illustrations) == 1
        assert ctx.illustrations[0].passage_id == "passage::intro"
        assert ctx.illustrations[0].asset_path == "assets/abc123.png"

        assert len(ctx.codex_entries) == 1
        assert ctx.codex_entries[0].entity_id == "entity::hero"
        assert ctx.codex_entries[0].rank == 1

    def test_no_dress_nodes(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        assert ctx.art_direction is None
        assert ctx.illustrations == []
        assert ctx.codex_entries == []

    def test_empty_graph_raises(self) -> None:
        g = Graph()
        with pytest.raises(ValueError, match="no passages"):
            build_export_context(g, "test")

    def test_passage_prose_preserved(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        intro = next(p for p in ctx.passages if p.id == "passage::intro")
        assert intro.prose == "You stand at the gates."

    def test_cover_illustration_separated(self) -> None:
        g = _graph_with_dress(_minimal_graph())
        # Add a standalone cover illustration (no Depicts edge)
        g.create_node(
            "illustration::cover",
            {
                "type": "illustration",
                "asset": "assets/cover.png",
                "caption": "Cover art",
                "category": "cover",
            },
        )
        ctx = build_export_context(g, "test")

        assert ctx.cover is not None
        assert ctx.cover.asset_path == "assets/cover.png"
        assert ctx.cover.category == "cover"
        # Cover should NOT appear in passage illustrations
        assert all(ill.category != "cover" for ill in ctx.illustrations)

    def test_no_cover_when_absent(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        assert ctx.cover is None

    def test_language_default_english(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        assert ctx.language == "en"

    def test_language_passthrough(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test", language="nl")

        assert ctx.language == "nl"
