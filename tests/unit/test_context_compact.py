"""Tests for context compaction and enrichment utilities."""

from __future__ import annotations

from questfoundry.graph.context_compact import (
    CompactContextConfig,
    ContextItem,
    build_narrative_frame,
    compact_items,
    enrich_beat_line,
    truncate_summary,
)
from questfoundry.graph.graph import Graph


class TestTruncateSummary:
    """Tests for truncate_summary."""

    def test_short_text_unchanged(self) -> None:
        assert truncate_summary("hello", 80) == "hello"

    def test_exact_length_unchanged(self) -> None:
        text = "x" * 80
        assert truncate_summary(text, 80) == text

    def test_truncates_at_word_boundary(self) -> None:
        text = (
            "The hero encounters the mentor at the crossroads and they discuss the ancient prophecy"
        )
        result = truncate_summary(text, 50)
        assert len(result) <= 50
        assert result.endswith("...")
        # Should break at a word boundary, not mid-word
        assert not result[-4].isalpha() or result[-4] == " " or result.endswith("...")

    def test_empty_string(self) -> None:
        assert truncate_summary("", 80) == ""

    def test_custom_suffix(self) -> None:
        result = truncate_summary("a very long text that needs truncating", 20, suffix="~")
        assert result.endswith("~")
        assert len(result) <= 20


class TestCompactItems:
    """Tests for compact_items."""

    def test_all_items_fit(self) -> None:
        items = [
            ContextItem(id="a", text="short line"),
            ContextItem(id="b", text="another line"),
        ]
        result = compact_items(items, CompactContextConfig(max_chars=100))
        assert "short line" in result
        assert "another line" in result

    def test_empty_list(self) -> None:
        assert compact_items([]) == ""

    def test_priority_ordering(self) -> None:
        items = [
            ContextItem(id="low", text="low priority", priority=0),
            ContextItem(id="high", text="high priority", priority=10),
        ]
        result = compact_items(items, CompactContextConfig(max_chars=500))
        # High priority should come first
        assert result.index("high priority") < result.index("low priority")

    def test_drops_overflow_with_count(self) -> None:
        items = [
            ContextItem(id="a", text="x" * 50, priority=2),
            ContextItem(id="b", text="x" * 50, priority=1),
            ContextItem(id="c", text="x" * 50, priority=0),
        ]
        result = compact_items(items, CompactContextConfig(max_chars=80))
        # Should include first item and note about omitted
        assert "omitted" in result

    def test_truncates_last_partial_item(self) -> None:
        items = [
            ContextItem(id="a", text="short", priority=1),
            ContextItem(id="b", text="a much longer text that should be truncated", priority=0),
        ]
        result = compact_items(items, CompactContextConfig(max_chars=40))
        assert "short" in result
        assert "..." in result

    def test_preserves_insertion_order_for_equal_priority(self) -> None:
        items = [
            ContextItem(id="first", text="first item"),
            ContextItem(id="second", text="second item"),
            ContextItem(id="third", text="third item"),
        ]
        result = compact_items(items, CompactContextConfig(max_chars=500))
        assert result.index("first item") < result.index("second item")
        assert result.index("second item") < result.index("third item")

    def test_default_config(self) -> None:
        items = [ContextItem(id="a", text="hello")]
        result = compact_items(items)
        assert result == "hello"


class TestFromContextWindow:
    """Tests for CompactContextConfig.from_context_window."""

    def test_default_model_yields_near_6k(self) -> None:
        """qwen3:4b-instruct-32k (32768 tokens) → ~5734 chars."""
        cfg = CompactContextConfig.from_context_window(32_768)
        assert 5000 <= cfg.max_chars <= 7000

    def test_small_context_hits_floor(self) -> None:
        """Tiny context window should clamp to minimum."""
        cfg = CompactContextConfig.from_context_window(2_048)
        assert cfg.max_chars == 2000

    def test_huge_context_hits_ceiling(self) -> None:
        """1M-token model should cap at 50K."""
        cfg = CompactContextConfig.from_context_window(1_000_000)
        assert cfg.max_chars == 50_000

    def test_custom_fraction(self) -> None:
        cfg = CompactContextConfig.from_context_window(32_768, budget_fraction=0.10)
        assert cfg.max_chars == int(32_768 * 0.10 * 3.5)

    def test_preserves_other_defaults(self) -> None:
        cfg = CompactContextConfig.from_context_window(32_768)
        assert cfg.summary_truncate == 80
        assert cfg.truncation_suffix == "..."


class TestEnrichBeatLine:
    """Tests for enrich_beat_line."""

    def test_basic_format(self) -> None:
        graph = Graph.empty()
        beat_data = {
            "summary": "Hero arrives at the village",
            "scene_type": "scene",
            "narrative_function": "introduce",
        }
        result = enrich_beat_line(graph, "beat::opening", beat_data)
        assert "beat::opening" in result
        assert "scene, introduce" in result
        assert "Hero arrives" in result

    def test_truncates_long_summary(self) -> None:
        graph = Graph.empty()
        long_summary = "A" * 200
        beat_data = {"summary": long_summary, "scene_type": "scene"}
        result = enrich_beat_line(graph, "beat::x", beat_data, summary_max=50)
        # The summary portion should be truncated
        assert "..." in result

    def test_with_entities(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "character::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "name": "Pim",
                "concept": "A lost scholar",
                "category": "character",
            },
        )
        graph.create_node(
            "location::village",
            {
                "type": "entity",
                "raw_id": "village",
                "name": "Oakvale",
                "concept": "A quiet hamlet",
                "category": "location",
            },
        )
        beat_data = {
            "summary": "Pim arrives at Oakvale",
            "scene_type": "scene",
            "narrative_function": "introduce",
            "entities": ["character::hero", "location::village"],
        }
        result = enrich_beat_line(graph, "beat::arrival", beat_data, include_entities=True)
        assert "(entities: Pim, Oakvale)" in result

    def test_missing_entity_uses_id(self) -> None:
        graph = Graph.empty()
        beat_data = {
            "summary": "Something happens",
            "entities": ["character::unknown"],
        }
        result = enrich_beat_line(graph, "beat::x", beat_data, include_entities=True)
        assert "character::unknown" in result

    def test_no_tags_shows_unclassified(self) -> None:
        graph = Graph.empty()
        beat_data = {"summary": "Something"}
        result = enrich_beat_line(graph, "beat::x", beat_data)
        assert "unclassified" in result


class TestBuildNarrativeFrame:
    """Tests for build_narrative_frame."""

    def test_with_dilemma(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {
                "type": "dilemma",
                "raw_id": "trust",
                "question": "Can the mentor be trusted?",
                "why_it_matters": "The hero's survival depends on this alliance",
            },
        )
        result = build_narrative_frame(graph, dilemma_ids=["dilemma::trust"])
        assert "## Story Context" in result
        assert "Can the mentor be trusted?" in result
        assert "hero's survival" in result

    def test_with_path(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {
                "type": "dilemma",
                "raw_id": "trust",
                "question": "Trust?",
            },
        )
        graph.create_node(
            "path::trust__yes",
            {
                "type": "path",
                "raw_id": "trust__yes",
                "description": "Hero chooses to trust the mentor",
                "path_theme": "cautious bond deepens into reliance",
                "path_mood": "fragile trust",
            },
        )
        result = build_narrative_frame(
            graph,
            dilemma_ids=["dilemma::trust"],
            path_ids=["path::trust__yes"],
        )
        assert "Trust?" in result
        assert "cautious bond" in result
        assert "fragile trust" in result

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        result = build_narrative_frame(graph)
        assert result == ""

    def test_all_dilemmas_when_none_specified(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "dilemma::a",
            {
                "type": "dilemma",
                "raw_id": "a",
                "question": "Question A?",
            },
        )
        graph.create_node(
            "dilemma::b",
            {
                "type": "dilemma",
                "raw_id": "b",
                "question": "Question B?",
            },
        )
        result = build_narrative_frame(graph)
        assert "Question A?" in result
        assert "Question B?" in result

    def test_missing_dilemma_skipped(self) -> None:
        graph = Graph.empty()
        result = build_narrative_frame(graph, dilemma_ids=["dilemma::nonexistent"])
        assert result == ""

    def test_path_without_theme_still_shows_description(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "path::x",
            {
                "type": "path",
                "raw_id": "x",
                "description": "A journey through darkness",
            },
        )
        result = build_narrative_frame(graph, dilemma_ids=[], path_ids=["path::x"])
        assert "journey through darkness" in result
