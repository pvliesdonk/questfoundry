"""Tests for FILL artifact extraction."""

from __future__ import annotations

from questfoundry.artifacts.enrichment import extract_fill_artifact
from questfoundry.graph.graph import Graph


def _make_fill_graph() -> Graph:
    """Create a graph with FILL output for artifact extraction testing."""
    g = Graph.empty()

    # Voice document
    g.create_node(
        "voice::voice",
        {
            "type": "voice",
            "raw_id": "voice",
            "pov": "third_limited",
            "tense": "past",
            "voice_register": "literary",
            "sentence_rhythm": "varied",
            "tone_words": ["terse", "atmospheric", "melancholic"],
            "avoid_words": ["suddenly", "very"],
            "avoid_patterns": ["adverb-heavy dialogue tags"],
        },
    )

    # Passages with prose
    g.create_node(
        "passage::p1",
        {
            "type": "passage",
            "raw_id": "p1",
            "from_beat": "beat::b1",
            "prose": "Kay entered the crumbling tower, dust motes swirling in the pale light.",
        },
    )
    g.create_node(
        "passage::p2",
        {
            "type": "passage",
            "raw_id": "p2",
            "from_beat": "beat::b2",
            "prose": "The hall stretched before Kay, its shadows alive with whispered echoes.",
            "flag": "incompatible_states",
        },
    )
    # Passage without prose (should be excluded)
    g.create_node(
        "passage::p3",
        {
            "type": "passage",
            "raw_id": "p3",
            "from_beat": "beat::b3",
        },
    )

    return g


class TestExtractFillArtifact:
    def test_extracts_voice_document(self) -> None:
        graph = _make_fill_graph()
        artifact = extract_fill_artifact(graph)
        voice = artifact["voice_document"]
        assert voice["pov"] == "third_limited"
        assert voice["tense"] == "past"
        assert voice["voice_register"] == "literary"
        assert "tone_words" in voice
        assert "type" not in voice  # graph metadata excluded
        assert "raw_id" not in voice

    def test_extracts_filled_passages_only(self) -> None:
        graph = _make_fill_graph()
        artifact = extract_fill_artifact(graph)
        passages = artifact["passages"]
        # Only p1 and p2 have prose; p3 is excluded
        assert len(passages) == 2
        passage_ids = [p["passage_id"] for p in passages]
        assert "passage::p1" in passage_ids
        assert "passage::p2" in passage_ids
        assert "passage::p3" not in passage_ids

    def test_passage_prose_snippets(self) -> None:
        graph = _make_fill_graph()
        artifact = extract_fill_artifact(graph)
        passages = artifact["passages"]
        p1 = next(p for p in passages if p["passage_id"] == "passage::p1")
        assert "prose_snippet" in p1
        assert "prose_length" in p1
        assert p1["prose_length"] > 0

    def test_passage_flags_included(self) -> None:
        graph = _make_fill_graph()
        artifact = extract_fill_artifact(graph)
        passages = artifact["passages"]
        p2 = next(p for p in passages if p["passage_id"] == "passage::p2")
        assert p2["flag"] == "incompatible_states"

    def test_prose_truncation(self) -> None:
        graph = Graph.empty()
        long_prose = "A" * 300
        graph.create_node(
            "passage::long",
            {"type": "passage", "raw_id": "long", "from_beat": "beat::b1", "prose": long_prose},
        )
        artifact = extract_fill_artifact(graph)
        passages = artifact["passages"]
        assert len(passages) == 1
        assert passages[0]["prose_snippet"].endswith("...")
        assert len(passages[0]["prose_snippet"]) == 203  # 200 + "..."
        assert passages[0]["prose_length"] == 300

    def test_review_summary(self) -> None:
        graph = _make_fill_graph()
        artifact = extract_fill_artifact(graph)
        summary = artifact["review_summary"]
        assert summary["total_passages"] == 3
        assert summary["passages_with_prose"] == 2
        assert summary["passages_flagged"] == 1

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        artifact = extract_fill_artifact(graph)
        assert artifact["voice_document"] == {}
        assert artifact["passages"] == []
        assert artifact["review_summary"]["total_passages"] == 0

    def test_no_voice_node(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "b1", "prose": "Some text."},
        )
        artifact = extract_fill_artifact(graph)
        assert artifact["voice_document"] == {}
        assert len(artifact["passages"]) == 1
