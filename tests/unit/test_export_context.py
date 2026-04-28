"""Tests for ExportContext builder."""

from __future__ import annotations

import logging

import pytest

from questfoundry.export.context import build_export_context
from questfoundry.graph.graph import Graph

_CONTEXT_LOGGER = "questfoundry.export.context"


def _has_event(caplog: pytest.LogCaptureFixture, event: str) -> bool:
    """Return True if any captured record's message contains ``event``.

    Mirrors the project convention (see test_polish_llm_phases.py et al.):
    use ``caplog.records`` rather than ``caplog.text`` so the assertion
    survives processor-chain or formatter changes in the structlog setup.
    """
    return any(event in str(r.message) for r in caplog.records)


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
    # Choices are stored as graph edges (POLISH writes them via
    # `graph.add_edge("choice", ...)`) — see #1532 for the regression
    # where the export context read non-existent choice nodes instead.
    g.add_edge(
        "choice",
        "passage::intro",
        "passage::choice_a",
        label="Enter the castle",
        requires=[],
        grants=["codeword::entered_castle"],
    )
    g.add_edge(
        "choice",
        "passage::intro",
        "passage::choice_b",
        label="Flee to the forest",
        requires=[],
        grants=[],
    )
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
            "derived_from": "consequence::entered",
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

    def test_choices_read_from_edges_not_nodes(self) -> None:
        # Regression for #1532: POLISH stores choices as edges, not nodes.
        # The earlier _extract_choices read non-existent choice nodes and
        # silently produced 0 choices, leaving every export with no
        # navigable links.
        g = _minimal_graph()
        # Sanity: graph really has zero choice nodes and non-zero choice edges.
        assert g.get_nodes_by_type("choice") == {}
        assert len(g.get_edges(edge_type="choice")) == 2
        ctx = build_export_context(g, "regression-1532")
        assert len(ctx.choices) == 2
        labels = {c.label for c in ctx.choices}
        assert labels == {"Enter the castle", "Flee to the forest"}

    def test_choice_requires_round_trips_polish_edge_key(self) -> None:
        # Regression for #1532 follow-up: POLISH writes the gate-condition
        # key as `"requires"` (see _create_choice_edge in
        # pipeline/stages/polish/deterministic.py:1430). The export must
        # read the same key — earlier code looked under legacy names and
        # silently dropped POLISH's gate values.
        g = Graph()
        g.create_node("passage::a", {"type": "passage", "raw_id": "a", "prose": "."})
        g.create_node("passage::b", {"type": "passage", "raw_id": "b", "prose": "."})
        g.add_edge(
            "choice",
            "passage::a",
            "passage::b",
            label="Open the gate",
            requires=["state_flag::has_key"],  # POLISH's actual key name
            grants=[],
        )
        ctx = build_export_context(g, "test")
        assert len(ctx.choices) == 1
        assert ctx.choices[0].requires == ["state_flag::has_key"]

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
        g.add_edge(
            "choice",
            "passage::intro",
            "passage::spoke_0",
            label="Look around",
            requires=[],
            grants=[],
        )
        g.add_edge(
            "choice",
            "passage::spoke_0",
            "passage::intro",
            label="Return",
            is_return=True,
            requires=[],
            grants=[],
        )

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
        assert ctx.codewords[0].derived_from == "consequence::entered"

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

    def test_state_flag_projection_soft_only(self) -> None:
        """Soft dilemma state flags are projected as codewords; hard ones are not."""
        g = _minimal_graph()
        # Dilemmas
        g.create_node(
            "dilemma::trust",
            {"type": "dilemma", "raw_id": "trust", "dilemma_role": "soft"},
        )
        g.create_node(
            "dilemma::loyalty",
            {"type": "dilemma", "raw_id": "loyalty", "dilemma_role": "hard"},
        )
        # State flags
        g.create_node(
            "state_flag::trusted_mentor",
            {
                "type": "state_flag",
                "raw_id": "trusted_mentor",
                "dilemma_id": "dilemma::trust",
                "codeword_type": "granted",
                "derived_from": "consequence::trust_given",
            },
        )
        g.create_node(
            "state_flag::loyal_to_crown",
            {
                "type": "state_flag",
                "raw_id": "loyal_to_crown",
                "dilemma_id": "dilemma::loyalty",
                "codeword_type": "granted",
                "derived_from": "consequence::loyalty",
            },
        )
        ctx = build_export_context(g, "test")
        # Only soft dilemma flag should be exported
        assert len(ctx.codewords) == 1
        assert ctx.codewords[0].id == "state_flag::trusted_mentor"
        assert ctx.codewords[0].derived_from == "consequence::trust_given"

    def test_hard_only_state_flags_yield_no_codewords(self) -> None:
        """Hard-only state flags produce empty codewords, not a legacy fallback."""
        g = _minimal_graph()
        g.create_node(
            "dilemma::routing",
            {"type": "dilemma", "raw_id": "routing", "dilemma_role": "hard"},
        )
        g.create_node(
            "state_flag::branch_taken",
            {
                "type": "state_flag",
                "raw_id": "branch_taken",
                "dilemma_id": "dilemma::routing",
                "codeword_type": "granted",
            },
        )
        ctx = build_export_context(g, "test")
        assert ctx.codewords == []

    def test_legacy_codeword_fallback(self) -> None:
        """When no state_flag nodes exist, legacy codeword nodes are used."""
        g = _graph_with_entities(_minimal_graph())
        ctx = build_export_context(g, "test")
        assert len(ctx.codewords) == 1
        assert ctx.codewords[0].id == "codeword::entered_castle"

    def test_language_default_english(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")

        assert ctx.language == "en"

    def test_language_passthrough(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test", language="nl")

        assert ctx.language == "nl"


class TestCodewordPlayabilityWarning:
    """R-1.7: codeword count > 10 must trigger a WARNING."""

    @staticmethod
    def _add_soft_dilemma_with_n_flags(g: Graph, n: int) -> None:
        g.create_node(
            "dilemma::wide",
            {"type": "dilemma", "raw_id": "wide", "dilemma_role": "soft"},
        )
        for i in range(n):
            g.create_node(
                f"state_flag::flag_{i}",
                {
                    "type": "state_flag",
                    "raw_id": f"flag_{i}",
                    "dilemma_id": "dilemma::wide",
                    "codeword_type": "granted",
                },
            )

    def test_at_threshold_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Exactly 10 codewords sits at the limit — no warning yet."""
        g = _minimal_graph()
        self._add_soft_dilemma_with_n_flags(g, 10)

        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")

        assert len(ctx.codewords) == 10
        assert not _has_event(caplog, "codeword_count_exceeds_threshold")

    def test_above_threshold_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """11 codewords exceeds the playability threshold — must warn (R-1.7)."""
        g = _minimal_graph()
        self._add_soft_dilemma_with_n_flags(g, 11)

        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")

        assert len(ctx.codewords) == 11
        assert _has_event(caplog, "codeword_count_exceeds_threshold")

    def test_zero_codewords_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        g = _minimal_graph()
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")
        assert ctx.codewords == []
        assert not _has_event(caplog, "codeword_count_exceeds_threshold")


class TestPartialDressWarning:
    """R-3.9: art_direction missing required fields must warn (graceful, not silent)."""

    def test_complete_art_direction_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        g = _minimal_graph()
        g.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "watercolor",
                "medium": "digital painting",
                "palette": ["blue", "gold"],
                "composition_notes": "wide framing",
                "style_exclusions": "no text overlays",
                "aspect_ratio": "16:9",
            },
        )
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")

        assert ctx.art_direction is not None
        assert not _has_event(caplog, "art_direction_partial")

    def test_missing_palette_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        g = _minimal_graph()
        g.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "watercolor",
                "medium": "digital painting",
                # palette omitted
                "composition_notes": "wide framing",
                "style_exclusions": "no text overlays",
                "aspect_ratio": "16:9",
            },
        )
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")

        # Partial data still propagates (graceful degradation)
        assert ctx.art_direction is not None
        assert _has_event(caplog, "art_direction_partial")
        assert _has_event(caplog, "palette")

    def test_blank_string_field_counts_as_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Whitespace-only style is as bad as missing — both warn."""
        g = _minimal_graph()
        g.create_node(
            "art_direction::main",
            {
                "type": "art_direction",
                "style": "   ",
                "medium": "digital painting",
                "palette": ["blue"],
                "composition_notes": "wide framing",
                "style_exclusions": "no text overlays",
                "aspect_ratio": "16:9",
            },
        )
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            build_export_context(g, "test")

        assert _has_event(caplog, "art_direction_partial")
        assert _has_event(caplog, "'style'")

    def test_multiple_missing_fields_all_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """When several fields are missing, the warning must list ALL of them.

        Otherwise the user fixes one, reruns DRESS, hits the warning again
        for the next field — fragile and unfriendly.
        """
        g = _minimal_graph()
        # Only style + medium present; the other four fields all missing
        g.create_node(
            "art_direction::main",
            {"type": "art_direction", "style": "ink", "medium": "digital"},
        )
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            build_export_context(g, "test")

        assert _has_event(caplog, "art_direction_partial")
        for missing in ("palette", "composition_notes", "style_exclusions", "aspect_ratio"):
            assert _has_event(caplog, missing), f"missing field {missing!r} not in warning"

    def test_no_art_direction_node_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """DRESS skipped entirely → art_direction=None, no partial warning (R-3.8)."""
        g = _minimal_graph()
        with caplog.at_level(logging.WARNING, logger=_CONTEXT_LOGGER):
            ctx = build_export_context(g, "test")
        assert ctx.art_direction is None
        assert not _has_event(caplog, "art_direction_partial")


class TestVoiceExtraction:
    """R-3.3 prep: ExportContext exposes the FILL voice document so the
    HTML exporter (and future format-specific styling) can react to it.
    """

    def test_voice_node_extracted(self) -> None:
        g = _minimal_graph()
        g.create_node(
            "voice::voice",
            {
                "type": "voice",
                "raw_id": "voice",
                "story_title": "The Test",
                "pov": "third_person_limited",
                "tense": "past",
                "voice_register": "literary",
                "sentence_rhythm": "flowing",
                "tone_words": ["wry"],
            },
        )
        ctx = build_export_context(g, "test")
        assert ctx.voice is not None
        assert ctx.voice["voice_register"] == "literary"
        assert ctx.voice["sentence_rhythm"] == "flowing"
        # Internal-only fields stripped
        assert "type" not in ctx.voice
        assert "raw_id" not in ctx.voice

    def test_no_voice_node_returns_none(self) -> None:
        g = _minimal_graph()
        ctx = build_export_context(g, "test")
        assert ctx.voice is None
