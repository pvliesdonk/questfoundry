"""Tests for GROW context-formatting helpers."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_context import (
    format_valid_beat_ids_by_dilemma,
    format_valid_entity_ids_by_category,
    format_valid_state_flag_ids_by_dilemma,
)


class TestFormatValidBeatIdsByDilemma:
    """Pin the grouped Valid IDs layout for Phase 3 + Phase 4a (#1476)."""

    def _seed_graph(self) -> Graph:
        """Build a graph with two dilemmas, two paths each, plus a Y-shape
        pre-commit beat that belongs to both paths of one dilemma, and a
        structural beat with no ``belongs_to`` edge."""
        g = Graph()
        for did in ("mentor_trust", "archive_nature"):
            g.create_node(f"dilemma::{did}", {"type": "dilemma", "raw_id": did})
        # Two paths per dilemma
        g.create_node(
            "path::mt_canonical",
            {"type": "path", "raw_id": "mt_canonical", "dilemma_id": "dilemma::mentor_trust"},
        )
        g.create_node(
            "path::mt_alt",
            {"type": "path", "raw_id": "mt_alt", "dilemma_id": "dilemma::mentor_trust"},
        )
        g.create_node(
            "path::an_canonical",
            {"type": "path", "raw_id": "an_canonical", "dilemma_id": "dilemma::archive_nature"},
        )
        # Beats
        for bid in (
            "beat::mt_a",
            "beat::mt_b",
            "beat::mt_pre",
            "beat::an_a",
            "beat::orphan",
        ):
            g.create_node(bid, {"type": "beat", "raw_id": bid.split("::", 1)[1]})

        # Single-dilemma mappings
        g.add_edge("belongs_to", "beat::mt_a", "path::mt_canonical")
        g.add_edge("belongs_to", "beat::mt_b", "path::mt_alt")
        g.add_edge("belongs_to", "beat::an_a", "path::an_canonical")
        # Y-shape pre-commit: belongs to BOTH paths of mentor_trust
        g.add_edge("belongs_to", "beat::mt_pre", "path::mt_canonical")
        g.add_edge("belongs_to", "beat::mt_pre", "path::mt_alt")
        # beat::orphan has no belongs_to edge

        return g

    def test_groups_by_dilemma(self) -> None:
        g = self._seed_graph()
        beat_ids = {"beat::mt_a", "beat::mt_b", "beat::an_a"}
        result = format_valid_beat_ids_by_dilemma(g, beat_ids)
        assert "- `dilemma::archive_nature`: `beat::an_a`" in result
        assert "- `dilemma::mentor_trust`: `beat::mt_a`, `beat::mt_b`" in result
        # No cross-dilemma or unmapped buckets when all beats are single-dilemma.
        # Use the real bucket label so a future rename trips this test.
        assert "(spans multiple dilemmas)" not in result
        assert "(unmapped)" not in result

    def test_pre_commit_beat_groups_under_its_dilemma(self) -> None:
        """Y-shape pre-commit beats have multiple ``belongs_to`` edges, but all
        to paths of the SAME dilemma (Story Graph Ontology Part 8: "Multi-
        belongs_to only within one dilemma"). They land under that dilemma's
        bucket, NOT in the cross-dilemma bucket."""
        g = self._seed_graph()
        beat_ids = {"beat::mt_a", "beat::mt_pre"}
        result = format_valid_beat_ids_by_dilemma(g, beat_ids)
        # Both beats land under mentor_trust (mt_pre has 2 belongs_to but to
        # the SAME dilemma's paths — single-dilemma in Y-shape semantics).
        assert "- `dilemma::mentor_trust`: `beat::mt_a`, `beat::mt_pre`" in result
        # Cross-dilemma bucket MUST be absent for this fixture:
        assert "(spans multiple dilemmas)" not in result

    def test_cross_dilemma_beat_lands_in_multi_bucket(self) -> None:
        """A beat with ``belongs_to`` edges to paths of DIFFERENT dilemmas
        is a Story Graph Ontology Part 8 violation. The helper surfaces such
        anomalies in a dedicated bucket rather than silently grouping them
        under one of the dilemmas."""
        g = self._seed_graph()
        # Synthetic spec-violation: a beat that belongs to paths of two
        # different dilemmas. Production graph code should never produce
        # this, but defensive surfacing helps catch upstream regressions.
        g.create_node("beat::cross", {"type": "beat", "raw_id": "cross"})
        g.add_edge("belongs_to", "beat::cross", "path::mt_canonical")
        g.add_edge("belongs_to", "beat::cross", "path::an_canonical")

        result = format_valid_beat_ids_by_dilemma(g, {"beat::mt_a", "beat::cross", "beat::an_a"})
        # Single-dilemma beats land under their dilemma:
        assert "- `dilemma::mentor_trust`: `beat::mt_a`" in result
        assert "- `dilemma::archive_nature`: `beat::an_a`" in result
        # Cross-dilemma beat lands in the spec-violation bucket:
        assert "- (spans multiple dilemmas): `beat::cross`" in result

    def test_unmapped_beat_lands_in_unmapped_bucket(self) -> None:
        g = self._seed_graph()
        result = format_valid_beat_ids_by_dilemma(g, {"beat::orphan"})
        assert "- (unmapped): `beat::orphan`" in result
        assert "dilemma::" not in result

    def test_empty_input_returns_empty_string(self) -> None:
        g = self._seed_graph()
        assert format_valid_beat_ids_by_dilemma(g, set()) == ""

    def test_only_listed_beats_grouped(self) -> None:
        """Beats not in the input set MUST NOT appear, even if the graph has
        ``belongs_to`` edges for them. The helper groups whatever the caller
        passes — it doesn't consult the graph for the canonical beat list."""
        g = self._seed_graph()
        result = format_valid_beat_ids_by_dilemma(g, {"beat::mt_a"})
        assert "beat::mt_a" in result
        assert "beat::mt_b" not in result
        assert "beat::an_a" not in result
        assert "beat::mt_pre" not in result

    def test_dilemmas_sorted_alphabetically(self) -> None:
        """Dilemma buckets sort lexicographically so identical inputs render
        identically across runs (deterministic prompt context)."""
        g = self._seed_graph()
        result = format_valid_beat_ids_by_dilemma(g, {"beat::an_a", "beat::mt_a"})
        an_idx = result.index("dilemma::archive_nature")
        mt_idx = result.index("dilemma::mentor_trust")
        assert an_idx < mt_idx

    def test_buckets_render_in_dilemma_then_multi_then_unmapped_order(self) -> None:
        """When all three bucket kinds are present, render order is:
        single-dilemma buckets → cross-dilemma → unmapped."""
        g = self._seed_graph()
        # Synthetic cross-dilemma beat (same construction as
        # test_cross_dilemma_beat_lands_in_multi_bucket).
        g.create_node("beat::cross", {"type": "beat", "raw_id": "cross"})
        g.add_edge("belongs_to", "beat::cross", "path::mt_canonical")
        g.add_edge("belongs_to", "beat::cross", "path::an_canonical")

        result = format_valid_beat_ids_by_dilemma(g, {"beat::mt_a", "beat::cross", "beat::orphan"})
        single_idx = result.index("- `dilemma::mentor_trust`")
        multi_idx = result.index("- (spans multiple dilemmas)")
        unmapped_idx = result.index("- (unmapped)")
        assert single_idx < multi_idx < unmapped_idx


class TestFormatValidEntityIdsByCategory:
    """Pin the grouped entity-Valid-IDs layout for Phase 8c (#1482)."""

    def _seed_graph(self) -> Graph:
        g = Graph()
        g.create_node(
            "character::clara_yu",
            {"type": "entity", "raw_id": "clara_yu", "entity_category": "character"},
        )
        g.create_node(
            "character::simon_blackwood",
            {"type": "entity", "raw_id": "simon_blackwood", "entity_category": "character"},
        )
        g.create_node(
            "location::manor",
            {"type": "entity", "raw_id": "manor", "entity_category": "location"},
        )
        g.create_node(
            "object::brooch",
            {"type": "entity", "raw_id": "brooch", "entity_category": "object"},
        )
        # Entity with no entity_category — falls through to entity_type.
        g.create_node(
            "entity::stray",
            {"type": "entity", "raw_id": "stray", "entity_type": "faction"},
        )
        # Entity with neither category nor type — surfaces as "unknown".
        g.create_node(
            "entity::orphan",
            {"type": "entity", "raw_id": "orphan"},
        )
        return g

    def test_groups_by_entity_category(self) -> None:
        g = self._seed_graph()
        result = format_valid_entity_ids_by_category(
            g,
            [
                "character::clara_yu",
                "character::simon_blackwood",
                "location::manor",
                "object::brooch",
            ],
        )
        # One bullet per category, sorted alphabetically by category, with
        # backtick-wrapped IDs.
        assert "- character: `character::clara_yu`, `character::simon_blackwood`" in result
        assert "- location: `location::manor`" in result
        assert "- object: `object::brooch`" in result

    def test_falls_back_to_entity_type_when_category_missing(self) -> None:
        g = self._seed_graph()
        result = format_valid_entity_ids_by_category(g, ["entity::stray"])
        assert "- faction: `entity::stray`" in result

    def test_unknown_bucket_for_entities_with_no_classification(self) -> None:
        g = self._seed_graph()
        result = format_valid_entity_ids_by_category(g, ["entity::orphan"])
        assert "- unknown: `entity::orphan`" in result

    def test_categories_render_alphabetically(self) -> None:
        g = self._seed_graph()
        result = format_valid_entity_ids_by_category(
            g,
            ["object::brooch", "character::clara_yu", "location::manor"],
        )
        char_idx = result.index("- character:")
        loc_idx = result.index("- location:")
        obj_idx = result.index("- object:")
        assert char_idx < loc_idx < obj_idx

    def test_empty_input_returns_empty_string(self) -> None:
        g = self._seed_graph()
        assert format_valid_entity_ids_by_category(g, []) == ""


class TestFormatValidStateFlagIdsByDilemma:
    """Pin the grouped state_flag-Valid-IDs layout for Phase 8c (#1482).

    The helper is graph-free — it consumes a ``flag_to_dilemma`` map the
    caller already builds from consequence → path → dilemma traversal in
    ``_phase_8c_overlays`` (line 890)."""

    def test_groups_by_dilemma(self) -> None:
        result = format_valid_state_flag_ids_by_dilemma(
            [
                "state_flag::mentor_friendly_committed",
                "state_flag::mentor_hostile_committed",
                "state_flag::artifact_safe_committed",
            ],
            flag_to_dilemma={
                "state_flag::mentor_friendly_committed": "dilemma::mentor_trust",
                "state_flag::mentor_hostile_committed": "dilemma::mentor_trust",
                "state_flag::artifact_safe_committed": "dilemma::artifact_nature",
            },
        )
        assert "- `dilemma::artifact_nature`: `state_flag::artifact_safe_committed`" in result
        assert (
            "- `dilemma::mentor_trust`: `state_flag::mentor_friendly_committed`,"
            " `state_flag::mentor_hostile_committed`" in result
        )
        # No unmapped bucket when every flag has a dilemma.
        assert "(unmapped)" not in result

    def test_unmapped_bucket_for_flags_without_dilemma(self) -> None:
        result = format_valid_state_flag_ids_by_dilemma(
            ["state_flag::raw_flag"],
            flag_to_dilemma={},
        )
        assert "- (unmapped): `state_flag::raw_flag`" in result
        assert "dilemma::" not in result

    def test_dilemma_then_unmapped_render_order(self) -> None:
        result = format_valid_state_flag_ids_by_dilemma(
            ["state_flag::mapped", "state_flag::orphan"],
            flag_to_dilemma={"state_flag::mapped": "dilemma::trust"},
        )
        mapped_idx = result.index("- `dilemma::trust`")
        unmapped_idx = result.index("- (unmapped)")
        assert mapped_idx < unmapped_idx

    def test_dilemmas_sorted_alphabetically(self) -> None:
        result = format_valid_state_flag_ids_by_dilemma(
            ["state_flag::a", "state_flag::b"],
            flag_to_dilemma={
                "state_flag::a": "dilemma::zeta",
                "state_flag::b": "dilemma::alpha",
            },
        )
        alpha_idx = result.index("dilemma::alpha")
        zeta_idx = result.index("dilemma::zeta")
        assert alpha_idx < zeta_idx

    def test_empty_input_returns_empty_string(self) -> None:
        assert format_valid_state_flag_ids_by_dilemma([], {}) == ""
