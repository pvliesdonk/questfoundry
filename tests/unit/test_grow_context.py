"""Tests for GROW context-formatting helpers."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_context import (
    format_valid_entity_ids_by_category,
    format_valid_state_flag_ids_by_dilemma,
)


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
