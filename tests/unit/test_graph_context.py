"""Tests for graph context formatting."""

from __future__ import annotations

from questfoundry.graph import Graph, format_valid_ids_context


class TestFormatValidIdsContext:
    """Tests for format_valid_ids_context function."""

    def test_returns_empty_for_unknown_stage(self) -> None:
        """Unknown stages return empty string."""
        graph = Graph.empty()
        result = format_valid_ids_context(graph, "unknown")
        assert result == ""

    def test_returns_empty_for_empty_graph(self) -> None:
        """Empty graph returns minimal context for seed stage."""
        graph = Graph.empty()
        result = format_valid_ids_context(graph, "seed")
        # Should still have header and rules even with no entities
        assert "VALID IDS" in result
        assert "Rules" in result

    def test_seed_includes_entities_by_category(self) -> None:
        """SEED context groups entities by category."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_type": "character",
            },
        )
        graph.create_node(
            "entity::tavern",
            {
                "type": "entity",
                "raw_id": "tavern",
                "entity_type": "location",
            },
        )
        graph.create_node(
            "entity::sword",
            {
                "type": "entity",
                "raw_id": "sword",
                "entity_type": "object",
            },
        )

        result = format_valid_ids_context(graph, "seed")

        assert "**Characters:**" in result
        assert "`hero`" in result
        assert "**Locations:**" in result
        assert "`tavern`" in result
        assert "**Objects:**" in result
        assert "`sword`" in result

    def test_seed_includes_tensions_with_alternatives(self) -> None:
        """SEED context lists tensions with their alternatives."""
        graph = Graph.empty()
        graph.create_node(
            "tension::trust",
            {
                "type": "tension",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "tension::trust::alt::yes",
            {
                "type": "alternative",
                "raw_id": "yes",
                "is_default_path": True,
            },
        )
        graph.create_node(
            "tension::trust::alt::no",
            {
                "type": "alternative",
                "raw_id": "no",
                "is_default_path": False,
            },
        )
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::yes")
        graph.add_edge("has_alternative", "tension::trust", "tension::trust::alt::no")

        result = format_valid_ids_context(graph, "seed")

        assert "Tension IDs" in result
        assert "`trust`" in result
        assert "`yes`" in result
        assert "(default)" in result
        assert "`no`" in result

    def test_seed_includes_rules(self) -> None:
        """SEED context includes rules for ID usage."""
        graph = Graph.empty()
        result = format_valid_ids_context(graph, "seed")

        assert "Rules" in result
        assert "entity" in result.lower()
        assert "tension" in result.lower()
        assert "alternative" in result.lower()

    def test_seed_sorts_entity_ids_alphabetically(self) -> None:
        """Entity IDs are sorted alphabetically within categories."""
        graph = Graph.empty()
        graph.create_node(
            "entity::zara",
            {
                "type": "entity",
                "raw_id": "zara",
                "entity_type": "character",
            },
        )
        graph.create_node(
            "entity::bob",
            {
                "type": "entity",
                "raw_id": "bob",
                "entity_type": "character",
            },
        )
        graph.create_node(
            "entity::alice",
            {
                "type": "entity",
                "raw_id": "alice",
                "entity_type": "character",
            },
        )

        result = format_valid_ids_context(graph, "seed")

        # Check order: alice should come before bob, bob before zara
        alice_pos = result.find("`alice`")
        bob_pos = result.find("`bob`")
        zara_pos = result.find("`zara`")

        assert alice_pos < bob_pos < zara_pos

    def test_seed_skips_entities_without_raw_id(self) -> None:
        """Entities without raw_id are skipped."""
        graph = Graph.empty()
        graph.create_node(
            "entity::valid",
            {
                "type": "entity",
                "raw_id": "valid",
                "entity_type": "character",
            },
        )
        graph.create_node(
            "entity::invalid",
            {
                "type": "entity",
                # Missing raw_id
                "entity_type": "character",
            },
        )

        result = format_valid_ids_context(graph, "seed")

        assert "`valid`" in result
        assert "invalid" not in result

    def test_seed_handles_unknown_category(self) -> None:
        """Unknown entity categories are handled gracefully."""
        graph = Graph.empty()
        graph.create_node(
            "entity::something",
            {
                "type": "entity",
                "raw_id": "something",
                "entity_type": "custom_type",  # Not a standard category
            },
        )

        # Should not raise, just won't appear in standard categories
        result = format_valid_ids_context(graph, "seed")
        assert "VALID IDS" in result

    def test_full_context_format(self) -> None:
        """Test complete context format with entities and tensions."""
        graph = Graph.empty()

        # Add entities
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_type": "character",
            },
        )
        graph.create_node(
            "entity::castle",
            {
                "type": "entity",
                "raw_id": "castle",
                "entity_type": "location",
            },
        )

        # Add tension with alternatives
        graph.create_node(
            "tension::quest",
            {
                "type": "tension",
                "raw_id": "quest",
            },
        )
        graph.create_node(
            "tension::quest::alt::accept",
            {
                "type": "alternative",
                "raw_id": "accept",
                "is_default_path": True,
            },
        )
        graph.add_edge("has_alternative", "tension::quest", "tension::quest::alt::accept")

        result = format_valid_ids_context(graph, "seed")

        # Verify structure
        assert result.startswith("## VALID IDS")
        assert "### Entity IDs" in result
        assert "### Tension IDs" in result
        assert "### Rules" in result

        # Verify content
        assert "`hero`" in result
        assert "`castle`" in result
        assert "`quest`" in result
        assert "`accept` (default)" in result
