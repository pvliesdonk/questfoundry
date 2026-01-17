"""Tests for graph context formatting."""

from __future__ import annotations

from questfoundry.graph import Graph, format_valid_ids_context
from questfoundry.graph.context import format_thread_ids_context


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


class TestFormatThreadIdsContext:
    """Tests for format_thread_ids_context function."""

    def test_returns_empty_for_empty_list(self) -> None:
        """Empty threads list returns empty string."""
        result = format_thread_ids_context([])
        assert result == ""

    def test_returns_empty_for_threads_without_thread_id(self) -> None:
        """Threads missing thread_id field return empty string."""
        threads = [
            {"tension_id": "some_tension"},
            {"name": "Some Thread"},
        ]
        result = format_thread_ids_context(threads)
        assert result == ""

    def test_single_thread_format(self) -> None:
        """Single thread produces correct format."""
        threads = [{"thread_id": "host_motive", "tension_id": "host_benevolent_or_self_serving"}]
        result = format_thread_ids_context(threads)

        assert "## VALID THREAD IDs" in result
        assert "Allowed: `host_motive`" in result
        assert "Rules:" in result
        assert "WRONG" in result

    def test_multiple_threads_pipe_delimited(self) -> None:
        """Multiple threads are pipe-delimited."""
        threads = [
            {"thread_id": "host_motive"},
            {"thread_id": "butler_fidelity"},
            {"thread_id": "archive_secret"},
        ]
        result = format_thread_ids_context(threads)

        # Should be sorted and pipe-delimited
        assert "`archive_secret` | `butler_fidelity` | `host_motive`" in result

    def test_threads_sorted_alphabetically(self) -> None:
        """Thread IDs are sorted alphabetically for deterministic output."""
        threads = [
            {"thread_id": "zebra_thread"},
            {"thread_id": "alpha_thread"},
            {"thread_id": "middle_thread"},
        ]
        result = format_thread_ids_context(threads)

        # Check order in the Allowed line
        alpha_pos = result.find("`alpha_thread`")
        middle_pos = result.find("`middle_thread`")
        zebra_pos = result.find("`zebra_thread`")

        assert alpha_pos < middle_pos < zebra_pos

    def test_skips_threads_without_thread_id(self) -> None:
        """Threads without thread_id are gracefully skipped."""
        threads = [
            {"thread_id": "valid_thread"},
            {"tension_id": "missing_thread_id"},
            {"thread_id": "another_valid"},
        ]
        result = format_thread_ids_context(threads)

        assert "`another_valid`" in result
        assert "`valid_thread`" in result
        assert "missing_thread_id" not in result

    def test_includes_wrong_examples(self) -> None:
        """Output includes WRONG examples for LLM guidance."""
        threads = [{"thread_id": "host_motive"}]
        result = format_thread_ids_context(threads)

        assert "WRONG (will fail validation):" in result
        assert "`clock_distortion`" in result
        assert "`the_host_motive`" in result

    def test_includes_rules(self) -> None:
        """Output includes rules for ID usage."""
        threads = [{"thread_id": "host_motive"}]
        result = format_thread_ids_context(threads)

        assert "Rules:" in result
        assert "ONLY IDs from the list above" in result
        assert "Do NOT add prefixes" in result
        assert "Do NOT derive IDs from tension concepts" in result

    def test_handles_empty_thread_id_string(self) -> None:
        """Empty string thread_id is treated as missing."""
        threads = [
            {"thread_id": "valid_thread"},
            {"thread_id": ""},
            {"thread_id": "another_valid"},
        ]
        result = format_thread_ids_context(threads)

        # Empty string should be skipped
        assert "`valid_thread`" in result
        assert "`another_valid`" in result
        # Should not have empty backticks
        assert "``" not in result
