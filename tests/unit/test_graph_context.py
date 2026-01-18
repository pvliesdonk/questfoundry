"""Tests for graph context formatting."""

from __future__ import annotations

from questfoundry.graph import Graph, format_summarize_manifest, format_valid_ids_context
from questfoundry.graph.context import (
    check_structural_completeness,
    format_thread_ids_context,
    get_expected_counts,
)


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
        # Should still have header and requirements even with no entities
        assert "VALID IDS MANIFEST" in result
        assert "Generation Requirements" in result

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

        # Categories now include counts
        assert "**Characters (1):**" in result
        assert "`hero`" in result
        assert "**Locations (1):**" in result
        assert "`tavern`" in result
        assert "**Objects (1):**" in result
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

    def test_seed_includes_generation_requirements(self) -> None:
        """SEED context includes generation requirements for completeness."""
        graph = Graph.empty()
        result = format_valid_ids_context(graph, "seed")

        assert "Generation Requirements" in result
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
        assert "VALID IDS MANIFEST" in result

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
        assert result.startswith("## VALID IDS MANIFEST")
        assert "### Entity IDs" in result
        assert "### Tension IDs" in result
        assert "### Generation Requirements" in result

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


class TestGetExpectedCounts:
    """Tests for get_expected_counts function."""

    def test_returns_zero_for_empty_graph(self) -> None:
        """Empty graph returns zero counts."""
        graph = Graph.empty()
        counts = get_expected_counts(graph)

        assert counts["entities"] == 0
        assert counts["tensions"] == 0

    def test_counts_entities_with_raw_id(self) -> None:
        """Only entities with raw_id are counted."""
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
            "entity::invalid",
            {
                "type": "entity",
                "entity_type": "character",
                # Missing raw_id
            },
        )
        graph.create_node(
            "entity::location",
            {
                "type": "entity",
                "raw_id": "castle",
                "entity_type": "location",
            },
        )

        counts = get_expected_counts(graph)
        assert counts["entities"] == 2  # Only hero and castle

    def test_counts_tensions_with_raw_id(self) -> None:
        """Only tensions with raw_id are counted."""
        graph = Graph.empty()
        graph.create_node(
            "tension::trust",
            {
                "type": "tension",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "tension::invalid",
            {
                "type": "tension",
                # Missing raw_id
            },
        )

        counts = get_expected_counts(graph)
        assert counts["tensions"] == 1


class TestManifestCounts:
    """Tests for manifest counts in format_valid_ids_context."""

    def test_context_includes_entity_count(self) -> None:
        """Context should include total entity count."""
        graph = Graph.empty()
        for i in range(3):
            graph.create_node(
                f"entity::char{i}",
                {
                    "type": "entity",
                    "raw_id": f"char{i}",
                    "entity_type": "character",
                },
            )

        result = format_valid_ids_context(graph, "seed")
        assert "TOTAL: 3" in result
        assert "Generate EXACTLY 3 entity decisions" in result

    def test_context_includes_tension_count(self) -> None:
        """Context should include total tension count."""
        graph = Graph.empty()
        for i in range(2):
            graph.create_node(
                f"tension::t{i}",
                {
                    "type": "tension",
                    "raw_id": f"tension_{i}",
                },
            )

        result = format_valid_ids_context(graph, "seed")
        assert "TOTAL: 2" in result
        assert "Generate EXACTLY 2 tension decisions" in result

    def test_context_includes_category_counts(self) -> None:
        """Context should include per-category counts."""
        graph = Graph.empty()
        # Add 2 characters
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::villain",
            {"type": "entity", "raw_id": "villain", "entity_type": "character"},
        )
        # Add 1 location
        graph.create_node(
            "entity::castle",
            {"type": "entity", "raw_id": "castle", "entity_type": "location"},
        )

        result = format_valid_ids_context(graph, "seed")
        assert "Characters (2)" in result
        assert "Locations (1)" in result

    def test_context_includes_verification_section(self) -> None:
        """Context should include verification counts for self-checking."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "tension::trust",
            {"type": "tension", "raw_id": "trust"},
        )

        result = format_valid_ids_context(graph, "seed")
        assert "Verification" in result
        assert "entities array should have 1 item" in result
        assert "tensions array should have 1 item" in result


class TestFormatSummarizeManifest:
    """Tests for format_summarize_manifest function."""

    def test_returns_no_entities_for_empty_graph(self) -> None:
        """Empty graph returns '(No entities)' and '(No tensions)'."""
        graph = Graph.empty()
        result = format_summarize_manifest(graph)

        assert result["entity_manifest"] == "(No entities)"
        assert result["tension_manifest"] == "(No tensions)"

    def test_formats_entities_by_category(self) -> None:
        """Entities are grouped by category with markdown formatting."""
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
            "entity::castle",
            {
                "type": "entity",
                "raw_id": "castle",
                "entity_type": "location",
            },
        )

        result = format_summarize_manifest(graph)

        assert "**Characters:**" in result["entity_manifest"]
        assert "`hero`" in result["entity_manifest"]
        assert "**Locations:**" in result["entity_manifest"]
        assert "`castle`" in result["entity_manifest"]

    def test_formats_tensions_as_list(self) -> None:
        """Tensions are formatted as simple bullet list."""
        graph = Graph.empty()
        graph.create_node(
            "tension::trust",
            {
                "type": "tension",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "tension::loyalty",
            {
                "type": "tension",
                "raw_id": "loyalty",
            },
        )

        result = format_summarize_manifest(graph)

        assert "- `loyalty`" in result["tension_manifest"]
        assert "- `trust`" in result["tension_manifest"]

    def test_skips_entities_without_raw_id(self) -> None:
        """Entities without raw_id are excluded."""
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
                "entity_type": "character",
                # Missing raw_id
            },
        )

        result = format_summarize_manifest(graph)

        assert "`valid`" in result["entity_manifest"]
        assert "invalid" not in result["entity_manifest"]

    def test_skips_tensions_without_raw_id(self) -> None:
        """Tensions without raw_id are excluded."""
        graph = Graph.empty()
        graph.create_node(
            "tension::valid",
            {
                "type": "tension",
                "raw_id": "valid",
            },
        )
        graph.create_node(
            "tension::invalid",
            {
                "type": "tension",
                # Missing raw_id
            },
        )

        result = format_summarize_manifest(graph)

        assert "`valid`" in result["tension_manifest"]
        assert "invalid" not in result["tension_manifest"]

    def test_sorts_entities_alphabetically(self) -> None:
        """Entity IDs are sorted alphabetically within categories."""
        graph = Graph.empty()
        graph.create_node(
            "entity::zara",
            {"type": "entity", "raw_id": "zara", "entity_type": "character"},
        )
        graph.create_node(
            "entity::alice",
            {"type": "entity", "raw_id": "alice", "entity_type": "character"},
        )

        result = format_summarize_manifest(graph)

        alice_pos = result["entity_manifest"].find("`alice`")
        zara_pos = result["entity_manifest"].find("`zara`")
        assert alice_pos < zara_pos

    def test_sorts_tensions_by_node_id(self) -> None:
        """Tensions are sorted by node ID for deterministic output."""
        graph = Graph.empty()
        graph.create_node(
            "tension::zebra",
            {"type": "tension", "raw_id": "zebra"},
        )
        graph.create_node(
            "tension::alpha",
            {"type": "tension", "raw_id": "alpha"},
        )

        result = format_summarize_manifest(graph)

        alpha_pos = result["tension_manifest"].find("`alpha`")
        zebra_pos = result["tension_manifest"].find("`zebra`")
        assert alpha_pos < zebra_pos

    def test_handles_all_entity_categories(self) -> None:
        """All standard entity categories are included in output."""
        graph = Graph.empty()
        for cat in ["character", "location", "object", "faction"]:
            graph.create_node(
                f"entity::{cat}_example",
                {"type": "entity", "raw_id": f"{cat}_example", "entity_type": cat},
            )

        result = format_summarize_manifest(graph)

        assert "**Characters:**" in result["entity_manifest"]
        assert "**Locations:**" in result["entity_manifest"]
        assert "**Objects:**" in result["entity_manifest"]
        assert "**Factions:**" in result["entity_manifest"]

    def test_returns_dict_with_both_keys(self) -> None:
        """Result always contains both entity_manifest and tension_manifest keys."""
        graph = Graph.empty()
        result = format_summarize_manifest(graph)

        assert "entity_manifest" in result
        assert "tension_manifest" in result
        assert len(result) == 2

    def test_skips_entities_with_unknown_type(self) -> None:
        """Entities with unknown entity_type are excluded from manifest."""
        graph = Graph.empty()
        graph.create_node(
            "entity::weird",
            {
                "type": "entity",
                "raw_id": "weird_thing",
                "entity_type": "unknown",
            },
        )
        graph.create_node(
            "entity::custom",
            {
                "type": "entity",
                "raw_id": "custom_item",
                "entity_type": "custom_type",
            },
        )
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_type": "character",
            },
        )

        result = format_summarize_manifest(graph)

        # Only standard categories are included
        assert "hero" in result["entity_manifest"]
        assert "weird_thing" not in result["entity_manifest"]
        assert "custom_item" not in result["entity_manifest"]

    def test_adds_blank_lines_between_categories(self) -> None:
        """Output includes blank lines between entity categories for readability."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::castle",
            {"type": "entity", "raw_id": "castle", "entity_type": "location"},
        )

        result = format_summarize_manifest(graph)

        # Should have blank line after Characters section, before Locations
        assert "**Characters:**\n  - `hero`\n\n**Locations:**" in result["entity_manifest"]


class TestCheckStructuralCompleteness:
    """Tests for check_structural_completeness function."""

    def test_returns_empty_when_counts_match(self) -> None:
        """No errors when actual counts match expected."""
        output = {
            "entities": [{"entity_id": "a"}, {"entity_id": "b"}],
            "tensions": [{"tension_id": "x"}],
        }
        expected = {"entities": 2, "tensions": 1}

        errors = check_structural_completeness(output, expected)

        assert errors == []

    def test_detects_missing_entities(self) -> None:
        """Reports error when entity count is less than expected."""
        output = {
            "entities": [{"entity_id": "a"}],
            "tensions": [],
        }
        expected = {"entities": 3, "tensions": 0}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "entities"
        assert "Expected 3" in errors[0][1]
        assert "got 1" in errors[0][1]

    def test_detects_missing_tensions(self) -> None:
        """Reports error when tension count is less than expected."""
        output = {
            "entities": [],
            "tensions": [{"tension_id": "x"}],
        }
        expected = {"entities": 0, "tensions": 2}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "tensions"
        assert "Expected 2" in errors[0][1]
        assert "got 1" in errors[0][1]

    def test_detects_extra_entities(self) -> None:
        """Reports error when entity count exceeds expected."""
        output = {
            "entities": [{"entity_id": "a"}, {"entity_id": "b"}, {"entity_id": "c"}],
            "tensions": [],
        }
        expected = {"entities": 2, "tensions": 0}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "entities"
        assert "Expected 2" in errors[0][1]
        assert "got 3" in errors[0][1]

    def test_detects_multiple_errors(self) -> None:
        """Reports errors for both entities and tensions."""
        output = {
            "entities": [],
            "tensions": [],
        }
        expected = {"entities": 2, "tensions": 1}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 2
        field_paths = {e[0] for e in errors}
        assert field_paths == {"entities", "tensions"}

    def test_handles_missing_output_keys(self) -> None:
        """Handles output missing entities or tensions keys."""
        output: dict[str, list[dict[str, str]]] = {}
        expected = {"entities": 1, "tensions": 1}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 2

    def test_handles_zero_expected(self) -> None:
        """No error when both expected and actual are zero."""
        output = {"entities": [], "tensions": []}
        expected = {"entities": 0, "tensions": 0}

        errors = check_structural_completeness(output, expected)

        assert errors == []
