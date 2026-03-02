"""Tests for graph context formatting."""

from __future__ import annotations

from questfoundry.graph import Graph, format_summarize_manifest, format_valid_ids_context
from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    SCOPE_DILEMMA,
    SCOPE_PATH,
    check_structural_completeness,
    format_answer_ids_by_dilemma,
    format_dilemma_analysis_context,
    format_hierarchical_path_id,
    format_interaction_candidates_context,
    format_path_ids_context,
    format_scoped_id,
    get_expected_counts,
    parse_hierarchical_path_id,
    parse_scoped_id,
    strip_scope_prefix,
)
from questfoundry.models.seed import Consequence, DilemmaDecision, Path, SeedOutput


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

        # Categories now include counts, and entity IDs use category prefix
        assert "**Characters (1):**" in result
        assert "`character::hero`" in result
        assert "**Locations (1):**" in result
        assert "`location::tavern`" in result
        assert "**Objects (1):**" in result
        assert "`object::sword`" in result

    def test_seed_includes_dilemmas_with_answers(self) -> None:
        """SEED context lists dilemmas with their answers."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {
                "type": "dilemma",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "dilemma::trust::alt::yes",
            {
                "type": "answer",
                "raw_id": "yes",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "dilemma::trust::alt::no",
            {
                "type": "answer",
                "raw_id": "no",
                "is_canonical": False,
            },
        )
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::yes")
        graph.add_edge("has_answer", "dilemma::trust", "dilemma::trust::alt::no")

        result = format_valid_ids_context(graph, "seed")

        assert "Dilemma IDs" in result
        assert "`dilemma::trust`" in result
        assert "`yes`" in result  # answers stay unscoped within dilemma arrow
        assert "(default)" in result
        assert "`no`" in result

    def test_seed_includes_generation_requirements(self) -> None:
        """SEED context includes generation requirements for completeness."""
        graph = Graph.empty()
        result = format_valid_ids_context(graph, "seed")

        assert "Generation Requirements" in result
        assert "entity" in result.lower()
        assert "dilemma" in result.lower()
        assert "answer" in result.lower()

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
        alice_pos = result.find("`character::alice`")
        bob_pos = result.find("`character::bob`")
        zara_pos = result.find("`character::zara`")

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

        assert "`character::valid`" in result
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
        """Test complete context format with entities and dilemmas."""
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

        # Add dilemma with answers (graph still uses "dilemma" type)
        graph.create_node(
            "dilemma::quest",
            {
                "type": "dilemma",
                "raw_id": "quest",
            },
        )
        graph.create_node(
            "dilemma::quest::alt::accept",
            {
                "type": "answer",
                "raw_id": "accept",
                "is_canonical": True,
            },
        )
        graph.add_edge("has_answer", "dilemma::quest", "dilemma::quest::alt::accept")

        result = format_valid_ids_context(graph, "seed")

        # Verify structure
        assert result.startswith("## VALID IDS MANIFEST")
        assert "### Entity IDs" in result
        assert "### Dilemma IDs" in result
        assert "### Generation Requirements" in result

        # Verify content with scoped IDs
        assert "`character::hero`" in result
        assert "`location::castle`" in result
        assert "`dilemma::quest`" in result
        assert "`accept` (default)" in result  # answers stay unscoped


class TestSectionScopedManifest:
    """Tests for section-scoped valid IDs manifest (#898)."""

    @staticmethod
    def _graph_with_entities_and_dilemmas() -> Graph:
        """Build a graph with both entities and dilemmas for scoping tests."""
        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::tavern",
            {"type": "entity", "raw_id": "tavern", "entity_type": "location"},
        )
        graph.create_node(
            "dilemma::trust_or_betray",
            {"type": "dilemma", "raw_id": "trust_or_betray"},
        )
        graph.create_node(
            "dilemma::trust_or_betray::alt::trust",
            {"type": "answer", "raw_id": "trust", "is_canonical": True},
        )
        graph.create_node(
            "dilemma::trust_or_betray::alt::betray",
            {"type": "answer", "raw_id": "betray", "is_canonical": False},
        )
        graph.add_edge(
            "has_answer", "dilemma::trust_or_betray", "dilemma::trust_or_betray::alt::trust"
        )
        graph.add_edge(
            "has_answer", "dilemma::trust_or_betray", "dilemma::trust_or_betray::alt::betray"
        )
        return graph

    def test_seed_entities_section_excludes_dilemmas(self) -> None:
        """section='entities' has entity IDs but no dilemma IDs."""
        graph = self._graph_with_entities_and_dilemmas()
        result = format_valid_ids_context(graph, "seed", section="entities")

        assert "VALID ENTITY IDS" in result
        assert "`character::hero`" in result
        assert "`location::tavern`" in result
        assert "EXACTLY 2 entity decisions" in result

        # Must NOT mention dilemmas
        assert "Dilemma IDs" not in result
        assert "dilemma::" not in result
        assert "dilemma decisions" not in result.lower()

    def test_seed_dilemmas_section_excludes_entities(self) -> None:
        """section='dilemmas' has dilemma IDs but no entity IDs."""
        graph = self._graph_with_entities_and_dilemmas()
        result = format_valid_ids_context(graph, "seed", section="dilemmas")

        assert "VALID DILEMMA IDS" in result
        assert "`dilemma::trust_or_betray`" in result
        assert "EXACTLY 1 dilemma decisions" in result

        # Must NOT mention entities
        assert "Entity IDs" not in result
        assert "character::" not in result
        assert "entity decisions" not in result.lower()

    def test_seed_unknown_section_returns_empty(self) -> None:
        """Sections like 'consequences' return empty (IDs come from elsewhere)."""
        graph = self._graph_with_entities_and_dilemmas()

        assert format_valid_ids_context(graph, "seed", section="consequences") == ""
        assert format_valid_ids_context(graph, "seed", section="beats") == ""
        assert format_valid_ids_context(graph, "seed", section="paths") == ""

    def test_seed_none_section_returns_full_manifest(self) -> None:
        """section=None returns the full manifest (backward compat)."""
        graph = self._graph_with_entities_and_dilemmas()
        result = format_valid_ids_context(graph, "seed", section=None)

        # Both entity and dilemma blocks present
        assert "Entity IDs" in result
        assert "Dilemma IDs" in result
        assert "`character::hero`" in result
        assert "`dilemma::trust_or_betray`" in result
        assert "entity decisions" in result.lower()
        assert "dilemma decisions" in result.lower()

    def test_default_section_matches_none(self) -> None:
        """Omitting section arg gives same result as section=None."""
        graph = self._graph_with_entities_and_dilemmas()
        default_result = format_valid_ids_context(graph, "seed")
        explicit_none = format_valid_ids_context(graph, "seed", section=None)
        assert default_result == explicit_none


class TestFormatPathIdsContext:
    """Tests for format_path_ids_context function."""

    def test_returns_empty_for_empty_list(self) -> None:
        """Empty paths list returns empty string."""
        result = format_path_ids_context([])
        assert result == ""

    def test_returns_empty_for_paths_without_path_id(self) -> None:
        """Paths missing path_id field return empty string."""
        paths = [
            {"dilemma_id": "some_dilemma"},
            {"name": "Some Path"},
        ]
        result = format_path_ids_context(paths)
        assert result == ""

    def test_single_path_format(self) -> None:
        """Single path produces correct format."""
        paths = [{"path_id": "host_motive", "dilemma_id": "host_benevolent_or_self_serving"}]
        result = format_path_ids_context(paths)

        assert "## VALID PATH IDs" in result
        assert "Allowed: `path::host_motive`" in result
        assert "Rules:" in result
        assert "WRONG" in result

    def test_multiple_paths_pipe_delimited(self) -> None:
        """Multiple paths are pipe-delimited."""
        paths = [
            {"path_id": "host_motive"},
            {"path_id": "butler_fidelity"},
            {"path_id": "archive_secret"},
        ]
        result = format_path_ids_context(paths)

        # Should be sorted and pipe-delimited with path:: prefix
        assert "`path::archive_secret` | `path::butler_fidelity` | `path::host_motive`" in result

    def test_paths_sorted_alphabetically(self) -> None:
        """Path IDs are sorted alphabetically for deterministic output."""
        paths = [
            {"path_id": "zebra_path"},
            {"path_id": "alpha_path"},
            {"path_id": "middle_path"},
        ]
        result = format_path_ids_context(paths)

        # Check order in the Allowed line
        alpha_pos = result.find("`path::alpha_path`")
        middle_pos = result.find("`path::middle_path`")
        zebra_pos = result.find("`path::zebra_path`")

        assert alpha_pos < middle_pos < zebra_pos

    def test_skips_paths_without_path_id(self) -> None:
        """Paths without path_id are gracefully skipped."""
        paths = [
            {"path_id": "valid_path"},
            {"dilemma_id": "missing_path_id"},
            {"path_id": "another_valid"},
        ]
        result = format_path_ids_context(paths)

        assert "`path::another_valid`" in result
        assert "`path::valid_path`" in result
        assert "missing_path_id" not in result

    def test_includes_wrong_examples(self) -> None:
        """Output includes WRONG examples for LLM guidance."""
        paths = [{"path_id": "host_motive"}]
        result = format_path_ids_context(paths)

        assert "WRONG (will fail validation):" in result
        assert "`clock_distortion`" in result
        assert "`host_motive`" in result  # missing scope prefix example

    def test_includes_rules(self) -> None:
        """Output includes rules for ID usage."""
        paths = [{"path_id": "host_motive"}]
        result = format_path_ids_context(paths)

        assert "Rules:" in result
        assert "ONLY IDs from the list above" in result
        assert "Include the `path::` prefix" in result
        assert "Do NOT derive IDs from dilemma concepts" in result

    def test_handles_empty_path_id_string(self) -> None:
        """Empty string path_id is treated as missing."""
        paths = [
            {"path_id": "valid_path"},
            {"path_id": ""},
            {"path_id": "another_valid"},
        ]
        result = format_path_ids_context(paths)

        # Empty string should be skipped - only valid paths appear
        assert "`path::valid_path`" in result
        assert "`path::another_valid`" in result
        # Exactly 2 path IDs in the Allowed line (empty one skipped)
        allowed_line = next(ln for ln in result.split("\n") if ln.startswith("Allowed:"))
        assert allowed_line.count("`path::") == 2


class TestGetExpectedCounts:
    """Tests for get_expected_counts function."""

    def test_returns_zero_for_empty_graph(self) -> None:
        """Empty graph returns zero counts."""
        graph = Graph.empty()
        counts = get_expected_counts(graph)

        assert counts["entities"] == 0
        assert counts["dilemmas"] == 0

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

    def test_counts_dilemmas_with_raw_id(self) -> None:
        """Only dilemmas with raw_id are counted."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {
                "type": "dilemma",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "dilemma::invalid",
            {
                "type": "dilemma",
                # Missing raw_id
            },
        )

        counts = get_expected_counts(graph)
        assert counts["dilemmas"] == 1


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

    def test_context_includes_dilemma_count(self) -> None:
        """Context should include total dilemma count."""
        graph = Graph.empty()
        for i in range(2):
            graph.create_node(
                f"dilemma::t{i}",
                {
                    "type": "dilemma",
                    "raw_id": f"dilemma_{i}",
                },
            )

        result = format_valid_ids_context(graph, "seed")
        assert "TOTAL: 2" in result
        assert "Generate EXACTLY 2 dilemma decisions" in result

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
            "dilemma::trust",
            {"type": "dilemma", "raw_id": "trust"},
        )

        result = format_valid_ids_context(graph, "seed")
        assert "Verification" in result
        assert "entities array should have 1 item" in result
        assert "dilemmas array should have 1 item" in result


class TestFormatSummarizeManifest:
    """Tests for format_summarize_manifest function."""

    def test_returns_no_entities_for_empty_graph(self) -> None:
        """Empty graph returns '(No entities)' and '(No dilemmas)'."""
        graph = Graph.empty()
        result = format_summarize_manifest(graph)

        assert result["entity_manifest"] == "(No entities)"
        assert result["dilemma_manifest"] == "(No dilemmas)"

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
        assert "`character::hero`" in result["entity_manifest"]
        assert "**Locations:**" in result["entity_manifest"]
        assert "`location::castle`" in result["entity_manifest"]

    def test_formats_dilemmas_as_list(self) -> None:
        """Dilemmas are formatted as simple bullet list."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {
                "type": "dilemma",
                "raw_id": "trust",
            },
        )
        graph.create_node(
            "dilemma::loyalty",
            {
                "type": "dilemma",
                "raw_id": "loyalty",
            },
        )

        result = format_summarize_manifest(graph)

        assert "- `dilemma::loyalty`" in result["dilemma_manifest"]
        assert "- `dilemma::trust`" in result["dilemma_manifest"]

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

        assert "`character::valid`" in result["entity_manifest"]
        assert "invalid" not in result["entity_manifest"]

    def test_skips_dilemmas_without_raw_id(self) -> None:
        """Dilemmas without raw_id are excluded."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::valid",
            {
                "type": "dilemma",
                "raw_id": "valid",
            },
        )
        graph.create_node(
            "dilemma::invalid",
            {
                "type": "dilemma",
                # Missing raw_id
            },
        )

        result = format_summarize_manifest(graph)

        assert "`dilemma::valid`" in result["dilemma_manifest"]
        assert "invalid" not in result["dilemma_manifest"]

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

        alice_pos = result["entity_manifest"].find("`character::alice`")
        zara_pos = result["entity_manifest"].find("`character::zara`")
        assert alice_pos < zara_pos

    def test_sorts_dilemmas_by_node_id(self) -> None:
        """Dilemmas are sorted by node ID for deterministic output."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::zebra",
            {"type": "dilemma", "raw_id": "zebra"},
        )
        graph.create_node(
            "dilemma::alpha",
            {"type": "dilemma", "raw_id": "alpha"},
        )

        result = format_summarize_manifest(graph)

        alpha_pos = result["dilemma_manifest"].find("`dilemma::alpha`")
        zebra_pos = result["dilemma_manifest"].find("`dilemma::zebra`")
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
        """Result always contains both entity_manifest and dilemma_manifest keys."""
        graph = Graph.empty()
        result = format_summarize_manifest(graph)

        assert "entity_manifest" in result
        assert "dilemma_manifest" in result
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
        assert "character::hero" in result["entity_manifest"]
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

        # Should have blank line between category sections (order may vary)
        manifest = result["entity_manifest"]
        assert "**Characters:**" in manifest
        assert "**Locations:**" in manifest
        assert "`character::hero`" in manifest
        assert "`location::castle`" in manifest
        # Verify blank line between sections (double newline pattern)
        assert "\n\n**" in manifest


class TestFormatRetainedEntityIds:
    """Tests for format_retained_entity_ids function."""

    def test_returns_empty_for_no_retained_entities(self) -> None:
        """Returns empty string when all entities are cut."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        decisions = [{"entity_id": "hero", "disposition": "cut"}]

        result = format_retained_entity_ids(graph, decisions)

        assert result == ""

    def test_filters_out_cut_entities(self) -> None:
        """Only retained entities appear in output."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::villain",
            {"type": "entity", "raw_id": "villain", "entity_type": "character"},
        )
        decisions = [
            {"entity_id": "hero", "disposition": "retained"},
            {"entity_id": "villain", "disposition": "cut"},
        ]

        result = format_retained_entity_ids(graph, decisions)

        assert "`character::hero`" in result
        assert "villain" not in result

    def test_handles_scoped_entity_ids_in_decisions(self) -> None:
        """Works with both scoped and unscoped IDs in decisions."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::villain",
            {"type": "entity", "raw_id": "villain", "entity_type": "character"},
        )
        # Mix of scoped and unscoped IDs in decisions
        decisions = [
            {"entity_id": "entity::hero", "disposition": "retained"},
            {"entity_id": "villain", "disposition": "cut"},
        ]

        result = format_retained_entity_ids(graph, decisions)

        assert "`character::hero`" in result
        assert "villain" not in result

    def test_includes_count_of_retained_entities(self) -> None:
        """Output includes count of retained entities."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::castle",
            {"type": "entity", "raw_id": "castle", "entity_type": "location"},
        )
        decisions = [
            {"entity_id": "hero", "disposition": "retained"},
            {"entity_id": "castle", "disposition": "retained"},
        ]

        result = format_retained_entity_ids(graph, decisions)

        assert "2 entities are RETAINED" in result

    def test_groups_by_category(self) -> None:
        """Retained entities are grouped by category."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        graph.create_node(
            "entity::castle",
            {"type": "entity", "raw_id": "castle", "entity_type": "location"},
        )
        decisions = [
            {"entity_id": "hero", "disposition": "retained"},
            {"entity_id": "castle", "disposition": "retained"},
        ]

        result = format_retained_entity_ids(graph, decisions)

        assert "**Characters" in result
        assert "**Locations" in result

    def test_includes_warning_about_cut_entities(self) -> None:
        """Output warns not to use cut entities."""
        from questfoundry.graph.context import format_retained_entity_ids

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_type": "character"},
        )
        decisions = [{"entity_id": "hero", "disposition": "retained"}]

        result = format_retained_entity_ids(graph, decisions)

        assert "RETAINED" in result
        assert "cut" in result.lower()


class TestCheckStructuralCompleteness:
    """Tests for check_structural_completeness function."""

    def test_returns_empty_when_counts_match(self) -> None:
        """No errors when actual counts match expected."""
        output = {
            "entities": [{"entity_id": "a"}, {"entity_id": "b"}],
            "dilemmas": [{"dilemma_id": "x"}],
        }
        expected = {"entities": 2, "dilemmas": 1}

        errors = check_structural_completeness(output, expected)

        assert errors == []

    def test_detects_missing_entities(self) -> None:
        """Reports error when entity count is less than expected."""
        output = {
            "entities": [{"entity_id": "a"}],
            "dilemmas": [],
        }
        expected = {"entities": 3, "dilemmas": 0}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "entities"
        assert "Expected 3" in errors[0][1]
        assert "got 1" in errors[0][1]

    def test_detects_missing_dilemmas(self) -> None:
        """Reports error when dilemma count is less than expected."""
        output = {
            "entities": [],
            "dilemmas": [{"dilemma_id": "x"}],
        }
        expected = {"entities": 0, "dilemmas": 2}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "dilemmas"
        assert "Expected 2" in errors[0][1]
        assert "got 1" in errors[0][1]

    def test_detects_extra_entities(self) -> None:
        """Reports error when entity count exceeds expected."""
        output = {
            "entities": [{"entity_id": "a"}, {"entity_id": "b"}, {"entity_id": "c"}],
            "dilemmas": [],
        }
        expected = {"entities": 2, "dilemmas": 0}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 1
        assert errors[0][0] == "entities"
        assert "Expected 2" in errors[0][1]
        assert "got 3" in errors[0][1]

    def test_detects_multiple_errors(self) -> None:
        """Reports errors for both entities and dilemmas."""
        output = {
            "entities": [],
            "dilemmas": [],
        }
        expected = {"entities": 2, "dilemmas": 1}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 2
        field_paths = {e[0] for e in errors}
        assert field_paths == {"entities", "dilemmas"}

    def test_handles_missing_output_keys(self) -> None:
        """Handles output missing entities or dilemmas keys."""
        output: dict[str, list[dict[str, str]]] = {}
        expected = {"entities": 1, "dilemmas": 1}

        errors = check_structural_completeness(output, expected)

        assert len(errors) == 2

    def test_handles_zero_expected(self) -> None:
        """No error when both expected and actual are zero."""
        output = {"entities": [], "dilemmas": []}
        expected = {"entities": 0, "dilemmas": 0}

        errors = check_structural_completeness(output, expected)

        assert errors == []

    def test_raises_on_negative_expected_count(self) -> None:
        """Raises ValueError when expected count is negative."""
        import pytest

        output = {"entities": [], "dilemmas": []}
        expected = {"entities": -1, "dilemmas": 0}

        with pytest.raises(ValueError, match="cannot be negative"):
            check_structural_completeness(output, expected)


class TestParseScopedId:
    """Tests for parse_scoped_id function."""

    def test_parses_entity_scoped_id(self) -> None:
        """Entity scoped ID is correctly parsed."""
        scope, raw_id = parse_scoped_id("entity::hero")
        assert scope == "entity"
        assert raw_id == "hero"

    def test_parses_dilemma_scoped_id(self) -> None:
        """Dilemma scoped ID is correctly parsed."""
        scope, raw_id = parse_scoped_id("dilemma::trust_betrayal")
        assert scope == "dilemma"
        assert raw_id == "trust_betrayal"

    def test_parses_path_scoped_id(self) -> None:
        """Path scoped ID is correctly parsed."""
        scope, raw_id = parse_scoped_id("path::host_motive")
        assert scope == "path"
        assert raw_id == "host_motive"

    def test_returns_empty_scope_for_unscoped_id(self) -> None:
        """Unscoped ID returns empty string for scope."""
        scope, raw_id = parse_scoped_id("hero")
        assert scope == ""
        assert raw_id == "hero"

    def test_handles_id_with_multiple_colons(self) -> None:
        """ID with multiple :: only splits on first occurrence."""
        scope, raw_id = parse_scoped_id("entity::node::with::colons")
        assert scope == "entity"
        assert raw_id == "node::with::colons"

    def test_handles_empty_raw_id(self) -> None:
        """Scope with empty raw_id is parsed correctly."""
        scope, raw_id = parse_scoped_id("entity::")
        assert scope == "entity"
        assert raw_id == ""

    def test_handles_empty_string(self) -> None:
        """Empty string returns empty scope and raw_id."""
        scope, raw_id = parse_scoped_id("")
        assert scope == ""
        assert raw_id == ""

    def test_handles_single_colon(self) -> None:
        """Single colon is not a scope delimiter."""
        scope, raw_id = parse_scoped_id("entity:hero")
        assert scope == ""
        assert raw_id == "entity:hero"


class TestStripScopePrefix:
    """Tests for strip_scope_prefix function."""

    def test_strips_entity_prefix(self) -> None:
        """Entity prefix is stripped correctly."""
        result = strip_scope_prefix("entity::hero")
        assert result == "hero"

    def test_strips_dilemma_prefix(self) -> None:
        """Dilemma prefix is stripped correctly."""
        result = strip_scope_prefix("dilemma::trust_betrayal")
        assert result == "trust_betrayal"

    def test_strips_path_prefix(self) -> None:
        """Path prefix is stripped correctly."""
        result = strip_scope_prefix("path::host_motive")
        assert result == "host_motive"

    def test_returns_unscoped_id_unchanged(self) -> None:
        """Unscoped ID is returned unchanged."""
        result = strip_scope_prefix("hero")
        assert result == "hero"

    def test_handles_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = strip_scope_prefix("")
        assert result == ""

    def test_handles_id_with_multiple_colons(self) -> None:
        """Only first :: is treated as scope delimiter."""
        result = strip_scope_prefix("entity::node::with::colons")
        assert result == "node::with::colons"


class TestFormatScopedId:
    """Tests for format_scoped_id function."""

    def test_formats_entity_scoped_id(self) -> None:
        """Entity ID is correctly formatted with scope."""
        result = format_scoped_id("entity", "hero")
        assert result == "entity::hero"

    def test_formats_dilemma_scoped_id(self) -> None:
        """Dilemma ID is correctly formatted with scope."""
        result = format_scoped_id("dilemma", "trust_betrayal")
        assert result == "dilemma::trust_betrayal"

    def test_formats_path_scoped_id(self) -> None:
        """Path ID is correctly formatted with scope."""
        result = format_scoped_id("path", "host_motive")
        assert result == "path::host_motive"

    def test_roundtrip_scoped_id(self) -> None:
        """Formatted scoped ID can be parsed back correctly."""
        original_scope = "entity"
        original_raw_id = "tavern"

        formatted = format_scoped_id(original_scope, original_raw_id)
        parsed_scope, parsed_raw_id = parse_scoped_id(formatted)

        assert parsed_scope == original_scope
        assert parsed_raw_id == original_raw_id


class TestScopeConstants:
    """Tests for scope constant values."""

    def test_entity_categories(self) -> None:
        """ENTITY_CATEGORIES contains expected categories."""
        assert "character" in ENTITY_CATEGORIES
        assert "location" in ENTITY_CATEGORIES
        assert "object" in ENTITY_CATEGORIES
        assert "faction" in ENTITY_CATEGORIES

    def test_scope_dilemma_value(self) -> None:
        """SCOPE_DILEMMA has expected value."""
        assert SCOPE_DILEMMA == "dilemma"

    def test_scope_path_value(self) -> None:
        """SCOPE_PATH has expected value."""
        assert SCOPE_PATH == "path"


class TestParseHierarchicalPathId:
    """Tests for parse_hierarchical_path_id function."""

    def test_rejects_wrong_scope_prefix(self) -> None:
        """Rejects non-path scope prefixes (e.g., legacy shorthand)."""
        import pytest

        with pytest.raises(ValueError, match="wrong scope prefix"):
            parse_hierarchical_path_id("dilemma::mentor_trust__benevolent")

    def test_parses_path_with_path_prefix(self) -> None:
        """Path ID with path:: prefix is correctly parsed."""
        dilemma_id, answer_id = parse_hierarchical_path_id("path::mentor_trust__selfish")
        assert dilemma_id == "dilemma::mentor_trust"
        assert answer_id == "selfish"

    def test_parses_unscoped_path_id(self) -> None:
        """Unscoped path ID is correctly parsed."""
        dilemma_id, answer_id = parse_hierarchical_path_id("mentor_trust__benevolent")
        assert dilemma_id == "dilemma::mentor_trust"
        assert answer_id == "benevolent"

    def test_raises_for_non_hierarchical_id(self) -> None:
        """Raises ValueError for ID without __ separator."""
        import pytest

        with pytest.raises(ValueError, match="not hierarchical"):
            parse_hierarchical_path_id("path::mentor_trust")

    def test_handles_multiple_underscores_in_dilemma(self) -> None:
        """Multiple underscores in dilemma part are preserved."""
        dilemma_id, answer_id = parse_hierarchical_path_id("path::my_complex_dilemma__answer")
        assert dilemma_id == "dilemma::my_complex_dilemma"
        assert answer_id == "answer"

    def test_handles_multiple_double_underscores(self) -> None:
        """Uses rightmost __ as separator."""
        dilemma_id, answer_id = parse_hierarchical_path_id("path::a__b__c")
        assert dilemma_id == "dilemma::a__b"
        assert answer_id == "c"


class TestFormatHierarchicalPathId:
    """Tests for format_hierarchical_path_id function."""

    def test_formats_with_scoped_dilemma_id(self) -> None:
        """Formats correctly with dilemma:: prefix on dilemma."""
        result = format_hierarchical_path_id("dilemma::mentor_trust", "benevolent")
        assert result == "path::mentor_trust__benevolent"

    def test_formats_with_unscoped_dilemma_id(self) -> None:
        """Formats correctly without prefix on dilemma."""
        result = format_hierarchical_path_id("mentor_trust", "selfish")
        assert result == "path::mentor_trust__selfish"

    def test_roundtrip_parse_and_format(self) -> None:
        """Parsing and formatting roundtrip preserves values."""
        original = "path::my_dilemma__my_answer"
        dilemma_id, answer_id = parse_hierarchical_path_id(original)
        reformatted = format_hierarchical_path_id(dilemma_id, answer_id)
        assert reformatted == original


class TestFormatAnswerIdsByDilemma:
    """Tests for format_answer_ids_by_dilemma function."""

    def test_empty_dilemmas_returns_empty(self) -> None:
        """Empty list returns empty string."""
        assert format_answer_ids_by_dilemma([]) == ""

    def test_single_dilemma_with_explored_and_unexplored(self) -> None:
        """Single dilemma formats explored and unexplored as human-readable text."""
        dilemmas = [
            {
                "dilemma_id": "dilemma::host_benevolent_or_selfish",
                "explored": ["protector", "manipulator"],
                "unexplored": ["neutral"],
            }
        ]
        result = format_answer_ids_by_dilemma(dilemmas)
        assert "Valid Answer IDs per Dilemma" in result
        assert "dilemma::host_benevolent_or_selfish" in result
        # Must use human-readable formatting, not Python list repr (#1088)
        assert "`protector`, `manipulator`" in result
        assert "`neutral`" in result
        # Must NOT contain Python list syntax
        assert "['" not in result

    def test_multiple_dilemmas_all_listed(self) -> None:
        """Multiple dilemmas are all included in output."""
        dilemmas = [
            {
                "dilemma_id": "dilemma::mentor_trust_or_betray",
                "explored": ["trust"],
                "unexplored": ["betray"],
            },
            {
                "dilemma_id": "dilemma::artifact_blessed_or_cursed",
                "explored": ["blessed", "cursed"],
                "unexplored": [],
            },
        ]
        result = format_answer_ids_by_dilemma(dilemmas)
        assert "dilemma::mentor_trust_or_betray" in result
        assert "dilemma::artifact_blessed_or_cursed" in result

    def test_dilemma_without_id_skipped(self) -> None:
        """Dilemmas with empty or missing ID produce empty string."""
        dilemmas = [
            {"dilemma_id": "", "explored": ["a"], "unexplored": []},
            {"explored": ["b"], "unexplored": []},
        ]
        result = format_answer_ids_by_dilemma(dilemmas)
        # No valid dilemmas → empty string (no header injected)
        assert result == ""

    def test_unscoped_dilemma_id_gets_prefix(self) -> None:
        """Dilemma IDs without scope prefix get normalized."""
        dilemmas = [
            {
                "dilemma_id": "host_benevolent_or_selfish",
                "explored": ["protector"],
                "unexplored": [],
            }
        ]
        result = format_answer_ids_by_dilemma(dilemmas)
        assert "dilemma::host_benevolent_or_selfish" in result


# ---------------------------------------------------------------------------
# Helper to build minimal SeedOutput for context tests
# ---------------------------------------------------------------------------


def _seed_output(
    dilemmas: list[DilemmaDecision] | None = None,
    paths: list[Path] | None = None,
    consequences: list[Consequence] | None = None,
) -> SeedOutput:
    """Build a minimal SeedOutput for testing context functions."""
    return SeedOutput(
        dilemmas=dilemmas or [],
        paths=paths or [],
        consequences=consequences or [],
    )


def _dilemma(
    dilemma_id: str,
    explored: list[str] | None = None,
    unexplored: list[str] | None = None,
) -> DilemmaDecision:
    return DilemmaDecision(
        dilemma_id=dilemma_id,
        explored=explored or ["answer_a"],
        unexplored=unexplored or [],
    )


def _path(
    path_id: str,
    dilemma_id: str,
    answer_id: str,
    description: str | None = None,
) -> Path:
    return Path(
        path_id=path_id,
        name=f"Path {answer_id}",
        dilemma_id=dilemma_id,
        answer_id=answer_id,
        path_importance="major",
        description=description or f"Explores {answer_id}",
    )


def _consequence(
    consequence_id: str,
    path_id: str,
    description: str,
    narrative_effects: list[str] | None = None,
) -> Consequence:
    return Consequence(
        consequence_id=consequence_id,
        path_id=path_id,
        description=description,
        narrative_effects=narrative_effects or [],
    )


class TestFormatDilemmaAnalysisContext:
    """Tests for format_dilemma_analysis_context."""

    def test_empty_dilemmas_returns_empty(self) -> None:
        """No dilemmas returns empty string."""
        result = format_dilemma_analysis_context(_seed_output())
        assert result == ""

    def test_formats_surviving_dilemmas(self) -> None:
        """Two dilemmas with paths produce formatted markdown."""
        seed = _seed_output(
            dilemmas=[
                _dilemma("alpha", explored=["trust", "betray"], unexplored=[]),
                _dilemma("beta", explored=["help"], unexplored=["ignore"]),
            ],
            paths=[
                _path("path::alpha__trust", "alpha", "trust"),
                _path("path::alpha__betray", "alpha", "betray"),
                _path("path::beta__help", "beta", "help"),
            ],
        )
        result = format_dilemma_analysis_context(seed)
        assert "## Dilemma Convergence Brief" in result
        assert "dilemma::alpha" in result
        assert "dilemma::beta" in result
        assert "Paths (2)" in result  # alpha has 2 paths
        assert "Paths (1)" in result  # beta has 1 path

    def test_includes_valid_ids_section(self) -> None:
        """Output includes Valid Dilemma IDs section."""
        seed = _seed_output(dilemmas=[_dilemma("gamma")])
        result = format_dilemma_analysis_context(seed)
        assert "### Valid Dilemma IDs" in result
        assert "`dilemma::gamma`" in result

    def test_path_count_per_dilemma(self) -> None:
        """Path counts are correctly computed per dilemma."""
        seed = _seed_output(
            dilemmas=[_dilemma("only_one", explored=["a", "b"])],
            paths=[
                _path("path::only_one__a", "only_one", "a"),
                _path("path::only_one__b", "only_one", "b"),
            ],
        )
        result = format_dilemma_analysis_context(seed)
        assert "Paths (2)" in result

    def test_includes_path_descriptions(self) -> None:
        """Path descriptions are included in the output."""
        seed = _seed_output(
            dilemmas=[_dilemma("d1", explored=["a", "b"])],
            paths=[
                _path("path::d1__a", "d1", "a", description="Take the red pill"),
                _path("path::d1__b", "d1", "b", description="Take the blue pill"),
            ],
        )
        result = format_dilemma_analysis_context(seed)
        assert "Take the red pill" in result
        assert "Take the blue pill" in result

    def test_unexplored_answers_displayed(self) -> None:
        """Dilemmas with unexplored answers show them in the output."""
        seed = _seed_output(
            dilemmas=[
                _dilemma("d1", explored=["trust"], unexplored=["betray", "ignore"]),
            ],
            paths=[_path("path::d1__trust", "d1", "trust")],
        )
        result = format_dilemma_analysis_context(seed)
        assert "**Unexplored answers:** betray, ignore" in result

    def test_no_unexplored_answers_omits_line(self) -> None:
        """Dilemmas with no unexplored answers omit the line entirely."""
        seed = _seed_output(
            dilemmas=[_dilemma("d1", explored=["a"], unexplored=[])],
            paths=[_path("path::d1__a", "d1", "a")],
        )
        result = format_dilemma_analysis_context(seed)
        assert "Unexplored answers" not in result

    def test_enriched_with_graph_data(self) -> None:
        """Graph data enriches output with question and stakes."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::alive_or_dead",
            {
                "type": "dilemma",
                "raw_id": "alive_or_dead",
                "question": "Should the hero sacrifice themselves?",
                "why_it_matters": "Life versus duty defines the entire story",
            },
        )
        graph.create_node(
            "character::hero",
            {"type": "entity", "raw_id": "hero", "entity_category": "character"},
        )
        graph.add_edge("anchored_to", "dilemma::alive_or_dead", "character::hero")
        seed = _seed_output(
            dilemmas=[_dilemma("alive_or_dead", explored=["alive", "dead"])],
            paths=[
                _path("path::alive_or_dead__alive", "alive_or_dead", "alive"),
                _path("path::alive_or_dead__dead", "alive_or_dead", "dead"),
            ],
        )
        result = format_dilemma_analysis_context(seed, graph)
        assert "**Question:** Should the hero sacrifice themselves?" in result
        assert "**Stakes:** Life versus duty defines the entire story" in result

    def test_enriched_with_consequence_effects(self) -> None:
        """Consequence narrative effects are included per path."""
        seed = _seed_output(
            dilemmas=[_dilemma("key_destroy_or_keep", explored=["destroy", "keep"])],
            paths=[
                _path(
                    "path::key_destroy_or_keep__destroy",
                    "key_destroy_or_keep",
                    "destroy",
                ),
                _path(
                    "path::key_destroy_or_keep__keep",
                    "key_destroy_or_keep",
                    "keep",
                ),
            ],
            consequences=[
                _consequence(
                    "cons::destroy_result",
                    "path::key_destroy_or_keep__destroy",
                    "The key is destroyed permanently",
                    narrative_effects=["Library sealed forever", "Knowledge lost"],
                ),
                _consequence(
                    "cons::keep_result",
                    "path::key_destroy_or_keep__keep",
                    "The key unlocks forbidden secrets",
                    narrative_effects=["Dark knowledge accessed"],
                ),
            ],
        )
        result = format_dilemma_analysis_context(seed)
        assert "Library sealed forever" in result
        assert "Knowledge lost" in result
        assert "Dark knowledge accessed" in result

    def test_consequence_with_empty_effects(self) -> None:
        """Consequence with empty narrative_effects produces no Effects line."""
        seed = _seed_output(
            dilemmas=[_dilemma("d1", explored=["a"])],
            paths=[
                _path(
                    "path::d1__a",
                    "d1",
                    "a",
                ),
            ],
            consequences=[
                _consequence(
                    "cons::empty_fx",
                    "path::d1__a",
                    "Something happens",
                    narrative_effects=[],
                ),
            ],
        )
        result = format_dilemma_analysis_context(seed)
        assert "Explores a" in result  # path description present
        assert "Effects:" not in result  # no effects → no Effects line

    def test_missing_graph_node_falls_back(self) -> None:
        """Missing graph node falls back to paths-only listing."""
        graph = Graph.empty()
        # Graph has no dilemma node for "orphan"
        seed = _seed_output(
            dilemmas=[_dilemma("orphan", explored=["x"])],
            paths=[_path("path::orphan__x", "orphan", "x")],
        )
        result = format_dilemma_analysis_context(seed, graph)
        assert "dilemma::orphan" in result
        assert "Paths (1)" in result
        # No Question or Stakes lines
        assert "**Question:**" not in result

    def test_no_paths_shows_explored_fallback(self) -> None:
        """Dilemma with no paths shows explored answers as fallback."""
        seed = _seed_output(
            dilemmas=[_dilemma("lonely", explored=["opt_a", "opt_b"])],
        )
        result = format_dilemma_analysis_context(seed)
        assert "no paths yet" in result
        assert "opt_a" in result


class TestFormatInteractionCandidatesContext:
    """Tests for format_interaction_candidates_context."""

    def _graph_with_dilemmas(self, dilemma_entities: dict[str, list[str]]) -> Graph:
        """Build a graph with dilemma nodes and anchored_to edges."""
        graph = Graph.empty()
        # Collect all unique entity IDs and create entity nodes
        all_entities: set[str] = set()
        for entities in dilemma_entities.values():
            all_entities.update(entities)
        for eid in sorted(all_entities):
            if not graph.has_node(eid):
                graph.create_node(eid, {"type": "entity", "raw_id": strip_scope_prefix(eid)})
        # Create dilemma nodes and anchored_to edges
        for did, entities in dilemma_entities.items():
            graph.create_node(
                f"dilemma::{did}",
                {
                    "type": "dilemma",
                    "raw_id": did,
                },
            )
            for eid in entities:
                graph.add_edge("anchored_to", f"dilemma::{did}", eid)
        return graph

    def test_shared_entity_pair_found(self) -> None:
        """Two dilemmas sharing an entity produce a candidate pair."""
        graph = self._graph_with_dilemmas(
            {
                "alpha": ["entity::hero", "entity::castle"],
                "beta": ["entity::hero", "entity::forest"],
            }
        )
        seed = _seed_output(dilemmas=[_dilemma("alpha"), _dilemma("beta")])
        result = format_interaction_candidates_context(seed, graph)
        assert "### Candidate Pairs" in result
        assert "dilemma::alpha" in result
        assert "dilemma::beta" in result
        assert "hero" in result

    def test_no_shared_entities_returns_no_candidates(self) -> None:
        """Disjoint entities produce no-candidates message."""
        graph = self._graph_with_dilemmas(
            {
                "alpha": ["entity::hero"],
                "beta": ["entity::villain"],
            }
        )
        seed = _seed_output(dilemmas=[_dilemma("alpha"), _dilemma("beta")])
        result = format_interaction_candidates_context(seed, graph)
        assert "No candidate pairs" in result

    def test_single_dilemma_returns_no_candidates(self) -> None:
        """Fewer than 2 dilemmas cannot have pairs."""
        graph = self._graph_with_dilemmas({"alpha": ["entity::hero"]})
        seed = _seed_output(dilemmas=[_dilemma("alpha")])
        result = format_interaction_candidates_context(seed, graph)
        assert "No candidate pairs" in result
