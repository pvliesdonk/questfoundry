"""Tests for artifact enrichment."""

from __future__ import annotations

import pytest

from questfoundry.artifacts.enrichment import enrich_seed_artifact
from questfoundry.graph import Graph


@pytest.fixture
def graph_with_entities() -> Graph:
    """Graph with BRAINSTORM entity data."""
    graph = Graph()
    # Add entities as BRAINSTORM would
    graph.create_node(
        "entity::the_detective",
        {
            "type": "entity",
            "raw_id": "the_detective",
            "entity_type": "character",
            "concept": "Seasoned detective known for solving intricate cases",
            "notes": "Inspector Reginald Hargrove is known for sharp wit",
            "disposition": "proposed",
        },
    )
    graph.create_node(
        "entity::the_manor",
        {
            "type": "entity",
            "raw_id": "the_manor",
            "entity_type": "location",
            "concept": "Grand Victorian manor with hidden passages",
            "notes": "Whitmore Manor has been in the family for generations",
            "disposition": "proposed",
        },
    )
    graph.create_node(
        "entity::the_victim",
        {
            "type": "entity",
            "raw_id": "the_victim",
            "entity_type": "character",
            "concept": "Wealthy socialite whose death triggers the mystery",
            "disposition": "proposed",
        },
    )
    return graph


class TestEnrichSeedArtifact:
    """Tests for enrich_seed_artifact function."""

    def test_enriches_entity_with_full_details(self, graph_with_entities: Graph) -> None:
        """Enrichment adds entity_type, concept, notes from graph."""
        artifact = {
            "entities": [
                {"entity_id": "the_detective", "disposition": "retained"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        assert entity["entity_id"] == "the_detective"
        assert entity["entity_type"] == "character"
        assert entity["concept"] == "Seasoned detective known for solving intricate cases"
        assert entity["notes"] == "Inspector Reginald Hargrove is known for sharp wit"
        assert entity["disposition"] == "retained"

    def test_handles_entity_without_notes(self, graph_with_entities: Graph) -> None:
        """Enrichment handles entities missing optional fields."""
        artifact = {
            "entities": [
                {"entity_id": "the_victim", "disposition": "cut"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        assert entity["entity_id"] == "the_victim"
        assert entity["entity_type"] == "character"
        assert entity["concept"] == "Wealthy socialite whose death triggers the mystery"
        assert "notes" not in entity  # Not added if not in graph
        assert entity["disposition"] == "cut"

    def test_handles_unknown_entity(self, graph_with_entities: Graph) -> None:
        """Enrichment handles entities not in graph gracefully."""
        artifact = {
            "entities": [
                {"entity_id": "unknown_entity", "disposition": "retained"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        assert entity["entity_id"] == "unknown_entity"
        assert "entity_type" not in entity
        assert "concept" not in entity
        assert entity["disposition"] == "retained"

    def test_enriches_multiple_entities(self, graph_with_entities: Graph) -> None:
        """Enrichment works for multiple entities."""
        artifact = {
            "entities": [
                {"entity_id": "the_detective", "disposition": "retained"},
                {"entity_id": "the_manor", "disposition": "retained"},
                {"entity_id": "the_victim", "disposition": "cut"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert len(result["entities"]) == 3
        assert result["entities"][0]["entity_type"] == "character"
        assert result["entities"][1]["entity_type"] == "location"
        assert result["entities"][2]["entity_type"] == "character"

    def test_preserves_other_artifact_fields(self, graph_with_entities: Graph) -> None:
        """Enrichment preserves non-entity fields in artifact."""
        artifact = {
            "entities": [{"entity_id": "the_detective", "disposition": "retained"}],
            "threads": [{"thread_id": "main_thread", "tier": "major"}],
            "beats": [{"beat_id": "opening", "summary": "Introduction"}],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert "threads" in result
        assert result["threads"] == artifact["threads"]
        assert "beats" in result
        assert result["beats"] == artifact["beats"]

    def test_empty_entities_list(self, graph_with_entities: Graph) -> None:
        """Enrichment handles empty entities list."""
        artifact = {"entities": []}

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert result["entities"] == []

    def test_missing_entities_key(self, graph_with_entities: Graph) -> None:
        """Enrichment handles missing entities key."""
        artifact = {"threads": []}

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert result["entities"] == []
        assert result["threads"] == []

    def test_field_order_consistent(self, graph_with_entities: Graph) -> None:
        """Enriched entities have consistent field order."""
        artifact = {
            "entities": [
                {"entity_id": "the_detective", "disposition": "retained"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        keys = list(entity.keys())
        # Full expected order: entity_id, entity_type, concept, notes, disposition
        expected_order = ["entity_id", "entity_type", "concept", "notes", "disposition"]
        assert keys == expected_order
