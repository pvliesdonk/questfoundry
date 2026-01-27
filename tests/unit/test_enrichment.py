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


@pytest.fixture
def graph_with_tensions() -> Graph:
    """Graph with BRAINSTORM dilemma data."""
    graph = Graph()
    # Add dilemma as BRAINSTORM would
    graph.create_node(
        "dilemma::host_motivation",
        {
            "type": "dilemma",
            "raw_id": "host_motivation",
            "question": "Is the host benevolent or self-serving?",
            "why_it_matters": "Determines whether protagonist can trust their guide",
            "central_entity_ids": ["entity::the_host", "entity::the_manor"],
        },
    )
    graph.create_node(
        "dilemma::killer_identity",
        {
            "type": "dilemma",
            "raw_id": "killer_identity",
            "question": "Who committed the murder?",
            "why_it_matters": "Core mystery that drives the plot",
            "central_entity_ids": ["entity::the_victim"],
        },
    )
    return graph


class TestEnrichSeedArtifact:
    """Tests for enrich_seed_artifact function."""

    def test_enriches_entity_with_full_details(self, graph_with_entities: Graph) -> None:
        """Enrichment adds entity_category, concept, notes from graph."""
        artifact = {
            "entities": [
                {"entity_id": "the_detective", "disposition": "retained"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        assert entity["entity_id"] == "the_detective"
        assert entity["entity_category"] == "character"
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
        assert entity["entity_category"] == "character"
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
        assert "entity_category" not in entity
        assert "concept" not in entity
        assert entity["disposition"] == "retained"

    def test_handles_prefixed_entity_ids(self, graph_with_entities: Graph) -> None:
        """Enrichment strips prefix from entity_id for graph lookup."""
        artifact = {
            "entities": [
                {"entity_id": "entity::the_detective", "disposition": "retained"},
            ],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        entity = result["entities"][0]
        # Original prefixed ID preserved in output
        assert entity["entity_id"] == "entity::the_detective"
        # But graph lookup succeeds with prefix stripped
        assert entity["entity_category"] == "character"
        assert entity["concept"] == "Seasoned detective known for solving intricate cases"
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
        assert result["entities"][0]["entity_category"] == "character"
        assert result["entities"][1]["entity_category"] == "location"
        assert result["entities"][2]["entity_category"] == "character"

    def test_preserves_other_artifact_fields(self, graph_with_entities: Graph) -> None:
        """Enrichment preserves non-entity fields in artifact."""
        artifact = {
            "entities": [{"entity_id": "the_detective", "disposition": "retained"}],
            "paths": [{"thread_id": "main_thread", "tier": "major"}],
            "beats": [{"beat_id": "opening", "summary": "Introduction"}],
        }

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert "paths" in result
        assert result["paths"] == artifact["paths"]
        assert "beats" in result
        assert result["beats"] == artifact["beats"]

    def test_empty_entities_list(self, graph_with_entities: Graph) -> None:
        """Enrichment handles empty entities list."""
        artifact = {"entities": []}

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert result["entities"] == []

    def test_missing_entities_key(self, graph_with_entities: Graph) -> None:
        """Enrichment handles missing entities key."""
        artifact = {"paths": []}

        result = enrich_seed_artifact(graph_with_entities, artifact)

        assert result["entities"] == []
        assert result["dilemmas"] == []
        assert result["paths"] == []

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
        # Full expected order: entity_id, entity_category, concept, notes, disposition
        expected_order = ["entity_id", "entity_category", "concept", "notes", "disposition"]
        assert keys == expected_order


class TestEnrichDilemmas:
    """Tests for dilemma enrichment."""

    def test_enriches_dilemma_with_full_details(self, graph_with_tensions: Graph) -> None:
        """Enrichment adds question, why_it_matters, central_entity_ids from graph."""
        artifact = {
            "tensions": [  # Input uses legacy "tensions" key
                {
                    "tension_id": "host_motivation",
                    "considered": ["benevolent"],
                    "implicit": ["self_serving"],
                },
            ],
        }

        result = enrich_seed_artifact(graph_with_tensions, artifact)

        dilemma = result["dilemmas"][0]
        assert dilemma["dilemma_id"] == "host_motivation"
        assert dilemma["question"] == "Is the host benevolent or self-serving?"
        assert dilemma["why_it_matters"] == "Determines whether protagonist can trust their guide"
        # Entity IDs should have prefix stripped
        assert dilemma["central_entity_ids"] == ["the_host", "the_manor"]
        assert dilemma["considered"] == ["benevolent"]
        assert dilemma["implicit"] == ["self_serving"]

    def test_handles_unknown_dilemma(self, graph_with_tensions: Graph) -> None:
        """Enrichment handles dilemmas not in graph gracefully."""
        artifact = {
            "tensions": [
                {"tension_id": "unknown_dilemma", "considered": ["option_a"], "implicit": []},
            ],
        }

        result = enrich_seed_artifact(graph_with_tensions, artifact)

        dilemma = result["dilemmas"][0]
        assert dilemma["dilemma_id"] == "unknown_dilemma"
        assert "question" not in dilemma
        assert "why_it_matters" not in dilemma
        assert dilemma["considered"] == ["option_a"]

    def test_handles_prefixed_dilemma_ids(self, graph_with_tensions: Graph) -> None:
        """Enrichment strips prefix from dilemma_id for graph lookup."""
        artifact = {
            "tensions": [
                {
                    "tension_id": "dilemma::host_motivation",
                    "considered": ["benevolent"],
                    "implicit": [],
                },
            ],
        }

        result = enrich_seed_artifact(graph_with_tensions, artifact)

        dilemma = result["dilemmas"][0]
        # Original prefixed ID preserved in output
        assert dilemma["dilemma_id"] == "dilemma::host_motivation"
        # But graph lookup succeeds with prefix stripped
        assert dilemma["question"] == "Is the host benevolent or self-serving?"
        assert dilemma["why_it_matters"] == "Determines whether protagonist can trust their guide"
        assert dilemma["considered"] == ["benevolent"]

    def test_enriches_multiple_dilemmas(self, graph_with_tensions: Graph) -> None:
        """Enrichment works for multiple dilemmas."""
        artifact = {
            "tensions": [
                {"tension_id": "host_motivation", "considered": ["benevolent"], "implicit": []},
                {"tension_id": "killer_identity", "considered": ["suspect_a"], "implicit": []},
            ],
        }

        result = enrich_seed_artifact(graph_with_tensions, artifact)

        assert len(result["dilemmas"]) == 2
        assert result["dilemmas"][0]["question"] == "Is the host benevolent or self-serving?"
        assert result["dilemmas"][1]["question"] == "Who committed the murder?"

    def test_empty_dilemmas_list(self, graph_with_tensions: Graph) -> None:
        """Enrichment handles empty dilemmas list."""
        artifact = {"tensions": []}  # Input uses legacy key

        result = enrich_seed_artifact(graph_with_tensions, artifact)

        assert result["dilemmas"] == []
