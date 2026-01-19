"""Artifact enrichment with graph context.

Enriches stage artifacts with data from previous stages stored in the graph.
This provides full context in human-readable artifacts without requiring the
LLM to repeat information from earlier stages.

Example: SEED entities include disposition (from SEED) plus entity_category,
concept, notes (from BRAINSTORM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


def enrich_seed_artifact(graph: Graph, artifact: dict[str, Any]) -> dict[str, Any]:
    """Enrich SEED artifact with BRAINSTORM entity and tension details.

    Merges entity details (entity_category, concept, notes) and tension details
    (question, why_it_matters, central_entity_ids) from graph nodes into SEED
    decisions. This provides full context in the YAML artifact without requiring
    the LLM to repeat information.

    Args:
        graph: Graph containing BRAINSTORM entity and tension nodes.
        artifact: SEED artifact dict with entity and tension decisions.

    Returns:
        Enriched artifact dict with full entity and tension details.

    Example:
        Input artifact:
            {"entities": [{"entity_id": "the_detective", "disposition": "retained"}]}

        Output:
            {"entities": [{
                "entity_id": "the_detective",
                "entity_category": "character",
                "concept": "Seasoned detective...",
                "notes": "Inspector Reginald...",
                "disposition": "retained"
            }]}
    """
    enriched = dict(artifact)

    # Enrich entities
    enriched["entities"] = _enrich_entities(graph, artifact.get("entities", []))

    # Enrich tensions
    enriched["tensions"] = _enrich_tensions(graph, artifact.get("tensions", []))

    return enriched


def _enrich_entities(graph: Graph, entity_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich entity decisions with BRAINSTORM details."""
    # Build lookup: raw_id -> node data for entities
    entity_nodes = graph.get_nodes_by_type("entity")
    entity_data: dict[str, dict[str, Any]] = {
        node["raw_id"]: node for node in entity_nodes.values() if node.get("raw_id")
    }

    enriched_entities = []
    for decision in entity_decisions:
        entity_id = decision.get("entity_id", "")
        node = entity_data.get(entity_id, {}) if entity_id else {}

        # Build enriched entity in consistent field order
        enriched: dict[str, Any] = {
            "entity_id": entity_id,
        }

        # Add BRAINSTORM fields if available
        # Map graph field names to output field names (entity_type -> entity_category)
        entity_field_map = {
            "entity_type": "entity_category",
            "concept": "concept",
            "notes": "notes",
        }
        for source_field, target_field in entity_field_map.items():
            if value := node.get(source_field):
                enriched[target_field] = value

        # Add SEED disposition
        enriched["disposition"] = decision.get("disposition")

        enriched_entities.append(enriched)

    log.debug(
        "entities_enriched",
        stage="seed",
        entity_count=len(enriched_entities),
        enriched_count=sum(1 for e in enriched_entities if "concept" in e),
    )

    return enriched_entities


def _enrich_tensions(graph: Graph, tension_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich tension decisions with BRAINSTORM details."""
    # Build lookup: raw_id -> node data for tensions
    tension_nodes = graph.get_nodes_by_type("tension")
    tension_data: dict[str, dict[str, Any]] = {
        node["raw_id"]: node for node in tension_nodes.values() if node.get("raw_id")
    }

    enriched_tensions = []
    for decision in tension_decisions:
        tension_id = decision.get("tension_id", "")
        node = tension_data.get(tension_id, {}) if tension_id else {}

        # Build enriched tension in consistent field order
        enriched: dict[str, Any] = {
            "tension_id": tension_id,
        }

        # Add BRAINSTORM fields if available (simple copy for most fields)
        for field in ("question", "why_it_matters"):
            if value := node.get(field):
                enriched[field] = value

        # Handle central_entity_ids with prefix stripping for readability
        if value := node.get("central_entity_ids"):
            enriched["central_entity_ids"] = [
                eid.split("::")[-1] if "::" in eid else eid for eid in value
            ]

        # Add SEED decision fields
        enriched["explored"] = decision.get("explored", [])
        enriched["implicit"] = decision.get("implicit", [])

        enriched_tensions.append(enriched)

    log.debug(
        "tensions_enriched",
        stage="seed",
        tension_count=len(enriched_tensions),
        enriched_count=sum(1 for t in enriched_tensions if "question" in t),
    )

    return enriched_tensions
