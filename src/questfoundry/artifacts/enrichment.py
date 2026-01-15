"""Artifact enrichment with graph context.

Enriches stage artifacts with data from previous stages stored in the graph.
This provides full context in human-readable artifacts without requiring the
LLM to repeat information from earlier stages.

Example: SEED entities include disposition (from SEED) plus entity_type,
concept, notes (from BRAINSTORM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


def enrich_seed_artifact(graph: Graph, artifact: dict[str, Any]) -> dict[str, Any]:
    """Enrich SEED artifact with BRAINSTORM entity details.

    Merges entity details (entity_type, concept, notes) from graph nodes
    into entity decisions. This provides full context in the YAML artifact
    without requiring the LLM to repeat information.

    Args:
        graph: Graph containing BRAINSTORM entity nodes.
        artifact: SEED artifact dict with entity decisions.

    Returns:
        Enriched artifact dict with full entity details.

    Example:
        Input artifact:
            {"entities": [{"entity_id": "the_detective", "disposition": "retained"}]}

        Output:
            {"entities": [{
                "entity_id": "the_detective",
                "entity_type": "character",
                "concept": "Seasoned detective...",
                "notes": "Inspector Reginald...",
                "disposition": "retained"
            }]}
    """
    # Build lookup: raw_id -> node data for entities
    entity_nodes = graph.get_nodes_by_type("entity")
    entity_data: dict[str, dict[str, Any]] = {
        node["raw_id"]: node for node in entity_nodes.values() if node.get("raw_id")
    }

    # Enrich entity decisions
    enriched_entities = []
    for decision in artifact.get("entities", []):
        entity_id = decision.get("entity_id")
        node = entity_data.get(entity_id, {})

        # Build enriched entity in consistent field order
        enriched: dict[str, Any] = {
            "entity_id": entity_id,
        }

        # Add BRAINSTORM fields if available
        for field in ("entity_type", "concept", "notes"):
            if value := node.get(field):
                enriched[field] = value

        # Add SEED disposition
        enriched["disposition"] = decision.get("disposition")

        enriched_entities.append(enriched)

    log.debug(
        "artifact_enriched",
        stage="seed",
        entity_count=len(enriched_entities),
        enriched_count=sum(1 for e in enriched_entities if "concept" in e),
    )

    return {**artifact, "entities": enriched_entities}
