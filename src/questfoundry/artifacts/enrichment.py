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
    """Enrich SEED artifact with BRAINSTORM entity and dilemma details.

    Merges entity details (entity_category, concept, notes) and dilemma details
    (question, why_it_matters, central_entity_ids) from graph nodes into SEED
    decisions. This provides full context in the YAML artifact without requiring
    the LLM to repeat information.

    Args:
        graph: Graph containing BRAINSTORM entity and dilemma nodes.
        artifact: SEED artifact dict with entity and dilemma decisions.

    Returns:
        Enriched artifact dict with full entity and dilemma details.

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

    # Enrich dilemmas (artifact may use "dilemmas" or legacy "tensions")
    dilemma_decisions = artifact.get("dilemmas", artifact.get("tensions", []))
    enriched["dilemmas"] = _enrich_dilemmas(graph, dilemma_decisions)

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
        # Strip prefix if present (e.g., "entity::the_detective" -> "the_detective")
        lookup_id = entity_id.split("::")[-1]
        node = entity_data.get(lookup_id, {}) if lookup_id else {}

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


def _enrich_dilemmas(graph: Graph, dilemma_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich dilemma decisions with BRAINSTORM details."""
    # Build lookup: raw_id -> node data for dilemmas
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    dilemma_data: dict[str, dict[str, Any]] = {
        node["raw_id"]: node for node in dilemma_nodes.values() if node.get("raw_id")
    }

    enriched_dilemmas = []
    for decision in dilemma_decisions:
        dilemma_id = decision.get("dilemma_id", "")
        # Strip prefix if present (e.g., "dilemma::host_motivation" -> "host_motivation")
        lookup_id = dilemma_id.split("::")[-1]
        node = dilemma_data.get(lookup_id, {}) if lookup_id else {}

        # Build enriched dilemma in consistent field order
        enriched: dict[str, Any] = {
            "dilemma_id": dilemma_id,
        }

        # Add BRAINSTORM fields if available (simple copy for most fields)
        for field in ("question", "why_it_matters"):
            if value := node.get(field):
                enriched[field] = value

        # Handle central_entity_ids with prefix stripping for readability
        if value := node.get("central_entity_ids"):
            enriched["central_entity_ids"] = [eid.split("::")[-1] for eid in value]

        # Add SEED decision fields (support both old 'explored' and new 'considered')
        enriched["considered"] = decision.get("considered", decision.get("explored", []))
        enriched["implicit"] = decision.get("implicit", [])

        enriched_dilemmas.append(enriched)

    log.debug(
        "dilemmas_enriched",
        stage="seed",
        dilemma_count=len(enriched_dilemmas),
        enriched_count=sum(1 for d in enriched_dilemmas if "question" in d),
    )

    return enriched_dilemmas
