"""Artifact enrichment with graph context.

Enriches stage artifacts with data from previous stages stored in the graph.
This provides full context in human-readable artifacts without requiring the
LLM to repeat information from earlier stages.

Example: SEED entities include disposition (from SEED) plus entity_category,
concept, notes (from BRAINSTORM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import strip_scope_prefix
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


def extract_grow_artifact(graph: Graph) -> dict[str, Any]:
    """Extract GROW story data from graph into a human-readable artifact.

    Reads arc, beat, passage, choice, and codeword nodes plus their
    connecting edges to build a flat artifact dict suitable for YAML export.

    Args:
        graph: Graph after GROW phases have completed.

    Returns:
        Dict with arcs, beats, passages, choices, codewords lists.
    """
    return {
        "arcs": _extract_arcs(graph),
        "beats": _extract_beats(graph),
        "passages": _extract_passages(graph),
        "choices": _extract_choices(graph),
        "codewords": _extract_codewords(graph),
    }


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

    # Enrich dilemmas
    enriched["dilemmas"] = _enrich_dilemmas(graph, artifact.get("dilemmas", []))

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
        lookup_id = strip_scope_prefix(entity_id)
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
        lookup_id = strip_scope_prefix(dilemma_id)
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
            enriched["central_entity_ids"] = [strip_scope_prefix(eid) for eid in value]

        # Add SEED decision fields (supports old 'considered' and 'implicit' field names)
        enriched["explored"] = decision.get("explored", decision.get("considered", []))
        enriched["unexplored"] = decision.get("unexplored", decision.get("implicit", []))

        enriched_dilemmas.append(enriched)

    log.debug(
        "dilemmas_enriched",
        stage="seed",
        dilemma_count=len(enriched_dilemmas),
        enriched_count=sum(1 for d in enriched_dilemmas if "question" in d),
    )

    return enriched_dilemmas


def extract_fill_artifact(graph: Graph) -> dict[str, Any]:
    """Extract FILL prose data from graph into a human-readable artifact.

    Reads voice document, passage prose, and entity updates from the graph
    to build a flat artifact dict suitable for YAML export.

    Args:
        graph: Graph after FILL phases have completed.

    Returns:
        Dict with voice_document, passages (with prose snippets),
        and review_summary.
    """
    return {
        "voice_document": _extract_voice_document(graph),
        "passages": _extract_filled_passages(graph),
    }


# ---------------------------------------------------------------------------
# FILL artifact extraction helpers
# ---------------------------------------------------------------------------


def _extract_voice_document(graph: Graph) -> dict[str, Any]:
    """Extract the voice document from graph."""
    voice_nodes = graph.get_nodes_by_type("voice")
    if not voice_nodes:
        return {}

    voice_data = next(iter(voice_nodes.values()))
    return {k: v for k, v in voice_data.items() if k not in ("type", "raw_id") and v is not None}


def _extract_filled_passages(graph: Graph) -> list[dict[str, Any]]:
    """Extract all passages with their full prose.

    Per spec: "FILL Output — All passages with prose populated."
    Includes passages without prose (flagged as missing) so the artifact
    is a complete manifest of every passage in the story.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    passages = []
    for passage_id in sorted(passage_nodes):
        data = passage_nodes[passage_id]
        entry: dict[str, Any] = {
            "passage_id": passage_id,
            "from_beat": data.get("from_beat", ""),
        }
        prose = data.get("prose", "")
        if prose:
            entry["prose"] = prose
        if flag := data.get("flag"):
            entry["flag"] = flag
        passages.append(entry)
    return passages


# ---------------------------------------------------------------------------
# GROW artifact extraction helpers
# ---------------------------------------------------------------------------


def _extract_arcs(graph: Graph) -> list[dict[str, Any]]:
    """Extract arc nodes with their ordered beat sequences.

    Prefer the arc node's `sequence` field (the narrative order). Fall back to
    arc_contains edges if needed.
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    arcs = []
    for arc_id in sorted(arc_nodes):
        data = arc_nodes[arc_id]
        beat_sequence = list(data.get("sequence") or [])
        if not beat_sequence:
            # Collect beats in this arc via arc_contains edges (unordered)
            contains_edges = graph.get_edges(from_id=arc_id, edge_type="arc_contains")
            beat_sequence = [e["to"] for e in contains_edges]
        entry: dict[str, Any] = {
            "arc_id": arc_id,
            "arc_type": data.get("arc_type", "branch"),
            "paths": sorted(data.get("paths", []) or []),
            "sequence": beat_sequence,
        }
        arcs.append(entry)
    return arcs


def _extract_beats(graph: Graph) -> list[dict[str, Any]]:
    """Extract beat nodes with path membership from belongs_to edges."""
    beat_nodes = graph.get_nodes_by_type("beat")
    beats = []
    for beat_id in sorted(beat_nodes):
        data = beat_nodes[beat_id]
        belongs_edges = graph.get_edges(from_id=beat_id, edge_type="belongs_to")
        belongs_to = sorted(e["to"] for e in belongs_edges)
        entry: dict[str, Any] = {
            "beat_id": beat_id,
            "summary": data.get("summary", ""),
        }
        if scene_type := data.get("scene_type"):
            entry["scene_type"] = scene_type
        if narrative_fn := data.get("narrative_function"):
            entry["narrative_function"] = narrative_fn
        if location := data.get("location"):
            entry["location"] = location
        if intersection_group := data.get("intersection_group"):
            entry["intersection_group"] = sorted(intersection_group)
        if belongs_to:
            entry["belongs_to"] = belongs_to
        if entities := data.get("entities"):
            entry["entities"] = sorted(entities)
        beats.append(entry)
    return beats


def _extract_passages(graph: Graph) -> list[dict[str, Any]]:
    """Extract passage nodes with their source beat."""
    passage_nodes = graph.get_nodes_by_type("passage")
    passages = []
    for passage_id in sorted(passage_nodes):
        data = passage_nodes[passage_id]
        entry: dict[str, Any] = {
            "passage_id": passage_id,
            "from_beat": data.get("from_beat", ""),
            "summary": data.get("summary", ""),
        }
        if data.get("is_synthetic") is True:
            entry["is_synthetic"] = True
        if entities := data.get("entities"):
            entry["entities"] = sorted(entities)
        passages.append(entry)
    return passages


def _extract_choices(graph: Graph) -> list[dict[str, Any]]:
    """Extract choice nodes with from/to passages from edges."""
    choice_nodes = graph.get_nodes_by_type("choice")
    choices = []
    for choice_id in sorted(choice_nodes):
        data = choice_nodes[choice_id]
        entry: dict[str, Any] = {
            "choice_id": choice_id,
            "label": data.get("label", ""),
        }
        from_passage = data.get("from_passage")
        to_passage = data.get("to_passage")
        if not from_passage:
            from_edges = graph.get_edges(from_id=choice_id, edge_type="choice_from")
            if from_edges:
                if len(from_edges) > 1:
                    log.warning(
                        "multiple_choice_from_edges", choice_id=choice_id, count=len(from_edges)
                    )
                from_passage = from_edges[0]["to"]
        if not to_passage:
            to_edges = graph.get_edges(from_id=choice_id, edge_type="choice_to")
            if to_edges:
                if len(to_edges) > 1:
                    log.warning(
                        "multiple_choice_to_edges", choice_id=choice_id, count=len(to_edges)
                    )
                to_passage = to_edges[0]["to"]
        if from_passage:
            entry["from_passage"] = from_passage
        if to_passage:
            entry["to_passage"] = to_passage
        if requires := data.get("requires"):
            entry["requires"] = sorted(requires)
        if grants := data.get("grants"):
            entry["grants"] = sorted(grants)
        if data.get("is_return") is True:
            entry["is_return"] = True
        choices.append(entry)
    return choices


def _extract_codewords(graph: Graph) -> list[dict[str, Any]]:
    """Extract codeword nodes with tracks and granted_by relationships."""
    codeword_nodes = graph.get_nodes_by_type("codeword")
    codewords = []
    for cw_id in sorted(codeword_nodes):
        data = codeword_nodes[cw_id]
        entry: dict[str, Any] = {
            "codeword_id": cw_id,
        }
        if raw_id := data.get("raw_id"):
            entry["raw_id"] = raw_id
        tracks = data.get("tracks")
        if not tracks:
            tracks_edges = graph.get_edges(from_id=cw_id, edge_type="tracks")
            if tracks_edges:
                if len(tracks_edges) > 1:
                    log.warning("multiple_tracks_edges", codeword_id=cw_id, count=len(tracks_edges))
                tracks = tracks_edges[0]["to"]
        if tracks:
            entry["tracks"] = tracks
        if cw_type := data.get("codeword_type"):
            entry["codeword_type"] = cw_type
        granted_by = data.get("granted_by")
        if not granted_by:
            grants_edges = graph.get_edges(to_id=cw_id, edge_type="grants")
            if grants_edges:
                granted_by = sorted(e["from"] for e in grants_edges)
        if granted_by:
            entry["granted_by"] = sorted(granted_by)
        codewords.append(entry)
    return codewords
