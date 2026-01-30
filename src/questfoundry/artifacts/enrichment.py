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
        "review_summary": _extract_review_summary(graph),
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
    """Extract passages with their prose (truncated for readability)."""
    passage_nodes = graph.get_nodes_by_type("passage")
    passages = []
    for passage_id in sorted(passage_nodes):
        data = passage_nodes[passage_id]
        prose = data.get("prose", "")
        if not prose:
            continue
        entry: dict[str, Any] = {
            "passage_id": passage_id,
            "from_beat": data.get("from_beat", ""),
        }
        # Truncate prose to first 200 chars for artifact readability
        if len(prose) > 200:
            entry["prose_snippet"] = prose[:200] + "..."
        else:
            entry["prose_snippet"] = prose
        entry["prose_length"] = len(prose)
        if flag := data.get("flag"):
            entry["flag"] = flag
        passages.append(entry)
    return passages


def _extract_review_summary(graph: Graph) -> dict[str, Any]:
    """Summarize review results from passage flags."""
    passage_nodes = graph.get_nodes_by_type("passage")
    total = len(passage_nodes)
    filled = sum(1 for p in passage_nodes.values() if p.get("prose"))
    flagged = sum(1 for p in passage_nodes.values() if p.get("flag"))
    reviewed = sum(1 for p in passage_nodes.values() if p.get("review_flags") is not None)

    return {
        "total_passages": total,
        "passages_with_prose": filled,
        "passages_flagged": flagged,
        "passages_reviewed": reviewed,
    }


# ---------------------------------------------------------------------------
# GROW artifact extraction helpers
# ---------------------------------------------------------------------------


def _extract_arcs(graph: Graph) -> list[dict[str, Any]]:
    """Extract arc nodes with their beat sequences from arc_contains edges."""
    arc_nodes = graph.get_nodes_by_type("arc")
    arcs = []
    for arc_id in sorted(arc_nodes):
        data = arc_nodes[arc_id]
        # Collect beats in this arc via arc_contains edges
        contains_edges = graph.get_edges(from_id=arc_id, edge_type="arc_contains")
        beat_sequence = sorted(e["to"] for e in contains_edges)
        entry: dict[str, Any] = {
            "arc_id": arc_id,
            "arc_type": data.get("arc_type", "branch"),
            "beat_sequence": beat_sequence,
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
        if belongs_to:
            entry["belongs_to"] = belongs_to
        if entities := data.get("entities"):
            entry["entities"] = entities
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
        if entities := data.get("entities"):
            entry["entities"] = entities
        passages.append(entry)
    return passages


def _extract_choices(graph: Graph) -> list[dict[str, Any]]:
    """Extract choice nodes with from/to passages from edges."""
    choice_nodes = graph.get_nodes_by_type("choice")
    choices = []
    for choice_id in sorted(choice_nodes):
        data = choice_nodes[choice_id]
        from_edges = graph.get_edges(from_id=choice_id, edge_type="choice_from")
        to_edges = graph.get_edges(from_id=choice_id, edge_type="choice_to")
        requires_edges = graph.get_edges(from_id=choice_id, edge_type="requires")
        entry: dict[str, Any] = {
            "choice_id": choice_id,
            "label": data.get("label", ""),
        }
        if from_edges:
            if len(from_edges) > 1:
                log.warning(
                    "multiple_choice_from_edges", choice_id=choice_id, count=len(from_edges)
                )
            entry["from_passage"] = from_edges[0]["to"]
        if to_edges:
            if len(to_edges) > 1:
                log.warning("multiple_choice_to_edges", choice_id=choice_id, count=len(to_edges))
            entry["to_passage"] = to_edges[0]["to"]
        if requires_edges:
            entry["requires"] = sorted(e["to"] for e in requires_edges)
        choices.append(entry)
    return choices


def _extract_codewords(graph: Graph) -> list[dict[str, Any]]:
    """Extract codeword nodes with tracks and granted_by relationships."""
    codeword_nodes = graph.get_nodes_by_type("codeword")
    codewords = []
    for cw_id in sorted(codeword_nodes):
        data = codeword_nodes[cw_id]
        tracks_edges = graph.get_edges(from_id=cw_id, edge_type="tracks")
        grants_edges = graph.get_edges(to_id=cw_id, edge_type="grants")
        entry: dict[str, Any] = {
            "codeword_id": cw_id,
        }
        if tracks_edges:
            if len(tracks_edges) > 1:
                log.warning("multiple_tracks_edges", codeword_id=cw_id, count=len(tracks_edges))
            entry["tracks"] = tracks_edges[0]["to"]
        if grants_edges:
            entry["granted_by"] = sorted(e["from"] for e in grants_edges)
        if raw_id := data.get("raw_id"):
            entry["raw_id"] = raw_id
        codewords.append(entry)
    return codewords
