"""Build ExportContext from a story graph.

Extracts all player-facing nodes and edges from the graph, ignoring
working nodes (dilemmas, paths, beats, arcs, etc.) and internal edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.export.base import (
    ExportChoice,
    ExportCodeword,
    ExportCodexEntry,
    ExportContext,
    ExportEntity,
    ExportIllustration,
    ExportPassage,
)

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def build_export_context(graph: Graph, project_name: str) -> ExportContext:
    """Extract player-facing data from the story graph.

    Args:
        graph: Loaded story graph.
        project_name: Project name used as story title.

    Returns:
        ExportContext containing all data needed by exporters.

    Raises:
        ValueError: If no passages exist in the graph.
    """
    passages = _extract_passages(graph)
    if not passages:
        msg = "Graph contains no passages — nothing to export"
        raise ValueError(msg)

    choices = _extract_choices(graph)
    _mark_start_and_endings(passages, choices)

    illustrations, cover = _extract_illustrations(graph)

    return ExportContext(
        title=project_name,
        passages=passages,
        choices=choices,
        entities=_extract_entities(graph),
        codewords=_extract_codewords(graph),
        illustrations=illustrations,
        cover=cover,
        codex_entries=_extract_codex_entries(graph),
        art_direction=_extract_art_direction(graph),
    )


def _extract_passages(graph: Graph) -> list[ExportPassage]:
    """Extract passage nodes with prose."""
    nodes = graph.get_nodes_by_type("passage")
    return [
        ExportPassage(
            id=node_id,
            prose=data.get("prose", ""),
        )
        for node_id, data in sorted(nodes.items())
    ]


def _extract_choices(graph: Graph) -> list[ExportChoice]:
    """Extract choice nodes (navigation links between passages)."""
    nodes = graph.get_nodes_by_type("choice")
    return [
        ExportChoice(
            from_passage=data["from_passage"],
            to_passage=data["to_passage"],
            label=data.get("label", "continue"),
            requires=data.get("requires", []),
            grants=data.get("grants", []),
        )
        for _node_id, data in sorted(nodes.items())
    ]


def _mark_start_and_endings(
    passages: list[ExportPassage],
    choices: list[ExportChoice],
) -> None:
    """Determine start passage (no incoming choices) and endings (no outgoing)."""
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()

    for choice in choices:
        has_incoming.add(choice.to_passage)
        has_outgoing.add(choice.from_passage)

    for passage in passages:
        if passage.id not in has_incoming:
            passage.is_start = True
        if passage.id not in has_outgoing:
            passage.is_ending = True


def _extract_entities(graph: Graph) -> list[ExportEntity]:
    """Extract entity nodes."""
    nodes = graph.get_nodes_by_type("entity")
    return [
        ExportEntity(
            id=node_id,
            entity_type=data.get("entity_type", "unknown"),
            concept=data.get("concept", ""),
            overlays=data.get("overlays", []),
        )
        for node_id, data in sorted(nodes.items())
    ]


def _extract_codewords(graph: Graph) -> list[ExportCodeword]:
    """Extract codeword nodes."""
    nodes = graph.get_nodes_by_type("codeword")
    return [
        ExportCodeword(
            id=node_id,
            codeword_type=data.get("codeword_type", "granted"),
            tracks=data.get("tracks"),
        )
        for node_id, data in sorted(nodes.items())
    ]


def _extract_illustrations(
    graph: Graph,
) -> tuple[list[ExportIllustration], ExportIllustration | None]:
    """Extract illustrations, separating cover from passage illustrations.

    Returns:
        Tuple of (passage_illustrations, cover_illustration).
        Cover is identified by category="cover" and absence of a Depicts edge.
    """
    illustration_nodes = graph.get_nodes_by_type("illustration")
    if not illustration_nodes:
        return [], None

    # Build illustration→passage mapping from Depicts edges
    depicts_edges = graph.get_edges(edge_type="Depicts")
    illust_to_passage: dict[str, str] = {}
    for edge in depicts_edges:
        illust_to_passage[edge["from"]] = edge["to"]

    passage_illustrations: list[ExportIllustration] = []
    cover: ExportIllustration | None = None
    for node_id, data in sorted(illustration_nodes.items()):
        category = data.get("category", "scene")
        passage_id = illust_to_passage.get(node_id)
        if category == "cover" and not passage_id:
            cover = ExportIllustration(
                passage_id="",
                asset_path=data.get("asset", ""),
                caption=data.get("caption", ""),
                category="cover",
            )
        elif passage_id:
            passage_illustrations.append(
                ExportIllustration(
                    passage_id=passage_id,
                    asset_path=data.get("asset", ""),
                    caption=data.get("caption", ""),
                    category=category,
                )
            )
    return passage_illustrations, cover


def _extract_codex_entries(graph: Graph) -> list[ExportCodexEntry]:
    """Extract codex entries linked to entities via HasEntry edges."""
    codex_nodes = graph.get_nodes_by_type("codex_entry")
    if not codex_nodes:
        return []

    # Build codex→entity mapping from HasEntry edges
    has_entry_edges = graph.get_edges(edge_type="HasEntry")
    codex_to_entity: dict[str, str] = {}
    for edge in has_entry_edges:
        codex_to_entity[edge["from"]] = edge["to"]

    result: list[ExportCodexEntry] = []
    for node_id, data in sorted(codex_nodes.items()):
        entity_id = codex_to_entity.get(node_id)
        if entity_id:
            result.append(
                ExportCodexEntry(
                    entity_id=entity_id,
                    rank=data.get("rank", 0),
                    visible_when=data.get("visible_when", []),
                    content=data.get("content", ""),
                )
            )
    return result


def _extract_art_direction(graph: Graph) -> dict[str, Any] | None:
    """Extract art direction node if DRESS stage was run."""
    nodes = graph.get_nodes_by_type("art_direction")
    if not nodes:
        return None
    # There's typically one art_direction::main node
    _node_id, data = next(iter(nodes.items()))
    return {k: v for k, v in data.items() if k not in ("type", "raw_id")}
