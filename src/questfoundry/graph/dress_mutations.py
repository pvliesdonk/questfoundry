"""DRESS stage graph mutation helpers.

DRESS manages its own graph directly (like FILL and GROW), not through
the orchestrator's ``apply_mutations()`` dispatch. These helper functions
create DRESS-specific nodes and edges in the graph.

Node ID conventions:
    art_direction::main          — singleton global visual identity
    entity_visual::{entity_id}   — per-entity visual profile
    illustration_brief::{passage_id} — per-passage image prompt
    codex::{entity_id}_rank{N}   — per-entity codex tier
    illustration::{passage_id}      — rendered image asset
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import strip_scope_prefix

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


# ---------------------------------------------------------------------------
# Art Direction (Phase 0)
# ---------------------------------------------------------------------------


def apply_dress_art_direction(
    graph: Graph,
    art_dir: dict[str, Any],
    entity_visuals: list[dict[str, Any]],
) -> None:
    """Create ArtDirection node, EntityVisual nodes, and describes_visual edges.

    Uses ``upsert_node`` for ArtDirection to allow re-running Phase 0.
    Uses ``create_node`` for EntityVisuals (duplicates are an error).

    Args:
        graph: Story graph to mutate.
        art_dir: ArtDirection fields (style, medium, palette, etc.).
        entity_visuals: List of dicts with ``entity_id`` plus EntityVisual fields.

    Raises:
        NodeNotFoundError: If a referenced entity doesn't exist in the graph.
    """
    # Singleton art direction node
    ad_data = {"type": "art_direction", **_clean_dict(art_dir)}
    graph.upsert_node("art_direction::main", ad_data)

    # Per-entity visual profiles
    for ev in entity_visuals:
        ev = dict(ev)  # copy to avoid mutating caller's data
        entity_id = strip_scope_prefix(ev.pop("entity_id"))
        node_id = f"entity_visual::{entity_id}"
        ev_data = {"type": "entity_visual", **_clean_dict(ev)}
        graph.upsert_node(node_id, ev_data)
        # Edge: entity_visual → entity
        entity_ref = graph.ref("entity", entity_id)
        # Remove existing edge if re-running
        _remove_edges(graph, from_id=node_id, edge_type="describes_visual")
        graph.add_edge("describes_visual", node_id, entity_ref)


# ---------------------------------------------------------------------------
# Illustration Briefs (Phase 1)
# ---------------------------------------------------------------------------


def apply_dress_brief(
    graph: Graph,
    passage_id: str,
    brief: dict[str, Any],
    priority: int,
) -> str:
    """Create an IllustrationBrief node and targets edge.

    Args:
        graph: Story graph to mutate.
        passage_id: Scoped or raw passage ID (e.g., ``passage::opening``).
        brief: IllustrationBrief fields (subject, composition, mood, etc.).
        priority: Final computed priority (1-3).

    Returns:
        The created brief node ID.

    Raises:
        NodeNotFoundError: If the passage doesn't exist.
    """
    raw_passage_id = strip_scope_prefix(passage_id)
    node_id = f"illustration_brief::{raw_passage_id}"
    brief_data = {
        "type": "illustration_brief",
        "priority": priority,
        **_clean_dict(brief),
    }
    graph.upsert_node(node_id, brief_data)

    # Edge: brief → passage
    passage_ref = graph.ref("passage", raw_passage_id)
    _remove_edges(graph, from_id=node_id, edge_type="targets")
    graph.add_edge("targets", node_id, passage_ref)

    return node_id


# ---------------------------------------------------------------------------
# Codex Entries (Phase 2)
# ---------------------------------------------------------------------------


def apply_dress_codex(
    graph: Graph,
    entity_id: str,
    entries: list[dict[str, Any]],
) -> list[str]:
    """Create CodexEntry nodes and HasEntry edges for one entity.

    Args:
        graph: Story graph to mutate.
        entity_id: Scoped or raw entity ID.
        entries: List of CodexEntry dicts (rank, visible_when, content).

    Returns:
        List of created codex node IDs.

    Raises:
        NodeNotFoundError: If the entity doesn't exist.
    """
    raw_entity_id = strip_scope_prefix(entity_id)
    entity_ref = graph.ref("entity", raw_entity_id)
    created_ids: list[str] = []

    for entry in entries:
        if "rank" not in entry:
            msg = f"Entity {raw_entity_id}: codex entry missing required 'rank' field"
            raise ValueError(msg)
        rank = entry["rank"]
        node_id = f"codex::{raw_entity_id}_rank{rank}"
        entry_data = {"type": "codex_entry", **_clean_dict(entry)}
        graph.upsert_node(node_id, entry_data)

        # Edge: codex_entry → entity
        _remove_edges(graph, from_id=node_id, edge_type="HasEntry")
        graph.add_edge("HasEntry", node_id, entity_ref)
        created_ids.append(node_id)

    return created_ids


# ---------------------------------------------------------------------------
# Illustrations (Phase 4)
# ---------------------------------------------------------------------------


def apply_dress_illustration(
    graph: Graph,
    brief_id: str,
    asset_path: str,
    caption: str,
    category: str,
    quality: str = "high",
) -> str:
    """Create an Illustration node with Depicts and from_brief edges.

    The brief's ``targets`` edge determines the depicted node, which may
    be a passage (normal illustrations) or another node type such as
    ``vision::main`` (cover illustration).

    Args:
        graph: Story graph to mutate.
        brief_id: IllustrationBrief node ID that was rendered.
        asset_path: Relative path to the image file.
        caption: Diegetic caption from the brief.
        category: Image category (scene, portrait, vista, item_detail).
        quality: Image quality tier (placeholder, low, high).

    Returns:
        The created illustration node ID.

    Raises:
        ValueError: If the brief has no targets edge.
    """
    # Derive target node from brief's targets edge
    targets_edges = graph.get_edges(from_id=brief_id, edge_type="targets")
    if not targets_edges:
        msg = f"Brief {brief_id} has no targets edge"
        raise ValueError(msg)

    target_id = targets_edges[0]["to"]
    raw_target_id = strip_scope_prefix(target_id)
    node_id = f"illustration::{raw_target_id}"

    illust_data = {
        "type": "illustration",
        "asset": asset_path,
        "caption": caption,
        "category": category,
        "quality": quality,
    }
    graph.upsert_node(node_id, illust_data)

    # Edge: illustration → target node (passage, vision, etc.)
    _remove_edges(graph, from_id=node_id, edge_type="Depicts")
    graph.add_edge("Depicts", node_id, target_id)

    # Edge: illustration → brief (traceability)
    _remove_edges(graph, from_id=node_id, edge_type="from_brief")
    graph.add_edge("from_brief", node_id, brief_id)

    return node_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dress_codex_entries(
    graph: Graph,
    entity_id: str,
    entries: list[dict[str, Any]],
) -> list[str]:
    """Validate codex entries before applying to graph.

    Checks:
    - At least one entry exists
    - Entry with rank=1 exists (base tier, always visible)
    - Codeword IDs in visible_when exist in graph
    - Ranks are unique per entity

    Args:
        graph: Story graph for codeword validation.
        entity_id: Entity these entries belong to.
        entries: List of CodexEntry dicts.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if not entries:
        errors.append(f"Entity {entity_id}: no codex entries provided")
        return errors

    ranks = [e["rank"] for e in entries if "rank" in e]
    if len(ranks) != len(entries):
        errors.append(f"Entity {entity_id}: some codex entries missing required 'rank' field")
    if 1 not in ranks:
        errors.append(f"Entity {entity_id}: missing rank=1 base tier (always visible)")

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    for r, count in rank_counts.items():
        if count > 1:
            errors.append(f"Entity {entity_id}: duplicate rank={r} ({count} entries)")

    # Validate codeword references
    codewords = graph.get_nodes_by_type("codeword")
    codeword_ids = {strip_scope_prefix(cid) for cid in codewords}

    for entry in entries:
        for cw in entry.get("visible_when", []):
            raw_cw = strip_scope_prefix(cw)
            if raw_cw not in codeword_ids:
                errors.append(
                    f"Entity {entity_id}: codex rank={entry.get('rank')} "
                    f"references unknown codeword '{cw}'"
                )

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from a dictionary for cleaner storage."""
    return {k: v for k, v in data.items() if v is not None}


def _remove_edges(graph: Graph, from_id: str, edge_type: str) -> None:
    """Remove all edges of a given type from a node (for idempotent re-runs)."""
    existing = graph.get_edges(from_id=from_id, edge_type=edge_type)
    for edge in existing:
        graph.remove_edge(edge["type"], edge["from"], edge["to"])
