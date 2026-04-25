"""DRESS Stage Output Contract validator.

Implements the DRESS §Stage Output Contract checks (from
``docs/design/procedures/dress.md``):

1. Exactly one ArtDirection node exists.
2. Every Entity with ≥1 ``appears`` edge has an EntityVisual with
   non-empty ``reference_prompt_fragment`` (R-1.3, R-1.4 — also
   enforced at Phase 0 exit by ``validate_entity_visual_coverage``).
3. Every Passage has an IllustrationBrief with a ``targets`` edge.
5. Every Entity has ≥1 CodexEntry with ``HasEntry`` edge.
6. CodexEntry rank 1 is always visible (``visible_when == []``).

Output-4 (brief field completeness + diegetic captions) is enforced
inline during Phase 1 by ``apply_dress_brief()`` validation against
the Pydantic ``IllustrationBrief`` model — re-checking it here would
duplicate the per-brief checks that already raised at creation time.
The numbering gap above is intentional, not an oversight.

DRESS may be skipped entirely (Output-10), so an empty graph (no
art_direction node) is treated as a *skipped* DRESS, not a contract
violation. The validator returns no errors for skipped DRESS.

Called at DRESS exit and at SHIP entry. Read-only — never mutates
the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.context import strip_scope_prefix

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def validate_dress_output(graph: Graph) -> list[str]:
    """Return contract violations after DRESS completion. Empty = compliant.

    DRESS may be skipped entirely (no art_direction node) — in that
    case the validator returns ``[]`` so the SHIP entry contract still
    passes for stories that opt out of illustrations.
    """
    art_direction_nodes = graph.get_nodes_by_type("art_direction")
    if not art_direction_nodes:
        # DRESS was skipped — Output-10 explicitly permits this.
        return []

    errors: list[str] = []

    # Output-1: exactly one ArtDirection node.
    if len(art_direction_nodes) > 1:
        errors.append(
            f"Output-1: exactly one ArtDirection node expected, found "
            f"{len(art_direction_nodes)}: {sorted(art_direction_nodes)}"
        )

    # Output-2: every appearing entity has an EntityVisual with a
    # non-empty reference_prompt_fragment. This mirrors
    # validate_entity_visual_coverage() called at Phase 0 exit.
    appears_edges = graph.get_edges(edge_type="appears")
    appearing_entity_ids = {edge["from"] for edge in appears_edges}
    for entity_id in sorted(appearing_entity_ids):
        raw_id = strip_scope_prefix(entity_id)
        ev_id = f"entity_visual::{raw_id}"
        ev = graph.get_node(ev_id)
        if ev is None:
            errors.append(
                f"Output-2: entity {entity_id!r} has appears edges but no "
                f"EntityVisual ({ev_id!r} missing)"
            )
            continue
        fragment = (ev.get("reference_prompt_fragment") or "").strip()
        if not fragment:
            errors.append(
                f"Output-2: entity {entity_id!r} EntityVisual has empty reference_prompt_fragment"
            )

    # Output-3: every Passage has an IllustrationBrief with a targets edge.
    passages = graph.get_nodes_by_type("passage")
    targets_edges = graph.get_edges(edge_type="targets")
    brief_targets = {edge["to"] for edge in targets_edges}
    missing_briefs = sorted(pid for pid in passages if pid not in brief_targets)
    if missing_briefs:
        errors.append(
            f"Output-3: {len(missing_briefs)} Passage(s) without an "
            f"IllustrationBrief: "
            f"{', '.join(repr(p) for p in missing_briefs[:5])}"
            f"{' …' if len(missing_briefs) > 5 else ''}"
        )

    # Output-5: every Entity has ≥1 CodexEntry via HasEntry.
    entities = graph.get_nodes_by_type("entity")
    has_entry_edges = graph.get_edges(edge_type="HasEntry")
    entities_with_codex = {edge["to"] for edge in has_entry_edges}
    missing_codex = sorted(eid for eid in entities if eid not in entities_with_codex)
    if missing_codex:
        errors.append(
            f"Output-5: {len(missing_codex)} Entity(ies) without a CodexEntry: "
            f"{', '.join(repr(e) for e in missing_codex[:5])}"
            f"{' …' if len(missing_codex) > 5 else ''}"
        )

    # Output-6: rank-1 codex entries must be unconditional (visible_when == []).
    codex_entries = graph.get_nodes_by_type("codex_entry")
    rank1_with_gates = sorted(
        cid
        for cid, data in codex_entries.items()
        if data.get("rank") == 1 and data.get("visible_when")
    )
    if rank1_with_gates:
        errors.append(
            f"Output-6: {len(rank1_with_gates)} rank-1 CodexEntry(ies) with "
            f"non-empty visible_when (must be unconditional): "
            f"{', '.join(repr(c) for c in rank1_with_gates[:5])}"
            f"{' …' if len(rank1_with_gates) > 5 else ''}"
        )

    return errors
