"""Context compaction and enrichment utilities for GROW LLM phases.

Provides building blocks for controlling context size and enriching items
with narrative data from the graph. Each GROW phase composes these
primitives differently — this module handles item-level compaction;
grouping and batching stay in each phase.

See #791 for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


@dataclass
class ContextItem:
    """A single item to include in LLM context.

    Attributes:
        id: Node ID for reference.
        text: Pre-formatted text for this item (one or more lines).
        priority: Higher values are included first when budget is tight.
    """

    id: str
    text: str
    priority: int = 0


@dataclass
class CompactContextConfig:
    """Budget and truncation settings for context compaction.

    Attributes:
        max_chars: Total character budget for compacted output.
        summary_truncate: Max chars for individual summary fields.
        truncation_suffix: Appended when text is truncated.
    """

    max_chars: int = 6000
    summary_truncate: int = 80
    truncation_suffix: str = "..."

    # Approximate chars per token for English text.
    _CHARS_PER_TOKEN: float = 3.5
    # Fraction of context window reserved for injected data.
    # System prompt, template text, and response need the rest.
    _BUDGET_FRACTION: float = 0.05
    _MIN_CHARS: int = 2000
    _MAX_CHARS: int = 50_000

    @classmethod
    def from_context_window(
        cls,
        context_window_tokens: int,
        budget_fraction: float | None = None,
    ) -> CompactContextConfig:
        """Derive config from model context window size.

        Args:
            context_window_tokens: Model's context window in tokens.
            budget_fraction: Fraction of context for injected data.
                Defaults to 0.05 (~5%), which yields ~6K chars for a
                32K-token model.

        Returns:
            Config with max_chars proportional to context window.
        """
        frac = budget_fraction if budget_fraction is not None else cls._BUDGET_FRACTION
        raw = int(context_window_tokens * frac * cls._CHARS_PER_TOKEN)
        max_chars = max(cls._MIN_CHARS, min(raw, cls._MAX_CHARS))
        return cls(max_chars=max_chars)


def truncate_summary(text: str, max_chars: int = 80, suffix: str = "...") -> str:
    """Truncate text to max_chars, preserving word boundaries.

    Args:
        text: Text to truncate.
        max_chars: Maximum character count (including suffix).
        suffix: Appended when truncation occurs.

    Returns:
        Truncated text, or original if already within limit.
    """
    if len(text) <= max_chars:
        return text
    # Leave room for suffix
    limit = max_chars - len(suffix)
    if limit <= 0:
        return suffix[:max_chars]
    # Find last space within limit to preserve word boundary
    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > limit // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def compact_items(
    items: list[ContextItem],
    config: CompactContextConfig | None = None,
) -> str:
    """Render items within character budget.

    Items are sorted by priority (descending), then by insertion order.
    Includes all items that fit within the budget. The last item that
    would exceed the budget is truncated. Remaining items are reported
    with a count note.

    Args:
        items: Context items to render.
        config: Budget configuration. Uses defaults if None.

    Returns:
        Rendered items as a string, or empty string if no items.
    """
    if not items:
        return ""

    cfg = config or CompactContextConfig()

    # Stable sort by priority descending (preserves insertion order for ties)
    sorted_items = sorted(items, key=lambda x: -x.priority)

    output_parts: list[str] = []
    used_chars = 0

    for item in sorted_items:
        item_len = len(item.text)
        if used_chars + item_len <= cfg.max_chars:
            output_parts.append(item.text)
            used_chars += item_len + 1  # +1 for newline separator
        else:
            # Truncate this item to fill remaining budget
            remaining = cfg.max_chars - used_chars
            if remaining > len(cfg.truncation_suffix) + 10:
                truncated = truncate_summary(item.text, remaining, cfg.truncation_suffix)
                output_parts.append(truncated)
            # Count omitted items
            omitted = len(sorted_items) - len(output_parts)
            if omitted > 0:
                output_parts.append(f"({omitted} more items omitted for brevity)")
            break

    return "\n".join(output_parts)


def enrich_beat_line(
    graph: Graph,
    beat_id: str,
    beat_data: dict[str, Any],
    *,
    include_entities: bool = False,
    summary_max: int = 80,
) -> str:
    """Format a single beat as a compact enriched line.

    Base format: ``- {beat_id} [{scene_type}, {fn}]: "{summary}"``
    With entities: appends ``(entities: Name1, Name2)``

    Args:
        graph: Graph for entity lookups.
        beat_id: Scoped beat ID.
        beat_data: Beat node data dict.
        include_entities: Whether to append entity names.
        summary_max: Max chars for the summary field.

    Returns:
        Formatted single-line string.
    """
    summary = truncate_summary(beat_data.get("summary", ""), summary_max)
    scene_type = beat_data.get("scene_type", "")
    narrative_fn = beat_data.get("narrative_function", "")

    tags: list[str] = []
    if scene_type:
        tags.append(scene_type)
    if narrative_fn:
        tags.append(narrative_fn)
    tag_str = ", ".join(tags) if tags else "unclassified"

    line = f'- {beat_id} [{tag_str}]: "{summary}"'

    if include_entities:
        entity_ids = beat_data.get("entities", [])
        if entity_ids:
            names: list[str] = []
            for eid in entity_ids:
                enode = graph.get_node(eid)
                if enode:
                    names.append(enode.get("name") or enode.get("raw_id", eid))
                else:
                    names.append(eid)
            line += f" (entities: {', '.join(names)})"

    return line


def build_narrative_frame(
    graph: Graph,
    dilemma_ids: list[str] | None = None,
    path_ids: list[str] | None = None,
) -> str:
    """Build a compact narrative context frame from graph data.

    Assembles dilemma briefs (question, why_it_matters) and optional
    path briefs (description, path_theme/mood if set) into a compact
    markdown block suitable for injection at the top of LLM prompts.

    Args:
        graph: Graph containing dilemma, path, and entity nodes.
        dilemma_ids: Scoped dilemma IDs to include. If None, includes all.
        path_ids: Scoped path IDs to include. If None, omits path section.

    Returns:
        Rendered markdown string, or empty string if no data.
    """
    lines: list[str] = []

    # Dilemma briefs
    if dilemma_ids is None:
        dilemma_nodes = graph.get_nodes_by_type("dilemma")
        dilemma_ids = sorted(dilemma_nodes.keys())

    for did in dilemma_ids:
        node = graph.get_node(did)
        if not node:
            continue
        question = node.get("question", "")
        stakes = node.get("why_it_matters", "")
        if not question:
            continue
        lines.append(f"**Dilemma ({did}):** {question}")
        if stakes:
            lines.append(f"  Stakes: {truncate_summary(stakes, 150)}")

    # Path briefs (if requested)
    if path_ids:
        for pid in path_ids:
            pnode = graph.get_node(pid)
            if not pnode:
                continue
            desc = pnode.get("description", "")
            theme = pnode.get("path_theme", "")
            mood = pnode.get("path_mood", "")
            parts: list[str] = [f"**Path ({pid}):**"]
            if desc:
                parts.append(truncate_summary(desc, 120))
            if theme:
                parts.append(f"Theme: {theme}")
            if mood:
                parts.append(f"Mood: {mood}")
            if len(parts) > 1:
                lines.append(" ".join(parts))

    if not lines:
        return ""

    return "## Story Context\n" + "\n".join(lines)
