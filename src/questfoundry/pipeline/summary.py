"""Build per-stage summary lines from stage artifact data.

After a stage completes successfully, the orchestrator passes its artifact
(the dict serialized from the stage's Pydantic output) through
:func:`build_stage_summary` to produce a short bulleted list shown in the
CLI completion block. The intent is a 2-5 line ballpark check — "BRAINSTORM
produced 3 dilemmas + 8 entities" — without opening the artifact files.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    StageSummaryBuilder = Callable[[dict[str, Any]], list[str]]


def build_stage_summary(stage_name: str, artifact_data: Any) -> list[str]:
    """Build a short list of summary lines for a stage's artifact.

    Returns an empty list for stages without a builder, or for artifact
    payloads that don't match the expected shape — the CLI suppresses the
    Summary block when the list is empty.
    """
    if not isinstance(artifact_data, dict):
        return []

    builder = _BUILDERS.get(stage_name)
    if builder is None:
        return []
    return builder(artifact_data)


def _summarize_dream(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    genre = _as_str(data.get("genre"))
    subgenre = _as_str(data.get("subgenre"))
    if genre:
        lines.append(f"Genre: {genre} ({subgenre})" if subgenre else f"Genre: {genre}")
    audience = _as_str(data.get("audience"))
    if audience:
        lines.append(f"Audience: {audience}")
    tone = _format_str_list(data.get("tone"))
    if tone:
        lines.append(f"Tone: {tone}")
    themes = _format_str_list(data.get("themes"))
    if themes:
        lines.append(f"Themes: {themes}")
    scope = data.get("scope")
    if isinstance(scope, dict):
        story_size = _as_str(scope.get("story_size"))
        if story_size:
            lines.append(f"Scope: {story_size}")
    return lines


def _summarize_brainstorm(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    entities = _list_of_dicts(data.get("entities"))
    dilemmas = _list_of_dicts(data.get("dilemmas"))

    if entities:
        category_counts = Counter(
            cat for cat in (_as_str(e.get("entity_category")) for e in entities) if cat is not None
        )
        if category_counts:
            lines.append(f"Entities: {len(entities)} ({_format_counts(category_counts)})")
        else:
            lines.append(f"Entities: {len(entities)}")

    if dilemmas:
        lines.append(f"Dilemmas: {len(dilemmas)}")

    return lines


def _summarize_seed(data: dict[str, Any]) -> list[str]:
    return _count_list_fields(
        data,
        (
            ("entities", "Entities"),
            ("dilemmas", "Dilemmas"),
            ("paths", "Paths"),
            ("consequences", "Consequences"),
            ("initial_beats", "Initial beats"),
        ),
    )


def _summarize_grow(data: dict[str, Any]) -> list[str]:
    # GROW emits GrowResult after epic #1057 — beats live in graph; this
    # artifact records derived counts only.
    return _count_int_fields(
        data,
        (
            ("arc_count", "Arcs"),
            ("state_flag_count", "State flags"),
            ("overlay_count", "Overlays"),
        ),
    )


def _summarize_fill(data: dict[str, Any]) -> list[str]:
    return _count_int_fields(
        data,
        (
            ("passages_filled", "Passages filled"),
            ("passages_flagged", "Passages flagged"),
            ("entity_updates_applied", "Entity updates"),
            ("review_cycles", "Review cycles"),
        ),
    )


def _summarize_dress(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if data.get("art_direction_created"):
        lines.append("Art direction: created")
    lines.extend(
        _count_int_fields(
            data,
            (
                ("entity_visuals_created", "Entity visuals"),
                ("codex_entries_created", "Codex entries"),
                ("briefs_created", "Illustration briefs"),
                ("illustrations_generated", "Illustrations generated"),
                ("illustrations_failed", "Illustrations failed"),
            ),
        )
    )
    return lines


def _summarize_polish(data: dict[str, Any]) -> list[str]:
    return _count_int_fields(
        data,
        (
            ("passage_count", "Passages"),
            ("choice_count", "Choices"),
            ("variant_count", "Variants"),
            ("residue_count", "Residue beats"),
            ("sidetrack_count", "Sidetrack beats"),
            ("false_branch_count", "False branches"),
        ),
    )


_BUILDERS: dict[str, StageSummaryBuilder] = {
    "dream": _summarize_dream,
    "brainstorm": _summarize_brainstorm,
    "seed": _summarize_seed,
    "grow": _summarize_grow,
    "fill": _summarize_fill,
    "dress": _summarize_dress,
    "polish": _summarize_polish,
}


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _format_str_list(value: Any, limit: int = 4) -> str | None:
    if not isinstance(value, list):
        return None
    items = [s for s in (_as_str(v) for v in value) if s is not None]
    if not items:
        return None
    return _truncate_items(items, limit=limit)


def _truncate_items(items: Iterable[str], limit: int) -> str:
    seq = list(items)
    if len(seq) <= limit:
        return ", ".join(seq)
    return ", ".join(seq[:limit]) + ", ..."


def _format_counts(counter: Counter[str]) -> str:
    return ", ".join(f"{count} {label}" for label, count in counter.most_common())


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _count_list_fields(
    data: dict[str, Any],
    mapping: Iterable[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []
    for key, label in mapping:
        value = data.get(key)
        if isinstance(value, list) and value:
            lines.append(f"{label}: {len(value)}")
    return lines


def _count_int_fields(
    data: dict[str, Any],
    mapping: Iterable[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []
    for key, label in mapping:
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            lines.append(f"{label}: {value}")
    return lines
