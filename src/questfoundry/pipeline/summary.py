from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def build_stage_summary(stage_name: str, artifact_data: dict[str, Any]) -> list[str]:
    """Build a concise, human-readable stage summary.

    Args:
        stage_name: Stage identifier (dream, brainstorm, seed, grow, fill, ship, dress).
        artifact_data: Stage artifact data produced by the model.

    Returns:
        List of summary lines suitable for CLI display.
    """
    if not isinstance(artifact_data, dict):
        return []

    if stage_name == "dream":
        lines: list[str] = []
        genre = _as_str(artifact_data.get("genre"))
        subgenre = _as_str(artifact_data.get("subgenre"))
        if genre:
            if subgenre:
                lines.append(f"Genre: {genre} ({subgenre})")
            else:
                lines.append(f"Genre: {genre}")
        audience = _as_str(artifact_data.get("audience"))
        if audience:
            lines.append(f"Audience: {audience}")
        tone = _format_list(artifact_data.get("tone"))
        if tone:
            lines.append(f"Tone: {tone}")
        themes = _format_list(artifact_data.get("themes"))
        if themes:
            lines.append(f"Themes: {themes}")
        scope = artifact_data.get("scope")
        if isinstance(scope, dict):
            story_size = _as_str(scope.get("story_size"))
            if story_size:
                lines.append(f"Scope: {story_size}")
        return _limit_lines(lines)

    if stage_name == "brainstorm":
        lines = []
        entities = _list_of_dicts(artifact_data.get("entities"))
        dilemmas = _list_of_dicts(artifact_data.get("dilemmas"))

        if entities:
            categories: list[str] = []
            for entity in entities:
                category = _as_str(entity.get("entity_category"))
                if category:
                    categories.append(category)
            category_counts = Counter(categories)
            if category_counts:
                lines.append(f"Entities: {len(entities)} ({_format_counts(category_counts)})")
            else:
                lines.append(f"Entities: {len(entities)}")

            protagonists = [
                name
                for name in (
                    _as_str(entity.get("name")) or _as_str(entity.get("concept"))
                    for entity in entities
                    if entity.get("is_protagonist")
                )
                if name
            ]
            if protagonists:
                lines.append(f"Protagonist: {_format_truncated(protagonists)}")

        if dilemmas:
            lines.append(f"Dilemmas: {len(dilemmas)}")

        return _limit_lines(lines)

    if stage_name == "seed":
        lines = _count_lines(
            artifact_data,
            (
                ("entities", "Entities"),
                ("dilemmas", "Dilemmas"),
                ("paths", "Paths"),
                ("consequences", "Consequences"),
                ("initial_beats", "Initial beats"),
                ("interaction_constraints", "Constraints"),
            ),
        )
        return _limit_lines(lines)

    if stage_name == "fill":
        lines = []
        result = artifact_data.get("result")
        if isinstance(result, dict):
            lines.extend(
                _count_simple_fields(
                    result,
                    (
                        ("passages_filled", "Passages filled"),
                        ("passages_flagged", "Passages flagged"),
                        ("entity_updates_applied", "Entity updates"),
                        ("review_cycles", "Review cycles"),
                        ("phases_completed", "Phases completed"),
                    ),
                )
            )
        if not lines:
            lines = _count_lines(
                artifact_data,
                (
                    ("passages", "Passages"),
                    ("prose", "Prose entries"),
                ),
            )
        return _limit_lines(lines)

    lines = _count_lines(
        artifact_data,
        (
            ("paths", "Paths"),
            ("passages", "Passages"),
            ("choices", "Choices"),
            ("arcs", "Arcs"),
            ("codewords", "Codewords"),
            ("beats", "Beats"),
            ("nodes", "Nodes"),
        ),
    )
    return _limit_lines(lines)


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _format_list(value: Any, limit: int = 4) -> str | None:
    if not isinstance(value, list):
        return None
    items = [item for item in (str(item).strip() for item in value) if item]
    if not items:
        return None
    return _format_truncated(items, limit=limit)


def _format_truncated(items: Iterable[str], limit: int = 3) -> str:
    items_list = list(items)
    if len(items_list) <= limit:
        return ", ".join(items_list)
    return ", ".join(items_list[:limit]) + "..."


def _format_counts(counter: Counter[str]) -> str:
    return ", ".join(f"{label} {count}" for label, count in counter.most_common())


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _count_lines(
    artifact_data: dict[str, Any],
    mapping: Iterable[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []
    for key, label in mapping:
        count = _list_count(artifact_data.get(key))
        if count:
            lines.append(f"{label}: {count}")
    return lines


def _count_simple_fields(
    data: dict[str, Any],
    mapping: Iterable[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []
    for key, label in mapping:
        value = data.get(key)
        if isinstance(value, int):
            lines.append(f"{label}: {value}")
    return lines


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _limit_lines(lines: list[str], limit: int = 5) -> list[str]:
    return lines[:limit]
