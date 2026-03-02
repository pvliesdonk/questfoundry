"""Export data models and Exporter protocol.

Defines the intermediate representation (ExportContext) that all exporters
consume, plus the Exporter protocol they must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ExportPassage:
    """A player-facing passage with prose."""

    id: str
    prose: str
    is_start: bool = False
    is_ending: bool = False


@dataclass
class ExportChoice:
    """A navigable link between two passages."""

    from_passage: str
    to_passage: str
    label: str
    requires_codewords: list[str] = field(default_factory=list)
    grants: list[str] = field(default_factory=list)
    is_return: bool = False


@dataclass
class ExportEntity:
    """A story entity (character, location, object, faction)."""

    id: str
    entity_type: str
    concept: str
    overlays: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExportCodeword:
    """A state-tracking codeword."""

    id: str
    codeword_type: str
    derived_from: str | None = None


@dataclass
class ExportIllustration:
    """An illustration asset linked to a passage."""

    passage_id: str
    asset_path: str
    caption: str
    category: str


@dataclass
class ExportCodexEntry:
    """A codex encyclopedia entry for an entity."""

    entity_id: str
    title: str
    rank: int
    visible_when: list[str] = field(default_factory=list)
    content: str = ""


@dataclass
class ExportContext:
    """All data needed by exporters, extracted from the story graph.

    This is the intermediate representation between the internal graph
    format and the various export formats (Twee, HTML, JSON).
    """

    title: str
    passages: list[ExportPassage]
    choices: list[ExportChoice]
    entities: list[ExportEntity] = field(default_factory=list)
    codewords: list[ExportCodeword] = field(default_factory=list)
    illustrations: list[ExportIllustration] = field(default_factory=list)
    cover: ExportIllustration | None = None
    codex_entries: list[ExportCodexEntry] = field(default_factory=list)
    art_direction: dict[str, Any] | None = None
    language: str = "en"


class Exporter(Protocol):
    """Protocol for story export format handlers."""

    format_name: str

    def export(self, context: ExportContext, output_dir: Path) -> Path:
        """Export the story to the given output directory.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.

        Returns:
            Path to the main output file.
        """
        ...
