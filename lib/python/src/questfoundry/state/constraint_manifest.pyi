"""Type stubs for constraint manifest generator (Layer 6/7 canon workflows)"""

from dataclasses import dataclass
from typing import Any

from .entity_registry import EntityRegistry
from .timeline import TimelineManager

@dataclass
class ConstraintManifest:
    """Manifest of creative constraints derived from canon."""

    invariants: list[str]
    mutables: list[str]
    timeline_constraints: list[str]
    entity_constraints: list[str]
    boundaries: list[str]
    guidance: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: ...
    def to_markdown(self) -> str: ...

class ConstraintManifestGenerator:
    """Generator for constraint manifests."""

    def __init__(self) -> None: ...
    def generate(
        self,
        invariant_canon: list[dict[str, Any]] | None = None,
        mutable_canon: list[dict[str, Any]] | None = None,
        entity_registry: EntityRegistry | None = None,
        timeline: TimelineManager | None = None,
        source: str = "canon-import",
    ) -> ConstraintManifest: ...
