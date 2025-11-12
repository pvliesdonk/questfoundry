"""Type stubs for entity registry (Layer 6/7 canon workflows)"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

class EntityType(Enum):
    """Types of entities in the registry."""

    CHARACTER: str
    PLACE: str
    FACTION: str
    ITEM: str

@dataclass
class Entity:
    """A canonical entity in the registry."""

    name: str
    entity_type: EntityType
    role: str
    description: str
    source: str
    immutable: bool = ...
    metadata: dict[str, Any] = ...

    def __post_init__(self) -> None: ...

class EntityRegistry:
    """Registry for managing canonical entities across canon workflows."""

    def __init__(self) -> None: ...
    def create(self, entity: Entity) -> Entity: ...
    def get_by_name(self, name: str) -> Entity | None: ...
    def get_by_type(self, entity_type: EntityType) -> list[Entity]: ...
    def get_by_immutability(self, immutable: bool) -> list[Entity]: ...
    def update(
        self,
        name: str,
        entity_type: EntityType,
        role: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity: ...
    def delete(self, name: str, entity_type: EntityType) -> bool: ...
    def merge(
        self,
        entities: list[Entity],
        deduplicate: bool = True,
    ) -> dict[str, Any]: ...
    def count(self) -> int: ...
    def count_by_type(self) -> dict[str, int]: ...
    def to_dict(self) -> dict[str, Any]: ...
