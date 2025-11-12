"""
Entity registry for canon workflows.

The entity registry tracks canonical entities (characters, places, factions, items)
across canon transfer packages and world genesis manifests. It provides CRUD
operations, reference integrity validation, and deduplication during imports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(Enum):
    """Types of entities in the registry."""

    CHARACTER = "character"
    PLACE = "place"
    FACTION = "faction"
    ITEM = "item"


@dataclass
class Entity:
    """
    A canonical entity in the registry.

    Entities represent named elements in the story world: characters, locations,
    factions, and items. Each entity is tracked with its source attribution to
    maintain provenance across canon transfers.

    Attributes:
        name: Entity name (3-80 characters)
        entity_type: Type of entity (character, place, faction, item)
        role: Role or function (e.g., "protagonist", "capital city",
            "antagonist faction")
        description: Brief description of the entity
        source: Source project or package (e.g., "dragon-quest-1",
            "world-genesis")
        immutable: Whether this entity can be modified (from invariant canon)
        metadata: Additional entity-specific data
    """

    name: str
    entity_type: EntityType
    role: str
    description: str
    source: str
    immutable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entity fields."""
        if not 3 <= len(self.name) <= 80:
            raise ValueError(
                f"Entity name must be 3-80 characters, got {len(self.name)}"
            )
        if not self.description:
            raise ValueError("Entity description cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "role": self.role,
            "description": self.description,
            "source": self.source,
            "immutable": self.immutable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            role=data["role"],
            description=data["description"],
            source=data["source"],
            immutable=data.get("immutable", False),
            metadata=data.get("metadata", {}),
        )


class EntityRegistry:
    """
    Registry for managing canonical entities across canon workflows.

    The entity registry provides CRUD operations for entities, maintains
    reference integrity, and handles deduplication during canon imports.

    Features:
        - Fast lookup by name and type
        - Deduplication on import
        - Immutability enforcement for invariant canon
        - Reference integrity validation
        - Source attribution tracking

    Example:
        >>> registry = EntityRegistry()
        >>> # Add character from world genesis
        >>> queen = Entity(
        ...     name="Queen Elara",
        ...     entity_type=EntityType.CHARACTER,
        ...     role="ruler",
        ...     description="First queen of united kingdom",
        ...     source="world-genesis",
        ...     immutable=True
        ... )
        >>> registry.create(queen)
        >>> # Query entities
        >>> characters = registry.get_by_type(EntityType.CHARACTER)
        >>> queen = registry.get_by_name("Queen Elara")
        >>> # Update mutable entity
        >>> place = registry.get_by_name("Market District")
        >>> if not place.immutable:
        ...     place.description = "Bustling trade center"
        ...     registry.update(place)
    """

    def __init__(self) -> None:
        """Initialize empty entity registry."""
        self._entities: dict[str, Entity] = {}  # name -> entity
        self._type_index: dict[EntityType, set[str]] = {
            EntityType.CHARACTER: set(),
            EntityType.PLACE: set(),
            EntityType.FACTION: set(),
            EntityType.ITEM: set(),
        }

    def create(self, entity: Entity) -> None:
        """
        Add entity to registry.

        Args:
            entity: Entity to add

        Raises:
            ValueError: If entity name already exists
        """
        if entity.name in self._entities:
            raise ValueError(f"Entity '{entity.name}' already exists")

        self._entities[entity.name] = entity
        self._type_index[entity.entity_type].add(entity.name)

    def get_by_name(self, name: str) -> Entity | None:
        """
        Get entity by name.

        Args:
            name: Entity name

        Returns:
            Entity if found, None otherwise
        """
        return self._entities.get(name)

    def get_by_type(self, entity_type: EntityType) -> list[Entity]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Type to filter by

        Returns:
            List of entities of the specified type
        """
        names = self._type_index[entity_type]
        return [self._entities[name] for name in names]

    def get_by_source(self, source: str) -> list[Entity]:
        """
        Get all entities from a specific source.

        Args:
            source: Source project or package

        Returns:
            List of entities from the specified source
        """
        return [e for e in self._entities.values() if e.source == source]

    def update(self, entity: Entity) -> None:
        """
        Update existing entity.

        Args:
            entity: Entity with updated fields

        Raises:
            ValueError: If entity doesn't exist or is immutable
        """
        if entity.name not in self._entities:
            raise ValueError(f"Entity '{entity.name}' does not exist")

        existing = self._entities[entity.name]
        if existing.immutable:
            raise ValueError(
                f"Cannot update immutable entity '{entity.name}' from {existing.source}"
            )

        self._entities[entity.name] = entity

    def delete(self, name: str) -> None:
        """
        Delete entity from registry.

        Args:
            name: Entity name to delete

        Raises:
            ValueError: If entity doesn't exist or is immutable
        """
        if name not in self._entities:
            raise ValueError(f"Entity '{name}' does not exist")

        entity = self._entities[name]
        if entity.immutable:
            raise ValueError(
                f"Cannot delete immutable entity '{name}' from {entity.source}"
            )

        del self._entities[name]
        self._type_index[entity.entity_type].discard(name)

    def merge(self, entities: list[Entity], deduplicate: bool = True) -> dict[str, Any]:
        """
        Merge entities from canon import.

        Handles deduplication and conflict resolution during canon imports.
        Immutable entities take precedence over mutable ones.

        Args:
            entities: List of entities to merge
            deduplicate: Whether to deduplicate by name

        Returns:
            Merge report with added, skipped, and conflict counts
        """
        added = 0
        skipped = 0
        conflicts: list[dict[str, str]] = []

        for entity in entities:
            if entity.name not in self._entities:
                # New entity - add it
                self.create(entity)
                added += 1
            elif deduplicate:
                existing = self._entities[entity.name]

                # Conflict: same name, different details
                if existing.immutable and not entity.immutable:
                    # Existing is immutable, skip incoming mutable
                    skipped += 1
                elif not existing.immutable and entity.immutable:
                    # Incoming is immutable, replace existing mutable
                    self._entities[entity.name] = entity
                    added += 1
                elif existing.immutable and entity.immutable:
                    # Both immutable - conflict!
                    conflicts.append(
                        {
                            "name": entity.name,
                            "existing_source": existing.source,
                            "incoming_source": entity.source,
                            "reason": "Both immutable with same name",
                        }
                    )
                    skipped += 1
                else:
                    # Both mutable - use newer (incoming)
                    self._entities[entity.name] = entity
                    added += 1
            else:
                # No deduplication - this is a conflict
                conflicts.append(
                    {
                        "name": entity.name,
                        "existing_source": self._entities[entity.name].source,
                        "incoming_source": entity.source,
                        "reason": "Duplicate name without deduplication",
                    }
                )
                skipped += 1

        return {
            "added": added,
            "skipped": skipped,
            "conflicts": conflicts,
        }

    def validate_references(self, references: list[str]) -> list[str]:
        """
        Validate that entity references exist.

        Args:
            references: List of entity names to validate

        Returns:
            List of missing entity names
        """
        missing = []
        for name in references:
            if name not in self._entities:
                missing.append(name)
        return missing

    def to_dict(self) -> dict[str, Any]:
        """
        Export registry to dictionary.

        Returns:
            Dictionary with entities organized by type
        """
        return {
            "characters": [e.to_dict() for e in self.get_by_type(EntityType.CHARACTER)],
            "places": [e.to_dict() for e in self.get_by_type(EntityType.PLACE)],
            "factions": [e.to_dict() for e in self.get_by_type(EntityType.FACTION)],
            "items": [e.to_dict() for e in self.get_by_type(EntityType.ITEM)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityRegistry":
        """
        Create registry from dictionary.

        Args:
            data: Dictionary with entities by type

        Returns:
            New EntityRegistry instance
        """
        registry = cls()

        for character_data in data.get("characters", []):
            entity = Entity.from_dict(character_data)
            registry.create(entity)

        for place_data in data.get("places", []):
            entity = Entity.from_dict(place_data)
            registry.create(entity)

        for faction_data in data.get("factions", []):
            entity = Entity.from_dict(faction_data)
            registry.create(entity)

        for item_data in data.get("items", []):
            entity = Entity.from_dict(item_data)
            registry.create(entity)

        return registry

    def count_by_type(self) -> dict[str, int]:
        """
        Get entity counts by type.

        Returns:
            Dictionary mapping type names to counts
        """
        return {
            "characters": len(self._type_index[EntityType.CHARACTER]),
            "places": len(self._type_index[EntityType.PLACE]),
            "factions": len(self._type_index[EntityType.FACTION]),
            "items": len(self._type_index[EntityType.ITEM]),
            "total": len(self._entities),
        }

    def __len__(self) -> int:
        """Return total entity count."""
        return len(self._entities)

    def __contains__(self, name: str) -> bool:
        """Check if entity exists."""
        return name in self._entities
