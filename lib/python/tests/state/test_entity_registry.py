"""Tests for entity registry (Layer 6/7 canon workflows)"""

import pytest

from questfoundry.state.entity_registry import Entity, EntityRegistry, EntityType


def test_entity_creation():
    """Test entity creation with validation"""
    entity = Entity(
        name="Queen Elara",
        entity_type=EntityType.CHARACTER,
        role="ruler",
        description="First queen of united kingdom",
        source="world-genesis",
        immutable=True,
    )

    assert entity.name == "Queen Elara"
    assert entity.entity_type == EntityType.CHARACTER
    assert entity.immutable is True
    assert entity.source == "world-genesis"


def test_entity_invalid_name():
    """Test entity with invalid name raises ValueError"""
    with pytest.raises(ValueError, match="Entity name must be 3-80 characters"):
        Entity(
            name="",
            entity_type=EntityType.CHARACTER,
            role="ruler",
            description="Test",
            source="test",
        )


def test_entity_registry_create():
    """Test creating entities in registry"""
    registry = EntityRegistry()

    entity = Entity(
        name="Dragon Council",
        entity_type=EntityType.FACTION,
        role="governing body",
        description="Elder dragons who maintain peace",
        source="world-genesis",
        immutable=True,
    )

    registry.create(entity)
    assert "Dragon Council" in registry
    assert len(registry) == 1


def test_entity_registry_create_duplicate():
    """Test creating duplicate entity raises ValueError"""
    registry = EntityRegistry()

    entity1 = Entity(
        name="Dragon Council",
        entity_type=EntityType.FACTION,
        role="governing body",
        description="Elder dragons",
        source="world-genesis",
    )
    registry.create(entity1)

    # Try to create duplicate
    entity2 = Entity(
        name="Dragon Council",
        entity_type=EntityType.FACTION,
        role="different",
        description="Different description",
        source="project-local",
    )

    with pytest.raises(ValueError, match="Entity.*already exists"):
        registry.create(entity2)


def test_entity_registry_get_by_name():
    """Test querying entities by name"""
    registry = EntityRegistry()

    entity = Entity(
        name="Ancient City",
        entity_type=EntityType.PLACE,
        role="capital",
        description="Capital of the kingdom",
        source="world-genesis",
    )
    registry.create(entity)

    found = registry.get_by_name("Ancient City")
    assert found is not None
    assert found.entity_type == EntityType.PLACE

    not_found = registry.get_by_name("Unknown City")
    assert not_found is None


def test_entity_registry_get_by_type():
    """Test querying entities by type"""
    registry = EntityRegistry()

    # Create characters
    registry.create(
        Entity(
            name="Hero",
            entity_type=EntityType.CHARACTER,
            role="protagonist",
            description="Main character",
            source="project-local",
        )
    )
    registry.create(
        Entity(
            name="Villain",
            entity_type=EntityType.CHARACTER,
            role="antagonist",
            description="Main villain",
            source="project-local",
        )
    )

    # Create place
    registry.create(
        Entity(
            name="Castle",
            entity_type=EntityType.PLACE,
            role="fortress",
            description="Royal castle",
            source="project-local",
        )
    )

    characters = registry.get_by_type(EntityType.CHARACTER)
    assert len(characters) == 2
    assert all(e.entity_type == EntityType.CHARACTER for e in characters)

    places = registry.get_by_type(EntityType.PLACE)
    assert len(places) == 1
    assert places[0].name == "Castle"


def test_entity_registry_update():
    """Test updating entity"""
    registry = EntityRegistry()

    entity = Entity(
        name="Council",
        entity_type=EntityType.FACTION,
        role="governing",
        description="Original description",
        source="world-genesis",
        immutable=False,
    )
    registry.create(entity)

    # Update entity
    entity.description = "Updated description"
    registry.update(entity)

    updated = registry.get_by_name("Council")
    assert updated.description == "Updated description"
    assert updated.role == "governing"  # Unchanged


def test_entity_registry_update_immutable_entity():
    """Test updating immutable entity raises ValueError"""
    registry = EntityRegistry()

    entity = Entity(
        name="Ancient Dragon",
        entity_type=EntityType.CHARACTER,
        role="founder",
        description="First dragon",
        source="world-genesis",
        immutable=True,
    )
    registry.create(entity)

    # Try to update immutable entity
    entity.description = "New description"
    with pytest.raises(ValueError, match="Cannot update immutable entity"):
        registry.update(entity)


def test_entity_registry_delete():
    """Test deleting entity"""
    registry = EntityRegistry()

    entity = Entity(
        name="Temp Entity",
        entity_type=EntityType.ITEM,
        role="temporary",
        description="Test entity",
        source="test",
    )
    registry.create(entity)
    assert len(registry) == 1

    registry.delete("Temp Entity")
    assert len(registry) == 0


def test_entity_registry_delete_immutable():
    """Test deleting immutable entity raises ValueError"""
    registry = EntityRegistry()

    entity = Entity(
        name="Sacred Artifact",
        entity_type=EntityType.ITEM,
        role="relic",
        description="Ancient relic",
        source="world-genesis",
        immutable=True,
    )
    registry.create(entity)

    with pytest.raises(ValueError, match="Cannot delete immutable entity"):
        registry.delete("Sacred Artifact")


def test_entity_registry_merge():
    """Test merging entities with deduplication"""
    registry = EntityRegistry()

    # Create base entity (mutable)
    registry.create(
        Entity(
            name="Dragon",
            entity_type=EntityType.CHARACTER,
            role="guardian",
            description="Local dragon",
            source="project-local",
            immutable=False,
        )
    )

    # Merge with immutable version
    imported = [
        Entity(
            name="Dragon",
            entity_type=EntityType.CHARACTER,
            role="ancient guardian",
            description="Canonical dragon from source material",
            source="canon-import",
            immutable=True,
        )
    ]

    report = registry.merge(imported, deduplicate=True)

    # Immutable should replace mutable
    dragon = registry.get_by_name("Dragon")
    assert dragon.immutable is True
    assert dragon.source == "canon-import"
    assert dragon.role == "ancient guardian"

    # Check report
    assert report["added"] == 1  # Immutable replaced mutable
    assert report["skipped"] == 0
    assert len(report["conflicts"]) == 0


def test_entity_registry_merge_new_entity():
    """Test merging new entities"""
    registry = EntityRegistry()

    # Merge new entities
    imported = [
        Entity(
            name="New Character",
            entity_type=EntityType.CHARACTER,
            role="hero",
            description="New hero",
            source="canon-import",
            immutable=False,
        )
    ]

    report = registry.merge(imported, deduplicate=True)

    assert report["added"] == 1
    assert report["skipped"] == 0
    assert len(report["conflicts"]) == 0


def test_entity_registry_merge_conflict():
    """Test merge with immutable conflict"""
    registry = EntityRegistry()

    # Create immutable entity
    registry.create(
        Entity(
            name="Dragon",
            entity_type=EntityType.CHARACTER,
            role="ancient",
            description="Original immutable",
            source="world-genesis",
            immutable=True,
        )
    )

    # Try to merge another immutable with same name
    imported = [
        Entity(
            name="Dragon",
            entity_type=EntityType.CHARACTER,
            role="different",
            description="Different immutable",
            source="canon-import",
            immutable=True,
        )
    ]

    report = registry.merge(imported, deduplicate=True)

    # Should have conflict
    assert report["added"] == 0
    assert report["skipped"] == 1
    assert len(report["conflicts"]) == 1
    assert "Dragon" in report["conflicts"][0]["name"]


def test_entity_registry_count_by_type():
    """Test counting entities by type"""
    registry = EntityRegistry()

    registry.create(
        Entity(
            name="Hero1",
            entity_type=EntityType.CHARACTER,
            role="hero",
            description="Test",
            source="test",
        )
    )
    registry.create(
        Entity(
            name="Hero2",
            entity_type=EntityType.CHARACTER,
            role="hero",
            description="Test",
            source="test",
        )
    )
    registry.create(
        Entity(
            name="City",
            entity_type=EntityType.PLACE,
            role="settlement",
            description="Test",
            source="test",
        )
    )
    registry.create(
        Entity(
            name="Guild",
            entity_type=EntityType.FACTION,
            role="organization",
            description="Test",
            source="test",
        )
    )
    registry.create(
        Entity(
            name="Sword",
            entity_type=EntityType.ITEM,
            role="weapon",
            description="Test",
            source="test",
        )
    )

    counts = registry.count_by_type()
    assert counts["characters"] == 2
    assert counts["places"] == 1
    assert counts["factions"] == 1
    assert counts["items"] == 1
    assert counts["total"] == 5


def test_entity_registry_to_dict():
    """Test serializing registry to dictionary"""
    registry = EntityRegistry()

    registry.create(
        Entity(
            name="Test Entity",
            entity_type=EntityType.CHARACTER,
            role="test",
            description="Test entity",
            source="test",
            immutable=True,
        )
    )

    data = registry.to_dict()
    assert "characters" in data
    assert len(data["characters"]) == 1
    assert data["characters"][0]["name"] == "Test Entity"
    assert data["characters"][0]["immutable"] is True
