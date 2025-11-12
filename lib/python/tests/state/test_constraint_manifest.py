"""Tests for constraint manifest generator (Layer 6/7 canon workflows)"""

from questfoundry.state.constraint_manifest import (
    ConstraintManifest,
    ConstraintManifestGenerator,
)
from questfoundry.state.entity_registry import Entity, EntityRegistry, EntityType
from questfoundry.state.timeline import TimelineAnchor, TimelineManager


def test_constraint_manifest_creation():
    """Test constraint manifest creation"""
    manifest = ConstraintManifest(
        invariants=["Magic cannot resurrect the dead"],
        mutables=["Regional magic traditions vary"],
        timeline_constraints=["All events after T0 (year 0)"],
        entity_constraints=["Cannot modify canonical dragon lore"],
    )

    assert len(manifest.invariants) == 1
    assert len(manifest.mutables) == 1
    assert "resurrect" in manifest.invariants[0]


def test_constraint_manifest_to_dict():
    """Test serializing manifest to dictionary"""
    manifest = ConstraintManifest(
        invariants=["Test invariant"],
        mutables=["Test mutable"],
        boundaries=["Respect canon rules"],
        guidance=["Feel free to elaborate"],
    )

    data = manifest.to_dict()
    assert "invariants" in data
    assert "mutables" in data
    assert "boundaries" in data
    assert "guidance" in data
    assert data["invariants"][0] == "Test invariant"


def test_constraint_manifest_to_markdown():
    """Test generating markdown documentation"""
    manifest = ConstraintManifest(
        invariants=["Dragons are extinct", "Magic has limits"],
        mutables=["Regional customs vary", "New spells can be invented"],
        timeline_constraints=["Events must occur after year 0"],
        entity_constraints=["Cannot modify Queen Elara's role"],
        boundaries=["Respect 2 invariant rules"],
        guidance=["Extend mutable canon freely"],
    )

    markdown = manifest.to_markdown()

    assert "# Creative Constraints" in markdown
    assert "## You CANNOT" in markdown
    assert "## You CAN" in markdown
    assert "❌" in markdown  # Invariant marker
    assert "✅" in markdown  # Mutable marker
    assert "Dragons are extinct" in markdown


def test_constraint_generator_empty():
    """Test generator with no canon"""
    generator = ConstraintManifestGenerator()

    manifest = generator.generate(
        invariant_canon=[],
        mutable_canon=[],
        source="test",
    )

    assert len(manifest.invariants) == 0
    assert len(manifest.mutables) == 0
    assert "No strict constraints" in manifest.boundaries[0]


def test_constraint_generator_invariant_canon():
    """Test extracting invariants from canon"""
    generator = ConstraintManifestGenerator()

    invariant_canon = [
        {"facts": ["Dragons sleep for decades", "Magic has limits"], "immutable": True}
    ]

    manifest = generator.generate(
        invariant_canon=invariant_canon,
        source="test",
    )

    assert len(manifest.invariants) == 2
    assert any("Dragons" in inv for inv in manifest.invariants)


def test_constraint_generator_mutable_canon():
    """Test extracting mutables from canon"""
    generator = ConstraintManifestGenerator()

    mutable_canon = [
        {"facts": [{"statement": "Regional magic traditions", "immutable": False}]}
    ]

    manifest = generator.generate(
        mutable_canon=mutable_canon,
        source="test",
    )

    assert len(manifest.mutables) > 0


def test_constraint_generator_with_entity_registry():
    """Test generating constraints from entity registry"""
    generator = ConstraintManifestGenerator()
    registry = EntityRegistry()

    # Add immutable entities
    registry.create(
        Entity(
            name="Queen Elara",
            entity_type=EntityType.CHARACTER,
            role="ruler",
            description="First queen",
            source="world-genesis",
            immutable=True,
        )
    )
    registry.create(
        Entity(
            name="Ancient City",
            entity_type=EntityType.PLACE,
            role="capital",
            description="Capital city",
            source="world-genesis",
            immutable=True,
        )
    )

    manifest = generator.generate(
        entity_registry=registry,
        source="test",
    )

    assert len(manifest.entity_constraints) > 0
    # Should mention canonical characters or places


def test_constraint_generator_with_timeline():
    """Test generating constraints from timeline"""
    generator = ConstraintManifestGenerator()
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Kingdom founding",
            year=0,
            source="world-genesis",
            immutable=True,
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=250,
            source="world-genesis",
            immutable=True,
        )
    )

    manifest = generator.generate(
        timeline=timeline,
        source="test",
    )

    assert len(manifest.timeline_constraints) > 0
    # Should mention T0 or baseline constraints


def test_constraint_generator_boundaries():
    """Test boundary generation"""
    generator = ConstraintManifestGenerator()

    invariant_canon = [{"facts": ["Rule 1", "Rule 2"], "immutable": True}]

    manifest = generator.generate(
        invariant_canon=invariant_canon,
        source="test",
    )

    assert len(manifest.boundaries) > 0
    # Should mention number of rules to respect


def test_constraint_generator_guidance():
    """Test guidance generation"""
    generator = ConstraintManifestGenerator()
    registry = EntityRegistry()

    registry.create(
        Entity(
            name="Hero",
            entity_type=EntityType.CHARACTER,
            role="protagonist",
            description="Main character",
            source="test",
            immutable=False,
        )
    )

    mutable_canon = [{"facts": ["Mutable element 1"], "immutable": False}]

    manifest = generator.generate(
        mutable_canon=mutable_canon,
        entity_registry=registry,
        source="test",
    )

    assert len(manifest.guidance) > 0
    # Should provide positive creative guidance


def test_constraint_generator_complete():
    """Test complete manifest generation with all components"""
    generator = ConstraintManifestGenerator()

    # Prepare all components
    invariant_canon = [{"facts": ["Dragons are extinct"], "immutable": True}]
    mutable_canon = [
        {"facts": [{"statement": "Regional customs vary", "immutable": False}]}
    ]

    registry = EntityRegistry()
    registry.create(
        Entity(
            name="Ancient Dragon",
            entity_type=EntityType.CHARACTER,
            role="legendary",
            description="Last dragon",
            source="world-genesis",
            immutable=True,
        )
    )

    timeline = TimelineManager()
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Dragon extinction",
            year=0,
            source="world-genesis",
            immutable=True,
        )
    )

    # Generate manifest
    manifest = generator.generate(
        invariant_canon=invariant_canon,
        mutable_canon=mutable_canon,
        entity_registry=registry,
        timeline=timeline,
        source="canon-import",
    )

    # Verify all sections populated
    assert len(manifest.invariants) > 0
    assert len(manifest.mutables) > 0
    assert len(manifest.entity_constraints) > 0
    assert len(manifest.timeline_constraints) > 0
    assert len(manifest.boundaries) > 0
    assert len(manifest.guidance) > 0
    assert manifest.metadata["source"] == "canon-import"

    # Verify markdown export works
    markdown = manifest.to_markdown()
    assert len(markdown) > 100
    assert "extinct" in markdown.lower()
