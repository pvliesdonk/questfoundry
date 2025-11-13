"""Tests for timeline management (Layer 6/7 canon workflows)"""

import pytest

from questfoundry.state.timeline import TimelineAnchor, TimelineManager


def test_timeline_anchor_creation():
    """Test timeline anchor creation"""
    anchor = TimelineAnchor(
        anchor_id="T0",
        event="Kingdom founding",
        year=0,
        description="First king unified the clans",
        source="world-genesis",
        immutable=True,
    )

    assert anchor.anchor_id == "T0"
    assert anchor.year == 0
    assert anchor.immutable is True


def test_timeline_manager_add_anchor():
    """Test adding timeline anchors"""
    timeline = TimelineManager()

    anchor = TimelineAnchor(
        anchor_id="T0",
        event="Kingdom founding",
        year=0,
        source="world-genesis",
        immutable=True,
    )
    timeline.add_anchor(anchor)

    assert len(timeline) == 1
    t0 = timeline.get_anchor("T0")
    assert t0 is not None
    assert t0.event == "Kingdom founding"


def test_timeline_manager_add_duplicate_anchor():
    """Test adding duplicate anchor raises ValueError"""
    timeline = TimelineManager()

    anchor1 = TimelineAnchor(
        anchor_id="T0",
        event="Kingdom founding",
        year=0,
        source="world-genesis",
    )
    timeline.add_anchor(anchor1)

    anchor2 = TimelineAnchor(
        anchor_id="T0",
        event="Different event",
        year=100,
        source="test",
    )

    with pytest.raises(ValueError, match="Anchor.*T0.*already exists"):
        timeline.add_anchor(anchor2)


def test_timeline_manager_invalid_anchor_id():
    """Test invalid anchor ID raises ValueError"""
    # anchor_id validation happens in TimelineAnchor, not TimelineManager
    # Empty anchor_id is caught by TimelineAnchor.__post_init__
    with pytest.raises(ValueError, match="Anchor ID cannot be empty"):
        TimelineAnchor(
            anchor_id="",
            event="Test event",
            source="test",
        )


def test_timeline_manager_get_baseline_anchors():
    """Test getting baseline anchors"""
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=250,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T3-REVOLT",
            event="Rebellion",
            offset=500,
            source="project-local",
        )
    )

    baseline = timeline.get_baseline_anchors()
    assert len(baseline) == 2
    assert all(a.anchor_id in ("T0", "T1") for a in baseline)


def test_timeline_manager_get_extension_anchors():
    """Test getting extension anchors"""
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T3-REVOLT",
            event="Rebellion",
            offset=500,
            source="project-local",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T4-WAR",
            event="Great War",
            offset=600,
            source="project-local",
        )
    )

    extensions = timeline.get_extension_anchors()
    assert len(extensions) == 2
    # All extensions should start with T3 or T4
    for ext in extensions:
        assert ext.anchor_id.startswith("T3") or ext.anchor_id.startswith("T4")


def test_timeline_manager_update_anchor():
    """Test updating anchor"""
    timeline = TimelineManager()

    anchor = TimelineAnchor(
        anchor_id="T1",
        event="Peace treaty",
        year=250,
        description="Original description",
        source="world-genesis",
        immutable=False,
    )
    timeline.add_anchor(anchor)

    # Create updated anchor
    updated_anchor = TimelineAnchor(
        anchor_id="T1",
        event="Peace treaty",
        year=251,
        description="Updated description",
        source="world-genesis",
        immutable=False,
    )
    timeline.update_anchor(updated_anchor)

    result = timeline.get_anchor("T1")
    assert result.description == "Updated description"
    assert result.year == 251
    assert result.event == "Peace treaty"


def test_timeline_manager_update_immutable_anchor():
    """Test updating immutable anchor raises ValueError"""
    timeline = TimelineManager()

    anchor = TimelineAnchor(
        anchor_id="T0",
        event="Foundation",
        year=0,
        source="world-genesis",
        immutable=True,
    )
    timeline.add_anchor(anchor)

    # Try to update immutable anchor
    updated = TimelineAnchor(
        anchor_id="T0",
        event="Foundation",
        year=1,
        source="world-genesis",
        immutable=True,
    )

    with pytest.raises(ValueError, match="Cannot update immutable anchor"):
        timeline.update_anchor(updated)


def test_timeline_manager_delete_anchor():
    """Test deleting anchor"""
    timeline = TimelineManager()

    anchor = TimelineAnchor(
        anchor_id="T3-TEMP",
        event="Temporary event",
        offset=500,
        source="test",
        immutable=False,
    )
    timeline.add_anchor(anchor)
    assert len(timeline) == 1

    timeline.delete_anchor("T3-TEMP")
    assert len(timeline) == 0


def test_timeline_manager_delete_immutable_anchor():
    """Test deleting immutable anchor raises ValueError"""
    timeline = TimelineManager()

    anchor = TimelineAnchor(
        anchor_id="T0",
        event="Foundation",
        year=0,
        source="world-genesis",
        immutable=True,
    )
    timeline.add_anchor(anchor)

    with pytest.raises(ValueError, match="Cannot delete immutable anchor"):
        timeline.delete_anchor("T0")


def test_timeline_validate_chronology():
    """Test chronology validation"""
    timeline = TimelineManager()

    # Add anchors in proper order
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=250,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T2",
            event="Great expansion",
            year=500,
            source="world-genesis",
        )
    )

    errors = timeline.validate_chronology()
    assert len(errors) == 0


def test_timeline_validate_chronology_with_errors():
    """Test chronology validation detects errors"""
    timeline = TimelineManager()

    # Add anchors with offset mismatch (offset doesn't match year calculation)
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=1000,
            source="test",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=1250,
            offset=500,  # Should be 250 (1250 - 1000), not 500
            source="test",
        )
    )

    errors = timeline.validate_chronology()
    assert len(errors) > 0
    assert any("offset" in error.lower() for error in errors)


def test_timeline_validate_missing_baseline():
    """Test validation detects missing T0"""
    timeline = TimelineManager()

    # Add T1 without T0
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=250,
            source="test",
        )
    )

    errors = timeline.validate_chronology()
    assert len(errors) > 0
    assert any("T0" in error and "missing" in error.lower() for error in errors)


def test_timeline_validate_gap_warning():
    """Test validation detects large gaps"""
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Event",
            year=10000,  # Large gap
            source="world-genesis",
        )
    )

    errors = timeline.validate_chronology()
    # Should have warning about large gap
    assert len(errors) > 0
    assert any("gap" in error.lower() for error in errors)


def test_timeline_merge():
    """Test merging timelines"""
    timeline = TimelineManager()

    # Add baseline
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            source="world-genesis",
            immutable=False,
        )
    )

    # Merge with canonical timeline
    imported = [
        TimelineAnchor(
            anchor_id="T0",
            event="Kingdom founding (canonical)",
            year=0,
            source="canon-import",
            immutable=True,
        ),
        TimelineAnchor(
            anchor_id="T1",
            event="Peace treaty",
            year=250,
            source="canon-import",
            immutable=True,
        ),
    ]

    report = timeline.merge(imported)

    # Immutable should take precedence
    t0 = timeline.get_anchor("T0")
    assert t0.immutable is True
    assert t0.source == "canon-import"
    assert "canonical" in t0.event.lower()

    # T1 should be added
    t1 = timeline.get_anchor("T1")
    assert t1 is not None
    assert t1.event == "Peace treaty"

    # Merge report: T0 updated (counted as added), T1 added
    assert report["added"] == 2
    assert report["skipped"] == 0


def test_timeline_to_dict():
    """Test serializing timeline to dictionary"""
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=0,
            description="Kingdom founding",
            source="world-genesis",
            immutable=True,
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T1",
            event="Peace",
            year=250,
            source="world-genesis",
            immutable=False,
        )
    )

    data = timeline.to_dict()
    assert "anchors" in data
    assert len(data["anchors"]) == 2
    assert data["anchors"][0]["anchor_id"] == "T0"
    assert data["anchors"][0]["immutable"] is True
    assert data["anchors"][1]["anchor_id"] == "T1"


def test_timeline_offset_resolution():
    """Test offset resolution relative to T0"""
    timeline = TimelineManager()

    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T0",
            event="Foundation",
            year=1000,
            source="world-genesis",
        )
    )
    timeline.add_anchor(
        TimelineAnchor(
            anchor_id="T3-EVENT",
            event="Rebellion",
            offset=500,  # 500 years after T0
            source="project-local",
        )
    )

    # Get T3-EVENT and check offset stored correctly
    t3 = timeline.get_anchor("T3-EVENT")
    assert t3.offset == 500

    # Test calculate_offset helper
    calculated_offset = timeline.calculate_offset(1500)
    assert calculated_offset == 500  # 1500 - 1000 = 500
