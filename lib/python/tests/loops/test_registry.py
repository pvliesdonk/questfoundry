"""Tests for loop registry."""

from pathlib import Path

import pytest

from questfoundry.loops.registry import LoopMetadata, LoopRegistry


@pytest.fixture
def spec_path():
    """Fixture providing path to spec directory."""
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent
    return repo_root / "spec"


@pytest.fixture
def registry(spec_path):
    """Fixture providing a loop registry."""
    return LoopRegistry(spec_path=spec_path)


def test_registry_initialization(registry):
    """Test loop registry initialization."""
    assert registry is not None
    assert len(registry._loops) == 11  # All 11 loops registered


def test_get_loop_metadata(registry):
    """Test getting loop metadata by ID."""
    metadata = registry.get_loop_metadata("story_spark")

    assert metadata.loop_id == "story_spark"
    assert metadata.display_name == "Story Spark"
    assert metadata.description
    assert metadata.typical_duration
    assert len(metadata.primary_roles) > 0


def test_get_loop_metadata_not_found(registry):
    """Test error when loop not found."""
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get_loop_metadata("nonexistent")


def test_list_all_loops(registry):
    """Test listing all loops."""
    loops = registry.list_loops()

    assert len(loops) == 11
    assert all(isinstance(loop, LoopMetadata) for loop in loops)


def test_list_loops_by_tag(registry):
    """Test filtering loops by tag."""
    structure_loops = registry.list_loops(filters={"tag": "structure"})

    assert len(structure_loops) > 0
    assert all("structure" in loop.tags for loop in structure_loops)


def test_list_loops_by_role(registry):
    """Test filtering loops by role."""
    plotwright_loops = registry.list_loops(filters={"role": "plotwright"})

    assert len(plotwright_loops) > 0
    for loop in plotwright_loops:
        assert (
            "plotwright" in loop.primary_roles or "plotwright" in loop.consulted_roles
        )


def test_list_loops_by_duration(registry):
    """Test filtering loops by duration."""
    short_loops = registry.list_loops(filters={"duration": "1-2 hours"})

    assert len(short_loops) > 0
    assert all("1-2 hours" in loop.typical_duration for loop in short_loops)


def test_get_loops_by_role(registry):
    """Test getting loops by role."""
    gatekeeper_loops = registry.get_loops_by_role("gatekeeper")

    assert len(gatekeeper_loops) > 0
    for loop in gatekeeper_loops:
        assert (
            "gatekeeper" in loop.primary_roles or "gatekeeper" in loop.consulted_roles
        )


def test_get_loops_by_tag(registry):
    """Test getting loops by tag."""
    quality_loops = registry.get_loops_by_tag("quality")

    assert len(quality_loops) > 0
    assert all("quality" in loop.tags for loop in quality_loops)


def test_build_registry_context(registry):
    """Test building lightweight registry context."""
    context = registry.build_registry_context()

    assert isinstance(context, str)
    assert "Story Spark" in context
    assert "Hook Harvest" in context
    assert len(context) > 0

    # Should be lightweight - approximately 90 lines
    lines = context.split("\n")
    assert len(lines) < 150  # Reasonable upper bound


def test_all_loops_have_metadata(registry):
    """Test that all 11 loops have complete metadata."""
    expected_loops = [
        "story_spark",
        "hook_harvest",
        "lore_deepening",
        "codex_expansion",
        "binding_run",
        "style_tune_up",
        "art_touch_up",
        "audio_pass",
        "translation_pass",
        "narration_dry_run",
        "full_production_run",
    ]

    for loop_id in expected_loops:
        metadata = registry.get_loop_metadata(loop_id)
        assert metadata.loop_id == loop_id
        assert metadata.display_name
        assert metadata.description
        assert metadata.typical_duration


def test_loop_metadata_structure():
    """Test loop metadata structure."""
    metadata = LoopMetadata(
        loop_id="test_loop",
        display_name="Test Loop",
        description="A test loop",
        typical_duration="1 hour",
        primary_roles=["role1"],
        consulted_roles=["role2"],
        entry_conditions=["condition1"],
        exit_conditions=["outcome1"],
        output_artifacts=["artifact1"],
        inputs=["input1"],
        tags=["tag1"],
    )

    assert metadata.loop_id == "test_loop"
    assert metadata.display_name == "Test Loop"
    assert len(metadata.primary_roles) == 1
    assert len(metadata.tags) == 1


def test_story_spark_metadata(registry):
    """Test Story Spark loop metadata details."""
    story_spark = registry.get_loop_metadata("story_spark")

    assert story_spark.loop_id == "story_spark"
    assert "plotwright" in story_spark.primary_roles
    assert "scene_smith" in story_spark.primary_roles
    assert "gatekeeper" in story_spark.consulted_roles
    assert "structure" in story_spark.tags
    assert len(story_spark.output_artifacts) > 0


def test_hook_harvest_metadata(registry):
    """Test Hook Harvest loop metadata details."""
    hook_harvest = registry.get_loop_metadata("hook_harvest")

    assert hook_harvest.loop_id == "hook_harvest"
    assert "showrunner" in hook_harvest.primary_roles
    assert "planning" in hook_harvest.tags
    assert "hook_card" in hook_harvest.output_artifacts


def test_binding_run_metadata(registry):
    """Test Binding Run loop metadata details."""
    binding_run = registry.get_loop_metadata("binding_run")

    assert binding_run.loop_id == "binding_run"
    assert "book_binder" in binding_run.primary_roles
    assert "finalization" in binding_run.tags
    assert "front_matter" in binding_run.output_artifacts


def test_full_production_run_metadata(registry):
    """Test Full Production Run loop metadata details."""
    full_prod = registry.get_loop_metadata("full_production_run")

    assert full_prod.loop_id == "full_production_run"
    assert "showrunner" in full_prod.primary_roles
    assert "complete" in full_prod.tags
    # Should have multiple output types
    assert len(full_prod.output_artifacts) >= 4


def test_registry_repr(registry):
    """Test registry string representation."""
    repr_str = repr(registry)

    assert "LoopRegistry" in repr_str
    assert "loops=11" in repr_str
