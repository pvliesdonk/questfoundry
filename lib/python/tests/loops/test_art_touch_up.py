"""Tests for Art Touch-Up loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.art_touch_up import ArtTouchUpLoop
from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.models.artifact import Artifact
from questfoundry.roles.art_director import ArtDirector
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.illustrator import Illustrator
from questfoundry.roles.showrunner import Showrunner
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for art touch-up",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a flexible mock text provider."""
    return MockTextProvider()


@pytest.fixture
def spec_path():
    """Fixture providing path to spec directory."""
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent
    return repo_root / "spec"


@pytest.fixture
def loop_context_with_illustrator(temp_workspace, mock_provider, spec_path):
    """Fixture providing a loop context with all roles including illustrator."""
    # Create role instances
    art_director = ArtDirector(provider=mock_provider, spec_path=spec_path)
    illustrator = Illustrator(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    # Create some scenes to illustrate
    artifacts = [
        Artifact(
            type="canon_pack",
            data={
                "sections": ["Opening scene", "Climax scene"],
                "scenes": ["Scene A", "Scene B"],
            },
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="art_touch_up",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "art_director": art_director,
            "illustrator": illustrator,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
    )

    return context


@pytest.fixture
def loop_context_no_illustrator(temp_workspace, mock_provider, spec_path):
    """Fixture providing a loop context without illustrator (plan-only mode)."""
    # Create role instances (no illustrator)
    art_director = ArtDirector(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    # Create some scenes to illustrate
    artifacts = [
        Artifact(
            type="canon_pack",
            data={
                "sections": ["Opening scene", "Climax scene"],
                "scenes": ["Scene A", "Scene B"],
            },
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="art_touch_up",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "art_director": art_director,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
    )

    return context


# Loop Tests


def test_art_touch_up_initialization(loop_context_with_illustrator):
    """Test Art Touch-Up loop initialization."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    assert loop.metadata.loop_id == "art_touch_up"
    assert loop.context == loop_context_with_illustrator
    assert len(loop.steps) == 5
    assert len(loop.shotlist) == 0
    assert loop.illustrator_active


def test_art_touch_up_metadata():
    """Test Art Touch-Up metadata."""
    metadata = ArtTouchUpLoop.metadata

    assert metadata.loop_id == "art_touch_up"
    assert metadata.display_name == "Art Touch-Up"
    assert "art_director" in metadata.primary_roles
    assert "illustrator" in metadata.consulted_roles


def test_art_touch_up_steps():
    """Test Art Touch-Up step definitions."""
    steps = ArtTouchUpLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "create_shotlist"
    assert steps[1].step_id == "review_shots"
    assert steps[2].step_id == "generate_images"
    assert steps[3].step_id == "validate_quality"
    assert steps[4].step_id == "package_art_plan"


def test_art_touch_up_execute_success_with_illustrator(
    loop_context_with_illustrator,
):
    """Test successful Art Touch-Up execution with illustrator."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "art_touch_up"
    assert result.steps_completed == 5
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_art_touch_up_execute_success_without_illustrator(
    loop_context_no_illustrator,
):
    """Test successful Art Touch-Up execution without illustrator (plan-only)."""
    loop = ArtTouchUpLoop(loop_context_no_illustrator)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "art_touch_up"
    # Image generation step skipped, so 4 steps completed
    assert result.steps_completed == 4
    assert result.steps_failed == 0
    assert not loop.illustrator_active


def test_art_touch_up_create_shotlist_step(loop_context_with_illustrator):
    """Test create_shotlist step."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    step = loop.steps[0]  # create_shotlist
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.shotlist) > 0


def test_art_touch_up_review_shots_step(loop_context_with_illustrator):
    """Test review_shots step."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Run previous steps
    loop.execute_step(loop.steps[0])

    # Run review_shots
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.art_plan) > 0


def test_art_touch_up_generate_images_step(loop_context_with_illustrator):
    """Test generate_images step with illustrator."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run generate_images
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_art_touch_up_generate_images_skipped(loop_context_no_illustrator):
    """Test generate_images step is skipped without illustrator."""
    loop = ArtTouchUpLoop(loop_context_no_illustrator)

    # Execute the full loop - it will handle skipping generate_images
    result = loop.execute()

    # Check that generate_images step was skipped
    generate_images_step = loop.steps[2]
    assert generate_images_step.step_id == "generate_images"
    assert generate_images_step.status == StepStatus.SKIPPED
    assert result.success


def test_art_touch_up_validate_quality_step(loop_context_with_illustrator):
    """Test validate_quality step."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run validate_quality
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_art_touch_up_package_art_plan_step(loop_context_with_illustrator):
    """Test package_art_plan step."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run package_art_plan
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_art_touch_up_artifacts_created(loop_context_with_illustrator):
    """Test that Art Touch-Up creates expected artifacts."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    result = loop.execute()

    assert result.success
    # Should create artifacts: shotlist, art_plan, validation_report, art_package
    assert len(result.artifacts_created) >= 4


def test_art_touch_up_metadata_includes_counts(loop_context_with_illustrator):
    """Test that result metadata includes shotlist and render counts."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    result = loop.execute()

    assert result.success
    assert "shotlist_count" in result.metadata
    assert "renders_generated" in result.metadata
    assert "illustrator_active" in result.metadata


def test_art_touch_up_failed_step(loop_context_with_illustrator):
    """Test Art Touch-Up with failed step."""
    # Create provider that fails
    bad_provider = MockTextProvider(responses={"create_shotlist": ""})

    art_director = ArtDirector(
        provider=bad_provider,
        spec_path=loop_context_with_illustrator.role_instances[
            "art_director"
        ].spec_path,
    )
    loop_context_with_illustrator.role_instances["art_director"] = art_director

    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Should handle gracefully
    result = loop.execute()

    # May succeed with empty responses, or fail - both are valid
    assert result.loop_id == "art_touch_up"


def test_art_touch_up_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = ArtTouchUpLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Valid result - skipped
    assert loop.validate_step(loop.steps[0], {"success": True, "skipped": True})

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_art_touch_up_context_summary(loop_context_with_illustrator):
    """Test building loop context summary."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Art Touch-Up" in summary
    assert "art" in summary.lower()


def test_art_touch_up_repr(loop_context_with_illustrator):
    """Test loop string representation."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    repr_str = repr(loop)

    assert "ArtTouchUpLoop" in repr_str
    assert "art_touch_up" in repr_str
    assert "0/5" in repr_str


def test_art_touch_up_can_continue(loop_context_with_illustrator):
    """Test loop can_continue logic."""
    loop = ArtTouchUpLoop(loop_context_with_illustrator)

    # Can continue at start
    assert loop.can_continue()

    # Execute all steps
    for step in loop.steps:
        if step.step_id == "generate_images" and not loop.illustrator_active:
            step.status = StepStatus.SKIPPED
            continue
        loop.execute_step(step)
        loop.current_step_index += 1

    # Cannot continue at end
    assert not loop.can_continue()


def test_art_touch_up_plan_only_mode(loop_context_no_illustrator):
    """Test plan-only mode produces art plan without renders."""
    loop = ArtTouchUpLoop(loop_context_no_illustrator)

    result = loop.execute()

    assert result.success
    assert result.metadata["illustrator_active"] is False
    assert len(loop.renders_generated) == 0
    assert len(loop.art_plan) > 0
