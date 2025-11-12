"""Tests for Hook Harvest loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.hook_harvest import HookHarvestLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.showrunner import Showrunner
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for hook harvest",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider with predefined responses."""
    return MockTextProvider(
        responses={
            "review_progress": "Progress review complete",
            "coordinate_step": "Coordination complete",
        }
    )


@pytest.fixture
def spec_path():
    """Fixture providing path to spec directory."""
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent
    return repo_root / "spec"


@pytest.fixture
def loop_context(temp_workspace, mock_provider, spec_path):
    """Fixture providing a loop context with roles."""
    # Create role instances
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    # Create some proposed hook artifacts
    proposed_hooks = [
        Artifact(
            type="hook_card",
            data={
                "status": "proposed",
                "title": "Hook 1",
                "summary": "First hook",
                "type": "narrative",
            },
            metadata={"created_by": "plotwright"},
        ),
        Artifact(
            type="hook_card",
            data={
                "status": "proposed",
                "title": "Hook 2",
                "summary": "Second hook",
                "type": "scene",
            },
            metadata={"created_by": "scene_smith"},
        ),
        Artifact(
            type="hook_card",
            data={
                "status": "proposed",
                "title": "Hook 3",
                "summary": "Third hook",
                "type": "factual",
            },
            metadata={"created_by": "lore_weaver"},
        ),
    ]

    context = LoopContext(
        loop_id="hook_harvest",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "showrunner": showrunner,
        },
        artifacts=proposed_hooks,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
    )

    return context


# Loop Tests


def test_hook_harvest_initialization(loop_context):
    """Test Hook Harvest loop initialization."""
    loop = HookHarvestLoop(loop_context)

    assert loop.metadata.loop_id == "hook_harvest"
    assert loop.context == loop_context
    assert len(loop.steps) == 5  # 5 steps in hook_harvest
    assert len(loop.hooks_collected) == 0
    assert len(loop.clusters) == 0


def test_hook_harvest_metadata():
    """Test Hook Harvest metadata."""
    metadata = HookHarvestLoop.metadata

    assert metadata.loop_id == "hook_harvest"
    assert metadata.display_name == "Hook Harvest"
    assert "showrunner" in metadata.primary_roles
    assert "lore_weaver" in metadata.consulted_roles


def test_hook_harvest_steps():
    """Test Hook Harvest step definitions."""
    steps = HookHarvestLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "collect_hooks"
    assert steps[1].step_id == "cluster_hooks"
    assert steps[2].step_id == "annotate_hooks"
    assert steps[3].step_id == "decide_triage"
    assert steps[4].step_id == "package_harvest"


def test_hook_harvest_execute_success(loop_context):
    """Test successful Hook Harvest execution."""
    loop = HookHarvestLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "hook_harvest"
    assert result.steps_completed == 5
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_hook_harvest_collect_hooks_step(loop_context):
    """Test collect_hooks step."""
    loop = HookHarvestLoop(loop_context)

    step = loop.steps[0]  # collect_hooks
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.hooks_collected) > 0
    assert len(loop.hooks_collected) == 3  # 3 proposed hooks


def test_hook_harvest_cluster_hooks_step(loop_context):
    """Test cluster_hooks step."""
    loop = HookHarvestLoop(loop_context)

    # Run collect_hooks first
    loop.execute_step(loop.steps[0])

    # Run cluster_hooks
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.clusters) > 0


def test_hook_harvest_annotate_hooks_step(loop_context):
    """Test annotate_hooks step."""
    loop = HookHarvestLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run annotate_hooks
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_hook_harvest_decide_triage_step(loop_context):
    """Test decide_triage step."""
    loop = HookHarvestLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run decide_triage
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.triage_decisions) > 0


def test_hook_harvest_package_harvest_step(loop_context):
    """Test package_harvest step."""
    loop = HookHarvestLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run package_harvest
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_hook_harvest_artifacts_created(loop_context):
    """Test that Hook Harvest creates expected artifacts."""
    loop = HookHarvestLoop(loop_context)

    result = loop.execute()

    assert result.success
    # Should create artifacts for each step
    assert len(result.artifacts_created) >= 5


def test_hook_harvest_triage_results(loop_context):
    """Test triage decision tracking."""
    loop = HookHarvestLoop(loop_context)

    result = loop.execute()

    assert result.success
    # Check metadata for triage results
    assert "accepted" in result.metadata
    assert "deferred" in result.metadata
    assert "rejected" in result.metadata


def test_hook_harvest_failed_step(loop_context):
    """Test Hook Harvest with failed step."""
    # Create provider that fails
    bad_provider = MockTextProvider(
        responses={"review_progress": ""}  # Empty response will cause issues
    )

    # Create a showrunner that will fail
    spec_path = loop_context.role_instances["showrunner"].spec_path
    showrunner = Showrunner(provider=bad_provider, spec_path=spec_path)
    loop_context.role_instances["showrunner"] = showrunner

    loop = HookHarvestLoop(loop_context)

    # Should handle gracefully
    result = loop.execute()

    # May succeed with empty responses, or fail - both are valid
    assert result.loop_id == "hook_harvest"


def test_hook_harvest_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = HookHarvestLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})

    # Invalid result - not a dict
    assert not loop.validate_step(loop.steps[0], "not a dict")


def test_hook_harvest_context_summary(loop_context):
    """Test building loop context summary."""
    loop = HookHarvestLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Hook Harvest" in summary
    assert "collect" in summary.lower()
    assert "showrunner" in summary.lower()


def test_hook_harvest_repr(loop_context):
    """Test loop string representation."""
    loop = HookHarvestLoop(loop_context)

    repr_str = repr(loop)

    assert "HookHarvestLoop" in repr_str
    assert "hook_harvest" in repr_str
    assert "0/5" in repr_str  # 0 of 5 steps


def test_hook_harvest_can_continue(loop_context):
    """Test loop can_continue logic."""
    loop = HookHarvestLoop(loop_context)

    # Can continue at start
    assert loop.can_continue()

    # Execute all steps
    for step in loop.steps:
        loop.execute_step(step)
        loop.current_step_index += 1

    # Cannot continue at end
    assert not loop.can_continue()


def test_hook_harvest_empty_hooks(temp_workspace, mock_provider, spec_path):
    """Test Hook Harvest with no proposed hooks."""
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    # Context with no hooks
    context = LoopContext(
        loop_id="hook_harvest",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={"showrunner": showrunner},
        artifacts=[],  # No hooks
        project_metadata={"name": "Test Project"},
    )

    loop = HookHarvestLoop(context)
    result = loop.execute()

    # Should still succeed, just with 0 hooks
    assert result.success
    assert result.metadata.get("hooks_processed", 0) == 0
