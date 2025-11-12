"""Tests for loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, LoopStep, StepStatus
from questfoundry.loops.story_spark import StorySparkLoop
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.plotwright import Plotwright
from questfoundry.roles.scene_smith import SceneSmith
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for loop execution",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider with predefined responses."""
    return MockTextProvider(
        responses={
            "generate_hooks": (
                '{"hooks": [{"title": "Test Hook", '
                '"summary": "A test hook", "tags": ["test"]}]}'
            ),
            "create_topology": "Test topology content",
            "create_tu_brief": "Test TU brief content",
            "create_section_briefs": "Test section briefs content",
            "draft_scene": "Test scene content",
            "pre_gate": (
                '{"status": "pass", "blockers": [], '
                '"quick_wins": [], "review_needed": []}'
            ),
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
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)
    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)

    context = LoopContext(
        loop_id="story_spark",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "plotwright": plotwright,
            "scene_smith": scene_smith,
            "gatekeeper": gatekeeper,
        },
        project_metadata={"name": "Test Project", "genre": "fantasy"},
    )

    return context


# Loop Step Tests


def test_loop_step_initialization():
    """Test loop step initialization."""
    step = LoopStep(
        step_id="test_step",
        description="A test step",
        assigned_roles=["role1"],
        consulted_roles=["role2"],
        artifacts_input=["input1"],
        artifacts_output=["output1"],
    )

    assert step.step_id == "test_step"
    assert step.status == StepStatus.PENDING
    assert step.validation_required is True
    assert step.result is None


def test_loop_step_status_transitions():
    """Test loop step status transitions."""
    step = LoopStep(step_id="test", description="Test")

    assert step.status == StepStatus.PENDING

    step.status = StepStatus.IN_PROGRESS
    assert step.status == StepStatus.IN_PROGRESS

    step.status = StepStatus.COMPLETED
    assert step.status == StepStatus.COMPLETED


# Story Spark Loop Tests


def test_story_spark_initialization(loop_context):
    """Test Story Spark loop initialization."""
    loop = StorySparkLoop(loop_context)

    assert loop.metadata.loop_id == "story_spark"
    assert loop.context == loop_context
    assert len(loop.steps) == 6  # 6 steps in story_spark
    assert loop.iteration_count == 0


def test_story_spark_metadata():
    """Test Story Spark metadata."""
    metadata = StorySparkLoop.metadata

    assert metadata.loop_id == "story_spark"
    assert metadata.display_name == "Story Spark"
    assert "plotwright" in metadata.primary_roles
    assert "scene_smith" in metadata.primary_roles
    assert "gatekeeper" in metadata.consulted_roles


def test_story_spark_steps():
    """Test Story Spark step definitions."""
    steps = StorySparkLoop.steps

    assert len(steps) == 6
    assert steps[0].step_id == "generate_hooks"
    assert steps[1].step_id == "create_topology"
    assert steps[2].step_id == "create_tu_brief"
    assert steps[3].step_id == "create_section_briefs"
    assert steps[4].step_id == "draft_scenes"
    assert steps[5].step_id == "pre_gate_check"


def test_story_spark_execute_success(loop_context):
    """Test successful Story Spark execution."""
    loop = StorySparkLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "story_spark"
    assert result.steps_completed == 6
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_story_spark_generate_hooks_step(loop_context):
    """Test generate_hooks step."""
    loop = StorySparkLoop(loop_context)

    step = loop.steps[0]  # generate_hooks
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert loop.hooks_generated is not None
    assert len(loop.hooks_generated) > 0


def test_story_spark_create_topology_step(loop_context):
    """Test create_topology step."""
    loop = StorySparkLoop(loop_context)

    # Run generate_hooks first
    loop.execute_step(loop.steps[0])

    # Run create_topology
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_story_spark_pre_gate_check_step(loop_context):
    """Test pre_gate_check step."""
    loop = StorySparkLoop(loop_context)

    # Run all previous steps
    for i in range(5):
        loop.execute_step(loop.steps[i])

    # Run pre_gate_check
    step = loop.steps[5]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result.get("passed") is True


def test_story_spark_artifacts_created(loop_context):
    """Test that Story Spark creates expected artifacts."""
    loop = StorySparkLoop(loop_context)

    result = loop.execute()

    assert result.success
    # Should create: hook_card, tu_brief (topology), tu_brief (brief),
    # canon_pack (section briefs), canon_pack (scenes), gatecheck_report
    assert len(result.artifacts_created) >= 4


def test_story_spark_failed_step(loop_context):
    """Test Story Spark with failed step."""
    # Create provider that fails
    bad_provider = MockTextProvider(responses={"generate_hooks": "Invalid JSON"})

    spec_path = loop_context.role_instances["plotwright"].spec_path
    plotwright = Plotwright(provider=bad_provider, spec_path=spec_path)
    loop_context.role_instances["plotwright"] = plotwright

    loop = StorySparkLoop(loop_context)

    result = loop.execute()

    assert not result.success
    assert result.steps_failed > 0


def test_story_spark_pre_gate_failure_iteration(loop_context):
    """Test Story Spark iteration on pre-gate failure."""
    # Create provider that fails pre-gate
    provider = MockTextProvider(
        responses={
            "generate_hooks": (
                '{"hooks": [{"title": "Test", "summary": "Test", "tags": []}]}'
            ),
            "create_topology": "Topology",
            "create_tu_brief": "Brief",
            "create_section_briefs": "Briefs",
            "draft_scene": "Scene",
            "pre_gate": (
                '{"status": "fail", "blockers": ["Issue 1"], '
                '"quick_wins": [], "review_needed": []}'
            ),
        }
    )

    # Update roles with failing provider
    for role_name in loop_context.role_instances:
        role_class = loop_context.role_instances[role_name].__class__
        loop_context.role_instances[role_name] = role_class(
            provider=provider,
            spec_path=loop_context.role_instances[role_name].spec_path,
        )

    loop = StorySparkLoop(loop_context)

    result = loop.execute()

    # Should fail due to pre-gate
    assert not result.success
    error_lower = result.error.lower()
    assert "iteration needed" in error_lower or "iterations" in error_lower
    has_refinement = result.metadata.get("refinement_needed")
    has_iterations = result.metadata.get("iterations", 0) > 0
    assert has_refinement or has_iterations


def test_story_spark_context_summary(loop_context):
    """Test building loop context summary."""
    loop = StorySparkLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Story Spark" in summary
    assert "generate quest hooks" in summary.lower()
    assert "plotwright" in summary.lower()


def test_story_spark_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = StorySparkLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})

    # Invalid result - not a dict
    assert not loop.validate_step(loop.steps[0], "not a dict")

    # Pre-gate step - passed
    assert loop.validate_step(loop.steps[5], {"success": True, "passed": True})

    # Pre-gate step - failed
    assert not loop.validate_step(loop.steps[5], {"success": True, "passed": False})


def test_loop_repr(loop_context):
    """Test loop string representation."""
    loop = StorySparkLoop(loop_context)

    repr_str = repr(loop)

    assert "StorySparkLoop" in repr_str
    assert "story_spark" in repr_str
    assert "0/6" in repr_str  # 0 of 6 steps


def test_loop_can_continue(loop_context):
    """Test loop can_continue logic."""
    loop = StorySparkLoop(loop_context)

    # Can continue at start
    assert loop.can_continue()

    # Execute all steps
    for step in loop.steps:
        loop.execute_step(step)
        loop.current_step_index += 1

    # Cannot continue at end
    assert not loop.can_continue()


def test_loop_rollback(loop_context):
    """Test loop rollback functionality."""
    loop = StorySparkLoop(loop_context)

    # Execute first step
    loop.execute_step(loop.steps[0])
    loop.current_step_index = 1

    assert loop.current_step_index == 1

    # Rollback
    loop.rollback_step()

    assert loop.current_step_index == 0
    assert loop.steps[0].status == StepStatus.PENDING


def test_loop_skip_step(loop_context):
    """Test loop skip step functionality."""
    loop = StorySparkLoop(loop_context)

    step = loop.steps[0]
    loop.skip_step(step)

    assert step.status == StepStatus.SKIPPED
    assert loop.current_step_index == 1
