"""Tests for Scene Forge loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.scene_forge import SceneForgeLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.scene_smith import SceneSmith
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for scene forge",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider with predefined responses."""
    return MockTextProvider(
        responses={
            "draft_scene": "Scene draft content with choices and descriptions",
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
    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)

    # Create some canon and hook artifacts
    artifacts = [
        Artifact(
            type="canon_pack",
            data={
                "entries": [
                    {
                        "title": "Location canon",
                        "answer": "Details about the location",
                    }
                ]
            },
            metadata={"created_by": "lore_weaver"},
        ),
        Artifact(
            type="hook_card",
            data={
                "status": "accepted",
                "title": "Scene hook",
                "summary": "A dramatic encounter",
            },
            metadata={"created_by": "plotwright"},
        ),
    ]

    context = LoopContext(
        loop_id="scene_forge",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "scene_smith": scene_smith,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
        config={"scene_count": 3},
    )

    return context


def test_scene_forge_initialization(loop_context):
    """Test Scene Forge loop initialization."""
    loop = SceneForgeLoop(loop_context)

    assert loop.metadata.loop_id == "scene_forge"
    assert loop.context == loop_context
    assert len(loop.steps) == 5
    assert len(loop.selected_scenes) == 0
    assert len(loop.scene_drafts) == 0
    assert loop.scene_count == 3


def test_scene_forge_metadata():
    """Test Scene Forge metadata."""
    metadata = SceneForgeLoop.metadata

    assert metadata.loop_id == "scene_forge"
    assert metadata.display_name == "Scene Forge"
    assert "scene_smith" in metadata.primary_roles


def test_scene_forge_steps():
    """Test Scene Forge step definitions."""
    steps = SceneForgeLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "select_scenes"
    assert steps[1].step_id == "gather_context"
    assert steps[2].step_id == "draft_scenes"
    assert steps[3].step_id == "style_pass"
    assert steps[4].step_id == "package_scenes"


def test_scene_forge_execute_success(loop_context):
    """Test successful Scene Forge execution."""
    loop = SceneForgeLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "scene_forge"
    assert result.steps_completed == 5
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_scene_forge_select_scenes_step(loop_context):
    """Test select_scenes step."""
    loop = SceneForgeLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.selected_scenes) > 0
    assert len(loop.selected_scenes) == 3


def test_scene_forge_gather_context_step(loop_context):
    """Test gather_context step."""
    loop = SceneForgeLoop(loop_context)

    # Run select_scenes first
    loop.execute_step(loop.steps[0])

    # Run gather_context
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_scene_forge_draft_scenes_step(loop_context):
    """Test draft_scenes step."""
    loop = SceneForgeLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run draft_scenes
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.scene_drafts) > 0


def test_scene_forge_style_pass_step(loop_context):
    """Test style_pass step."""
    loop = SceneForgeLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run style_pass
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_scene_forge_package_scenes_step(loop_context):
    """Test package_scenes step."""
    loop = SceneForgeLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run package_scenes
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_scene_forge_artifacts_created(loop_context):
    """Test that Scene Forge creates expected artifacts."""
    loop = SceneForgeLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert len(result.artifacts_created) >= 5


def test_scene_forge_scene_drafts(loop_context):
    """Test scene draft creation."""
    loop = SceneForgeLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.metadata.get("scenes_drafted", 0) > 0


def test_scene_forge_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = SceneForgeLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_scene_forge_context_summary(loop_context):
    """Test building loop context summary."""
    loop = SceneForgeLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Scene Forge" in summary
    assert "scene_smith" in summary.lower()


def test_scene_forge_repr(loop_context):
    """Test loop string representation."""
    loop = SceneForgeLoop(loop_context)

    repr_str = repr(loop)

    assert "SceneForgeLoop" in repr_str
    assert "scene_forge" in repr_str


def test_scene_forge_custom_scene_count(temp_workspace, mock_provider, spec_path):
    """Test Scene Forge with custom scene count."""
    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)

    context = LoopContext(
        loop_id="scene_forge",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={"scene_smith": scene_smith},
        artifacts=[],
        project_metadata={"name": "Test Project"},
        config={"scene_count": 5},  # Custom count
    )

    loop = SceneForgeLoop(context)

    assert loop.scene_count == 5

    result = loop.execute()
    assert result.success
