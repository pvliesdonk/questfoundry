"""Tests for Post Mortem loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.post_mortem import PostMortemLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.showrunner import Showrunner
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for post mortem",
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
def loop_context(temp_workspace, mock_provider, spec_path):
    """Fixture providing loop context."""
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="gatecheck_report",
            data={"decision": "pass", "bars": []},
            metadata={"created_by": "gatekeeper"},
        ),
        Artifact(
            type="tu_brief",
            data={"id": "TU-001", "status": "complete"},
            metadata={"created_by": "showrunner"},
        ),
    ]

    context = LoopContext(
        loop_id="post_mortem",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
    )

    return context


def test_post_mortem_initialization(loop_context):
    """Test Post Mortem initialization."""
    loop = PostMortemLoop(loop_context)

    assert loop.metadata.loop_id == "post_mortem"
    assert len(loop.steps) == 5
    assert len(loop.action_items) == 0


def test_post_mortem_metadata():
    """Test Post Mortem metadata."""
    metadata = PostMortemLoop.metadata

    assert metadata.loop_id == "post_mortem"
    assert "gatekeeper" in metadata.primary_roles


def test_post_mortem_steps():
    """Test Post Mortem step definitions."""
    steps = PostMortemLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "collect_metrics"
    assert steps[1].step_id == "final_validation"
    assert steps[2].step_id == "create_report"
    assert steps[3].step_id == "archive"
    assert steps[4].step_id == "package"


def test_post_mortem_execute_success(loop_context):
    """Test successful Post Mortem execution."""
    loop = PostMortemLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_post_mortem_collect_metrics_step(loop_context):
    """Test collect_metrics step."""
    loop = PostMortemLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.metrics) > 0


def test_post_mortem_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = PostMortemLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_post_mortem_metadata_includes_counts(loop_context):
    """Test metadata includes action items count."""
    loop = PostMortemLoop(loop_context)

    result = loop.execute()

    assert "action_items_count" in result.metadata
    assert "lessons_learned_count" in result.metadata
