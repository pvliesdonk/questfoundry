"""Tests for Binding Run loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.binding_run import BindingRunLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.book_binder import BookBinder
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
            description="Test project for binding run",
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
    book_binder = BookBinder(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="cold_snapshot",
            data={"snapshot_id": "cold-001", "sections": ["Section 1"]},
            metadata={"created_by": "system"},
        ),
    ]

    context = LoopContext(
        loop_id="binding_run",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "book_binder": book_binder,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
        config={"formats": ["markdown", "html"]},
    )

    return context


def test_binding_run_initialization(loop_context):
    """Test Binding Run initialization."""
    loop = BindingRunLoop(loop_context)

    assert loop.metadata.loop_id == "binding_run"
    assert len(loop.steps) == 5
    assert "markdown" in loop.export_formats


def test_binding_run_metadata():
    """Test Binding Run metadata."""
    metadata = BindingRunLoop.metadata

    assert metadata.loop_id == "binding_run"
    assert "book_binder" in metadata.primary_roles


def test_binding_run_steps():
    """Test Binding Run step definitions."""
    steps = BindingRunLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "select_formats"
    assert steps[1].step_id == "generate_views"
    assert steps[2].step_id == "export_files"
    assert steps[3].step_id == "validate"
    assert steps[4].step_id == "package"


def test_binding_run_execute_success(loop_context):
    """Test successful Binding Run execution."""
    loop = BindingRunLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_binding_run_select_formats_step(loop_context):
    """Test select_formats step."""
    loop = BindingRunLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.export_config) > 0


def test_binding_run_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = BindingRunLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_binding_run_metadata_includes_formats(loop_context):
    """Test metadata includes export formats."""
    loop = BindingRunLoop(loop_context)

    result = loop.execute()

    assert "formats" in result.metadata
    assert "markdown" in result.metadata["formats"]
