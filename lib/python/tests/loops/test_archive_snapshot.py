"""Tests for Archive Snapshot loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.archive_snapshot import ArchiveSnapshotLoop
from questfoundry.loops.base import LoopContext, StepStatus
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
            description="Test project for archive snapshot",
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
            type="cold_snapshot",
            data={"snapshot_id": "cold-001"},
            metadata={"created_by": "system"},
        ),
        Artifact(
            type="hot_snapshot",
            data={"snapshot_id": "hot-001"},
            metadata={"created_by": "system"},
        ),
    ]

    context = LoopContext(
        loop_id="archive_snapshot",
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


def test_archive_snapshot_initialization(loop_context):
    """Test Archive Snapshot initialization."""
    loop = ArchiveSnapshotLoop(loop_context)

    assert loop.metadata.loop_id == "archive_snapshot"
    assert len(loop.steps) == 5
    assert loop.snapshot_id == ""


def test_archive_snapshot_metadata():
    """Test Archive Snapshot metadata."""
    metadata = ArchiveSnapshotLoop.metadata

    assert metadata.loop_id == "archive_snapshot"
    assert "showrunner" in metadata.primary_roles


def test_archive_snapshot_steps():
    """Test Archive Snapshot step definitions."""
    steps = ArchiveSnapshotLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "select_artifacts"
    assert steps[1].step_id == "create_snapshot"
    assert steps[2].step_id == "validate"
    assert steps[3].step_id == "promote_to_cold"
    assert steps[4].step_id == "package"


def test_archive_snapshot_execute_success(loop_context):
    """Test successful Archive Snapshot execution."""
    loop = ArchiveSnapshotLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_archive_snapshot_select_artifacts_step(loop_context):
    """Test select_artifacts step."""
    loop = ArchiveSnapshotLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.artifacts_selected) >= 0


def test_archive_snapshot_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = ArchiveSnapshotLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_archive_snapshot_metadata_includes_snapshot_id(loop_context):
    """Test metadata includes snapshot ID."""
    loop = ArchiveSnapshotLoop(loop_context)

    result = loop.execute()

    assert "snapshot_id" in result.metadata
    assert "artifacts_archived" in result.metadata
