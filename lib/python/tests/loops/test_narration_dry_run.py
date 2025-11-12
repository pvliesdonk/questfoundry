"""Tests for Narration Dry Run loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.narration_dry_run import NarrationDryRunLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.player_narrator import PlayerNarrator
from questfoundry.roles.showrunner import Showrunner
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for narration dry run",
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
    player_narrator = PlayerNarrator(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="export_bundle",
            data={"view_id": "view-001", "formats": ["markdown"]},
            metadata={"created_by": "book_binder"},
        ),
    ]

    context = LoopContext(
        loop_id="narration_dry_run",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "player_narrator": player_narrator,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
    )

    return context


def test_narration_dry_run_initialization(loop_context):
    """Test Narration Dry Run initialization."""
    loop = NarrationDryRunLoop(loop_context)

    assert loop.metadata.loop_id == "narration_dry_run"
    assert len(loop.steps) == 5
    assert len(loop.issue_counts) == 8


def test_narration_dry_run_metadata():
    """Test Narration Dry Run metadata."""
    metadata = NarrationDryRunLoop.metadata

    assert metadata.loop_id == "narration_dry_run"
    assert "player_narrator" in metadata.primary_roles


def test_narration_dry_run_steps():
    """Test Narration Dry Run step definitions."""
    steps = NarrationDryRunLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "select_path"
    assert steps[1].step_id == "simulate_playthrough"
    assert steps[2].step_id == "identify_issues"
    assert steps[3].step_id == "report"
    assert steps[4].step_id == "package"


def test_narration_dry_run_execute_success(loop_context):
    """Test successful Narration Dry Run execution."""
    loop = NarrationDryRunLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_narration_dry_run_select_path_step(loop_context):
    """Test select_path step."""
    loop = NarrationDryRunLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.paths_selected) > 0


def test_narration_dry_run_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = NarrationDryRunLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_narration_dry_run_issue_types():
    """Test all issue types are defined."""
    issue_types = NarrationDryRunLoop.ISSUE_TYPES

    assert len(issue_types) == 8
    assert "choice-ambiguity" in issue_types
    assert "gate-friction" in issue_types


def test_narration_dry_run_metadata_includes_paths(loop_context):
    """Test metadata includes paths tested."""
    loop = NarrationDryRunLoop(loop_context)

    result = loop.execute()

    assert "paths_tested" in result.metadata
    assert "issues_found" in result.metadata
