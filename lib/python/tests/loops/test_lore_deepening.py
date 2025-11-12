"""Tests for Lore Deepening loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.lore_deepening import LoreDeepeningLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.lore_weaver import LoreWeaver
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for lore deepening",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider with predefined responses."""
    return MockTextProvider(
        responses={
            "expand canon": (
                '{"entries": [{"title": "Canon Entry", '
                '"answer": "Canonical answer", '
                '"timeline": ["Event 1"], '
                '"causal_links": ["Cause 1"], '
                '"entities": ["Entity 1"], '
                '"constraints": ["Constraint 1"], '
                '"sensitivity": "spoiler-heavy", '
                '"player_safe_summary": "Safe summary", '
                '"downstream_impacts": {}}]}'
            ),
            "check canon consistency": (
                '{"is_consistent": true, "issues": [], "recommendations": []}'
            ),
            "generate player summary": "Player-safe summary text",
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
    lore_weaver = LoreWeaver(provider=mock_provider, spec_path=spec_path)

    # Create some accepted hook artifacts
    accepted_hooks = [
        Artifact(
            type="hook_card",
            data={
                "status": "accepted",
                "title": "Character backstory",
                "summary": "Need backstory for protagonist",
            },
            metadata={"created_by": "plotwright"},
        ),
        Artifact(
            type="hook_card",
            data={
                "status": "accepted",
                "title": "World history",
                "summary": "Establish timeline for key events",
            },
            metadata={"created_by": "lore_weaver"},
        ),
    ]

    context = LoopContext(
        loop_id="lore_deepening",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "lore_weaver": lore_weaver,
        },
        artifacts=accepted_hooks,
        project_metadata={"name": "Test Project", "genre": "fantasy"},
    )

    return context


def test_lore_deepening_initialization(loop_context):
    """Test Lore Deepening loop initialization."""
    loop = LoreDeepeningLoop(loop_context)

    assert loop.metadata.loop_id == "lore_deepening"
    assert loop.context == loop_context
    assert len(loop.steps) == 5
    assert len(loop.canon_questions) == 0
    assert len(loop.canon_entries) == 0


def test_lore_deepening_metadata():
    """Test Lore Deepening metadata."""
    metadata = LoreDeepeningLoop.metadata

    assert metadata.loop_id == "lore_deepening"
    assert metadata.display_name == "Lore Deepening"
    assert "lore_weaver" in metadata.primary_roles


def test_lore_deepening_steps():
    """Test Lore Deepening step definitions."""
    steps = LoreDeepeningLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "frame_questions"
    assert steps[1].step_id == "draft_canon"
    assert steps[2].step_id == "check_contradictions"
    assert steps[3].step_id == "create_impact_notes"
    assert steps[4].step_id == "package_canon"


def test_lore_deepening_execute_success(loop_context):
    """Test successful Lore Deepening execution."""
    loop = LoreDeepeningLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "lore_deepening"
    assert result.steps_completed == 5
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_lore_deepening_frame_questions_step(loop_context):
    """Test frame_questions step."""
    loop = LoreDeepeningLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.canon_questions) > 0


def test_lore_deepening_draft_canon_step(loop_context):
    """Test draft_canon step."""
    loop = LoreDeepeningLoop(loop_context)

    # Run frame_questions first
    loop.execute_step(loop.steps[0])

    # Run draft_canon
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.canon_entries) > 0


def test_lore_deepening_check_contradictions_step(loop_context):
    """Test check_contradictions step."""
    loop = LoreDeepeningLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run check_contradictions
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_lore_deepening_create_impact_notes_step(loop_context):
    """Test create_impact_notes step."""
    loop = LoreDeepeningLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run create_impact_notes
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_lore_deepening_package_canon_step(loop_context):
    """Test package_canon step."""
    loop = LoreDeepeningLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run package_canon
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_lore_deepening_artifacts_created(loop_context):
    """Test that Lore Deepening creates expected artifacts."""
    loop = LoreDeepeningLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert len(result.artifacts_created) >= 5


def test_lore_deepening_canon_entries(loop_context):
    """Test canon entry creation."""
    loop = LoreDeepeningLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.metadata.get("canon_entries_created", 0) > 0


def test_lore_deepening_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = LoreDeepeningLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_lore_deepening_context_summary(loop_context):
    """Test building loop context summary."""
    loop = LoreDeepeningLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Lore Deepening" in summary
    assert "lore_weaver" in summary.lower()


def test_lore_deepening_repr(loop_context):
    """Test loop string representation."""
    loop = LoreDeepeningLoop(loop_context)

    repr_str = repr(loop)

    assert "LoreDeepeningLoop" in repr_str
    assert "lore_deepening" in repr_str
