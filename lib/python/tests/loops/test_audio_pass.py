"""Tests for Audio Pass loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.audio_pass import AudioPassLoop
from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.models.artifact import Artifact
from questfoundry.roles.audio_director import AudioDirector
from questfoundry.roles.audio_producer import AudioProducer
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
            description="Test project for audio pass",
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
def loop_context_with_producer(temp_workspace, mock_provider, spec_path):
    """Fixture providing loop context with audio producer."""
    audio_director = AudioDirector(provider=mock_provider, spec_path=spec_path)
    audio_producer = AudioProducer(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="canon_pack",
            data={"sections": ["Opening", "Climax"], "scenes": ["A", "B"]},
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="audio_pass",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "audio_director": audio_director,
            "audio_producer": audio_producer,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
    )

    return context


@pytest.fixture
def loop_context_no_producer(temp_workspace, mock_provider, spec_path):
    """Fixture providing loop context without audio producer."""
    audio_director = AudioDirector(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="canon_pack",
            data={"sections": ["Opening", "Climax"]},
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="audio_pass",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "audio_director": audio_director,
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
    )

    return context


def test_audio_pass_initialization(loop_context_with_producer):
    """Test Audio Pass initialization."""
    loop = AudioPassLoop(loop_context_with_producer)

    assert loop.metadata.loop_id == "audio_pass"
    assert len(loop.steps) == 5
    assert loop.audio_producer_active


def test_audio_pass_metadata():
    """Test Audio Pass metadata."""
    metadata = AudioPassLoop.metadata

    assert metadata.loop_id == "audio_pass"
    assert "audio_director" in metadata.primary_roles


def test_audio_pass_steps():
    """Test Audio Pass step definitions."""
    steps = AudioPassLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "create_cuelist"
    assert steps[1].step_id == "review_cues"
    assert steps[2].step_id == "generate_audio"
    assert steps[3].step_id == "validate_quality"
    assert steps[4].step_id == "package_audio_plan"


def test_audio_pass_execute_with_producer(loop_context_with_producer):
    """Test execution with audio producer."""
    loop = AudioPassLoop(loop_context_with_producer)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_audio_pass_execute_without_producer(loop_context_no_producer):
    """Test execution without audio producer (plan-only)."""
    loop = AudioPassLoop(loop_context_no_producer)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 4
    assert not loop.audio_producer_active


def test_audio_pass_create_cuelist_step(loop_context_with_producer):
    """Test create_cuelist step."""
    loop = AudioPassLoop(loop_context_with_producer)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.cuelist) > 0


def test_audio_pass_generate_audio_skipped(loop_context_no_producer):
    """Test generate_audio step is skipped without producer."""
    loop = AudioPassLoop(loop_context_no_producer)

    # Execute the full loop - it will handle skipping generate_audio
    result = loop.execute()

    # Check that generate_audio step was skipped
    generate_audio_step = loop.steps[2]
    assert generate_audio_step.step_id == "generate_audio"
    assert generate_audio_step.status == StepStatus.SKIPPED
    assert result.success


def test_audio_pass_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = AudioPassLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert loop.validate_step(loop.steps[0], {"success": True, "skipped": True})
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_audio_pass_context_summary(loop_context_with_producer):
    """Test building loop context summary."""
    loop = AudioPassLoop(loop_context_with_producer)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Audio Pass" in summary


def test_audio_pass_metadata_includes_counts(loop_context_with_producer):
    """Test metadata includes counts."""
    loop = AudioPassLoop(loop_context_with_producer)

    result = loop.execute()

    assert "cuelist_count" in result.metadata
    assert "assets_generated" in result.metadata
