"""Tests for Translation Pass loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.translation_pass import TranslationPassLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.showrunner import Showrunner
from questfoundry.roles.style_lead import StyleLead
from questfoundry.roles.translator import Translator
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for translation pass",
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
    translator = Translator(provider=mock_provider, spec_path=spec_path)
    style_lead = StyleLead(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="canon_pack",
            data={"sections": ["Section 1", "Section 2"]},
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="translation_pass",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "translator": translator,
            "style_lead": style_lead,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project"},
        config={"target_language": "nl"},
    )

    return context


def test_translation_pass_initialization(loop_context):
    """Test Translation Pass initialization."""
    loop = TranslationPassLoop(loop_context)

    assert loop.metadata.loop_id == "translation_pass"
    assert len(loop.steps) == 5
    assert loop.target_language == "nl"


def test_translation_pass_metadata():
    """Test Translation Pass metadata."""
    metadata = TranslationPassLoop.metadata

    assert metadata.loop_id == "translation_pass"
    assert "translator" in metadata.primary_roles


def test_translation_pass_steps():
    """Test Translation Pass step definitions."""
    steps = TranslationPassLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "extract_strings"
    assert steps[1].step_id == "translate"
    assert steps[2].step_id == "validate"
    assert steps[3].step_id == "style_check"
    assert steps[4].step_id == "package_language_pack"


def test_translation_pass_execute_success(loop_context):
    """Test successful Translation Pass execution."""
    loop = TranslationPassLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.steps_completed == 5


def test_translation_pass_extract_strings_step(loop_context):
    """Test extract_strings step."""
    loop = TranslationPassLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.strings_extracted) > 0


def test_translation_pass_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = TranslationPassLoop(loop_context_mock)

    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_translation_pass_metadata_includes_language(loop_context):
    """Test metadata includes target language."""
    loop = TranslationPassLoop(loop_context)

    result = loop.execute()

    assert "target_language" in result.metadata
    assert result.metadata["target_language"] == "nl"
