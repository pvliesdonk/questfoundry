"""Tests for Style Tune-Up loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.style_tune_up import StyleTuneUpLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.showrunner import Showrunner
from questfoundry.roles.style_lead import StyleLead
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for style tune-up",
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
    """Fixture providing a loop context with roles."""
    # Create role instances
    style_lead = StyleLead(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)

    # Create some manuscript sections with style drift
    sections = [
        Artifact(
            type="manuscript_section",
            data={
                "title": "Section 1",
                "text": "Some text with inconsistent style",
            },
            metadata={"created_by": "scene_smith"},
        ),
        Artifact(
            type="manuscript_section",
            data={
                "title": "Section 2",
                "text": "More text with tone wobble",
            },
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="style_tune_up",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "style_lead": style_lead,
            "showrunner": showrunner,
            "gatekeeper": gatekeeper,
        },
        artifacts=sections,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
    )

    return context


# Loop Tests


def test_style_tune_up_initialization(loop_context):
    """Test Style Tune-Up loop initialization."""
    loop = StyleTuneUpLoop(loop_context)

    assert loop.metadata.loop_id == "style_tune_up"
    assert loop.context == loop_context
    assert len(loop.steps) == 6
    assert len(loop.drift_issues) == 0
    assert not loop.addendum_created


def test_style_tune_up_metadata():
    """Test Style Tune-Up metadata."""
    metadata = StyleTuneUpLoop.metadata

    assert metadata.loop_id == "style_tune_up"
    assert metadata.display_name == "Style Tune-Up"
    assert "style_lead" in metadata.primary_roles
    assert "gatekeeper" in metadata.consulted_roles


def test_style_tune_up_steps():
    """Test Style Tune-Up step definitions."""
    steps = StyleTuneUpLoop.steps

    assert len(steps) == 6
    assert steps[0].step_id == "diagnose_drift"
    assert steps[1].step_id == "create_addendum"
    assert steps[2].step_id == "generate_edit_notes"
    assert steps[3].step_id == "apply_revisions"
    assert steps[4].step_id == "pre_gate_validation"
    assert steps[5].step_id == "package_results"


def test_style_tune_up_execute_success(loop_context):
    """Test successful Style Tune-Up execution."""
    loop = StyleTuneUpLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "style_tune_up"
    assert result.steps_completed == 6
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_style_tune_up_diagnose_drift_step(loop_context):
    """Test diagnose_drift step."""
    loop = StyleTuneUpLoop(loop_context)

    step = loop.steps[0]  # diagnose_drift
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    # drift_issues should be populated (even if empty from mock)
    assert isinstance(loop.drift_issues, list)


def test_style_tune_up_create_addendum_step(loop_context):
    """Test create_addendum step."""
    loop = StyleTuneUpLoop(loop_context)

    # Run diagnose_drift first
    loop.execute_step(loop.steps[0])

    # Run create_addendum
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert loop.addendum_created


def test_style_tune_up_generate_edit_notes_step(loop_context):
    """Test generate_edit_notes step."""
    loop = StyleTuneUpLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run generate_edit_notes
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert loop.edit_notes_count >= 0


def test_style_tune_up_apply_revisions_step(loop_context):
    """Test apply_revisions step."""
    loop = StyleTuneUpLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run apply_revisions
    step = loop.steps[3]
    loop.execute_step(step)

    # Should succeed even without scene_smith
    assert step.status == StepStatus.COMPLETED


def test_style_tune_up_pre_gate_validation_step(loop_context):
    """Test pre_gate_validation step."""
    loop = StyleTuneUpLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run pre_gate_validation
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_style_tune_up_package_results_step(loop_context):
    """Test package_results step."""
    loop = StyleTuneUpLoop(loop_context)

    # Run previous steps
    for i in range(5):
        loop.execute_step(loop.steps[i])

    # Run package_results
    step = loop.steps[5]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_style_tune_up_artifacts_created(loop_context):
    """Test that Style Tune-Up creates expected artifacts."""
    loop = StyleTuneUpLoop(loop_context)

    result = loop.execute()

    assert result.success
    # Should create: drift_diagnosis, style_addendum, edit_notes, gate_report, tu_brief
    assert len(result.artifacts_created) >= 5


def test_style_tune_up_metadata_tracking(loop_context):
    """Test metadata tracking in result."""
    loop = StyleTuneUpLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert "drift_issues_found" in result.metadata
    assert "addendum_created" in result.metadata
    assert "edit_notes_count" in result.metadata


def test_style_tune_up_failed_step(loop_context):
    """Test Style Tune-Up with failed step."""
    # Create provider that fails
    bad_provider = MockTextProvider(responses={"diagnose_drift": ""})

    # Create a style_lead that will fail
    spec_path = loop_context.role_instances["style_lead"].spec_path
    style_lead = StyleLead(provider=bad_provider, spec_path=spec_path)
    loop_context.role_instances["style_lead"] = style_lead

    loop = StyleTuneUpLoop(loop_context)

    # Should handle gracefully
    result = loop.execute()

    # May succeed with empty responses, or fail - both are valid
    assert result.loop_id == "style_tune_up"


def test_style_tune_up_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = StyleTuneUpLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})

    # Invalid result - not a dict
    assert not loop.validate_step(loop.steps[0], "not a dict")


def test_style_tune_up_context_summary(loop_context):
    """Test building loop context summary."""
    loop = StyleTuneUpLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Style Tune-Up" in summary
    assert "style" in summary.lower()


def test_style_tune_up_repr(loop_context):
    """Test loop string representation."""
    loop = StyleTuneUpLoop(loop_context)

    repr_str = repr(loop)

    assert "StyleTuneUpLoop" in repr_str
    assert "style_tune_up" in repr_str
    assert "0/6" in repr_str  # 0 of 6 steps


def test_style_tune_up_can_continue(loop_context):
    """Test loop can_continue logic."""
    loop = StyleTuneUpLoop(loop_context)

    # Can continue at start
    assert loop.can_continue()

    # Execute all steps
    for step in loop.steps:
        loop.execute_step(step)
        loop.current_step_index += 1

    # Cannot continue at end
    assert not loop.can_continue()


def test_style_tune_up_empty_sections(temp_workspace, mock_provider, spec_path):
    """Test Style Tune-Up with no manuscript sections."""
    style_lead = StyleLead(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)

    # Context with no sections
    context = LoopContext(
        loop_id="style_tune_up",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "style_lead": style_lead,
            "showrunner": showrunner,
            "gatekeeper": gatekeeper,
        },
        artifacts=[],  # No sections
        project_metadata={"name": "Test Project"},
    )

    loop = StyleTuneUpLoop(context)
    result = loop.execute()

    # Should still succeed
    assert result.success
    assert result.metadata.get("drift_issues_found", 0) == 0
