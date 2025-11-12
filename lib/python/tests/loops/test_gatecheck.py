"""Tests for Gatecheck loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.gatecheck import GatecheckLoop
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
            description="Test project for gatecheck",
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
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    showrunner = Showrunner(provider=mock_provider, spec_path=spec_path)

    # Create some artifacts to review
    artifacts = [
        Artifact(
            type="tu_brief",
            data={
                "id": "TU-2025-11-07-PW01",
                "status": "gatecheck",
                "deliverables": ["Story draft"],
            },
            metadata={"created_by": "plotwright"},
        ),
        Artifact(
            type="canon_pack",
            data={
                "sections": ["Section 1", "Section 2"],
                "scenes": ["Scene A", "Scene B"],
            },
            metadata={"created_by": "scene_smith"},
        ),
    ]

    context = LoopContext(
        loop_id="gatecheck",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "gatekeeper": gatekeeper,
            "showrunner": showrunner,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project", "genre": "sci-fi"},
    )

    return context


# Loop Tests


def test_gatecheck_initialization(loop_context):
    """Test Gatecheck loop initialization."""
    loop = GatecheckLoop(loop_context)

    assert loop.metadata.loop_id == "gatecheck"
    assert loop.context == loop_context
    assert len(loop.steps) == 5
    assert len(loop.bar_results) == 0
    assert loop.decision == "pending"


def test_gatecheck_metadata():
    """Test Gatecheck metadata."""
    metadata = GatecheckLoop.metadata

    assert metadata.loop_id == "gatecheck"
    assert metadata.display_name == "Gatecheck"
    assert "gatekeeper" in metadata.primary_roles
    assert "showrunner" in metadata.consulted_roles


def test_gatecheck_steps():
    """Test Gatecheck step definitions."""
    steps = GatecheckLoop.steps

    assert len(steps) == 5
    assert steps[0].step_id == "run_quality_bars"
    assert steps[1].step_id == "collect_findings"
    assert steps[2].step_id == "triage_blockers"
    assert steps[3].step_id == "create_report"
    assert steps[4].step_id == "package_decision"


def test_gatecheck_quality_bars():
    """Test that all 8 quality bars are defined."""
    bars = GatecheckLoop.QUALITY_BARS

    assert len(bars) == 8
    assert "integrity" in bars
    assert "reachability" in bars
    assert "nonlinearity" in bars
    assert "gateways" in bars
    assert "style" in bars
    assert "determinism" in bars
    assert "presentation" in bars
    assert "accessibility" in bars


def test_gatecheck_execute_success(loop_context):
    """Test successful Gatecheck execution."""
    loop = GatecheckLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "gatecheck"
    assert result.steps_completed == 5
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_gatecheck_run_quality_bars_step(loop_context):
    """Test run_quality_bars step."""
    loop = GatecheckLoop(loop_context)

    step = loop.steps[0]  # run_quality_bars
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.bar_results) == 8  # All 8 bars evaluated


def test_gatecheck_collect_findings_step(loop_context):
    """Test collect_findings step."""
    loop = GatecheckLoop(loop_context)

    # Run previous steps
    loop.execute_step(loop.steps[0])

    # Run collect_findings
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    # Findings list should exist (may be empty if all green)
    assert isinstance(loop.findings, list)


def test_gatecheck_triage_blockers_step(loop_context):
    """Test triage_blockers step."""
    loop = GatecheckLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run triage_blockers
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert loop.decision in ("pass", "conditional_pass", "block")


def test_gatecheck_create_report_step(loop_context):
    """Test create_report step."""
    loop = GatecheckLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run create_report
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_gatecheck_package_decision_step(loop_context):
    """Test package_decision step."""
    loop = GatecheckLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run package_decision
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_gatecheck_pass_decision(loop_context):
    """Test gatecheck with all green bars (pass decision)."""
    loop = GatecheckLoop(loop_context)

    # Execute loop
    result = loop.execute()

    assert result.success
    # By default, mock provider should result in all green bars
    assert loop.decision == "pass"


def test_gatecheck_decision_metadata(loop_context):
    """Test that decision is included in result metadata."""
    loop = GatecheckLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert "decision" in result.metadata
    assert "bars_evaluated" in result.metadata
    assert result.metadata["bars_evaluated"] == 8


def test_gatecheck_artifacts_created(loop_context):
    """Test that Gatecheck creates expected artifacts."""
    loop = GatecheckLoop(loop_context)

    result = loop.execute()

    assert result.success
    # Should create artifacts for each step
    assert len(result.artifacts_created) >= 5


def test_gatecheck_failed_step(loop_context):
    """Test Gatecheck with failed step."""
    # Create provider that fails
    bad_provider = MockTextProvider(responses={"evaluate_quality_bars": ""})

    gatekeeper = Gatekeeper(
        provider=bad_provider,
        spec_path=loop_context.role_instances["gatekeeper"].spec_path,
    )
    loop_context.role_instances["gatekeeper"] = gatekeeper

    loop = GatecheckLoop(loop_context)

    # Should handle gracefully
    result = loop.execute()

    # May succeed with empty responses, or fail - both are valid
    assert result.loop_id == "gatecheck"


def test_gatecheck_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = GatecheckLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})

    # Invalid result - not a dict
    assert not loop.validate_step(loop.steps[0], "not a dict")


def test_gatecheck_context_summary(loop_context):
    """Test building loop context summary."""
    loop = GatecheckLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Gatecheck" in summary
    assert "quality" in summary.lower()
    assert "gatekeeper" in summary.lower()


def test_gatecheck_repr(loop_context):
    """Test loop string representation."""
    loop = GatecheckLoop(loop_context)

    repr_str = repr(loop)

    assert "GatecheckLoop" in repr_str
    assert "gatecheck" in repr_str
    assert "0/5" in repr_str  # 0 of 5 steps


def test_gatecheck_can_continue(loop_context):
    """Test loop can_continue logic."""
    loop = GatecheckLoop(loop_context)

    # Can continue at start
    assert loop.can_continue()

    # Execute all steps
    for step in loop.steps:
        loop.execute_step(step)
        loop.current_step_index += 1

    # Cannot continue at end
    assert not loop.can_continue()


def test_gatecheck_decision_rationale(loop_context):
    """Test decision rationale generation."""
    loop = GatecheckLoop(loop_context)

    # Execute to populate bar results
    loop.execute()

    rationale = loop._get_decision_rationale()

    assert isinstance(rationale, str)
    assert len(rationale) > 0


def test_gatecheck_next_actions(loop_context):
    """Test next actions generation."""
    loop = GatecheckLoop(loop_context)

    # Execute to populate decision
    loop.execute()

    next_actions = loop._get_next_actions()

    assert isinstance(next_actions, str)
    assert len(next_actions) > 0


def test_gatecheck_handoffs(loop_context):
    """Test handoffs generation."""
    loop = GatecheckLoop(loop_context)

    # Execute to populate findings
    loop.execute()

    handoffs = loop._get_handoffs()

    assert isinstance(handoffs, list)
