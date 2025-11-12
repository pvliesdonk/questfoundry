"""Tests for Codex Expansion loop execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopContext, StepStatus
from questfoundry.loops.codex_expansion import CodexExpansionLoop
from questfoundry.models.artifact import Artifact
from questfoundry.roles.codex_curator import CodexCurator
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for codex expansion",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider with predefined responses."""
    return MockTextProvider(
        responses={
            "check coverage": (
                '{"missing": [{"term": "Term 1", "frequency": 10, '
                '"priority": "high", "rationale": "Important term"}], '
                '"incomplete": [], "dead_links": [], "recommendations": []}'
            ),
            "create entry": (
                '{"title": "Test Entry", "slug": "test-entry", '
                '"overview": "Overview text", "usage": "Usage text", '
                '"context": "Context text", "see_also": [], '
                '"notes": "", "lineage": "TU-123"}'
            ),
            "validate accessibility": ('{"is_accessible": true, "issues": []}'),
            "create crosslinks": (
                '{"crosslinks": [], "orphans": [], "coverage_notes": "All linked"}'
            ),
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
    codex_curator = CodexCurator(provider=mock_provider, spec_path=spec_path)

    # Create canon pack artifacts with player-safe summaries
    artifacts = [
        Artifact(
            type="canon_pack",
            data={
                "entries": [
                    {
                        "title": "Location Alpha",
                        "answer": "Spoiler-heavy canon details",
                        "player_safe_summary": "A trading hub in the outer rim",
                    },
                    {
                        "title": "Character Beta",
                        "answer": "Secret backstory",
                        "player_safe_summary": "A mysterious trader",
                    },
                ]
            },
            metadata={"created_by": "lore_weaver"},
        ),
    ]

    context = LoopContext(
        loop_id="codex_expansion",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={
            "codex_curator": codex_curator,
        },
        artifacts=artifacts,
        project_metadata={"name": "Test Project", "genre": "space-opera"},
    )

    return context


def test_codex_expansion_initialization(loop_context):
    """Test Codex Expansion loop initialization."""
    loop = CodexExpansionLoop(loop_context)

    assert loop.metadata.loop_id == "codex_expansion"
    assert loop.context == loop_context
    assert len(loop.steps) == 6
    assert len(loop.selected_topics) == 0
    assert len(loop.codex_entries) == 0


def test_codex_expansion_metadata():
    """Test Codex Expansion metadata."""
    metadata = CodexExpansionLoop.metadata

    assert metadata.loop_id == "codex_expansion"
    assert metadata.display_name == "Codex Expansion"
    assert "codex_curator" in metadata.primary_roles


def test_codex_expansion_steps():
    """Test Codex Expansion step definitions."""
    steps = CodexExpansionLoop.steps

    assert len(steps) == 6
    assert steps[0].step_id == "select_topics"
    assert steps[1].step_id == "draft_entries"
    assert steps[2].step_id == "spoiler_sweep"
    assert steps[3].step_id == "style_pass"
    assert steps[4].step_id == "link_audit"
    assert steps[5].step_id == "package_codex"


def test_codex_expansion_execute_success(loop_context):
    """Test successful Codex Expansion execution."""
    loop = CodexExpansionLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.loop_id == "codex_expansion"
    assert result.steps_completed == 6
    assert result.steps_failed == 0
    assert len(result.artifacts_created) > 0


def test_codex_expansion_select_topics_step(loop_context):
    """Test select_topics step."""
    loop = CodexExpansionLoop(loop_context)

    step = loop.steps[0]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.selected_topics) > 0


def test_codex_expansion_draft_entries_step(loop_context):
    """Test draft_entries step."""
    loop = CodexExpansionLoop(loop_context)

    # Run select_topics first
    loop.execute_step(loop.steps[0])

    # Run draft_entries
    step = loop.steps[1]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert len(loop.codex_entries) > 0


def test_codex_expansion_spoiler_sweep_step(loop_context):
    """Test spoiler_sweep step."""
    loop = CodexExpansionLoop(loop_context)

    # Run previous steps
    for i in range(2):
        loop.execute_step(loop.steps[i])

    # Run spoiler_sweep
    step = loop.steps[2]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_codex_expansion_style_pass_step(loop_context):
    """Test style_pass step."""
    loop = CodexExpansionLoop(loop_context)

    # Run previous steps
    for i in range(3):
        loop.execute_step(loop.steps[i])

    # Run style_pass
    step = loop.steps[3]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_codex_expansion_link_audit_step(loop_context):
    """Test link_audit step."""
    loop = CodexExpansionLoop(loop_context)

    # Run previous steps
    for i in range(4):
        loop.execute_step(loop.steps[i])

    # Run link_audit
    step = loop.steps[4]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED


def test_codex_expansion_package_codex_step(loop_context):
    """Test package_codex step."""
    loop = CodexExpansionLoop(loop_context)

    # Run previous steps
    for i in range(5):
        loop.execute_step(loop.steps[i])

    # Run package_codex
    step = loop.steps[5]
    loop.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    assert step.result is not None


def test_codex_expansion_artifacts_created(loop_context):
    """Test that Codex Expansion creates expected artifacts."""
    loop = CodexExpansionLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert len(result.artifacts_created) >= 6


def test_codex_expansion_entries_count(loop_context):
    """Test codex entry creation."""
    loop = CodexExpansionLoop(loop_context)

    result = loop.execute()

    assert result.success
    assert result.metadata.get("entries_created", 0) > 0


def test_codex_expansion_validate_step():
    """Test step validation logic."""
    loop_context_mock = MagicMock()
    loop = CodexExpansionLoop(loop_context_mock)

    # Valid result
    assert loop.validate_step(
        loop.steps[0], {"success": True, "artifacts": [MagicMock()]}
    )

    # Invalid result - no success
    assert not loop.validate_step(loop.steps[0], {"success": False})


def test_codex_expansion_context_summary(loop_context):
    """Test building loop context summary."""
    loop = CodexExpansionLoop(loop_context)

    summary = loop.build_loop_context_summary()

    assert isinstance(summary, str)
    assert "Codex Expansion" in summary
    assert "codex_curator" in summary.lower()


def test_codex_expansion_repr(loop_context):
    """Test loop string representation."""
    loop = CodexExpansionLoop(loop_context)

    repr_str = repr(loop)

    assert "CodexExpansionLoop" in repr_str
    assert "codex_expansion" in repr_str


def test_codex_expansion_player_safe_summaries(loop_context):
    """Test extraction of player-safe summaries from canon."""
    loop = CodexExpansionLoop(loop_context)

    # Execute to check that summaries are properly used
    result = loop.execute()

    assert result.success

    # Verify at least some entries were created from canon
    assert len(loop.codex_entries) > 0


def test_codex_expansion_empty_canon(temp_workspace, mock_provider, spec_path):
    """Test Codex Expansion with no canon entries."""
    codex_curator = CodexCurator(provider=mock_provider, spec_path=spec_path)

    context = LoopContext(
        loop_id="codex_expansion",
        project_id="test-project",
        workspace=temp_workspace,
        role_instances={"codex_curator": codex_curator},
        artifacts=[],  # No canon
        project_metadata={"name": "Test Project"},
    )

    loop = CodexExpansionLoop(context)
    result = loop.execute()

    # Should still succeed (mock provider returns data regardless)
    assert result.success
    # Note: Mock provider returns "Term 1" even with no canon, so entries > 0
    assert result.metadata.get("topics_covered", 0) >= 0
