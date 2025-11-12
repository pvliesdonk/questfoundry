"""Tests for role system."""

from pathlib import Path

import pytest
from conftest import MockTextProvider

from questfoundry.models.artifact import Artifact
from questfoundry.roles.base import RoleContext
from questfoundry.roles.gatekeeper import Gatekeeper
from questfoundry.roles.plotwright import Plotwright
from questfoundry.roles.scene_smith import SceneSmith


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider."""
    return MockTextProvider()


@pytest.fixture
def spec_path():
    """Fixture providing path to spec directory."""
    # Find spec relative to test file
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent
    return repo_root / "spec"


@pytest.fixture
def sample_context():
    """Fixture providing a sample role context."""
    return RoleContext(
        task="test_task",
        artifacts=[
            Artifact(
                type="hook_card",
                data={"title": "Test Hook", "description": "A test hook for testing"},
                metadata={"id": "HOOK-001"},
            )
        ],
        project_metadata={"name": "Test Project", "genre": "fantasy"},
    )


# Base Role Tests


def test_role_initialization(mock_provider, spec_path):
    """Test role initialization."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    assert plotwright.provider == mock_provider
    assert plotwright.spec_path == spec_path
    assert plotwright.role_name == "plotwright"
    assert plotwright.display_name == "Plotwright"


def test_role_load_brief(mock_provider, spec_path):
    """Test loading role brief from spec."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    brief = plotwright.load_brief()

    assert isinstance(brief, str)
    assert len(brief) > 0
    assert "Plotwright" in brief or "plotwright" in brief.lower()


def test_role_load_brief_missing_spec(mock_provider):
    """Test error when spec not found."""
    plotwright = Plotwright(provider=mock_provider, spec_path=Path("/nonexistent"))

    with pytest.raises(FileNotFoundError):
        plotwright.load_brief()


def test_role_extract_section(mock_provider, spec_path):
    """Test extracting sections from markdown."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    content = """
# Document

## 1) Operating principles

- Principle 1
- Principle 2

## 2) Inputs & outputs

Input and output info

## 3) Another section

More content
"""

    principles = plotwright.extract_section(content, "Operating principles")
    assert "Principle 1" in principles
    assert "Principle 2" in principles

    inputs = plotwright.extract_section(content, "Inputs & outputs")
    assert "Input and output" in inputs

    missing = plotwright.extract_section(content, "Nonexistent")
    assert missing == ""


def test_role_build_system_prompt(mock_provider, spec_path, sample_context):
    """Test building system prompt."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    prompt = plotwright.build_system_prompt(sample_context)

    assert isinstance(prompt, str)
    assert "Plotwright" in prompt
    assert len(prompt) > 0


def test_role_format_artifacts(mock_provider, spec_path):
    """Test formatting artifacts for prompts."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    artifacts = [
        Artifact(
            type="hook_card",
            data={"title": "Hook 1", "tags": ["test"]},
            metadata={"id": "HOOK-001"},
        ),
        Artifact(
            type="tu_brief",
            data={"title": "TU 1", "goal": "Test goal"},
            metadata={"id": "TU-001"},
        ),
    ]

    formatted = plotwright.format_artifacts(artifacts)

    assert "Hook 1" in formatted
    assert "TU 1" in formatted
    assert "hook_card" in formatted
    assert "tu_brief" in formatted


def test_role_format_artifacts_empty(mock_provider, spec_path):
    """Test formatting empty artifact list."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    formatted = plotwright.format_artifacts([])

    assert "No artifacts" in formatted


def test_role_build_user_prompt(mock_provider, spec_path, sample_context):
    """Test building user prompt."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    prompt = plotwright.build_user_prompt(sample_context)

    assert "test_task" in prompt
    assert "Test Project" in prompt
    assert "Test Hook" in prompt


# Plotwright Tests


def test_plotwright_generate_hooks(mock_provider, spec_path):
    """Test Plotwright hook generation."""
    # Set up mock to return valid JSON
    mock_provider.response = """
{
  "hooks": [
    {"title": "Hook 1", "summary": "First hook", "tags": ["fantasy"]},
    {"title": "Hook 2", "summary": "Second hook", "tags": ["adventure"]}
  ]
}
"""

    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="generate_hooks",
        project_metadata={"name": "Test", "genre": "fantasy"},
    )

    result = plotwright.execute_task(context)

    assert result.success
    assert "hooks" in result.metadata
    assert len(result.metadata["hooks"]) == 2


def test_plotwright_create_tu_brief(mock_provider, spec_path):
    """Test Plotwright TU brief creation."""
    mock_provider.response = "TU Brief content here..."

    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="create_tu_brief",
        artifacts=[
            Artifact(
                type="hook_card",
                data={"title": "Test Hook", "description": "Hook description"},
                metadata={"id": "HOOK-001"},
            )
        ],
    )

    result = plotwright.execute_task(context)

    assert result.success
    assert result.metadata.get("content_type") == "tu_brief"
    assert len(result.output) > 0


def test_plotwright_unknown_task(mock_provider, spec_path):
    """Test Plotwright with unknown task."""
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(task="unknown_task")

    result = plotwright.execute_task(context)

    assert not result.success
    assert "Unknown task" in result.error


# Gatekeeper Tests


def test_gatekeeper_pre_gate(mock_provider, spec_path):
    """Test Gatekeeper pre-gate check."""
    mock_provider.response = """
{
  "status": "warning",
  "blockers": ["Issue 1"],
  "quick_wins": ["Fix 1"],
  "review_needed": ["Area 1"]
}
"""

    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="pre_gate",
        artifacts=[
            Artifact(
                type="tu_brief",
                data={"title": "Test TU", "content": "TU content"},
                metadata={"id": "TU-001"},
            )
        ],
    )

    result = gatekeeper.execute_task(context)

    assert result.success
    assert result.metadata.get("check_type") == "pre_gate"
    assert result.metadata.get("status") == "warning"


def test_gatekeeper_gate_check(mock_provider, spec_path):
    """Test Gatekeeper full gate check."""
    mock_provider.response = """
{
  "overall_status": "pass",
  "merge_safe": true,
  "bars": {
    "integrity": {"status": "pass", "issues": [], "fixes": []},
    "reachability": {"status": "pass", "issues": [], "fixes": []}
  }
}
"""

    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="gate_check",
        artifacts=[
            Artifact(
                type="canon_pack",
                data={"title": "Test Canon", "content": "Canon content"},
                metadata={"id": "CANON-001"},
            )
        ],
    )

    result = gatekeeper.execute_task(context)

    assert result.success
    assert result.metadata.get("overall_status") == "pass"
    assert result.metadata.get("merge_safe") is True


# Scene Smith Tests


def test_scene_smith_draft_scene(mock_provider, spec_path):
    """Test Scene Smith scene drafting."""
    mock_provider.response = "Scene content here..."

    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="draft_scene",
        artifacts=[
            Artifact(
                type="tu_brief",
                data={
                    "title": "Section Brief",
                    "goal": "Test goal",
                    "beats": ["Beat 1", "Beat 2"],
                },
                metadata={"id": "BRIEF-001"},
            )
        ],
    )

    result = scene_smith.execute_task(context)

    assert result.success
    assert result.metadata.get("content_type") == "scene_draft"
    assert len(result.output) > 0


def test_scene_smith_draft_choices(mock_provider, spec_path):
    """Test Scene Smith choice drafting."""
    mock_provider.response = "Choice options here..."

    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="draft_choices",
        artifacts=[
            Artifact(
                type="canon_pack",
                data={"title": "Scene Context", "situation": "Test situation"},
                metadata={"id": "CANON-001"},
            )
        ],
    )

    result = scene_smith.execute_task(context)

    assert result.success
    assert result.metadata.get("content_type") == "choice_draft"


def test_scene_smith_write_gate_scene(mock_provider, spec_path):
    """Test Scene Smith gateway scene writing."""
    mock_provider.response = "Gateway scene content..."

    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)

    context = RoleContext(
        task="write_gate_scene",
        additional_context={"gate_type": "knowledge_check"},
    )

    result = scene_smith.execute_task(context)

    assert result.success
    assert result.metadata.get("content_type") == "gate_scene"
    assert result.metadata.get("gate_type") == "knowledge_check"


# Integration Tests


def test_role_context_complete_workflow(mock_provider, spec_path):
    """Test a complete workflow through multiple roles."""
    # 1. Plotwright generates hooks
    plotwright = Plotwright(provider=mock_provider, spec_path=spec_path)
    mock_provider.response = '{"hooks": [{"title": "H1", "summary": "S1", "tags": []}]}'

    hook_context = RoleContext(
        task="generate_hooks",
        project_metadata={"name": "Test"},
    )

    hook_result = plotwright.execute_task(hook_context)
    assert hook_result.success

    # 2. Scene Smith drafts scene
    scene_smith = SceneSmith(provider=mock_provider, spec_path=spec_path)
    mock_provider.response = "Scene content"

    hook_data = hook_result.metadata.get("hooks", [])[0]
    scene_context = RoleContext(
        task="draft_scene",
        artifacts=[
            Artifact(
                type="hook_card",
                data=hook_data,
                metadata={"id": "HOOK-001"},
            )
        ],
    )

    scene_result = scene_smith.execute_task(scene_context)
    assert scene_result.success

    # 3. Gatekeeper checks
    gatekeeper = Gatekeeper(provider=mock_provider, spec_path=spec_path)
    mock_provider.response = (
        '{"status": "pass", "blockers": [], "quick_wins": [], "review_needed": []}'
    )

    gate_context = RoleContext(
        task="pre_gate",
        artifacts=[
            Artifact(
                type="canon_pack",
                data={"title": "Scene", "content": scene_result.output},
                metadata={"id": "CANON-001"},
            )
        ],
    )

    gate_result = gatekeeper.execute_task(gate_context)
    assert gate_result.success
