"""Tests for orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockTextProvider

from questfoundry.loops.base import LoopResult
from questfoundry.models.artifact import Artifact
from questfoundry.orchestrator import Orchestrator
from questfoundry.state.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace():
    """Fixture providing a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = WorkspaceManager(tmpdir)
        workspace.init_workspace(
            name="Test Project",
            description="Test project for orchestrator",
        )
        yield workspace


@pytest.fixture
def mock_provider():
    """Fixture providing a mock text provider."""
    return MockTextProvider()


@pytest.fixture
def spec_path():
    """Fixture providing path to spec directory."""
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent
    return repo_root / "spec"


@pytest.fixture
def orchestrator(temp_workspace, spec_path):
    """Fixture providing an orchestrator instance."""
    orch = Orchestrator(
        workspace=temp_workspace,
        spec_path=spec_path,
    )
    return orch


# Orchestrator Initialization Tests


def test_orchestrator_initialization(temp_workspace, spec_path):
    """Test orchestrator initialization."""
    orch = Orchestrator(
        workspace=temp_workspace,
        spec_path=spec_path,
    )

    assert orch.workspace == temp_workspace
    assert orch.spec_path == spec_path
    assert orch.showrunner is None
    assert orch.provider_registry is not None
    assert orch.role_registry is not None
    assert orch.loop_registry is not None


def test_orchestrator_initialize_with_provider(orchestrator, mock_provider):
    """Test orchestrator initialization with provider."""
    orchestrator.initialize(provider=mock_provider)

    assert orchestrator.showrunner is not None
    assert orchestrator.showrunner.role_name == "showrunner"


def test_orchestrator_repr(orchestrator):
    """Test orchestrator string representation."""
    repr_str = repr(orchestrator)

    assert "Orchestrator" in repr_str
    assert "not initialized" in repr_str

    orchestrator.initialize(provider=MockTextProvider())
    repr_str = repr(orchestrator)

    assert "initialized" in repr_str
    assert "loops=" in repr_str
    assert "roles=" in repr_str


# Loop Selection Tests


def test_select_loop_not_initialized(orchestrator):
    """Test selecting loop without initialization raises error."""
    with pytest.raises(RuntimeError, match="not initialized"):
        orchestrator.select_loop("Create a new quest")


def test_select_loop_success(orchestrator, mock_provider):
    """Test successful loop selection."""
    orchestrator.initialize(provider=mock_provider)

    loop_id = orchestrator.select_loop(
        goal="Create a new quest with branching narrative",
        project_state={"phase": "initial"},
    )

    assert loop_id == "story_spark"


def test_select_loop_with_artifacts(orchestrator, mock_provider):
    """Test loop selection with existing artifacts."""
    orchestrator.initialize(provider=mock_provider)

    artifacts = [
        Artifact(
            type="hook_card",
            data={"title": "Existing Hook", "description": "Test"},
            metadata={"id": "HOOK-001"},
        )
    ]

    loop_id = orchestrator.select_loop(
        goal="Expand on existing hooks",
        artifacts=artifacts,
    )

    assert loop_id == "story_spark"


def test_extract_loop_id_patterns(orchestrator, mock_provider):
    """Test loop ID extraction from various output patterns."""
    orchestrator.initialize(provider=mock_provider)

    # Test various output patterns
    test_cases = [
        ("Selected Loop: story_spark", "story_spark"),
        ("**Selected Loop**: story_spark", "story_spark"),
        ("Loop ID: story_spark", "story_spark"),
        ("The story_spark loop is best for this", "story_spark"),
    ]

    for output, expected in test_cases:
        loop_id = orchestrator._extract_loop_id(output)
        assert loop_id == expected


def test_extract_loop_id_failure(orchestrator, mock_provider):
    """Test loop ID extraction failure."""
    orchestrator.initialize(provider=mock_provider)

    with pytest.raises(RuntimeError, match="Could not extract loop ID"):
        orchestrator._extract_loop_id("Some output without a valid loop")


# Loop Execution Tests


def test_execute_loop_story_spark(orchestrator, mock_provider):
    """Test executing Story Spark loop."""
    orchestrator.initialize(provider=mock_provider)

    result = orchestrator.execute_loop(
        loop_id="story_spark",
        project_id="test-project",
    )

    assert isinstance(result, LoopResult)
    assert result.success
    assert result.loop_id == "story_spark"
    assert result.steps_completed > 0


def test_execute_loop_with_artifacts(orchestrator, mock_provider):
    """Test executing loop with existing artifacts."""
    orchestrator.initialize(provider=mock_provider)

    artifacts = [
        Artifact(
            type="hook_card",
            data={"title": "Test Hook", "description": "A test hook"},
            metadata={"id": "HOOK-001"},
        )
    ]

    result = orchestrator.execute_loop(
        loop_id="story_spark",
        project_id="test-project",
        artifacts=artifacts,
    )

    assert result.success
    # Should have artifacts from input plus new ones created
    assert len(result.artifacts_created) > 0


def test_execute_loop_with_config(orchestrator, mock_provider):
    """Test executing loop with configuration."""
    orchestrator.initialize(provider=mock_provider)

    config = {
        "max_iterations": 2,
        "section_count": 5,
    }

    result = orchestrator.execute_loop(
        loop_id="story_spark",
        project_id="test-project",
        config=config,
    )

    assert result.success


def test_execute_loop_not_found(orchestrator, mock_provider):
    """Test executing non-existent loop."""
    orchestrator.initialize(provider=mock_provider)

    with pytest.raises(KeyError, match="Loop 'nonexistent' not registered"):
        orchestrator.execute_loop(
            loop_id="nonexistent",
            project_id="test-project",
        )


def test_execute_loop_not_implemented(orchestrator, mock_provider):
    """Test executing loop that exists but isn't implemented yet."""
    orchestrator.initialize(provider=mock_provider)

    # hook_harvest is registered but not implemented
    with pytest.raises(KeyError, match="not yet implemented"):
        orchestrator.execute_loop(
            loop_id="hook_harvest",
            project_id="test-project",
        )


def test_execute_loop_with_human_callback(orchestrator, mock_provider):
    """Test that interactive mode uses Showrunner to initialize roles."""
    orchestrator.initialize(provider=mock_provider)
    mock_callback = MagicMock()

    # Mock the role instantiation methods
    orchestrator.showrunner.initialize_role_with_config = MagicMock()
    orchestrator.role_registry.get_role = MagicMock()

    orchestrator.execute_loop(
        loop_id="story_spark",
        project_id="test-project",
        config={"human_callback": mock_callback},
    )

    # Assert that the showrunner was used to initialize roles
    assert orchestrator.showrunner.initialize_role_with_config.called
    # Assert that the role registry was NOT used to get roles
    assert not orchestrator.role_registry.get_role.called


def test_execute_loop_in_batch_mode_uses_cache(orchestrator, mock_provider):
    """Test that batch mode uses the role registry to get roles."""
    orchestrator.initialize(provider=mock_provider)

    # Mock the role instantiation methods
    orchestrator.showrunner.initialize_role_with_config = MagicMock()
    orchestrator.role_registry.get_role = MagicMock()

    orchestrator.execute_loop(
        loop_id="story_spark",
        project_id="test-project",
        config={},
    )

    # Assert that the showrunner was NOT used to initialize roles
    assert not orchestrator.showrunner.initialize_role_with_config.called
    # Assert that the role registry was used to get roles
    assert orchestrator.role_registry.get_role.called


def test_execute_loop_interactive_not_initialized(orchestrator):
    """Test interactive mode without initialization raises error."""
    mock_callback = MagicMock()
    with pytest.raises(RuntimeError, match="not initialized"):
        orchestrator.execute_loop(
            loop_id="story_spark",
            project_id="test-project",
            config={"human_callback": mock_callback},
        )


def test_execute_loop_unregistered_role(orchestrator, mock_provider):
    """Test that unregistered roles are skipped gracefully."""
    orchestrator.initialize(provider=mock_provider)

    # Mock the role registry to remove a role
    original_roles = orchestrator.role_registry._roles
    orchestrator.role_registry._roles = {
        k: v for k, v in original_roles.items() if k != "plotwright"
    }

    try:
        # This should not raise an exception
        orchestrator.execute_loop(
            loop_id="story_spark",
            project_id="test-project",
        )
    finally:
        # Restore the original roles
        orchestrator.role_registry._roles = original_roles


# End-to-End Tests


def test_execute_goal_full_workflow(orchestrator, mock_provider):
    """Test full workflow: goal -> loop selection -> execution."""
    orchestrator.initialize(provider=mock_provider)

    result = orchestrator.execute_goal(
        goal="Create a new quest with branching narrative",
        project_id="test-project",
        project_state={"phase": "initial"},
    )

    assert isinstance(result, LoopResult)
    assert result.success
    assert result.loop_id == "story_spark"
    assert result.steps_completed == 6
    assert result.steps_failed == 0


def test_execute_goal_with_artifacts(orchestrator, mock_provider):
    """Test goal execution with existing artifacts."""
    orchestrator.initialize(provider=mock_provider)

    artifacts = [
        Artifact(
            type="hook_card",
            data={"title": "Existing Hook", "description": "Test"},
            metadata={"id": "HOOK-001"},
        )
    ]

    result = orchestrator.execute_goal(
        goal="Expand on existing narrative hooks",
        project_id="test-project",
        artifacts=artifacts,
    )

    assert result.success
    assert len(result.artifacts_created) > 0


# Registry Integration Tests


def test_orchestrator_uses_registries(orchestrator):
    """Test that orchestrator correctly uses registries."""
    assert len(orchestrator.loop_registry.list_loops()) > 0
    assert len(orchestrator.role_registry.list_roles()) > 0

    # Verify story_spark loop is available
    loop_ids = [loop.loop_id for loop in orchestrator.loop_registry.list_loops()]
    assert "story_spark" in loop_ids

    # Verify required roles are available
    required_roles = ["plotwright", "scene_smith", "gatekeeper", "showrunner"]
    available_roles = orchestrator.role_registry.list_roles()

    for role in required_roles:
        assert role in available_roles
