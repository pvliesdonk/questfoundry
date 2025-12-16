"""Tests for lifecycle transition tools."""

import pytest

from questfoundry.runtime.storage.lifecycle import (
    ArtifactLifecycle,
    LifecycleManager,
    LifecycleState,
    LifecycleTransition,
)
from questfoundry.runtime.tools.base import ToolContext
from questfoundry.runtime.tools.lifecycle_transition import (
    GetLifecycleStateTool,
    RequestLifecycleTransitionTool,
)


class MockStudio:
    """Minimal Studio mock."""

    def __init__(self):
        self.artifact_types = []


class MockProject:
    """Mock project for testing artifact operations."""

    def __init__(self):
        self.artifacts: dict[str, dict] = {}

    def get_artifact(self, artifact_id: str) -> dict | None:
        return self.artifacts.get(artifact_id)

    def update_artifact(self, artifact_id: str, data: dict) -> dict | None:
        if artifact_id not in self.artifacts:
            return None
        self.artifacts[artifact_id].update(data)
        return self.artifacts[artifact_id]


class MockToolDefinition:
    """Minimal Tool definition mock."""

    def __init__(self, tool_id: str = "test_tool"):
        self.id = tool_id
        self.name = tool_id
        self.description = f"Test tool {tool_id}"
        self.timeout_ms = 30000
        self.input_schema = None


@pytest.fixture
def mock_project():
    """Create mock project with test artifacts."""
    project = MockProject()
    project.artifacts = {
        "section_001": {
            "_id": "section_001",
            "_type": "section",
            "_lifecycle_state": "draft",
            "title": "Test Section",
        },
        "section_002": {
            "_id": "section_002",
            "_type": "section",
            "_lifecycle_state": "review",
            "title": "Section in Review",
        },
        "section_003": {
            "_id": "section_003",
            "_type": "section",
            "_lifecycle_state": "cold",
            "title": "Cold Section",
        },
        "note_001": {
            "_id": "note_001",
            "_type": "note",
            # No lifecycle state - defaults to draft
            "content": "A note",
        },
    }
    return project


@pytest.fixture
def lifecycle_manager():
    """Create lifecycle manager with section lifecycle."""
    manager = LifecycleManager()

    section_lifecycle = ArtifactLifecycle(
        artifact_type_id="section",
        states={
            "draft": LifecycleState(id="draft", name="Draft"),
            "review": LifecycleState(id="review", name="Review"),
            "gatecheck": LifecycleState(id="gatecheck", name="Gatecheck"),
            "approved": LifecycleState(id="approved", name="Approved"),
            "cold": LifecycleState(id="cold", name="Cold", terminal=True),
        },
        transitions=[
            LifecycleTransition(from_state="draft", to_state="review"),
            LifecycleTransition(from_state="review", to_state="draft"),
            LifecycleTransition(from_state="review", to_state="gatecheck"),
            LifecycleTransition(from_state="gatecheck", to_state="draft"),
            LifecycleTransition(
                from_state="gatecheck",
                to_state="approved",
                allowed_agents=["gatekeeper"],
            ),
            LifecycleTransition(
                from_state="approved",
                to_state="cold",
                allowed_agents=["gatekeeper"],
                requires_validation=["integrity", "style"],
            ),
        ],
        initial_state="draft",
    )
    manager.register_lifecycle(section_lifecycle)

    return manager


@pytest.fixture
def tool_context(mock_project, lifecycle_manager):
    """Create tool context with project and lifecycle manager."""
    ctx = ToolContext(
        studio=MockStudio(),
        project=mock_project,
        agent_id="scene_smith",
        lifecycle_manager=lifecycle_manager,
    )
    return ctx


class TestRequestLifecycleTransitionTool:
    """Tests for RequestLifecycleTransitionTool."""

    @pytest.mark.asyncio
    async def test_missing_artifact_id(self, tool_context):
        """Reject missing artifact_id."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute({"target_state": "review"})

        assert result.success is False
        assert "artifact_id is required" in result.error

    @pytest.mark.asyncio
    async def test_missing_target_state(self, tool_context):
        """Reject missing target_state."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is False
        assert "target_state is required" in result.error

    @pytest.mark.asyncio
    async def test_no_project(self):
        """Reject when no project available."""
        ctx = ToolContext(studio=MockStudio(), project=None)
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            ctx,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "review",
            }
        )

        assert result.success is False
        assert "No project available" in result.error

    @pytest.mark.asyncio
    async def test_artifact_not_found(self, tool_context):
        """Reject non-existent artifact."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "nonexistent",
                "target_state": "review",
            }
        )

        assert result.success is False
        assert "Artifact not found" in result.error

    @pytest.mark.asyncio
    async def test_already_in_target_state(self, tool_context):
        """Return success with transitioned=False when already in target state."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "draft",
            }
        )

        assert result.success is True
        assert result.data["transitioned"] is False
        assert "already in" in result.data["message"]

    @pytest.mark.asyncio
    async def test_valid_transition(self, tool_context, mock_project):
        """Successfully transition artifact."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "review",
            }
        )

        assert result.success is True
        assert result.data["transitioned"] is True
        assert result.data["previous_state"] == "draft"
        assert result.data["new_state"] == "review"
        assert result.data["result"] == "committed"
        assert mock_project.artifacts["section_001"]["_lifecycle_state"] == "review"

    @pytest.mark.asyncio
    async def test_invalid_transition(self, tool_context):
        """Reject invalid transition path."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        # draft -> cold is not a valid transition
        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "cold",
            }
        )

        assert result.success is False
        assert "Transition not allowed" in result.error

    @pytest.mark.asyncio
    async def test_agent_not_allowed(self, tool_context):
        """Reject when agent not in allowed_agents."""
        # Move section to gatecheck first
        tool_context.project.artifacts["section_001"]["_lifecycle_state"] = "gatecheck"

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        # scene_smith cannot transition gatecheck -> approved (only gatekeeper)
        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "approved",
            }
        )

        assert result.success is False
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_gatekeeper_can_approve(self, tool_context):
        """Gatekeeper can transition to approved."""
        tool_context.agent_id = "gatekeeper"
        tool_context.project.artifacts["section_001"]["_lifecycle_state"] = "gatecheck"

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "approved",
            }
        )

        assert result.success is True
        assert result.data["transitioned"] is True
        assert result.data["new_state"] == "approved"
        assert result.data["result"] == "committed"

    @pytest.mark.asyncio
    async def test_requires_validation_runs_checks(self, tool_context, mock_project):
        """Run validations and commit if all pass."""
        tool_context.agent_id = "gatekeeper"
        mock_project.artifacts["section_001"]["_lifecycle_state"] = "approved"

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        # approved -> cold requires validation (integrity, style)
        # Without a full Studio setup, validate_artifact may not find the type
        # But the tool should still attempt validation
        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "cold",
            }
        )

        assert result.success is True
        # Either committed (validations passed) or rejected (validations failed)
        assert result.data["result"] in ("committed", "rejected")

        if result.data["result"] == "committed":
            assert result.data["transitioned"] is True
            assert result.data["new_state"] == "cold"
            assert mock_project.artifacts["section_001"]["_lifecycle_state"] == "cold"
        else:
            assert result.data["transitioned"] is False
            assert "validation_results" in result.data

    @pytest.mark.asyncio
    async def test_force_bypasses_validation(self, tool_context, mock_project):
        """Force=true bypasses validation requirement."""
        tool_context.agent_id = "gatekeeper"
        mock_project.artifacts["section_001"]["_lifecycle_state"] = "approved"

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "cold",
                "force": True,
            }
        )

        assert result.success is True
        assert result.data["transitioned"] is True
        assert result.data["new_state"] == "cold"
        assert result.data["result"] == "committed"
        assert mock_project.artifacts["section_001"]["_lifecycle_state"] == "cold"

    @pytest.mark.asyncio
    async def test_validation_rejection_with_broken_reference(
        self, tool_context, mock_project, lifecycle_manager
    ):
        """Reject transition when integrity validation fails on broken reference."""
        from unittest.mock import MagicMock

        # Create a mock artifact type with a reference field
        section_type = MagicMock()
        section_type.id = "section"

        # Mock field with reference type
        ref_field = MagicMock()
        ref_field.name = "parent_ref"
        ref_field.type = MagicMock()
        ref_field.type.value = "reference"
        ref_field.required = False

        title_field = MagicMock()
        title_field.name = "title"
        title_field.type = MagicMock()
        title_field.type.value = "string"
        title_field.required = True

        section_type.fields = [title_field, ref_field]
        section_type.validation = None

        # Set up mock studio with the artifact type
        tool_context.studio.artifact_types = [section_type]

        # Set up artifact with broken reference
        tool_context.agent_id = "gatekeeper"
        mock_project.artifacts["section_with_ref"] = {
            "_id": "section_with_ref",
            "_type": "section",
            "_lifecycle_state": "approved",
            "title": "Section with broken ref",
            "parent_ref": "nonexistent_artifact",  # Broken reference
        }

        # Add lifecycle for this to require integrity validation
        from questfoundry.runtime.storage.lifecycle import (
            ArtifactLifecycle,
            LifecycleState,
            LifecycleTransition,
        )

        section_lifecycle = ArtifactLifecycle(
            artifact_type_id="section",
            states={
                "approved": LifecycleState(id="approved", name="Approved"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(
                    from_state="approved",
                    to_state="cold",
                    requires_validation=["integrity"],  # Only integrity check
                ),
            ],
            initial_state="approved",
        )
        lifecycle_manager.register_lifecycle(section_lifecycle)

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "section_with_ref",
                "target_state": "cold",
            }
        )

        assert result.success is True
        assert result.data["result"] == "rejected"
        assert result.data["transitioned"] is False
        assert "integrity" in result.data["rejection_reason"].lower()
        # Artifact should NOT have transitioned
        assert mock_project.artifacts["section_with_ref"]["_lifecycle_state"] == "approved"

    @pytest.mark.asyncio
    async def test_transition_from_terminal_state(self, tool_context):
        """Cannot transition from terminal state."""
        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        # section_003 is already cold (terminal)
        result = await tool.execute(
            {
                "artifact_id": "section_003",
                "target_state": "draft",
            }
        )

        assert result.success is False
        assert "terminal" in result.error.lower() or "Transition not allowed" in result.error

    @pytest.mark.asyncio
    async def test_no_lifecycle_manager_allows_transition(self, mock_project):
        """Without lifecycle manager, any transition allowed."""
        ctx = ToolContext(
            studio=MockStudio(),
            project=mock_project,
            agent_id="scene_smith",
            lifecycle_manager=None,  # No manager
        )

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            ctx,
        )

        # Even draft -> cold works without manager
        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "target_state": "cold",
            }
        )

        assert result.success is True
        assert result.data["transitioned"] is True

    @pytest.mark.asyncio
    async def test_artifact_without_type(self, tool_context, mock_project):
        """Artifact without _type can still transition (no lifecycle validation)."""
        mock_project.artifacts["untyped"] = {
            "_id": "untyped",
            "_lifecycle_state": "draft",
            "content": "Untyped artifact",
        }

        tool = RequestLifecycleTransitionTool(
            MockToolDefinition("request_lifecycle_transition"),
            tool_context,
        )

        result = await tool.execute(
            {
                "artifact_id": "untyped",
                "target_state": "review",
            }
        )

        # Without type, lifecycle manager can't validate - transition proceeds
        assert result.success is True
        assert result.data["transitioned"] is True


class TestGetLifecycleStateTool:
    """Tests for GetLifecycleStateTool."""

    @pytest.mark.asyncio
    async def test_missing_artifact_id(self, tool_context):
        """Reject missing artifact_id."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({})

        assert result.success is False
        assert "artifact_id is required" in result.error

    @pytest.mark.asyncio
    async def test_no_project(self):
        """Reject when no project available."""
        ctx = ToolContext(studio=MockStudio(), project=None)
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            ctx,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is False
        assert "No project available" in result.error

    @pytest.mark.asyncio
    async def test_artifact_not_found(self, tool_context):
        """Reject non-existent artifact."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "nonexistent"})

        assert result.success is False
        assert "Artifact not found" in result.error

    @pytest.mark.asyncio
    async def test_get_state_basic(self, tool_context):
        """Get basic lifecycle state."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is True
        assert result.data["artifact_id"] == "section_001"
        assert result.data["artifact_type"] == "section"
        assert result.data["current_state"] == "draft"

    @pytest.mark.asyncio
    async def test_get_state_with_transitions(self, tool_context):
        """Get state with available transitions."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is True
        assert "available_transitions" in result.data
        # draft can go to review
        transitions = result.data["available_transitions"]
        assert len(transitions) == 1
        assert transitions[0]["target_state"] == "review"

    @pytest.mark.asyncio
    async def test_get_state_terminal(self, tool_context):
        """Get terminal state info."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "section_003"})

        assert result.success is True
        assert result.data["current_state"] == "cold"
        assert result.data["is_terminal"] is True
        # No transitions from terminal
        assert result.data["available_transitions"] == []

    @pytest.mark.asyncio
    async def test_get_state_default_draft(self, tool_context):
        """Artifact without _lifecycle_state defaults to draft."""
        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "note_001"})

        assert result.success is True
        assert result.data["current_state"] == "draft"

    @pytest.mark.asyncio
    async def test_get_state_no_lifecycle_manager(self, mock_project):
        """State retrieval works without lifecycle manager."""
        ctx = ToolContext(
            studio=MockStudio(),
            project=mock_project,
            lifecycle_manager=None,
        )

        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            ctx,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is True
        assert result.data["current_state"] == "draft"
        # No transitions info without manager
        assert "available_transitions" not in result.data

    @pytest.mark.asyncio
    async def test_transitions_filtered_by_agent(self, tool_context):
        """Available transitions filtered by current agent."""
        # Move to gatecheck
        tool_context.project.artifacts["section_001"]["_lifecycle_state"] = "gatecheck"

        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        # scene_smith can only go back to draft
        result = await tool.execute({"artifact_id": "section_001"})

        transitions = result.data["available_transitions"]
        targets = {t["target_state"] for t in transitions}
        assert "draft" in targets
        assert "approved" not in targets  # Only gatekeeper

    @pytest.mark.asyncio
    async def test_gatekeeper_sees_more_transitions(self, tool_context):
        """Gatekeeper sees additional transition options."""
        tool_context.agent_id = "gatekeeper"
        tool_context.project.artifacts["section_001"]["_lifecycle_state"] = "gatecheck"

        tool = GetLifecycleStateTool(
            MockToolDefinition("get_lifecycle_state"),
            tool_context,
        )

        result = await tool.execute({"artifact_id": "section_001"})

        transitions = result.data["available_transitions"]
        targets = {t["target_state"] for t in transitions}
        assert "draft" in targets
        assert "approved" in targets  # Gatekeeper can see this
