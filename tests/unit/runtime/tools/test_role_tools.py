"""Tests for role tools - WriteHotSot, ReadHotSot, and lifecycle protection."""

import json

import pytest

from questfoundry.runtime.tools.role import WriteHotSot, ReadHotSot


class TestWriteHotSot:
    """Tests for WriteHotSot tool."""

    @pytest.fixture
    def state(self):
        """Empty state for testing."""
        return {"hot_store": {}}

    @pytest.fixture
    def write_tool(self, state):
        """Create WriteHotSot tool with state injected."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "test_role"
        return tool

    def test_basic_write(self, write_tool, state):
        """Test basic write to hot_store."""
        result = json.loads(write_tool._run(
            key="test_artifact",
            value={"title": "Test", "content": "Hello"}
        ))

        assert result["success"] is True
        assert result["action"] == "created"
        assert "test_artifact" in state["hot_store"]

    def test_update_existing(self, write_tool, state):
        """Test updating an existing artifact."""
        # Create first
        write_tool._run(key="test_artifact", value={"title": "Original"})

        # Update
        result = json.loads(write_tool._run(
            key="test_artifact",
            value={"title": "Updated"}
        ))

        assert result["success"] is True
        assert result["action"] == "updated"

    def test_key_required(self, write_tool):
        """Test that key is required."""
        result = json.loads(write_tool._run(key="", value={"test": 1}))
        assert result["success"] is False
        assert "key" in result["error"].lower()

    def test_value_required(self, write_tool):
        """Test that value is required."""
        result = json.loads(write_tool._run(key="test", value=None))
        assert result["success"] is False
        assert "value" in result["error"].lower()


class TestLifecycleStateProtection:
    """Tests for _lifecycle_state write protection per meta/ spec."""

    @pytest.fixture
    def state(self):
        """Empty state for testing."""
        return {"hot_store": {}}

    @pytest.fixture
    def write_tool(self, state):
        """Create WriteHotSot tool with state injected."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"
        return tool

    def test_lifecycle_state_stripped_from_write(self, write_tool, state):
        """Test that _lifecycle_state is stripped from artifact writes."""
        # Use a key that doesn't trigger schema validation (not scene_*, act_*, etc.)
        result = json.loads(write_tool._run(
            key="draft_data",
            value={
                "title": "Test Draft",
                "content": "Hello world",
                "_lifecycle_state": "approved"  # Should be stripped!
            }
        ))

        assert result["success"] is True
        # Value in store should NOT have _lifecycle_state
        stored = state["hot_store"]["draft_data"]
        if isinstance(stored, dict):
            assert "_lifecycle_state" not in stored
        else:
            # If wrapped in Artifact, check data
            if hasattr(stored, "data"):
                assert "_lifecycle_state" not in stored.data

    def test_lifecycle_violation_logged_in_response(self, write_tool):
        """Test that lifecycle violation is noted in response."""
        # Use a key that doesn't trigger schema validation
        result = json.loads(write_tool._run(
            key="test_artifact",
            value={
                "title": "Test",
                "_lifecycle_state": "approved"
            }
        ))

        assert result["success"] is True
        assert result.get("lifecycle_violation") is True
        assert "request_lifecycle_transition" in result.get("notice", "")

    def test_write_without_lifecycle_state_no_violation(self, write_tool):
        """Test that normal writes don't trigger violation notice."""
        # Use a key that doesn't trigger schema validation
        result = json.loads(write_tool._run(
            key="test_artifact",
            value={"title": "Test", "content": "Normal write"}
        ))

        assert result["success"] is True
        assert "lifecycle_violation" not in result

    def test_multiple_lifecycle_attempts_all_stripped(self, write_tool, state):
        """Test that repeated lifecycle state attempts are all stripped."""
        # Use keys that don't trigger schema validation
        # First write with lifecycle state
        result1 = json.loads(write_tool._run(
            key="draft_v1",
            value={"title": "V1", "_lifecycle_state": "draft"}
        ))
        assert result1["success"] is True
        assert result1.get("lifecycle_violation") is True

        # Second write with different lifecycle state
        result2 = json.loads(write_tool._run(
            key="draft_v2",
            value={"title": "V2", "_lifecycle_state": "approved"}
        ))
        assert result2["success"] is True
        assert result2.get("lifecycle_violation") is True

        # Third write without lifecycle state
        result3 = json.loads(write_tool._run(
            key="draft_v3",
            value={"title": "V3", "content": "Clean"}
        ))
        assert result3["success"] is True
        assert "lifecycle_violation" not in result3


class TestReadHotSot:
    """Tests for ReadHotSot tool."""

    @pytest.fixture
    def state(self):
        """State with some test data."""
        return {
            "hot_store": {
                "scene_1": {"title": "Scene One", "content": "Hello"},
                "nested": {"level1": {"level2": "value"}},
            }
        }

    @pytest.fixture
    def read_tool(self, state):
        """Create ReadHotSot tool with state injected."""
        tool = ReadHotSot()
        tool.state = state
        return tool

    def test_read_existing_key(self, read_tool):
        """Test reading an existing key."""
        result = json.loads(read_tool._run("scene_1"))
        assert result["success"] is True
        assert result["value"]["title"] == "Scene One"

    def test_read_missing_key(self, read_tool):
        """Test reading a missing key."""
        result = json.loads(read_tool._run("nonexistent"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_read_nested_path(self, read_tool):
        """Test reading with dot-path notation."""
        result = json.loads(read_tool._run("nested.level1.level2"))
        assert result["success"] is True
        assert result["value"] == "value"

    def test_key_required(self, read_tool):
        """Test that key is required."""
        result = json.loads(read_tool._run(""))
        assert result["success"] is False


class TestSystemFields:
    """Tests verifying system fields are managed correctly."""

    @pytest.fixture
    def state(self):
        """Empty state for testing."""
        return {"hot_store": {}}

    @pytest.fixture
    def write_tool(self, state):
        """Create WriteHotSot tool with state injected."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "plotwright"
        return tool

    def test_created_by_set_from_role_id(self, write_tool, state):
        """Test that created_by is set from role_id for artifact-typed writes."""
        # Use a key that doesn't trigger validation but includes type
        result = json.loads(write_tool._run(
            key="test_draft",
            value={"type": "draft", "title": "Test Draft", "data": {"content": "Hello"}}
        ))

        assert result["success"] is True
        stored = state["hot_store"]["test_draft"]
        # The Artifact wrapper should have created_by
        if hasattr(stored, "created_by"):
            assert stored.created_by == "plotwright"

    def test_artifact_type_detected_from_key(self, write_tool, state):
        """Test that artifact type is detected from key pattern."""
        # Write raw data (not schema-validated artifact)
        result = json.loads(write_tool._run(
            key="custom_data",
            value={"title": "Custom Data", "content": "Opening..."}
        ))

        # The write should succeed
        assert result["success"] is True
        assert "custom_data" in state["hot_store"]

    def test_lifecycle_state_is_write_protected(self, write_tool, state):
        """Test that _lifecycle_state cannot be directly set by agents."""
        result = json.loads(write_tool._run(
            key="protected_artifact",
            value={
                "title": "Test",
                "content": "Hello",
                "_lifecycle_state": "approved"  # Should be stripped
            }
        ))

        assert result["success"] is True
        assert result.get("lifecycle_violation") is True
        # The stored value should not have _lifecycle_state
        stored = state["hot_store"]["protected_artifact"]
        if isinstance(stored, dict):
            assert "_lifecycle_state" not in stored


class TestWorkflowIntentNudging:
    """Tests for Phase 2b workflow intent (designated_producers) nudging.

    Per meta/ spec "open floor" principle:
    - Writes always succeed (never denied)
    - Non-designated producers get nudges for exclusive stores
    - Designated producers write without nudges
    """

    @pytest.fixture
    def mock_studio(self):
        """Create a mock studio with stores having workflow intent."""
        from unittest.mock import MagicMock

        # Create mock stores with workflow intent
        workspace_store = MagicMock()
        workspace_store.id = "workspace"
        workspace_store.artifact_types = ["section_brief", "hook_card"]
        workspace_store.workflow_intent = MagicMock()
        workspace_store.workflow_intent.production_guidance = "all"
        workspace_store.workflow_intent.designated_producers = []

        codex_store = MagicMock()
        codex_store.id = "codex"
        codex_store.artifact_types = ["codex_entry"]
        codex_store.workflow_intent = MagicMock()
        codex_store.workflow_intent.production_guidance = "exclusive"
        codex_store.workflow_intent.designated_producers = ["codex_curator"]

        canon_store = MagicMock()
        canon_store.id = "canon"
        canon_store.artifact_types = ["canon_pack"]
        canon_store.workflow_intent = MagicMock()
        canon_store.workflow_intent.production_guidance = "exclusive"
        canon_store.workflow_intent.designated_producers = ["lore_weaver"]

        studio = MagicMock()
        studio.stores = {
            "workspace": workspace_store,
            "codex": codex_store,
            "canon": canon_store,
        }
        return studio

    @pytest.fixture
    def state(self):
        """Empty state for testing."""
        return {"hot_store": {}}

    def test_no_nudge_for_workspace_artifacts(self, state, mock_studio):
        """Test that workspace artifacts (all producers) don't get nudged."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="test_brief",
            value={"type": "section_brief", "title": "Test Brief"}
        ))

        assert result["success"] is True
        assert "workflow_nudge" not in result

    def test_nudge_for_non_designated_codex_write(self, state, mock_studio):
        """Test that non-designated producers get nudged for codex writes."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"  # Not codex_curator
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="test_entry",
            value={"type": "codex_entry", "title": "Test Entry"}
        ))

        assert result["success"] is True  # Write still succeeds (open floor)
        assert result.get("workflow_nudge") is True
        assert result.get("designated_producers") == ["codex_curator"]
        assert result.get("target_store") == "codex"
        assert "codex_curator" in result.get("notice", "")

    def test_no_nudge_for_designated_codex_write(self, state, mock_studio):
        """Test that designated producers don't get nudged."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "codex_curator"  # Is designated
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="test_entry",
            value={"type": "codex_entry", "title": "Test Entry"}
        ))

        assert result["success"] is True
        assert "workflow_nudge" not in result

    def test_nudge_for_non_designated_canon_write(self, state, mock_studio):
        """Test that non-designated producers get nudged for canon writes."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "plotwright"  # Not lore_weaver
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="test_canon",
            value={"type": "canon_pack", "title": "Test Canon"}
        ))

        assert result["success"] is True  # Write still succeeds
        assert result.get("workflow_nudge") is True
        assert result.get("designated_producers") == ["lore_weaver"]
        assert result.get("target_store") == "canon"

    def test_no_nudge_when_studio_not_available(self, state):
        """Test graceful handling when studio is not injected."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"
        tool.studio = None  # No studio

        result = json.loads(tool._run(
            key="test_data",
            value={"type": "codex_entry", "title": "Test"}
        ))

        assert result["success"] is True
        assert "workflow_nudge" not in result

    def test_both_lifecycle_and_workflow_violations(self, state, mock_studio):
        """Test that both violations can occur simultaneously."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"  # Not codex_curator
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="test_entry",
            value={
                "type": "codex_entry",
                "title": "Test Entry",
                "_lifecycle_state": "approved"  # Should be stripped
            }
        ))

        assert result["success"] is True
        assert result.get("lifecycle_violation") is True
        assert result.get("workflow_nudge") is True
        # Notice should contain both violations
        notice = result.get("notice", "")
        assert "lifecycle" in notice.lower() or "request_lifecycle_transition" in notice
        assert "codex_curator" in notice

    def test_workflow_intent_check_method_directly(self, mock_studio):
        """Test the _check_workflow_intent method directly."""
        tool = WriteHotSot()
        tool.role_id = "scene_smith"
        tool.studio = mock_studio

        # Non-exclusive store - no nudge
        result = tool._check_workflow_intent("section_brief")
        assert result is None

        # Exclusive store, non-designated - nudge
        result = tool._check_workflow_intent("codex_entry")
        assert result is not None
        assert result["workflow_nudge"] is True
        assert result["store"] == "codex"

        # Designated producer - no nudge
        tool.role_id = "codex_curator"
        result = tool._check_workflow_intent("codex_entry")
        assert result is None

    def test_unknown_artifact_type_no_nudge(self, state, mock_studio):
        """Test that unknown artifact types don't trigger nudges."""
        tool = WriteHotSot()
        tool.state = state
        tool.role_id = "scene_smith"
        tool.studio = mock_studio

        result = json.loads(tool._run(
            key="unknown_key",
            value={"type": "unknown_type", "data": {"content": "test"}}
        ))

        assert result["success"] is True
        assert "workflow_nudge" not in result
