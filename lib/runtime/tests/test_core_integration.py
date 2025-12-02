"""
Integration tests for QuestFoundry runtime.

Tests the core components against real YAML definitions.
"""


import pytest

from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager


class TestSchemaRegistry:
    """Test SchemaRegistry with real YAML files."""

    @pytest.fixture
    def registry(self):
        return SchemaRegistry()

    def test_load_role_plotwright(self, registry):
        """Test loading plotwright role."""
        role = registry.load_role("plotwright")

        assert role is not None
        assert role.id == "plotwright"
        assert role.name == "Plotwright"
        assert role.role_type == "reasoning_agent"

    def test_load_loop_story_spark(self, registry):
        """Test loading story_spark loop."""
        loop = registry.load_loop("story_spark")

        assert loop is not None
        assert loop.id == "story_spark"
        assert loop.name == "story_spark"
        assert len(loop.nodes) > 0

    def test_load_all_roles(self, registry):
        """Test loading all 16 roles."""
        role_ids = [
            "plotwright",
            "scene_smith",
            "gatekeeper",
            "style_lead",
            "lore_weaver",
            "codex_curator",
            "audio_producer",
            "audio_director",
            "art_director",
            "illustrator",
            "player_narrator",
            "researcher",
            "translator",
            "book_binder",
            "export_service",
            "showrunner",
        ]

        for role_id in role_ids:
            role = registry.load_role(role_id)
            assert role is not None, f"Failed to load role: {role_id}"
            assert role.id == role_id

    def test_load_all_loops(self, registry):
        """Test loading all 10 loops."""
        loop_ids = [
            "story_spark",
            "hook_harvest",
            "lore_deepening",
            "codex_expansion",
            "audio_pass",
            "narration_dry_run",
            "style_tune_up",
            "binding_run",
            "art_touch_up",
            "translation_pass",
        ]

        for loop_id in loop_ids:
            loop = registry.load_loop(loop_id)
            assert loop is not None, f"Failed to load loop: {loop_id}"
            assert loop.id == loop_id


class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def manager(self):
        return StateManager()

    def test_initialize_state(self, manager):
        """Test state initialization."""
        state = manager.initialize_state(
            loop_id="story_spark", context={"scene_text": "cargo bay confrontation"}
        )

        assert state is not None
        assert state["tu_lifecycle"] == "hot-proposed"
        assert state["loop_id"] == "story_spark"
        assert len(state["quality_bars"]) == 8
        assert state["tu_id"].startswith("TU-")

    def test_update_state(self, manager):
        """Test state updates."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        new_state = manager.update_state(state, {"current_node": "plotwright"})

        assert new_state["current_node"] == "plotwright"
        assert state["current_node"] == ""  # Original unchanged

    def test_transition_tu_valid(self, manager):
        """Test valid TU transitions."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        new_state = manager.transition_tu(state, "stabilizing")
        assert new_state["tu_lifecycle"] == "stabilizing"

        new_state = manager.transition_tu(new_state, "gatecheck")
        assert new_state["tu_lifecycle"] == "gatecheck"

    def test_transition_tu_invalid(self, manager):
        """Test invalid TU transitions."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        # Try invalid transition
        with pytest.raises(ValueError):
            manager.transition_tu(state, "cold-merged")

    def test_add_artifact(self, manager):
        """Test adding artifacts to state."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        artifact = {
            "artifact_type": "scene",
            "content": "The crew discovers contraband...",
            "role_id": "plotwright",
            "timestamp": "2025-11-20T10:35:00Z",
            "tu_id": state["tu_id"],
            "state_key": "artifacts.hot.scenes.test",
            "metadata": {},
        }

        new_state = manager.add_artifact(state, artifact)

        assert "plotwright" in new_state["artifacts"]
        assert new_state["artifacts"]["plotwright"]["artifact_type"] == "scene"

    def test_quality_bars(self, manager):
        """Test quality bar updates."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        # Update bars
        new_state = manager.update_quality_bars(
            state,
            {
                "Integrity": {
                    "status": "green",
                    "feedback": "Story logic is sound",
                    "checked_by": "gatekeeper",
                    "timestamp": "2025-11-20T10:40:00Z",
                }
            },
        )

        assert new_state["quality_bars"]["Integrity"]["status"] == "green"

    def test_bar_threshold_all_green(self, manager):
        """Test all_green threshold."""
        state = manager.initialize_state(loop_id="story_spark", context={})

        # Mark all bars green
        bars = {}
        for bar_name in ["Integrity", "Reachability", "Nonlinearity"]:
            bars[bar_name] = {
                "status": "green",
                "feedback": None,
                "checked_by": "gatekeeper",
                "timestamp": None,
            }

        new_state = manager.update_quality_bars(state, bars)

        result = manager.check_bar_threshold(
            new_state, ["Integrity", "Reachability", "Nonlinearity"], "all_green"
        )

        assert result is True


class TestNodeFactory:
    """Test NodeFactory functionality."""

    @pytest.fixture
    def factory(self):
        return NodeFactory()

    def test_load_role(self, factory):
        """Test loading a role."""
        role = factory.load_role("plotwright")

        assert role is not None
        assert role.id == "plotwright"

    def test_create_role_node(self, factory):
        """Test creating a role node."""
        node = factory.create_role_node("plotwright")

        assert node is not None
        assert callable(node)

    def test_node_execution(self, factory):
        """Test executing a role node."""
        node = factory.create_role_node("plotwright")

        # Create mock state
        state = {
            "tu_id": "TU-2025-042",
            "tu_lifecycle": "hot-proposed",
            "current_node": "entry",
            "loop_id": "story_spark",
            "loop_context": {"scene_text": "cargo bay"},
            "artifacts": {},
            "quality_bars": {},
            "messages": [],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "2025-11-20T10:30:00Z",
            "updated_at": "2025-11-20T10:30:00Z",
        }

        result_state = node(state)

        assert result_state is not None
        assert result_state["current_node"] == "plotwright"
        assert "plotwright" in result_state["artifacts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
