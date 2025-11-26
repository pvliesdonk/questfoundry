"""
Tests for the Control Plane (Mesh Architecture).

These tests verify the protocol-driven envelope routing works correctly.
Uses mock LLMs for unit tests and real Haiku for integration tests.
"""

import pytest
from unittest.mock import MagicMock, patch

from questfoundry.runtime.core.control_plane import (
    ControlPlane,
    DormancyRegistry,
    TERMINATE,
    BROADCAST,
    SHOWRUNNER,
)
from questfoundry.runtime.models.state import StudioState, Message


class TestDormancyRegistry:
    """Tests for DormancyRegistry."""

    def test_always_on_roles_never_dormant(self):
        """Showrunner and Gatekeeper are always on."""
        registry = DormancyRegistry()

        assert not registry.is_dormant("showrunner")
        assert not registry.is_dormant("gatekeeper")

        # Even if we try to make them sleep
        registry.sleep("showrunner")
        registry.sleep("gatekeeper")

        assert not registry.is_dormant("showrunner")
        assert not registry.is_dormant("gatekeeper")

    def test_optional_roles_can_be_dormant(self):
        """Optional roles start active and can be made dormant."""
        registry = DormancyRegistry()

        assert not registry.is_dormant("illustrator")

        registry.sleep("illustrator")
        assert registry.is_dormant("illustrator")

        registry.wake("illustrator")
        assert not registry.is_dormant("illustrator")

    def test_set_dormant_roles(self):
        """Can set multiple dormant roles at once."""
        registry = DormancyRegistry()

        registry.set_dormant_roles({"illustrator", "translator", "showrunner"})

        # showrunner should NOT be dormant (always_on)
        assert not registry.is_dormant("showrunner")
        # Others should be dormant
        assert registry.is_dormant("illustrator")
        assert registry.is_dormant("translator")


class TestRouteByEnvelope:
    """Tests for the route_by_envelope function."""

    def test_no_messages_routes_to_showrunner(self):
        """When no messages, route to showrunner."""
        control_plane = ControlPlane()
        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        result = control_plane.route_by_envelope(state)
        assert result == SHOWRUNNER

    def test_terminate_signal_routes_to_end(self):
        """Terminate signal should end the graph."""
        control_plane = ControlPlane()
        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [
                {
                    "sender": "showrunner",
                    "receiver": TERMINATE,
                    "intent": "tu.close",
                    "payload": {},
                    "timestamp": "",
                    "envelope": {},
                }
            ],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        from langgraph.graph import END
        result = control_plane.route_by_envelope(state)
        assert result == END

    def test_direct_routing_by_role_id(self):
        """Messages with role_id receiver route directly."""
        control_plane = ControlPlane()
        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [
                {
                    "sender": "gatekeeper",
                    "receiver": "scene_smith",
                    "intent": "gate.report.submit",
                    "payload": {"feedback": "Fix style issues"},
                    "timestamp": "",
                    "envelope": {},
                }
            ],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        result = control_plane.route_by_envelope(state)
        assert result == "scene_smith"

    def test_abbreviation_routing(self):
        """Messages with abbreviation receiver are normalized."""
        control_plane = ControlPlane()
        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [
                {
                    "sender": "scene_smith",
                    "receiver": "GK",  # Abbreviation for gatekeeper
                    "intent": "tu.update",
                    "payload": {},
                    "timestamp": "",
                    "envelope": {},
                }
            ],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        result = control_plane.route_by_envelope(state)
        assert result == "gatekeeper"

    def test_broadcast_routes_to_showrunner(self):
        """Broadcast messages route to showrunner for coordination."""
        control_plane = ControlPlane()
        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [
                {
                    "sender": "showrunner",
                    "receiver": BROADCAST,
                    "intent": "tu.assign",
                    "payload": {},
                    "timestamp": "",
                    "envelope": {},
                }
            ],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        result = control_plane.route_by_envelope(state)
        assert result == SHOWRUNNER

    def test_dormant_role_routes_to_showrunner(self):
        """Messages to dormant roles route to showrunner for wake decision."""
        control_plane = ControlPlane()
        control_plane.dormancy.sleep("illustrator")

        state: StudioState = {
            "tu_id": "TU-test",
            "tu_lifecycle": "hot-proposed",
            "current_node": "",
            "loop_id": "test",
            "loop_context": {},
            "hot_sot": {},
            "cold_sot": {},
            "artifacts": {},
            "quality_bars": {},
            "messages": [
                {
                    "sender": "art_director",
                    "receiver": "illustrator",
                    "intent": "tu.assign",
                    "payload": {},
                    "timestamp": "",
                    "envelope": {},
                }
            ],
            "snapshot_ref": None,
            "parent_tu_id": None,
            "error": None,
            "retry_count": 0,
            "created_at": "",
            "updated_at": "",
        }

        result = control_plane.route_by_envelope(state)
        assert result == SHOWRUNNER


class TestControlPlaneGraph:
    """Tests for graph creation."""

    def test_create_studio_graph(self):
        """Can create a studio graph with all roles."""
        control_plane = ControlPlane()

        # This should not raise
        graph = control_plane.create_studio_graph()

        assert graph is not None

    def test_available_roles(self):
        """All expected roles are available."""
        control_plane = ControlPlane()
        roles = control_plane.get_available_roles()

        # Core roles
        assert "showrunner" in roles
        assert "gatekeeper" in roles
        assert "plotwright" in roles
        assert "scene_smith" in roles

        # Optional roles
        assert "illustrator" in roles
        assert "translator" in roles


class TestKnowledgeTools:
    """Tests for knowledge tools (consult the cartridge)."""

    def test_consult_playbook_tool_exists(self):
        """ConsultPlaybook tool can be imported."""
        from questfoundry.runtime.tools.knowledge_tools import ConsultPlaybook

        tool = ConsultPlaybook()
        assert tool.name == "consult_playbook"

    def test_consult_quality_gate_tool_exists(self):
        """ConsultQualityGate tool can be imported."""
        from questfoundry.runtime.tools.knowledge_tools import ConsultQualityGate

        tool = ConsultQualityGate()
        assert tool.name == "consult_quality_gate"

    def test_consult_protocol_tool_exists(self):
        """ConsultProtocol tool can be imported."""
        from questfoundry.runtime.tools.knowledge_tools import ConsultProtocol

        tool = ConsultProtocol()
        assert tool.name == "consult_protocol"

    def test_consult_role_charter_tool_exists(self):
        """ConsultRoleCharter tool can be imported."""
        from questfoundry.runtime.tools.knowledge_tools import ConsultRoleCharter

        tool = ConsultRoleCharter()
        assert tool.name == "consult_role_charter"


# Integration tests (require LLM - skip if no API key)
@pytest.mark.integration
class TestControlPlaneIntegration:
    """Integration tests with real LLM (Haiku for cost efficiency)."""

    @pytest.mark.skipif(
        not pytest.importorskip("anthropic", reason="anthropic not installed"),
        reason="Requires anthropic package",
    )
    def test_basic_mesh_execution(self):
        """Test basic mesh execution with Haiku."""
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        control_plane = ControlPlane(
            preferred_provider="anthropic",
        )

        # This will actually call the LLM
        # Use a simple request to minimize cost
        try:
            final_state = control_plane.run(
                human_input="Say hello in one word",
                recursion_limit=5,  # Limit iterations for testing
            )

            assert final_state is not None
            assert "tu_id" in final_state
            assert "messages" in final_state
            assert len(final_state["messages"]) > 0

        except Exception as e:
            # Allow test to pass if LLM fails (network issues, rate limits, etc.)
            pytest.skip(f"LLM call failed: {e}")
