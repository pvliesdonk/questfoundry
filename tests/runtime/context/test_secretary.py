"""Tests for Secretary pattern context management."""

from datetime import datetime

import pytest

from questfoundry.runtime.context import (
    ContextSecretary,
    ContextSummaryResult,
    MailboxSecretary,
    MailboxSummaryResult,
    Secretary,
    SummarizationLevel,
    SummarizationPolicy,
    ToolResultSummary,
)
from questfoundry.runtime.messaging import Message, MessagePriority, MessageType
from questfoundry.runtime.models.base import Tool


class TestSummarizationPolicy:
    """Tests for SummarizationPolicy enum."""

    def test_enum_values(self):
        """Test that all expected policy values exist."""
        assert SummarizationPolicy.DROP.value == "drop"
        assert SummarizationPolicy.ULTRA_CONCISE.value == "ultra_concise"
        assert SummarizationPolicy.CONCISE.value == "concise"
        assert SummarizationPolicy.PRESERVE.value == "preserve"


class TestSecretary:
    """Tests for Secretary context management."""

    @pytest.fixture
    def secretary(self) -> Secretary:
        """Create a Secretary instance."""
        return Secretary()

    @pytest.fixture
    def tool_drop(self) -> Tool:
        """Tool with DROP summarization policy."""
        return Tool(
            id="list_agents",
            name="List Agents",
            description="List available agents",
            summarization_policy="drop",
            summary_template="Listed {count} agents",
        )

    @pytest.fixture
    def tool_ultra_concise(self) -> Tool:
        """Tool with ULTRA_CONCISE summarization policy."""
        return Tool(
            id="delegate",
            name="Delegate Work",
            description="Delegate work to another agent",
            summarization_policy="ultra_concise",
            summary_template="Delegated to {assigned_to}: {task}",
        )

    @pytest.fixture
    def tool_concise(self) -> Tool:
        """Tool with CONCISE summarization policy."""
        return Tool(
            id="search_workspace",
            name="Search Workspace",
            description="Search for artifacts",
            summarization_policy="concise",
            summary_template="Found {total_count} artifacts",
        )

    @pytest.fixture
    def tool_preserve(self) -> Tool:
        """Tool with PRESERVE summarization policy (default)."""
        return Tool(
            id="validate_artifact",
            name="Validate Artifact",
            description="Validate an artifact",
            summarization_policy="preserve",
        )

    def test_register_tool(self, secretary: Secretary, tool_drop: Tool):
        """Test registering a tool with the Secretary."""
        secretary.register_tool(tool_drop)
        assert tool_drop.id in secretary._tool_cache
        assert secretary._tool_cache[tool_drop.id] == tool_drop

    def test_get_policy_registered(self, secretary: Secretary, tool_drop: Tool):
        """Test getting policy for a registered tool."""
        secretary.register_tool(tool_drop)
        policy = secretary.get_policy(tool_drop.id)
        assert policy == SummarizationPolicy.DROP

    def test_get_policy_unregistered(self, secretary: Secretary):
        """Test getting policy for an unregistered tool defaults to PRESERVE."""
        policy = secretary.get_policy("unknown_tool")
        assert policy == SummarizationPolicy.PRESERVE

    def test_summarize_drop_policy(self, secretary: Secretary, tool_drop: Tool):
        """Test DROP policy removes result from context."""
        secretary.register_tool(tool_drop)
        result = {"agents": [{"id": "agent1"}, {"id": "agent2"}], "count": 2}

        summary = secretary.summarize_tool_result(tool_drop.id, result)

        assert summary.tool_id == tool_drop.id
        assert summary.policy_applied == SummarizationPolicy.DROP
        assert summary.content is None  # Dropped
        assert summary.was_summarized is True
        assert summary.summarized_size == 0

    def test_summarize_ultra_concise_policy(self, secretary: Secretary, tool_ultra_concise: Tool):
        """Test ULTRA_CONCISE policy uses summary template."""
        secretary.register_tool(tool_ultra_concise)
        result = {
            "delegation_id": "del-123",
            "assigned_to": "scene_smith",
            "task": "Write section prose",
            "status": "sent",
        }

        summary = secretary.summarize_tool_result(tool_ultra_concise.id, result)

        assert summary.tool_id == tool_ultra_concise.id
        assert summary.policy_applied == SummarizationPolicy.ULTRA_CONCISE
        assert summary.content == "Delegated to scene_smith: Write section prose"
        assert summary.was_summarized is True
        assert summary.summarized_size < summary.original_size

    def test_summarize_with_arguments(self, secretary: Secretary):
        """Test that arguments are merged with result for template substitution."""
        # Create a tool that uses template variables from arguments
        tool = Tool(
            id="search_tool",
            name="Search Tool",
            description="Search for things",
            summarization_policy="ultra_concise",
            summary_template="Searched for '{query}': found {count} results",
        )
        secretary.register_tool(tool)

        # Result only has count, query comes from arguments
        result = {"count": 5, "items": ["a", "b", "c", "d", "e"]}
        arguments = {"query": "test search", "limit": 10}

        summary = secretary.summarize_tool_result(tool.id, result, arguments=arguments)

        assert summary.policy_applied == SummarizationPolicy.ULTRA_CONCISE
        # Template should use query from arguments and count from result
        assert summary.content == "Searched for 'test search': found 5 results"
        assert summary.was_summarized is True

    def test_arguments_override_result_keys(self, secretary: Secretary):
        """Test that arguments take precedence when keys exist in both."""
        tool = Tool(
            id="echo_tool",
            name="Echo Tool",
            description="Echo input",
            summarization_policy="ultra_concise",
            summary_template="Input: {value}",
        )
        secretary.register_tool(tool)

        # Both have 'value' - arguments should win
        result = {"value": "result_value", "extra": "data"}
        arguments = {"value": "arg_value"}

        summary = secretary.summarize_tool_result(tool.id, result, arguments=arguments)

        # Arguments take precedence
        assert summary.content == "Input: arg_value"

    def test_summarize_concise_policy(self, secretary: Secretary, tool_concise: Tool):
        """Test CONCISE policy preserves key facts."""
        secretary.register_tool(tool_concise)
        result = {
            "results": [
                {"artifact_id": "art-1", "type": "section"},
                {"artifact_id": "art-2", "type": "section"},
            ],
            "total_count": 2,
            "query": "lighthouse",
        }

        summary = secretary.summarize_tool_result(tool_concise.id, result)

        assert summary.tool_id == tool_concise.id
        assert summary.policy_applied == SummarizationPolicy.CONCISE
        assert summary.content is not None
        assert "Found 2 artifacts" in summary.content  # Template applied
        assert "total_count: 2" in summary.content  # Key facts preserved
        assert summary.was_summarized is True

    def test_summarize_preserve_policy(self, secretary: Secretary, tool_preserve: Tool):
        """Test PRESERVE policy keeps full result."""
        secretary.register_tool(tool_preserve)
        result = {
            "valid": True,
            "schema_errors": [],
            "bar_results": [{"bar": "integrity", "status": "green"}],
        }

        summary = secretary.summarize_tool_result(tool_preserve.id, result)

        assert summary.tool_id == tool_preserve.id
        assert summary.policy_applied == SummarizationPolicy.PRESERVE
        assert summary.content is not None
        assert summary.was_summarized is False
        assert summary.summarized_size == summary.original_size

    def test_force_policy_override(self, secretary: Secretary, tool_preserve: Tool):
        """Test forcing a different policy than the tool's default."""
        secretary.register_tool(tool_preserve)
        result = {"valid": True, "schema_errors": []}

        summary = secretary.summarize_tool_result(
            tool_preserve.id,
            result,
            force_policy=SummarizationPolicy.DROP,
        )

        assert summary.policy_applied == SummarizationPolicy.DROP
        assert summary.content is None

    def test_template_substitution_simple(self, secretary: Secretary):
        """Test simple template variable substitution."""
        result = secretary._substitute_template("Count: {count}", {"count": 42})
        assert result == "Count: 42"

    def test_template_substitution_nested(self, secretary: Secretary):
        """Test nested key template substitution."""
        result = secretary._substitute_template(
            "Artifact: {artifact.id}",
            {"artifact": {"id": "art-123", "type": "section"}},
        )
        assert result == "Artifact: art-123"

    def test_template_substitution_missing_key(self, secretary: Secretary):
        """Test template with missing key preserves placeholder."""
        result = secretary._substitute_template("Value: {missing}", {"other": "data"})
        assert result == "Value: {missing}"

    def test_template_substitution_multiple(self, secretary: Secretary):
        """Test multiple variable substitution."""
        result = secretary._substitute_template(
            "{action} to {target}: {task}",
            {"action": "Delegated", "target": "agent", "task": "work"},
        )
        assert result == "Delegated to agent: work"

    def test_statistics_tracking(self, secretary: Secretary):
        """Test that Secretary tracks summarization statistics."""
        tool_drop = Tool(
            id="tool_drop",
            name="Drop Tool",
            description="Test",
            summarization_policy="drop",
        )
        tool_preserve = Tool(
            id="tool_preserve",
            name="Preserve Tool",
            description="Test",
            summarization_policy="preserve",
        )
        tool_concise = Tool(
            id="tool_concise",
            name="Concise Tool",
            description="Test",
            summarization_policy="concise",
            summary_template="Result",
        )

        secretary.register_tool(tool_drop)
        secretary.register_tool(tool_preserve)
        secretary.register_tool(tool_concise)

        # Make some tool calls
        secretary.summarize_tool_result("tool_drop", {"data": "x" * 100})
        secretary.summarize_tool_result("tool_drop", {"data": "y" * 100})
        secretary.summarize_tool_result("tool_preserve", {"data": "z" * 100})
        secretary.summarize_tool_result("tool_concise", {"key": "value"})

        stats = secretary.get_stats()

        assert stats["tools_dropped"] == 2
        assert stats["tools_preserved"] == 1
        assert stats["tools_summarized"] == 1
        assert stats["total_tokens_saved"] > 0

    def test_reset_stats(self, secretary: Secretary, tool_drop: Tool):
        """Test resetting statistics."""
        secretary.register_tool(tool_drop)
        secretary.summarize_tool_result(tool_drop.id, {"data": "test"})

        assert secretary.tools_dropped == 1

        secretary.reset_stats()

        assert secretary.tools_dropped == 0
        assert secretary.tools_preserved == 0
        assert secretary.tools_summarized == 0
        assert secretary.total_tokens_saved == 0

    def test_ultra_concise_fallback_no_template(self, secretary: Secretary):
        """Test ULTRA_CONCISE fallback when no template is provided."""
        tool = Tool(
            id="no_template",
            name="No Template",
            description="Test",
            summarization_policy="ultra_concise",
            # No summary_template
        )
        secretary.register_tool(tool)

        result = {"success": True, "data": {"key": "value"}}
        summary = secretary.summarize_tool_result("no_template", result)

        assert summary.policy_applied == SummarizationPolicy.ULTRA_CONCISE
        assert summary.content is not None
        assert "succeeded" in summary.content  # Fallback message

    def test_concise_truncates_long_strings(self, secretary: Secretary):
        """Test CONCISE policy truncates long string values."""
        tool = Tool(
            id="long_string",
            name="Long String",
            description="Test",
            summarization_policy="concise",
        )
        secretary.register_tool(tool)

        result = {"message": "x" * 200}  # Longer than 100 chars
        summary = secretary.summarize_tool_result("long_string", result)

        assert summary.content is not None
        assert "..." in summary.content  # Truncation marker
        assert len(summary.content) < 200  # Significantly shorter


class TestToolResultSummary:
    """Tests for ToolResultSummary dataclass."""

    def test_creation(self):
        """Test creating a ToolResultSummary."""
        summary = ToolResultSummary(
            tool_id="test_tool",
            original_size=100,
            summarized_size=50,
            policy_applied=SummarizationPolicy.CONCISE,
            content="Summarized content",
            was_summarized=True,
        )

        assert summary.tool_id == "test_tool"
        assert summary.original_size == 100
        assert summary.summarized_size == 50
        assert summary.policy_applied == SummarizationPolicy.CONCISE
        assert summary.content == "Summarized content"
        assert summary.was_summarized is True

    def test_dropped_result(self):
        """Test ToolResultSummary for dropped result."""
        summary = ToolResultSummary(
            tool_id="dropped_tool",
            original_size=500,
            summarized_size=0,
            policy_applied=SummarizationPolicy.DROP,
            content=None,
            was_summarized=True,
        )

        assert summary.content is None
        assert summary.summarized_size == 0


class TestTieredSummarization:
    """Tests for tiered summarization (progressive degradation)."""

    def test_summarization_level_enum(self):
        """Test SummarizationLevel enum values and ordering."""
        assert SummarizationLevel.NONE < SummarizationLevel.TOOL < SummarizationLevel.FULL
        assert SummarizationLevel.NONE == 0
        assert SummarizationLevel.TOOL == 1
        assert SummarizationLevel.FULL == 2

    def test_default_level_is_none(self):
        """Test that a fresh Secretary starts at level NONE."""
        secretary = Secretary()
        assert secretary.current_level == SummarizationLevel.NONE
        assert not secretary.should_summarize_tools()
        assert not secretary.should_summarize_messages()

    def test_level_none_when_below_threshold(self):
        """Test level is NONE when context usage is below threshold."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)
        secretary.update_context_size(500)  # 50% - below 70%

        assert secretary.current_level == SummarizationLevel.NONE
        assert not secretary.should_summarize_tools()
        assert secretary.usage_fraction == 0.5

    def test_level_tool_when_above_threshold(self):
        """Test level is TOOL when context is above threshold."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)
        secretary.update_context_size(800)  # 80% - above 70%

        assert secretary.current_level == SummarizationLevel.TOOL
        assert secretary.should_summarize_tools()
        # FULL level reserved for future message summarization
        assert not secretary.should_summarize_messages()
        assert secretary.usage_fraction == 0.8

    def test_level_transitions_at_threshold(self):
        """Test that level transitions when crossing threshold."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)

        # Start at NONE
        secretary.update_context_size(500)
        assert secretary.current_level == SummarizationLevel.NONE

        # Cross threshold to TOOL
        secretary.update_context_size(750)
        assert secretary.current_level == SummarizationLevel.TOOL

    def test_level_can_decrease_if_context_shrinks(self):
        """Test that level can decrease if context is somehow reduced."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)

        secretary.update_context_size(800)
        assert secretary.current_level == SummarizationLevel.TOOL

        # Context shrinks (e.g., after summarization)
        secretary.update_context_size(500)
        assert secretary.current_level == SummarizationLevel.NONE

    def test_usage_fraction_calculation(self):
        """Test usage_fraction calculation."""
        secretary = Secretary(context_limit=10000)
        secretary.update_context_size(2500)
        assert secretary.usage_fraction == 0.25

    def test_usage_fraction_with_zero_limit(self):
        """Test usage_fraction returns 0 when limit is 0."""
        secretary = Secretary(context_limit=0)
        secretary.update_context_size(1000)
        assert secretary.usage_fraction == 0.0

    def test_get_stats_includes_context_info(self):
        """Test that get_stats includes context information."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)
        secretary.update_context_size(800)

        stats = secretary.get_stats()

        assert stats["context_tokens"] == 800
        assert stats["context_limit"] == 1000
        assert stats["usage_percent"] == 80.0
        assert stats["current_level"] == "TOOL"

    def test_reset_context(self):
        """Test resetting context tracking."""
        secretary = Secretary(context_limit=1000)
        secretary.update_context_size(800)
        assert secretary.current_context_tokens == 800

        secretary.reset_context()
        assert secretary.current_context_tokens == 0
        assert secretary.current_level == SummarizationLevel.NONE

    def test_threshold_at_exact_boundary(self):
        """Test behavior at exact threshold boundary."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)

        # Exactly at threshold - should be TOOL
        secretary.update_context_size(700)
        assert secretary.current_level == SummarizationLevel.TOOL

    def test_custom_threshold(self):
        """Test with custom threshold value."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.5,  # Earlier summarization
        )

        secretary.update_context_size(400)
        assert secretary.current_level == SummarizationLevel.NONE

        secretary.update_context_size(600)
        assert secretary.current_level == SummarizationLevel.TOOL

    def test_context_limit_determines_threshold(self):
        """Test that context_limit directly affects when summarization kicks in."""
        # Small context limit - summarization kicks in earlier (absolute tokens)
        small_secretary = Secretary(context_limit=1000, summarization_threshold=0.7)
        small_secretary.update_context_size(700)
        assert small_secretary.should_summarize_tools()

        # Large context limit - same absolute tokens, no summarization yet
        large_secretary = Secretary(context_limit=10000, summarization_threshold=0.7)
        large_secretary.update_context_size(700)  # Only 7% of limit
        assert not large_secretary.should_summarize_tools()


class TestRecencyWindow:
    """Tests for recency window (recent tool results always preserved)."""

    # Use a consistent test agent ID
    AGENT_ID = "test_agent"

    def test_default_preserve_recent_n(self):
        """Test default preserve_recent_n value."""
        secretary = Secretary()
        assert secretary.preserve_recent_n == 5

    def test_track_tool_call(self):
        """Test tracking a tool call for an agent."""
        secretary = Secretary()
        secretary.track_tool_call("call-001", self.AGENT_ID)
        assert self.AGENT_ID in secretary._recent_tool_calls
        assert "call-001" in secretary._recent_tool_calls[self.AGENT_ID]

    def test_is_recent_with_empty_history(self):
        """Test is_recent returns True when no history exists for agent."""
        secretary = Secretary()
        # No agent_id means safe default (True)
        assert secretary.is_recent("call-001")
        # With agent_id but empty history, also True (safe default)
        assert secretary.is_recent("call-001", self.AGENT_ID)

    def test_is_recent_within_window(self):
        """Test that tool calls within the window are considered recent."""
        secretary = Secretary(preserve_recent_n=3)

        # Add 3 calls for this agent
        secretary.track_tool_call("call-001", self.AGENT_ID)
        secretary.track_tool_call("call-002", self.AGENT_ID)
        secretary.track_tool_call("call-003", self.AGENT_ID)

        # All should be recent for this agent
        assert secretary.is_recent("call-001", self.AGENT_ID)
        assert secretary.is_recent("call-002", self.AGENT_ID)
        assert secretary.is_recent("call-003", self.AGENT_ID)

    def test_is_recent_outside_window(self):
        """Test that old tool calls are not considered recent."""
        secretary = Secretary(preserve_recent_n=3)

        # Add 5 calls (exceeds window of 3)
        for i in range(5):
            secretary.track_tool_call(f"call-{i:03d}", self.AGENT_ID)

        # First two are outside window
        assert not secretary.is_recent("call-000", self.AGENT_ID)
        assert not secretary.is_recent("call-001", self.AGENT_ID)

        # Last three are recent
        assert secretary.is_recent("call-002", self.AGENT_ID)
        assert secretary.is_recent("call-003", self.AGENT_ID)
        assert secretary.is_recent("call-004", self.AGENT_ID)

    def test_get_old_tool_call_ids(self):
        """Test getting IDs outside the recency window for an agent."""
        secretary = Secretary(preserve_recent_n=3)

        # Add 5 calls
        for i in range(5):
            secretary.track_tool_call(f"call-{i:03d}", self.AGENT_ID)

        old_ids = secretary.get_old_tool_call_ids(self.AGENT_ID)
        assert old_ids == ["call-000", "call-001"]

    def test_get_old_tool_call_ids_all_recent(self):
        """Test that empty list returned when all calls are recent."""
        secretary = Secretary(preserve_recent_n=5)

        # Add 3 calls (less than window)
        for i in range(3):
            secretary.track_tool_call(f"call-{i:03d}", self.AGENT_ID)

        assert secretary.get_old_tool_call_ids(self.AGENT_ID) == []

    def test_recent_results_always_preserved(self):
        """Test that recent tool results are always preserved regardless of policy."""
        secretary = Secretary(preserve_recent_n=3)

        # Register a tool with DROP policy
        tool = Tool(
            id="droppable_tool",
            name="Droppable",
            description="Test",
            summarization_policy="drop",
        )
        secretary.register_tool(tool)

        # First call - is recent, should be preserved despite DROP policy
        result = {"data": "important"}
        summary = secretary.summarize_tool_result(
            "droppable_tool",
            result,
            tool_call_id="call-001",
            agent_id=self.AGENT_ID,
        )

        # Should be preserved (not dropped) because it's recent
        assert summary.policy_applied == SummarizationPolicy.PRESERVE
        assert summary.content is not None
        assert summary.was_summarized is False

    def test_old_results_use_declared_policy(self):
        """Test that old tool results use their declared summarization policy."""
        secretary = Secretary(preserve_recent_n=2)

        # Register a tool with DROP policy
        tool = Tool(
            id="droppable_tool",
            name="Droppable",
            description="Test",
            summarization_policy="drop",
        )
        secretary.register_tool(tool)

        # Add enough calls to push call-001 out of window
        secretary.track_tool_call("call-001", self.AGENT_ID)
        secretary.track_tool_call("call-002", self.AGENT_ID)
        secretary.track_tool_call("call-003", self.AGENT_ID)

        # Now call-001 is old, so it should use DROP policy
        result = {"data": "old data"}
        summary = secretary.summarize_tool_result(
            "droppable_tool",
            result,
            tool_call_id="call-001",  # Already tracked, is old
            agent_id=self.AGENT_ID,
        )

        # Should use DROP policy because it's not recent
        assert summary.policy_applied == SummarizationPolicy.DROP
        assert summary.content is None

    def test_recency_window_sliding(self):
        """Test that the recency window slides as new calls come in."""
        secretary = Secretary(preserve_recent_n=2)

        # call-001 starts as recent
        secretary.track_tool_call("call-001", self.AGENT_ID)
        assert secretary.is_recent("call-001", self.AGENT_ID)

        # Still recent after call-002
        secretary.track_tool_call("call-002", self.AGENT_ID)
        assert secretary.is_recent("call-001", self.AGENT_ID)

        # Pushed out of window by call-003
        secretary.track_tool_call("call-003", self.AGENT_ID)
        assert not secretary.is_recent("call-001", self.AGENT_ID)
        assert secretary.is_recent("call-002", self.AGENT_ID)
        assert secretary.is_recent("call-003", self.AGENT_ID)


class TestPerAgentIsolation:
    """Tests for per-agent context and history isolation.

    Critical invariants:
    1. Different agents have completely independent histories/contexts
    2. Multiple activations of the same agent (same ID) share history
    """

    def test_different_agents_have_independent_tool_histories(self):
        """Test that different agents have completely independent tool call histories."""
        secretary = Secretary(preserve_recent_n=3)

        # Track tool calls for agent A
        secretary.track_tool_call("call-A1", "agent_a")
        secretary.track_tool_call("call-A2", "agent_a")
        secretary.track_tool_call("call-A3", "agent_a")

        # Track tool calls for agent B
        secretary.track_tool_call("call-B1", "agent_b")
        secretary.track_tool_call("call-B2", "agent_b")

        # Agent A's calls should not be visible to agent B
        assert secretary.is_recent("call-A1", "agent_a")
        assert not secretary.is_recent("call-A1", "agent_b")  # Not in B's history

        # Agent B's calls should not be visible to agent A
        assert secretary.is_recent("call-B1", "agent_b")
        assert not secretary.is_recent("call-B1", "agent_a")  # Not in A's history

        # Each agent has their own history tracked
        assert "agent_a" in secretary._recent_tool_calls
        assert "agent_b" in secretary._recent_tool_calls
        assert len(secretary._recent_tool_calls["agent_a"]) == 3
        assert len(secretary._recent_tool_calls["agent_b"]) == 2

    def test_different_agents_have_independent_context_tracking(self):
        """Test that different agents have independent context size tracking."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)

        # Update context for agent A to above threshold
        secretary.update_context_size(800, "agent_a")  # 80%

        # Update context for agent B to below threshold
        secretary.update_context_size(400, "agent_b")  # 40%

        # Agent A should be at TOOL level, agent B at NONE
        assert secretary.should_summarize_tools_for_agent("agent_a")
        assert not secretary.should_summarize_tools_for_agent("agent_b")

        # Verify independent usage fractions
        assert secretary.get_usage_fraction("agent_a") == 0.8
        assert secretary.get_usage_fraction("agent_b") == 0.4

    def test_same_agent_id_shares_history_across_calls(self):
        """Test that the same agent ID shares history across multiple activations.

        When an agent (e.g., "lorekeeper") is activated multiple times,
        all activations should share the same tool call history.
        """
        secretary = Secretary(preserve_recent_n=3)
        agent_id = "lorekeeper"  # Same agent activated multiple times

        # First activation - track some calls
        secretary.track_tool_call("call-001", agent_id)
        secretary.track_tool_call("call-002", agent_id)

        # Simulate a second activation of the same agent
        # The history should be preserved from the first activation
        assert secretary.is_recent("call-001", agent_id)
        assert secretary.is_recent("call-002", agent_id)

        # Third call in "second activation" - history continues
        secretary.track_tool_call("call-003", agent_id)

        # All three calls should be in the shared history
        assert secretary.is_recent("call-001", agent_id)
        assert secretary.is_recent("call-002", agent_id)
        assert secretary.is_recent("call-003", agent_id)

        # Fourth call pushes first call out of window
        secretary.track_tool_call("call-004", agent_id)
        assert not secretary.is_recent("call-001", agent_id)  # Now old

    def test_same_agent_context_accumulates(self):
        """Test that context usage accumulates for the same agent ID."""
        secretary = Secretary(context_limit=1000, summarization_threshold=0.7)
        agent_id = "scene_smith"

        # First activation updates context
        secretary.update_context_size(300, agent_id)
        assert secretary.get_usage_fraction(agent_id) == 0.3

        # Second activation of same agent updates context (overwrites, not accumulates)
        # This is the correct behavior - context size is the CURRENT size, not cumulative
        secretary.update_context_size(600, agent_id)
        assert secretary.get_usage_fraction(agent_id) == 0.6

    def test_agent_isolation_with_summarization(self):
        """Test that summarization decisions are independent per agent."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.7,
            preserve_recent_n=2,
        )

        # Register a tool with DROP policy
        tool = Tool(
            id="list_tool",
            name="List Tool",
            description="Test",
            summarization_policy="drop",
        )
        secretary.register_tool(tool)

        # Agent A is under pressure, agent B is not
        secretary.update_context_size(800, "agent_a")
        secretary.update_context_size(200, "agent_b")

        # Track some calls for agent A (to make call-A1 old)
        secretary.track_tool_call("call-A1", "agent_a")
        secretary.track_tool_call("call-A2", "agent_a")
        secretary.track_tool_call("call-A3", "agent_a")  # Pushes A1 out

        # For agent A, old calls should be summarized (context pressure)
        result = {"items": [1, 2, 3]}
        summary_a = secretary.summarize_tool_result(
            "list_tool",
            result,
            tool_call_id="call-A1",  # Old for agent A
            agent_id="agent_a",
        )
        assert summary_a.policy_applied == SummarizationPolicy.DROP

        # For agent B, same tool call ID would be recent (different history)
        summary_b = secretary.summarize_tool_result(
            "list_tool",
            result,
            tool_call_id="call-B1",  # New for agent B
            agent_id="agent_b",
        )
        # Agent B's result is preserved (recent, regardless of context level)
        assert summary_b.policy_applied == SummarizationPolicy.PRESERVE

    def test_empty_agent_gets_safe_defaults(self):
        """Test that an unknown agent gets safe defaults (treat as recent)."""
        secretary = Secretary(preserve_recent_n=3)

        # Track calls for a known agent
        secretary.track_tool_call("call-001", "known_agent")

        # An unknown agent should get safe defaults
        # (empty history = everything is "recent" = preserved)
        assert secretary.is_recent("any-call", "unknown_agent")
        assert secretary.get_old_tool_call_ids("unknown_agent") == []

    def test_global_stats_accumulate_across_agents(self):
        """Test that global statistics accumulate across all agents."""
        secretary = Secretary(preserve_recent_n=1)

        tool = Tool(
            id="drop_tool",
            name="Drop Tool",
            description="Test",
            summarization_policy="drop",
        )
        secretary.register_tool(tool)

        # Make some calls from different agents
        # First call for each agent is recent (preserved)
        secretary.summarize_tool_result(
            "drop_tool",
            {"data": "a"},
            tool_call_id="call-A1",
            agent_id="agent_a",
        )
        secretary.summarize_tool_result(
            "drop_tool",
            {"data": "b"},
            tool_call_id="call-B1",
            agent_id="agent_b",
        )

        # Both should be preserved (recent)
        assert secretary.tools_preserved == 2
        assert secretary.tools_dropped == 0

        # Now add more calls to make the first ones old
        secretary.track_tool_call("call-A2", "agent_a")
        secretary.track_tool_call("call-B2", "agent_b")

        # Re-summarize old calls (now they should be dropped)
        secretary.summarize_tool_result(
            "drop_tool",
            {"data": "a"},
            tool_call_id="call-A1",  # Now old
            agent_id="agent_a",
        )
        secretary.summarize_tool_result(
            "drop_tool",
            {"data": "b"},
            tool_call_id="call-B1",  # Now old
            agent_id="agent_b",
        )

        # Stats should accumulate globally
        assert secretary.tools_dropped == 2
        assert secretary.tools_preserved == 2  # From earlier


class TestMailboxSecretary:
    """Tests for MailboxSecretary (mailbox-level message summarization)."""

    @pytest.fixture
    def secretary(self) -> MailboxSecretary:
        """Create a MailboxSecretary with test defaults."""
        return MailboxSecretary(
            auto_summarize_threshold=10,
            min_summarize_batch=3,
            preserve_recent_n=3,
            priority_threshold=MessagePriority.HIGH,
        )

    @pytest.fixture
    def sample_messages(self) -> list[Message]:
        """Create sample messages for testing."""
        messages = []
        for i in range(15):
            msg = Message(
                id=f"msg-{i:03d}",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="scene_smith",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, i, 0),
                priority=MessagePriority.PROGRESS,
                turn_created=i,
            )
            messages.append(msg)
        return messages

    def test_should_summarize_below_threshold(self, secretary: MailboxSecretary):
        """Test should_summarize returns False below threshold."""
        assert not secretary.should_summarize(5)  # Below 10

    def test_should_summarize_at_threshold(self, secretary: MailboxSecretary):
        """Test should_summarize returns False at exact threshold (needs to exceed)."""
        # The threshold is 10, so exactly 10 messages doesn't trigger summarization
        assert not secretary.should_summarize(10)
        # But 11 messages does
        assert secretary.should_summarize(11)

    def test_should_summarize_above_threshold(self, secretary: MailboxSecretary):
        """Test should_summarize returns True above threshold."""
        assert secretary.should_summarize(20)

    def test_select_preserves_recent_messages(
        self, secretary: MailboxSecretary, sample_messages: list[Message]
    ):
        """Test that recent messages are preserved."""
        to_summarize, to_preserve = secretary.select_messages_for_summarization(sample_messages)

        # Last 3 (preserve_recent_n=3) should be preserved
        preserved_ids = {m.id for m in to_preserve}
        assert "msg-012" in preserved_ids
        assert "msg-013" in preserved_ids
        assert "msg-014" in preserved_ids

    def test_select_preserves_high_priority(self, secretary: MailboxSecretary):
        """Test that high priority messages are preserved."""
        messages = [
            Message(
                id="msg-low",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="a",
                to_agent="b",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
                priority=MessagePriority.LOW,
                turn_created=0,
            ),
            Message(
                id="msg-high",
                type=MessageType.ESCALATION,
                from_agent="a",
                to_agent="b",
                timestamp=datetime(2025, 1, 15, 12, 1, 0),
                priority=MessagePriority.ESCALATION,  # High priority
                turn_created=1,
            ),
        ]

        to_summarize, to_preserve = secretary.select_messages_for_summarization(messages)

        preserved_ids = {m.id for m in to_preserve}
        assert "msg-high" in preserved_ids

    def test_select_preserves_delegations(self, secretary: MailboxSecretary):
        """Test that delegation messages are always preserved."""
        messages = [
            Message(
                id="msg-progress",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="a",
                to_agent="b",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
                priority=MessagePriority.PROGRESS,
                turn_created=0,
            ),
            Message(
                id="msg-delegation",
                type=MessageType.DELEGATION_REQUEST,
                from_agent="showrunner",
                to_agent="plotwright",
                timestamp=datetime(2025, 1, 15, 12, 1, 0),
                priority=MessagePriority.DELEGATION,
                turn_created=1,
            ),
        ]

        to_summarize, to_preserve = secretary.select_messages_for_summarization(messages)

        preserved_ids = {m.id for m in to_preserve}
        assert "msg-delegation" in preserved_ids

    def test_preserves_delegation_response(self, secretary: MailboxSecretary):
        """Test that delegation RESPONSES are also preserved."""
        # Create messages with a delegation response that should be preserved
        messages = [
            # Old, low-priority messages (should be summarized)
            *[
                Message(
                    id=f"msg-old-{i}",
                    type=MessageType.PROGRESS_UPDATE,
                    from_agent="agent",
                    to_agent="showrunner",
                    timestamp=datetime(2025, 1, 15, 12, i, 0),
                    priority=MessagePriority.PROGRESS,
                    turn_created=i,
                )
                for i in range(15)
            ],
            # Delegation response (should be preserved)
            Message(
                id="msg-delegation-response",
                type=MessageType.DELEGATION_RESPONSE,
                from_agent="plotwright",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, 16, 0),
                priority=MessagePriority.DELEGATION,
                turn_created=16,
            ),
        ]

        to_summarize, to_preserve = secretary.select_messages_for_summarization(messages)

        preserved_ids = {m.id for m in to_preserve}
        assert "msg-delegation-response" in preserved_ids

    def test_generate_summary_basic(self, secretary: MailboxSecretary):
        """Test basic summary generation."""
        messages = [
            Message(
                id=f"msg-{i}",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="scene_smith",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, i, 0),
            )
            for i in range(5)
        ]

        summary, action_items = secretary.generate_summary(messages)

        # Summary should mention count and sender
        assert "5" in summary
        assert "scene_smith" in summary

    def test_summarize_mailbox_below_threshold(
        self, secretary: MailboxSecretary, sample_messages: list[Message]
    ):
        """Test summarize_mailbox returns empty result when below threshold."""
        # Use only 5 messages (below threshold of 10)
        result = secretary.summarize_mailbox(sample_messages[:5], current_turn=10)

        assert result.messages_summarized == 0
        assert not result.digest_created

    def test_summarize_mailbox_above_threshold(
        self, secretary: MailboxSecretary, sample_messages: list[Message]
    ):
        """Test summarize_mailbox creates digest when above threshold."""
        result = secretary.summarize_mailbox(sample_messages, current_turn=20)

        assert result.messages_summarized > 0
        assert result.messages_preserved > 0
        assert result.digest_created
        assert result.summary_text is not None

    def test_summarize_mailbox_preserves_minimum(
        self, secretary: MailboxSecretary, sample_messages: list[Message]
    ):
        """Test that at least preserve_recent_n messages are preserved."""
        result = secretary.summarize_mailbox(sample_messages, current_turn=20)

        # Should preserve at least 3 (preserve_recent_n)
        assert result.messages_preserved >= 3


class TestMailboxSummaryResult:
    """Tests for MailboxSummaryResult dataclass."""

    def test_creation(self):
        """Test creating a MailboxSummaryResult."""
        result = MailboxSummaryResult(
            messages_summarized=10,
            messages_preserved=5,
            digest_created=True,
            summary_text="Summary of 10 messages",
            action_items=["Review feedback", "Address escalation"],
        )

        assert result.messages_summarized == 10
        assert result.messages_preserved == 5
        assert result.digest_created is True
        assert result.summary_text == "Summary of 10 messages"
        assert len(result.action_items) == 2

    def test_no_digest_created(self):
        """Test result when no digest was created."""
        result = MailboxSummaryResult(
            messages_summarized=0,
            messages_preserved=5,
            digest_created=False,
        )

        assert result.messages_summarized == 0
        assert result.digest_created is False
        assert result.summary_text is None
        assert result.action_items == []


class TestFullSummarizationLevel:
    """Tests for FULL level (context + mailbox summarization)."""

    def test_full_level_threshold(self):
        """Test that FULL level is triggered at 90% threshold."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.7,
            full_summarization_threshold=0.9,
        )

        # Below TOOL threshold
        secretary.update_context_size(600, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.NONE

        # At TOOL threshold
        secretary.update_context_size(700, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.TOOL

        # At FULL threshold
        secretary.update_context_size(900, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.FULL

    def test_should_summarize_messages_for_agent(self):
        """Test should_summarize_messages_for_agent at FULL level."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.7,
            full_summarization_threshold=0.9,
        )

        # Below FULL threshold
        secretary.update_context_size(800, "agent_a")
        assert not secretary.should_summarize_messages_for_agent("agent_a")
        assert secretary.should_summarize_tools_for_agent("agent_a")  # TOOL level

        # At FULL threshold
        secretary.update_context_size(900, "agent_a")
        assert secretary.should_summarize_messages_for_agent("agent_a")
        assert secretary.should_summarize_tools_for_agent("agent_a")  # Still true

    def test_global_full_level(self):
        """Test global FULL level detection."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.7,
            full_summarization_threshold=0.9,
        )

        secretary.update_context_size(950)  # Global context
        assert secretary.current_level == SummarizationLevel.FULL
        assert secretary.should_summarize_messages()

    def test_level_decreases_when_context_shrinks(self):
        """Test that level decreases when context is reduced."""
        secretary = Secretary(
            context_limit=1000,
            summarization_threshold=0.7,
            full_summarization_threshold=0.9,
        )

        # Start at FULL
        secretary.update_context_size(950, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.FULL

        # Drop to TOOL
        secretary.update_context_size(800, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.TOOL

        # Drop to NONE
        secretary.update_context_size(500, "agent_a")
        assert secretary.get_current_level("agent_a") == SummarizationLevel.NONE


class TestContextSecretary:
    """Tests for ContextSecretary (conversation context summarization)."""

    @pytest.fixture
    def secretary(self) -> ContextSecretary:
        """Create a ContextSecretary with test defaults."""
        return ContextSecretary(
            preserve_recent_turns=3,
            min_turns_to_summarize=5,
        )

    @pytest.fixture
    def sample_turns(self) -> list[dict]:
        """Create sample conversation turns."""
        turns = []
        for i in range(10):
            turns.append(
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Turn {i} content",
                    "tool_calls": [] if i % 2 == 0 else [{"name": "search_workspace"}],
                }
            )
        return turns

    def test_should_summarize_below_threshold(self, secretary: ContextSecretary):
        """Test should_summarize returns False when not enough turns."""
        # Need 5 turns beyond the 3 preserved = 8 total minimum
        assert not secretary.should_summarize(5)
        assert not secretary.should_summarize(7)

    def test_should_summarize_above_threshold(self, secretary: ContextSecretary):
        """Test should_summarize returns True when enough turns."""
        # 8 turns = 3 preserved + 5 summarizable
        assert secretary.should_summarize(8)
        assert secretary.should_summarize(20)

    def test_select_preserves_recent_turns(
        self, secretary: ContextSecretary, sample_turns: list[dict]
    ):
        """Test that recent turns are preserved."""
        to_summarize, to_preserve = secretary.select_turns_for_summarization(sample_turns)

        # Last 3 turns should be preserved
        preserved_contents = [t["content"] for t in to_preserve]
        assert "Turn 7 content" in preserved_contents
        assert "Turn 8 content" in preserved_contents
        assert "Turn 9 content" in preserved_contents

    def test_select_preserves_delegation_turns(self, secretary: ContextSecretary):
        """Test that turns with delegations are preserved."""
        turns = [
            {"role": "user", "content": "Regular message", "tool_calls": []},
            {"role": "assistant", "content": "Delegation to plotwright", "tool_calls": []},
            {"role": "user", "content": "Another message", "tool_calls": []},
            {"role": "assistant", "content": "Response", "tool_calls": []},
            {"role": "user", "content": "Recent 1", "tool_calls": []},
            {"role": "assistant", "content": "Recent 2", "tool_calls": []},
            {"role": "user", "content": "Recent 3", "tool_calls": []},
        ]

        to_summarize, to_preserve = secretary.select_turns_for_summarization(turns)

        # Turn with "delegation" should be preserved
        preserved_contents = [t["content"] for t in to_preserve]
        assert "Delegation to plotwright" in preserved_contents

    def test_select_preserves_artifact_creation_turns(self, secretary: ContextSecretary):
        """Test that turns with artifact creation are preserved."""
        turns = [
            {"role": "user", "content": "Create something", "tool_calls": []},
            {
                "role": "assistant",
                "content": "Creating artifact",
                "tool_calls": [{"name": "save_artifact"}],
            },
            {"role": "user", "content": "Another message", "tool_calls": []},
            {"role": "assistant", "content": "Response", "tool_calls": []},
            {"role": "user", "content": "Recent 1", "tool_calls": []},
            {"role": "assistant", "content": "Recent 2", "tool_calls": []},
            {"role": "user", "content": "Recent 3", "tool_calls": []},
        ]

        to_summarize, to_preserve = secretary.select_turns_for_summarization(turns)

        # Turn with save_artifact should be preserved
        preserved_contents = [t["content"] for t in to_preserve]
        assert "Creating artifact" in preserved_contents

    def test_generate_summary(self, secretary: ContextSecretary, sample_turns: list[dict]):
        """Test summary generation."""
        summary = secretary.generate_summary(sample_turns[:5])

        assert "5" in summary  # Number of turns
        assert "user" in summary or "assistant" in summary  # Roles mentioned
        assert "search_workspace" in summary  # Tool mentioned

    def test_summarize_context_below_threshold(self, secretary: ContextSecretary):
        """Test summarize_context returns empty result when below threshold."""
        turns = [{"role": "user", "content": f"Turn {i}"} for i in range(5)]

        result = secretary.summarize_context(turns)

        assert result.turns_summarized == 0
        assert not result.summary_created

    def test_summarize_context_above_threshold(
        self, secretary: ContextSecretary, sample_turns: list[dict]
    ):
        """Test summarize_context creates summary when above threshold."""
        result = secretary.summarize_context(sample_turns)

        assert result.turns_summarized > 0
        assert result.turns_preserved > 0
        assert result.summary_created
        assert result.summary_text is not None

    def test_summarize_context_preserves_minimum(
        self, secretary: ContextSecretary, sample_turns: list[dict]
    ):
        """Test that at least preserve_recent_turns are preserved."""
        result = secretary.summarize_context(sample_turns)

        # Should preserve at least 3 (preserve_recent_turns)
        assert result.turns_preserved >= 3


class TestContextSummaryResult:
    """Tests for ContextSummaryResult dataclass."""

    def test_creation(self):
        """Test creating a ContextSummaryResult."""
        result = ContextSummaryResult(
            turns_summarized=10,
            turns_preserved=5,
            summary_created=True,
            summary_text="Summary of 10 turns",
            tokens_before=5000,
            tokens_after=2000,
        )

        assert result.turns_summarized == 10
        assert result.turns_preserved == 5
        assert result.summary_created is True
        assert result.summary_text == "Summary of 10 turns"
        assert result.tokens_before == 5000
        assert result.tokens_after == 2000

    def test_no_summary_created(self):
        """Test result when no summary was created."""
        result = ContextSummaryResult(
            turns_summarized=0,
            turns_preserved=5,
            summary_created=False,
        )

        assert result.turns_summarized == 0
        assert result.summary_created is False
        assert result.summary_text is None
