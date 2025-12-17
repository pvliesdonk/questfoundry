"""Tests for Secretary pattern context management."""

import pytest

from questfoundry.runtime.context import (
    Secretary,
    SummarizationLevel,
    SummarizationPolicy,
    ToolResultSummary,
)
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
