"""Tests for Secretary pattern context management."""

import pytest

from questfoundry.runtime.context import Secretary, SummarizationPolicy, ToolResultSummary
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
