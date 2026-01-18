"""Tests for Summarize phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.agents.prompts import get_summarize_prompt
from questfoundry.agents.summarize import _format_messages_for_summary, summarize_discussion


class TestGetSummarizePrompt:
    """Test summarize prompt loading."""

    def test_prompt_loads_from_template(self) -> None:
        """Prompt should load from external template file."""
        prompt = get_summarize_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_includes_task_description(self) -> None:
        """Prompt should describe the summarization task."""
        prompt = get_summarize_prompt()

        assert "summariz" in prompt.lower()
        assert "brief" in prompt.lower()

    def test_prompt_includes_categories(self) -> None:
        """Prompt should mention key categories to extract."""
        prompt = get_summarize_prompt()

        assert "Genre" in prompt
        assert "Tone" in prompt
        assert "audience" in prompt.lower()
        assert "themes" in prompt.lower()


class TestSummarizeDiscussion:
    """Test summarize_discussion function."""

    @pytest.mark.asyncio
    async def test_summarize_returns_summary_and_tokens(self) -> None:
        """summarize_discussion should return summary text and token count."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary of discussion")
        # Token usage via usage_metadata attribute (Ollama-style)
        mock_response.usage_metadata = {"total_tokens": 100}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            HumanMessage(content="I want to write a mystery"),
            AIMessage(content="Great! Tell me more about the setting."),
        ]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary of discussion"
        assert tokens == 100

    @pytest.mark.asyncio
    async def test_summarize_uses_model_directly(self) -> None:
        """summarize_discussion should use model directly without bind."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        await summarize_discussion(mock_model, messages)

        # Should call ainvoke directly on model, not via bind
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_includes_conversation_in_prompt(self) -> None:
        """summarize_discussion should include formatted conversation."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            HumanMessage(content="User message"),
            AIMessage(content="Assistant response"),
        ]

        await summarize_discussion(mock_model, messages)

        # Check the call to ainvoke included the conversation
        call_args = mock_model.ainvoke.call_args
        invoke_messages = call_args[0][0]

        # Should have system message and human message with conversation
        assert len(invoke_messages) == 2
        assert isinstance(invoke_messages[0], SystemMessage)
        assert isinstance(invoke_messages[1], HumanMessage)
        assert "User message" in invoke_messages[1].content
        assert "Assistant response" in invoke_messages[1].content

    @pytest.mark.asyncio
    async def test_summarize_handles_missing_metadata(self) -> None:
        """summarize_discussion should handle responses without metadata."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        # No usage_metadata or response_metadata
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary"
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_summarize_handles_null_token_values(self) -> None:
        """summarize_discussion should handle None token values."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_response.usage_metadata = {"total_tokens": None}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary"
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_summarize_handles_usage_metadata_attribute(self) -> None:
        """summarize_discussion should extract tokens from usage_metadata attribute."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        # Ollama-style: usage_metadata as attribute on AIMessage
        mock_response.usage_metadata = {"total_tokens": 200}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        _summary, tokens = await summarize_discussion(mock_model, messages)

        assert tokens == 200

    @pytest.mark.asyncio
    async def test_summarize_handles_openai_response_metadata(self) -> None:
        """summarize_discussion should extract tokens from response_metadata."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        # OpenAI-style: token_usage in response_metadata
        mock_response.response_metadata = {"token_usage": {"total_tokens": 150}}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        _summary, tokens = await summarize_discussion(mock_model, messages)

        assert tokens == 150

    @pytest.mark.asyncio
    async def test_summarize_prefers_usage_metadata_over_response_metadata(self) -> None:
        """summarize_discussion should prefer usage_metadata attribute."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        # Both present - should prefer usage_metadata
        mock_response.usage_metadata = {"total_tokens": 100}
        mock_response.response_metadata = {"token_usage": {"total_tokens": 200}}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        _summary, tokens = await summarize_discussion(mock_model, messages)

        assert tokens == 100  # From usage_metadata, not response_metadata

    @pytest.mark.asyncio
    async def test_summarize_handles_empty_messages(self) -> None:
        """summarize_discussion should handle empty message list."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Nothing to summarize")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        summary, _tokens = await summarize_discussion(mock_model, [])

        assert summary == "Nothing to summarize"

    @pytest.mark.asyncio
    async def test_summarize_handles_non_string_content(self) -> None:
        """summarize_discussion should handle non-string response content."""
        mock_model = MagicMock()
        mock_response = AIMessage(content=["list", "content"])  # type: ignore[arg-type]
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, _tokens = await summarize_discussion(mock_model, messages)

        # Should convert to string
        assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_summarize_formats_all_message_types(self) -> None:
        """summarize_discussion should format different message types."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            SystemMessage(content="System context"),
            HumanMessage(content="User input"),
            AIMessage(content="AI response"),
        ]

        await summarize_discussion(mock_model, messages)

        call_args = mock_model.ainvoke.call_args
        invoke_messages = call_args[0][0]
        conversation_text = invoke_messages[1].content

        assert "System: System context" in conversation_text
        assert "User: User input" in conversation_text
        assert "Assistant: AI response" in conversation_text


class TestFormatMessagesToolCalls:
    """Test _format_messages_for_summary with tool calls and results."""

    def test_format_messages_with_tool_calls(self) -> None:
        """Should include tool call name and arguments in formatted output."""
        ai_message = AIMessage(content="Let me search for relevant information.")
        ai_message.tool_calls = [
            {
                "name": "search_corpus",
                "args": {"query": "mystery genre conventions"},
                "id": "call_123",
            }
        ]
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "Let me search for relevant information." in result
        assert "[Tool Call: search_corpus]" in result
        assert "mystery genre conventions" in result

    def test_format_messages_with_tool_messages(self) -> None:
        """Should include tool result name and content in formatted output."""
        tool_message = ToolMessage(
            content='{"result": "success", "data": {"genre": "mystery"}}',
            name="search_corpus",
            tool_call_id="call_123",
        )
        messages = [tool_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Result: search_corpus]" in result
        assert '"result": "success"' in result
        assert '"genre": "mystery"' in result

    def test_format_preserves_research_context(self) -> None:
        """Should preserve full tool research chain in conversation context."""
        # Simulate a discuss phase with tool research
        messages = [
            HumanMessage(content="I want to write a mystery story"),
            AIMessage(content="Let me research mystery conventions."),
        ]
        # Add tool call to the AI message
        messages[1].tool_calls = [
            {
                "name": "search_corpus",
                "args": {"query": "mystery tropes"},
                "id": "call_456",
            }
        ]
        messages.append(
            ToolMessage(
                content='{"result": "success", "data": {"tropes": ["locked room", "red herring"]}}',
                name="search_corpus",
                tool_call_id="call_456",
            )
        )
        messages.append(
            AIMessage(
                content="Based on my research, I found common mystery tropes include locked room mysteries and red herrings."
            )
        )

        result = _format_messages_for_summary(messages)

        # Verify the full context is preserved
        assert "User: I want to write a mystery story" in result
        assert "Let me research mystery conventions" in result
        assert "[Tool Call: search_corpus]" in result
        assert "mystery tropes" in result
        assert "[Tool Result: search_corpus]" in result
        assert "locked room" in result
        assert "red herring" in result
        assert "Based on my research" in result

    def test_format_messages_with_multiple_tool_calls(self) -> None:
        """Should include all tool calls from a single AI message."""
        ai_message = AIMessage(content="")
        ai_message.tool_calls = [
            {"name": "search_corpus", "args": {"query": "genre"}, "id": "call_1"},
            {"name": "get_example", "args": {"category": "mystery"}, "id": "call_2"},
        ]
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Call: search_corpus]" in result
        assert "[Tool Call: get_example]" in result
        assert "genre" in result
        assert "mystery" in result

    def test_format_messages_ai_without_content_but_with_tool_calls(self) -> None:
        """Should handle AI message with only tool calls (no text content)."""
        ai_message = AIMessage(content="")
        ai_message.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "call_1"}]
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        # Should not have empty "Assistant: " prefix
        assert "Assistant: " not in result or result.count("Assistant:") == 0
        assert "[Tool Call: search]" in result

    def test_format_messages_tool_message_without_name(self) -> None:
        """Should handle tool message without name attribute."""
        tool_message = ToolMessage(
            content="some result",
            tool_call_id="call_123",
        )
        # Ensure name is None/missing
        tool_message.name = None
        messages = [tool_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Result: unknown_tool]" in result
        assert "some result" in result

    def test_format_messages_tool_call_without_args(self) -> None:
        """Should handle tool call with missing args."""
        ai_message = AIMessage(content="")
        ai_message.tool_calls = [{"name": "simple_tool", "id": "call_1"}]
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Call: simple_tool]" in result
        # Empty args should render as {}
        assert "{}" in result
