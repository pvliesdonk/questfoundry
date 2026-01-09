"""Tests for Summarize phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from questfoundry.agents.prompts import get_summarize_prompt
from questfoundry.agents.summarize import summarize_discussion


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
