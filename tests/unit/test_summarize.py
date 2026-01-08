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
        mock_response.response_metadata = {"token_usage": {"total_tokens": 100}}
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            HumanMessage(content="I want to write a mystery"),
            AIMessage(content="Great! Tell me more about the setting."),
        ]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary of discussion"
        assert tokens == 100

    @pytest.mark.asyncio
    async def test_summarize_uses_lower_temperature(self) -> None:
        """summarize_discussion should bind model with lower temperature."""
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_bound.ainvoke = AsyncMock(return_value=mock_response)
        mock_model.bind.return_value = mock_bound

        messages = [HumanMessage(content="Test")]

        await summarize_discussion(mock_model, messages)

        mock_model.bind.assert_called_once_with(temperature=0.3)

    @pytest.mark.asyncio
    async def test_summarize_fallback_when_bind_fails(self) -> None:
        """summarize_discussion should fallback when bind not supported."""
        mock_model = MagicMock()
        mock_model.bind.side_effect = AttributeError("no bind")
        mock_response = AIMessage(content="Summary")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, _tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary"
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_includes_conversation_in_prompt(self) -> None:
        """summarize_discussion should include formatted conversation."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            HumanMessage(content="User message"),
            AIMessage(content="Assistant response"),
        ]

        await summarize_discussion(mock_model, messages)

        # Check the call to ainvoke included the conversation
        call_args = mock_model.bind.return_value.ainvoke.call_args
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
        # No response_metadata
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary"
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_summarize_handles_null_token_values(self) -> None:
        """summarize_discussion should handle None token values."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_response.response_metadata = {"token_usage": {"total_tokens": None}}
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, tokens = await summarize_discussion(mock_model, messages)

        assert summary == "Summary"
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_summarize_handles_usage_metadata_key(self) -> None:
        """summarize_discussion should extract tokens from usage_metadata."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_response.response_metadata = {"usage_metadata": {"total_tokens": 200}}
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        _summary, tokens = await summarize_discussion(mock_model, messages)

        assert tokens == 200

    @pytest.mark.asyncio
    async def test_summarize_handles_empty_messages(self) -> None:
        """summarize_discussion should handle empty message list."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Nothing to summarize")
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        summary, _tokens = await summarize_discussion(mock_model, [])

        assert summary == "Nothing to summarize"

    @pytest.mark.asyncio
    async def test_summarize_handles_non_string_content(self) -> None:
        """summarize_discussion should handle non-string response content."""
        mock_model = MagicMock()
        mock_response = AIMessage(content=["list", "content"])  # type: ignore[arg-type]
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        summary, _tokens = await summarize_discussion(mock_model, messages)

        # Should convert to string
        assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_summarize_formats_all_message_types(self) -> None:
        """summarize_discussion should format different message types."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Summary")
        mock_model.bind.return_value.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            SystemMessage(content="System context"),
            HumanMessage(content="User input"),
            AIMessage(content="AI response"),
        ]

        await summarize_discussion(mock_model, messages)

        call_args = mock_model.bind.return_value.ainvoke.call_args
        invoke_messages = call_args[0][0]
        conversation_text = invoke_messages[1].content

        assert "System: System context" in conversation_text
        assert "User: User input" in conversation_text
        assert "Assistant: AI response" in conversation_text
