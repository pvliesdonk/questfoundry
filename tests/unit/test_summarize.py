"""Tests for Summarize phase."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from questfoundry.agents.prompts import get_repair_seed_brief_prompt, get_summarize_prompt
from questfoundry.agents.summarize import (
    format_repair_errors,
    get_fuzzy_id_suggestions,
    repair_seed_brief,
    summarize_discussion,
)


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


class TestGetFuzzyIdSuggestions:
    """Test fuzzy ID matching."""

    def test_exact_match_returns_first(self) -> None:
        """Exact match should be returned first."""
        suggestions = get_fuzzy_id_suggestions("diary_truth", ["diary_truth", "diary_access"])
        assert suggestions[0] == "diary_truth"

    def test_close_match_found(self) -> None:
        """Close matches should be found."""
        suggestions = get_fuzzy_id_suggestions(
            "archive_access", ["archive_nature", "diary_truth", "mentor_trust"]
        )
        assert "archive_nature" in suggestions

    def test_no_match_returns_empty(self) -> None:
        """When no close match exists, return empty list."""
        suggestions = get_fuzzy_id_suggestions("completely_different", ["a", "b", "c"])
        assert suggestions == []

    def test_returns_multiple_suggestions(self) -> None:
        """Should return multiple suggestions when available."""
        suggestions = get_fuzzy_id_suggestions(
            "test_id", ["test_id_1", "test_id_2", "test_id_3"], n=3
        )
        assert len(suggestions) <= 3

    def test_empty_available_returns_empty(self) -> None:
        """Empty available list should return empty."""
        suggestions = get_fuzzy_id_suggestions("test", [])
        assert suggestions == []


class TestFormatRepairErrors:
    """Test error formatting for repair prompt."""

    @dataclass
    class MockError:
        """Mock error object for testing."""

        field_path: str
        issue: str
        available: list[str]
        provided: str

    def test_formats_single_error(self) -> None:
        """Should format a single error with all fields."""
        error = self.MockError(
            field_path="threads.0.tension_id",
            issue="Tension not found",
            available=["mentor_trust", "diary_truth"],
            provided="archive_access",
        )

        result = format_repair_errors([error])

        assert "### Error 1" in result
        assert "`threads.0.tension_id`" in result
        assert "`archive_access`" in result
        assert "Tension not found" in result
        assert "`mentor_trust`" in result
        assert "`diary_truth`" in result

    def test_includes_fuzzy_suggestion(self) -> None:
        """Should include fuzzy match suggestion when available."""
        error = self.MockError(
            field_path="initial_beats.0.thread_id",
            issue="Thread not found",
            available=["archive_nature", "diary_truth"],
            provided="archive_access",
        )

        result = format_repair_errors([error])

        # Should suggest archive_nature as closest match to archive_access
        assert "**Suggested replacement**" in result
        assert "archive_nature" in result

    def test_formats_multiple_errors(self) -> None:
        """Should format multiple errors with numbering."""
        errors = [
            self.MockError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["t1"],
                provided="bad_id",
            ),
            self.MockError(
                field_path="initial_beats.1.entities",
                issue="Entity not found",
                available=["e1", "e2"],
                provided="wrong_entity",
            ),
        ]

        result = format_repair_errors(errors)

        assert "### Error 1" in result
        assert "### Error 2" in result

    def test_truncates_long_available_list(self) -> None:
        """Should truncate available list if longer than 8."""
        error = self.MockError(
            field_path="field",
            issue="Not found",
            available=[f"id_{i}" for i in range(15)],
            provided="bad_id",
        )

        result = format_repair_errors([error])

        # Should show truncation message
        assert "... and 7 more" in result

    def test_handles_empty_available(self) -> None:
        """Should handle error with no available options."""
        error = self.MockError(
            field_path="field",
            issue="Not found",
            available=[],
            provided="bad_id",
        )

        result = format_repair_errors([error])

        assert "### Error 1" in result
        assert "**Available options**" not in result


class TestGetRepairSeedBriefPrompt:
    """Test repair brief prompt loading."""

    def test_prompt_loads_successfully(self) -> None:
        """Prompt should load from template file."""
        system_prompt, user_prompt = get_repair_seed_brief_prompt(
            valid_ids_context="Valid IDs: a, b, c",
            error_list="### Error 1\n- Invalid: x",
            brief="Test brief with x",
        )

        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_prompt_includes_valid_ids(self) -> None:
        """System prompt should include valid IDs context."""
        system_prompt, _ = get_repair_seed_brief_prompt(
            valid_ids_context="VALID IDS: entity_1, entity_2",
            error_list="errors",
            brief="brief",
        )

        assert "VALID IDS: entity_1, entity_2" in system_prompt

    def test_prompt_includes_errors(self) -> None:
        """System prompt should include error list."""
        system_prompt, _ = get_repair_seed_brief_prompt(
            valid_ids_context="ids",
            error_list="### Error 1\n- **Invalid ID**: `bad_id`",
            brief="brief",
        )

        assert "### Error 1" in system_prompt
        assert "`bad_id`" in system_prompt

    def test_user_prompt_includes_brief(self) -> None:
        """User prompt should include the brief to repair."""
        _, user_prompt = get_repair_seed_brief_prompt(
            valid_ids_context="ids",
            error_list="errors",
            brief="This is the brief with invalid IDs",
        )

        assert "This is the brief with invalid IDs" in user_prompt


class TestRepairSeedBrief:
    """Test repair_seed_brief function."""

    @dataclass
    class MockError:
        """Mock error for testing."""

        field_path: str
        issue: str
        available: list[str]
        provided: str

    @pytest.mark.asyncio
    async def test_repair_returns_repaired_brief_and_tokens(self) -> None:
        """repair_seed_brief should return repaired brief and token count."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Repaired brief content")
        mock_response.usage_metadata = {"total_tokens": 150}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        errors = [
            self.MockError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["mentor_trust"],
                provided="bad_tension",
            )
        ]

        repaired, tokens = await repair_seed_brief(
            model=mock_model,
            brief="Original brief with bad_tension",
            errors=errors,
            valid_ids_context="Valid IDs: mentor_trust",
        )

        assert repaired == "Repaired brief content"
        assert tokens == 150

    @pytest.mark.asyncio
    async def test_repair_calls_model_with_repair_prompt(self) -> None:
        """repair_seed_brief should use repair prompt structure."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Repaired")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        errors = [
            self.MockError(
                field_path="field",
                issue="Issue",
                available=["valid"],
                provided="invalid",
            )
        ]

        await repair_seed_brief(
            model=mock_model,
            brief="Brief",
            errors=errors,
            valid_ids_context="IDs",
        )

        # Should call ainvoke with 2 messages (system + user)
        call_args = mock_model.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_repair_handles_missing_token_metadata(self) -> None:
        """repair_seed_brief should handle responses without token metadata."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Repaired")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        errors = [
            self.MockError(
                field_path="f",
                issue="i",
                available=["a"],
                provided="p",
            )
        ]

        _, tokens = await repair_seed_brief(
            model=mock_model,
            brief="Brief",
            errors=errors,
            valid_ids_context="IDs",
        )

        assert tokens == 0

    @pytest.mark.asyncio
    async def test_repair_includes_error_in_system_prompt(self) -> None:
        """repair_seed_brief should include formatted errors in prompt."""
        mock_model = MagicMock()
        mock_response = AIMessage(content="Repaired")
        mock_model.ainvoke = AsyncMock(return_value=mock_response)

        errors = [
            self.MockError(
                field_path="threads.0.tension_id",
                issue="Tension not found in BRAINSTORM",
                available=["mentor_trust", "diary_truth"],
                provided="invented_tension",
            )
        ]

        await repair_seed_brief(
            model=mock_model,
            brief="Brief with invented_tension",
            errors=errors,
            valid_ids_context="Valid tension IDs: mentor_trust, diary_truth",
        )

        call_args = mock_model.ainvoke.call_args
        system_message = call_args[0][0][0]

        # System prompt should contain the error details
        assert "`invented_tension`" in system_message.content
        assert "Tension not found" in system_message.content
