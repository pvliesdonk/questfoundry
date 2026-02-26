"""Tests for Summarize phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.agents.prompts import get_seed_section_summarize_prompts, get_summarize_prompt
from questfoundry.agents.summarize import (
    SEED_SUMMARY_SECTIONS,
    _extract_tokens,
    _format_dilemma_answers_from_graph,
    _format_messages_for_summary,
    summarize_discussion,
    summarize_seed_chunked,
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


class TestFormatMessagesToolCalls:
    """Test _format_messages_for_summary with tool calls and results."""

    def test_format_messages_with_tool_calls(self) -> None:
        """Should include tool call name and arguments in formatted output."""
        ai_message = AIMessage(
            content="Let me search for relevant information.",
            tool_calls=[
                {
                    "name": "search_corpus",
                    "args": {"query": "mystery genre conventions"},
                    "id": "call_123",
                }
            ],
        )
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "Let me search for relevant information." in result
        assert "[Tool Call: search_corpus]" in result
        assert "mystery genre conventions" in result

    def test_format_messages_with_tool_messages(self) -> None:
        """Should include tool result name and extracted content in formatted output."""
        tool_message = ToolMessage(
            content='{"result": "success", "data": {"genre": "mystery"}}',
            name="search_corpus",
            tool_call_id="call_123",
        )
        messages = [tool_message]

        result = _format_messages_for_summary(messages)

        # Tool results now use [Research: ...] format with content-only extraction
        assert "[Research: search_corpus]" in result
        # Extracts just the data content, not the wrapper
        assert "genre" in result
        assert "mystery" in result

    def test_format_preserves_research_context(self) -> None:
        """Should preserve full tool research chain in conversation context."""
        # Simulate a discuss phase with tool research
        messages = [
            HumanMessage(content="I want to write a mystery story"),
            AIMessage(
                content="Let me research mystery conventions.",
                tool_calls=[
                    {
                        "name": "search_corpus",
                        "args": {"query": "mystery tropes"},
                        "id": "call_456",
                    }
                ],
            ),
            ToolMessage(
                content='{"result": "success", "data": {"tropes": ["locked room", "red herring"]}}',
                name="search_corpus",
                tool_call_id="call_456",
            ),
            AIMessage(
                content="Based on my research, I found common mystery tropes include locked room mysteries and red herrings."
            ),
        ]

        result = _format_messages_for_summary(messages)

        # Verify the full context is preserved
        assert "User: I want to write a mystery story" in result
        assert "Let me research mystery conventions" in result
        assert "[Tool Call: search_corpus]" in result
        assert "mystery tropes" in result
        # Tool results now use [Research: ...] format with content-only extraction
        assert "[Research: search_corpus]" in result
        assert "locked room" in result
        assert "red herring" in result
        assert "Based on my research" in result

    def test_format_messages_with_multiple_tool_calls(self) -> None:
        """Should include all tool calls from a single AI message."""
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_corpus", "args": {"query": "genre"}, "id": "call_1"},
                {"name": "get_example", "args": {"category": "mystery"}, "id": "call_2"},
            ],
        )
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Call: search_corpus]" in result
        assert "[Tool Call: get_example]" in result
        assert "genre" in result
        assert "mystery" in result

    def test_format_messages_ai_without_content_but_with_tool_calls(self) -> None:
        """Should handle AI message with only tool calls (no text content)."""
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
        )
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        # No "Assistant:" prefix should appear when there's no content
        assert "Assistant:" not in result
        assert "[Tool Call: search]" in result

    def test_format_messages_tool_message_without_name(self) -> None:
        """Should handle tool message without name attribute."""
        # The 'name' attribute defaults to None when not provided
        tool_message = ToolMessage(
            content="some result",
            tool_call_id="call_123",
        )
        messages = [tool_message]

        result = _format_messages_for_summary(messages)

        # Tool results now use [Research: ...] format
        assert "[Research: unknown_tool]" in result
        assert "some result" in result

    def test_format_messages_tool_call_without_args(self) -> None:
        """Should handle tool call with missing args defensively."""
        # LangChain validates tool_calls in constructor, so we assign after creation
        # to test defensive handling of potentially malformed data
        ai_message = AIMessage(content="")
        ai_message.tool_calls = [{"name": "simple_tool", "id": "call_1"}]  # type: ignore[list-item]
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        assert "[Tool Call: simple_tool]" in result
        # Empty args should render as {}
        assert "{}" in result

    def test_format_messages_skips_whitespace_only_content(self) -> None:
        """Should skip AI messages with whitespace-only content."""
        ai_message = AIMessage(
            content="   ",  # Whitespace only
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
        )
        messages = [ai_message]

        result = _format_messages_for_summary(messages)

        # No "Assistant:   " line should appear for whitespace-only content
        assert "Assistant:" not in result
        # Tool calls should still be included
        assert "[Tool Call: search]" in result
        assert '"q": "test"' in result


class TestExtractTokens:
    """Test _extract_tokens helper."""

    def test_extract_from_usage_metadata(self) -> None:
        """Should extract tokens from usage_metadata (Ollama-style)."""
        msg = AIMessage(content="test")
        msg.usage_metadata = {"total_tokens": 42}
        assert _extract_tokens(msg) == 42

    def test_extract_from_response_metadata(self) -> None:
        """Should extract tokens from response_metadata (OpenAI-style)."""
        msg = AIMessage(content="test")
        msg.response_metadata = {"token_usage": {"total_tokens": 99}}
        assert _extract_tokens(msg) == 99

    def test_returns_zero_when_no_metadata(self) -> None:
        """Should return 0 when no token metadata is available."""
        msg = AIMessage(content="test")
        assert _extract_tokens(msg) == 0

    def test_handles_none_total_tokens(self) -> None:
        """Should return 0 when total_tokens is None."""
        msg = AIMessage(content="test")
        msg.usage_metadata = {"total_tokens": None}
        assert _extract_tokens(msg) == 0


def _make_mock_graph() -> MagicMock:
    """Build a mock graph with brainstorm entities, dilemmas, and answers.

    Creates 2 entities and 1 dilemma with 2 answers for testing.
    """
    graph = MagicMock()

    entities = {
        "character::hero": {
            "raw_id": "hero",
            "entity_type": "character",
            "concept": "Protagonist",
        },
        "location::castle": {
            "raw_id": "castle",
            "entity_type": "location",
            "concept": "Main setting",
        },
    }
    dilemmas = {
        "dilemma::trust_or_betray": {
            "raw_id": "trust_or_betray",
            "question": "Trust or betray?",
        },
    }

    def get_nodes_by_type(t: str) -> dict:
        if t == "entity":
            return entities
        if t == "dilemma":
            return dilemmas
        return {}

    graph.get_nodes_by_type.side_effect = get_nodes_by_type

    # Answer edges and nodes
    answer_edges = [
        {"from": "dilemma::trust_or_betray", "to": "answer::trust", "type": "has_answer"},
        {"from": "dilemma::trust_or_betray", "to": "answer::betray", "type": "has_answer"},
    ]
    answer_nodes = {
        "answer::trust": {"raw_id": "trust", "is_canonical": True},
        "answer::betray": {"raw_id": "betray", "is_canonical": False},
    }

    def get_edges(*, edge_type: str = "", from_id: str = "") -> list:
        if edge_type == "has_answer":
            if from_id:
                return [e for e in answer_edges if e["from"] == from_id]
            return answer_edges
        return []

    graph.get_edges.side_effect = get_edges

    def get_node(node_id: str) -> dict | None:
        return answer_nodes.get(node_id)

    graph.get_node.side_effect = get_node

    return graph


class TestFormatDilemmaAnswersFromGraph:
    """Test _format_dilemma_answers_from_graph."""

    def test_formats_dilemma_answers(self) -> None:
        """Should format dilemma answer IDs from graph."""
        graph = _make_mock_graph()
        result = _format_dilemma_answers_from_graph(graph)

        assert "dilemma::trust_or_betray" in result
        assert "`betray`" in result
        assert "`trust`" in result

    def test_handles_empty_graph(self) -> None:
        """Should return placeholder for graph with no dilemmas."""
        graph = MagicMock()
        graph.get_nodes_by_type.return_value = {}
        result = _format_dilemma_answers_from_graph(graph)
        assert result == "(No dilemmas)"

    def test_marks_default_answer(self) -> None:
        """Should mark the default answer."""
        graph = _make_mock_graph()
        result = _format_dilemma_answers_from_graph(graph)
        assert "(default)" in result


class TestGetSeedSectionSummarizePrompts:
    """Test get_seed_section_summarize_prompts loader."""

    def test_returns_all_sections(self) -> None:
        """Should return prompts for all 5 sections."""
        prompts = get_seed_section_summarize_prompts(
            entity_count=2,
            dilemma_count=1,
            entity_manifest="- hero\n- castle",
            dilemma_manifest="- trust_or_betray",
        )
        assert set(prompts.keys()) == set(SEED_SUMMARY_SECTIONS)

    def test_entities_prompt_has_manifest(self) -> None:
        """Entities section should include entity manifest."""
        prompts = get_seed_section_summarize_prompts(
            entity_count=2,
            entity_manifest="- character::hero\n- location::castle",
        )
        assert "character::hero" in prompts["entities"]
        assert "2" in prompts["entities"]

    def test_dilemmas_prompt_has_answers(self) -> None:
        """Dilemmas section should include answer IDs."""
        prompts = get_seed_section_summarize_prompts(
            dilemma_count=1,
            dilemma_manifest="- dilemma::trust_or_betray",
            dilemma_answers="- `dilemma::trust_or_betray` -> [trust, betray]",
        )
        assert "trust_or_betray" in prompts["dilemmas"]
        assert "trust" in prompts["dilemmas"]

    def test_all_prompts_are_nonempty_strings(self) -> None:
        """All section prompts should be non-empty strings."""
        prompts = get_seed_section_summarize_prompts()
        for section, prompt in prompts.items():
            assert isinstance(prompt, str), f"{section} should be a string"
            assert len(prompt) > 50, f"{section} should be non-trivial"


class TestSummarizeSeedChunked:
    """Test summarize_seed_chunked function."""

    @pytest.mark.asyncio
    async def test_returns_all_sections(self) -> None:
        """Should return briefs for all 5 sections."""
        graph = _make_mock_graph()
        mock_model = MagicMock()

        # Each call returns a different brief
        responses = []
        for section in SEED_SUMMARY_SECTIONS:
            msg = AIMessage(content=f"Brief for {section}")
            msg.usage_metadata = {"total_tokens": 50}
            responses.append(msg)

        mock_model.ainvoke = AsyncMock(side_effect=responses)

        messages = [HumanMessage(content="Let's triage the brainstorm")]

        briefs, tokens = await summarize_seed_chunked(
            model=mock_model,
            messages=messages,
            graph=graph,
        )

        assert set(briefs.keys()) == set(SEED_SUMMARY_SECTIONS)
        assert tokens == 250  # 5 sections * 50 tokens

    @pytest.mark.asyncio
    async def test_makes_five_sequential_calls(self) -> None:
        """Should make exactly 5 model calls (one per section)."""
        graph = _make_mock_graph()
        mock_model = MagicMock()

        response = AIMessage(content="Section brief")
        response.usage_metadata = {"total_tokens": 10}
        mock_model.ainvoke = AsyncMock(return_value=response)

        messages = [HumanMessage(content="Test")]

        await summarize_seed_chunked(
            model=mock_model,
            messages=messages,
            graph=graph,
        )

        assert mock_model.ainvoke.call_count == 5

    @pytest.mark.asyncio
    async def test_each_call_has_system_and_human_messages(self) -> None:
        """Each call should have a SystemMessage and HumanMessage."""
        graph = _make_mock_graph()
        mock_model = MagicMock()

        captured_calls: list[list] = []

        async def capture_calls(call_messages, config=None):  # noqa: ARG001
            captured_calls.append(list(call_messages))
            msg = AIMessage(content="Brief")
            msg.usage_metadata = {"total_tokens": 10}
            return msg

        mock_model.ainvoke = capture_calls

        messages = [HumanMessage(content="Discussion content")]

        await summarize_seed_chunked(
            model=mock_model,
            messages=messages,
            graph=graph,
        )

        assert len(captured_calls) == 5
        for call_msgs in captured_calls:
            assert len(call_msgs) == 2
            assert isinstance(call_msgs[0], SystemMessage)
            assert isinstance(call_msgs[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_later_calls_accumulate_prior_context(self) -> None:
        """Later calls should receive all prior sections' output as context."""
        graph = _make_mock_graph()
        mock_model = MagicMock()

        captured_calls: list[list] = []

        async def capture_calls(call_messages, config=None):  # noqa: ARG001
            captured_calls.append(list(call_messages))
            idx = len(captured_calls) - 1
            section = list(SEED_SUMMARY_SECTIONS)[idx]
            msg = AIMessage(
                content="Entities are retained" if section == "entities" else f"Brief for {section}"
            )
            msg.usage_metadata = {"total_tokens": 10}
            return msg

        mock_model.ainvoke = capture_calls

        messages = [HumanMessage(content="Test")]

        await summarize_seed_chunked(
            model=mock_model,
            messages=messages,
            graph=graph,
        )

        # First call (entities) should NOT have prior decisions
        entities_msg = captured_calls[0][1].content
        assert "Prior Decisions" not in entities_msg

        # Second call (dilemmas) SHOULD have entities prior context
        dilemmas_msg = captured_calls[1][1].content
        assert "Prior Decisions" in dilemmas_msg
        assert "Entities are retained" in dilemmas_msg

        # Third call (paths) SHOULD have both entities and dilemmas context
        paths_msg = captured_calls[2][1].content
        assert "Prior Decisions" in paths_msg
        assert "Entities are retained" in paths_msg
        assert "Brief for dilemmas" in paths_msg

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self) -> None:
        """Should handle empty message list."""
        graph = _make_mock_graph()
        mock_model = MagicMock()

        response = AIMessage(content="Brief")
        response.usage_metadata = {"total_tokens": 10}
        mock_model.ainvoke = AsyncMock(return_value=response)

        briefs, _tokens = await summarize_seed_chunked(
            model=mock_model,
            messages=[],
            graph=graph,
        )

        assert len(briefs) == 5
