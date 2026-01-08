"""Tests for Discuss agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.agents.discuss import create_discuss_agent, run_discuss_phase
from questfoundry.agents.prompts import get_discuss_prompt


class TestGetDiscussPrompt:
    """Test prompt template creation."""

    def test_prompt_includes_user_prompt(self) -> None:
        """Prompt should contain the user's story idea."""
        prompt = get_discuss_prompt(user_prompt="A mystery in space")

        assert "A mystery in space" in prompt

    def test_prompt_includes_core_discussion_topics(self) -> None:
        """Prompt should include key discussion areas."""
        prompt = get_discuss_prompt(user_prompt="Test")

        assert "Genre and tone" in prompt
        assert "Themes and motifs" in prompt
        assert "Target audience" in prompt
        assert "Scope and complexity" in prompt

    def test_prompt_includes_tools_section_when_available(self) -> None:
        """Prompt should list research tools when available."""
        prompt = get_discuss_prompt(
            user_prompt="Test",
            research_tools_available=True,
        )

        assert "search_corpus" in prompt
        assert "web_search" in prompt
        assert "Research Tools Available" in prompt

    def test_prompt_excludes_tools_section_when_unavailable(self) -> None:
        """Prompt should not list research tools when unavailable."""
        prompt = get_discuss_prompt(
            user_prompt="Test",
            research_tools_available=False,
        )

        assert "search_corpus" not in prompt
        assert "Research Tools Available" not in prompt

    def test_prompt_includes_guidelines(self) -> None:
        """Prompt should include interaction guidelines."""
        prompt = get_discuss_prompt(user_prompt="Test")

        assert "clarifying questions" in prompt
        assert "conversational" in prompt


class TestCreateDiscussAgent:
    """Test agent creation."""

    @patch("langchain.agents.create_agent")
    def test_creates_agent_with_tools(self, mock_create: MagicMock) -> None:
        """Agent should be created with provided tools."""
        mock_model = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]

        create_discuss_agent(mock_model, mock_tools, "Test prompt")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["tools"] == mock_tools

    @patch("langchain.agents.create_agent")
    def test_creates_agent_without_tools(self, mock_create: MagicMock) -> None:
        """Agent should be created with tools=None when no tools provided."""
        mock_model = MagicMock()

        create_discuss_agent(mock_model, [], "Test prompt")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tools"] is None

    @patch("langchain.agents.create_agent")
    def test_creates_agent_with_system_prompt(self, mock_create: MagicMock) -> None:
        """Agent should receive a system prompt containing user's idea."""
        mock_model = MagicMock()

        create_discuss_agent(mock_model, [], "A cozy mystery story")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert "A cozy mystery story" in call_kwargs["system_prompt"]


class TestRunDiscussPhase:
    """Test discuss phase execution."""

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_returns_messages(self, mock_create: MagicMock) -> None:
        """run_discuss_phase should return messages from agent."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="Response"),
            ]
        }
        mock_create.return_value = mock_agent

        messages, _calls, _tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test prompt",
        )

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_invokes_with_initial_message(
        self, mock_create: MagicMock
    ) -> None:
        """Agent should be invoked with user's prompt as initial message."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": []}
        mock_create.return_value = mock_agent

        await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="My story idea",
        )

        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "My story idea"

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_respects_max_iterations(self, mock_create: MagicMock) -> None:
        """Agent should be invoked with recursion_limit from max_iterations."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": []}
        mock_create.return_value = mock_agent

        await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
            max_iterations=50,
        )

        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args
        config = call_args[1]["config"]
        assert config["recursion_limit"] == 50

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_extracts_token_usage(self, mock_create: MagicMock) -> None:
        """run_discuss_phase should extract token metrics from response metadata."""
        ai_msg = AIMessage(content="Response")
        ai_msg.response_metadata = {"token_usage": {"total_tokens": 150}}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Test"),
                ai_msg,
            ]
        }
        mock_create.return_value = mock_agent

        _messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        assert calls == 1
        assert tokens == 150

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_handles_missing_metadata(self, mock_create: MagicMock) -> None:
        """run_discuss_phase should handle AIMessages without response_metadata."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="Response"),  # No metadata
            ]
        }
        mock_create.return_value = mock_agent

        _messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        # Should not raise, just return zero counts
        assert calls == 0
        assert tokens == 0

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_handles_empty_messages(self, mock_create: MagicMock) -> None:
        """run_discuss_phase should handle empty message list."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": []}
        mock_create.return_value = mock_agent

        messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        assert len(messages) == 0
        assert calls == 0
        assert tokens == 0

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_handles_missing_messages_key(
        self, mock_create: MagicMock
    ) -> None:
        """run_discuss_phase should handle result without messages key."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {}  # No messages key
        mock_create.return_value = mock_agent

        messages, _calls, _tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        assert len(messages) == 0
