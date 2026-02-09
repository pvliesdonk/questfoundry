"""Tests for Discuss agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.agents.discuss import create_discuss_agent, run_discuss_phase
from questfoundry.agents.prompts import get_discuss_prompt


class TestGetDiscussPrompt:
    """Test prompt template creation."""

    def test_prompt_includes_core_discussion_topics(self) -> None:
        """Prompt should include key creative questions as deliverables."""
        prompt = get_discuss_prompt()

        assert "genre and subgenre" in prompt
        assert "emotional tone" in prompt
        assert "target audience" in prompt.lower()
        assert "themes" in prompt

    def test_prompt_includes_tools_section_when_available(self) -> None:
        """Prompt should list research tools when available."""
        prompt = get_discuss_prompt(research_tools_available=True)

        assert "search_corpus" in prompt
        assert "web_search" in prompt
        assert "Craft Corpus Research (REQUIRED)" in prompt

    def test_prompt_excludes_tools_section_when_unavailable(self) -> None:
        """Prompt should not list research tools when unavailable."""
        prompt = get_discuss_prompt(research_tools_available=False)

        assert "search_corpus" not in prompt
        assert "Craft Corpus Research" not in prompt

    def test_prompt_includes_guidelines(self) -> None:
        """Prompt should include interaction guidelines."""
        prompt = get_discuss_prompt()

        assert "clarifying questions" in prompt
        assert "conversational" in prompt

    def test_prompt_loads_from_template(self) -> None:
        """Prompt should load from external template file."""
        # If template loading fails, get_discuss_prompt would raise
        # TemplateNotFoundError - this tests the integration works
        prompt = get_discuss_prompt()

        # Verify it returns a non-empty string
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should have substantial content


class TestCreateDiscussAgent:
    """Test agent creation."""

    @patch("langchain.agents.create_agent")
    def test_creates_agent_with_tools(self, mock_create: MagicMock) -> None:
        """Agent should be created with provided tools."""
        mock_model = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]

        create_discuss_agent(mock_model, mock_tools)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["tools"] == mock_tools

    @patch("langchain.agents.create_agent")
    def test_creates_agent_without_tools(self, mock_create: MagicMock) -> None:
        """Agent should be created with tools=None when no tools provided."""
        mock_model = MagicMock()

        create_discuss_agent(mock_model, [])

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["tools"] is None

    @patch("langchain.agents.create_agent")
    def test_creates_agent_with_system_prompt(self, mock_create: MagicMock) -> None:
        """Agent should receive a system prompt with discussion guidelines."""
        mock_model = MagicMock()

        create_discuss_agent(mock_model, [])

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        # System prompt should contain discussion guidelines, not user prompt
        assert "creative collaborator" in call_kwargs["system_prompt"]
        assert "genre and subgenre" in call_kwargs["system_prompt"]


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
        """run_discuss_phase should handle AIMessages without usage metadata."""
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

        # Should count the AIMessage but return zero tokens
        assert calls == 1
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

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_handles_null_token_values(
        self, mock_create: MagicMock
    ) -> None:
        """run_discuss_phase should handle token_usage with None values."""
        ai_msg = AIMessage(content="Response")
        ai_msg.response_metadata = {"token_usage": {"total_tokens": None}}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
        mock_create.return_value = mock_agent

        _messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        # Should handle None gracefully, counting the call but treating None as 0
        assert calls == 1
        assert tokens == 0

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_run_discuss_phase_handles_usage_metadata_attribute(
        self, mock_create: MagicMock
    ) -> None:
        """run_discuss_phase should extract tokens from usage_metadata attribute (Ollama)."""
        ai_msg = AIMessage(content="Response")
        ai_msg.usage_metadata = {"total_tokens": 200}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [ai_msg]}
        mock_create.return_value = mock_agent

        _messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
        )

        assert calls == 1
        assert tokens == 200


class TestInteractiveMode:
    """Tests for interactive multi-turn conversation mode."""

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_interactive_multi_turn_conversation(self, mock_create: MagicMock) -> None:
        """Interactive mode processes multiple turns before /done."""
        mock_agent = AsyncMock()
        # Each turn returns the AI response
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Turn response")]}
        mock_create.return_value = mock_agent

        # User provides two messages then exits
        inputs = iter(["follow-up 1", "follow-up 2", "/done"])
        user_input_fn = AsyncMock(side_effect=lambda: next(inputs))
        on_assistant = MagicMock()

        _messages, calls, _tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Initial idea",
            interactive=True,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant,
        )

        # 3 turns: initial + 2 follow-ups (third input is /done, exits before invoke)
        assert mock_agent.ainvoke.call_count == 3
        assert calls == 3  # One AIMessage per turn

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_cmd", ["/done", "/quit", "/exit"])
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_interactive_exit_on_done_command(
        self, mock_create: MagicMock, exit_cmd: str
    ) -> None:
        """Interactive mode exits on /done, /quit, /exit commands."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Response")]}
        mock_create.return_value = mock_agent

        user_input_fn = AsyncMock(return_value=exit_cmd)
        on_assistant = MagicMock()

        await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
            interactive=True,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant,
        )

        # Only 1 turn (initial) — exit command prevents second invoke
        assert mock_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_input", ["", "  ", None])
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_interactive_exit_on_empty_input(
        self, mock_create: MagicMock, empty_input: str | None
    ) -> None:
        """Interactive mode exits on empty input or None."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Response")]}
        mock_create.return_value = mock_agent

        user_input_fn = AsyncMock(return_value=empty_input)
        on_assistant = MagicMock()

        await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
            interactive=True,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant,
        )

        assert mock_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_interactive_without_input_fn_raises(self) -> None:
        """Interactive mode without user_input_fn raises ValueError."""
        with pytest.raises(ValueError, match="user_input_fn"):
            await run_discuss_phase(
                model=MagicMock(),
                tools=[],
                user_prompt="Test",
                interactive=True,
                user_input_fn=None,
                on_assistant_message=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_interactive_without_assistant_callback_raises(self) -> None:
        """Interactive mode without on_assistant_message raises ValueError."""
        with pytest.raises(ValueError, match="on_assistant_message"):
            await run_discuss_phase(
                model=MagicMock(),
                tools=[],
                user_prompt="Test",
                interactive=True,
                user_input_fn=AsyncMock(return_value="/done"),
                on_assistant_message=None,
            )

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_interactive_callbacks_invoked(self, mock_create: MagicMock) -> None:
        """on_assistant_message, on_llm_start, on_llm_end are called each turn."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"messages": [AIMessage(content="Hello!")]}
        mock_create.return_value = mock_agent

        user_input_fn = AsyncMock(return_value="/done")
        on_assistant = MagicMock()
        on_start = MagicMock()
        on_end = MagicMock()

        await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
            interactive=True,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant,
            on_llm_start=on_start,
            on_llm_end=on_end,
        )

        # All callbacks invoked once for the single turn
        on_assistant.assert_called_once_with("Hello!")
        on_start.assert_called_once_with("discuss")
        on_end.assert_called_once_with("discuss")

    @pytest.mark.asyncio
    @patch("questfoundry.agents.discuss.create_discuss_agent")
    async def test_interactive_metrics_across_turns(self, mock_create: MagicMock) -> None:
        """Token and call counts aggregate correctly across turns."""
        ai_msg_1 = AIMessage(content="Turn 1")
        ai_msg_1.usage_metadata = {"total_tokens": 100}
        ai_msg_2 = AIMessage(content="Turn 2")
        ai_msg_2.usage_metadata = {"total_tokens": 150}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = [
            {"messages": [ai_msg_1]},
            {"messages": [ai_msg_2]},
        ]
        mock_create.return_value = mock_agent

        inputs = iter(["continue", "/done"])
        user_input_fn = AsyncMock(side_effect=lambda: next(inputs))
        on_assistant = MagicMock()

        _messages, calls, tokens = await run_discuss_phase(
            model=MagicMock(),
            tools=[],
            user_prompt="Test",
            interactive=True,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant,
        )

        assert calls == 2
        assert tokens == 250
