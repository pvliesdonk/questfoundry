"""Tests for AgentRuntime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.agent import AgentRuntime
from questfoundry.runtime.models.base import Agent, Studio
from questfoundry.runtime.providers import (
    ContextOverflowError,
    LLMMessage,
    LLMResponse,
    OllamaProvider,
    StreamChunk,
)
from questfoundry.runtime.session import Session
from questfoundry.runtime.storage import Project


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock(spec=OllamaProvider)
    provider.name = "mock"
    return provider


@pytest.fixture
def basic_studio() -> Studio:
    """Create a basic studio with agents."""
    return Studio(
        id="test_studio",
        name="Test Studio",
        agents=[
            Agent(
                id="showrunner",
                name="Showrunner",
                description="The orchestrator",
                archetypes=["orchestrator"],
                is_entry_agent=True,
            ),
            Agent(
                id="lorekeeper",
                name="Lorekeeper",
                description="Keeper of canon",
                archetypes=["curator"],
            ),
        ],
    )


@pytest.fixture
def project(tmp_path) -> Project:
    """Create a test project."""
    project_path = tmp_path / "test_project"
    project = Project.create(
        path=project_path,
        name="Test Project",
        studio_id="test_studio",
    )
    yield project
    project.close()


@pytest.fixture
def session(project: Project) -> Session:
    """Create a test session."""
    return Session.create(project=project, entry_agent="showrunner")


class TestAgentRuntimeBasics:
    """Tests for AgentRuntime initialization and basic operations."""

    def test_create_runtime(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """AgentRuntime can be created."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            model="qwen3:8b",
        )

        assert runtime._provider == mock_provider
        assert runtime._studio == basic_studio
        assert runtime._model == "qwen3:8b"

    def test_get_agent(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """get_agent finds agents by ID."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        agent = runtime.get_agent("showrunner")
        assert agent is not None
        assert agent.id == "showrunner"
        assert agent.name == "Showrunner"

    def test_get_agent_not_found(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """get_agent returns None for unknown ID."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        agent = runtime.get_agent("nonexistent")
        assert agent is None

    def test_get_entry_agent(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """get_entry_agent finds first entry agent."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        agent = runtime.get_entry_agent()
        assert agent is not None
        assert agent.id == "showrunner"
        assert agent.is_entry_agent is True


class TestBuildMessages:
    """Tests for message building."""

    def test_build_messages_basic(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """build_messages creates system and user messages."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        messages = runtime.build_messages(agent, "Hello!")

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "Showrunner" in messages[0].content
        assert messages[1].role == "user"
        assert messages[1].content == "Hello!"

    def test_build_messages_with_history(
        self, mock_provider: MagicMock, basic_studio: Studio
    ) -> None:
        """build_messages includes conversation history."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]

        messages = runtime.build_messages(agent, "Second message", history=history)

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "First message"
        assert messages[2].role == "assistant"
        assert messages[3].role == "user"
        assert messages[3].content == "Second message"


class TestContextValidation:
    """Tests for context size validation."""

    def test_validate_within_limit(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """Validation passes when within limit."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            context_limit=10000,
        )

        messages = [
            LLMMessage(role="system", content="Short prompt"),
            LLMMessage(role="user", content="Hello"),
        ]

        # Should not raise
        runtime.validate_context_size(messages)

    def test_validate_exceeds_limit(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """Validation raises when exceeding limit."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            context_limit=10,  # Very small limit
        )

        messages = [
            LLMMessage(role="system", content="A" * 1000),  # Way over limit
        ]

        with pytest.raises(ContextOverflowError, match="exceeds"):
            runtime.validate_context_size(messages)

    def test_validate_no_limit(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """Validation is skipped when no limit set."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            context_limit=None,
        )

        messages = [
            LLMMessage(role="system", content="A" * 100000),
        ]

        # Should not raise even with huge message
        runtime.validate_context_size(messages)


class TestAgentActivation:
    """Tests for agent activation."""

    @pytest.mark.asyncio
    async def test_activate_success(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """activate() invokes provider and returns result."""
        mock_response = LLMResponse(
            content="Hello! I am the Showrunner.",
            model="qwen3:8b",
            provider="mock",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        mock_provider.invoke = AsyncMock(return_value=mock_response)

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        result = await runtime.activate(agent, "Hello!", session)

        assert result.content == "Hello! I am the Showrunner."
        assert result.agent_id == "showrunner"
        assert result.turn is not None
        assert result.usage is not None
        assert result.usage.total_tokens == 120

    @pytest.mark.asyncio
    async def test_activate_creates_turn(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """activate() creates and completes a turn."""
        mock_response = LLMResponse(
            content="Response",
            model="qwen3:8b",
            provider="mock",
        )
        mock_provider.invoke = AsyncMock(return_value=mock_response)

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        assert session.turn_count == 0

        await runtime.activate(agent, "Input", session)

        assert session.turn_count == 1
        turn = session.turns[0]
        assert turn.input == "Input"
        assert turn.output == "Response"

    @pytest.mark.asyncio
    async def test_activate_context_overflow(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """activate() raises on context overflow."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            context_limit=10,  # Very small
        )
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        with pytest.raises(ContextOverflowError):
            await runtime.activate(agent, "Hello!", session)

        # Turn should be marked as error
        assert session.turn_count == 1
        assert "exceeds" in (session.turns[0].output or "")


class TestAgentActivationStreaming:
    """Tests for streaming agent activation."""

    @pytest.mark.asyncio
    async def test_activate_streaming(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """activate_streaming yields chunks and completes turn."""

        async def mock_stream(*_args, **_kwargs):
            yield StreamChunk(content="Hello")
            yield StreamChunk(content=" world")
            yield StreamChunk(content="!", done=True, total_tokens=50)

        mock_provider.stream = mock_stream

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        chunks = []
        async for chunk in runtime.activate_streaming(agent, "Hi", session):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].done is True

        # Turn should be completed
        assert session.turn_count == 1
        turn = session.turns[0]
        assert turn.output == "Hello world!"

    @pytest.mark.asyncio
    async def test_activate_streaming_error(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """activate_streaming handles errors properly."""

        async def mock_stream_error(*_args, **_kwargs):
            yield StreamChunk(content="Start")
            raise RuntimeError("Stream failed")

        mock_provider.stream = mock_stream_error

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")
        assert agent is not None

        with pytest.raises(RuntimeError, match="Stream failed"):
            async for _chunk in runtime.activate_streaming(agent, "Hi", session):
                pass

        # Turn should be marked as error
        assert session.turn_count == 1
        assert "Stream failed" in (session.turns[0].output or "")
