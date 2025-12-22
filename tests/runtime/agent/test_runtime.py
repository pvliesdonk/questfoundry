"""Tests for AgentRuntime."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.agent import AgentRuntime
from questfoundry.runtime.agent.runtime import ToolCall, ToolCallRequest
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
        """get_agent returns None for unknown ID.""""""
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
            context_limit=10,
        )

        messages = [
            LLMMessage(role="system", content="This is a very long system prompt"),
            LLMMessage(role="user", content="Hello"),
        ]

        with pytest.raises(ContextOverflowError):
            runtime.validate_context_size(messages)


class TestGetAgentHistoryWithSummarization:
    """Tests for _get_agent_history_with_summarization."""

    def test_no_history_returns_none(self, mock_provider: MagicMock, basic_studio: Studio) -> None:
        """If agent has no prior turns, returns None."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        session = MagicMock(spec=Session)
        session.get_agent_turn_count.return_value = 1
        session.get_agent_history.return_value = []

        result = runtime._get_agent_history_with_summarization(session, "showrunner")

        assert result is None

    def test_returns_history_when_present(
        self, mock_provider: MagicMock, basic_studio: Studio
    ) -> None:
        """If agent has history and context is not high, returns history unchanged."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        session = MagicMock(spec=Session)
        …