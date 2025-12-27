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


class TestToolResultsToMessages:
    """Tests for _tool_results_to_messages error handling."""

    def test_error_includes_full_feedback_data(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Tool errors should include result.result data for LLM self-correction."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        # Simulate a failed tool call with detailed feedback
        tool_requests = [
            ToolCallRequest(
                id="call_123",
                name="save_artifact",
                arguments={"artifact_type": "section_brief", "data": {"title": "Wrong"}},
            )
        ]
        tool_results = [
            ToolCall(
                tool_id="save_artifact",
                args={"artifact_type": "section_brief", "data": {"title": "Wrong"}},
                success=False,
                error="Artifact validation failed: 2 error(s)",
                result={
                    "feedback": {
                        "success": False,
                        "error_count": 2,
                        "errors": [
                            {"field": "brief_id", "issue": "Required field missing"},
                            {"field": "section_title", "issue": "Required field missing"},
                        ],
                        "required_fields": [
                            {"name": "brief_id", "type": "string"},
                            {"name": "section_title", "type": "string"},
                        ],
                        "hint": "Check the artifact schema",
                    }
                },
            )
        ]

        # Call the method
        messages = runtime._tool_results_to_messages(tool_requests, tool_results)

        # Verify the error message includes the full feedback
        assert len(messages) == 1
        msg = messages[0]
        assert msg.role == "tool"
        assert msg.name == "save_artifact"

        content = json.loads(msg.content)
        assert "error" in content
        assert content["error"] == "Artifact validation failed: 2 error(s)"
        # KEY: The feedback data should be included for LLM self-correction
        assert "feedback" in content
        assert content["feedback"]["error_count"] == 2
        assert len(content["feedback"]["errors"]) == 2
        assert content["feedback"]["errors"][0]["field"] == "brief_id"

    def test_success_result_not_affected(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Successful tool results should work as before."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_requests = [
            ToolCallRequest(
                id="call_456",
                name="some_tool",
                arguments={"query": "test"},
            )
        ]
        tool_results = [
            ToolCall(
                tool_id="some_tool",
                args={"query": "test"},
                success=True,
                result={"data": "success", "count": 42},
            )
        ]

        messages = runtime._tool_results_to_messages(tool_requests, tool_results)

        assert len(messages) == 1
        msg = messages[0]
        content = json.loads(msg.content)
        assert content == {"data": "success", "count": 42}

    def test_error_with_none_result(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Tool errors with None result should still include error message."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_requests = [
            ToolCallRequest(
                id="call_789",
                name="failing_tool",
                arguments={"input": "test"},
            )
        ]
        tool_results = [
            ToolCall(
                tool_id="failing_tool",
                args={"input": "test"},
                success=False,
                error="Connection timeout",
                result=None,  # No result data, just error
            )
        ]

        messages = runtime._tool_results_to_messages(tool_requests, tool_results)

        assert len(messages) == 1
        msg = messages[0]
        content = json.loads(msg.content)
        assert content == {"error": "Connection timeout"}


class TestContextSummarizationPressureGating:
    """Tests for context summarization pressure gating."""

    def test_no_summarization_below_full_level(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Context summarization should not trigger when below FULL level (90%)."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        # Create enough history to trigger turn-count-based summarization
        # (8+ message groups), but context pressure is low (default 0%)
        history = []
        for i in range(20):  # Well above the 8-group threshold
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Response {i}"})

        # Apply summarization - should NOT summarize because pressure is below 90%
        result = runtime._apply_context_summarization(history, "test_agent")

        # History should be unchanged (no summarization)
        assert result == history
        assert len(result) == 40  # All messages preserved

    def test_summarization_triggers_at_full_level(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Context summarization should trigger when at FULL level (90%+)."""
        runtime = AgentRuntime(
            provider=mock_provider,
            studio=basic_studio,
            context_limit=1000,  # Small limit (in tokens)
        )

        # Create history that actually fills 90%+ of context
        # Each message needs ~25 tokens to reach ~950 tokens with 40 messages
        # JSON overhead adds ~20 chars per message, so we need ~100 char content
        padding = "x" * 100  # ~25 tokens per message
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"Message {i} {padding}"})
            history.append({"role": "assistant", "content": f"Response {i} {padding}"})

        # Apply summarization - should summarize because actual history is >= 90%
        result = runtime._apply_context_summarization(history, "test_agent")

        # History should be summarized (fewer messages)
        assert result != history
        # Should have summary message + preserved recent turns
        assert len(result) < len(history)


class TestIterationOutcome:
    """Tests for IterationOutcome dataclass (issue #234)."""

    def test_made_progress_with_successes(self) -> None:
        """made_progress returns True when there are effective successes."""
        from questfoundry.runtime.agent.runtime import IterationOutcome

        outcome = IterationOutcome(
            total_tool_calls=3,
            successful_tool_calls=2,
            failed_tool_calls=1,
            rejected_tool_calls=0,
        )
        assert outcome.made_progress is True

    def test_made_progress_with_all_rejected(self) -> None:
        """made_progress returns False when all successes were rejections."""
        from questfoundry.runtime.agent.runtime import IterationOutcome

        outcome = IterationOutcome(
            total_tool_calls=2,
            successful_tool_calls=2,
            failed_tool_calls=0,
            rejected_tool_calls=2,  # All successes were rejections
        )
        assert outcome.made_progress is False

    def test_made_progress_with_partial_rejection(self) -> None:
        """made_progress returns True when some successes are not rejections."""
        from questfoundry.runtime.agent.runtime import IterationOutcome

        outcome = IterationOutcome(
            total_tool_calls=3,
            successful_tool_calls=3,
            failed_tool_calls=0,
            rejected_tool_calls=1,  # Only 1 of 3 successes was rejection
        )
        assert outcome.made_progress is True

    def test_made_progress_with_all_failures(self) -> None:
        """made_progress returns False when all tools failed."""
        from questfoundry.runtime.agent.runtime import IterationOutcome

        outcome = IterationOutcome(
            total_tool_calls=2,
            successful_tool_calls=0,
            failed_tool_calls=2,
            rejected_tool_calls=0,
        )
        assert outcome.made_progress is False


class TestMadeProgressHelper:
    """Tests for _made_progress helper method (issue #234)."""

    def test_failed_tool_is_no_progress(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Failed tool calls don't count as progress."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_result = ToolCall(
            tool_id="some_tool",
            args={},
            success=False,
            error="Something went wrong",
        )
        assert runtime._made_progress(tool_result) is False

    def test_successful_tool_is_progress(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Successful tool calls count as progress by default."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_result = ToolCall(
            tool_id="some_tool",
            args={},
            success=True,
            result={"data": "success"},
        )
        assert runtime._made_progress(tool_result) is True

    def test_rejected_can_reject_tool_is_no_progress(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Tool with can_reject=True returning rejected is no progress."""
        from questfoundry.runtime.models.base import Tool
        from questfoundry.runtime.tools.registry import ToolRegistry

        # Add a tool that can reject
        basic_studio.tools = [
            Tool(
                id="validate_artifact",
                name="Validate Artifact",
                description="Test tool that can reject",
                can_reject=True,
            )
        ]

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        runtime._tool_registry = ToolRegistry(studio=basic_studio)

        # Tool succeeded but rejected the work
        tool_result = ToolCall(
            tool_id="validate_artifact",
            args={},
            success=True,
            result={
                "feedback": {
                    "action_outcome": "rejected",
                    "message": "Validation failed",
                }
            },
        )
        assert runtime._made_progress(tool_result) is False

    def test_explicit_made_progress_flag(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Tool result with explicit made_progress field is respected."""
        from questfoundry.runtime.models.base import Tool
        from questfoundry.runtime.tools.registry import ToolRegistry

        basic_studio.tools = [
            Tool(
                id="save_artifact",
                name="Save Artifact",
                description="Test tool",
                can_reject=True,
            )
        ]

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        runtime._tool_registry = ToolRegistry(studio=basic_studio)

        # Tool explicitly says no progress was made
        tool_result = ToolCall(
            tool_id="save_artifact",
            args={},
            success=True,
            result={"made_progress": False},
        )
        assert runtime._made_progress(tool_result) is False


class TestEvaluateIterationOutcome:
    """Tests for _evaluate_iteration_outcome method (issue #234)."""

    def test_all_successful_tools(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """All successful tools produce progress."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_calls = [
            ToolCall(tool_id="tool1", args={}, success=True, result={}),
            ToolCall(tool_id="tool2", args={}, success=True, result={}),
        ]

        outcome = runtime._evaluate_iteration_outcome(tool_calls)

        assert outcome.total_tool_calls == 2
        assert outcome.successful_tool_calls == 2
        assert outcome.failed_tool_calls == 0
        assert outcome.rejected_tool_calls == 0
        assert outcome.made_progress is True

    def test_mixed_success_failure(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Mix of successful and failed tools."""
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        tool_calls = [
            ToolCall(tool_id="tool1", args={}, success=True, result={}),
            ToolCall(tool_id="tool2", args={}, success=False, error="Failed"),
        ]

        outcome = runtime._evaluate_iteration_outcome(tool_calls)

        assert outcome.total_tool_calls == 2
        assert outcome.successful_tool_calls == 1
        assert outcome.failed_tool_calls == 1
        assert outcome.made_progress is True

    def test_all_rejected(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """All successful tools rejected work."""
        from questfoundry.runtime.models.base import Tool
        from questfoundry.runtime.tools.registry import ToolRegistry

        basic_studio.tools = [Tool(id="validate", name="Validate", description="", can_reject=True)]
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        runtime._tool_registry = ToolRegistry(studio=basic_studio)

        tool_calls = [
            ToolCall(
                tool_id="validate",
                args={},
                success=True,
                result={"feedback": {"action_outcome": "rejected"}},
            ),
        ]

        outcome = runtime._evaluate_iteration_outcome(tool_calls)

        assert outcome.total_tool_calls == 1
        assert outcome.successful_tool_calls == 1
        assert outcome.rejected_tool_calls == 1
        assert outcome.made_progress is False

    def test_mixed_rejected_and_successful_tools(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
    ) -> None:
        """Iteration with at least one non-rejected success counts as progress."""
        from questfoundry.runtime.models.base import Tool
        from questfoundry.runtime.tools.registry import ToolRegistry

        # One can_reject tool plus a generic non-rejecting tool
        basic_studio.tools = [Tool(id="validate", name="Validate", description="", can_reject=True)]
        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        runtime._tool_registry = ToolRegistry(studio=basic_studio)

        tool_calls = [
            # Rejected work from a can_reject tool
            ToolCall(
                tool_id="validate",
                args={},
                success=True,
                result={"feedback": {"action_outcome": "rejected"}},
            ),
            # Successful tool with no rejection signal
            ToolCall(
                tool_id="other_tool",
                args={},
                success=True,
                result={},
            ),
        ]

        outcome = runtime._evaluate_iteration_outcome(tool_calls)

        assert outcome.total_tool_calls == 2
        assert outcome.successful_tool_calls == 2
        assert outcome.rejected_tool_calls == 1
        # At least one non-rejected success → iteration made progress
        assert outcome.made_progress is True


class TestProgressBasedIterationLimits:
    """Tests for progress-based iteration counting (issue #234)."""

    @pytest.mark.asyncio
    async def test_stalled_counter_resets_on_progress(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """Stalled iterations reset when progress is made."""
        from questfoundry.runtime.providers.base import ToolCallRequest

        # Simulate: 2 rejections, then success, then 2 more rejections
        call_count = 0

        async def mock_invoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # First 2 calls: rejection
                return LLMResponse(
                    content="Trying validation...",
                    model="test",
                    provider="mock",
                    tool_calls=[
                        ToolCallRequest(
                            id=f"call_{call_count}",
                            name="validate_artifact",
                            arguments={},
                        )
                    ],
                )
            elif call_count == 3:
                # Third call: success
                return LLMResponse(
                    content="Saving artifact...",
                    model="test",
                    provider="mock",
                    tool_calls=[
                        ToolCallRequest(
                            id=f"call_{call_count}",
                            name="save_artifact",
                            arguments={},
                        )
                    ],
                )
            else:
                # After that: done
                return LLMResponse(
                    content="Done!",
                    model="test",
                    provider="mock",
                )

        mock_provider.invoke = AsyncMock(side_effect=mock_invoke)

        # Set up tools
        from questfoundry.runtime.models.base import Capability, Tool

        basic_studio.tools = [
            Tool(id="validate_artifact", name="Validate", description="", can_reject=True),
            Tool(id="save_artifact", name="Save", description="", can_reject=True),
        ]
        basic_studio.agents[0].capabilities = [
            Capability(id="cap1", name="", tool_ref="validate_artifact"),
            Capability(id="cap2", name="", tool_ref="save_artifact"),
        ]

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)

        async def fake_execute(tool_call_requests, *_args, **_kwargs):
            if not tool_call_requests:
                return []

            tool_name = tool_call_requests[0].name
            if tool_name == "validate_artifact":
                return [
                    ToolCall(
                        tool_id="validate_artifact",
                        args={},
                        success=True,
                        result={"feedback": {"action_outcome": "rejected"}},
                    )
                ]
            if tool_name == "save_artifact":
                return [
                    ToolCall(
                        tool_id="save_artifact",
                        args={},
                        success=True,
                        result={"feedback": {"action_outcome": "saved"}},
                    )
                ]
            return []

        runtime._execute_tool_calls = AsyncMock(side_effect=fake_execute)

        agent = runtime.get_agent("showrunner")
        assert agent is not None

        # Should complete without hitting stalled limit because
        # progress resets the counter
        result = await runtime.activate(
            agent,
            "Test",
            session,
            max_stalled_iterations=3,
            max_total_iterations=10,
            enforce_tool_usage=False,
        )

        # Should have completed
        assert "stalled" not in result.content.lower()

    @pytest.mark.asyncio
    async def test_max_tool_iterations_backward_compat(
        self,
        mock_provider: MagicMock,
        basic_studio: Studio,
        session: Session,
    ) -> None:
        """Deprecated max_tool_iterations still works."""
        import warnings

        mock_response = LLMResponse(
            content="Done",
            model="test",
            provider="mock",
        )
        mock_provider.invoke = AsyncMock(return_value=mock_response)

        runtime = AgentRuntime(provider=mock_provider, studio=basic_studio)
        agent = runtime.get_agent("showrunner")

        # Should trigger deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            await runtime.activate(
                agent,
                "Test",
                session,
                max_tool_iterations=5,  # Deprecated parameter
            )

            # Check that deprecation warning was issued
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "max_tool_iterations" in str(deprecation_warnings[0].message)


class TestChainOfThoughtReasoningLogging:
    """Tests for Chain-of-Thought reasoning logging in tool execution."""

    @pytest.mark.asyncio
    async def test_reasoning_logged_at_info_level(
        self,
        mock_provider: MagicMock,
        project: Project,
        caplog,
    ) -> None:
        """Reasoning argument is logged at INFO level before tool execution."""
        import logging
        from typing import Any

        from questfoundry.runtime.models.base import Agent, Capability, Studio, Tool
        from questfoundry.runtime.tools.base import BaseTool, ToolContext, ToolResult
        from questfoundry.runtime.tools.registry import ToolRegistry

        # Create a mock tool
        class MockTool(BaseTool):
            async def execute(self, _args: dict[str, Any]) -> ToolResult:
                return ToolResult(success=True, data={"result": "done"})

        # Create tool definition
        tool_def = Tool(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
        )

        # Create agent with capability to use the test tool
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            description="Test agent",
            archetypes=["creator"],
            capabilities=[
                Capability(
                    id="use_test_tool",
                    name="Use Test Tool",
                    description="Can use test_tool",
                    category="tools",
                    tool_ref="test_tool",
                ),
            ],
        )

        # Create studio with the tool and agent
        studio = Studio(
            id="test_studio",
            name="Test Studio",
            agents=[agent],
            tools=[tool_def],
        )

        tool_context = ToolContext(studio=studio, project=project)
        mock_tool = MockTool(definition=tool_def, context=tool_context)

        # Setup runtime with tool registry
        runtime = AgentRuntime(provider=mock_provider, studio=studio, project=project)
        runtime._tool_registry = ToolRegistry(studio=studio, project=project)
        runtime._tool_registry._tool_cache["test_tool"] = mock_tool

        # Execute tool with reasoning
        with caplog.at_level(logging.INFO, logger="questfoundry.runtime.agent.runtime"):
            await runtime.execute_tool(
                agent=agent,
                tool_id="test_tool",
                args={"reasoning": "Testing CoT logging", "other_arg": "value"},
                session_id="test-session",
                turn_number=1,
            )

        # Check that reasoning was logged
        assert any("Testing CoT logging" in record.message for record in caplog.records), (
            f"Expected reasoning in log, got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_no_log_when_reasoning_absent(
        self,
        mock_provider: MagicMock,
        project: Project,
        caplog,
    ) -> None:
        """No reasoning log when reasoning argument is not provided."""
        import logging
        from typing import Any

        from questfoundry.runtime.models.base import Agent, Capability, Studio, Tool
        from questfoundry.runtime.tools.base import BaseTool, ToolContext, ToolResult
        from questfoundry.runtime.tools.registry import ToolRegistry

        class MockTool(BaseTool):
            async def execute(self, _args: dict[str, Any]) -> ToolResult:
                return ToolResult(success=True, data={"result": "done"})

        tool_def = Tool(
            id="test_tool",
            name="Test Tool",
            description="A test tool",
        )

        agent = Agent(
            id="test_agent",
            name="Test Agent",
            description="Test agent",
            archetypes=["creator"],
            capabilities=[
                Capability(
                    id="use_test_tool",
                    name="Use Test Tool",
                    description="Can use test_tool",
                    category="tools",
                    tool_ref="test_tool",
                ),
            ],
        )

        studio = Studio(
            id="test_studio",
            name="Test Studio",
            agents=[agent],
            tools=[tool_def],
        )

        tool_context = ToolContext(studio=studio, project=project)
        mock_tool = MockTool(definition=tool_def, context=tool_context)

        runtime = AgentRuntime(provider=mock_provider, studio=studio, project=project)
        runtime._tool_registry = ToolRegistry(studio=studio, project=project)
        runtime._tool_registry._tool_cache["test_tool"] = mock_tool

        # Execute tool WITHOUT reasoning
        with caplog.at_level(logging.INFO, logger="questfoundry.runtime.agent.runtime"):
            await runtime.execute_tool(
                agent=agent,
                tool_id="test_tool",
                args={"other_arg": "value"},  # No reasoning
                session_id="test-session",
                turn_number=1,
            )

        # Check that no reasoning-related log was emitted
        reasoning_logs = [r for r in caplog.records if "reasoning" in r.message.lower()]
        assert len(reasoning_logs) == 0, f"Unexpected reasoning log: {reasoning_logs}"
