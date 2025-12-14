"""
Agent runtime for executing agent activations.

The AgentRuntime handles the full lifecycle of an agent activation:
1. Load agent and build context
2. Build system prompt with tool descriptions
3. Validate context size
4. Execute LLM call (streaming or non-streaming)
5. Parse and execute tool calls
6. Track turn in session

With observability integration:
- JSONL event logging to project_dir/logs/events.jsonl
- LangSmith tracing (when LANGSMITH_TRACING=true)

With tool execution (Phase 2):
- Tool registry filters tools by agent capabilities
- Tool results are added to conversation context
- Tool calls are logged for observability
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.agent.context import AgentContext, ContextBuilder
from questfoundry.runtime.agent.prompt import PromptBuilder, build_prompt
from questfoundry.runtime.observability import EventType
from questfoundry.runtime.providers import (
    ContextOverflowError,
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    StreamChunk,
)
from questfoundry.runtime.session import Session, TokenUsage, Turn

if TYPE_CHECKING:
    from questfoundry.runtime.models import Agent, Studio
    from questfoundry.runtime.observability import EventLogger, TracingManager
    from questfoundry.runtime.storage import Project
    from questfoundry.runtime.tools import BaseTool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool call made by the agent."""

    tool_id: str
    args: dict[str, Any]
    result: Any = None
    success: bool = False
    error: str | None = None
    execution_time_ms: float | None = None


@dataclass
class ActivationResult:
    """Result of an agent activation."""

    content: str
    agent_id: str
    turn: Turn
    usage: TokenUsage | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class AgentRuntime:
    """
    Runtime for executing agent activations.

    Handles context building, prompt construction, LLM invocation,
    tool execution, and session/turn management.

    Integrations:
    - Observability: JSONL event logging, LangSmith tracing
    - Tools: Registry-based tool filtering and execution per agent capabilities
    """

    def __init__(
        self,
        provider: LLMProvider,
        studio: Studio,
        domain_path: Path | None = None,
        project: Project | None = None,
        model: str = "qwen3:8b",
        context_limit: int | None = None,
        event_logger: EventLogger | None = None,
        tracing_manager: TracingManager | None = None,
    ):
        """
        Initialize agent runtime.

        Args:
            provider: LLM provider to use
            studio: Loaded studio definition
            domain_path: Path to domain directory (for knowledge loading)
            project: Optional project for tool storage access
            model: Model to use for LLM calls
            context_limit: Maximum context size (tokens), raises error if exceeded
            event_logger: Optional JSONL event logger
            tracing_manager: Optional LangSmith tracing manager
        """
        self._provider = provider
        self._studio = studio
        self._domain_path = domain_path
        self._project = project
        self._model = model
        self._context_limit = context_limit
        self._event_logger = event_logger
        self._tracing_manager = tracing_manager

        self._context_builder = ContextBuilder(domain_path=domain_path)
        self._prompt_builder = PromptBuilder()
        self._tool_registry: ToolRegistry | None = None

    @property
    def tool_registry(self) -> ToolRegistry | None:
        """
        Get the tool registry, creating it lazily.

        Returns None if tools module is not available.
        """
        if self._tool_registry is None:
            try:
                from questfoundry.runtime.tools import ToolRegistry

                self._tool_registry = ToolRegistry(
                    studio=self._studio,
                    project=self._project,
                    domain_path=self._domain_path,
                )
            except ImportError:
                logger.warning("Tools module not available, tool execution disabled", exc_info=True)
                return None

        return self._tool_registry

    def get_agent_tools(self, agent: Agent, session_id: str | None = None) -> list[BaseTool]:
        """
        Get tools available to an agent.

        Args:
            agent: Agent to get tools for
            session_id: Current session ID for context

        Returns:
            List of tools the agent can use (empty if tools not available)
        """
        if not self.tool_registry:
            return []
        return self.tool_registry.get_agent_tools(agent, session_id)

    async def execute_tool(
        self,
        tool_id: str,
        args: dict[str, Any],
        agent: Agent,
        session_id: str | None = None,
    ) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_id: ID of tool to execute
            args: Tool arguments
            agent: Agent requesting tool execution
            session_id: Current session ID

        Returns:
            ToolResult from tool execution

        Raises:
            CapabilityViolationError: If agent lacks capability
            KeyError: If tool not found
        """
        if not self.tool_registry:
            from questfoundry.runtime.tools import ToolResult

            return ToolResult(
                success=False,
                data={},
                error="Tool registry not available",
            )

        # Enforce capability
        self.tool_registry.enforce_capability(agent, tool_id)

        # Get and execute tool
        tool = self.tool_registry.get_tool(tool_id, agent.id, session_id)

        # Log tool call start
        if self._event_logger:
            self._event_logger.log(
                EventType.TOOL_CALL_START,
                agent_id=agent.id,
                tool_id=tool_id,
                args=args,
            )

        result = await tool.run(args)

        # Log tool call result
        if self._event_logger:
            self._event_logger.log(
                EventType.TOOL_CALL_COMPLETE,
                agent_id=agent.id,
                tool_id=tool_id,
                success=result.success,
                error=result.error,
                execution_time_ms=result.execution_time_ms,
            )

        return result

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        for agent in self._studio.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_entry_agent(self) -> Agent | None:
        """Get the first entry agent."""
        for agent in self._studio.agents:
            if agent.is_entry_agent:
                return agent
        return None

    def build_context(self, agent: Agent) -> AgentContext:
        """Build context for an agent."""
        return self._context_builder.build(agent, self._studio)

    def build_messages(
        self,
        agent: Agent,
        user_input: str,
        context: AgentContext | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[LLMMessage]:
        """
        Build messages for LLM invocation.

        Args:
            agent: The agent to activate
            user_input: User's input message
            context: Pre-built context (built if not provided)
            history: Conversation history [{role, content}, ...]

        Returns:
            List of LLMMessage for the provider
        """
        if context is None:
            context = self.build_context(agent)

        # Build system prompt
        prompt = build_prompt(
            agent=agent,
            constitution_text=context.constitution_text,
            must_know_entries=context.must_know_entries,
            role_specific_menu=context.role_specific_menu,
        )

        messages: list[LLMMessage] = []

        # System message
        messages.append(LLMMessage(role="system", content=prompt.text))

        # History
        if history:
            for h in history:
                messages.append(LLMMessage(role=h["role"], content=h["content"]))

        # User input
        messages.append(LLMMessage(role="user", content=user_input))

        return messages

    def estimate_tokens(self, messages: list[LLMMessage]) -> int:
        """Estimate token count for messages."""
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4  # Rough estimate

    def validate_context_size(self, messages: list[LLMMessage]) -> None:
        """
        Validate that messages fit within context limit.

        Raises:
            ContextOverflowError: If messages exceed context limit
        """
        if self._context_limit is None:
            return

        estimated = self.estimate_tokens(messages)
        if estimated > self._context_limit:
            raise ContextOverflowError(
                f"Prompt ({estimated} tokens) exceeds model context "
                f"({self._context_limit} tokens). Reduce knowledge or use larger model."
            )

    async def activate(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
    ) -> ActivationResult:
        """
        Activate an agent (non-streaming).

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options

        Returns:
            ActivationResult with response content and turn

        Raises:
            ContextOverflowError: If prompt exceeds context limit
            ProviderError: If LLM call fails
        """
        # Start turn
        turn = session.start_turn(agent.id, user_input)
        start_time = time.time()

        # Log turn start
        if self._event_logger:
            self._event_logger.turn_start(
                session_id=session.id,
                turn_id=turn.turn_number,
                agent_id=agent.id,
                input_text=user_input,
            )

        try:
            # Build context and messages
            context = self.build_context(agent)

            # Log context building
            if self._event_logger:
                self._event_logger.context_build(
                    session_id=session.id,
                    agent_id=agent.id,
                    knowledge_count=len(context.must_know_entries),
                    total_chars=context.total_tokens * 4,  # Rough estimate,
                )

            history = session.get_history()[:-1] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history)

            # Validate context size
            self.validate_context_size(messages)

            # Log LLM call start
            estimated_tokens = self.estimate_tokens(messages)
            if self._event_logger:
                self._event_logger.llm_call_start(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    model=self._model,
                    provider=self._provider.name,
                    prompt_tokens=estimated_tokens,
                )

            # Invoke LLM
            response = await self._provider.invoke(messages, self._model, options)

            # Complete turn
            usage = TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            )
            session.complete_turn(turn, response.content, usage)
            duration_ms = (time.time() - start_time) * 1000

            # Log completion
            if self._event_logger:
                self._event_logger.llm_call_complete(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    model=self._model,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    duration_ms=response.duration_ms,
                )
                self._event_logger.turn_complete(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    output_length=len(response.content),
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    duration_ms=duration_ms,
                )

            return ActivationResult(
                content=response.content,
                agent_id=agent.id,
                turn=turn,
                usage=usage,
            )

        except Exception as e:
            session.error_turn(turn, str(e))
            # Log error
            if self._event_logger:
                self._event_logger.turn_error(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    error=str(e),
                )
            raise

    async def activate_streaming(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Activate an agent with streaming response.

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options

        Yields:
            StreamChunk with content

        Raises:
            ContextOverflowError: If prompt exceeds context limit
            ProviderError: If LLM call fails
        """
        # Start turn
        turn = session.start_turn(agent.id, user_input)
        start_time = time.time()

        # Log turn start
        if self._event_logger:
            self._event_logger.turn_start(
                session_id=session.id,
                turn_id=turn.turn_number,
                agent_id=agent.id,
                input_text=user_input,
            )

        try:
            # Build context and messages
            context = self.build_context(agent)

            # Log context building
            if self._event_logger:
                self._event_logger.context_build(
                    session_id=session.id,
                    agent_id=agent.id,
                    knowledge_count=len(context.must_know_entries),
                    total_chars=context.total_tokens * 4,  # Rough estimate,
                )

            history = session.get_history()[:-1] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history)

            # Validate context size
            self.validate_context_size(messages)

            # Log LLM call start
            estimated_tokens = self.estimate_tokens(messages)
            if self._event_logger:
                self._event_logger.llm_call_start(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    model=self._model,
                    provider=self._provider.name,
                    prompt_tokens=estimated_tokens,
                )

            # Collect response for turn completion
            full_content = ""
            final_usage: TokenUsage | None = None

            # Stream from provider
            async for chunk in self._provider.stream(messages, self._model, options):
                full_content += chunk.content

                if chunk.done:
                    final_usage = TokenUsage(
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        total_tokens=chunk.total_tokens,
                    )

                yield chunk

            # Complete turn after streaming finishes
            session.complete_turn(turn, full_content, final_usage)
            duration_ms = (time.time() - start_time) * 1000

            # Log completion
            if self._event_logger:
                self._event_logger.llm_call_complete(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    model=self._model,
                    completion_tokens=final_usage.completion_tokens if final_usage else None,
                    total_tokens=final_usage.total_tokens if final_usage else None,
                    duration_ms=duration_ms,
                )
                self._event_logger.turn_complete(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    output_length=len(full_content),
                    prompt_tokens=final_usage.prompt_tokens if final_usage else None,
                    completion_tokens=final_usage.completion_tokens if final_usage else None,
                    duration_ms=duration_ms,
                )

        except Exception as e:
            session.error_turn(turn, str(e))
            # Log error
            if self._event_logger:
                self._event_logger.turn_error(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    error=str(e),
                )
            raise


async def activate_agent(
    agent_id: str,
    user_input: str,
    runtime: AgentRuntime,
    session: Session,
    streaming: bool = True,
    options: InvokeOptions | None = None,
) -> ActivationResult | AsyncIterator[StreamChunk]:
    """
    Convenience function to activate an agent.

    Args:
        agent_id: ID of the agent to activate
        user_input: User's input message
        runtime: The agent runtime
        session: Session to track the turn
        streaming: Whether to use streaming
        options: LLM invocation options

    Returns:
        ActivationResult (non-streaming) or AsyncIterator[StreamChunk] (streaming)

    Raises:
        ValueError: If agent not found
        ContextOverflowError: If prompt exceeds context limit
    """
    agent = runtime.get_agent(agent_id)
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    if streaming:
        return runtime.activate_streaming(agent, user_input, session, options)
    else:
        return await runtime.activate(agent, user_input, session, options)
