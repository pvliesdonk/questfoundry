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

import json
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
    LLMResponse,
    StreamChunk,
    ToolCallRequest,
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

    def get_tool_schemas(self, agent: Agent, session_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get LangChain-compatible tool schemas for an agent.

        Args:
            agent: Agent to get tools for
            session_id: Current session ID

        Returns:
            List of tool schemas (empty if no tools available)
        """
        if not self.tool_registry:
            return []
        return self.tool_registry.get_langchain_tools(agent, session_id)

    def build_messages(
        self,
        agent: Agent,
        user_input: str,
        context: AgentContext | None = None,
        history: list[dict[str, str]] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[LLMMessage]:
        """
        Build messages for LLM invocation.

        Args:
            agent: The agent to activate
            user_input: User's input message
            context: Pre-built context (built if not provided)
            history: Conversation history [{role, content}, ...]
            tool_schemas: Optional list of tool schemas to include in prompt

        Returns:
            List of LLMMessage for the provider
        """
        if context is None:
            context = self.build_context(agent)

        # Build system prompt with tool descriptions, playbooks, stores, and artifact types
        prompt = build_prompt(
            agent=agent,
            constitution_text=context.constitution_text,
            must_know_entries=context.must_know_entries,
            role_specific_menu=context.role_specific_menu,
            tool_schemas=tool_schemas,
            playbooks_menu=context.playbooks_menu,
            stores_menu=context.stores_menu,
            artifact_types_menu=context.artifact_types_menu,
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

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCallRequest],
        agent: Agent,
        session_id: str | None = None,
    ) -> list[ToolCall]:
        """
        Execute a list of tool calls.

        Args:
            tool_calls: List of tool call requests from LLM
            agent: Agent making the calls
            session_id: Current session ID

        Returns:
            List of ToolCall results
        """
        results = []
        for tc in tool_calls:
            tool_call = ToolCall(tool_id=tc.name, args=tc.arguments)
            start_time = time.time()

            try:
                result = await self.execute_tool(tc.name, tc.arguments, agent, session_id)
                tool_call.result = result.data
                tool_call.success = result.success
                tool_call.error = result.error
                tool_call.execution_time_ms = result.execution_time_ms
            except Exception as e:
                tool_call.success = False
                tool_call.error = str(e)
                tool_call.execution_time_ms = (time.time() - start_time) * 1000
                logger.warning(f"Tool call failed: {tc.name} - {e}")

            results.append(tool_call)

        return results

    def _tool_results_to_messages(
        self,
        tool_calls: list[ToolCallRequest],
        tool_results: list[ToolCall],
    ) -> list[LLMMessage]:
        """
        Convert tool call results to LLM messages.

        Args:
            tool_calls: Original tool call requests (with IDs)
            tool_results: Executed tool results

        Returns:
            List of tool result messages for the LLM
        """
        messages = []
        for tc, result in zip(tool_calls, tool_results, strict=True):
            # Format the result as JSON
            if result.success:
                content = json.dumps(result.result) if result.result else "{}"
            else:
                content = json.dumps({"error": result.error or "Tool execution failed"})

            messages.append(
                LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
        return messages

    async def activate(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
        max_tool_iterations: int = 10,
    ) -> ActivationResult:
        """
        Activate an agent (non-streaming) with tool call support.

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options
            max_tool_iterations: Maximum number of tool call iterations

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

            # Get tool schemas for this agent
            tool_schemas = self.get_tool_schemas(agent, session.id)

            history = session.get_history()[:-1] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history, tool_schemas)

            # Validate context size
            self.validate_context_size(messages)

            # Track all tool calls made during this activation
            all_tool_calls: list[ToolCall] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            response: LLMResponse | None = None

            # Tool call loop
            for iteration in range(max_tool_iterations):
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

                # Invoke LLM with tools
                response = await self._provider.invoke(
                    messages, self._model, options, tools=tool_schemas if tool_schemas else None
                )

                # Accumulate token usage
                if response.prompt_tokens:
                    total_prompt_tokens += response.prompt_tokens
                if response.completion_tokens:
                    total_completion_tokens += response.completion_tokens

                # Log LLM call complete
                if self._event_logger:
                    self._event_logger.llm_call_complete(
                        session_id=session.id,
                        turn_id=turn.turn_number,
                        model=self._model,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                        duration_ms=response.duration_ms,
                    )

                # Check for tool calls
                if not response.has_tool_calls or not response.tool_calls:
                    # No tool calls, we're done
                    break

                # Execute tool calls
                tool_calls = response.tool_calls  # Already checked not None
                logger.info(f"Executing {len(tool_calls)} tool calls (iteration {iteration + 1})")
                tool_results = await self._execute_tool_calls(tool_calls, agent, session.id)
                all_tool_calls.extend(tool_results)

                # Add assistant message with tool calls (empty content)
                messages.append(LLMMessage(role="assistant", content=response.content or ""))

                # Add tool result messages
                tool_messages = self._tool_results_to_messages(tool_calls, tool_results)
                messages.extend(tool_messages)

            # Complete turn
            usage = TokenUsage(
                prompt_tokens=total_prompt_tokens or response.prompt_tokens if response else None,
                completion_tokens=total_completion_tokens or response.completion_tokens
                if response
                else None,
                total_tokens=(total_prompt_tokens + total_completion_tokens)
                if total_prompt_tokens
                else None,
            )
            final_content = response.content if response else ""
            session.complete_turn(turn, final_content, usage)
            duration_ms = (time.time() - start_time) * 1000

            # Log turn completion
            if self._event_logger:
                self._event_logger.turn_complete(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    output_length=len(final_content),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    duration_ms=duration_ms,
                )

            return ActivationResult(
                content=final_content,
                agent_id=agent.id,
                turn=turn,
                usage=usage,
                tool_calls=all_tool_calls,
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

        Note: For tool calling with multi-iteration loops, use activate() instead.
        Streaming mode provides tools to the LLM but handles tool calls in a single pass.
        If tool calls are detected, they are executed and a follow-up response is streamed.

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

            # Get tool schemas for this agent
            tool_schemas = self.get_tool_schemas(agent, session.id)

            history = session.get_history()[:-1] if len(session.turns) > 1 else None
            messages = self.build_messages(agent, user_input, context, history, tool_schemas)

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
            pending_tool_calls: list[ToolCallRequest] | None = None

            # Stream from provider with tools
            async for chunk in self._provider.stream(
                messages, self._model, options, tools=tool_schemas if tool_schemas else None
            ):
                full_content += chunk.content

                if chunk.done:
                    final_usage = TokenUsage(
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        total_tokens=chunk.total_tokens,
                    )
                    # Check for tool calls in final chunk
                    if chunk.tool_calls:
                        pending_tool_calls = chunk.tool_calls

                yield chunk

            # Handle tool calls if present (single pass for streaming)
            if pending_tool_calls:
                logger.info(
                    f"Executing {len(pending_tool_calls)} tool calls from streaming response"
                )
                tool_results = await self._execute_tool_calls(pending_tool_calls, agent, session.id)

                # Add assistant message and tool results
                messages.append(LLMMessage(role="assistant", content=full_content))
                tool_messages = self._tool_results_to_messages(pending_tool_calls, tool_results)
                messages.extend(tool_messages)

                # Stream follow-up response (without tools to prevent infinite loops)
                async for chunk in self._provider.stream(messages, self._model, options):
                    full_content += chunk.content

                    # Update usage if done and we have previous usage to accumulate
                    if chunk.done and final_usage and chunk.prompt_tokens:
                        final_usage = TokenUsage(
                            prompt_tokens=(final_usage.prompt_tokens or 0) + chunk.prompt_tokens,
                            completion_tokens=(final_usage.completion_tokens or 0)
                            + (chunk.completion_tokens or 0),
                            total_tokens=(final_usage.total_tokens or 0)
                            + (chunk.total_tokens or 0),
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
