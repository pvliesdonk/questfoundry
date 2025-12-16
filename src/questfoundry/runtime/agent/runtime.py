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
    stop_reason: str | None = None  # "delegate", "terminate", "max_iterations", None


# Tool names that are "stop tools" - calling them returns control to orchestrator/human
STOP_TOOL_NAMES = frozenset(
    {
        "delegate",  # Delegation to another agent
        "terminate",  # End workflow
        "return_to_orchestrator",  # Delegatee returns to orchestrator
        "request_clarification",  # Ask human for clarification
    }
)

# Maximum consecutive failures before giving up
MAX_CONSECUTIVE_FAILURES = 3


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
        broker: Any | None = None,
        interactive: bool = True,
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
            broker: Message broker for delegation routing
            interactive: Whether running in interactive mode (affects tool availability)
        """
        self._provider = provider
        self._studio = studio
        self._domain_path = domain_path
        self._project = project
        self._model = model
        self._context_limit = context_limit
        self._event_logger = event_logger
        self._tracing_manager = tracing_manager
        self._broker = broker
        self._interactive = interactive

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
                    broker=self._broker,
                    interactive=self._interactive,
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

        # Build system prompt with tool descriptions, playbooks, stores, artifact types, and agents
        prompt = build_prompt(
            agent=agent,
            constitution_text=context.constitution_text,
            must_know_entries=context.must_know_entries,
            role_specific_menu=context.role_specific_menu,
            tool_schemas=tool_schemas,
            playbooks_menu=context.playbooks_menu,
            stores_menu=context.stores_menu,
            artifact_types_menu=context.artifact_types_menu,
            agents_menu=context.agents_menu,
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

    def _is_orchestrator(self, agent: Agent) -> bool:
        """Check if an agent is an orchestrator."""
        return "orchestrator" in [
            a.value if hasattr(a, "value") else str(a) for a in agent.archetypes
        ]

    def _build_tool_nudge_message(self, agent: Agent) -> str:
        """
        Build a nudge message when the LLM doesn't make tool calls.

        Based on the v3 executor's validate-with-feedback pattern.
        """
        if self._is_orchestrator(agent):
            return (
                "You must use tools to do your work and communicate results. "
                "Your response contained no tool calls.\n\n"
                "As an orchestrator, you MUST either:\n"
                "1. Call `delegate` to assign work to a specialist agent, OR\n"
                "2. Call `terminate` if the workflow is complete\n\n"
                "Do NOT generate content directly - delegate to specialists. "
                "Do NOT respond with plain text - you MUST make a tool call."
            )
        else:
            return (
                "You must use tools to do your work and communicate results. "
                "Your response contained no tool calls.\n\n"
                "Please either:\n"
                "1. Call a tool to continue your work, OR\n"
                "2. Call `return_to_orchestrator` to report your work is complete\n\n"
                "Do NOT respond with plain text - you MUST make a tool call."
            )

    def _check_for_stop_tool(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[str | None, dict[str, Any] | None]:
        """
        Check if any tool call is a stop tool.

        Returns:
            Tuple of (stop_tool_name, stop_tool_result) or (None, None)
        """
        for tc in tool_calls:
            if tc.tool_id in STOP_TOOL_NAMES and tc.success:
                # Parse result to check if it actually succeeded
                result = tc.result
                if isinstance(result, dict) and result.get("success", True):
                    return tc.tool_id, result
        return None, None

    async def activate(
        self,
        agent: Agent,
        user_input: str,
        session: Session,
        options: InvokeOptions | None = None,
        max_tool_iterations: int = 10,
        enforce_tool_usage: bool = True,
    ) -> ActivationResult:
        """
        Activate an agent (non-streaming) with tool call support.

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options
            max_tool_iterations: Maximum number of tool call iterations
            enforce_tool_usage: If True, nudge LLM when it doesn't make tool calls

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

        # Start turn tracing if enabled
        turn_ctx = None
        callbacks = None
        if self._tracing_manager:
            turn_ctx = self._tracing_manager.turn(
                turn_id=turn.turn_number,
                user_input=user_input,
                agent_id=agent.id,
            )
            turn_ctx.__enter__()
            callbacks = self._tracing_manager.get_langchain_callbacks()

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
            stop_reason: str | None = None
            consecutive_failures = 0  # Track no-tool-call failures

            # Tool call loop with enforcement
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

                # Invoke LLM with tools and tracing callbacks
                response = await self._provider.invoke(
                    messages,
                    self._model,
                    options,
                    tools=tool_schemas if tool_schemas else None,
                    callbacks=callbacks,
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
                    # No tool calls - check if we should enforce tool usage
                    if enforce_tool_usage and tool_schemas:
                        consecutive_failures += 1
                        logger.warning(
                            "No tool calls in iteration %d (failure %d/%d)",
                            iteration + 1,
                            consecutive_failures,
                            MAX_CONSECUTIVE_FAILURES,
                        )

                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.error(
                                "Max consecutive failures (%d) reached without tool calls",
                                MAX_CONSECUTIVE_FAILURES,
                            )
                            stop_reason = "max_failures"
                            break

                        # Add assistant message and nudge
                        messages.append(
                            LLMMessage(role="assistant", content=response.content or "")
                        )
                        nudge = self._build_tool_nudge_message(agent)
                        messages.append(LLMMessage(role="user", content=nudge))
                        continue
                    else:
                        # Tool enforcement disabled or no tools - we're done
                        break

                # Reset failure count on successful tool calls
                consecutive_failures = 0

                # Execute tool calls
                tool_calls = response.tool_calls  # Already checked not None
                logger.info(f"Executing {len(tool_calls)} tool calls (iteration {iteration + 1})")
                tool_results = await self._execute_tool_calls(tool_calls, agent, session.id)
                all_tool_calls.extend(tool_results)

                # Check for stop tools
                stop_tool, stop_result = self._check_for_stop_tool(tool_results)
                if stop_tool:
                    logger.info("Stop tool '%s' called, ending activation", stop_tool)
                    stop_reason = stop_tool
                    # Still add the messages for context
                    messages.append(LLMMessage(role="assistant", content=response.content or ""))
                    tool_messages = self._tool_results_to_messages(tool_calls, tool_results)
                    messages.extend(tool_messages)
                    break

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

            # End turn tracing with success
            if turn_ctx and self._tracing_manager:
                self._tracing_manager.end_turn(
                    output=final_content,
                    token_usage={
                        "prompt_tokens": usage.prompt_tokens or 0,
                        "completion_tokens": usage.completion_tokens or 0,
                        "total_tokens": usage.total_tokens or 0,
                    }
                    if usage
                    else None,
                )
                turn_ctx.__exit__(None, None, None)

            return ActivationResult(
                content=final_content,
                agent_id=agent.id,
                turn=turn,
                usage=usage,
                tool_calls=all_tool_calls,
                stop_reason=stop_reason,
            )

        except Exception as e:
            # End turn tracing with error
            if turn_ctx and self._tracing_manager:
                self._tracing_manager.end_turn(error=str(e))
                turn_ctx.__exit__(type(e), e, e.__traceback__)

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
        enforce_tool_usage: bool = True,
        max_iterations: int = 5,
    ) -> AsyncIterator[StreamChunk]:
        """
        Activate an agent with streaming response.

        With enforce_tool_usage=True (default), if the LLM doesn't make tool calls,
        it will be nudged and given another chance to use tools. This implements
        the validate-with-feedback pattern from v3.

        Args:
            agent: The agent to activate
            user_input: User's input message
            session: Session to track the turn in
            options: LLM invocation options
            enforce_tool_usage: If True, nudge LLM when it doesn't make tool calls
            max_iterations: Maximum streaming iterations for enforcement loop

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

        # Start turn tracing if enabled
        turn_ctx = None
        callbacks = None
        if self._tracing_manager:
            turn_ctx = self._tracing_manager.turn(
                turn_id=turn.turn_number,
                user_input=user_input,
                agent_id=agent.id,
            )
            turn_ctx.__enter__()
            callbacks = self._tracing_manager.get_langchain_callbacks()

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

            # Collect response for turn completion
            full_content = ""
            final_usage: TokenUsage | None = None
            consecutive_failures = 0

            # Streaming loop with enforcement
            for iteration in range(max_iterations):
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

                iteration_content = ""
                pending_tool_calls: list[ToolCallRequest] | None = None

                # Stream from provider with tools and tracing callbacks
                async for chunk in self._provider.stream(
                    messages,
                    self._model,
                    options,
                    tools=tool_schemas if tool_schemas else None,
                    callbacks=callbacks,
                ):
                    iteration_content += chunk.content

                    if chunk.done:
                        if final_usage is None:
                            final_usage = TokenUsage(
                                prompt_tokens=chunk.prompt_tokens,
                                completion_tokens=chunk.completion_tokens,
                                total_tokens=chunk.total_tokens,
                            )
                        elif chunk.prompt_tokens:
                            # Accumulate usage across iterations
                            final_usage = TokenUsage(
                                prompt_tokens=(final_usage.prompt_tokens or 0)
                                + chunk.prompt_tokens,
                                completion_tokens=(final_usage.completion_tokens or 0)
                                + (chunk.completion_tokens or 0),
                                total_tokens=(final_usage.total_tokens or 0)
                                + (chunk.total_tokens or 0),
                            )
                        # Check for tool calls in final chunk
                        if chunk.tool_calls:
                            pending_tool_calls = chunk.tool_calls

                    yield chunk

                full_content += iteration_content

                # Handle tool calls if present
                if pending_tool_calls:
                    consecutive_failures = 0  # Reset on tool call
                    logger.info(
                        f"Executing {len(pending_tool_calls)} tool calls from streaming response"
                    )
                    tool_results = await self._execute_tool_calls(
                        pending_tool_calls, agent, session.id
                    )

                    # Check for stop tools
                    stop_tool, _ = self._check_for_stop_tool(tool_results)
                    if stop_tool:
                        logger.info("Stop tool '%s' called, ending streaming", stop_tool)
                        # Add messages for completeness
                        messages.append(LLMMessage(role="assistant", content=iteration_content))
                        tool_messages = self._tool_results_to_messages(
                            pending_tool_calls, tool_results
                        )
                        messages.extend(tool_messages)
                        break

                    # Add assistant message and tool results
                    messages.append(LLMMessage(role="assistant", content=iteration_content))
                    tool_messages = self._tool_results_to_messages(pending_tool_calls, tool_results)
                    messages.extend(tool_messages)

                    # Continue streaming with tool results in context
                    # (next iteration will stream with updated messages)
                    continue

                # No tool calls - check if we should enforce
                if enforce_tool_usage and tool_schemas:
                    consecutive_failures += 1
                    logger.warning(
                        "No tool calls in streaming iteration %d (failure %d/%d)",
                        iteration + 1,
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                    )

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            "Max consecutive failures (%d) reached in streaming",
                            MAX_CONSECUTIVE_FAILURES,
                        )
                        break

                    # Add nudge and continue streaming
                    messages.append(LLMMessage(role="assistant", content=iteration_content))
                    nudge = self._build_tool_nudge_message(agent)
                    messages.append(LLMMessage(role="user", content=nudge))

                    # Yield a marker chunk to indicate retry
                    yield StreamChunk(
                        content="\n\n[Retrying with tool guidance...]\n\n",
                        done=False,
                    )
                    continue

                # No enforcement or no tools - we're done
                break

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

            # End turn tracing with success
            if turn_ctx and self._tracing_manager:
                self._tracing_manager.end_turn(
                    output=full_content,
                    token_usage={
                        "prompt_tokens": final_usage.prompt_tokens or 0,
                        "completion_tokens": final_usage.completion_tokens or 0,
                        "total_tokens": final_usage.total_tokens or 0,
                    }
                    if final_usage
                    else None,
                )
                turn_ctx.__exit__(None, None, None)

        except Exception as e:
            # End turn tracing with error
            if turn_ctx and self._tracing_manager:
                self._tracing_manager.end_turn(error=str(e))
                turn_ctx.__exit__(type(e), e, e.__traceback__)

            session.error_turn(turn, str(e))
            # Log error
            if self._event_logger:
                self._event_logger.turn_error(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    error=str(e),
                )
            raise

    async def process_pending_delegations(
        self,
        session: Session,
        max_depth: int = 5,
        _current_depth: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Process any pending delegation_request messages.

        After an orchestrator delegates work, this method processes
        the pending delegations by activating the delegatee agents.

        Args:
            session: Current session
            max_depth: Maximum delegation depth to prevent infinite loops
            _current_depth: Current depth (internal recursion tracking)

        Returns:
            List of delegation results
        """
        if not self._broker:
            logger.debug("No broker available, skipping delegation processing")
            return []

        if _current_depth >= max_depth:
            logger.warning("Max delegation depth %d reached, stopping", max_depth)
            return []

        from questfoundry.runtime.messaging import create_delegation_response
        from questfoundry.runtime.messaging.types import MessageType

        results = []

        # Check all agents for pending delegation_requests
        for agent in self._studio.agents:
            mailbox = await self._broker.get_mailbox(agent.id)

            # Collect all messages first, then filter
            all_messages = []
            while True:
                msg = await mailbox.get_nowait()
                if msg is None:
                    break
                all_messages.append(msg)

            # Separate delegation requests from other messages
            delegation_requests = []
            other_messages = []
            for msg in all_messages:
                if msg.type == MessageType.DELEGATION_REQUEST:
                    delegation_requests.append(msg)
                else:
                    other_messages.append(msg)

            # Put non-delegation messages back
            for msg in other_messages:
                await mailbox.put(msg)

            # Process delegation requests
            for msg in delegation_requests:
                delegation_id = msg.delegation_id or msg.id
                task = msg.payload.get("task", "")
                context_data = msg.payload.get("context", {})

                logger.info(
                    "Processing delegation %s: %s -> %s (task: %s)",
                    delegation_id,
                    msg.from_agent,
                    msg.to_agent,
                    task[:50] + "..." if len(task) > 50 else task,
                )

                try:
                    # Get the delegatee agent
                    delegatee = self.get_agent(agent.id)
                    if not delegatee:
                        raise ValueError(f"Agent not found: {agent.id}")

                    # Build the task prompt including context
                    task_prompt = f"You have been delegated a task.\n\nTask: {task}"
                    if context_data:
                        task_prompt += f"\n\nContext: {json.dumps(context_data, indent=2)}"

                    # Activate the delegatee (non-streaming for delegations)
                    activation_result = await self.activate(
                        agent=delegatee,
                        user_input=task_prompt,
                        session=session,
                        max_tool_iterations=5,
                    )

                    # Create success response
                    response_msg = create_delegation_response(
                        from_agent=agent.id,
                        to_agent=msg.from_agent,
                        delegation_id=delegation_id,
                        success=True,
                        result={"content": activation_result.content},
                        in_reply_to=msg.id,
                        playbook_id=msg.playbook_id,
                        playbook_instance_id=msg.playbook_instance_id,
                        phase_id=msg.phase_id,
                    )
                    await self._broker.send(response_msg)

                    results.append(
                        {
                            "delegation_id": delegation_id,
                            "from_agent": msg.from_agent,
                            "to_agent": agent.id,
                            "task": task,
                            "success": True,
                            "result": activation_result.content,
                        }
                    )

                    logger.info(
                        "Delegation %s completed successfully",
                        delegation_id,
                    )

                    # Recursively process any delegations the delegatee made
                    nested_results = await self.process_pending_delegations(
                        session=session,
                        max_depth=max_depth,
                        _current_depth=_current_depth + 1,
                    )
                    results.extend(nested_results)

                except Exception as e:
                    logger.error(
                        "Delegation %s failed: %s",
                        delegation_id,
                        e,
                    )

                    # Create failure response
                    response_msg = create_delegation_response(
                        from_agent=agent.id,
                        to_agent=msg.from_agent,
                        delegation_id=delegation_id,
                        success=False,
                        error=str(e),
                        in_reply_to=msg.id,
                    )
                    await self._broker.send(response_msg)

                    results.append(
                        {
                            "delegation_id": delegation_id,
                            "from_agent": msg.from_agent,
                            "to_agent": agent.id,
                            "task": task,
                            "success": False,
                            "error": str(e),
                        }
                    )

        return results


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
