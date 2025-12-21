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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.agent.context import AgentContext, ContextBuilder
from questfoundry.runtime.agent.prompt import PromptBuilder, build_prompt
from questfoundry.runtime.agent.turn_validator import (
    TurnValidationConfig,
    TurnValidator,
)
from questfoundry.runtime.context import (
    CachedToolResult,
    CacheScope,
    ContextSecretary,
    Secretary,
    SummarizationPolicy,
    ToolResultCache,
)
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
from questfoundry.runtime.storage import LifecycleManager, StoreManager

if TYPE_CHECKING:
    from questfoundry.runtime.checkpoint import CheckpointManager, ContextUsage
    from questfoundry.runtime.delegation.tracker import PlaybookTracker
    from questfoundry.runtime.messaging import Message
    from questfoundry.runtime.models import Agent, Studio
    from questfoundry.runtime.observability import EventLogger, TracingManager
    from questfoundry.runtime.storage import Project
from questfoundry.runtime.tools import BaseTool, ToolExecutionError, ToolRegistry, ToolResult

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_id": self.tool_id,
            "args": self.args,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Create from dictionary."""
        return cls(
            tool_id=data["tool_id"],
            args=data.get("args", {}),
            result=data.get("result"),
            success=data.get("success", False),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms"),
        )


@dataclass
class ActivationResult:
    """Result of an agent activation."""

    content: str
    agent_id: str
    turn: Turn
    usage: TokenUsage | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None  # "delegate", "terminate", "max_iterations", None
    session_terminated: bool = False  # True if a tool with terminates_session=True was called


# Tool names that are "stop tools" - calling them returns control to orchestrator/human
STOP_TOOL_NAMES = frozenset(
    {
        "delegate",  # Delegation to another agent
        "terminate",  # End workflow (legacy name)
        "terminate_session",  # End session explicitly
        "return_to_orchestrator",  # Delegatee returns to orchestrator
        "request_clarification",  # Ask human for clarification (legacy)
        "communicate",  # Human communication (questions are blocking)
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
        checkpoint_manager: CheckpointManager | None = None,
        playbook_tracker: PlaybookTracker | None = None,
        turn_validation_config: TurnValidationConfig | None = None,
        on_tool_call: Any | None = None,
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
            checkpoint_manager: Optional checkpoint manager for auto-checkpointing
            playbook_tracker: Optional playbook tracker for checkpoint state
            turn_validation_config: Optional configuration for orchestrator enforcement
            on_tool_call: Optional callback(tool_id, success, agent_id, turn_number, execution_time_ms, result)
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
        self._checkpoint_manager = checkpoint_manager
        self._playbook_tracker = playbook_tracker
        self._on_tool_call = on_tool_call

        # Context usage tracking per agent
        self._context_usage: dict[str, ContextUsage] = {}

        self._context_builder = ContextBuilder(domain_path=domain_path)
        self._prompt_builder = PromptBuilder()
        self._tool_registry: ToolRegistry | None = None
        self._store_manager: StoreManager | None = None
        self._lifecycle_manager: LifecycleManager | None = None
        try:
            self._store_manager = StoreManager.from_studio(studio)
        except (KeyError, ValueError) as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load store manager from studio definition: %s", exc)

        # Initialize lifecycle manager from artifact types
        try:
            self._lifecycle_manager = LifecycleManager.from_artifact_types(studio.artifact_types)
        except (KeyError, ValueError, AttributeError) as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load lifecycle manager from studio definition: %s", exc)

        # Secretary for tiered context management
        # context_limit from model determines when to start summarizing
        self._secretary = Secretary(
            context_limit=context_limit or 128000,  # Default 128k for modern models
            summarization_threshold=0.7,  # Start tool summarization at 70%
        )
        self._secretary_initialized = False

        # Context Secretary for full conversation summarization (Level 2)
        # Summarizes older turns when context pressure reaches FULL threshold (90%)
        self._context_secretary = ContextSecretary(
            preserve_recent_turns=3,  # Always preserve last 3 turns
            min_turns_to_summarize=5,  # Need at least 5 older turns before summarizing
        )

        # Tool result cache for deduplication (Phase 2 of epic #240)
        # - ACTIVATION scope: Intra-turn deduplication (same tool called twice in one turn)
        # - SESSION scope: Static tool caching (consult_* tools across turns)
        self._tool_cache = ToolResultCache()

        # Turn validator for orchestrator enforcement
        self._turn_validator = TurnValidator(studio, turn_validation_config)

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
                    store_manager=self._store_manager,
                    lifecycle_manager=self._lifecycle_manager,
                )
            except ImportError:
                logger.warning("Tools module not available, tool execution disabled", exc_info=True)
                return None

        # Initialize secretary with tool definitions (once)
        if not self._secretary_initialized and self._tool_registry:
            self._initialize_secretary()

        return self._tool_registry

    def _initialize_secretary(self) -> None:
        """Initialize the Secretary with tool definitions for summarization."""
        for tool in self._studio.tools:
            self._secretary.register_tool(tool)
        self._secretary_initialized = True
        logger.debug(
            "Secretary initialized with %d tools",
            len(self._studio.tools),
        )

    @property
    def secretary(self) -> Secretary:
        """Get the Secretary for context management."""
        return self._secretary

    @property
    def context_secretary(self) -> ContextSecretary:
        """Get the ContextSecretary for conversation summarization."""
        return self._context_secretary

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
        turn_number: int | None = None,
    ) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_id: ID of tool to execute
            args: Tool arguments
            agent: Agent requesting tool execution
            session_id: Current session ID
            turn_number: Current turn number for logging

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

        tool_trace_ctx = (
            self._tracing_manager.tool_call(
                tool_id,
                agent_id=agent.id,
                agent_name=agent.name,
                args=args,
            )
            if self._tracing_manager
            else nullcontext()
        )

        with tool_trace_ctx:
            result = await tool.run(args)

        # Log tool call result with full data for debugging
        if self._event_logger:
            self._event_logger.tool_call_with_result(
                session_id=session_id or "",
                turn_id=turn_number or 0,
                agent_id=agent.id,
                tool_id=tool_id,
                args=args,
                success=result.success,
                result=result.data,
                error=result.error,
                execution_time_ms=result.execution_time_ms,
            )

        if result.fatal:
            raise ToolExecutionError(result.error or f"Fatal error executing tool '{tool_id}'")

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
        history: list[dict[str, Any]] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[LLMMessage]:
        """
        Build messages for LLM invocation.

        Args:
            agent: The agent to activate
            user_input: User's input message
            context: Pre-built context (built if not provided)
            history: Conversation history - supports both simple {role, content}
                     and rich format with tool_calls, tool_call_id, name
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

        # History - handle both simple and rich formats
        if history:
            for h in history:
                # Reconstruct ToolCallRequest objects if present
                tool_calls = None
                if h.get("tool_calls"):
                    tool_calls = [
                        ToolCallRequest(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc.get("arguments", {}),
                        )
                        for tc in h["tool_calls"]
                    ]

                messages.append(
                    LLMMessage(
                        role=h["role"],
                        content=h.get("content", ""),
                        tool_call_id=h.get("tool_call_id"),
                        name=h.get("name"),
                        tool_calls=tool_calls,
                    )
                )

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

    def _apply_context_summarization(
        self,
        history: list[dict[str, Any]] | None,
        agent_id: str,
    ) -> list[dict[str, Any]] | None:
        """
        Apply context summarization if the history exceeds thresholds.

        Uses the ContextSecretary to summarize older turns while preserving
        recent and important turns.

        Per PR #180's tiered design, context summarization only happens at
        FULL level (>= 90% context usage). This prevents premature summarization
        when context pressure is low.

        Args:
            history: Conversation history (list of message dicts)
            agent_id: Agent ID for logging

        Returns:
            Possibly summarized history
        """
        if not history:
            return history

        # Estimate actual history size to check context pressure
        # Can't rely on Secretary's cached value - it has the previous turn's size
        history_tokens = sum(len(json.dumps(m)) // 4 for m in history)
        context_limit = self._context_limit or self._secretary.context_limit
        usage_fraction = history_tokens / context_limit if context_limit > 0 else 0

        # Gate on context pressure: only summarize at FULL level (>= 90%)
        # Per PR #180 tiered design - don't summarize when context pressure is low
        if usage_fraction < self._secretary.full_summarization_threshold:
            return history

        # Check if summarization is needed (group count check)
        result = self._context_secretary.summarize_context(history)

        if not result.summary_created:
            return history

        # Use preserved turns from result (avoids recomputing partition)
        preserved = result.preserved_turns or []

        # Create summary message as first entry
        summary_message = {
            "role": "system",
            "content": f"[Context summary for continuity]\n{result.summary_text}",
        }

        logger.info(
            "Agent %s: summarized %d turns, preserving %d",
            agent_id,
            result.turns_summarized,
            result.turns_preserved,
        )

        return [summary_message] + preserved

    # Static tools whose results don't change during a session
    STATIC_TOOLS = frozenset(
        {
            "consult_playbook",
            "consult_schema",
            "consult_knowledge",
            "consult_corpus",
        }
    )

    # Size threshold for tool result summarization (~500 tokens = 2000 chars)
    TOOL_RESULT_SIZE_THRESHOLD = 2000

    def _deduplicate_static_tool_results(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate static tool results from history.

        For static tools (consult_*), the result is the same every time.
        We keep only the first occurrence and remove subsequent duplicates,
        including the corresponding tool_call from assistant messages.

        Args:
            history: Conversation history

        Returns:
            History with duplicate static tool results removed
        """
        import hashlib

        # Track seen static tool results: (tool_name, content_hash) -> tool_call_id
        seen_static: dict[tuple[str, str], str] = {}
        # Track tool_call_ids to remove from assistant messages
        duplicate_tool_call_ids: set[str] = set()

        # First pass: identify duplicates
        for msg in history:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("name", "")
            if tool_name not in self.STATIC_TOOLS:
                continue

            content = msg.get("content", "")
            # MD5 is fine here - used for deduplication, not security
            content_hash = hashlib.md5(content.encode()).hexdigest()  # noqa: S324
            key = (tool_name, content_hash)

            tool_call_id = msg.get("tool_call_id")
            if key in seen_static:
                # Duplicate - mark for removal
                if tool_call_id:
                    duplicate_tool_call_ids.add(tool_call_id)
            else:
                # First occurrence - keep it
                seen_static[key] = tool_call_id or ""

        if not duplicate_tool_call_ids:
            return history  # No duplicates found

        # Second pass: filter out duplicates
        result: list[dict[str, Any]] = []
        for msg in history:
            role = msg.get("role")

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in duplicate_tool_call_ids:
                    continue  # Skip duplicate tool result

            elif role == "assistant" and msg.get("tool_calls"):
                # Filter out duplicate tool_calls from assistant message
                original_calls = msg.get("tool_calls", [])
                filtered_calls = [
                    tc for tc in original_calls if tc.get("id") not in duplicate_tool_call_ids
                ]
                if filtered_calls:
                    # Keep message with filtered tool_calls
                    new_msg = dict(msg)
                    new_msg["tool_calls"] = filtered_calls
                    result.append(new_msg)
                elif msg.get("content"):
                    # Keep message without tool_calls if it has content
                    new_msg = dict(msg)
                    new_msg.pop("tool_calls", None)
                    result.append(new_msg)
                # Else: skip empty assistant message
                continue

            result.append(msg)

        logger.info(
            "Deduplicated %d static tool results from history",
            len(duplicate_tool_call_ids),
        )
        return result

    def _summarize_historical_tool_results(
        self,
        history: list[dict[str, Any]],
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """
        Apply summarization to large tool results in history.

        Called when context pressure is high (>= 70%). Summarizes large tool
        results from previous turns to save space.

        Args:
            history: Conversation history
            agent_id: Agent ID for context tracking

        Returns:
            History with large tool results summarized
        """
        result: list[dict[str, Any]] = []
        for msg in history:
            if msg.get("role") != "tool":
                result.append(msg)
                continue

            content = msg.get("content", "")
            if len(content) <= self.TOOL_RESULT_SIZE_THRESHOLD:
                result.append(msg)
                continue

            # Summarize large tool result
            tool_name = msg.get("name", "unknown")

            # Use tool's summarization policy if available
            summary = self._secretary.summarize_tool_result(
                tool_id=tool_name,
                result=content,
                tool_call_id=msg.get("tool_call_id"),
                agent_id=agent_id,
            )

            new_msg = dict(msg)
            if summary.policy_applied == SummarizationPolicy.DROP:
                new_msg["content"] = json.dumps(
                    {
                        "_summarized": f"[Large {tool_name} result dropped to save context]",
                        "_original_size": len(content),
                    }
                )
            elif summary.content:
                new_msg["content"] = json.dumps(
                    {
                        "_summarized": summary.content,
                        "_tool": tool_name,
                        "_original_size": len(content),
                    }
                )
            # else: keep original (PRESERVE policy)

            result.append(new_msg)

        return result

    def _get_agent_history_with_summarization(
        self,
        session: Session,
        agent_id: str,
    ) -> list[dict[str, Any]] | None:
        """
        Get agent-specific history with context summarization applied.

        Each agent only sees their own turns (not other agents' internal
        conversations), and older turns are summarized when needed to
        prevent context overflow.

        Processing pipeline:
        1. Get agent-specific history
        2. Deduplicate static tool results (consult_*)
        3. Summarize large tool results when context is high
        4. Apply full context summarization if still over threshold

        Args:
            session: Current session
            agent_id: Agent to get history for

        Returns:
            History with summarization applied, or None if no prior turns
        """
        # Only include history if this agent has prior completed turns
        agent_turn_count = session.get_agent_turn_count(agent_id)
        if agent_turn_count <= 1:
            return None

        # Get agent-specific history (filters out other agents' turns)
        agent_history = session.get_agent_history(agent_id)
        if not agent_history:
            return None

        # Check context pressure to decide on summarization
        history_tokens = sum(len(json.dumps(m)) // 4 for m in agent_history)
        context_limit = self._context_limit or self._secretary.context_limit
        usage_fraction = history_tokens / context_limit if context_limit > 0 else 0

        # Only apply deduplication and tool summarization when context is high (>= 70%)
        if usage_fraction >= self._secretary.summarization_threshold:
            # Step 1: Deduplicate static tool results
            agent_history = self._deduplicate_static_tool_results(agent_history)

            # Step 2: Summarize large tool results
            agent_history = self._summarize_historical_tool_results(agent_history, agent_id)

        # Step 3: Apply full context summarization if still over threshold (>= 90%)
        return self._apply_context_summarization(agent_history, agent_id)

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCallRequest],
        agent: Agent,
        session_id: str | None = None,
        turn_number: int | None = None,
    ) -> list[ToolCall]:
        """
        Execute a list of tool calls with caching.

        Uses ToolResultCache for deduplication:
        - ACTIVATION scope: Prevents same tool called twice in one turn
        - SESSION scope: Caches static tools (consult_*) across turns

        Args:
            tool_calls: List of tool call requests from LLM
            agent: Agent making the calls
            session_id: Current session ID
            turn_number: Current turn number for logging

        Returns:
            List of ToolCall results
        """
        results = []

        # Create activation ID for intra-turn caching
        activation_id = f"{session_id}:{turn_number}" if session_id and turn_number else None

        for tc in tool_calls:
            tool_call = ToolCall(tool_id=tc.name, args=tc.arguments)
            start_time = time.time()

            # Get caching policy for this tool
            policy = self._tool_cache.get_policy(tc.name)
            cached_hit: CachedToolResult | None = None

            # Check activation cache first (intra-turn deduplication)
            if policy.participate_in_activation_cache and activation_id and session_id:
                cached_hit = self._tool_cache.lookup(
                    session_id=session_id,
                    scope=CacheScope.ACTIVATION,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    activation_id=activation_id,
                )

            # Check session cache for static tools (inter-turn caching)
            if not cached_hit and policy.participate_in_session_cache and session_id:
                cached_hit = self._tool_cache.lookup(
                    session_id=session_id,
                    scope=CacheScope.SESSION,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                )

            if cached_hit:
                # Use cached result
                tool_call.result = {"_cached": True, "_ref": cached_hit.presentation_id}
                tool_call.success = cached_hit.success
                tool_call.execution_time_ms = 0.1  # Negligible time for cache hit
                logger.debug(f"Cache hit for {tc.name}: {cached_hit.presentation_id}")
            else:
                # Execute the tool
                try:
                    result = await self.execute_tool(
                        tc.name, tc.arguments, agent, session_id, turn_number
                    )
                    tool_call.result = result.data
                    tool_call.success = result.success
                    tool_call.error = result.error
                    tool_call.execution_time_ms = result.execution_time_ms

                    # Cache successful results
                    if result.success and session_id:
                        cached_result = CachedToolResult(
                            tool_name=tc.name,
                            args_json=json.dumps(tc.arguments, sort_keys=True),
                            content=result.data,
                            success=True,
                        )

                        # Store in activation cache
                        if policy.participate_in_activation_cache and activation_id:
                            self._tool_cache.record(
                                session_id=session_id,
                                scope=CacheScope.ACTIVATION,
                                tool_name=tc.name,
                                tool_args=tc.arguments,
                                result=cached_result,
                                activation_id=activation_id,
                            )

                        # Store in session cache for static tools
                        if policy.participate_in_session_cache:
                            self._tool_cache.record(
                                session_id=session_id,
                                scope=CacheScope.SESSION,
                                tool_name=tc.name,
                                tool_args=tc.arguments,
                                result=cached_result,
                            )

                except Exception as e:
                    tool_call.success = False
                    tool_call.error = str(e)
                    tool_call.execution_time_ms = (time.time() - start_time) * 1000
                    logger.warning(f"Tool call failed: {tc.name} - {e}")

            results.append(tool_call)

            # Notify UI callback if registered
            if self._on_tool_call:
                try:
                    self._on_tool_call(
                        tool_id=tool_call.tool_id,
                        success=tool_call.success,
                        agent_id=agent.id,
                        turn_number=turn_number,
                        execution_time_ms=tool_call.execution_time_ms,
                        result=tool_call.result,
                    )
                except Exception as e:
                    logger.debug(f"Tool callback failed: {e}")

        return results

    def _tool_results_to_messages(
        self,
        tool_calls: list[ToolCallRequest],
        tool_results: list[ToolCall],
        agent_id: str | None = None,
    ) -> list[LLMMessage]:
        """
        Convert tool call results to LLM messages.

        Uses tiered Secretary pattern:
        - Below tool_threshold: Preserve full results (no summarization)
        - Above tool_threshold: Apply tool summarization policies

        Args:
            tool_calls: Original tool call requests (with IDs)
            tool_results: Executed tool results
            agent_id: Agent making the calls (for per-agent summarization)

        Returns:
            List of tool result messages for the LLM
        """
        # Check if we should apply tool summarization based on context usage (per-agent)
        if agent_id:
            apply_summarization = self._secretary.should_summarize_tools_for_agent(agent_id)
        else:
            apply_summarization = self._secretary.should_summarize_tools()

        if apply_summarization:
            usage = (
                self._secretary.get_usage_fraction(agent_id)
                if agent_id
                else self._secretary.usage_fraction
            )
            level = (
                self._secretary.get_current_level(agent_id)
                if agent_id
                else self._secretary.current_level
            )
            logger.debug(
                "Applying tool summarization for %s (context at %.1f%%, level=%s)",
                agent_id or "global",
                usage * 100,
                level.name,
            )

        messages = []
        for tc, result in zip(tool_calls, tool_results, strict=True):
            # Handle errors (always preserve error messages and feedback data)
            if not result.success:
                # Include the full result.result (contains validation feedback, etc.)
                # so the LLM can self-correct based on detailed error info
                error_response = result.result.copy() if result.result else {}
                runtime_error = result.error or "Tool execution failed"
                # Avoid overwriting if result already has an "error" key
                if "error" in error_response:
                    error_response["runtime_error"] = runtime_error
                else:
                    error_response["error"] = runtime_error
                content = json.dumps(error_response)
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=content,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
                continue

            # Apply Secretary summarization when above threshold
            # Pass tool_call_id for recency tracking - recent results always preserved
            if apply_summarization and result.result:
                summary = self._secretary.summarize_tool_result(
                    tool_id=tc.name,
                    result=result.result,
                    tool_call_id=tc.id,
                    agent_id=agent_id,
                    arguments=tc.arguments,
                )

                # Wrap summarized content in consistent JSON format
                if summary.policy_applied == SummarizationPolicy.DROP:
                    content = json.dumps({"_summarized": "dropped", "_tool": tc.name})
                elif summary.policy_applied == SummarizationPolicy.PRESERVE:
                    # PRESERVE keeps original JSON as-is
                    content = summary.content or "{}"
                elif summary.content is not None:
                    # ULTRA_CONCISE and CONCISE wrap text in JSON for consistency
                    content = json.dumps({"_summarized": summary.content, "_tool": tc.name})
                else:
                    content = "{}"
            else:
                # No summarization - use raw JSON (full fidelity)
                # Still track tool call for recency window (per-agent)
                if tc.id:
                    self._secretary.track_tool_call(tc.id, agent_id)
                content = json.dumps(result.result) if result.result else "{}"

            messages.append(
                LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
        return messages

    @property
    def turn_validator(self) -> TurnValidator:
        """Get the turn validator for orchestrator enforcement."""
        return self._turn_validator

    def _is_orchestrator(self, agent: Agent) -> bool:
        """Check if an agent is an orchestrator."""
        return self._turn_validator.is_orchestrator(agent)

    def _requires_terminating_tool(self, agent: Agent) -> bool:
        """Check if an agent requires a terminating tool to end its turn."""
        return self._turn_validator.requires_terminating_tool(agent)

    def _update_context_usage(self, agent_id: str, usage: TokenUsage) -> None:
        """
        Update context usage tracking for an agent.

        Args:
            agent_id: Agent ID to update
            usage: Token usage from turn
        """
        from questfoundry.runtime.checkpoint import ContextUsage

        if agent_id not in self._context_usage:
            self._context_usage[agent_id] = ContextUsage(agent_id=agent_id)

        ctx_usage = self._context_usage[agent_id]
        ctx_usage.add_usage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )

        # Log warning if approaching limit
        if ctx_usage.at_warning and not ctx_usage.at_limit:
            logger.warning(
                "Agent %s approaching context limit: %d/%d tokens (%.1f%%)",
                agent_id,
                ctx_usage.total_tokens,
                ctx_usage.limit,
                ctx_usage.usage_percent,
            )

    async def _create_auto_checkpoint(self, session: Session) -> None:
        """
        Create an automatic checkpoint after orchestrator turn.

        Respects auto_checkpoint and checkpoint_frequency from config.

        Args:
            session: Current session
        """
        if not self._checkpoint_manager or not self._broker:
            return

        config = self._checkpoint_manager.config

        # Check if auto-checkpointing is enabled
        if not config.auto_checkpoint:
            return

        # Check checkpoint frequency (skip if not at frequency interval)
        if (
            config.checkpoint_frequency > 1
            and session.turn_count % config.checkpoint_frequency != 0
        ):
            return

        try:
            # Let manager generate checkpoint ID (includes session_id)
            checkpoint = await self._checkpoint_manager.create_checkpoint(
                session=session,
                broker=self._broker,
                tracker=self._playbook_tracker,
                context_usage=self._context_usage,
            )
            logger.info("Auto-checkpoint created: %s", checkpoint.id)
        except Exception as e:
            logger.error("Failed to create auto-checkpoint: %s", e)

    def restore_context_usage(self, context_usage: dict[str, Any]) -> None:
        """
        Restore context usage tracking from checkpoint.

        Args:
            context_usage: Dict of agent_id -> ContextUsage from checkpoint
        """
        self._context_usage = context_usage

    def _build_tool_nudge_message(
        self,
        agent: Agent,
        tool_calls: list[ToolCallRequest] | None = None,
    ) -> str:
        """
        Build a nudge message when the LLM doesn't comply with tool requirements.

        Uses TurnValidator to generate appropriate nudges based on:
        - Whether any tools were called
        - Whether a terminating tool was called (for orchestrators)

        Args:
            agent: Agent being nudged
            tool_calls: Tool calls made (if any) - used to detect missing terminator

        Returns:
            Nudge message string
        """
        # Use TurnValidator to validate and get appropriate nudge
        result = self._turn_validator.validate_turn(agent, tool_calls)

        if result.nudge_message:
            return result.nudge_message

        # Fallback for non-orchestrators without tools
        if not self._requires_terminating_tool(agent):
            return (
                "You must use tools to do your work and communicate results. "
                "Your response contained no tool calls.\n\n"
                "Please either:\n"
                "1. Call a tool to continue your work, OR\n"
                "2. Call `return_to_orchestrator` to report your work is complete\n\n"
                "IMPORTANT: Do NOT repeat your previous response. "
                "Simply make the required tool call without any preamble."
            )

        # Shouldn't reach here, but provide sensible default
        return "You must use tools to do your work. Your response did not complete correctly."

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
                    # For communicate tool, only stop if it's blocking (question type)
                    # Non-blocking types (status, notification, error with low severity) continue
                    # Note: tc.result is already result.data, not the full ToolResult
                    if tc.tool_id == "communicate" and not result.get("blocking", False):
                        continue  # Non-blocking communicate, don't stop
                    return tc.tool_id, result
        return None, None

    def _check_for_session_terminating_tool(
        self,
        tool_calls: list[ToolCall],
    ) -> bool:
        """
        Check if any tool call has terminates_session=True.

        This signals that the session should end completely (not just the turn).
        The terminate_session tool is the primary example.

        Returns:
            True if a session-terminating tool was called successfully.
        """
        registry = self.tool_registry
        if not registry:
            return False

        for tc in tool_calls:
            if tc.success:
                tool_def = registry.get_tool_definition(tc.tool_id)
                if tool_def and getattr(tool_def, "terminates_session", False):
                    logger.info("Session terminating tool called: %s", tc.tool_id)
                    return True
        return False

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
                agent_name=agent.name,
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

            # Get agent-specific history (not other agents' conversations)
            # Note: Current turn has no messages yet (just started), so
            # get_agent_history naturally excludes it via its messages filter.
            history = self._get_agent_history_with_summarization(session, agent.id)

            # Build messages
            messages = self.build_messages(agent, user_input, context, history, tool_schemas)

            # Track where new messages start (after system prompt + history)
            # This is used to store only this turn's messages, not the full trace
            # Structure: [system, *history, user_input, ...]
            # New messages start at: 1 (system) + len(history)
            history_len = len(history) if history else 0
            new_message_start = 1 + history_len

            # Log system prompt for debugging
            if self._event_logger:
                system_prompt = ""
                if messages and messages[0].role == "system":
                    system_prompt = messages[0].content
                self._event_logger.prompt_build(
                    session_id=session.id,
                    agent_id=agent.id,
                    prompt_text=system_prompt,
                    tool_count=len(tool_schemas) if tool_schemas else 0,
                )

            # Log messages sent to LLM
            if self._event_logger:
                self._event_logger.messages_sent(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    agent_id=agent.id,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )

            # Validate context size
            self.validate_context_size(messages)

            # Update Secretary's context tracking for tiered summarization (per-agent)
            estimated_tokens = self.estimate_tokens(messages)
            self._secretary.update_context_size(estimated_tokens, agent.id)

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

                # Log full LLM response for debugging
                if self._event_logger:
                    tool_calls_data = None
                    if response.tool_calls:
                        tool_calls_data = [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ]
                    self._event_logger.llm_response(
                        session_id=session.id,
                        turn_id=turn.turn_number,
                        agent_id=agent.id,
                        content=response.content,
                        tool_calls=tool_calls_data,
                        has_tool_calls=response.has_tool_calls,
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

                        # Add assistant message and nudge (no tool calls)
                        messages.append(
                            LLMMessage(role="assistant", content=response.content or "")
                        )
                        nudge = self._build_tool_nudge_message(agent, tool_calls=None)
                        messages.append(LLMMessage(role="user", content=nudge))
                        continue
                    else:
                        # Tool enforcement disabled or no tools - we're done
                        break

                # Have tool calls - execute them first, then validate
                tool_calls = response.tool_calls  # Already checked not None

                # Execute tool calls (always execute, even if validation will fail)
                logger.info(f"Executing {len(tool_calls)} tool calls (iteration {iteration + 1})")
                tool_results = await self._execute_tool_calls(
                    tool_calls, agent, session.id, turn.turn_number
                )
                all_tool_calls.extend(tool_results)

                # Validate turn with TurnValidator (checks for terminating tools)
                # Note: This orchestrator enforcement runs independently of enforce_tool_usage.
                # Orchestrators with runtime enforcement always require terminating tools.
                validation = self._turn_validator.validate_turn(agent, tool_calls)

                # Log turn validation for debugging
                if self._event_logger:
                    self._event_logger.turn_validation(
                        session_id=session.id,
                        turn_id=turn.turn_number,
                        agent_id=agent.id,
                        valid=validation.valid,
                        is_orchestrator=self._is_orchestrator(agent),
                        tool_calls_made=[tc.name for tc in tool_calls],
                        terminating_tool=validation.terminating_tool_id,
                        nudge_message=validation.nudge_message,
                    )

                if not validation.valid:
                    # Orchestrator didn't use terminating tool - tools already executed above
                    consecutive_failures += 1
                    logger.warning(
                        "Turn validation failed in iteration %d (failure %d/%d): %s",
                        iteration + 1,
                        consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES,
                        "missing terminating tool"
                        if validation.non_terminating_tools
                        else "no tools",
                    )

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            "Max consecutive failures (%d) reached without valid turn",
                            MAX_CONSECUTIVE_FAILURES,
                        )
                        stop_reason = "max_failures"
                        break

                    # Add assistant and tool messages (tools already executed above)
                    messages.append(
                        LLMMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )
                    tool_messages = self._tool_results_to_messages(
                        tool_calls, tool_results, agent.id
                    )
                    messages.extend(tool_messages)

                    # Add nudge about missing terminating tool
                    nudge = self._build_tool_nudge_message(agent, tool_calls)
                    messages.append(LLMMessage(role="user", content=nudge))
                    continue

                # Reset failure count on valid turn
                consecutive_failures = 0

                # Check if we should stop the loop.
                #
                # For orchestrators: only stop on explicit stop tools (blocking
                # communicate, delegate, terminate_session). The _check_for_stop_tool
                # method already returns None for non-blocking communicate, so
                # orchestrators continue after status updates.
                #
                # For specialists: any terminating tool returns control to orchestrator.
                stop_tool, stop_result = self._check_for_stop_tool(tool_results)
                if self._is_orchestrator(agent):
                    # Orchestrators: only stop on explicit stop tools
                    # Non-blocking communicate returns None from _check_for_stop_tool
                    should_stop = stop_tool is not None
                else:
                    # Specialists: any terminating tool returns to orchestrator
                    should_stop = (
                        stop_tool is not None or validation.terminating_tool_id is not None
                    )

                if should_stop:
                    stop_reason = stop_tool or validation.terminating_tool_id
                    logger.info("Stopping activation: %s", stop_reason)
                    # Still add the messages for context - include tool_calls for proper conversation flow
                    messages.append(
                        LLMMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )
                    tool_messages = self._tool_results_to_messages(
                        tool_calls, tool_results, agent.id
                    )
                    messages.extend(tool_messages)
                    break

                # Add assistant message with tool calls for proper LangChain conversation flow
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )

                # Add tool result messages
                tool_messages = self._tool_results_to_messages(tool_calls, tool_results, agent.id)
                messages.extend(tool_messages)

                # Update context size for next iteration's summarization decision (per-agent)
                self._secretary.update_context_size(self.estimate_tokens(messages), agent.id)

            # Complete turn with full message trace and tool calls
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
            # Store only this turn's new messages, not the full trace including history
            # This prevents O(n²) storage growth where each turn duplicates prior turns
            turn_messages = messages[new_message_start:]
            session.complete_turn(
                turn, final_content, usage, messages=turn_messages, tool_calls=all_tool_calls
            )
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

            # Update context usage tracking
            if usage:
                self._update_context_usage(agent.id, usage)

            # Check if session should terminate
            session_terminated = self._check_for_session_terminating_tool(all_tool_calls)

            # Auto-checkpoint after orchestrator turns
            if self._is_orchestrator(agent) and self._checkpoint_manager and self._broker:
                await self._create_auto_checkpoint(session)

            return ActivationResult(
                content=final_content,
                agent_id=agent.id,
                turn=turn,
                usage=usage,
                tool_calls=all_tool_calls,
                stop_reason=stop_reason,
                session_terminated=session_terminated,
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
        max_iterations: int = 10,
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
                agent_name=agent.name,
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

            # Get agent-specific history (not other agents' conversations)
            # Note: Current turn has no messages yet (just started), so
            # get_agent_history naturally excludes it via its messages filter.
            history = self._get_agent_history_with_summarization(session, agent.id)

            # Build messages
            messages = self.build_messages(agent, user_input, context, history, tool_schemas)

            # Track where new messages start (after system prompt + history)
            # This is used to store only this turn's messages, not the full trace
            # Structure: [system, *history, user_input, ...]
            # New messages start at: 1 (system) + len(history)
            history_len = len(history) if history else 0
            new_message_start = 1 + history_len

            # Log system prompt for debugging (streaming path)
            if self._event_logger:
                system_prompt = ""
                if messages and messages[0].role == "system":
                    system_prompt = messages[0].content
                self._event_logger.prompt_build(
                    session_id=session.id,
                    agent_id=agent.id,
                    prompt_text=system_prompt,
                    tool_count=len(tool_schemas) if tool_schemas else 0,
                )

            # Log messages sent to LLM (streaming path)
            if self._event_logger:
                self._event_logger.messages_sent(
                    session_id=session.id,
                    turn_id=turn.turn_number,
                    agent_id=agent.id,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )

            # Validate context size
            self.validate_context_size(messages)

            # Update Secretary's context tracking for tiered summarization (per-agent)
            self._secretary.update_context_size(self.estimate_tokens(messages), agent.id)

            # Collect response for turn completion
            full_content = ""
            final_usage: TokenUsage | None = None
            consecutive_failures = 0
            all_tool_calls: list[ToolCall] = []  # Track all tool calls for memory

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
                    # Log LLM response for debugging (streaming path)
                    if self._event_logger:
                        tool_calls_data = [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in pending_tool_calls
                        ]
                        self._event_logger.llm_response(
                            session_id=session.id,
                            turn_id=turn.turn_number,
                            agent_id=agent.id,
                            content=iteration_content,
                            tool_calls=tool_calls_data,
                            has_tool_calls=True,
                        )

                    # Execute tool calls first (always execute, even if validation will fail)
                    logger.info(
                        f"Executing {len(pending_tool_calls)} tool calls from streaming response"
                    )
                    tool_results = await self._execute_tool_calls(
                        pending_tool_calls, agent, session.id, turn.turn_number
                    )
                    all_tool_calls.extend(tool_results)  # Track for memory

                    # Validate turn with TurnValidator (checks for terminating tools)
                    # Note: This orchestrator enforcement runs independently of enforce_tool_usage.
                    # Orchestrators with runtime enforcement always require terminating tools.
                    validation = self._turn_validator.validate_turn(agent, pending_tool_calls)

                    # Log turn validation for debugging (streaming path)
                    if self._event_logger:
                        self._event_logger.turn_validation(
                            session_id=session.id,
                            turn_id=turn.turn_number,
                            agent_id=agent.id,
                            valid=validation.valid,
                            is_orchestrator=self._is_orchestrator(agent),
                            tool_calls_made=[tc.name for tc in pending_tool_calls],
                            terminating_tool=validation.terminating_tool_id,
                            nudge_message=validation.nudge_message,
                        )

                    if not validation.valid:
                        # Orchestrator didn't use terminating tool - tools already executed above
                        consecutive_failures += 1
                        logger.warning(
                            "Turn validation failed in streaming iteration %d (failure %d/%d)",
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

                        # Add assistant and tool messages (tools already executed above)
                        messages.append(
                            LLMMessage(
                                role="assistant",
                                content=iteration_content,
                                tool_calls=pending_tool_calls,
                            )
                        )
                        tool_messages = self._tool_results_to_messages(
                            pending_tool_calls, tool_results, agent.id
                        )
                        messages.extend(tool_messages)

                        # Add nudge about missing terminating tool
                        nudge = self._build_tool_nudge_message(agent, pending_tool_calls)
                        messages.append(LLMMessage(role="user", content=nudge))

                        # Yield a marker chunk to indicate retry
                        yield StreamChunk(
                            content="\n\n[Retrying - need terminating tool...]\n\n",
                            done=False,
                        )
                        continue

                    # Valid turn - reset failure count
                    consecutive_failures = 0

                    # Check if we should stop the loop.
                    #
                    # For orchestrators: only stop on explicit stop tools (delegate,
                    # blocking communicate, terminate_session). The _check_for_stop_tool
                    # method already returns None for non-blocking communicate, so
                    # orchestrators continue after status updates.
                    #
                    # For specialists: any terminating tool returns control to orchestrator.
                    stop_tool, _ = self._check_for_stop_tool(tool_results)
                    if self._is_orchestrator(agent):
                        # Orchestrators: only stop on explicit stop tools
                        # Non-blocking communicate returns None from _check_for_stop_tool
                        should_stop = stop_tool is not None
                    else:
                        # Specialists: any terminating tool returns to orchestrator
                        should_stop = (
                            stop_tool is not None or validation.terminating_tool_id is not None
                        )

                    if should_stop:
                        stop_reason = stop_tool or validation.terminating_tool_id
                        logger.info("Stopping streaming: %s", stop_reason)
                        # Add messages for completeness - include tool_calls for proper conversation flow
                        messages.append(
                            LLMMessage(
                                role="assistant",
                                content=iteration_content,
                                tool_calls=pending_tool_calls,
                            )
                        )
                        tool_messages = self._tool_results_to_messages(
                            pending_tool_calls, tool_results, agent.id
                        )
                        messages.extend(tool_messages)
                        break

                    # Add assistant message with tool calls for proper LangChain conversation flow
                    messages.append(
                        LLMMessage(
                            role="assistant",
                            content=iteration_content,
                            tool_calls=pending_tool_calls,
                        )
                    )
                    tool_messages = self._tool_results_to_messages(
                        pending_tool_calls, tool_results, agent.id
                    )
                    messages.extend(tool_messages)

                    # Update context size for next iteration's summarization decision (per-agent)
                    self._secretary.update_context_size(self.estimate_tokens(messages), agent.id)

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
                    nudge = self._build_tool_nudge_message(agent, tool_calls=None)
                    messages.append(LLMMessage(role="user", content=nudge))

                    # Yield a marker chunk to indicate retry
                    yield StreamChunk(
                        content="\n\n[Retrying with tool guidance...]\n\n",
                        done=False,
                    )
                    continue

                # No enforcement or no tools - we're done
                break

            # Store only this turn's new messages, not the full trace including history
            # This prevents O(n²) storage growth where each turn duplicates prior turns
            turn_messages = messages[new_message_start:]
            session.complete_turn(
                turn, full_content, final_usage, messages=turn_messages, tool_calls=all_tool_calls
            )
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

            # Update context usage tracking
            if final_usage:
                self._update_context_usage(agent.id, final_usage)

            # Auto-checkpoint after orchestrator turns
            if self._is_orchestrator(agent) and self._checkpoint_manager and self._broker:
                await self._create_auto_checkpoint(session)

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

    async def get_delegation_responses(self, agent_id: str) -> list[Message]:
        """
        Get pending delegation_response messages for an agent.

        This properly drains the mailbox to avoid duplicates.

        Args:
            agent_id: Agent to check mailbox for

        Returns:
            List of delegation_response messages
        """
        if not self._broker:
            return []

        from questfoundry.runtime.messaging.types import MessageType

        mailbox = await self._broker.get_mailbox(agent_id)

        # Drain the mailbox properly (same pattern as process_pending_delegations)
        all_messages: list[Message] = []
        while True:
            msg = await mailbox.get_nowait()
            if msg is None:
                break
            all_messages.append(msg)

        # Separate responses from other messages
        responses: list[Message] = []
        other_messages: list[Message] = []
        for msg in all_messages:
            if msg.type == MessageType.DELEGATION_RESPONSE:
                responses.append(msg)
            else:
                other_messages.append(msg)

        # Put back non-response messages
        for msg in other_messages:
            await mailbox.put(msg)

        return responses

    def build_delegation_response_prompt(self, responses: list[Message]) -> str:
        """
        Build a prompt summarizing delegation responses for the orchestrator.

        Args:
            responses: List of delegation_response messages

        Returns:
            Prompt text describing the delegation results
        """
        if not responses:
            return ""

        lines = ["Your delegated work has completed. Here are the results:\n"]

        for i, msg in enumerate(responses, 1):
            from_agent = msg.from_agent
            payload = msg.payload or {}
            success = payload.get("success", False)
            result = payload.get("result", {})

            # New semantic model: task_completion, result.assessment, recommendation
            task_completion = result.get("task_completion", "completed" if success else "failed")
            result_info = result.get("result", {})
            assessment = result_info.get("assessment", "info")
            recommendation = result.get("recommendation", "proceed")
            summary = result_info.get("summary") or result.get("summary", "No summary provided")

            # artifacts_produced is a sibling of result in payload, not inside result
            artifacts = payload.get("artifacts_produced", [])
            ready_for_review = result.get("artifacts_ready_for_review", [])
            details = result_info.get("details", [])

            lines.append(f"## Delegation {i}: {from_agent}")
            lines.append(f"- **Task**: {task_completion}")
            lines.append(f"- **Assessment**: {assessment}")
            lines.append(f"- **Recommendation**: {recommendation}")
            lines.append(f"- **Summary**: {summary}")

            if artifacts:
                lines.append(f"- **Artifacts produced**: {', '.join(artifacts)}")
            if ready_for_review:
                lines.append(f"- **Ready for review**: {', '.join(ready_for_review)}")
            if details:
                detail_strs = [
                    d.get("description", str(d)) if isinstance(d, dict) else str(d) for d in details
                ]
                lines.append(f"- **Issues**: {'; '.join(detail_strs)}")

            lines.append("")

        lines.append(
            "Based on these results, decide what to do next:\n"
            "- If recommendation is 'proceed': continue the workflow\n"
            "- If recommendation is 'rework': delegate back for fixes\n"
            "- If recommendation is 'escalate': ask the human for guidance\n"
            "- If recommendation is 'hold': address blockers first"
        )

        return "\n".join(lines)


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
