"""Executor using native bind_tools for structured tool calling.

This is the PRIMARY executor for models with native tool support (GPT-4, Claude 3,
Qwen 3, Llama 3.1+, etc.). It uses bind_tools for structured tool calls instead of
text parsing, resulting in more reliable and efficient tool execution.

**Primary approach (bind_tools, this module):**
- Use BindToolsExecutor for models with native tool support
- Structured tool calls via bind_tools (GPT-4, Claude 3, Qwen, Llama 3.1+)
- Higher reliability than text-based parsing

**Fallback approach (text-based):**
- Use ProtocolExecutor for models without bind_tools support
- Explicit Action/Action Input text format for universal compatibility

CRITICAL: This is NOT a ReAct agent loop. Roles communicate via protocol messages.
When send_protocol_message is called, the role is DONE for this turn.
See lib/runtime/AGENTS.md for the execution model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.runtime.config import get_settings
from questfoundry.runtime.core.reasoning_extractor import ReasoningExtractor
from questfoundry.runtime.structured_logging import get_reasoning_logger, get_tool_logger

log = logging.getLogger(__name__)


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Serialize LangChain messages to JSON-serializable dicts.

    Args:
        messages: List of LangChain message objects

    Returns:
        List of dicts with type and content
    """
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"type": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data = {"type": "ai", "content": msg.content}
            # Preserve tool_calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                data["tool_calls"] = msg.tool_calls
            result.append(data)
        elif isinstance(msg, ToolMessage):
            result.append({
                "type": "tool",
                "content": msg.content,
                "tool_call_id": getattr(msg, "tool_call_id", ""),
            })
        else:
            # Fallback for unknown types
            result.append({"type": "unknown", "content": str(msg.content)})
    return result


def deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    """Deserialize dicts back to LangChain message objects.

    Args:
        data: List of dicts from serialize_messages

    Returns:
        List of LangChain message objects
    """
    messages: list[BaseMessage] = []
    for item in data:
        msg_type = item.get("type", "unknown")
        content = item.get("content", "")

        if msg_type == "system":
            messages.append(SystemMessage(content=content))
        elif msg_type == "human":
            messages.append(HumanMessage(content=content))
        elif msg_type == "ai":
            msg = AIMessage(content=content)
            if "tool_calls" in item:
                msg.tool_calls = item["tool_calls"]
            messages.append(msg)
        elif msg_type == "tool":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=item.get("tool_call_id", ""),
            ))
        else:
            # Skip unknown types
            log.warning(f"Unknown message type during deserialization: {msg_type}")
    return messages


def _get_prompt_log():
    """Lazy getter for prompt logger (configured at runtime by CLI)."""
    try:
        from questfoundry.runtime.structured_logging import get_prompt_logger, is_configured

        if is_configured():
            return get_prompt_logger()
    except ImportError:
        pass
    return None


def _get_memory_config() -> tuple[int, int, int]:
    """Get memory settings from centralized config.

    Returns:
        Tuple of (error_threshold, warning_threshold, memory_cap)

    Note: Summarization thresholds are now calculated dynamically per model
          using calculate_model_aware_thresholds() based on context window size.
    """
    settings = get_settings()
    return (
        settings.memory.prompt_error_threshold,
        settings.memory.prompt_warning_threshold,
        settings.memory.memory_cap,
    )


def _get_bind_tools_denylist() -> frozenset[str]:
    """Get bind_tools denylist from centralized config.

    Returns:
        Frozenset of model name patterns that don't support bind_tools reliably
    """
    settings = get_settings()
    return frozenset(settings.llm.bind_tools_denylist)


# Tools that signal "role is done for this turn"
PROTOCOL_MESSAGE_TOOLS = {"send_protocol_message", "send_message"}


@dataclass
class ExecutorResult:
    """Result from bind_tools executor."""

    success: bool
    messages: list[dict] = field(default_factory=list)  # Protocol messages sent
    tool_results: list[dict] = field(default_factory=list)
    iterations: int = 0
    failure_count: int = 0  # Counts only failures, not valid tool calls
    error: str | None = None
    raw_responses: list[str] = field(default_factory=list)
    # Serialized conversation history for role continuity (proper message objects)
    conversation_history: list[dict] = field(default_factory=list)
    # Legacy: text summary for backwards compatibility
    work_summary: str = ""


def supports_bind_tools(llm: Any, model_name: str | None = None) -> bool:
    """Check if LLM supports native bind_tools.

    Uses a denylist approach: assume bind_tools works unless:
    1. The LLM lacks the bind_tools method
    2. The model is in the known denylist (small models with issues)

    Args:
        llm: The LLM instance to check
        model_name: Optional model name for denylist check

    Returns:
        True if LLM supports bind_tools, False otherwise
    """
    # Check if LLM has bind_tools method
    if not hasattr(llm, "bind_tools"):
        return False

    # Check denylist if model name provided
    if model_name:
        model_lower = model_name.lower()
        denylist = _get_bind_tools_denylist()
        if any(denied in model_lower for denied in denylist):
            return False

    return True


class BindToolsExecutor:
    """Execute role using native bind_tools for structured tool calling.

    This executor uses LLM bind_tools() for structured tool calls instead of
    text parsing. This provides:
    - Higher reliability than text-based Action/Action Input parsing
    - Structured tool call objects (name, args, id)
    - Support for multiple parallel tool calls in one response
    - Native support across major LLM providers

    CRITICAL: This is NOT a ReAct loop with "Final Answer" termination.
    - Role is done when it calls send_protocol_message
    - Only FAILURES count against max_failures (not valid tool calls)
    - Messages from send_protocol_message ARE the output

    The executor maintains conversation state between invocations, allowing
    the same role to continue its conversation across multiple turns.
    """

    def __init__(
        self,
        llm: Any,
        tools: list[Any],
        role_id: str,
        system_prompt: str,
        state: Any = None,
        max_iterations: int | None = None,
        trace_handler: Any | None = None,
        model_name: str | None = None,
        provider_manager: Any | None = None,
    ):
        """Initialize bind_tools executor.

        Args:
            llm: LLM instance (will be bound to tools)
            tools: List of LangChain tool objects
            role_id: ID of the role executing (for logging)
            system_prompt: Role's system prompt (set once, used for all turns)
            state: Current StudioState for tool execution context (injected into tools)
            max_iterations: Max tool call iterations (default from config)
            trace_handler: Optional callback(intent, payload) for tracing events
            model_name: Model name for calculating context-aware thresholds
            provider_manager: Provider manager for accessing model limits
        """
        settings = get_settings()
        self.llm = llm
        self.tools = tools
        self.role_id = role_id
        self.system_prompt = system_prompt
        self.state = state
        self.max_iterations = max_iterations or settings.runtime.max_iterations
        self.debug = settings.runtime.debug
        self.trace_handler = trace_handler
        self.tool_map = {tool.name: tool for tool in tools}
        self.model_name = model_name
        self.provider_manager = provider_manager

        # Calculate model-aware thresholds
        if model_name and provider_manager:
            from questfoundry.runtime.config import calculate_model_aware_thresholds

            self.message_threshold, self.char_threshold = (
                calculate_model_aware_thresholds(model_name, provider_manager)
            )
        else:
            # Fallback to legacy defaults if model info not provided
            log.warning(
                f"No model info provided for {role_id}, using legacy default thresholds"
            )
            # Fallback to conservative defaults if model info not provided
            self.message_threshold = 50  # Conservative default
            self.char_threshold = 50000  # ~12K tokens

        # Conversation state - maintained across invocations
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        # Bind tools once at init
        self.llm_with_tools = llm.bind_tools(tools)

    def update_state(self, state: Any) -> None:
        """Update the state reference (called before each execution).

        Args:
            state: New state dict to use for tool execution
        """
        self.state = state

    async def execute(
        self,
        user_prompt: str,
    ) -> ExecutorResult:
        """Run the tool execution loop until protocol message is sent.

        The executor maintains conversation state between calls. Each call adds
        the user_prompt to the existing conversation and continues.

        The loop continues until:
        - A protocol message is sent (success, role done for this turn)
        - Max failures reached (error, role couldn't complete)
        - Max iterations reached (error, role took too many steps)

        Args:
            user_prompt: User/task prompt for this turn

        Returns:
            ExecutorResult with success status, messages, and tool results
        """
        # Add new user message to conversation
        self.messages.append(HumanMessage(content=user_prompt))

        log.info(
            f"BindToolsExecutor continuing for {self.role_id}, "
            f"conversation has {len(self.messages)} messages"
        )

        # Check prompt size and summarize if needed (using centralized config)
        error_thresh, warn_thresh, _ = _get_memory_config()
        prompt_size = sum(len(str(m.content)) for m in self.messages)
        message_count = len(self.messages)

        if prompt_size > error_thresh:
            log.error(
                f"PROMPT SIZE ERROR for {self.role_id}: {prompt_size} chars. "
                f"Exceeds threshold ({error_thresh})."
            )
        elif prompt_size > warn_thresh:
            log.warning(f"Large prompt for {self.role_id}: {prompt_size} chars.")

        # Two-stage hybrid context management
        model_info = f" ({self.model_name})" if self.model_name else ""

        # Stage 1: Clear old tool results at 60% threshold (lossless for reasoning)
        stage1_char_threshold = int(self.char_threshold * 0.60)
        if prompt_size > stage1_char_threshold:
            log.info(
                f"Context at Stage 1 threshold for {self.role_id}{model_info}: "
                f"{prompt_size} chars (>{stage1_char_threshold}, 60% of limit). "
                f"Clearing old tool results..."
            )
            cleared_count = self._clear_old_tool_results()

            if cleared_count > 0:
                # Recompute after clearing
                prompt_size = sum(len(str(m.content)) for m in self.messages)
                message_count = len(self.messages)
                log.info(
                    f"After tool result cleanup: {message_count} msgs, {prompt_size} chars"
                )

        # Stage 2: Full summarization at 100% threshold if needed (lossy)
        if message_count > self.message_threshold or prompt_size > self.char_threshold:
            log.info(
                f"Context at Stage 2 threshold for {self.role_id}{model_info}: "
                f"{message_count} msgs (>{self.message_threshold}), "
                f"{prompt_size} chars (>{self.char_threshold}). "
                f"Full summarization needed..."
            )
            await self._summarize_history()

            # Recompute after summarization
            prompt_size = sum(len(str(m.content)) for m in self.messages)
            message_count = len(self.messages)
            log.info(
                f"After summarization: {message_count} msgs, {prompt_size} chars"
            )

        tool_results: list[dict] = []
        raw_responses: list[str] = []
        protocol_messages: list[dict] = []
        failure_count = 0
        work_done: list[str] = []

        log.debug(f"Available tools: {list(self.tool_map.keys())}")

        # Loop until protocol message sent or max failures/iterations reached
        for iteration in range(1, self.max_iterations + 1):
            log.debug(f"[{self.role_id}] Iteration {iteration}, failures: {failure_count}/3")

            # Invoke LLM with bound tools
            try:
                response = await self.llm_with_tools.ainvoke(self.messages)
            except Exception as e:
                log.error(f"LLM invocation failed: {e}")
                failure_count += 1
                if failure_count >= 3:
                    # Ask user if they want to retry
                    import questionary
                    from rich.console import Console
                    from rich.panel import Panel

                    console = Console()
                    console.print(
                        Panel(
                            f"[bold red]LLM Connection Failed[/bold red]\n\n"
                            f"Role: {self.role_id}\n"
                            f"Error: {e}\n"
                            f"Attempts: {failure_count}",
                            title="⚠️ Connection Error",
                            border_style="red",
                        )
                    )

                    retry = questionary.confirm(
                        "Retry connection?",
                        default=True,
                    ).ask()

                    if retry:
                        log.info("User requested retry after LLM connection failure")
                        failure_count = 0
                        continue

                    return ExecutorResult(
                        success=False,
                        error=f"LLM invocation failed: {e}",
                        iterations=iteration,
                        failure_count=failure_count,
                        tool_results=tool_results,
                        raw_responses=raw_responses,
                    )
                continue

            # Extract response content
            response_text = self._extract_content(response)
            raw_responses.append(response_text)

            # Log prompt and response to structured logging (full content)
            prompt_log = _get_prompt_log()
            if prompt_log:
                tool_calls = getattr(response, "tool_calls", []) or []
                prompt_log.info(
                    "llm_call",
                    role=self.role_id,
                    iteration=iteration,
                    message_count=len(self.messages),
                    response_chars=len(response_text),
                    tool_calls_count=len(tool_calls),
                    # Full content for debugging
                    messages=serialize_messages(self.messages),
                    response=response_text,
                    tool_calls=[
                        {"name": tc.get("name"), "args": tc.get("args")}
                        for tc in tool_calls
                    ] if tool_calls else [],
                )

            # Trace for debugging
            if self.trace_handler:
                try:
                    self.trace_handler(
                        "llm_iteration",
                        {
                            "role_id": self.role_id,
                            "iteration": iteration,
                            "failure_count": failure_count,
                            "response": response_text,
                        },
                    )
                except Exception as e:
                    log.warning(f"Trace callback failed: {e}")

            if self.debug:
                log.info(f"LLM Response (iter {iteration}):\n{response_text[:500]}...")

            # Add AI message to conversation
            self.messages.append(response)

            # Check for tool calls
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # No tool calls in this response - prompt LLM to use tools or explain
                log.warning(f"No tool calls found in response for {self.role_id}")
                failure_count += 1

                # Add guidance message for retry
                guidance = HumanMessage(
                    content=(
                        "You must use tools to communicate. Your response contained no tool calls. "
                        "Please either:\n"
                        "1. Call send_protocol_message to send your output to another role, OR\n"
                        "2. Call another tool to do your work, OR\n"
                        "3. If you cannot proceed, call send_protocol_message with intent='error' "
                        "explaining why.\n\n"
                        "Do NOT respond with plain text - you MUST make a tool call."
                    )
                )
                self.messages.append(guidance)
                continue

            log.info(f"[{self.role_id}] Found {len(tool_calls)} tool call(s) in response")

            # Extract and log reasoning if enabled
            settings = get_settings()
            if settings.logging.reasoning_enabled:
                try:
                    extractor = ReasoningExtractor()
                    reasoning = extractor.extract_reasoning(
                        message_content=response_text,
                        tool_calls=tool_calls
                    )
                    if reasoning:
                        reasoning_log = get_reasoning_logger()
                        context = {
                            "tu_id": self.state.get("loop_context", {}).get("tu_id"),
                            "loop_id": self.state.get("loop_context", {}).get("loop_id"),
                            "iteration": iteration,
                        }
                        log_entry = extractor.format_for_logging(
                            reasoning,
                            role_id=self.role_id,
                            context=context
                        )
                        reasoning_log.info("reasoning", **log_entry)
                        log.debug(f"[{self.role_id}] Captured reasoning: {reasoning['reasoning_type']}")
                except Exception as e:
                    log.warning(f"[{self.role_id}] Reasoning extraction failed: {e}")

            # Execute all tool calls from this response
            any_failed = False
            found_protocol_message = False

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call-{iteration}")

                log.info(f"[{self.role_id}] Executing tool: {tool_name}")
                log.debug(f"[{self.role_id}] Tool args: {tool_args}")

                # Get structured logger
                try:
                    tool_log = get_tool_logger()
                except RuntimeError:
                    # Logging not configured, use standard logger
                    tool_log = None

                # Execute tool
                observation, tool_success = await self._execute_tool(tool_name, tool_args)

                # Log tool execution
                if tool_log:
                    try:
                        tool_log.info(
                            "tool.invoke",
                            tool=tool_name,
                            args=tool_args,
                            success=tool_success,
                            result=observation,
                        )
                    except Exception as e:
                        log.warning(f"Failed to log tool execution: {e}")

                tool_results.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": observation,
                        "success": tool_success,
                        "iteration": iteration,
                    }
                )

                if not tool_success:
                    any_failed = True
                    continue

                work_done.append(f"Called {tool_name}")

                # Check if this was a protocol message
                if tool_name in PROTOCOL_MESSAGE_TOOLS:
                    found_protocol_message = True
                    # Build protocol message from tool args
                    # CRITICAL: receiver is REQUIRED - no default to avoid loops
                    receiver = tool_args.get("receiver")
                    if not receiver:
                        log.error(
                            f"Protocol message from {self.role_id} missing receiver. "
                            "This is a tool call bug - receiver is required."
                        )
                        # Use __terminate__ as safe fallback to avoid infinite loops
                        receiver = "__terminate__"

                    # SAFETY: Prevent roles from sending messages to themselves
                    # This creates infinite loops. Convert to termination signal.
                    if receiver == self.role_id:
                        log.warning(
                            f"Role {self.role_id} attempted to send message to itself. "
                            "Converting to termination to prevent infinite loop."
                        )
                        receiver = "__terminate__"

                    protocol_msg = {
                        "sender": self.role_id,
                        "receiver": receiver,
                        "intent": tool_args.get("intent", "message"),
                        "content": tool_args.get("content", ""),
                        "payload": tool_args.get("payload", {}),
                    }
                    protocol_messages.append(protocol_msg)
                    log.info(f"Protocol message to {protocol_msg['receiver']} collected")

                # Add tool result to messages
                self.messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_id,
                    )
                )

            # After processing all actions, decide what to do
            if any_failed:
                failure_count += 1
            else:
                failure_count = 0

            # Check if max failures reached
            if failure_count >= 3:
                log.error(
                    f"Max failures (3) reached for {self.role_id} after {failure_count} consecutive failures"
                )
                return ExecutorResult(
                    success=False,
                    error=f"Max failures (3) reached after {failure_count} consecutive tool failures",
                    messages=protocol_messages,
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=failure_count,
                    raw_responses=raw_responses,
                )

            # If we found protocol messages, role is done for this turn
            if found_protocol_message:
                log.info(
                    f"Protocol message(s) sent by {self.role_id}, "
                    f"turn complete ({len(protocol_messages)} message(s))"
                )
                return ExecutorResult(
                    success=True,
                    messages=protocol_messages,
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=0,
                    raw_responses=raw_responses,
                )

        # Max iterations reached
        log.error(f"Max iterations ({self.max_iterations}) reached for {self.role_id}")
        return ExecutorResult(
            success=False,
            error=f"Reached maximum iterations ({self.max_iterations})",
            messages=protocol_messages,
            tool_results=tool_results,
            iterations=self.max_iterations,
            failure_count=failure_count,
            raw_responses=raw_responses,
        )

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response.

        Args:
            response: Response from LLM invoke

        Returns:
            Content string, or empty string if no content
        """
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return "".join(str(part) for part in content)
            return str(content) if content else ""
        return str(response)

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool asynchronously and return (observation, success).

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments from LLM (state/role_id injected automatically)

        Returns:
            Tuple of (observation string, success boolean)
        """
        import inspect

        tool = self.tool_map.get(tool_name)
        if tool is None:
            available = list(self.tool_map.keys())
            error_msg = f"Error: Unknown tool '{tool_name}'. Available tools: {available}"
            return error_msg, False

        try:
            # Inject state and role_id for InjectedToolArg parameters
            # Use combined_args to avoid shadowing 'payload' key from tool_args
            combined_args = {**tool_args, "state": self.state, "role_id": self.role_id}

            # Call _run directly with filtered params (like ProtocolExecutor)
            # This avoids Pydantic validation issues with InjectedToolArg
            if hasattr(tool, "_run"):
                sig = inspect.signature(tool._run)
                valid_params = {k: v for k, v in combined_args.items() if k in sig.parameters}
                result = tool._run(**valid_params)
            elif hasattr(tool, "invoke"):
                result = tool.invoke(**tool_args)  # Use original args without state/role_id
            else:
                result = tool(**tool_args)

            # Apply state updates from tools that return state changes
            # This ensures subsequent tool calls within the same turn see updated state
            if isinstance(result, dict):
                if "hot_sot" in result and isinstance(result["hot_sot"], dict):
                    self.state["hot_sot"] = result["hot_sot"]
                    log.debug(f"[{self.role_id}] Applied hot_sot update from {tool_name}")
                if "cold_sot" in result and isinstance(result["cold_sot"], dict):
                    self.state["cold_sot"] = result["cold_sot"]
                    log.debug(f"[{self.role_id}] Applied cold_sot update from {tool_name}")

            # Convert result to JSON string
            if isinstance(result, str):
                try:
                    json.loads(result)
                    return result, True
                except json.JSONDecodeError:
                    return json.dumps({"result": result}, indent=2, default=str), True
            return json.dumps(result, indent=2, default=str), True

        except Exception as e:
            log.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return json.dumps({"error": str(e), "tool": tool_name}, indent=2), False

    def _build_work_summary(
        self, work_done: list[str], raw_responses: list[str] | None = None
    ) -> str:
        """Build a conversation summary for role continuity.

        Args:
            work_done: List of actions taken (tool names)
            raw_responses: List of LLM response texts (optional)

        Returns:
            Summary string including actions AND LLM reasoning, capped at max chars
        """
        _, _, memory_cap = _get_memory_config()
        parts = []

        # Include actions taken
        if work_done:
            parts.append(f"Actions taken: {', '.join(work_done)}")

        # Include LLM responses (the actual reasoning/decisions)
        if raw_responses:
            # Use the last response as it typically contains final reasoning
            last_response = raw_responses[-1] if raw_responses else ""
            if last_response:
                # Truncate if too long, keep the important end
                max_response_chars = memory_cap // 2
                if len(last_response) > max_response_chars:
                    last_response = "..." + last_response[-max_response_chars:]
                parts.append(f"Last response:\n{last_response}")

        if not parts:
            return ""

        summary = "\n\n".join(parts)
        if len(summary) > memory_cap:
            summary = summary[:memory_cap] + "..."
        return summary

    def _clear_old_tool_results(self) -> int:
        """Clear old tool results to reduce context size (Stage 1 cleanup).

        Replaces verbose ToolMessage content with [result cleared] placeholders.
        Preserves all AIMessage reasoning text for ReasoningExtractor.

        This is a lossless operation for reasoning extraction, only removes
        verbose tool results. More aggressive than full summarization.

        Returns:
            Number of tool results cleared
        """
        # Keep the most recent messages (last 6 for context continuity)
        keep_recent = 6
        if len(self.messages) <= keep_recent:
            # Not enough messages to clear
            return 0

        # Find ToolMessage objects in older messages (exclude last 6)
        cleared_count = 0
        for i, msg in enumerate(self.messages[:-keep_recent]):
            if isinstance(msg, ToolMessage):
                # Replace verbose content with placeholder
                original_size = len(str(msg.content))
                msg.content = "[result cleared]"
                cleared_count += 1
                log.debug(
                    f"Cleared tool result at index {i}, "
                    f"reduced from {original_size} to {len(msg.content)} chars"
                )

        if cleared_count > 0:
            # Recompute prompt size
            new_size = sum(len(str(m.content)) for m in self.messages)
            log.info(
                f"Cleared {cleared_count} old tool results for {self.role_id}, "
                f"new context size: {new_size} chars"
            )

        return cleared_count

    async def _summarize_history(self) -> None:
        """Summarize older conversation history to reduce context size.

        Uses the LLM to create a concise summary of older messages,
        keeping the system message and most recent messages intact.

        This is called automatically when context exceeds thresholds.
        """
        # Separate system messages from conversation
        system_msgs = [m for m in self.messages if isinstance(m, SystemMessage)]
        non_system = [m for m in self.messages if not isinstance(m, SystemMessage)]

        # Keep the most recent messages (last 6 for context continuity)
        keep_recent = 6
        if len(non_system) <= keep_recent:
            # Not enough to summarize
            return

        older_msgs = non_system[:-keep_recent]
        recent_msgs = non_system[-keep_recent:]

        # Build text representation of older messages for summarization
        older_text_parts = []
        for msg in older_msgs:
            if isinstance(msg, HumanMessage):
                older_text_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = self._extract_content(msg)
                # Include tool call info if present
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                    tools_str = ", ".join(tool_names)
                    older_text_parts.append(f"Assistant: {content}\n[Called tools: {tools_str}]")
                else:
                    older_text_parts.append(f"Assistant: {content}")
            elif isinstance(msg, ToolMessage):
                # Summarize tool results briefly
                content = str(msg.content)[:200]  # Truncate long tool outputs
                older_text_parts.append(f"Tool result: {content}...")

        older_text = "\n\n".join(older_text_parts)

        # Create summarization prompt
        summary_prompt = (
            f"Summarize the following conversation history concisely. "
            f"Focus on: decisions made, tools used, key information discovered, "
            f"and any ongoing tasks. Be brief but preserve essential context.\n\n"
            f"Conversation to summarize:\n{older_text}\n\n"
            f"Summary (2-4 sentences):"
        )

        try:
            # Use the base LLM (without tools) for summarization
            response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
            summary_text = self._extract_content(response)

            log.info(
                f"Summarized {len(older_msgs)} messages into {len(summary_text)} chars "
                f"for {self.role_id}"
            )

            # Log summarization event (full content)
            prompt_log = _get_prompt_log()
            if prompt_log:
                prompt_log.info(
                    "context_summarized",
                    role=self.role_id,
                    messages_summarized=len(older_msgs),
                    summary_chars=len(summary_text),
                    # Full content for debugging
                    original_messages=serialize_messages(older_msgs),
                    summary=summary_text,
                )

            # Replace older messages with summary
            summary_msg = HumanMessage(
                content=f"[Previous conversation summary: {summary_text}]"
            )
            self.messages = system_msgs + [summary_msg] + recent_msgs

        except Exception as e:
            log.warning(f"Summarization failed for {self.role_id}: {e}, falling back to truncation")
            # Fallback: just keep recent messages without summary
            self.messages = system_msgs + recent_msgs
