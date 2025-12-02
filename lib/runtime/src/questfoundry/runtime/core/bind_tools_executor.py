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
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from questfoundry.runtime.structured_logging import get_tool_logger

log = logging.getLogger(__name__)

# Optional LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # LangSmith not available, use no-op decorator
    def traceable(**kwargs):
        def decorator(func):
            return func

        return decorator


# Prompt size thresholds (in characters, ~4 chars per token)
PROMPT_SIZE_ERROR_THRESHOLD = int(os.environ.get("QF_PROMPT_ERROR_THRESHOLD", "32000"))
PROMPT_SIZE_WARNING_THRESHOLD = int(os.environ.get("QF_PROMPT_WARNING_THRESHOLD", "16000"))

# Short-term memory cap (characters)
PRIOR_CONVERSATION_MAX_CHARS = int(os.environ.get("QF_MEMORY_CAP", "8000"))

# Tools that signal "role is done for this turn"
PROTOCOL_MESSAGE_TOOLS = {"send_protocol_message", "send_message"}

# Models known to NOT support bind_tools reliably (denylist approach)
# Assume bind_tools works unless the model is in this list or lacks the method
BIND_TOOLS_DENYLIST = frozenset(
    {
        "llama-3.2-1b",
        "llama-3.2-3b",
        "llama3.2:1b",
        "llama3.2:3b",
        "phi-2",
        "tinyllama",
    }
)


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
    work_summary: str = ""  # Summary of work done (for short-term memory)


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
        if any(denied in model_lower for denied in BIND_TOOLS_DENYLIST):
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
    """

    def __init__(
        self,
        llm: Any,
        tools: list[Any],
        role_id: str,
        max_iterations: int | None = None,
        trace_handler: Any | None = None,
    ):
        """Initialize bind_tools executor.

        Args:
            llm: LLM instance (will be bound to tools)
            tools: List of LangChain tool objects
            role_id: ID of the role executing (for logging)
            max_iterations: Max tool call iterations (default from env or 5)
            trace_handler: Optional callback(intent, payload) for tracing events
        """
        self.llm = llm
        self.tools = tools
        self.role_id = role_id
        self.max_iterations = max_iterations or int(os.getenv("QF_BIND_TOOLS_MAX_ITERATIONS", "5"))
        self.debug = os.getenv("QF_DEBUG", "").lower() in ("true", "1", "yes")
        self.trace_handler = trace_handler
        self.tool_map = {tool.name: tool for tool in tools}

    @traceable(name="bind_tools_executor.execute", tags=["executor", "bind_tools"])
    async def execute(
        self,
        system_prompt: str,
        user_prompt: str,
        prior_conversation: str = "",
    ) -> ExecutorResult:
        """Run the tool execution loop until protocol message is sent.

        The loop continues until:
        - A protocol message is sent (success, role done for this turn)
        - Max failures reached (error, role couldn't complete)
        - Max iterations reached (error, role took too many steps)

        Args:
            system_prompt: Role's system prompt
            user_prompt: User/task prompt for this execution
            prior_conversation: Prior conversation history for this role

        Returns:
            ExecutorResult with success status, messages, and tool results
        """
        # Bind tools to LLM
        try:
            llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception as e:
            log.error(f"Failed to bind tools to LLM: {e}")
            return ExecutorResult(
                success=False,
                error=f"Failed to bind tools: {e}",
                iterations=0,
            )

        # Check prompt size
        prompt_size = len(system_prompt) + len(user_prompt) + len(prior_conversation)
        if prompt_size > PROMPT_SIZE_ERROR_THRESHOLD:
            log.error(
                f"PROMPT SIZE ERROR for {self.role_id}: {prompt_size} chars. "
                f"Exceeds threshold ({PROMPT_SIZE_ERROR_THRESHOLD})."
            )
        elif prompt_size > PROMPT_SIZE_WARNING_THRESHOLD:
            log.warning(f"Large prompt for {self.role_id}: {prompt_size} chars.")

        # Initialize conversation with prior history (with memory cap)
        if prior_conversation:
            original_len = len(prior_conversation)
            if original_len > PRIOR_CONVERSATION_MAX_CHARS:
                truncated = prior_conversation[-PRIOR_CONVERSATION_MAX_CHARS:]
                newline_idx = truncated.find("\n")
                if newline_idx > 0 and newline_idx < 500:
                    truncated = truncated[newline_idx + 1 :]
                prior_conversation = "[... earlier conversation truncated ...]\n\n" + truncated
                log.info(
                    f"Truncated prior_conversation from {original_len} to "
                    f"{len(prior_conversation)} chars for {self.role_id}"
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prior_conversation + "\n\n---\n\n" + user_prompt),
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

        tool_results: list[dict] = []
        raw_responses: list[str] = []
        protocol_messages: list[dict] = []
        failure_count = 0
        work_done: list[str] = []

        log.info(f"BindToolsExecutor starting for role {self.role_id}")
        log.debug(f"Available tools: {list(self.tool_map.keys())}")

        # Loop until protocol message sent or max failures/iterations reached
        for iteration in range(1, self.max_iterations + 1):
            log.debug(f"Iteration {iteration}, failures: {failure_count}/3")

            # Invoke LLM with bound tools
            try:
                response = await llm_with_tools.ainvoke(messages)
            except Exception as e:
                log.error(f"LLM invocation failed: {e}")
                failure_count += 1
                if failure_count >= 3:
                    return ExecutorResult(
                        success=False,
                        error=f"LLM invocation failed: {e}",
                        iterations=iteration,
                        failure_count=failure_count,
                        tool_results=tool_results,
                        raw_responses=raw_responses,
                        work_summary=self._build_work_summary(work_done),
                    )
                continue

            # Extract response content
            response_text = self._extract_content(response)
            raw_responses.append(response_text)

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
            messages.append(response)

            # Check for tool calls
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # No tool calls in this response
                log.warning(f"No tool calls found in response for {self.role_id}")
                failure_count += 1
                continue

            log.info(f"Found {len(tool_calls)} tool call(s) in response")

            # Execute all tool calls from this response
            any_failed = False
            found_protocol_message = False

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call-{iteration}")

                log.info(f"Executing tool: {tool_name}")
                log.debug(f"Tool args: {tool_args}")

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
                    protocol_msg = {
                        "sender": self.role_id,
                        "receiver": tool_args.get("receiver", "showrunner"),
                        "intent": tool_args.get("intent", "message"),
                        "content": tool_args.get("content", ""),
                        "payload": tool_args.get("payload", {}),
                    }
                    protocol_messages.append(protocol_msg)
                    log.info(f"Protocol message to {protocol_msg['receiver']} collected")

                # Add tool result to messages
                messages.append(
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
                    work_summary=self._build_work_summary(work_done),
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
            work_summary=self._build_work_summary(work_done),
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

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool and return (observation, success).

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tuple of (observation string, success boolean)
        """
        tool = self.tool_map.get(tool_name)
        if tool is None:
            available = list(self.tool_map.keys())
            error_msg = f"Error: Unknown tool '{tool_name}'. Available tools: {available}"
            return error_msg, False

        try:
            # Invoke tool with provided args
            result = tool.invoke(tool_args)

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

    def _build_work_summary(self, work_done: list[str]) -> str:
        """Build a summary of work done for short-term memory.

        Args:
            work_done: List of actions taken

        Returns:
            Summary string, capped at PRIOR_CONVERSATION_MAX_CHARS
        """
        if not work_done:
            return ""
        summary = f"Actions taken: {', '.join(work_done)}"
        if len(summary) > PRIOR_CONVERSATION_MAX_CHARS:
            summary = summary[:PRIOR_CONVERSATION_MAX_CHARS] + "..."
        return summary


def select_executor(model_name: str) -> type:
    """Select executor based on model's bind_tools support.

    Models known to support bind_tools reliably get BindToolsExecutor.
    Models that don't support it or are too small get ProtocolExecutor (text-based fallback).

    Args:
        model_name: Name of the model (e.g., "qwen3:8b", "llama-3.2:1b", "gpt-4")

    Returns:
        Executor class to use (BindToolsExecutor or ProtocolExecutor)
    """
    # Import here to avoid circular imports
    from questfoundry.runtime.core.protocol_executor import ProtocolExecutor

    # Models known to support bind_tools well
    BIND_TOOLS_MODELS = {
        "qwen3",
        "qwen2.5",
        "qwen2",
        "gpt-4",
        "gpt-3.5",
        "claude-3",
        "claude-2",
        "llama-3.1",
        "llama3.1",
        "gemini",
    }

    model_lower = model_name.lower()

    # Check if any supported model pattern matches
    if any(m in model_lower for m in BIND_TOOLS_MODELS):
        return BindToolsExecutor

    # Check denylist for models known to NOT work
    if any(m in model_lower for m in BIND_TOOLS_DENYLIST):
        return ProtocolExecutor

    # Default to text-based for unknown models (safer)
    return ProtocolExecutor
