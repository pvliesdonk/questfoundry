"""Tool executor - reusable tool execution loop for LLM agents.

Adapted from the old BindToolsExecutor pattern. Provides a clean loop that:
- Binds tools to an LLM
- Executes tool calls from LLM responses
- Handles validation errors with LLM-friendly nudges
- Tracks consult_* tool usage for policy guards
- Terminates when a specified "done" tool is called

This executor is used by both SR (orchestrator) and specialist roles.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Tools that consult compiled resources for guidance.
# Successful calls are tracked so policy guards can verify prerequisites.
CONSULT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "consult_playbook",
        "consult_role_charter",
        "consult_schema",
        "consult_glossary",
    }
)


@dataclass
class ExecutorResult:
    """Result from tool executor run."""

    success: bool
    """Whether execution completed successfully (done tool called)."""

    done_tool_result: dict[str, Any] | None = None
    """Result from the 'done' tool that terminated execution."""

    tool_results: list[dict[str, Any]] = field(default_factory=list)
    """All tool execution results from this run."""

    iterations: int = 0
    """Number of LLM invocations performed."""

    failure_count: int = 0
    """Number of consecutive failures (reset on success)."""

    error: str | None = None
    """Error message if execution failed."""

    consults: list[dict[str, Any]] = field(default_factory=list)
    """Successful consult_* tool calls for policy tracking."""


def _format_validation_error(
    error_msg: str,
    tool_name: str,
    missing_fields: list[str] | None = None,
    invalid_fields: list[dict[str, str]] | None = None,
) -> str:
    """Format validation errors with LLM-friendly hints.

    Nudges the LLM to use consult_schema to understand requirements.
    """
    parts = [
        f"Tool '{tool_name}' validation failed: {error_msg}",
    ]

    if missing_fields:
        parts.append(f"Missing required fields: {', '.join(missing_fields)}")

    if invalid_fields:
        for f in invalid_fields:
            parts.append(f"  - {f['field']}: {f['issue']}")

    parts.append("")
    parts.append("Hint: Use consult_schema tool to check field types and allowed values.")

    return "\n".join(parts)


class ToolExecutor:
    """Execute LLM tool-calling loop until a 'done' tool is called.

    This executor handles:
    - Binding tools to LLM
    - Invoking LLM and processing tool calls
    - Executing tools and appending ToolMessages
    - Nudging LLM when it doesn't make tool calls
    - Tracking consult_* usage for policy guards
    - Terminating when any specified stop_tool is called

    Parameters
    ----------
    llm : BaseChatModel
        LangChain-compatible LLM with tool support.
    tools : list[BaseTool]
        Tools available to the agent.
    done_tool_name : str
        Name of the tool that signals completion (e.g., "return_to_sr", "terminate").
    system_prompt : str
        System prompt for the agent.
    max_iterations : int
        Maximum LLM invocations before giving up.
    max_failures : int
        Maximum consecutive failures before giving up.
    stop_tools : list[str] | None
        Additional tools that stop execution (for orchestrator intercepts).
        These are checked for successful calls and will stop the loop.

    Examples
    --------
    Execute a role until it calls return_to_sr::

        executor = ToolExecutor(
            llm=ollama_llm,
            tools=[write_hot_sot, read_hot_sot, return_to_sr],
            done_tool_name="return_to_sr",
            system_prompt="You are the Plotwright...",
        )
        result = await executor.run("Design a story topology")
        if result.success:
            delegation_result = result.done_tool_result
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        done_tool_name: str,
        system_prompt: str,
        max_iterations: int = 50,
        max_failures: int = 3,
        on_tool_call: Callable[[str, dict[str, Any], Any], None] | None = None,
        stop_tools: list[str] | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.done_tool_name = done_tool_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.on_tool_call = on_tool_call

        # Tools that stop execution when called successfully
        # Includes done_tool plus any additional stop tools
        self.stop_tool_names: set[str] = {done_tool_name}
        if stop_tools:
            self.stop_tool_names.update(stop_tools)

        # Build tool lookup
        self.tool_map: dict[str, BaseTool] = {tool.name: tool for tool in tools}

        # Bind tools to LLM
        self.llm_with_tools = llm.bind_tools(tools)

        # Conversation history (maintained across run() calls for multi-turn)
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def reset(self) -> None:
        """Reset conversation history for a fresh run."""
        self.messages = [SystemMessage(content=self.system_prompt)]

    async def run(self, user_prompt: str) -> ExecutorResult:
        """Run the tool execution loop until done tool is called.

        Parameters
        ----------
        user_prompt : str
            The task/prompt for this execution turn.

        Returns
        -------
        ExecutorResult
            Result containing success status, done tool result, and metadata.
        """
        # Add user message
        self.messages.append(HumanMessage(content=user_prompt))

        tool_results: list[dict[str, Any]] = []
        consults: list[dict[str, Any]] = []
        failure_count = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Iteration {iteration}, failures: {failure_count}")

            # Invoke LLM
            try:
                response = await self.llm_with_tools.ainvoke(self.messages)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                failure_count += 1
                if failure_count >= self.max_failures:
                    return ExecutorResult(
                        success=False,
                        error=f"LLM invocation failed after {failure_count} attempts: {e}",
                        tool_results=tool_results,
                        iterations=iteration,
                        failure_count=failure_count,
                        consults=consults,
                    )
                continue

            # Add AI response to history
            self.messages.append(response)

            # Extract tool calls
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # No tool calls - nudge LLM to use tools
                logger.warning(f"No tool calls in iteration {iteration}")
                failure_count += 1

                guidance = HumanMessage(
                    content=(
                        "You must use tools to do your work and communicate results. "
                        "Your response contained no tool calls. Please either:\n"
                        f"1. Call {self.done_tool_name} to report your work is complete, OR\n"
                        "2. Call another tool to continue your work, OR\n"
                        f"3. Call {self.done_tool_name} with status='error' if you cannot proceed.\n\n"
                        "Do NOT respond with plain text - you MUST make a tool call."
                    )
                )
                self.messages.append(guidance)

                if failure_count >= self.max_failures:
                    return ExecutorResult(
                        success=False,
                        error=f"No tool calls after {failure_count} consecutive attempts",
                        tool_results=tool_results,
                        iterations=iteration,
                        failure_count=failure_count,
                        consults=consults,
                    )
                continue

            # Process each tool call
            found_done = False
            done_result: dict[str, Any] | None = None
            any_failed = False

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call-{iteration}")

                logger.info(f"Executing tool: {tool_name}")
                logger.debug(f"Tool args: {tool_args}")

                # Execute tool
                observation, success = await self._execute_tool(tool_name, tool_args)

                # Record result
                result_entry = {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": observation,
                    "success": success,
                    "iteration": iteration,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                tool_results.append(result_entry)

                # Callback for logging/tracing
                if self.on_tool_call:
                    try:
                        self.on_tool_call(tool_name, tool_args, observation)
                    except Exception as e:
                        logger.warning(f"on_tool_call callback failed: {e}")

                # Track consult_* usage
                if success and tool_name in CONSULT_TOOL_NAMES:
                    consults.append(
                        {
                            "tool": tool_name,
                            "args": tool_args,
                            "iteration": iteration,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                # Add ToolMessage (required by OpenAI API contract)
                self.messages.append(ToolMessage(content=observation, tool_call_id=tool_id))

                if not success:
                    any_failed = True
                    continue

                # Check if this was a stop tool (done tool or other stop tools)
                if tool_name in self.stop_tool_names:
                    found_done = True
                    # Parse the observation as the done result
                    try:
                        parsed = json.loads(observation)
                        # Only treat as done if the tool succeeded
                        if parsed.get("success", True):
                            done_result = parsed
                            done_result["_stop_tool"] = tool_name
                    except json.JSONDecodeError:
                        done_result = {"raw": observation, "_stop_tool": tool_name}

            # Update failure count
            if any_failed:
                failure_count += 1
            else:
                failure_count = 0

            # Check termination conditions
            if failure_count >= self.max_failures:
                return ExecutorResult(
                    success=False,
                    error=f"Max failures ({self.max_failures}) reached",
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=failure_count,
                    consults=consults,
                )

            if found_done:
                logger.info(f"Done tool '{self.done_tool_name}' called, execution complete")
                return ExecutorResult(
                    success=True,
                    done_tool_result=done_result,
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=0,
                    consults=consults,
                )

        # Max iterations reached
        return ExecutorResult(
            success=False,
            error=f"Max iterations ({self.max_iterations}) reached without calling {self.done_tool_name}",
            tool_results=tool_results,
            iterations=self.max_iterations,
            failure_count=failure_count,
            consults=consults,
        )

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool and return (observation, success).

        Returns
        -------
        tuple[str, bool]
            The observation string and whether execution succeeded.
        """
        tool = self.tool_map.get(tool_name)

        if tool is None:
            available = ", ".join(sorted(self.tool_map.keys()))
            error_msg = f"Unknown tool '{tool_name}'. Available tools: {available}"
            logger.warning(error_msg)
            return json.dumps({"success": False, "error": error_msg}), False

        try:
            # Execute tool (handle both sync and async)
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_args)
            elif hasattr(tool, "_arun"):
                result = await tool._arun(**tool_args)
            elif hasattr(tool, "_run"):
                result = tool._run(**tool_args)
            else:
                result = tool.invoke(tool_args)

            # Format result as string
            if isinstance(result, dict):
                observation = json.dumps(result, default=str)
                success = result.get("success", True)
            elif isinstance(result, str):
                observation = result
                success = True
            else:
                observation = str(result)
                success = True

            return observation, success

        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "hint": "Check tool arguments and try again. Use consult_schema for field requirements.",
            }
            return json.dumps(error_result), False
