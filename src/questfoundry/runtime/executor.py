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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ExecutorCallbacks(Protocol):
    """Protocol for executor event callbacks.

    Implement these methods to receive progress updates during execution.
    All methods are optional - unimplemented methods are skipped.
    """

    def on_llm_start(self, iteration: int) -> None:
        """Called when LLM inference begins."""
        ...

    def on_llm_end(self, iteration: int, has_tool_calls: bool) -> None:
        """Called when LLM inference completes."""
        ...

    def on_llm_token(self, token: str) -> None:
        """Called for each token during streaming."""
        ...

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        """Called before tool execution."""
        ...

    def on_tool_end(self, tool_name: str, result: str, success: bool) -> None:
        """Called after tool execution."""
        ...

    def on_error(self, error: str) -> None:
        """Called when an error occurs."""
        ...

    def on_done(self, tool_name: str, result: dict[str, Any]) -> None:
        """Called when execution completes via done tool."""
        ...


# Tools that consult compiled resources for guidance.
# Successful calls are tracked so policy guards can verify prerequisites.
CONSULT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "consult_playbook",
        "consult_role_charter",
        "consult_schema",
        "consult_glossary",
        "consult_tool",
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
        callbacks: ExecutorCallbacks | None = None,
        stream: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.done_tool_name = done_tool_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.on_tool_call = on_tool_call
        self.callbacks = callbacks
        self.stream = stream

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

    async def _invoke_llm(self) -> BaseMessage:
        """Invoke LLM with optional streaming.

        When streaming is enabled, emits on_llm_token callbacks for each chunk
        and accumulates the full response.

        Returns
        -------
        BaseMessage
            The complete LLM response.
        """
        if not self.stream:
            # Standard non-streaming invoke
            return await self.llm_with_tools.ainvoke(self.messages)

        # Streaming: accumulate chunks and emit tokens
        from langchain_core.messages import AIMessage, AIMessageChunk

        chunks: list[AIMessageChunk] = []
        async for chunk in self.llm_with_tools.astream(self.messages):
            if isinstance(chunk, AIMessageChunk):
                chunks.append(chunk)
                # Emit token callback for content
                if chunk.content:
                    self._emit("on_llm_token", chunk.content)

        # Concatenate all chunks into final message
        if not chunks:
            # Empty response - return empty AI message
            return AIMessage(content="")

        # Use LangChain's built-in chunk concatenation
        # AIMessageChunk.__add__ returns AIMessageChunk
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result + chunk

        # Convert final chunk to AIMessage for consistent return type
        return AIMessage(
            content=result.content,
            additional_kwargs=result.additional_kwargs,
            response_metadata=getattr(result, "response_metadata", {}),
            tool_calls=getattr(result, "tool_calls", []),
        )

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Safely emit a callback event.

        Callbacks are optional - if not set or method not implemented, silently skip.
        """
        if self.callbacks is None:
            return
        handler = getattr(self.callbacks, event, None)
        if handler is not None:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback {event} failed: {e}")

    def _log_llm_request(self, iteration: int) -> None:
        """Log LLM request to structured logs."""
        try:
            from questfoundry.runtime.logging import (
                is_structured_logging_configured,
                log_llm_request,
            )

            if not is_structured_logging_configured():
                return

            # Convert messages to dicts for logging
            messages_for_log = []
            for msg in self.messages:
                msg_dict = {
                    "role": getattr(msg, "type", "unknown"),
                    "content": getattr(msg, "content", ""),
                }
                messages_for_log.append(msg_dict)

            log_llm_request(
                messages=messages_for_log,
                iteration=iteration,
            )
        except Exception as e:
            logger.debug(f"Failed to log LLM request: {e}")

    def _log_llm_response(self, response: BaseMessage, iteration: int, duration_ms: float) -> None:
        """Log LLM response to structured logs."""
        try:
            from questfoundry.runtime.logging import (
                is_structured_logging_configured,
                log_llm_response,
            )

            if not is_structured_logging_configured():
                return

            # Extract tool calls if present
            tool_calls = getattr(response, "tool_calls", None) or []
            tool_calls_for_log = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})} for tc in tool_calls
            ]

            log_llm_response(
                content=getattr(response, "content", None),
                tool_calls=tool_calls_for_log if tool_calls_for_log else None,
                iteration=iteration,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.debug(f"Failed to log LLM response: {e}")

    def _log_tool_execution(
        self, tool_name: str, args: dict[str, Any], result: str, success: bool, duration_ms: float
    ) -> None:
        """Log tool execution to structured logs."""
        try:
            from questfoundry.runtime.logging import (
                is_structured_logging_configured,
                log_tool_execution,
            )

            if not is_structured_logging_configured():
                return

            log_tool_execution(
                tool_name=tool_name,
                args=args,
                result=result,
                success=success,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.debug(f"Failed to log tool execution: {e}")

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
            self._emit("on_llm_start", iteration)

            # Log LLM request to structured logs
            self._log_llm_request(iteration)

            start_time = time.perf_counter()
            try:
                response = await self._invoke_llm()
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                self._emit("on_error", f"LLM invocation failed: {e}")
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

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add AI response to history
            self.messages.append(response)

            # Log LLM response to structured logs
            self._log_llm_response(response, iteration, duration_ms)

            # Extract tool calls
            tool_calls = getattr(response, "tool_calls", None) or []
            self._emit("on_llm_end", iteration, bool(tool_calls))

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
                    error_msg = f"No tool calls after {failure_count} consecutive attempts"
                    self._emit("on_error", error_msg)
                    return ExecutorResult(
                        success=False,
                        error=error_msg,
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

                # Use debug when streaming to avoid duplicate output
                # (StreamingCallbacks handles visible output in streaming mode)
                if self.callbacks:
                    logger.debug(f"Executing tool: {tool_name}")
                else:
                    logger.info(f"Executing tool: {tool_name}")
                logger.debug(f"Tool args: {tool_args}")

                # Execute tool
                self._emit("on_tool_start", tool_name, tool_args)
                tool_start_time = time.perf_counter()
                observation, success = await self._execute_tool(tool_name, tool_args)
                tool_duration_ms = (time.perf_counter() - tool_start_time) * 1000
                self._emit("on_tool_end", tool_name, observation, success)

                # Log tool execution to structured logs
                self._log_tool_execution(tool_name, tool_args, observation, success, tool_duration_ms)

                # Log tool errors/warnings at appropriate level
                if not success:
                    logger.warning(f"Tool '{tool_name}' failed: {observation[:200]}")
                else:
                    # Check for semantic errors in successful tool calls (e.g., status='error')
                    try:
                        parsed_obs = json.loads(observation)
                        if parsed_obs.get("status") == "error":
                            logger.warning(
                                f"Tool '{tool_name}' returned error status: "
                                f"{parsed_obs.get('message', observation)[:200]}"
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass  # Not JSON or not a dict, ignore

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

                # Legacy callback for logging/tracing (deprecated, use callbacks instead)
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
                # IMPORTANT: Only treat as "done" if the tool actually succeeded.
                # If validation fails (success=False), continue the loop so the LLM
                # gets the error feedback and can retry with corrected values.
                # This is the "validate-with-feedback" pattern from v2.
                if tool_name in self.stop_tool_names:
                    # Parse the observation to check success
                    try:
                        parsed = json.loads(observation)
                        # Only terminate if the tool actually succeeded
                        if parsed.get("success", True):
                            found_done = True
                            done_result = parsed
                            done_result["_stop_tool"] = tool_name
                        else:
                            # Validation failed - log and continue loop
                            # The ToolMessage with error is already appended above,
                            # so LLM will see the feedback and can retry
                            logger.info(
                                f"Stop tool '{tool_name}' validation failed, "
                                f"continuing loop for retry. Error: {parsed.get('error', 'unknown')}"
                            )
                            # Mark as failed so failure_count increments
                            any_failed = True
                    except json.JSONDecodeError:
                        # Non-JSON response from stop tool is a bug - treat as error
                        # Feed back to LLM so it can see something went wrong
                        logger.error(
                            f"Stop tool '{tool_name}' returned non-JSON response: "
                            f"{observation[:200]}..."
                        )
                        any_failed = True

            # Update failure count
            if any_failed:
                failure_count += 1
            else:
                failure_count = 0

            # Check termination conditions
            if failure_count >= self.max_failures:
                error_msg = f"Max failures ({self.max_failures}) reached"
                self._emit("on_error", error_msg)
                return ExecutorResult(
                    success=False,
                    error=error_msg,
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=failure_count,
                    consults=consults,
                )

            if found_done:
                if self.callbacks:
                    logger.debug(f"Done tool '{self.done_tool_name}' called, execution complete")
                else:
                    logger.info(f"Done tool '{self.done_tool_name}' called, execution complete")
                stop_tool_name = (done_result or {}).get("_stop_tool", self.done_tool_name)
                self._emit("on_done", stop_tool_name, done_result or {})
                return ExecutorResult(
                    success=True,
                    done_tool_result=done_result,
                    tool_results=tool_results,
                    iterations=iteration,
                    failure_count=0,
                    consults=consults,
                )

        # Max iterations reached
        error_msg = (
            f"Max iterations ({self.max_iterations}) reached without calling {self.done_tool_name}"
        )
        self._emit("on_error", error_msg)
        return ExecutorResult(
            success=False,
            error=error_msg,
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
            error_result: dict[str, Any] = {
                "success": False,
                "error": str(e),
            }

            # Provide "did you mean" hints for parameter name errors
            hint = self._get_parameter_hint(tool, e)
            error_result["hint"] = hint

            return json.dumps(error_result), False

    def _get_parameter_hint(self, tool: BaseTool, error: Exception) -> str:
        """Generate a helpful hint for tool execution errors.

        Handles common LLM errors per section 9.4 Validate-with-Feedback Pattern:
        - 'unexpected keyword argument' - wrong parameter name
        - 'missing required positional arguments' - LLM passed wrong args entirely
        """
        error_str = str(error)
        default_hint = (
            "Check tool arguments and try again. Use consult_schema for field requirements."
        )

        # Check for TypeError patterns
        if not isinstance(error, TypeError):
            return default_hint

        import re

        # Get the tool's actual parameters from its schema
        try:
            schema = tool.get_input_schema()
            actual_params = (
                list(schema.model_fields.keys()) if hasattr(schema, "model_fields") else []
            )
        except Exception:
            actual_params = []

        # Pattern 1: "missing N required positional arguments: 'key' and 'value'"
        # This happens when LLM passes wrong argument names (e.g., 'item' instead of 'key'+'value')
        missing_match = re.search(r"missing \d+ required positional arguments?: (.+)", error_str)
        if missing_match:
            missing_args_str = missing_match.group(1)
            # Extract argument names from "'key' and 'value'" or "'key', 'value'"
            missing_args = re.findall(r"'(\w+)'", missing_args_str)

            if actual_params:
                # Generate tool-specific syntax hint
                if tool.name == "write_hot_sot":
                    return (
                        f"Missing required arguments: {', '.join(missing_args)}. "
                        f'Correct syntax: write_hot_sot(key="artifact_id", value={{...}}). '
                        f"'key' is a string ID like 'scene_1', 'value' is the artifact dict."
                    )
                else:
                    param_hints = ", ".join(f"{p}=..." for p in actual_params[:3])
                    return (
                        f"Missing required arguments: {', '.join(missing_args)}. "
                        f"Correct syntax: {tool.name}({param_hints}). "
                        f"Required parameters: {', '.join(actual_params)}"
                    )
            else:
                return f"Missing required arguments: {', '.join(missing_args)}. {default_hint}"

        # Pattern 2: "unexpected keyword argument 'foo'"
        match = re.search(r"unexpected keyword argument '(\w+)'", error_str)
        if not match:
            return default_hint

        wrong_param = match.group(1)

        if not actual_params:
            return f"Unknown parameter '{wrong_param}'. {default_hint}"

        # Find similar parameter names
        suggestions = self._find_similar_params(wrong_param, actual_params)

        if suggestions:
            suggestion_str = ", ".join(f"'{p}'" for p in suggestions)
            return (
                f"Unknown parameter '{wrong_param}'. "
                f"Did you mean {suggestion_str}? "
                f"Available parameters: {', '.join(actual_params)}"
            )
        else:
            return (
                f"Unknown parameter '{wrong_param}'. "
                f"Available parameters: {', '.join(actual_params)}"
            )

    def _find_similar_params(self, wrong: str, actual: list[str]) -> list[str]:
        """Find parameters similar to the wrong one.

        Uses suffix/prefix matching and common word matching for suggestions.
        """
        suggestions = []
        wrong_lower = wrong.lower()
        wrong_parts = set(wrong_lower.replace("_", " ").split())

        for param in actual:
            param_lower = param.lower()
            param_parts = set(param_lower.replace("_", " ").split())

            # Check for common words (e.g., "artifact" in both "target_artifact" and "artifact_id")
            common_words = wrong_parts & param_parts
            if common_words:
                suggestions.append(param)
                continue

            # Check for suffix match (e.g., "id" matches "artifact_id")
            if wrong_lower.endswith(param_lower) or param_lower.endswith(wrong_lower):
                suggestions.append(param)
                continue

            # Check for prefix match
            if wrong_lower.startswith(param_lower) or param_lower.startswith(wrong_lower):
                suggestions.append(param)

        return suggestions
