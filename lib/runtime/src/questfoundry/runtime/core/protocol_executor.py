"""Fallback executor using text-based tool calling via Action/Action Input format.

This module provides a FALLBACK approach for models that don't support bind_tools.
The preferred approach for capable models is BindToolsExecutor with native structured tool calls.

For models without bind_tools support, ProtocolExecutor provides universal tool calling via
explicit Action/Action Input text format, enabling compatibility across all LLM providers
and model sizes (including Llama 3.2 1B/3B, Qwen, smaller models without bind_tools).

**Parsing approach:**
- Action/Action Input text parsing: Models output explicit tool calls in text format
- Supports parallel fanout: Multiple action/input pairs in a single response
- Role termination: Role is DONE when send_protocol_message is called

**Key difference from ReAct:**
- NOT a ReAct agent loop with "Final Answer" termination
- Roles communicate via protocol messages to other roles in the system
- See lib/runtime/AGENTS.md for the protocol-based execution model
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

# Prompt size thresholds (in characters, ~4 chars per token)
PROMPT_SIZE_ERROR_THRESHOLD = int(os.environ.get("QF_PROMPT_ERROR_THRESHOLD", "32000"))
PROMPT_SIZE_WARNING_THRESHOLD = int(os.environ.get("QF_PROMPT_WARNING_THRESHOLD", "16000"))

# Short-term memory cap (characters)
PRIOR_CONVERSATION_MAX_CHARS = int(os.environ.get("QF_MEMORY_CAP", "8000"))

# Tools that signal "role is done for this turn"
PROTOCOL_MESSAGE_TOOLS = {"send_protocol_message", "send_message"}


TOOL_INSTRUCTIONS = """
## Available Tools

{tool_descriptions}

## How to Use Tools

When you need to perform an action, use this exact format:

Thought: [Explain what you're thinking and why you need to use a tool]
Action: tool_name
Action Input: {{"arg1": "value1", "arg2": "value2"}}

After the tool executes, you will see:
Observation: [Tool result as JSON]

You can then use another tool if needed.

## Completing Your Turn

Your turn is complete when you call `send_protocol_message`.
This routes your message to another role (via the `receiver` field).

Example:
Thought: I've completed my analysis. Routing to plotwright for topology design.
Action: send_protocol_message
Action Input: {{"receiver": "plotwright", "intent": "task.assign", "content": "Design topology."}}

After sending a protocol message, your turn ends and the graph routes to the receiver.

## Important Guidelines

- Always start with a Thought before taking an Action
- Action Input MUST be valid JSON (use double quotes for strings)
- Wait for Observation before your next action
- Do NOT include state or role_id in Action Input - these are injected automatically
- Complete your turn by sending a protocol message to the appropriate receiver
- For parallel fanout: you MAY send multiple protocol messages in one response
"""


@dataclass
class ExecutorResult:
    """Result from protocol executor."""

    success: bool
    messages: list[dict] = field(default_factory=list)  # Protocol messages sent
    tool_results: list[dict] = field(default_factory=list)
    iterations: int = 0
    failure_count: int = 0  # Counts only failures, not valid tool calls
    error: str | None = None
    raw_responses: list[str] = field(default_factory=list)
    work_summary: str = ""  # Summary of work done (for short-term memory)


class ProtocolExecutor:
    """Execute role using text-based tool calling with protocol-message termination.

    This provides universal tool calling that works across all LLM providers
    and models by using explicit text-based instructions rather than hidden
    token injection.

    CRITICAL: This is NOT a ReAct loop with "Final Answer" termination.
    - Role is done when it calls send_protocol_message
    - Only FAILURES count against max_failures (not valid tool calls)
    - Messages from send_protocol_message ARE the output
    """

    def __init__(
        self,
        tool_map: dict[str, Any],
        state: Any,
        role_id: str,
        max_failures: int | None = None,
        trace_callback: Any | None = None,
    ):
        """Initialize protocol executor.

        Args:
            tool_map: Dictionary mapping tool names to tool instances
            state: Current StudioState for tool execution context
            role_id: ID of the role executing (for tool injection)
            max_failures: Max consecutive failures before giving up (default 3)
            trace_callback: Optional callback(intent, payload) for tracing events
        """
        self.tool_map = tool_map
        self.state = state
        self.role_id = role_id
        self.max_failures = max_failures or int(os.getenv("QF_MAX_FAILURES", "3"))
        self.debug = os.getenv("QF_DEBUG", "").lower() in ("true", "1", "yes")
        self.trace_callback = trace_callback

    def execute(
        self,
        llm: Any,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        prior_conversation: str = "",
    ) -> ExecutorResult:
        """Run the tool execution loop until protocol message is sent.

        The loop continues until:
        - A protocol message is sent (success, role done for this turn)
        - Max failures reached (error, role couldn't complete)

        Args:
            llm: LLM instance (NOT bound to tools)
            system_prompt: Role's system prompt from RuntimeContextAssembler
            user_prompt: User/task prompt for this execution
            tools: Tool specifications in OpenAI function format
            prior_conversation: Prior conversation history for this role

        Returns:
            ExecutorResult with success status, messages, and tool results
        """
        # Build tool descriptions
        tool_desc = self._build_tool_descriptions(tools)
        tool_section = TOOL_INSTRUCTIONS.format(tool_descriptions=tool_desc)
        full_system = system_prompt + "\n\n" + tool_section

        # Check prompt size
        prompt_size = len(full_system) + len(user_prompt) + len(prior_conversation)
        if prompt_size > PROMPT_SIZE_ERROR_THRESHOLD:
            log.error(
                f"PROMPT SIZE ERROR for {self.role_id}: {prompt_size} chars. "
                f"Exceeds threshold ({PROMPT_SIZE_ERROR_THRESHOLD})."
            )
        elif prompt_size > PROMPT_SIZE_WARNING_THRESHOLD:
            log.warning(
                f"Large prompt for {self.role_id}: {prompt_size} chars."
            )

        # Initialize conversation with prior history (with memory cap)
        if prior_conversation:
            original_len = len(prior_conversation)
            if original_len > PRIOR_CONVERSATION_MAX_CHARS:
                truncated = prior_conversation[-PRIOR_CONVERSATION_MAX_CHARS:]
                newline_idx = truncated.find("\n")
                if newline_idx > 0 and newline_idx < 500:
                    truncated = truncated[newline_idx + 1:]
                prior_conversation = "[... earlier conversation truncated ...]\n\n" + truncated
                log.info(
                    f"Truncated prior_conversation from {original_len} to "
                    f"{len(prior_conversation)} chars for {self.role_id}"
                )
            conversation = prior_conversation + "\n\n---\n\n" + user_prompt
        else:
            conversation = user_prompt

        tool_results: list[dict] = []
        raw_responses: list[str] = []
        protocol_messages: list[dict] = []
        failure_count = 0
        iteration = 0
        work_done: list[str] = []

        log.info(f"ProtocolExecutor starting for role {self.role_id}")
        log.debug(f"Available tools: {list(self.tool_map.keys())}")

        # Loop until protocol message sent or max failures reached
        while failure_count < self.max_failures:
            iteration += 1
            log.debug(f"Iteration {iteration}, failures: {failure_count}/{self.max_failures}")

            # Invoke LLM (NO tool binding - pure text)
            messages = [
                SystemMessage(content=full_system),
                HumanMessage(content=conversation),
            ]

            try:
                response = llm.invoke(messages)
                response_text = self._extract_content(response)
            except Exception as e:
                log.error(f"LLM invocation failed: {e}")
                failure_count += 1
                if failure_count >= self.max_failures:
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

            raw_responses.append(response_text)

            # Trace for debugging
            if self.trace_callback:
                try:
                    self.trace_callback("llm_iteration", {
                        "role_id": self.role_id,
                        "iteration": iteration,
                        "failure_count": failure_count,
                        "response": response_text,
                    })
                except Exception as e:
                    log.warning(f"Trace callback failed: {e}")

            if self.debug:
                log.info(f"LLM Response (iter {iteration}):\n{response_text[:500]}...")

            # Parse ALL tool calls from response (supports parallel fanout)
            all_tool_calls = self._parse_all_tool_calls(response_text)

            if not all_tool_calls:
                # No valid tool call - count as failure
                log.warning(f"No tool call found in response for {self.role_id}")
                failure_count += 1
                error_msg = "Error - No tool call found. Use Action/Action Input format."
                conversation += f"\n\n{response_text}\n\nObservation: {error_msg}"
                continue

            log.info(f"Found {len(all_tool_calls)} tool call(s) in response")

            # Execute all tool calls from this response
            observations = []
            any_failed = False
            found_protocol_message = False

            for tool_call in all_tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                log.info(f"Executing tool: {tool_name}")
                log.debug(f"Tool args: {tool_args}")

                # Execute tool
                observation, tool_success = self._execute_tool(tool_name, tool_args)

                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": observation,
                    "success": tool_success,
                    "iteration": iteration,
                })

                observations.append(f"{tool_name}: {observation}")

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

            # No protocol message - continue loop with observations
            combined_obs = "\n".join(observations)
            if self.debug:
                log.info(f"Observations: {combined_obs[:500]}...")

            conversation += f"\n\n{response_text}\nObservation: {combined_obs}"

        # Max failures reached
        log.error(f"Max failures ({self.max_failures}) reached for {self.role_id}")
        return ExecutorResult(
            success=False,
            error=f"Reached maximum failures ({self.max_failures})",
            messages=protocol_messages,
            tool_results=tool_results,
            iterations=iteration,
            failure_count=failure_count,
            raw_responses=raw_responses,
            work_summary=self._build_work_summary(work_done),
        )

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return "".join(str(part) for part in content)
            return str(content)
        return str(response)

    def _build_tool_descriptions(self, tools: list[dict]) -> str:
        """Convert OpenAI function format to human-readable text."""
        lines = []
        for spec in tools:
            func = spec.get("function", spec)
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            lines.append(f"**Tool: {name}**")
            lines.append(f"Description: {desc}")
            if params:
                lines.append("Arguments:")
                for pname, pinfo in params.items():
                    if pname in ("state", "role_id"):
                        continue
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")
            lines.append("")
        return "\n".join(lines)

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        """Parse first Action/Action Input from LLM response.

        For backward compatibility. Use _parse_all_tool_calls for multiple actions.
        """
        calls = self._parse_all_tool_calls(text)
        return calls[0] if calls else None

    def _parse_all_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse ALL Action/Action Input pairs from LLM response.

        Supports parallel fanout where LLM outputs multiple actions:
            Action: send_protocol_message
            Action Input: {"receiver": "plotwright", ...}

            Action: send_protocol_message
            Action Input: {"receiver": "scene_smith", ...}

        Returns:
            List of dicts with "name" and "args", empty list if no tool calls found
        """
        tool_calls = []

        # Find all Action: lines with their positions
        action_pattern = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
        action_matches = list(action_pattern.finditer(text))

        if not action_matches:
            return []

        for i, action_match in enumerate(action_matches):
            tool_name = action_match.group(1).strip()

            # Determine the search region for Action Input
            start_pos = action_match.end()
            end_pos = action_matches[i + 1].start() if i + 1 < len(action_matches) else len(text)
            search_region = text[start_pos:end_pos]

            # Find Action Input in this region
            input_match = re.search(
                r"Action Input:\s*(\{[\s\S]*?\})(?=\n\n|\nThought:|\nAction:|\nObservation:|$)",
                search_region,
                re.IGNORECASE,
            )

            if not input_match:
                # Try simpler pattern for nested JSON
                input_match = re.search(
                    r"Action Input:\s*(\{[\s\S]*\})",
                    search_region,
                    re.IGNORECASE,
                )

            if not input_match:
                log.warning(f"Found Action '{tool_name}' but no Action Input")
                tool_calls.append({"name": tool_name, "args": {}})
                continue

            json_str = input_match.group(1).strip()

            # Try to parse JSON, handling nested structures
            tool_args = self._parse_json_robust(json_str)
            tool_calls.append({"name": tool_name, "args": tool_args})

        return tool_calls

    def _parse_json_robust(self, json_str: str) -> dict:
        """Robustly parse JSON, handling nested structures and common errors."""
        # First try direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Try to find balanced braces for nested JSON
        depth = 0
        end_idx = 0
        for i, char in enumerate(json_str):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            try:
                return json.loads(json_str[:end_idx])
            except json.JSONDecodeError:
                pass

        # Fall back to original fix attempt
        return self._attempt_json_fix(json_str)

    def _attempt_json_fix(self, json_str: str) -> dict:
        """Attempt to fix common JSON formatting issues."""
        if "}" in json_str:
            idx = json_str.rfind("}")
            try:
                return json.loads(json_str[: idx + 1])
            except json.JSONDecodeError:
                pass

        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        log.warning("Could not fix malformed JSON, using empty args")
        return {}

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool and return (observation, success).

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments from LLM (state/role_id injected automatically)

        Returns:
            Tuple of (observation string, success boolean)
        """
        tool = self.tool_map.get(tool_name)
        if tool is None:
            available = list(self.tool_map.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}", False

        try:
            payload = {**tool_args, "state": self.state, "role_id": self.role_id}

            if hasattr(tool, "_run"):
                import inspect
                sig = inspect.signature(tool._run)
                valid_params = {k: v for k, v in payload.items() if k in sig.parameters}
                result = tool._run(**valid_params)
            elif hasattr(tool, "invoke"):
                result = tool.invoke(**payload)
            else:
                result = tool(**payload)

            # Convert to JSON string
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
        """Build a summary of work done for short-term memory."""
        if not work_done:
            return ""
        summary = f"Actions taken: {', '.join(work_done)}"
        if len(summary) > PRIOR_CONVERSATION_MAX_CHARS:
            summary = summary[:PRIOR_CONVERSATION_MAX_CHARS] + "..."
        return summary
