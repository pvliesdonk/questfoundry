"""ReAct pattern executor - legacy fallback for text-based tool calling.

This module provides the ReAct pattern with text-based tool calling (Action/Action Input)
as a FALLBACK for models without native bind_tools support. It's retained for backwards
compatibility and specific use cases requiring the ReAct pattern with "Final Answer" termination.

For new implementations, prefer ProtocolExecutor which handles protocol-based message routing
instead of ReAct's "Final Answer" pattern. ProtocolExecutor is designed for the protocol-based
execution model used throughout the system.

**Preferred approaches:**
- BindToolsExecutor: For models with native bind_tools support (GPT-4, Claude 3, Qwen, Llama 3.1+)
- ProtocolExecutor: For fallback text-based tool calling with protocol message routing
- ReActExecutor: Legacy option when ReAct pattern is specifically required

**When ReActExecutor might be needed:**
- Legacy integrations requiring ReAct pattern with "Final Answer" termination
- Models without bind_tools support that need ReAct-style agent loop

See tests/test_bind_tools_ollama.py for bind_tools validation and ProtocolExecutor examples.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

# Prompt size thresholds (in characters, ~4 chars per token)
# These are conservative estimates; actual limits depend on model
# Environment variables allow override without code changes
PROMPT_SIZE_ERROR_THRESHOLD = int(os.environ.get("QF_PROMPT_ERROR_THRESHOLD", "32000"))
PROMPT_SIZE_WARNING_THRESHOLD = int(os.environ.get("QF_PROMPT_WARNING_THRESHOLD", "16000"))

# Short-term memory cap (characters)
# Prevents unbounded growth of prior_conversation across re-invocations
# Set to ~8000 chars to leave headroom for system prompt (~13k) and user prompt
PRIOR_CONVERSATION_MAX_CHARS = int(os.environ.get("QF_MEMORY_CAP", "8000"))


REACT_INSTRUCTIONS = """
## Available Tools

{tool_descriptions}

## How to Use Tools (ReAct Pattern)

When you need to perform an action, use this exact format:

Thought: [Explain what you're thinking and why you need to use a tool]
Action: tool_name
Action Input: {{"arg1": "value1", "arg2": "value2"}}

After I execute the tool, I will provide:
Observation: [Tool result as JSON]

You can then:
- Use another tool if you need more information
- Provide your final answer when task is complete

## Final Answer Format

When you have completed your task, output:

Thought: [Explain your final reasoning]
Final Answer:
```json
{{
  "messages": [],
  "hot_sot_updates": {{}},
  "cold_sot_updates": {{}}
}}
```

## Important Guidelines

- Always start with a Thought before taking an Action
- Use exactly ONE action per response
- Action Input MUST be valid JSON (use double quotes for strings)
- Wait for Observation before your next Thought
- Do NOT include state or role_id in Action Input - these are injected automatically
"""


@dataclass
class ReActResult:
    """Result from ReAct execution."""

    success: bool
    final_answer: dict[str, Any] | None = None
    messages: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    raw_responses: list[str] = field(default_factory=list)
    conversation: str = ""  # Accumulated conversation for short-term memory


class ReActExecutor:
    """Execute role using ReAct pattern - fallback for text-based tool calling.

    This provides tool calling via explicit text-based instructions (Action/Action Input)
    as a fallback for models without native bind_tools support. Retained for backwards
    compatibility and models requiring the ReAct pattern.
    """

    def __init__(
        self,
        tool_map: dict[str, Any],
        state: Any,
        role_id: str,
        max_iterations: int | None = None,
        trace_callback: Any | None = None,
    ):
        """Initialize ReAct executor.

        Args:
            tool_map: Dictionary mapping tool names to tool instances
            state: Current StudioState for tool execution context
            role_id: ID of the role executing (for tool injection)
            max_iterations: Max tool call iterations (default from env or 5)
            trace_callback: Optional callback(intent, payload) for tracing events
        """
        self.tool_map = tool_map
        self.state = state
        self.role_id = role_id
        self.max_iterations = max_iterations or int(os.getenv("QF_REACT_MAX_ITERATIONS", "5"))
        self.debug = os.getenv("QF_REACT_DEBUG", "").lower() in ("true", "1", "yes")
        self.trace_callback = trace_callback

    def execute(
        self,
        llm: Any,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        prior_conversation: str = "",
    ) -> ReActResult:
        """Run the ReAct reasoning loop.

        Args:
            llm: LLM instance (NOT bound to tools)
            system_prompt: Role's system prompt from RuntimeContextAssembler
            user_prompt: User/task prompt for this execution
            tools: Tool specifications in OpenAI function format
            prior_conversation: Prior conversation history for this role (short-term memory)

        Returns:
            ReActResult with success status, messages, and tool results
        """
        # Build tool descriptions from OpenAI format
        tool_desc = self._build_tool_descriptions(tools)
        react_section = REACT_INSTRUCTIONS.format(tool_descriptions=tool_desc)
        full_system = system_prompt + "\n\n" + react_section

        # CRITICAL: Prompts are NEVER truncated before LLM invocation.
        # Log errors if prompt might exceed context window.
        prompt_size = len(full_system) + len(user_prompt) + len(prior_conversation)
        if prompt_size > PROMPT_SIZE_ERROR_THRESHOLD:
            log.error(
                f"PROMPT SIZE ERROR for {self.role_id}: {prompt_size} chars "
                f"(system={len(full_system)}, user={len(user_prompt)}, prior={len(prior_conversation)}). "
                f"Exceeds threshold ({PROMPT_SIZE_ERROR_THRESHOLD}). "
                "This may cause model context overflow and unpredictable failures. "
                "Consider reducing prompt content or using a model with larger context. "
                "Override threshold with QF_PROMPT_ERROR_THRESHOLD env var."
            )
        elif prompt_size > PROMPT_SIZE_WARNING_THRESHOLD:
            log.warning(
                f"Large prompt for {self.role_id}: {prompt_size} chars "
                f"(threshold: {PROMPT_SIZE_WARNING_THRESHOLD}). "
                "Monitor for context window issues."
            )

        # Initialize conversation with prior history (short-term memory)
        # Apply memory cap to prevent unbounded growth
        if prior_conversation:
            original_len = len(prior_conversation)
            if original_len > PRIOR_CONVERSATION_MAX_CHARS:
                # Truncate from the beginning, keeping most recent context
                truncated = prior_conversation[-PRIOR_CONVERSATION_MAX_CHARS:]
                # Find a clean break point (newline) to avoid mid-sentence truncation
                newline_idx = truncated.find("\n")
                if newline_idx > 0 and newline_idx < 500:
                    truncated = truncated[newline_idx + 1 :]
                prior_conversation = "[... earlier conversation truncated ...]\n\n" + truncated
                log.info(
                    f"Truncated prior_conversation from {original_len} to "
                    f"{len(prior_conversation)} chars for {self.role_id}"
                )
            conversation = prior_conversation + "\n\n---\n\n" + user_prompt
        else:
            conversation = user_prompt
        tool_results = []
        raw_responses = []

        if self.debug:
            log.info(f"ReAct starting for role {self.role_id}")
            log.debug(f"Available tools: {list(self.tool_map.keys())}")

        for iteration in range(self.max_iterations):
            log.debug(f"ReAct iteration {iteration + 1}/{self.max_iterations}")

            # Invoke LLM without bind_tools (text-based fallback approach)
            messages = [
                SystemMessage(content=full_system),
                HumanMessage(content=conversation),
            ]

            try:
                response = llm.invoke(messages)
                response_text = self._extract_content(response)
            except Exception as e:
                log.error(f"LLM invocation failed: {e}")
                return ReActResult(
                    success=False,
                    error=f"LLM invocation failed: {e}",
                    iterations=iteration + 1,
                    tool_results=tool_results,
                    raw_responses=raw_responses,
                    conversation=conversation,
                )

            raw_responses.append(response_text)

            # Trace raw LLM response for debugging
            if self.trace_callback:
                try:
                    self.trace_callback(
                        "llm_iteration",
                        {
                            "role_id": self.role_id,
                            "iteration": iteration + 1,
                            "max_iterations": self.max_iterations,
                            "response": response_text,  # Full response, no truncation
                        },
                    )
                except Exception as e:
                    log.warning(f"Trace callback failed: {e}")

            if self.debug:
                log.info(f"LLM Response (iter {iteration + 1}):\n{response_text[:500]}...")

            # Check for Final Answer
            if "Final Answer:" in response_text:
                # Include current response in conversation before returning
                final_conversation = conversation + f"\n\n{response_text}"
                return self._parse_final_answer(
                    response_text, tool_results, iteration + 1, raw_responses, final_conversation
                )

            # Parse tool call from response text
            tool_call = self._parse_tool_call(response_text)

            if tool_call is None:
                # No tool call and no Final Answer - try to extract any JSON as final
                log.warning("No tool call or Final Answer found, attempting JSON extraction")
                final_conversation = conversation + f"\n\n{response_text}"
                return self._parse_final_answer(
                    response_text, tool_results, iteration + 1, raw_responses, final_conversation
                )

            # Execute tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            log.info(f"[{self.role_id}] Executing tool: {tool_name}")
            log.debug(f"[{self.role_id}] Tool args: {tool_args}")

            observation = self._execute_tool(tool_name, tool_args)
            tool_results.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": observation,
                    "iteration": iteration + 1,
                }
            )

            if self.debug:
                log.info(f"Observation: {observation[:500]}...")

            # Append to conversation (include LLM response + observation)
            # This builds the ReAct chain: Thought → Action → Observation → Thought → ...
            conversation += f"\n\n{response_text}\nObservation: {observation}"

        # Max iterations reached
        log.warning(f"ReAct reached max iterations ({self.max_iterations}) for role {self.role_id}")
        return ReActResult(
            success=False,
            error=f"Reached maximum iterations ({self.max_iterations})",
            iterations=self.max_iterations,
            tool_results=tool_results,
            raw_responses=raw_responses,
            conversation=conversation,
        )

    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                # Handle multi-part content (e.g., from some providers)
                return "".join(str(part) for part in content)
            return str(content)
        return str(response)

    def _build_tool_descriptions(self, tools: list[dict]) -> str:
        """Convert OpenAI function format to human-readable text.

        Args:
            tools: List of tool specs in OpenAI function format

        Returns:
            Formatted string describing all available tools
        """
        lines = []
        for spec in tools:
            # Handle both {"function": {...}} and direct format
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
                    # Skip internal parameters that are injected
                    if pname in ("state", "role_id"):
                        continue
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")
            lines.append("")
        return "\n".join(lines)

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        """Parse Action/Action Input from LLM response.

        Looks for:
            Action: tool_name
            Action Input: {"arg1": "value1"}

        Args:
            text: Raw LLM response text

        Returns:
            Dict with "name" and "args", or None if no tool call found
        """
        # Find Action: line
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()

        # Skip if this is "Final" (part of "Final Answer")
        if tool_name.lower() in ("final",):
            return None

        # Find Action Input: {...}
        # Use a more robust pattern that handles multi-line JSON
        input_match = re.search(
            r"Action Input:\s*(\{[\s\S]*?\})(?=\n\n|\nThought:|\nAction:|\nObservation:|$)",
            text,
            re.IGNORECASE,
        )

        if not input_match:
            # Try simpler pattern for single-line JSON
            input_match = re.search(
                r"Action Input:\s*(\{[^}]+\})",
                text,
                re.IGNORECASE,
            )

        if not input_match:
            log.warning(f"Found Action '{tool_name}' but no Action Input")
            return {"name": tool_name, "args": {}}

        json_str = input_match.group(1).strip()

        try:
            tool_args = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse Action Input JSON: {e}")
            log.debug(f"Raw JSON string: {json_str}")
            # Try to fix common issues
            tool_args = self._attempt_json_fix(json_str)

        return {"name": tool_name, "args": tool_args}

    def _attempt_json_fix(self, json_str: str) -> dict:
        """Attempt to fix common JSON formatting issues.

        Args:
            json_str: Malformed JSON string

        Returns:
            Parsed dict or empty dict if unfixable
        """
        # Try removing trailing content after closing brace
        if "}" in json_str:
            idx = json_str.rfind("}")
            try:
                return json.loads(json_str[: idx + 1])
            except json.JSONDecodeError:
                pass

        # Try with single quotes replaced
        try:
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        log.warning("Could not fix malformed JSON, using empty args")
        return {}

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool and return observation string.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments from LLM (state/role_id injected automatically)

        Returns:
            JSON string of tool result, or error message
        """
        tool = self.tool_map.get(tool_name)
        if tool is None:
            available = list(self.tool_map.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"

        try:
            # Inject state and role_id for stateful tools
            payload = {**tool_args, "state": self.state, "role_id": self.role_id}

            # Try different invocation methods
            if hasattr(tool, "_run"):
                # Direct _run method (BaseTool)
                import inspect

                sig = inspect.signature(tool._run)
                # Only pass params the tool accepts
                valid_params = {}
                for key, value in payload.items():
                    if key in sig.parameters:
                        valid_params[key] = value
                result = tool._run(**valid_params)
            elif hasattr(tool, "invoke"):
                # LangChainToolAdapter.invoke() takes **kwargs, not a positional dict
                result = tool.invoke(**payload)
            else:
                result = tool(**payload)

            # Convert result to JSON string
            if isinstance(result, str):
                # Check if already JSON
                try:
                    json.loads(result)
                    return result
                except json.JSONDecodeError:
                    return json.dumps({"result": result}, indent=2, default=str)
            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            log.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return json.dumps({"error": str(e), "tool": tool_name}, indent=2)

    def _parse_final_answer(
        self,
        text: str,
        tool_results: list,
        iterations: int,
        raw_responses: list[str],
        conversation: str = "",
    ) -> ReActResult:
        """Extract JSON from Final Answer section.

        Supports:
        - ```json ... ``` code blocks
        - ``` ... ``` code blocks
        - Raw JSON objects {...}

        Args:
            text: LLM response text
            tool_results: Accumulated tool results
            iterations: Number of iterations completed
            raw_responses: All raw LLM responses

        Returns:
            ReActResult with parsed data or error
        """
        try:
            json_str = None

            # Method 1: JSON code block after Final Answer
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                json_str = text[start:end].strip()

            # Method 2: Generic code block
            elif "```" in text:
                # Find the code block after "Final Answer" if present
                fa_pos = text.lower().find("final answer")
                search_start = fa_pos if fa_pos != -1 else 0
                block_start = text.index("```", search_start) + 3
                # Skip optional language identifier
                newline_pos = text.find("\n", block_start)
                if newline_pos != -1 and newline_pos - block_start < 20:
                    block_start = newline_pos + 1
                block_end = text.index("```", block_start)
                json_str = text[block_start:block_end].strip()

            # Method 3: Raw JSON object by brace matching
            elif "{" in text:
                # Find first { after "Final Answer" if present
                fa_pos = text.lower().find("final answer")
                search_start = fa_pos if fa_pos != -1 else 0
                start = text.index("{", search_start)
                brace_count = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                json_str = text[start:end]

            if not json_str:
                return ReActResult(
                    success=False,
                    error="No JSON found in final answer",
                    iterations=iterations,
                    tool_results=tool_results,
                    raw_responses=raw_responses,
                    conversation=conversation,
                )

            data = json.loads(json_str)

            # Extract messages from various possible locations
            messages = data.get("messages", [])
            if not messages and "message" in data:
                messages = [data["message"]]

            return ReActResult(
                success=True,
                final_answer=data,
                messages=messages,
                tool_results=tool_results,
                iterations=iterations,
                raw_responses=raw_responses,
                conversation=conversation,
            )

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse final answer JSON: {e}")
            return ReActResult(
                success=False,
                error=f"JSON parse error: {e}",
                tool_results=tool_results,
                iterations=iterations,
                raw_responses=raw_responses,
                conversation=conversation,
            )
        except ValueError as e:
            log.error(f"Failed to extract JSON from response: {e}")
            return ReActResult(
                success=False,
                error=f"JSON extraction error: {e}",
                tool_results=tool_results,
                iterations=iterations,
                raw_responses=raw_responses,
                conversation=conversation,
            )
