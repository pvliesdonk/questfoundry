"""
Node Factory - transforms role profiles into LangGraph-compatible Runnable nodes.

Based on spec: components/node_factory.md
STRICT component - role → Runnable transformation is the core contract.
"""

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from questfoundry.runtime.core.protocol_executor import ProtocolExecutor
from questfoundry.runtime.core.provider_manager import ProviderManager
from questfoundry.runtime.core.runtime_context_assembler import RuntimeContextAssembler
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.models.state import StudioState
from questfoundry.runtime.plugins.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)

# Optional LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # LangSmith not available, use no-op decorator
    def traceable(**kwargs):
        def decorator(func):
            return func

        return decorator


class NodeFactory:
    """Transform role profiles into LangGraph-compatible Runnable nodes."""

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        state_manager: Any | None = None,
        preferred_provider: str | None = None,
        context_assembler: RuntimeContextAssembler | None = None,
    ):
        """Initialize node factory.

        Args:
            schema_registry: SchemaRegistry instance (creates new if not provided)
            state_manager: Optional StateManager for tracing messages
            preferred_provider: Preferred provider (e.g., "ollama") or fallback chain
                (e.g., "ollama,openai")
            context_assembler: RuntimeContextAssembler instance (creates new if not provided)
        """
        self.schema_registry = schema_registry or SchemaRegistry()
        self.provider_manager = ProviderManager()
        self.context_assembler = context_assembler or RuntimeContextAssembler()
        self._role_cache: dict[str, RoleProfile] = {}
        self.state_manager = state_manager  # For message tracing
        self.preferred_provider = preferred_provider  # Store for use in select_llm
        self.tool_registry = get_tool_registry()

    def _extract_task_context_for_role(self, role_id: str, state: StudioState) -> str | None:
        """
        Extract task context from protocol messages addressed to this role.

        Looks for messages in state["messages"] where the receiver matches this role,
        and extracts task description/payload to provide context for the role's prompt.

        Args:
            role_id: The role identifier (e.g., "plotwright")
            state: Current studio state containing messages

        Returns:
            Task context string if found, None otherwise
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Role ID aliases for matching (e.g., "PW" -> "plotwright")
        role_aliases = {
            "plotwright": {"plotwright", "pw"},
            "lore_weaver": {"lore_weaver", "lw"},
            "scene_smith": {"scene_smith", "ss"},
            "style_lead": {"style_lead", "st"},
            "gatekeeper": {"gatekeeper", "gk"},
            "codex_curator": {"codex_curator", "cc"},
            "researcher": {"researcher", "rs"},
            "player_narrator": {"player_narrator", "pn"},
            "art_director": {"art_director", "ad"},
            "audio_director": {"audio_director", "audr"},
            "illustrator": {"illustrator", "il"},
            "audio_producer": {"audio_producer", "aupr"},
            "book_binder": {"book_binder", "bb"},
            "translator": {"translator", "tr"},
            "showrunner": {"showrunner", "sr"},
        }

        # Get aliases for this role
        aliases = role_aliases.get(role_id, {role_id})

        # Find messages addressed to this role (most recent first)
        task_parts = []
        for msg in reversed(messages):
            receiver = msg.get("receiver", "")

            # Normalize receiver to string
            if isinstance(receiver, dict):
                receiver = receiver.get("role", "") or receiver.get("id", "")
            receiver = str(receiver).lower()

            # Check if message is for this role (or broadcast)
            if receiver in aliases or receiver == "*":
                payload = msg.get("payload", {})
                intent = msg.get("intent", "")

                # Extract task details based on intent type
                if intent in ("tu.assign", "tu.open", "task.assign"):
                    # Direct task assignment
                    desc = payload.get("description", "") or payload.get("task", "")
                    loop = payload.get("loop", "")
                    deliverables = payload.get("deliverables", [])

                    if desc:
                        context = f"**Task**: {desc}"
                        if loop:
                            context += f"\n**Loop**: {loop}"
                        if deliverables:
                            context += f"\n**Deliverables**: {', '.join(deliverables)}"
                        task_parts.append(context)

                elif intent == "artifact.request":
                    # Request for specific artifact
                    artifact_type = payload.get("artifact_type", "")
                    requirements = payload.get("requirements", "")
                    if artifact_type:
                        context = f"**Requested Artifact**: {artifact_type}"
                        if requirements:
                            context += f"\n**Requirements**: {requirements}"
                        task_parts.append(context)

                elif intent == "feedback.request":
                    # Feedback request
                    subject = payload.get("subject", "")
                    question = payload.get("question", "")
                    if subject or question:
                        context = f"**Feedback Requested**: {subject or question}"
                        task_parts.append(context)

                # Also check for generic task/description fields
                if not task_parts:
                    for key in ("task", "description", "message", "content"):
                        if key in payload and payload[key]:
                            task_parts.append(str(payload[key]))
                            break

        if not task_parts:
            return None

        # Return unique tasks (preserve order)
        seen = set()
        unique_tasks = []
        for t in task_parts:
            if t not in seen:
                seen.add(t)
                unique_tasks.append(t)

        return "\n\n".join(unique_tasks)

    def load_role(self, role_id: str) -> RoleProfile:
        """
        Load and validate role YAML file.

        Steps:
        1. Construct path: spec/05-definitions/roles/{role_id}.yaml
        2. Load YAML with PyYAML
        3. Validate against role_profile.schema.json using SchemaRegistry
        4. Parse into structured RoleProfile object
        5. Return RoleProfile

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            RoleProfile object

        Raises:
            FileNotFoundError: If role YAML doesn't exist
            ValidationError: If YAML doesn't match schema
        """
        return self.schema_registry.load_role(role_id)

    def should_execute_role(self, role: RoleProfile, state: StudioState) -> bool:
        """
        Determine if role should execute based on dormancy policy.

        Dormancy Policies:
        - active: Always execute (default)
        - optional: Execute if explicitly requested in context
        - default_dormant: Execute only if wake_condition is met

        Args:
            role: RoleProfile object
            state: Current loop state

        Returns:
            True if role should execute, False if dormant
        """
        return role.should_execute(state)

    def assemble_role_context(self, role_id: str, state: StudioState) -> dict[str, Any]:
        """
        Assemble complete role context using RuntimeContextAssembler.

        This replaces the old template-based approach with dynamic context assembly.
        Uses the 5-layer prompt architecture:
        1. IDENTITY - Role charter and operating principles
        2. PROTOCOL - Communication permissions and intents
        3. STATE - Current execution context
        4. MISSION - Loop, node, and task guidance
        5. INTERFACE - Tools and structured output requirements

        Args:
            role_id: Role identifier (e.g., "plotwright")
            state: Current studio state

        Returns:
            Dictionary with:
            - prompt: Complete assembled prompt
            - tools: Tool configurations for LLM binding
            - role_def: Role definition for reference
        """
        loop_id = state.get("loop_id")
        node_id = state.get("node_id")

        try:
            context = self.context_assembler.assemble_context(
                role_id=role_id,
                loop_id=loop_id,
                node_id=node_id,
                state=state,
            )

            logger.info(
                f"Assembled context for {role_id}: "
                f"prompt_size={len(context['prompt'])}, "
                f"tools={len(context['tools'])}"
            )

            return context
        except Exception as e:
            logger.error(f"Context assembly failed for role {role_id}: {e}")
            raise

    def render_prompt(self, role: RoleProfile, state: StudioState) -> str:
        """
        Render prompt using RuntimeContextAssembler.

        This method now delegates to assemble_role_context for dynamic prompt
        generation. Kept for backward compatibility with existing code paths.

        Args:
            role: RoleProfile object
            state: Current studio state

        Returns:
            Formatted prompt string
        """
        context = self.assemble_role_context(role.id, state)
        return context["prompt"]

    def extract_artifacts(self, role: RoleProfile, llm_output: str, tu_id: str) -> dict[str, Any]:
        """
        Extract artifacts from LLM JSON output and map to hot_sot keys.

        This bridges the gap between:
        1. LLM structured output fields (e.g., "briefs_written")
        2. Spec-defined hot_sot keys (e.g., "hot_sot.section_briefs")

        Args:
            role: RoleProfile with outputs specification
            llm_output: Raw JSON string from LLM
            tu_id: Trace Unit ID for artifact metadata

        Returns:
            Dict with hot_sot updates

        Example:
            Plotwright outputs {"briefs_written": ["id1", "id2"]}
            → Returns {"hot_sot": {"section_briefs": ["id1", "id2"]}}

        Note:
            Hot SoT is in-memory for Phase 1. Future: Redis/Memory backend.
            Cold SoT will use SQLite/File for persistence.
        """
        try:
            # 1. Parse JSON output
            output_data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON for {role.id}: {e}")
            logger.debug(f"Raw output: {llm_output[:500]}")
            return {}

        hot_sot_updates = {}

        # 2. Get role's output specification from YAML
        role_outputs = role.raw.get("interface", {}).get("outputs", [])

        # 3. Map LLM output fields to hot_sot keys
        # The structured_output schema defines what fields the LLM outputs
        # The outputs list defines where those should be stored (destination: hot → hot_sot)

        for output_spec in role_outputs:
            artifact_type = output_spec.get("artifact_type")
            state_key = output_spec.get("state_key", "")
            destination = output_spec.get("destination", "hot")

            if not state_key or destination != "hot":
                continue

            # Extract the hot_sot key from state_key
            # Examples:
            #   "artifacts.section_briefs" → "section_briefs" (legacy format, still works)
            #   "hot_sot.section_briefs" → "section_briefs"
            if state_key.startswith("artifacts."):
                hot_key = state_key.replace("artifacts.", "")
            elif state_key.startswith("hot_sot."):
                hot_key = state_key.replace("hot_sot.", "")
            else:
                hot_key = artifact_type

            # Try to find matching data in LLM output
            # Common patterns:
            # - section_briefs → briefs_written
            # - topology_notes → topology_updated / topology_notes
            # - hooks → hooks_proposed

            value = None

            # Try exact match first
            if hot_key in output_data:
                value = output_data[hot_key]
            # Try common field name patterns
            elif f"{artifact_type}_written" in output_data:
                value = output_data[f"{artifact_type}_written"]
            elif f"{artifact_type}s_written" in output_data:  # plural
                value = output_data[f"{artifact_type}s_written"]
            elif f"{artifact_type}_proposed" in output_data:
                value = output_data[f"{artifact_type}_proposed"]
            elif "briefs_written" in output_data and artifact_type == "section_briefs":
                # Specific mapping for plotwright briefs
                value = output_data["briefs_written"]

            if value is not None:
                hot_sot_updates[hot_key] = value

        # 4. Store full LLM output for debugging (in artifacts, not hot_sot)
        # This preserves raw output without polluting the hot_sot structure
        return {
            "hot_sot": hot_sot_updates,
            "artifacts": {f"_{role.id}_raw_output": output_data},  # Debug info
        }

    def select_llm(self, role: RoleProfile) -> dict[str, Any | None]:
        """
        Select appropriate LLM based on role type and model config.

        Role Types:
        1. reasoning_agent: Full LLM with complex reasoning
        2. production_executor: Thin LLM wrapper + heavy tool orchestration
        3. service: Pure tool execution, no LLM needed

        Uses ProviderManager for:
        - Provider auto-detection and selection
        - Model tier resolution (work-type → actual model)

        Args:
            role: RoleProfile with role_type and model_config

        Returns:
            LLM configuration dict or None for service type

        Note:
            Provider can be overridden via QF_LLM_PROVIDER environment variable.
            Model can be overridden via QF_DEFAULT_MODEL environment variable.
        """
        if role.role_type == "service":
            # Service type - no LLM needed
            return None

        # 1. Select provider (priority: env var > CLI > role YAML > auto)
        # Build provider preference with fallback chain if provided
        preferred_provider = (
            os.getenv("QF_LLM_PROVIDER") or self.preferred_provider or role.get_provider()
        )

        # If provider comes from env/CLI and is not "auto", treat it as
        # an explicit choice and enable strict provider selection. In that
        # mode we will not silently fall back to unrelated providers.
        env_or_cli_provider = os.getenv("QF_LLM_PROVIDER") or self.preferred_provider
        strict_provider_selection = bool(env_or_cli_provider) and preferred_provider != "auto"

        # Parse fallback chain if comma-separated (e.g., "ollama,openai")
        fallback_chain = None
        if preferred_provider and "," in preferred_provider:
            providers = [p.strip() for p in preferred_provider.split(",")]
            preferred_provider = providers[0]  # First is preferred
            fallback_chain = providers[1:]  # Rest are fallbacks

        provider = self.provider_manager.select_provider(
            preferred_provider, fallback_chain, strict=strict_provider_selection
        )

        # 2. Resolve model tier to actual model name
        # Check for explicit model override first (backward compatibility)
        if os.getenv("QF_DEFAULT_MODEL"):
            model = os.getenv("QF_DEFAULT_MODEL")
            logger.info(f"Using model override from QF_DEFAULT_MODEL: {model}")
        else:
            # Get tier from role (or use specific model if tier not specified)
            model_tier = role.get_model_tier()

            # Check if role specifies a specific model (backward compatibility)
            specific_model = role.model_config.get("model")
            if specific_model:
                # Use specific model directly
                model = specific_model
                logger.info(f"Using specific model from role config: {model}")
            else:
                # Resolve tier to provider-specific model
                model = self.provider_manager.resolve_model(provider, model_tier)
                logger.info(f"Resolved tier '{model_tier}' to {provider}:{model}")

        # 3. Get other parameters
        temperature = role.get_temperature()
        max_tokens = role.get_max_tokens()

        logger.info(
            f"Selected LLM - provider: {provider}, model: {model}, temperature: {temperature}"
        )

        return {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "role_type": role.role_type,
            # Provider selection metadata used for graceful fallback
            "fallback_chain": fallback_chain,
            "strict_provider_selection": strict_provider_selection,
        }

    def _trace_tool_event(self, role_id: str, intent: str, payload: dict, tu_id: str) -> None:
        """Trace a tool-related event if trace handler is available."""
        if not (self.state_manager and hasattr(self.state_manager, "_trace_handler")):
            return
        if not self.state_manager._trace_handler:
            return

        message = {
            "sender": role_id,
            "receiver": "system",
            "intent": intent,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "envelope": {"tu_id": tu_id},
        }
        try:
            self.state_manager._trace_handler.trace_message(message)
        except Exception as e:
            logger.warning(f"Trace handler error: {e}")

    @staticmethod
    def _merge_tool_result(aggregated: dict[str, Any], result: Any) -> dict[str, Any]:
        """
        Merge a single tool result into the aggregated updates bucket.

        Supports the common shapes returned by our tools:
        - {"messages": [...]}
        - {"hot_sot": {...}}
        - {"cold_sot": {...}}
        - {"artifacts": {...}}
        """
        if not isinstance(result, dict):
            return aggregated

        for key in ("hot_sot", "cold_sot", "artifacts"):
            if key in result and isinstance(result[key], dict):
                aggregated.setdefault(key, {}).update(result[key])

        if isinstance(result.get("messages"), list):
            aggregated.setdefault("messages", []).extend(result["messages"])

        # Keep any additional scalar notes/errors for visibility
        for key, value in result.items():
            if key in {"hot_sot", "cold_sot", "artifacts", "messages"}:
                continue
            # First-writer wins to avoid clobbering earlier tool outputs
            aggregated.setdefault(key, value)

        return aggregated

    @staticmethod
    def _safe_json_load(data: str) -> dict[str, Any] | None:
        """Best-effort JSON loader that tolerates invalid output."""
        try:
            return json.loads(data)
        except Exception:
            return None

    @staticmethod
    def _validate_tool_usage(response: Any) -> bool:
        """
        Validate that the response contains tool calls.

        This enforces the constraint that tool-capable roles MUST use tools.

        Args:
            response: Response from LLM (likely has tool_calls attribute)

        Returns:
            True if response contains tool calls, False otherwise
        """
        tool_calls = getattr(response, "tool_calls", None)
        return bool(tool_calls) and len(tool_calls) > 0

    def _get_tool_instances_from_specs(self, tool_specs: list[dict]) -> list[Any]:
        """
        Convert OpenAI function format tool specifications to actual tool instances.

        Args:
            tool_specs: List of tools in OpenAI function format

        Returns:
            List of tool instances from the registry
        """
        tool_instances = []
        registry = get_tool_registry()

        for spec in tool_specs:
            # Extract tool name from OpenAI format
            if isinstance(spec, dict) and "function" in spec:
                tool_name = spec["function"]["name"]
            elif isinstance(spec, dict) and "name" in spec:
                tool_name = spec["name"]
            else:
                logger.warning(f"Unrecognized tool spec format: {spec}")
                continue

            # Get tool instance from registry
            tool_instance = registry.get_tool(tool_name)
            if tool_instance:
                tool_instances.append(tool_instance)
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry")

        logger.debug(
            f"Converted {len(tool_specs)} tool specs to {len(tool_instances)} tool instances"
        )
        return tool_instances

    def _get_tool_map_from_specs(self, tool_specs: list[dict]) -> dict[str, Any]:
        """
        Convert OpenAI function format tool specifications to a tool map.

        This builds a dictionary mapping tool names to tool instances for use
        with the ProtocolExecutor (text-based fallback for models without bind_tools).

        Args:
            tool_specs: List of tools in OpenAI function format

        Returns:
            Dictionary mapping tool names to tool instances
        """
        tool_map = {}
        registry = get_tool_registry()

        for spec in tool_specs:
            # Extract tool name from OpenAI format
            if isinstance(spec, dict) and "function" in spec:
                tool_name = spec["function"]["name"]
            elif isinstance(spec, dict) and "name" in spec:
                tool_name = spec["name"]
            else:
                logger.warning(f"Unrecognized tool spec format: {spec}")
                continue

            # Get tool instance from registry
            tool_instance = registry.get_tool(tool_name)
            if tool_instance:
                tool_map[tool_name] = tool_instance
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry for tool map")

        logger.debug(f"Built tool map with {len(tool_map)} tools: {list(tool_map.keys())}")
        return tool_map

    def _create_synthetic_tool_response(self, role: Any, state: StudioState) -> Any:
        """
        Create a synthetic tool response as a fallback when LLM fails to generate proper tool calls.

        This is a last resort to maintain protocol communication when tool calling fails.

        Args:
            role: The role that failed to generate tool calls
            state: Current studio state

        Returns:
            A synthetic response with a basic protocol message tool call
        """
        from langchain_core.messages import AIMessage

        logger.warning(f"Creating synthetic tool response for {role.id}")

        # Create a minimal protocol message
        synthetic_tool_call = {
            "id": f"synthetic_{role.id}_{datetime.utcnow().timestamp()}",
            "type": "function",
            "function": {
                "name": "send_protocol_message",
                "arguments": json.dumps(
                    {
                        "receiver": SHOWRUNNER,
                        "intent": "error.tool_generation",
                        "content": f"Role {role.id} failed to generate proper tool calls. This is a synthetic fallback message.",
                        "payload": {
                            "role": role.id,
                            "reason": "tool_generation_failure",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        },
                    }
                ),
            },
        }

        # Create AIMessage with tool_calls attribute
        return AIMessage(
            content=f"[Synthetic response for {role.id} due to tool generation failure]",
            tool_calls=[synthetic_tool_call],
        )

    def _execute_tool_loop(
        self,
        llm: Any,
        response: Any,
        role: RoleProfile,
        role_tools: list,
        state: StudioState,
        original_prompt: str,
        max_iterations: int = 5,
    ) -> tuple[str, dict[str, Any], bool]:
        """
        Execute tool calls in a loop until the LLM produces a final response.

        This implements a simple ReAct-style agent loop:
        1. Check if LLM response contains tool calls
        2. If yes: execute tools, trace results, feed back to LLM
        3. If no: return the final text response
        4. Repeat until no more tool calls or max iterations reached

        Args:
            llm: The LLM instance (potentially with tools bound)
            response: Initial LLM response
            role: The role profile
            role_tools: List of tools available to the role
            state: Current state for tool execution context
            original_prompt: The original prompt for context
            max_iterations: Max tool call iterations (safety limit)

        Returns:
            Final text response from the LLM
        """
        from langchain_core.messages import HumanMessage, ToolMessage

        # Build tool lookup for execution
        tool_map = {}
        for tool in role_tools:
            if hasattr(tool, "_base_tool"):
                base_tool = tool._base_tool
                tool_map[tool.name] = base_tool
                # Also map by tool's own name in case it differs
                if hasattr(base_tool, "name"):
                    tool_map[base_tool.name] = base_tool
            elif hasattr(tool, "name"):
                tool_map[tool.name] = tool

        tu_id = state.get("tu_id", "unknown")
        aggregated_updates: dict[str, Any] = {}
        messages = [HumanMessage(content=original_prompt)]
        iteration = 0
        any_tool_calls = False

        while iteration < max_iterations:
            # Check if response has tool calls
            tool_calls = getattr(response, "tool_calls", None)

            if not tool_calls:
                # No tool calls - return final content
                content = response.content if hasattr(response, "content") else str(response)
                if isinstance(content, list):
                    content = "".join(str(part) for part in content)
                return content, {k: v for k, v in aggregated_updates.items() if v}, any_tool_calls

            # Process tool calls
            messages.append(response)  # Add AI message with tool calls

            for tool_call in tool_calls:
                any_tool_calls = True
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        pass
                tool_id = tool_call.get("id", f"call_{iteration}")

                # Trace tool call
                self._trace_tool_event(
                    role.id,
                    "tool_call",
                    {"tool_name": tool_name, "input": tool_args},
                    tu_id,
                )

                # Execute tool
                result = None
                success = True
                try:
                    if tool_name in tool_map:
                        tool_instance = tool_map[tool_name]

                        payload = dict(tool_args)
                        args_schema = getattr(tool_instance, "args_schema", None)
                        model_fields = (
                            getattr(args_schema, "model_fields", {}) if args_schema else {}
                        )
                        if "state" in model_fields:
                            payload["state"] = state
                        if "role_id" in model_fields:
                            payload["role_id"] = role.id

                        # Fallback injection for common stateful tools when schema hints fail
                        needs_state = {
                            "read_hot_sot",
                            "write_hot_sot",
                            "read_cold_sot",
                            "write_cold_sot",
                            "create_snapshot",
                            "update_tu",
                            "trigger_gatecheck",
                            "send_protocol_message",
                            "send_protocol_envelope",
                        }
                        tname = getattr(tool_instance, "name", tool_name)
                        if tname in needs_state:
                            payload["state"] = state
                            payload["role_id"] = role.id
                            if tname == "send_protocol_message":
                                payload.setdefault("payload", {})

                        # Fallback: inspect _run signature for injectable params even if
                        # args_schema doesn't declare them (e.g., state tools).
                        try:
                            import inspect

                            run_sig = inspect.signature(tool_instance._run)  # type: ignore[attr-defined]
                            if "state" in run_sig.parameters and "state" not in payload:
                                payload["state"] = state
                            if "role_id" in run_sig.parameters and "role_id" not in payload:
                                payload["role_id"] = role.id
                        except Exception:
                            pass

                        try:
                            result = tool_instance._run(**payload)  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                result = tool_instance.invoke(payload)
                            except TypeError:
                                # Tool doesn't accept injected args, fallback to raw
                                result = tool_instance.invoke(tool_args)
                    else:
                        result = (
                            f"Tool '{tool_name}' not found in available tools: "
                            f"{list(tool_map.keys())}"
                        )
                        success = False
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"
                    success = False
                    logger.error(f"Tool {tool_name} failed: {e}")

                aggregated_updates = self._merge_tool_result(aggregated_updates, result)

                # Trace tool result
                self._trace_tool_event(
                    role.id,
                    "tool_result",
                    {"tool_name": tool_name, "result": str(result)[:1000], "success": success},
                    tu_id,
                )

                # Add tool result to messages
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            # Continue the conversation with tool results
            logger.info(
                f"Role {role.id} iteration {iteration + 1}: processed {len(tool_calls)} tool calls"
            )
            response = llm.invoke(messages)
            iteration += 1

        # Max iterations reached - return whatever we have
        logger.warning(f"Role {role.id} reached max tool iterations ({max_iterations})")
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        return content, {k: v for k, v in aggregated_updates.items() if v}, any_tool_calls

    def create_role_node(self, role_id: str) -> Callable[[StudioState], dict[str, Any]]:
        """
        Create complete Runnable node for StateGraph.

        This is the main entry point used by GraphFactory.

        Returns a function with signature:
        def role_node(state: StudioState) -> dict[str, Any]:
            # Execute role logic
            # Return partial state update (only changed fields)

        This follows LangGraph's recommended pattern for avoiding concurrent
        update conflicts: nodes return only the fields they want to change,
        not the entire state dict.

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            Runnable node function compatible with LangGraph StateGraph

        Note:
            LangSmith tracing is available via:
            1. Global control_plane.run() trace via @traceable decorator
            2. Message bus routing via route_by_envelope() trace
            3. Individual executor tracing via bind_tools_execute() trace
            The role_node itself is wrapped by control_plane._wrap_node_with_envelope()
            which handles envelope-based tracing for LangSmith observability.
        """
        # Load role once
        role = self.load_role(role_id)
        # Tools are now loaded dynamically by RuntimeContextAssembler based on capabilities

        async def role_node(state: StudioState) -> dict[str, Any]:
            """Execute role asynchronously and update state.

            Returns:
                Partial state update dict (only changed fields)
            """
            # 1. Check dormancy
            if not self.should_execute_role(role, state):
                logger.debug(f"Role {role.id} is dormant, skipping execution")
                return {}  # Return empty dict = no state changes

            try:
                # 2. Send "role_started" progress indicator (live feedback)
                if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                    if self.state_manager._trace_handler:
                        try:
                            start_message = {
                                "sender": role.id,
                                "receiver": "system",
                                "intent": "role_started",
                                "payload": {
                                    "role_name": role.name,
                                },
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "envelope": {
                                    "tu_id": state["tu_id"],
                                },
                            }
                            self.state_manager._trace_handler.trace_message(start_message)
                        except Exception as e:
                            logger.warning(f"Trace handler error: {e}")

                # 3. Assemble context using RuntimeContextAssembler
                context = self.assemble_role_context(role.id, state)
                prompt = context["prompt"]
                assembler_tools = context["tools"]

                logger.debug(
                    f"Role {role.id} assembled context with {len(assembler_tools)} tools from assembler"
                )

                # 4. Invoke LLM or tools based on role type
                llm_config = self.select_llm(role)

                if llm_config:
                    provider = llm_config.get("provider")

                    # Create LLM client via ProviderManager (cached)
                    try:
                        llm = self.provider_manager.create_llm_client(
                            provider=provider,
                            model=llm_config["model"],
                            temperature=llm_config["temperature"],
                            max_tokens=llm_config["max_tokens"],
                        )

                        # Primary: Use bind_tools for native structured tool support
                        # Fallback: Use text-based protocol executor for universal compatibility
                        # This works across ALL models (Llama, Qwen, GPT, Claude)
                        # by using explicit Action/Action Input instructions when bind_tools unavailable

                        if assembler_tools:
                            # Build tool map for ProtocolExecutor (text-based fallback)
                            tool_map = self._get_tool_map_from_specs(assembler_tools)

                            # Create role-specific user prompt
                            if role.id == "showrunner":
                                user_prompt = (
                                    f"You are executing as {role.name} for TU: {state.get('tu_id', 'unknown')}.\n\n"
                                    "Start by reading `customer_directives` from hot_sot using the read_hot_sot tool. "
                                    "Then route your interpretation via protocol messages (send_protocol_message) "
                                    "to wake/assign the appropriate roles."
                                )
                            else:
                                tu_id = state.get("tu_id", "unknown")

                                # Extract task context from incoming protocol messages
                                # Look for messages addressed to this role with task details
                                task_context = self._extract_task_context_for_role(role.id, state)

                                if task_context:
                                    user_prompt = (
                                        f"You are executing as {role.name} for TU: {tu_id}.\n\n"
                                        f"## Your Assigned Task\n{task_context}\n\n"
                                        "Complete this task using the available tools. "
                                        "When done, send a protocol message to report completion."
                                    )
                                else:
                                    user_prompt = (
                                        f"You are executing as {role.name} for TU: {tu_id}.\n\n"
                                        "Complete your assigned task using the available tools. "
                                        "When done, send a protocol message to the receiver."
                                    )

                            # Execute with protocol executor (text-based fallback when bind_tools unavailable)
                            logger.info(f"Executing {role.id} with protocol executor ({provider})")

                            # Send role_prompt BEFORE LLM invocation so user can abort if wrong
                            if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                                if self.state_manager._trace_handler:
                                    try:
                                        prompt_message = {
                                            "sender": role.id,
                                            "receiver": "system",
                                            "intent": "role_prompt",
                                            "payload": {
                                                "role_name": role.name,
                                                "prompt": prompt,
                                                "user_prompt": user_prompt,
                                                "tools": assembler_tools,
                                            },
                                            "timestamp": datetime.now(UTC).isoformat(),
                                        }
                                        self.state_manager._trace_handler.trace_message(
                                            prompt_message
                                        )
                                    except Exception as e:
                                        logger.debug(f"Failed to trace prompt: {e}")

                            # Create trace callback if trace handler is available
                            trace_callback = None
                            if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                                if self.state_manager._trace_handler:

                                    def make_trace_cb(handler, tu_id):
                                        def trace_cb(intent, payload):
                                            message = {
                                                "sender": payload.get("role_id", "react"),
                                                "receiver": "system",
                                                "intent": intent,
                                                "payload": payload,
                                                "timestamp": datetime.now(UTC).isoformat(),
                                                "envelope": {"tu_id": tu_id},
                                            }
                                            handler.trace_message(message)

                                        return trace_cb

                                    trace_callback = make_trace_cb(
                                        self.state_manager._trace_handler,
                                        state.get("tu_id", "unknown"),
                                    )

                            # Retrieve prior conversation for short-term memory
                            prior_conversation = state.get("role_conversations", {}).get(
                                role.id, ""
                            )

                            # Select executor based on model's bind_tools support
                            from questfoundry.runtime.core.bind_tools_executor import (
                                BindToolsExecutor,
                                select_executor,
                            )

                            model_name = llm_config.get("model", "")
                            ExecutorClass = select_executor(model_name)

                            if ExecutorClass == BindToolsExecutor:
                                # Use native bind_tools for structured tool calling
                                logger.info(
                                    f"Using BindToolsExecutor for {role.id} ({model_name})"
                                )
                                executor = BindToolsExecutor(
                                    llm=llm,
                                    tools=list(tool_map.values()),
                                    role_id=role.id,
                                    trace_handler=trace_callback,
                                )
                                exec_result = await executor.execute(
                                    system_prompt=prompt,
                                    user_prompt=user_prompt,
                                    prior_conversation=prior_conversation,
                                )
                            else:
                                # Fallback to text-based protocol executor
                                logger.info(
                                    f"Using ProtocolExecutor for {role.id} ({model_name})"
                                )
                                executor = ProtocolExecutor(
                                    tool_map=tool_map,
                                    state=state,
                                    role_id=role.id,
                                    trace_callback=trace_callback,
                                )
                                exec_result = await executor.execute(
                                    llm=llm,
                                    system_prompt=prompt,
                                    user_prompt=user_prompt,
                                    tools=assembler_tools,
                                    prior_conversation=prior_conversation,
                                )

                            # Process executor result
                            # NOTE: State updates (hot_sot, cold_sot) happen via tools during
                            # execution, not via parsing a "final answer" JSON blob.
                            if exec_result.success:
                                logger.info(
                                    f"Execution succeeded for {role.id} "
                                    f"({exec_result.iterations} iterations, "
                                    f"{len(exec_result.tool_results)} tool calls)"
                                )
                                # Messages ARE the output - no final_answer to parse
                                result = json.dumps(
                                    {
                                        "messages": exec_result.messages,
                                        "tool_results": exec_result.tool_results,
                                    }
                                )
                                tool_updates = {
                                    "messages": exec_result.messages,
                                    "tool_results": exec_result.tool_results,
                                    "role_conversations": {role.id: exec_result.work_summary},
                                }
                            else:
                                logger.error(f"Execution failed for {role.id}: {exec_result.error}")
                                result = json.dumps(
                                    {
                                        "error": exec_result.error,
                                        "iterations": exec_result.iterations,
                                        "failure_count": exec_result.failure_count,
                                        "tool_results": exec_result.tool_results,
                                    }
                                )
                                # Still store work summary even on failure for debugging
                                tool_updates = {
                                    "role_conversations": {role.id: exec_result.work_summary},
                                }

                        else:
                            # No tools - simple LLM invocation (async)
                            logger.info(f"No tools for {role.id}, invoking LLM directly")
                            response = await llm.ainvoke(prompt)
                            result = (
                                response.content if hasattr(response, "content") else str(response)
                            )
                            tool_updates = {}

                        # Send role_completed with extracted insight and prompt context
                        if self.state_manager and hasattr(self.state_manager, "_trace_handler"):
                            if self.state_manager._trace_handler:
                                try:
                                    # Include FULL result for trace (no truncation)
                                    # Critical: Roles need to see complete messages to
                                    # self-stabilize. Also include the rendered prompt
                                    # so we can debug cross-role communication and
                                    # prompt construction.
                                    completed_message = {
                                        "sender": role.id,
                                        "receiver": "system",
                                        "intent": "role_completed",
                                        "payload": {
                                            "role_name": role.name,
                                            "insight": result,  # Full result, not truncated
                                            "length": (
                                                len(result) if hasattr(result, "__len__") else 0
                                            ),
                                            # NOTE: This is the rendered role prompt *before*
                                            # JSON-format instructions are appended.
                                            # TODO: Ideally we'd also capture a structured view
                                            # of the prompt components (system vs. user vs. tools)
                                            # rather than a single flattened string.
                                            "prompt": prompt,
                                        },
                                        "timestamp": datetime.utcnow().isoformat() + "Z",
                                        "envelope": {
                                            "tu_id": state["tu_id"],
                                        },
                                    }
                                    self.state_manager._trace_handler.trace_message(
                                        completed_message
                                    )
                                except Exception as e:
                                    logger.warning(f"Trace handler error: {e}")

                    except Exception as e:
                        # Gracefully handle provider-level failures (e.g. 404, rate limit)
                        logger.error(f"LLM invocation failed for {role.id}: {e}")
                        # Mark provider as unavailable for subsequent selections in this run
                        if provider:
                            self.provider_manager.mark_unavailable(provider, reason=str(e))
                        # Re-raise so the caller can decide whether to abort or retry
                        raise
                else:
                    # Service type - attempt to run declared tools (metadata only for now)
                    tool_updates = {}
                    result = {
                        "tools_declared": [],  # Tools now handled by assembler
                        "note": (
                            "Service role executed without LLM; "
                            "tool invocations are not yet orchestrated."
                        ),
                    }
                    logger.info(f"Service-type role {role.id} returning tool metadata")

                # 4. Extract artifacts from LLM output
                result_str = result if isinstance(result, str) else json.dumps(result)
                extracted_artifacts = self.extract_artifacts(role, result_str, state["tu_id"])

                # Merge tool-driven updates (messages / sot updates)
                parsed_result = self._safe_json_load(result_str)
                log_free_text = None if parsed_result is not None else result_str

                merged_hot_sot: dict[str, Any] = {}
                merged_cold_sot: dict[str, Any] = {}
                merged_artifacts: dict[str, Any] = {}

                # Note: ProtocolExecutor (text-based) handles all tool calling
                # when bind_tools is unavailable, via Action/Action Input format

                # hot_sot / cold_sot / artifacts from extraction + tool outputs + parsed result
                for source in (
                    extracted_artifacts,
                    tool_updates if "tool_updates" in locals() else {},
                    parsed_result or {},
                ):
                    if not isinstance(source, dict):
                        continue
                    if isinstance(source.get("hot_sot"), dict):
                        merged_hot_sot.update(source["hot_sot"])
                    if isinstance(source.get("cold_sot"), dict):
                        merged_cold_sot.update(source["cold_sot"])
                    if isinstance(source.get("artifacts"), dict):
                        merged_artifacts.update(source["artifacts"])

                # 5. Create protocol message with proper envelope for mesh routing
                # The receiver field drives Control Plane routing:
                # - Specific role ID for direct peer-to-peer
                # - "showrunner" / "SR" to report back to coordinator
                # - "*" for broadcast
                # - "__terminate__" to end graph execution
                #
                # Default behavior: report back to showrunner (coordinator pattern)
                # Roles can override this by including routing info in their output
                messages_out: list[dict[str, Any]] = []

                # Messages from protocol executor tool execution
                if isinstance(tool_updates, dict) and isinstance(
                    tool_updates.get("messages"), list
                ):
                    messages_out.extend(tool_updates["messages"])

                # Messages embedded in the LLM JSON (common when provider lacks tool-calling)
                if isinstance(parsed_result, dict) and isinstance(
                    parsed_result.get("messages"), list
                ):
                    messages_out.extend(parsed_result["messages"])

                # Route free text (non-JSON) to logging for visibility
                if log_free_text:
                    messages_out.append(
                        {
                            "sender": role.id,
                            "receiver": "system",
                            "intent": "log",
                            "payload": {"text": log_free_text[:2000]},
                        }
                    )

                # If no messages were produced, route a protocol update to avoid loops
                if not messages_out:
                    note_text = result if isinstance(result, str) else "no_routed_messages"
                    messages_out.append(
                        {
                            "sender": role.id,
                            "receiver": "showrunner",
                            "intent": "tu.update",
                            "payload": {"note": note_text},
                        }
                    )

                def _normalize_receiver(msg: dict[str, Any]) -> None:
                    # Handle "recipient" alias (LLMs often use this instead of "receiver")
                    if "recipient" in msg and "receiver" not in msg:
                        msg["receiver"] = msg.pop("recipient")
                    recv = msg.get("receiver")
                    if isinstance(recv, dict):
                        msg["receiver"] = (
                            recv.get("role") or recv.get("id") or recv.get("name") or "showrunner"
                        )

                for msg in messages_out:
                    _normalize_receiver(msg)

                # Normalize envelopes and defaults
                for msg in messages_out:
                    msg.setdefault("sender", role.id)
                    msg.setdefault("receiver", "showrunner")
                    if isinstance(msg.get("receiver"), dict):
                        recv_dict = msg["receiver"] or {}
                        msg["receiver"] = (
                            recv_dict.get("role")
                            or recv_dict.get("id")
                            or recv_dict.get("name")
                            or "showrunner"
                        )
                    envelope = msg.get("envelope", {}) or {}
                    envelope.setdefault("tu_id", state.get("tu_id", ""))
                    envelope.setdefault("snapshot_ref", state.get("snapshot_ref"))
                    msg["envelope"] = envelope
                    msg.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

                # 6. Tracing is handled by control_plane._wrap_node_with_envelope
                # Do NOT trace here to avoid duplicate messages in trace log

                # 7. Return extracted artifacts mapped to spec-defined state keys
                # Note: Reducers (Annotated types) handle merging artifacts and messages
                # Don't manually merge - just return the new values
                # Unpack extracted_artifacts to put hot_sot and artifacts at top level
                state_update: dict[str, Any] = {
                    "messages": messages_out,
                }

                if merged_hot_sot:
                    state_update["hot_sot"] = merged_hot_sot
                if merged_cold_sot:
                    state_update["cold_sot"] = merged_cold_sot
                if merged_artifacts:
                    state_update["artifacts"] = merged_artifacts

                # Short-term memory: persist role conversation history
                if isinstance(tool_updates, dict) and "role_conversations" in tool_updates:
                    state_update["role_conversations"] = tool_updates["role_conversations"]

                return state_update

            except Exception as e:
                logger.error(f"Error executing role {role.id}: {e}")
                # Return partial update with error info and terminate the graph to avoid ping-pong
                error_message = {
                    "sender": role.id,
                    "receiver": "__terminate__",
                    "intent": "error",
                    "payload": {"error": str(e)},
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "envelope": {"tu_id": state.get("tu_id", "")},
                }
                return {
                    "error": str(e),
                    "retry_count": state.get("retry_count", 0) + 1,
                    "messages": [error_message],
                }

        return role_node

    def create_multi_role_node(
        self, sub_nodes: list[dict[str, Any]], node_id: str
    ) -> Callable[[StudioState], dict[str, Any]]:
        """
        Create a multi-role parallel execution node.

        For nodes with role="Multi" and parallel_execution=true, this creates
        a wrapper that executes multiple roles in parallel and merges their results.

        Args:
            sub_nodes: List of sub-node definitions, each with 'role' and 'task'
            node_id: Node identifier for logging

        Returns:
            Runnable node function that executes roles in parallel

        Example sub_nodes structure:
            [
                {"role": "Lore Weaver", "task": "Review narrative hooks..."},
                {"role": "Plotwright", "task": "Review structural impact..."}
            ]
        """
        import concurrent.futures

        # Load all role nodes upfront
        role_execution_pairs = []
        for sub_node in sub_nodes:
            role_name = sub_node.get("role", "")
            task = sub_node.get("task", "")

            # Convert role name to role_id (lowercase, replace spaces with underscores)
            role_id = role_name.lower().replace(" ", "_")

            try:
                # Create role node for this sub-role
                role_node = self.create_role_node(role_id)
                role_execution_pairs.append((role_name, role_id, role_node, task))
                logger.debug(f"Loaded role for parallel execution: {role_name} ({role_id})")
            except Exception as e:
                logger.warning(
                    f"Failed to load role {role_name} ({role_id}) for parallel node: {e}"
                )
                # Continue with other roles even if one fails to load

        if not role_execution_pairs:
            logger.error(f"No valid roles loaded for multi-role node {node_id}")

            # Return a placeholder that logs the error
            def error_node(state: StudioState) -> dict[str, Any]:
                return {"error": f"Multi-role node {node_id} has no valid roles"}

            return error_node

        def multi_role_node(state: StudioState) -> dict[str, Any]:
            """Execute multiple roles in parallel and merge results."""
            logger.info(
                "[bold cyan]Executing parallel node:[/bold cyan] %s with %d roles",
                node_id,
                len(role_execution_pairs),
            )

            # Execute all roles in parallel using ThreadPoolExecutor
            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(role_execution_pairs)
            ) as executor:
                # Submit all role executions
                futures = {}
                for role_name, role_id, role_node, task in role_execution_pairs:
                    logger.debug(f"Submitting parallel task for {role_name}: {task[:80]}...")
                    future = executor.submit(role_node, state)
                    futures[future] = (role_name, role_id)

                # Collect results as they complete
                for future in concurrent.futures.as_completed(
                    futures, timeout=600
                ):  # 10 min total timeout
                    role_name, role_id = futures[future]
                    try:
                        result = future.result(timeout=300)  # 5 min per role
                        logger.info(f"[green]✓[/green] {role_name} completed in parallel")
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Role {role_name} timed out in parallel execution")
                        results.append({"error": f"Role {role_name} timed out"})
                    except Exception as e:
                        logger.error(f"Role {role_name} failed in parallel execution: {e}")
                        results.append({"error": f"Role {role_name} failed: {str(e)}"})

            # Merge all results into single state update
            merged_artifacts = {}
            merged_messages = []
            errors = []

            for result in results:
                # Merge artifacts
                if "artifacts" in result and isinstance(result["artifacts"], dict):
                    merged_artifacts.update(result["artifacts"])

                # Merge messages
                if "messages" in result and isinstance(result["messages"], list):
                    merged_messages.extend(result["messages"])

                # Collect errors
                if "error" in result:
                    errors.append(result["error"])

            # Build final merged state update
            merged_state = {"artifacts": merged_artifacts, "messages": merged_messages}

            # If any errors occurred, include them in the merged state
            if errors:
                merged_state["error"] = "; ".join(errors)
                logger.warning(
                    f"Parallel node {node_id} completed with errors: {merged_state['error']}"
                )
            else:
                logger.info(
                    f"[bold green]Parallel node {node_id} completed successfully[/bold green]"
                )

            return merged_state

        return multi_role_node
