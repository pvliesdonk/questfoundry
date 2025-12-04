"""
RuntimeContextAssembler - Dynamically assembles agent prompts from YAML definitions.

This class builds the 6-layer prompt structure:
1. IDENTITY - Role identity and charter reference
2. PROTOCOL - Valid protocol intents and communication rules
3. STATE - Current context (TU, loop, state snapshot)
4. STATE MANAGEMENT - How to read/write state (tools, keys, workflow)
5. MISSION - Success criteria and task guidance
6. INTERFACE - Tool usage and protocol enforcement

Plus STUDIO KNOWLEDGE layer for Showrunner (between STATE MANAGEMENT and MISSION).

Based on: Layer 5 compiled definitions -> Layer 6 runtime prompt orchestration
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from questfoundry.runtime.core.capability_mapper import CapabilityMapper
from questfoundry.runtime.core.schema_registry import DEFINITIONS_ROOT
from questfoundry.runtime.models.state import StudioState
from questfoundry.runtime.plugins.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


class RuntimeContextAssembler:
    """
    Assembles agent prompts dynamically from YAML definitions.

    This class reads:
    - Role profiles (spec/05-definitions/roles/*.yaml)
    - Loop patterns (spec/05-definitions/loops/*.yaml)
    - Protocol definitions (spec/05-definitions/protocol.yaml)
    - Capability mappings (via CapabilityMapper)

    And produces:
    - Structured prompts with 5 layers
    - Tool configurations for LLM API binding
    """

    def __init__(
        self,
        definitions_path: str | Path | None = None,
        capability_mapper: CapabilityMapper | None = None,
    ):
        """
        Initialize runtime context assembler.

        Args:
            definitions_path: Path to spec/05-definitions (default: auto-detect)
            capability_mapper: CapabilityMapper instance (creates new if not provided)
        """
        if definitions_path is None:
            # Use the same spec finding mechanism as SchemaRegistry
            if DEFINITIONS_ROOT:
                definitions_path = DEFINITIONS_ROOT
            else:
                # Fallback to relative path if not found
                definitions_path = Path("spec/05-definitions")

        self.definitions_path = Path(definitions_path)
        self.capability_mapper = capability_mapper or CapabilityMapper()

        # Cached YAML data
        self._role_cache: dict[str, dict[str, Any]] = {}
        self._loop_cache: dict[str, dict[str, Any]] = {}
        self._protocol_def: dict[str, Any] | None = None

        # Load protocol once
        self._load_protocol()

    def _load_protocol(self) -> None:
        """Load protocol definitions from protocol.yaml."""
        protocol_path = self.definitions_path / "protocol.yaml"

        try:
            if not protocol_path.exists():
                logger.warning(f"Protocol file not found: {protocol_path}")
                return

            with open(protocol_path) as f:
                self._protocol_def = yaml.safe_load(f)

            logger.info("Loaded protocol definitions")

        except Exception as e:
            logger.error(f"Failed to load protocol: {e}")

    def _load_role(self, role_id: str) -> dict[str, Any]:
        """
        Load role definition from YAML.

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            Role definition dictionary

        Raises:
            FileNotFoundError: If role YAML doesn't exist
        """
        if role_id in self._role_cache:
            return self._role_cache[role_id]

        role_path = self.definitions_path / "roles" / f"{role_id}.yaml"

        if not role_path.exists():
            raise FileNotFoundError(f"Role definition not found: {role_path}")

        with open(role_path) as f:
            role_def = yaml.safe_load(f)

        self._role_cache[role_id] = role_def
        logger.debug(f"Loaded role definition: {role_id}")

        return role_def

    def _load_loop(self, loop_id: str) -> dict[str, Any]:
        """
        Load loop pattern from YAML.

        Args:
            loop_id: Loop identifier (e.g., "story_spark")

        Returns:
            Loop definition dictionary

        Raises:
            FileNotFoundError: If loop YAML doesn't exist
        """
        if loop_id in self._loop_cache:
            return self._loop_cache[loop_id]

        loop_path = self.definitions_path / "loops" / f"{loop_id}.yaml"

        if not loop_path.exists():
            raise FileNotFoundError(f"Loop definition not found: {loop_path}")

        with open(loop_path) as f:
            loop_def = yaml.safe_load(f)

        self._loop_cache[loop_id] = loop_def
        logger.debug(f"Loaded loop definition: {loop_id}")

        return loop_def

    def _build_identity_layer(self, role_def: dict[str, Any]) -> str:
        """
        Build IDENTITY layer from role definition.

        Args:
            role_def: Role definition dictionary

        Returns:
            Formatted identity block
        """
        identity = role_def.get("identity", {})
        prompt_content = role_def.get("prompt_content", {})

        name = identity.get("name", "Unknown Role")
        abbreviation = identity.get("abbreviation", "??")
        role_type = identity.get("role_type", "reasoning_agent")
        core_mandate = prompt_content.get("core_mandate", "")

        identity_block = f"""# IDENTITY

You are **{name}** ({abbreviation}), a {role_type} in the QuestFoundry production system.

## Core Mandate
{core_mandate}

## Operating Principles
"""

        # Add operating principles
        principles = prompt_content.get("operating_principles", [])
        for principle in principles:
            name_p = principle.get("name", "")
            desc = principle.get("description", "")
            identity_block += f"- **{name_p}**: {desc}\n"

        # Add anti-patterns
        anti_patterns = prompt_content.get("anti_patterns", [])
        if anti_patterns:
            identity_block += "\n## Anti-Patterns (Avoid These)\n"
            for pattern in anti_patterns:
                name_ap = pattern.get("name", "")
                desc = pattern.get("description", "")
                identity_block += f"- **{name_ap}**: {desc}\n"

        # Add heuristics if present
        heuristics = prompt_content.get("heuristics", [])
        if heuristics:
            identity_block += "\n## Heuristics\n"
            for heuristic in heuristics:
                name_h = heuristic.get("name", "")
                desc = heuristic.get("description", "")
                examples = heuristic.get("examples", [])
                identity_block += f"- **{name_h}**: {desc}\n"
                if examples:
                    for example in examples:
                        identity_block += f"  - Example: {example}\n"

        return identity_block

    def _build_protocol_layer(self, role_def: dict[str, Any], state: StudioState) -> str:
        """
        Build PROTOCOL layer from role protocol permissions.

        Args:
            role_def: Role definition dictionary
            state: Current studio state

        Returns:
            Formatted protocol block
        """
        protocol = role_def.get("protocol", {})
        intents = protocol.get("intents", {})
        can_send = intents.get("can_send", [])
        can_receive = intents.get("can_receive", [])

        protocol_block = """# PROTOCOL

You communicate through the QuestFoundry protocol system using structured messages.

## Protocol Intents You Can Send
"""

        # Filter intents that this role can send
        protocol_intents = self._protocol_def.get("intents", {}) if self._protocol_def else {}

        for intent in can_send:
            intent_def = protocol_intents.get(intent, {})
            desc = intent_def.get("description", "")
            domain = intent_def.get("domain", "")
            protocol_block += f"- **{intent}** ({domain}): {desc}\n"

        protocol_block += "\n## Protocol Intents You Can Receive\n"

        for intent in can_receive:
            intent_def = protocol_intents.get(intent, {})
            desc = intent_def.get("description", "")
            domain = intent_def.get("domain", "")
            protocol_block += f"- **{intent}** ({domain}): {desc}\n"

        # Add envelope constraints
        envelope_defaults = protocol.get("envelope_defaults", {})
        if envelope_defaults:
            protocol_block += "\n## Envelope Defaults\n"
            safety = envelope_defaults.get("safety", {})
            context = envelope_defaults.get("context", {})

            if safety:
                protocol_block += f"- Player Safe: {safety.get('player_safe', False)}\n"
                protocol_block += f"- Spoilers: {safety.get('spoilers', 'allowed')}\n"

            if context:
                protocol_block += f"- Hot/Cold: {context.get('hot_cold', 'hot')}\n"

        # Add lifecycle permissions
        lifecycles = protocol.get("lifecycles", {})
        if lifecycles:
            protocol_block += "\n## Lifecycle Permissions\n"

            hook = lifecycles.get("hook", {})
            if hook.get("can_create"):
                protocol_block += "- You can create Hook Cards\n"

            tu = lifecycles.get("tu", {})
            if tu.get("can_open"):
                protocol_block += "- You can open Trace Units (TUs)\n"
            if tu.get("can_close"):
                protocol_block += "- You can close Trace Units (TUs)\n"

            gate = lifecycles.get("gate", {})
            if gate.get("can_evaluate"):
                protocol_block += "- You can evaluate quality gates\n"

        return protocol_block

    def _build_state_layer(self, state: StudioState, role_id: str | None = None) -> str:
        """
        Build STATE layer from current runtime state.

        Args:
            state: Current studio state
            role_id: Optional role identifier for execution history

        Returns:
            Formatted state block
        """
        state_block = """# STATE

Current execution context:
"""

        # Add TU context if available
        tu_id = getattr(state, "tu_id", None)
        tu_brief = getattr(state, "tu_brief", None)

        if tu_id:
            state_block += f"\n## Current Trace Unit\n- TU ID: {tu_id}\n"
            if tu_brief:
                state_block += f"- Brief: {tu_brief}\n"

        # Add loop context if available
        loop_id = getattr(state, "loop_id", None)
        node_id = getattr(state, "node_id", None)

        if loop_id:
            state_block += f"\n## Current Loop\n- Loop: {loop_id}\n"
            if node_id:
                state_block += f"- Node: {node_id}\n"

        # Add Hot/Cold context
        hot_cold = getattr(state, "hot_cold", "hot")
        state_block += f"\n## State Context\n- Hot/Cold: {hot_cold}\n"

        # Add snapshot context if available
        snapshot_id = getattr(state, "snapshot_id", None)
        if snapshot_id:
            state_block += f"- Snapshot: {snapshot_id}\n"

        # Add relevant artifacts from state
        hot_sot = getattr(state, "hot_sot", {})
        if hot_sot:
            state_block += "\n## Available Artifacts\n"
            for artifact_type, artifacts in hot_sot.items():
                if isinstance(artifacts, list):
                    count = len(artifacts)
                    state_block += f"- {artifact_type}: {count} item(s)\n"

        # Add execution history for this role (messages it has sent)
        if role_id:
            messages = state.get("messages", [])
            role_messages = [
                m for m in messages
                if str(m.get("sender", "")).lower() == role_id.lower()
            ]
            if role_messages:
                state_block += f"\n## Your Previous Actions (as {role_id})\n"
                state_block += "You have already taken the following actions in this session:\n\n"
                # Show last N messages to avoid context overflow
                # Intentionally set to 5 for context size management
                max_history = 5

                # Preserve escalation messages regardless of history limit
                escalations = [
                    m for m in role_messages
                    if m.get("escalation") or m.get("priority") == "critical"
                ]
                normal = [m for m in role_messages if m not in escalations]

                # Keep all escalations + last N normal messages
                recent_messages = escalations + normal[-max_history:]
                for i, msg in enumerate(recent_messages, 1):
                    intent = msg.get("intent", "unknown")
                    receiver = msg.get("receiver", "unknown")
                    payload = msg.get("payload", {})
                    # Summarize payload content
                    if isinstance(payload, dict):
                        content = payload.get("content", payload.get("description", ""))
                        if len(str(content)) > 100:
                            content = str(content)[:100] + "..."
                    else:
                        content = str(payload)[:100]
                    state_block += f"{i}. **{intent}** → {receiver}"
                    if content:
                        state_block += f": {content}"
                    state_block += "\n"
                if len(role_messages) > max_history:
                    state_block += f"\n_(showing {max_history} of {len(role_messages)} actions)_\n"
                state_block += "\n**Do not repeat actions you have already taken.**\n"

        return state_block

    def _build_state_management_layer(self, role_def: dict[str, Any]) -> str:
        """
        Build STATE MANAGEMENT layer explaining how to read/write state.

        This layer teaches agents:
        1. HOW to read state (use tools, wait for results)
        2. HOW to write state (use tools with correct keys)
        3. WHICH keys map to which artifacts (based on role interface)

        Args:
            role_def: Role definition dictionary

        Returns:
            Formatted state management block
        """
        block = """# STATE MANAGEMENT

**CRITICAL: This section explains HOW to read and write project state.**

## Reading State

To access project data, you MUST use the state tools and WAIT for results:

1. **Call the appropriate read tool** (`read_hot_sot` or `read_cold_sot`)
2. **Wait for the Observation** - the tool will return the data
3. **Then use the data** in your reasoning

**DO NOT** write fake observations or assume what data contains.
**DO NOT** proceed without reading state when required inputs exist.

"""

        # Extract interface for key mapping
        interface = role_def.get("interface", {})
        inputs = interface.get("inputs", [])
        outputs = interface.get("outputs", [])

        # Build read instructions from inputs
        hot_inputs = []
        cold_inputs = []

        for inp in inputs:
            artifact_type = inp.get("artifact_type", "")
            state_key = inp.get("state_key", "")
            required = inp.get("required", False)

            if state_key.startswith("hot_sot."):
                key = state_key.replace("hot_sot.", "")
                hot_inputs.append((artifact_type, key, required))
            elif state_key.startswith("cold_sot."):
                key = state_key.replace("cold_sot.", "")
                cold_inputs.append((artifact_type, key, required))

        if hot_inputs:
            block += "### Reading from Hot SoT (Living State)\n\n"
            block += "Use `read_hot_sot` tool with these keys:\n\n"
            for artifact, key, required in hot_inputs:
                req_str = " **[REQUIRED]**" if required else ""
                block += f'- `read_hot_sot(key="{key}")` → {artifact}{req_str}\n'
            block += "\n"

        if cold_inputs:
            block += "### Reading from Cold SoT (Finalized State)\n\n"
            block += "Use `read_cold_sot` tool with these keys:\n\n"
            for artifact, key, required in cold_inputs:
                req_str = " **[REQUIRED]**" if required else ""
                block += f'- `read_cold_sot(key="{key}")` → {artifact}{req_str}\n'
            block += "\n"

        # Build write instructions from outputs
        block += """## Writing State

To save your work, you MUST use the `write_hot_sot` tool:

1. **Call `write_hot_sot`** with the correct key and your artifact data
2. **The tool handles merging** - lists append, dicts merge
3. **Wait for confirmation** before considering the write complete

"""

        hot_outputs = []
        for out in outputs:
            artifact_type = out.get("artifact_type", "")
            state_key = out.get("state_key", "")
            merge_strategy = out.get("merge_strategy", "replace")

            if state_key.startswith("hot_sot."):
                key = state_key.replace("hot_sot.", "")
                hot_outputs.append((artifact_type, key, merge_strategy))

        if hot_outputs:
            block += "### Your Output Keys\n\n"
            block += "Use `write_hot_sot` tool with these keys:\n\n"
            for artifact, key, strategy in hot_outputs:
                block += f'- `write_hot_sot(key="{key}", value={{...}})` → {artifact} (merge: {strategy})\n'
            block += "\n"

        return block

    def _build_mission_layer(
        self, loop_id: str | None, node_id: str | None, role_def: dict[str, Any]
    ) -> str:
        """
        Build MISSION layer from loop and node context.

        Args:
            loop_id: Current loop identifier
            node_id: Current node identifier
            role_def: Role definition dictionary

        Returns:
            Formatted mission block
        """
        mission_block = """# MISSION

Your current task:
"""

        # Add loop context if available
        if loop_id:
            try:
                loop_def = self._load_loop(loop_id)
                loop_name = loop_def.get("metadata", {}).get("name", loop_id)
                loop_desc = loop_def.get("metadata", {}).get("description", "")

                mission_block += f"\n## Loop: {loop_name}\n{loop_desc}\n"

                # Find current node in topology
                if node_id:
                    nodes = loop_def.get("topology", {}).get("nodes", [])
                    for node in nodes:
                        if node.get("node_id") == node_id:
                            node_desc = node.get("description", "")
                            mission_block += f"\n## Current Node: {node_id}\n{node_desc}\n"
                            break

                # Add success criteria
                success_criteria = loop_def.get("success_criteria", {})
                custom_checks = success_criteria.get("custom_checks", [])
                if custom_checks:
                    mission_block += "\n## Success Criteria\n"
                    for check in custom_checks:
                        check_name = check.get("name", "")
                        error_msg = check.get("error_message", "")
                        mission_block += f"- **{check_name}**: {error_msg}\n"

            except FileNotFoundError:
                logger.warning(f"Loop definition not found: {loop_id}")

        # Add task guidance from role
        prompt_content = role_def.get("prompt_content", {})
        task_guidance = prompt_content.get("task_guidance", "")
        if task_guidance:
            mission_block += f"\n## Task Guidance\n{task_guidance}\n"

        # Add quality bars owned by this role
        quality_bars_owned = prompt_content.get("quality_bars_owned", [])
        if quality_bars_owned:
            mission_block += "\n## Quality Bars You Own\n"
            for bar in quality_bars_owned:
                mission_block += f"- {bar}\n"

        return mission_block

    def _build_interface_block(self, role_def: dict[str, Any]) -> str:
        """
        Build INTERFACE block emphasizing tool usage and protocol requirements.

        Args:
            role_def: Role definition dictionary

        Returns:
            Formatted interface block
        """
        interface_block = """# INTERFACE

**CRITICAL: Read this section carefully. This defines how you MUST interact.**

## Protocol Communication (MANDATORY)

Protocol messages are the ONLY way to communicate with other roles and the system.

- All outputs MUST be wrapped in protocol envelopes
- Natural language responses WITHOUT protocol envelopes will be IGNORED
- You cannot communicate outside the protocol system

## Tool Usage (MANDATORY)

You MUST use tools to perform any action:

- State changes require state tools
- External operations require capability tools
- Protocol messages require the send_message tool
- DO NOT attempt to perform actions without tools

## Reasoning and Tool Calls

Before calling tools, briefly explain your thinking in natural language. This helps with:
- Debugging your decision-making process
- Understanding why you chose specific tools
- Tracing your reasoning across iterations

Examples of good reasoning:
- "I need to read the current TU brief to understand the task requirements."
- "Based on the state, I see the hooks array is empty, so I should create initial hooks."
- "The validation failed because the required field was missing. Let me fix that."
- "First, I'll read the customer directives. Then I'll create a TU brief based on them."

Keep reasoning concise (1-3 sentences) and focused on:
- What you're about to do and why
- What you learned from reading state
- How you're recovering from errors
- How you're breaking down complex tasks

## Available Tools

Your tools are listed below and bound to your LLM API session. Use them according to
their schemas and descriptions.
"""

        # Add input expectations
        interface = role_def.get("interface", {})
        inputs = interface.get("inputs", [])
        outputs = interface.get("outputs", [])

        if inputs:
            interface_block += "\n## Expected Inputs\n"
            for inp in inputs:
                artifact_type = inp.get("artifact_type", "")
                required = inp.get("required", False)
                state_key = inp.get("state_key", "")
                req_str = "**Required**" if required else "Optional"
                interface_block += f"- {artifact_type} ({req_str}) from {state_key}\n"

        if outputs:
            interface_block += "\n## Expected Outputs\n"
            for out in outputs:
                artifact_type = out.get("artifact_type", "")
                state_key = out.get("state_key", "")
                validation_required = out.get("validation_required", False)
                val_str = " (validated)" if validation_required else ""
                interface_block += f"- {artifact_type} to {state_key}{val_str}\n"

        # Add side effects note
        side_effects = interface.get("side_effects", [])
        if side_effects:
            interface_block += "\n## Side Effects You Can Trigger\n"
            for effect in side_effects:
                interface_block += f"- {effect}\n"

        # Add structured output expectations
        behavior = role_def.get("behavior", {})
        structured_output = behavior.get("structured_output", {})
        if structured_output.get("enabled"):
            schema_ref = structured_output.get("schema_ref", "")
            output_format = structured_output.get("format", "json")
            interface_block += "\n## Structured Output\n"
            interface_block += f"- Format: {output_format}\n"
            interface_block += f"- Schema: {schema_ref}\n"
            interface_block += "- Output MUST conform to schema or will be rejected\n"

        return interface_block

    def _gather_tools(self, role_def: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Gather all tools that should be bound to this role.

        Args:
            role_def: Role definition dictionary

        Returns:
            List of tool configurations
        """
        tools: list[dict[str, Any]] = []

        # 1. Protocol tool (if role can send messages)
        protocol = role_def.get("protocol", {})
        intents = protocol.get("intents", {})
        can_send = intents.get("can_send", [])

        if can_send:
            # Generic protocol message sender - OpenAI function format
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "send_protocol_message",
                        "description": "Send protocol message to another role",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "receiver": {
                                    "type": "string",
                                    "description": "Target role ID or '*' for broadcast",
                                },
                                "intent": {
                                    "type": "string",
                                    "description": "Protocol intent (e.g., 'tu.open', 'hook.create')",
                                },
                                "payload": {
                                    "type": "object",
                                    "description": "Message payload data",
                                },
                            },
                            "required": ["receiver", "intent", "payload"],
                        },
                    },
                }
            )

        # 2. State tools based on hot_cold_permissions
        constraints = role_def.get("constraints", {})
        hot_cold_permissions = constraints.get("hot_cold_permissions", {})

        hot_perms = hot_cold_permissions.get("hot", {})
        cold_perms = hot_cold_permissions.get("cold", {})

        if hot_perms.get("read"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_hot_sot",
                        "description": "Read from Hot State of Things",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Key to read from hot state",
                                }
                            },
                            "required": ["key"],
                        },
                    },
                }
            )

        if hot_perms.get("write"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "write_hot_sot",
                        "description": "Write to Hot State of Things",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Key to write to hot state",
                                },
                                "value": {
                                    "type": "object",
                                    "description": "Value to write (can be any JSON value)",
                                },
                            },
                            "required": ["key", "value"],
                        },
                    },
                }
            )

        if cold_perms.get("read"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_cold_sot",
                        "description": "Read from Cold State of Things (snapshots)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Key to read from cold state",
                                }
                            },
                            "required": ["key"],
                        },
                    },
                }
            )

        # Cold writes are very restricted (typically only Showrunner)
        if cold_perms.get("write"):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "write_cold_sot",
                        "description": "Write to Cold State of Things (create snapshots)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Key to write to cold state",
                                },
                                "value": {
                                    "type": "object",
                                    "description": "Value to write (will be snapshotted)",
                                },
                            },
                            "required": ["key", "value"],
                        },
                    },
                }
            )

        # 3. External tools based on capabilities
        # Check if role has behavior.tools section
        behavior = role_def.get("behavior", {})
        capability_refs = behavior.get("tools", [])

        # Note: The current role YAMLs don't have a "tools" section yet
        # This is a placeholder for when we add explicit capability references
        for cap_ref in capability_refs:
            if isinstance(cap_ref, dict):
                capability_id = cap_ref.get("capability_ref")
                enabled = cap_ref.get("enabled", True)
                required = cap_ref.get("required", False)

                if enabled:
                    tool_config = self.capability_mapper.get_tool_config_for_capability(
                        capability_id, check_availability=True
                    )

                    if tool_config:
                        # Convert to OpenAI function format
                        tools.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": capability_id,
                                    "description": f"External capability: {capability_id}",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "input": {
                                                "type": "string",
                                                "description": f"Input for {capability_id}",
                                            }
                                        },
                                        "required": ["input"],
                                    },
                                },
                            }
                        )
                    elif required:
                        logger.warning(
                            f"Required capability {capability_id} not available for role "
                            f"{role_def.get('id')}"
                        )

        # 4. Knowledge tools (always available) - OpenAI function format
        # These allow agents to consult the Cartridge (spec) for detailed information
        tools.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "consult_playbook",
                        "description": (
                            "Look up a loop/playbook definition to understand its purpose, "
                            "participating roles, quality gates, and workflow steps. "
                            "Use this to get FULL DETAILS about a loop before initiating it."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "loop_id": {
                                    "type": "string",
                                    "description": (
                                        "Loop ID to look up (e.g., 'story_spark', 'hook_harvest', "
                                        "'lore_deepening'). Use exact IDs from STUDIO KNOWLEDGE."
                                    ),
                                }
                            },
                            "required": ["loop_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "consult_protocol",
                        "description": (
                            "Look up protocol information: valid intents, envelope structure, "
                            "or message flow patterns."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "Query about protocol (e.g., 'tu.open', 'envelope', "
                                        "'intents', 'lifecycle', 'flow', 'example')"
                                    ),
                                }
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "consult_role_charter",
                        "description": (
                            "Look up a role's charter to understand its mandate, capabilities, "
                            "and responsibilities. Use before delegating work to understand "
                            "what a role can do."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "role_id": {
                                    "type": "string",
                                    "description": (
                                        "Role ID to look up (e.g., 'plotwright', 'gatekeeper', "
                                        "'scene_smith'). Use exact IDs from STUDIO KNOWLEDGE."
                                    ),
                                }
                            },
                            "required": ["role_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "consult_quality_gate",
                        "description": (
                            "Look up a quality gate/bar definition to understand validation "
                            "criteria, pass conditions, and remediation guidance."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "bar_name": {
                                    "type": "string",
                                    "description": (
                                        "Quality bar name (e.g., 'integrity', 'reachability', "
                                        "'consistency', 'style')"
                                    ),
                                }
                            },
                            "required": ["bar_name"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "consult_glossary",
                        "description": (
                            "Look up terminology, artifact types, or conventions used in "
                            "QuestFoundry. Use when encountering unfamiliar terms."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "term": {
                                    "type": "string",
                                    "description": ("Term to look up, or 'all' for full glossary"),
                                }
                            },
                            "required": ["term"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "consult_schema",
                        "description": (
                            "Look up JSON schema requirements for artifact types. Shows required "
                            "fields, validation patterns, and minimal valid examples. Use before "
                            "creating artifacts to understand structure."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "artifact_type": {
                                    "type": "string",
                                    "description": (
                                        "Artifact type to look up schema for (e.g., 'section_draft', "
                                        "'hook_card', 'gateway_map', 'tu_brief')"
                                    ),
                                }
                            },
                            "required": ["artifact_type"],
                        },
                    },
                },
            ]
        )

        # 5. Typed tools based on interface.outputs (schema-aware artifact tools)
        # Dynamically add typed tools (write_section_draft, write_hook_card, etc.)
        # based on what artifacts this role produces according to their YAML definition
        interface = role_def.get("interface", {})
        outputs = interface.get("outputs", [])
        registry = get_tool_registry()

        for output in outputs:
            artifact_type = output.get("artifact_type")
            if not artifact_type:
                continue

            # Check if a typed tool exists for this artifact
            tool_name = f"write_{artifact_type}"
            if registry.has_tool(tool_name):
                # Get tool instance from registry
                tool_wrapper = registry.get_tool(tool_name)
                if not tool_wrapper:
                    continue

                # Extract the underlying LangChain BaseTool
                if hasattr(tool_wrapper, "to_langchain_tool"):
                    base_tool = tool_wrapper.to_langchain_tool()
                else:
                    # Tool wrapper doesn't expose BaseTool, skip
                    continue

                # Convert LangChain BaseTool to OpenAI function format
                # Extract args_schema (Pydantic model) and convert to JSON schema
                tool_description = getattr(base_tool, "description", f"Write {artifact_type} artifact")

                # Get the args_schema (Pydantic model)
                args_schema = getattr(base_tool, "args_schema", None)
                if args_schema:
                    # Convert Pydantic model to JSON schema
                    schema_dict = args_schema.model_json_schema()
                    parameters = {
                        "type": "object",
                        "properties": schema_dict.get("properties", {}),
                        "required": schema_dict.get("required", []),
                    }
                else:
                    # No schema, use generic object
                    parameters = {"type": "object", "properties": {}}

                # Add tool in OpenAI function format
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_description,
                            "parameters": parameters,
                        },
                    }
                )

                logger.debug(f"Added typed tool {tool_name} for {role_def.get('id', 'unknown')}")

        return tools

    def _build_studio_knowledge_layer(self) -> str:
        """
        Build STUDIO KNOWLEDGE layer for Showrunner.

        This provides the Showrunner with knowledge about:
        - Available production loops and their purposes
        - Available roles that can be woken/assigned
        - How to route customer directives

        Returns:
            Formatted studio knowledge block
        """
        knowledge_block = """# STUDIO KNOWLEDGE

**CRITICAL: This section defines the production loops and roles you can coordinate.**

## Available Production Loops

These are the ONLY valid loops you can initiate. Do NOT invent loop names.

"""
        # Load all loop definitions
        loops_path = self.definitions_path / "loops"
        if loops_path.exists():
            for loop_file in sorted(loops_path.glob("*.yaml")):
                try:
                    with open(loop_file) as f:
                        loop_def = yaml.safe_load(f)
                    loop_id = loop_def.get("id", loop_file.stem)
                    metadata = loop_def.get("metadata", {})
                    name = metadata.get("name", loop_id)
                    loop_type = metadata.get("type", "Unknown")
                    description = metadata.get("description", "").strip()
                    # Get first sentence only for brevity
                    if description:
                        first_sentence = description.split(".")[0] + "."
                    else:
                        first_sentence = "No description."

                    knowledge_block += (
                        f"- **{name}** (`{loop_id}`): {loop_type} - {first_sentence}\n"
                    )
                except Exception as e:
                    logger.debug(f"Failed to load loop {loop_file}: {e}")

        knowledge_block += """
## Available Roles

These roles can be woken via protocol messages. Use their IDs (lowercase) in receiver field.

"""
        # Load all role definitions
        roles_path = self.definitions_path / "roles"
        if roles_path.exists():
            for role_file in sorted(roles_path.glob("*.yaml")):
                try:
                    with open(role_file) as f:
                        role_def = yaml.safe_load(f)
                    identity = role_def.get("identity", {})
                    role_id = identity.get("id", role_file.stem)
                    name = identity.get("name", role_id)
                    abbreviation = identity.get("abbreviation", "")
                    prompt_content = role_def.get("prompt_content", {})
                    core_mandate = prompt_content.get("core_mandate", "").strip()
                    # Get first sentence only
                    if core_mandate:
                        first_sentence = core_mandate.split(".")[0] + "."
                    else:
                        first_sentence = "No mandate."

                    knowledge_block += (
                        f"- **{name}** (`{role_id}`, {abbreviation}): {first_sentence}\n"
                    )
                except Exception as e:
                    logger.debug(f"Failed to load role {role_file}: {e}")

        knowledge_block += """
## Routing Customer Directives

When a customer gives you a directive, you must:

1. **Read `customer_directives` from hot_sot** to get the actual directive text
2. **Determine intent** - what does the customer want?
3. **Select appropriate loop(s)** from the list above (use exact IDs)
4. **Send protocol message** to wake the appropriate roles:
   - Use `send_protocol_message` tool with `receiver` set to a role ID
   - Use intent `tu.open` to assign work to a role
   - Include relevant context in payload

**IMPORTANT**:
- Only use loop IDs and role IDs from the lists above
- Do NOT invent or hallucinate loop/role names
- If unsure which loop to use, use `story_spark` for new content or `hook_harvest` for hooks
"""
        return knowledge_block

    def assemble_context(
        self, role_id: str, loop_id: str | None, node_id: str | None, state: StudioState
    ) -> dict[str, Any]:
        """
        Main entry point: Assemble complete context for a role.

        Args:
            role_id: Role identifier (e.g., "plotwright")
            loop_id: Current loop identifier (e.g., "story_spark")
            node_id: Current node identifier
            state: Current studio state

        Returns:
            Dictionary containing:
            - prompt: Complete assembled prompt (5 layers)
            - tools: List of tool configurations to bind
            - role_def: Full role definition for reference
        """
        logger.info(f"Assembling context for role={role_id}, loop={loop_id}, node={node_id}")

        # Load role definition
        role_def = self._load_role(role_id)

        # Build 6 layers (5-layer structure + STATE MANAGEMENT)
        identity_layer = self._build_identity_layer(role_def)
        protocol_layer = self._build_protocol_layer(role_def, state)
        state_layer = self._build_state_layer(state, role_id)
        state_management_layer = self._build_state_management_layer(role_def)
        mission_layer = self._build_mission_layer(loop_id, node_id, role_def)
        interface_layer = self._build_interface_block(role_def)

        # Build studio knowledge layer for showrunner (needs to know loops/roles)
        studio_knowledge_layer = ""
        if role_id == "showrunner":
            studio_knowledge_layer = self._build_studio_knowledge_layer()

        # Assemble complete prompt
        prompt = f"""
{identity_layer}

{protocol_layer}

{state_layer}

{state_management_layer}

{studio_knowledge_layer}

{mission_layer}

{interface_layer}
"""

        # Gather tools
        tools = self._gather_tools(role_def)

        logger.info(f"Assembled context with {len(tools)} tools for {role_id}")

        return {
            "prompt": prompt.strip(),
            "tools": tools,
            "role_def": role_def,
        }

    def get_role_info(self, role_id: str) -> dict[str, Any]:
        """
        Get role definition without assembling full context.

        Args:
            role_id: Role identifier

        Returns:
            Role definition dictionary
        """
        return self._load_role(role_id)

    def get_loop_info(self, loop_id: str) -> dict[str, Any]:
        """
        Get loop definition without assembling context.

        Args:
            loop_id: Loop identifier

        Returns:
            Loop definition dictionary
        """
        return self._load_loop(loop_id)
