# Runtime Context Assembler & Capability Mapper

This module provides dynamic prompt assembly from YAML definitions.

## Role Execution Model (CRITICAL)

**This is NOT a ReAct agent loop.** Roles communicate via protocol messages, not "final answers".

### Execution Flow

1. Role receives a message (graph routes based on `receiver` field of last message)
2. Role does work (reads state via tools, processes information)
3. Role calls `send_protocol_message(receiver=..., intent=..., content=...)`
4. **Role is DONE** — execution returns, graph routes to next role
5. Continue until `receiver = "__terminate__"`

### What This Means for Implementation

- **No "Final Answer" detection** — The `send_protocol_message` call IS the output
- **No iteration limits on work** — Only count FAILURES (errors, malformed calls)
- **No looping after message sent** — Role completes immediately after sending
- **Graph handles routing** — `control_plane.py` routes based on envelope `receiver`

### Why This Matters

When implementing role execution:

```python
# WRONG - ReAct pattern
while iterations < max_iterations:
    result = llm.invoke(...)
    if "Final Answer:" in result:
        return parse_final_answer(result)  # WRONG!
    iterations += 1

# CORRECT - Protocol message pattern
result = llm.invoke(...)  # LLM calls tools including send_protocol_message
# When send_protocol_message is called, role is done
# Graph routes based on message.receiver
return result
```

### Parallel Execution

Multiple messages can fan out in parallel:

- `receiver="*"` broadcasts to all active roles
- Multiple `send_protocol_message` calls to different receivers can execute in parallel

See `spec/04-protocol/FLOWS/` for examples (e.g., Hook Harvest shows SR sending to LW, PW, RS
simultaneously).

### See Also

- `lib/runtime/AGENTS.md` — Execution model anti-patterns table
- `spec/04-protocol/` — Full protocol specification

---

## Overview

### RuntimeContextAssembler

The `RuntimeContextAssembler` class dynamically builds agent prompts from structured YAML definitions. It reads:

- **Role profiles** from `spec/05-definitions/roles/*.yaml`
- **Loop patterns** from `spec/05-definitions/loops/*.yaml`
- **Protocol definitions** from `spec/05-definitions/protocol.yaml`

And produces:

1. **5-layer prompts** with structured context
2. **Tool configurations** for LLM API binding

### CapabilityMapper

The `CapabilityMapper` class maps abstract capabilities to concrete tool implementations:

- Reads **capability definitions** from `spec/05-definitions/capabilities.yaml`
- Reads **tool mappings** from `lib/runtime/config/tool_mappings.yaml`
- Selects **providers** based on availability and priority
- Provides **fallback chains** when primary providers fail

## Prompt Structure (5 Layers)

### 1. IDENTITY Layer

- Role name, abbreviation, and charter reference
- Core mandate
- Operating principles
- Anti-patterns to avoid
- Heuristics with examples

### 2. PROTOCOL Layer

- Protocol intents the role can send
- Protocol intents the role can receive
- Envelope defaults (safety, hot/cold context)
- Lifecycle permissions (hooks, TUs, gates, views)

### 3. STATE Layer

- Current Trace Unit (TU) context
- Current loop and node
- Hot/Cold state context
- Available artifacts from Hot SoT

### 4. MISSION Layer

- Current loop description
- Current node task
- Success criteria for the loop
- Task guidance from role
- Quality bars owned by the role

### 5. INTERFACE Layer

- **Critical enforcement block** emphasizing protocol requirements
- Tool usage mandates
- Expected inputs and outputs
- Side effects the role can trigger
- Structured output schema requirements

## Tool Gathering

Tools are gathered in 4 categories:

### 1. Protocol Tools (Implicit)

- Derived from `protocol.intents.can_send`
- Generic `send_protocol_message` tool
- Never explicitly listed in role configs

### 2. State Tools (Implicit)

- Based on `constraints.hot_cold_permissions`
- `read_hot_sot` / `write_hot_sot` (if permitted)
- `read_cold_sot` / `write_cold_sot` (if permitted)
- Never explicitly listed in role configs

### 3. External Tools (Explicit)

- Referenced via `behavior.tools` section (when added to role YAMLs)
- Mapped through CapabilityMapper
- Examples: `image_generation`, `web_search`, `audio_synthesis`
- Selected based on provider availability

### 4. Knowledge Tools (Implicit, Always Available)

- `consult_protocol` - Query protocol specifications
- `consult_role_charter` - Query role mandates
- `consult_quality_gate` - Query quality bar requirements
- Always available through runtime knowledge base

## Usage Example

```python
from questfoundry.runtime.core import RuntimeContextAssembler
from questfoundry.runtime.models.state import StudioState

# Create assembler
assembler = RuntimeContextAssembler()

# Create state
state = StudioState()
state.tu_id = "tu-001-story-spark"
state.loop_id = "story_spark"
state.node_id = "topology_draft"
state.hot_cold = "hot"

# Assemble context
context = assembler.assemble_context(
    role_id="plotwright",
    loop_id="story_spark",
    node_id="topology_draft",
    state=state
)

# Access prompt and tools
prompt = context["prompt"]  # Full 5-layer prompt
tools = context["tools"]    # List of tool configurations
role_def = context["role_def"]  # Full role definition

# Print prompt layers
print(prompt)

# Print tools
for tool in tools:
    print(f"- {tool['tool_id']} [{tool['category']}]")
```

## CapabilityMapper Usage

```python
from questfoundry.runtime.core import CapabilityMapper

# Create mapper
mapper = CapabilityMapper()

# Get available provider for a capability
provider = mapper.get_available_provider("image_generation")

if provider:
    print(f"Using {provider.id} ({provider.provider_name})")
    print(f"Priority: {provider.priority}")
    print(f"Config: {provider.config}")

# Get tool configuration
tool_config = mapper.get_tool_config_for_capability("web_search")

if tool_config:
    print(f"Tool class: {tool_config['tool_class']}")
    print(f"Provider: {tool_config['provider_name']}")
    print(f"Config: {tool_config['config']}")

# Get capability summary
summary = mapper.get_capability_summary()

print(f"Total capabilities: {summary['total_capabilities']}")

# Check external capabilities
for cap_id, info in summary["external_capabilities"].items():
    available = "✓" if info["available"] else "✗"
    provider = info.get("provider_name", "N/A")
    print(f"{available} {cap_id}: {provider}")
```

## Provider Selection

The CapabilityMapper selects providers based on:

1. **Priority** - Lower numbers = higher priority
2. **Availability** - Checks API keys, packages, commands
3. **Fallback Strategy** - Falls through to next provider on failure

### Availability Checks

- **api_key**: Check if environment variable is set
- **python_package**: Check if package is importable
- **command_available**: Check if command exists in PATH
- **always_available**: Always returns true (stubs)

### Example Provider Chain (image_generation)

1. **dalle3** (priority 1) - If `OPENAI_API_KEY` set
2. **stable_diffusion_api** (priority 2) - If `STABILITY_API_KEY` set
3. **midjourney_api** (priority 3) - If `MIDJOURNEY_API_KEY` set
4. **stub_generator** (priority 999) - Always available (testing)

## Protocol Communication Enforcement

The INTERFACE block emphasizes that:

- Protocol messages are the **ONLY** way to communicate
- Natural language responses **WITHOUT** protocol envelopes will be **IGNORED**
- Roles cannot communicate outside the protocol system
- All actions require tool usage

This is critical for the runtime to enforce structured communication.

## Implementation Details

### Caching

Both classes cache loaded YAML data:

- `RuntimeContextAssembler._role_cache`: Role definitions
- `RuntimeContextAssembler._loop_cache`: Loop patterns
- `CapabilityMapper._availability_cache`: Provider availability

Caches can be cleared:

```python
# Clear provider availability cache
mapper.clear_availability_cache()
```

### File Paths

Default paths (relative to project root):

- Roles: `spec/05-definitions/roles/{role_id}.yaml`
- Loops: `spec/05-definitions/loops/{loop_id}.yaml`
- Protocol: `spec/05-definitions/protocol.yaml`
- Capabilities: `spec/05-definitions/capabilities.yaml`
- Tool Mappings: `lib/runtime/config/tool_mappings.yaml`

Custom paths can be provided to constructors.

## Testing

The implementation includes test scripts:

- `lib/runtime/examples/test_context_direct.py` - Direct module test
- `lib/runtime/examples/standalone_test.py` - Standalone verification

Run tests from project root:

```bash
python lib/runtime/examples/standalone_test.py
```

## Dependencies

- **PyYAML**: YAML parsing
- **pathlib**: Path manipulation (stdlib)
- **logging**: Debug logging (stdlib)

Install dependencies:

```bash
pip install pyyaml
```

## Integration with NodeFactory

The `RuntimeContextAssembler` is designed to replace inline prompt building in `NodeFactory`:

**Before:**

```python
# NodeFactory builds prompts inline
prompt = self.render_prompt(role, state)
```

**After:**

```python
# NodeFactory uses RuntimeContextAssembler
assembler = RuntimeContextAssembler()
context = assembler.assemble_context(role_id, loop_id, node_id, state)
prompt = context["prompt"]
tools = context["tools"]
```

This separates concerns:

- **NodeFactory**: Transforms roles into LangGraph nodes
- **RuntimeContextAssembler**: Builds prompts from YAML
- **CapabilityMapper**: Maps capabilities to tools

## Future Enhancements

1. **Add `behavior.tools` section to role YAMLs** - Explicit capability references
2. **Dynamic tool injection based on node context** - Different tools per node
3. **Context compression for long prompts** - Summarize when needed
4. **Multi-language support** - Translate prompts dynamically
5. **Prompt templating system** - Custom templates per role

## See Also

- `spec/05-definitions/capabilities.yaml` - Capability definitions
- `lib/runtime/config/tool_mappings.yaml` - Provider configurations
- `spec/05-definitions/roles/*.yaml` - Role profiles
- `spec/05-definitions/loops/*.yaml` - Loop patterns
- `spec/05-definitions/protocol.yaml` - Protocol definitions
