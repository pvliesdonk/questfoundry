# RuntimeContextAssembler & CapabilityMapper Implementation Summary

## Overview

I've implemented two new classes that dynamically assemble agent prompts from structured YAML definitions:

1. **RuntimeContextAssembler** - Builds 5-layer prompts from role/loop/protocol YAMLs
2. **CapabilityMapper** - Maps abstract capabilities to concrete tool providers

## Files Created

### Core Implementation

1. **`lib/runtime/src/questfoundry/runtime/core/runtime_context_assembler.py`**
   - Main assembler class
   - Reads role, loop, and protocol YAMLs
   - Builds 5-layer prompt structure
   - Gathers tools based on role permissions
   - ~580 lines with comprehensive docstrings

2. **`lib/runtime/src/questfoundry/runtime/core/capability_mapper.py`**
   - Maps capabilities to providers
   - Checks provider availability (API keys, packages, commands)
   - Implements priority-based fallback chains
   - Caches availability results
   - ~380 lines with comprehensive docstrings

### Supporting Files

3. **`lib/runtime/src/questfoundry/runtime/core/README.md`**
   - Complete documentation
   - Usage examples
   - API reference
   - Integration guide

4. **`lib/runtime/examples/test_context_direct.py`**
   - Direct module test
   - Verifies all 5 layers present
   - Checks tool gathering
   - ~150 lines

5. **`lib/runtime/examples/standalone_test.py`**
   - Standalone verification script
   - Checks file paths and dependencies
   - ~120 lines

### Updated Files

6. **`lib/runtime/src/questfoundry/runtime/core/__init__.py`**
   - Added lazy imports to avoid dependency issues
   - Exported RuntimeContextAssembler and CapabilityMapper
   - Prevents loading heavy modules at import time

7. **`lib/runtime/src/questfoundry/runtime/__init__.py`**
   - Converted to lazy imports
   - Prevents importing modules with missing dependencies
   - Better separation of concerns

## Architecture

### RuntimeContextAssembler

**Input Sources:**

- Role profiles: `spec/05-definitions/roles/*.yaml`
- Loop patterns: `spec/05-definitions/loops/*.yaml`
- Protocol definitions: `spec/05-definitions/protocol.yaml`
- Runtime state: `StudioState` object

**Output:**

```python
{
    "prompt": str,        # Complete 5-layer prompt
    "tools": list[dict],  # Tool configurations
    "role_def": dict      # Full role definition
}
```

### 5-Layer Prompt Structure

1. **IDENTITY** - Role name, mandate, principles, anti-patterns, heuristics
2. **PROTOCOL** - Valid intents (send/receive), envelope defaults, lifecycle permissions
3. **STATE** - Current TU, loop, node, Hot/Cold context, available artifacts
4. **MISSION** - Loop description, node task, success criteria, quality bars
5. **INTERFACE** - Protocol enforcement, tool mandates, input/output expectations

### Tool Gathering Strategy

Tools are gathered from 4 sources:

1. **Protocol Tools (Implicit)** - Derived from `protocol.intents.can_send`
   - `send_protocol_message` (generic sender)

2. **State Tools (Implicit)** - Based on `constraints.hot_cold_permissions`
   - `read_hot_sot` / `write_hot_sot` (if hot.read/write = true)
   - `read_cold_sot` / `write_cold_sot` (if cold.read/write = true)

3. **External Tools (Explicit)** - Referenced via `behavior.tools` (when added)
   - Mapped through CapabilityMapper
   - Examples: `image_generation`, `web_search`, `audio_synthesis`

4. **Knowledge Tools (Implicit, Always)** - Always available
   - `consult_protocol`, `consult_role_charter`, `consult_quality_gate`

### CapabilityMapper

**Input Sources:**

- Capability definitions: `spec/05-definitions/capabilities.yaml`
- Tool mappings: `lib/runtime/config/tool_mappings.yaml`

**Key Features:**

- **Provider Selection** - Chooses best available provider by priority
- **Availability Checking** - Verifies API keys, packages, commands
- **Fallback Chains** - Falls through to next provider on failure
- **Caching** - Caches availability checks for performance

**Provider Types:**

- `api_service` - External API (requires API key)
- `local_tool` - Local package/command
- `stub` - Always-available placeholder (testing)
- `knowledge` - Knowledge base access

## Key Design Decisions

### 1. YAML-Driven Approach

Rather than hardcoding prompts, we read from structured YAMLs. This allows:

- Runtime flexibility without code changes
- Clear separation of specification (Layer 5) from runtime (Layer 6)
- Easy iteration on role definitions
- Versioning of role configurations

### 2. Layer-Based Prompt Structure

The 5-layer structure ensures:

- **Consistency** - Every role gets same structure
- **Clarity** - Clear sections for different concerns
- **Enforcement** - INTERFACE block emphasizes protocol requirements
- **Context** - STATE layer provides current execution context

### 3. Implicit vs Explicit Tools

- **Implicit Tools** - Never listed in role YAMLs, derived from permissions
  - Protocol tools (from `protocol.intents`)
  - State tools (from `constraints.hot_cold_permissions`)
  - Knowledge tools (always available)

- **Explicit Tools** - Must be declared in role YAMLs (future enhancement)
  - External capabilities (from `behavior.tools` section)
  - Require CapabilityMapper to resolve providers

### 4. Protocol Communication Enforcement

The INTERFACE block explicitly states:

- Protocol messages are **MANDATORY**
- Natural language responses **WITHOUT** protocol envelopes will be **IGNORED**
- Roles **CANNOT** communicate outside protocol system
- All actions **REQUIRE** tool usage

This is critical for runtime enforcement of structured communication.

### 5. Provider Fallback Chains

CapabilityMapper implements intelligent fallback:

1. Try primary provider (priority 1)
2. If unavailable/failed, try secondary (priority 2)
3. Continue through chain until success
4. Terminal fallback to stub (priority 999) for testing

This ensures graceful degradation when external services unavailable.

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
state.hot_sot = {"hooks": [], "topology_notes": []}

# Assemble context
context = assembler.assemble_context(
    role_id="plotwright",
    loop_id="story_spark",
    node_id="topology_draft",
    state=state
)

# Use prompt and tools
prompt = context["prompt"]
tools = context["tools"]

# Pass to LLM API
llm_response = llm.invoke(prompt, tools=tools)
```

## Integration with NodeFactory

The RuntimeContextAssembler is designed to replace inline prompt building in NodeFactory:

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

**Benefits:**

- Separates prompt assembly from node creation
- Enables dynamic prompt generation
- Consistent prompt structure across all roles
- Easier to test and iterate

## Testing

While full integration tests require PyYAML and the complete runtime, the implementation includes:

### Unit Test Structure

- Test capability mapper initialization
- Test provider availability checking
- Test prompt layer generation
- Test tool gathering logic
- Test caching behavior

### Integration Test Structure

- Test with real YAML files
- Test with mock StudioState
- Verify all 5 layers present
- Verify tool configurations correct
- Verify role/loop loading

### Test Files

- `test_context_direct.py` - Direct module tests
- `standalone_test.py` - Standalone verification

**To run tests:**

```bash
# Install dependencies
pip install pyyaml

# Run from project root
python lib/runtime/examples/standalone_test.py
```

## Dependencies

### Required

- **PyYAML** - YAML parsing (`pip install pyyaml`)
- **pathlib** - Path manipulation (stdlib)
- **logging** - Debug logging (stdlib)

### Optional (for external capabilities)

- **API keys** - For external services (OPENAI_API_KEY, etc.)
- **Local tools** - For local processing (pandoc, imagemagick, etc.)
- **Python packages** - For local capabilities (pillow, librosa, etc.)

## Error Handling

Both classes include comprehensive error handling:

### RuntimeContextAssembler

- **FileNotFoundError** - If role/loop YAML missing
- **ValidationError** - If YAML doesn't match schema (future)
- **KeyError** - If required fields missing from YAML
- **Exception** - Generic fallback with logging

### CapabilityMapper

- **FileNotFoundError** - If configuration files missing
- **yaml.YAMLError** - If YAML parsing fails
- **ImportError** - If capability mapper imports fail
- Graceful fallback to stubs when providers unavailable

## Future Enhancements

### Short Term

1. **Add `behavior.tools` section to role YAMLs** - Explicit capability references
2. **Schema validation** - Validate YAML against JSON schemas
3. **Better caching** - Smarter cache invalidation
4. **More tests** - Comprehensive unit and integration tests

### Medium Term

1. **Dynamic tool injection** - Different tools per node
2. **Context compression** - Summarize when prompts too long
3. **Prompt templating** - Custom templates per role
4. **Multi-language support** - Translate prompts dynamically

### Long Term

1. **Hot reloading** - Update prompts without restart
2. **A/B testing** - Test different prompt variations
3. **Prompt optimization** - Learn from successful executions
4. **Tool usage analytics** - Track which tools are used

## Conclusion

The RuntimeContextAssembler and CapabilityMapper provide a robust foundation for dynamic prompt assembly in QuestFoundry. They:

- **Separate concerns** - Specification (YAMLs) from runtime (Python)
- **Enable flexibility** - Change roles without code changes
- **Enforce structure** - Consistent 5-layer prompts
- **Support fallbacks** - Graceful degradation when services unavailable
- **Promote clarity** - Clear documentation and examples

These classes bridge the gap between Layer 5 (compiled definitions) and Layer 6 (runtime execution), enabling the runtime to dynamically adapt to role definitions while maintaining strict protocol enforcement.

## File Locations

**Core Implementation:**

- `lib/runtime/src/questfoundry/runtime/core/runtime_context_assembler.py`
- `lib/runtime/src/questfoundry/runtime/core/capability_mapper.py`

**Documentation:**

- `lib/runtime/src/questfoundry/runtime/core/README.md`
- `lib/runtime/IMPLEMENTATION_SUMMARY.md` (this file)

**Tests:**

- `lib/runtime/examples/test_context_direct.py`
- `lib/runtime/examples/standalone_test.py`

**Configuration:**

- `spec/05-definitions/capabilities.yaml` (already exists)
- `spec/05-definitions/protocol.yaml` (already exists)
- `lib/runtime/config/tool_mappings.yaml` (already exists)

**Role & Loop Definitions (read by assembler):**

- `spec/05-definitions/roles/*.yaml` (15 roles)
- `spec/05-definitions/loops/*.yaml` (12 loops)
