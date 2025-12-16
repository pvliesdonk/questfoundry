# Tool Calling Fixes (December 2025)

## Overview

This document captures fixes made to the V4 runtime tool calling system and identifies a gap that Phase 4 (Storage & Lifecycle) will address.

## Issues Fixed

### 1. Streaming Tool Call Arguments

**Problem**: When using LangChain's streaming mode, tool call arguments arrived as empty `{}`.

**Root Cause**: LangChain streaming sends tool call arguments as JSON string fragments via `tool_call_chunks`, not as parsed objects in `tool_calls`. The providers were only checking `tool_calls`.

**Fix**: Both OpenAI and Ollama providers now accumulate `tool_call_chunks` by index and parse the JSON at stream end:

```python
# Track tool call chunks by index
tool_call_builders: dict[int, dict[str, str]] = {}

async for chunk in llm.astream(messages):
    tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
    if tool_call_chunks:
        for tcc in tool_call_chunks:
            idx = tcc.get("index", 0)
            if idx not in tool_call_builders:
                tool_call_builders[idx] = {"id": "", "name": "", "args_json": ""}
            builder = tool_call_builders[idx]
            if tcc.get("id"):
                builder["id"] = tcc["id"]
            if tcc.get("name"):
                builder["name"] = tcc["name"]
            if tcc.get("args"):
                builder["args_json"] += tcc["args"]  # Accumulate JSON fragments

# Parse accumulated tool calls at end
for builder in tool_call_builders.values():
    args = json.loads(builder["args_json"]) if builder["args_json"] else {}
    final_tool_calls.append(ToolCallRequest(id=builder["id"], name=builder["name"], arguments=args))
```

**Files Changed**:
- `src/questfoundry/runtime/providers/openai.py`
- `src/questfoundry/runtime/providers/ollama.py`

### 2. Tool Call Conversation Flow Loop

**Problem**: Models would loop calling the same tool repeatedly, ignoring tool results.

**Root Cause**: LangChain requires `AIMessage` objects to have `tool_calls` attached so it can associate subsequent `ToolMessage` results. The V4 runtime's intermediate `LLMMessage` format didn't preserve `tool_calls`, breaking this association.

**Fix**:
1. Added `tool_calls` field to `LLMMessage` dataclass
2. Updated message conversion to preserve `tool_calls` on `AIMessage`
3. Updated runtime to store `tool_calls` on assistant messages

**Files Changed**:
- `src/questfoundry/runtime/providers/base.py` - Added `tool_calls` field
- `src/questfoundry/runtime/providers/openai.py` - Preserve in conversion
- `src/questfoundry/runtime/providers/ollama.py` - Preserve in conversion
- `src/questfoundry/runtime/agent/runtime.py` - Store on messages

### 3. CLI Non-Streaming Option

**Problem**: Streaming mode is unreliable with some models (especially Ollama/qwen3:8b).

**Fix**: Added `--no-stream` CLI option to disable streaming output.

```bash
uv run qf ask --no-stream -p ollama "create a story"
```

**Files Changed**:
- `src/questfoundry/cli.py` - Added `--no-stream` flag and `_invoke_response()` function

## Gap Identified: Artifact Persistence

### Current State

Agents like `scene_smith` have capabilities to create artifacts:
- `create_sections` - "Create and update section prose"
- `write_workspace` - "Can write draft sections"

However, there is **no tool to actually save artifacts**. The model generates content in its response text, then tries to validate non-existent artifacts.

### Observed Behavior

From OpenAI test logs:
```
tool_call_start: validate_artifact, args: {artifact_id: "scene_1", artifact_type_id: "section"}
tool_call_complete: success=false, error="No artifact data provided or found"
```

The model:
1. Generates section content in its response
2. Tries to validate `artifact_id="scene_1"`
3. Fails because nothing was ever saved

### Resolution

This gap will be addressed by **Phase 4: Storage & Lifecycle** (Issue #148), which includes:
- Store Manager with CRUD operations
- Artifact Storage with schema validation on write
- Lifecycle state management (draft -> review -> approved -> cold)

### Workaround (Not Recommended)

The `validate_artifact` tool supports inline `artifact_data`:
```json
validate_artifact(artifact_type_id="section", artifact_data={...content...})
```

However, this is awkward (massive JSON in tool calls) and doesn't provide persistence.

## Test Results

| Provider | Streaming | Non-Streaming |
|----------|-----------|---------------|
| OpenAI (gpt-4o) | Works | Works |
| Ollama (qwen3:8b) | Flaky | Works |

**Recommendation**: Use `--no-stream` with Ollama until streaming reliability improves.

## Related Issues

- #143 - V4 Runtime Cleanroom Rebuild (master tracking)
- #147 - Phase 3: Delegation & Messaging (tool call flow)
- #148 - Phase 4: Storage & Lifecycle (artifact persistence)
