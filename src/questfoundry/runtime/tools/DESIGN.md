# Tool Execution System Design

## Overview

The tool execution system enables agents to interact with the QuestFoundry runtime through
declaratively-defined tools. Tools are defined in `domain/tools/*.json` and implemented in
Python classes that inherit from `BaseTool`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentRuntime                              │
│                                                                  │
│  ┌────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ ContextBuilder │    │  ToolRegistry   │    │ LLMProvider  │ │
│  └────────────────┘    └────────┬────────┘    └──────────────┘ │
│                                 │                               │
│                    ┌────────────┴────────────┐                 │
│                    │                         │                 │
│              ┌─────▼─────┐           ┌──────▼──────┐          │
│              │ BaseTool  │           │ BaseTool    │          │
│              │ (schema)  │           │ (validate)  │          │
│              └───────────┘           └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### ToolRegistry

Central registry that:

1. **Loads tool definitions** from domain JSON files
2. **Maps tool IDs to implementations** via `TOOL_IMPLEMENTATIONS` dict
3. **Filters tools by agent capabilities** - agents only see authorized tools
4. **Creates tool instances** with runtime context (studio, project, etc.)

### BaseTool

Abstract base class providing:

- `execute(args)`: Core tool logic (abstract)
- `validate_input(args)`: Input validation against schema
- `check_availability()`: Pre-flight checks (API keys, services)
- `run(args)`: Full execution with validation, timing, error handling
- `to_langchain_schema()`: Convert to LangChain tool format

### ToolContext

Dependency injection container providing:

- `studio`: Loaded studio definition
- `project`: Project storage (optional)
- `agent_id`: Current agent
- `session_id`: Current session
- `domain_path`: Path to domain files

### ToolResult

Standardized result format:

```python
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any]
    error: str | None = None
    execution_time_ms: float | None = None
```

## Tool Implementations

### Core Tools (Phase 2)

| Tool | Purpose | Implementation Status |
|------|---------|----------------------|
| `consult_schema` | Get artifact type schema | ✅ Complete |
| `validate_artifact` | Validate against schema + 8 bars | ✅ Complete (structural checks) |
| `search_workspace` | Search project artifacts | ✅ Complete |
| `consult_corpus` | RAG search in knowledge base | ✅ Complete (keyword), TODO: vector |
| `delegate` | Delegate work to agent | ⚠️ Partial (Phase 3 routing) |
| `web_search` | Search web via SearXNG | ✅ Complete |
| `web_fetch` | Fetch and extract URL content | ✅ Complete |

### Stub Tools (Future Phases)

| Tool | Purpose | Status |
|------|---------|--------|
| `generate_image` | Image generation | Stub |
| `generate_audio` | Audio generation | Stub |
| `assemble_export` | Export assembly | Stub |

## Capability Enforcement

Tools are gated by agent capabilities defined in `domain/agents/*.json`:

```json
{
  "capabilities": [
    {
      "id": "use_delegate_tool",
      "category": "tool",
      "tool_ref": "delegate"
    }
  ]
}
```

The registry:

1. Extracts `tool_ref` from agent capabilities
2. Only returns tools matching those refs
3. Logs capability violations for audit

## Quality Bars (validate_artifact)

The 8 quality bars checked during validation:

1. **Integrity** - No contradictions or dead links
2. **Reachability** - Critical content accessible
3. **Nonlinearity** - Branches matter
4. **Gateways** - Conditions enforceable
5. **Style** - Voice consistent
6. **Determinism** - Reproducible where promised
7. **Presentation** - Spoiler-safe
8. **Accessibility** - Navigation clear

Current implementation:

- Structural checks are programmatic
- Content analysis marked as requiring LLM (yellow status)

## Search Methods (consult_corpus)

### Keyword Search (Always Available)

- Tokenize query and corpus
- Score: `(word_coverage * 0.7) + (density * 0.3)`
- Remove stop words
- Return top N by score

### Vector Search (TODO)

- Requires sqlite-vec extension
- Requires embedding model
- Semantic similarity via cosine distance

## Error Handling

```python
class ToolError(Exception): ...
class ToolValidationError(ToolError): ...    # Input validation failed
class ToolExecutionError(ToolError): ...     # Execution failed
class ToolUnavailableError(ToolError): ...   # Missing deps/config
class CapabilityViolationError(ToolError):   # No permission
```

## Observability

Tool calls are logged to EventLogger:

- `tool_call_start`: tool_id, agent_id, args
- `tool_call_complete`: success, error, execution_time_ms

## Adding New Tools

1. Create tool definition in `domain/tools/<tool_id>.json`
2. Create implementation in `src/questfoundry/runtime/tools/<tool_id>.py`
3. Decorate with `@register_tool("<tool_id>")`
4. Import in `tools/__init__.py`

```python
from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

@register_tool("my_tool")
class MyTool(BaseTool):
    async def execute(self, args: dict) -> ToolResult:
        # Implementation
        return ToolResult(success=True, data={...})
```

## Phase 3 Dependencies

The `delegate` tool requires Phase 3 (#147) for:

- Message broker integration
- Delegation lifecycle management
- Result collection from delegates
- Timeout handling

See: `delegate.py` for detailed TODO comments.
