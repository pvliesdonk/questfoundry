# Phase 2: Tool Execution

> **Issue**: #146
> **Status**: Complete
> **PR**: #152
> **Tests**: 256 passing (59 new for Phase 2)

## Overview

Implement tool infrastructure so agents can invoke defined actions. Tools are callable units that agents use to interact with the system. Defined declaratively in `domain-v4/tools/*.json` and implemented in Python.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Capability-based access | Tool filtering per agent | Security: agents only see tools they have capability for |
| Async execution | `async def execute()` | Consistency with runtime, future parallel tool calls |
| Registry pattern | Decorator-based registration | Clean separation of definition vs implementation |
| Availability checks | Pre-execution validation | Graceful degradation when services unavailable |

---

## 1. Tool Base Infrastructure

### File Structure

```
src/questfoundry/runtime/tools/
├── __init__.py              # Exports: BaseTool, ToolResult, ToolRegistry
├── base.py                  # BaseTool ABC, ToolResult, ToolContext, exceptions
├── registry.py              # ToolRegistry, @register_tool decorator
├── consult_schema.py        # Artifact type schema lookup
├── consult_corpus.py        # RAG-style knowledge search
├── validate_artifact.py     # Schema-based validation
├── search_workspace.py      # Artifact search in hot store
├── delegate.py              # Agent delegation stub (Phase 3)
├── request_lifecycle_transition.py  # State transitions stub (Phase 4)
├── write_hot.py             # Hot store writes stub (Phase 4)
├── web_fetch.py             # External URL fetching
└── web_search.py            # Web search via DuckDuckGo
```

### Base Classes

```python
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any]
    error: str | None = None
    execution_time_ms: float | None = None

@dataclass
class ToolContext:
    studio: Studio
    project: Project | None = None
    agent_id: str | None = None
    session_id: str | None = None
    domain_path: Path | None = None

class BaseTool(ABC):
    def __init__(self, definition: Tool, context: ToolContext): ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> ToolResult: ...

    def validate_input(self, args: dict[str, Any]) -> None: ...
    def check_availability(self) -> bool: ...
    async def run(self, args: dict[str, Any]) -> ToolResult: ...
    def to_langchain_schema(self) -> dict[str, Any]: ...
```

### Registration Pattern

```python
from questfoundry.runtime.tools.registry import register_tool

@register_tool("consult_schema")
class ConsultSchemaTool(BaseTool):
    async def execute(self, args: dict[str, Any]) -> ToolResult:
        # Implementation
        ...
```

---

## 2. Tool Registry

### Capability-Based Filtering

```python
class ToolRegistry:
    def get_agent_tools(self, agent: Agent, session_id: str | None) -> list[BaseTool]:
        """Get all tools an agent has capability to use."""
        tools = []
        allowed_tool_ids = self._get_agent_tool_refs(agent)
        for tool_id in allowed_tool_ids:
            tool = self.get_tool(tool_id, agent.id, session_id)
            tools.append(tool)
        return tools

    def check_capability(self, agent: Agent, tool_id: str) -> bool:
        """Check if an agent has capability to use a tool."""
        allowed = self._get_agent_tool_refs(agent)
        return tool_id in allowed

    def enforce_capability(self, agent: Agent, tool_id: str) -> None:
        """Raise CapabilityViolationError if agent lacks capability."""
        if not self.check_capability(agent, tool_id):
            raise CapabilityViolationError(agent.id, tool_id)
```

### LangChain Integration

```python
def get_langchain_tools(self, agent: Agent, session_id: str | None) -> list[dict]:
    """Get LangChain-compatible tool schemas for bind_tools()."""
    tools = self.get_agent_tools(agent, session_id)
    return [tool.to_langchain_schema() for tool in tools if tool.check_availability()]
```

---

## 3. Implemented Tools

### consult_schema

Returns artifact type schema with field definitions, validation rules, and lifecycle states.

```python
# Usage
result = await tool.execute({
    "artifact_type_id": "section",
    "include_validation_rules": True
})

# Returns
{
    "artifact_type": {
        "id": "section",
        "name": "Section",
        "fields": [...],
        "lifecycle": {...},
        "validation": {...}
    },
    "field_summary": "Required: title (string), Optional: ...",
    "lifecycle_summary": "Initial: draft, States: draft -> review -> ..."
}
```

### consult_corpus

RAG-style search in knowledge base with keyword matching (vector search placeholder).

```python
# Usage
result = await tool.execute({
    "query": "how to write dialogue",
    "max_results": 3
})

# Returns
{
    "excerpts": [
        {"excerpt": "...", "source_file": "knowledge/craft/dialogue.md", "relevance_score": 0.85}
    ],
    "search_method": "keyword"
}
```

### validate_artifact

Schema-based validation against artifact type definitions.

```python
# Usage
result = await tool.execute({
    "artifact_type_id": "section",
    "artifact_data": {"title": "Chapter 1", "content": "..."},
    "validation_level": "standard"
})

# Returns
{
    "valid": True,
    "issues": [],
    "artifact_type": "section"
}
```

### search_workspace

Search artifacts in hot store with filters.

```python
# Usage
result = await tool.execute({
    "artifact_type": "section",
    "filters": {"lifecycle_state": "draft"},
    "max_results": 10
})

# Returns
{
    "artifacts": [...],
    "total_count": 5
}
```

### web_fetch

Fetch and extract content from URLs with async rate limiting.

```python
# Usage
result = await tool.execute({
    "url": "https://example.com/article",
    "extract_mode": "text"
})

# Returns
{
    "url": "...",
    "content": "...",
    "content_type": "text/html",
    "extracted_text": "..."
}
```

### web_search

Web search via DuckDuckGo API.

```python
# Usage
result = await tool.execute({
    "query": "medieval castle architecture",
    "max_results": 5
})

# Returns
{
    "results": [
        {"title": "...", "url": "...", "snippet": "..."}
    ],
    "query": "medieval castle architecture"
}
```

### Stub Tools (Phase 3-4)

- `delegate` - Agent delegation (Phase 3)
- `request_lifecycle_transition` - State transitions (Phase 4)
- `write_hot` - Hot store writes (Phase 4)

---

## 4. Exception Hierarchy

```python
class ToolError(Exception):
    """Base exception for tool execution errors."""

class ToolValidationError(ToolError):
    """Tool input validation failed."""

class ToolExecutionError(ToolError):
    """Tool execution failed."""

class ToolUnavailableError(ToolError):
    """Tool is not available (missing dependencies, config, etc.)."""

class CapabilityViolationError(ToolError):
    """Agent attempted to use a tool they don't have capability for."""

    def __init__(self, agent_id: str, tool_id: str):
        self.agent_id = agent_id
        self.tool_id = tool_id
```

---

## 5. Agent Runtime Integration

### Tool Execution Flow

```python
class AgentRuntime:
    @property
    def tool_registry(self) -> ToolRegistry | None:
        """Lazy initialization of tool registry."""
        if self._tool_registry is None:
            self._tool_registry = ToolRegistry(
                studio=self._studio,
                project=self._project,
                domain_path=self._domain_path,
            )
        return self._tool_registry

    async def execute_tool(
        self,
        tool_id: str,
        args: dict[str, Any],
        agent: Agent,
        session_id: str | None = None,
    ) -> ToolResult:
        """Execute a tool with capability enforcement."""
        self.tool_registry.enforce_capability(agent, tool_id)
        tool = self.tool_registry.get_tool(tool_id, agent.id, session_id)

        # Log tool call start
        if self._event_logger:
            self._event_logger.log(EventType.TOOL_CALL_START, ...)

        result = await tool.run(args)

        # Log tool call result
        if self._event_logger:
            self._event_logger.log(EventType.TOOL_CALL_COMPLETE, ...)

        return result
```

---

## 6. Observability

### Event Logging

```json
{"event": "tool_call_start", "agent_id": "showrunner", "tool_id": "consult_schema", "args": {...}}
{"event": "tool_call_complete", "agent_id": "showrunner", "tool_id": "consult_schema", "success": true, "execution_time_ms": 12.5}
```

---

## Implementation Order

| Component | Tests | Status |
|-----------|-------|--------|
| Base tool infrastructure | 15 | Complete |
| Tool registry + capability filtering | 12 | Complete |
| consult_schema | 8 | Complete |
| consult_corpus | 10 | Complete |
| validate_artifact | 9 | Complete |
| search_workspace | 7 | Complete |
| web_fetch | 6 | Complete |
| web_search | 5 | Complete |
| Stub tools (delegate, etc.) | 4 | Complete |
| Runtime integration | 8 | Complete |

**Total: 59 new tests for Phase 2 (256 total)**

---

## Dependencies

- Phase 0 (domain loader, types)
- Phase 1 (agent runtime, sessions)

## References

- `domain-v4/tools/*.json` - Tool definitions
- `meta/schemas/core/tool.schema.json` - Tool schema contract
- `meta/schemas/core/agent.schema.json` - Capability definitions
