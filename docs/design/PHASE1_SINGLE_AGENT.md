# Phase 1: Single Agent Execution

> **Issue**: #145
> **Status**: ✅ Complete
> **Branch**: `epic/phase1-single-agent`
> **Tests**: 197 passing (110 new for Phase 1)

## Overview

Get a single agent to receive input, process it via LLM, and return a streaming response. Foundation for all agent interactions. Includes session tracking and parallel async support from the start.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Streaming | From start | Better UX, harder to retrofit |
| Sessions | From start | Need turn tracking for context |
| Async | Parallel-ready | Multiple agents can run concurrently |
| Context overflow | Hard error | Force prompt engineering, no silent truncation |
| Response format | Free text | Structured output in Phase 2 with tools |

---

## 1. LLM Provider Abstraction

### File Structure

```
src/questfoundry/runtime/providers/
├── __init__.py      # ProviderRegistry, get_provider()
├── base.py          # LLMProvider ABC, LLMMessage, LLMResponse
├── ollama.py        # OllamaProvider (langchain-ollama)
├── openai.py        # OpenAIProvider (langchain-openai)
└── google.py        # GoogleProvider (langchain-google-genai)
```

### Provider Interface

```python
class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def invoke(
        self,
        messages: list[LLMMessage],
        options: InvokeOptions,
    ) -> LLMResponse:
        """Send messages and get complete response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        options: InvokeOptions,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks."""

    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if provider is reachable."""

    @abstractmethod
    def get_context_size(self, model: str) -> int:
        """Get context window size for model."""
```

### Context Size Configuration

Add to `qf.yaml`:

```yaml
models:
  creative:
    ollama: qwen3:8b
    openai: gpt-4o
    google: gemini-1.5-pro
    context_size:
      ollama: 32768      # qwen3:8b default
      openai: 128000     # gpt-4o
      google: 1000000    # gemini-1.5-pro
```

### Token Counting

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: TokenUsage
    duration_ms: float
    raw: Any  # Provider-specific response
```

### Error Handling

- `ProviderUnavailableError` - Provider not reachable
- `ContextOverflowError` - Prompt exceeds model context size
- `ProviderError` - General provider failure
- Retry 3x with exponential backoff (1s, 2s, 4s)

---

## 2. Session Management

### File Structure

```
src/questfoundry/runtime/session/
├── __init__.py      # Exports: Session, Turn
├── session.py       # Session class
└── turn.py          # Turn tracking
```

### Session Model

```python
@dataclass
class Session:
    id: str
    project_id: str
    entry_agent: str
    started_at: datetime
    status: Literal["active", "completed", "error"]
    turns: list[Turn]

@dataclass
class Turn:
    id: int
    agent_id: str
    input: str
    output: str | None
    started_at: datetime
    ended_at: datetime | None
    usage: TokenUsage | None
    status: Literal["pending", "streaming", "completed", "error"]
```

Sessions are stored in project SQLite (tables already exist from Phase 0).

---

## 3. Agent Runtime

### File Structure

```
src/questfoundry/runtime/agent/
├── __init__.py      # Exports: AgentRuntime, activate_agent()
├── runtime.py       # AgentRuntime class
├── context.py       # ContextBuilder - knowledge injection
└── prompt.py        # PromptBuilder - system prompt construction
```

### Agent Activation Flow

```
1. Load agent definition from Studio
2. Build context (knowledge injection)
3. Build system prompt
4. Validate context size < model limit
5. Create/update session turn
6. Stream LLM response
7. Update turn with result
```

### Context Builder

Knowledge injection based on agent's `knowledge_requirements`:

```python
class ContextBuilder:
    def build(self, agent: Agent, studio: Studio) -> AgentContext:
        """Build context for agent activation."""
        context = AgentContext()

        # Constitution (if agent requires)
        if agent.knowledge_requirements.constitution:
            context.add_knowledge(studio.constitution, layer="constitution")

        # Must-know entries
        for entry_id in agent.knowledge_requirements.must_know:
            entry = studio.get_knowledge(entry_id)
            context.add_knowledge(entry, layer="must_know")

        # Role-specific entries (include as menu for now)
        for entry_id in agent.knowledge_requirements.role_specific:
            entry = studio.get_knowledge(entry_id)
            context.add_menu_item(entry)

        return context
```

### Prompt Builder

System prompt structure:

```
[Agent Identity]
You are {agent.name}. {agent.description}

[Knowledge - Constitution]
{constitution.text}

[Knowledge - Must Know]
{must_know entries}

[Constraints]
{agent.constraints as bullet points}

[Capabilities]
{agent.capabilities summary}

[Available Knowledge]
You can consult: {menu of role_specific entries}
```

### Context Size Validation

```python
def validate_context_size(self, messages: list[LLMMessage], model: str) -> None:
    """Raise ContextOverflowError if prompt exceeds model limit."""
    total_tokens = self.count_tokens(messages)
    max_tokens = self.provider.get_context_size(model)

    if total_tokens > max_tokens:
        raise ContextOverflowError(
            f"Prompt ({total_tokens} tokens) exceeds model context "
            f"({max_tokens} tokens). Reduce knowledge or use larger model."
        )
```

---

## 4. CLI Commands

### Interactive REPL

```bash
# Start interactive session (auto-creates 'default' project)
qf ask

# With specific project
qf ask my_story

# Output:
# QuestFoundry Interactive Session
# Project: my_story
# Agent: Showrunner (showrunner)
# Type 'exit' or Ctrl+D to quit
#
# > Hello, who are you?
# [streaming response...]
```

### Single-Shot

```bash
# Quick query (auto-creates 'default' project)
qf ask "Hello, who are you?"

# With specific project
qf ask my_story "Hello, who are you?"

# Specify entry agent
qf ask my_story --entry-agent player_narrator "Tell me about this world"
```

### Verbose Flags

```bash
# Show token usage after responses
qf ask -v "Hello"

# Show timing and model info
qf ask -vv "Hello"

# Show full prompts being sent
qf ask -vvv "Hello"
```

### Full Options

```
qf ask [OPTIONS] [PROJECT] [PROMPT]

Arguments:
  PROJECT  Project ID (auto-creates 'default' if omitted)
  PROMPT   Prompt (omit for REPL mode)

Options:
  -e, --entry-agent TEXT  Entry agent ID
  -d, --domain PATH       Path to domain directory [default: domain-v4]
  -p, --projects-dir PATH Projects directory [default: projects]
  -m, --model TEXT        Model to use
  -v, --verbose           Increase verbosity (-v: tokens, -vv: timing, -vvv: prompts)
```

---

## 5. Observability

### Event Logging

JSONL events to `project_dir/logs/events.jsonl`:

```json
{"event": "session_start", "session_id": "...", "agent": "showrunner", "ts": "..."}
{"event": "turn_start", "turn_id": 1, "input": "Hello", "ts": "..."}
{"event": "llm_call", "model": "qwen3:8b", "prompt_tokens": 1234, "ts": "..."}
{"event": "turn_complete", "turn_id": 1, "completion_tokens": 567, "ts": "..."}
```

### LangSmith Integration

When `LANGSMITH_TRACING=true`:

```
Session: my_story/session_abc123
└── Turn 1
    └── LLM Call (qwen3:8b)
        ├── Input: [system, user messages]
        └── Output: [assistant response]
```

---

## Implementation Order

| Component | Tests | Status |
|-----------|-------|--------|
| Providers (base + ollama + streaming) | 33 | ✅ |
| Session management | 36 | ✅ |
| PromptBuilder | 14 | ✅ |
| ContextBuilder | 10 | ✅ |
| AgentRuntime | 13 | ✅ |
| CLI `qf ask` | 10 | ✅ |

**Actual: 110 new tests for Phase 1 (197 total)**

---

## Acceptance Criteria

```bash
# Interactive REPL
qf ask --project test_story
> Hello, who are you?
# [Streaming response from Showrunner]

# Single-shot
qf ask --project test_story "What can you help with?"
# [Streaming response]

# Different entry agent
qf ask --project test_story --entry-agent player_narrator "Tell me about this world"
# [Streaming response from Player Narrator]

# Context overflow error
qf ask --project test_story "..." # with massive prompt
# Error: Prompt (50000 tokens) exceeds model context (32768 tokens)
```

---

## Dependencies

- Phase 0 (domain loader, configuration, project storage)

## References

- `domain-v4/agents/showrunner.json` - Example entry agent
- `domain-v4/knowledge/layers.json` - Knowledge injection rules
- `domain-v4/governance/constitution.json` - Constitution content
