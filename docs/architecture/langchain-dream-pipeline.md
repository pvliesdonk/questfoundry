# LangChain DREAM Pipeline Architecture

**Last Updated**: 2026-01-06
**Status**: Implemented

---

## Overview

The DREAM stage establishes the creative vision for an interactive story. It uses a **three-phase pattern** (Discuss → Summarize → Serialize) to collaboratively develop and document the vision, then serialize it to a validated YAML artifact.

The new implementation uses **LangChain-native patterns** instead of custom agent infrastructure:
- `langchain.agents` for the Discuss phase
- `ChatPromptTemplate` for prompt management
- Structured output strategies (ToolStrategy for Ollama, ProviderStrategy for OpenAI)
- `ConversationRunner` wraps these into a unified three-phase flow

---

## Three-Phase Pattern

### Phase 1: Discuss

**Purpose**: Explore and refine the creative vision through dialogue.

**Characteristics**:
- Uses `langchain.agents.create_agent` for autonomous exploration
- Access to research tools: `search_corpus`, `web_search`, `web_fetch`
- Higher temperature (0.8) for creative exploration
- Runs to completion (no interrupt/resume)
- Two modes:
  - **Interactive**: Multi-turn dialogue with user, up to 10 turns
  - **Direct**: Single-turn model-only discussion, no user interaction

**Behavior**:
- Agent has access to research tools to gather information about genre conventions, writing craft, trends
- Can call tools multiple times in a single response
- Discussion ends when either:
  - User calls `ready_to_summarize()` (interactive mode)
  - User types `/done` (interactive mode)
  - Max turns reached (both modes)

**Output**: Unstructured discussion content that feeds into the Summarize phase.

### Phase 2: Summarize

**Purpose**: Distill discussion into a concise creative vision brief.

**Characteristics**:
- Direct model call (no tools, no agent)
- Lower temperature (0.3) for focused synthesis
- Uses same LLM provider and model as Discuss phase
- Receives the full discussion history plus a summarization prompt

**Behavior**:
- LLM reads entire discussion and extracts key creative decisions
- Produces narrative summary covering:
  - Genre and subgenre
  - Tone and atmosphere
  - Target audience
  - Core themes
  - Scope and complexity
  - Content notes

**Output**: Brief narrative text to be serialized in next phase.

### Phase 3: Serialize

**Purpose**: Convert brief narrative into validated structured YAML artifact.

**Characteristics**:
- Uses `with_structured_output()` for guaranteed JSON/YAML format
- Strategy per provider:
  - **Ollama**: `ToolStrategy` (more reliable for qwen3:8b)
  - **OpenAI**: `ProviderStrategy` (native JSON mode)
- Lower temperature (0.1) for deterministic output
- Validation/repair loop: up to 3 attempts to produce valid output

**Behavior**:
1. Submit brief narrative plus structured output schema (DreamArtifact)
2. LLM generates JSON matching schema
3. Validate against Pydantic model
4. If invalid: provide structured error feedback, retry (max 3 times)
5. If valid: write to artifacts/dream.yaml

**Output**: Validated YAML artifact with complete creative vision metadata.

---

## Implementation Details

### Entry Point: DreamStage.execute()

```python
async def execute(
    self,
    context: dict[str, Any],
    provider: LLMProvider,
    compiler: PromptCompiler,
) -> tuple[dict[str, Any], int, int]:
    """Execute DREAM stage using 3-phase pattern.

    Args:
        context: User prompt, interaction mode, tools, callbacks
        provider: LLM provider interface
        compiler: Prompt template compiler

    Returns:
        (artifact_data, llm_calls, tokens_used)
    """
```

### Context Dictionary

Passed to `DreamStage.execute()`:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `user_prompt` | str | Yes | User's story idea |
| `interactive` | bool | No | Enable multi-turn dialogue (default: False) |
| `user_input_fn` | async callable | No | Coroutine to get user input |
| `research_tools` | list[Tool] | No | Tools available during Discuss |
| `max_turns` | int | No | Max discussion turns (default: 10 if interactive) |
| `validation_retries` | int | No | Max Serialize retries (default: 3) |
| `on_assistant_message` | callable | No | Callback when assistant responds |
| `on_llm_start` | callable | No | Callback at LLM call start |
| `on_llm_end` | callable | No | Callback at LLM call end |

### ConversationRunner Orchestration

`ConversationRunner` manages all three phases:

```python
runner = ConversationRunner(
    provider=provider,
    research_tools=research_tools,
    finalization_tool=SubmitDreamTool(),  # Serialization schema
    max_discuss_turns=1 if not interactive else 10,
    validation_retries=3,
)

artifact_data, state = await runner.run(
    initial_messages=[...],
    user_input_fn=user_input_fn if interactive else None,
    validator=self._validate_dream,  # Pydantic validation
    summary_prompt=self._get_summary_prompt(),
    on_assistant_message=...,
    on_llm_start=...,
    on_llm_end=...,
)
```

### Prompt Templates

Uses `ChatPromptTemplate` from `langchain_core.prompts`:

**Location**: `prompts/templates/dream.md`

**Variables injected**:
- `mode_instructions`: Discuss phase behavior (interactive vs direct)
- `mode_reminder`: Reminder at end of Discuss phase
- `user_message`: The user's story idea
- `research_tools_section`: Description of available tools

**Structure**:
1. System message: Role and constraints
2. User message: User's prompt with mode-specific instructions
3. Tool definitions: Research tools if provided
4. Finalization tool: SubmitDreamTool schema for serialization

### Tool-Based Finalization

The `SubmitDreamTool` is a Pydantic model that:
- Defines the JSON schema for the DreamArtifact
- Forces structured output through the tool mechanism
- Prevents hallucinated or incomplete artifacts

```python
class SubmitDreamTool:
    """Tool that captures complete dream artifact."""

    # Schema matches DreamArtifact pydantic model
    genre: str
    tone: str
    themes: list[str]
    audience: str
    scope: str
    # ... other fields
```

When LLM calls this tool, its arguments are extracted and validated.

---

## Provider Configuration

### Default: Ollama qwen3:8b

```bash
# Environment
OLLAMA_HOST=http://athena.int.liesdonk.nl:11434

# CLI
qf dream --provider ollama/qwen3:8b "my story idea"
```

**Configuration Flow**:
1. CLI flag: `--provider ollama/qwen3:8b` (highest priority)
2. Environment: `QF_PROVIDER=ollama/qwen3:8b`
3. Project file: `project.yaml` `providers.default`
4. Hardcoded: `ollama/qwen3:8b` (fallback)

### Alternative: OpenAI gpt-4o

```bash
# Environment
OPENAI_API_KEY=sk-...

# CLI
qf dream --provider openai/gpt-4o "my story idea"
```

**Note**: API keys must be explicitly configured. No defaults are provided.

### Provider Interface (Protocol)

All providers implement the `LLMProvider` protocol:

```python
class LLMProvider(Protocol):
    """Unified interface for LLM providers."""

    @property
    def default_model(self) -> str: ...

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse: ...
```

### LangChainProvider Adapter

`LangChainProvider` wraps any LangChain chat model to our protocol:

```python
# Ollama
from langchain_community.chat_models import ChatOllama

model = ChatOllama(
    model="qwen3:8b",
    base_url="http://athena.int.liesdonk.nl:11434",
    temperature=0.8,
    num_predict=4096,
)
provider = LangChainProvider(model, "ollama/qwen3:8b")

# OpenAI
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.8,
    max_tokens=4096,
)
provider = LangChainProvider(model, "openai/gpt-4o")
```

---

## Validation & Repair Loop

### Serialization Validation

When the LLM submits the SubmitDreamTool, its arguments are validated against the `DreamArtifact` Pydantic model.

**Valid Path**:
1. LLM calls SubmitDreamTool with valid data
2. Pydantic validation passes
3. Data persisted to artifacts/dream.yaml
4. Stage completes

**Invalid Path**:
1. LLM calls SubmitDreamTool with invalid data
2. Pydantic validation fails
3. Structured error feedback generated (per ADR-007)
4. Feedback added to conversation
5. LLM retries (max 3 attempts)

### Error Feedback Format

When validation fails, feedback follows ADR-007:

```json
{
  "result": "validation_failed",
  "issues": {
    "invalid": [
      {
        "field": "genre",
        "provided": "sci-fi / fantasy mix",
        "problem": "contains '/' character",
        "requirement": "single genre string without slashes"
      }
    ],
    "missing": [
      {
        "field": "themes",
        "requirement": "list of 2-5 theme strings"
      }
    ],
    "unknown": []
  },
  "issue_count": 2,
  "action": "Call SubmitDreamTool again with corrected data..."
}
```

The feedback is designed to:
- Be machine-parseable (structured JSON)
- Be human-comprehensible (clear descriptions)
- Guide the LLM toward correction (specific field requirements)
- Prevent infinite loops (semantic result enum)

---

## Artifact Schema

The `DreamArtifact` Pydantic model is generated from `schemas/dream.schema.json`.

**Key Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `type` | "dream" | Artifact type identifier |
| `version` | int | Schema version |
| `genre` | str | Primary genre (e.g., "sci-fi", "fantasy") |
| `tone` | str | Narrative tone (e.g., "dark", "whimsical") |
| `themes` | list[str] | Core themes (2-5 strings) |
| `audience` | str | Target audience (e.g., "adult", "young adult") |
| `scope` | str | Story scope (e.g., "intimate", "sprawling") |
| `creative_vision` | str | Narrative creative direction |
| `content_notes` | str | Content warnings or notes |

**Validation Rules**:
- All fields required except `content_notes`
- `genre`, `tone`, `audience`, `scope`: min_length=1
- `themes`: min 2, max 5 items per schema
- `creative_vision`: captures the Summarize phase output

**File Format**: YAML with metadata preservation via ruamel.yaml

---

## Logging & Observability

### Events

Uses structlog for consistent event-driven logging:

```python
# Phase transitions
log.info("dream_execute_start", interactive=True, prompt_length=125)
log.debug("prompt_compiled", template="dream")
log.info("phase_complete", phase="discuss", turns=3, tokens=1240)
log.info("phase_complete", phase="summarize", tokens=580)
log.info("phase_complete", phase="serialize", attempts=1, tokens=620)
log.info("stage_complete", stage="dream", llm_calls=3, tokens_used=2440)

# Tool calls (DEBUG level)
log.debug("tool_call_start", tool="search_corpus", query="noir mystery")
log.debug("tool_call_result", tool="search_corpus", results=3)

# Validation
log.debug("validation_start", phase="serialize", attempt=1)
log.warning("validation_failed", field="genre", issue="missing")
log.info("validation_passed", attempt=2)
```

### LLM Call Logging

When enabled (via CLI `--log` or env `LANGSMITH_TRACING=true`):

- Full request/response in `{project}/logs/llm_calls.jsonl`
- Structured application logs in `{project}/logs/debug.jsonl`
- Each line is a JSON object with timestamp, event name, and structured context

---

## Temperature & Sampling Settings

**Discuss Phase**: temperature=0.8
- Higher temperature encourages exploration of diverse genre conventions
- Tool use provides grounding

**Summarize Phase**: temperature=0.3
- Lower temperature focuses on synthesis
- Reduces hallucination of irrelevant themes

**Serialize Phase**: temperature=0.1
- Very low temperature ensures consistent JSON structure
- Reduces variation in field choices

---

## Future Considerations

1. **Multi-model support**: Currently optimized for qwen3:8b + gpt-4o; extensible to other models
2. **Tool expansion**: Can add corpus search, web search, or custom domain tools
3. **Interactive UX**: Discuss phase supports real-time user interaction; can be extended with streaming
4. **Caching**: LLM responses could be cached between runs for idempotency
5. **Parallel execution**: Future versions might parallelize independent research tool calls

---

## References

- **Discuss Phase**: Uses `langchain.agents` (agentless initially, can upgrade)
- **Summarize Phase**: Direct model call via `LangChainProvider.complete()`
- **Serialize Phase**: `with_structured_output()` + validation/repair loop
- **Provider Interface**: Protocol-based, implementation-agnostic
- **Validation**: Pydantic models, schema-first (source of truth in JSON Schema)
- **Logging**: structlog with consistent event naming

See also:
- [docs/design/01-pipeline-architecture.md](../design/01-pipeline-architecture.md) - Design rationale
- [docs/design/05-prompt-compiler.md](../design/05-prompt-compiler.md) - Prompt assembly
- [docs/architecture/decisions.md](./decisions.md) - Related ADRs
