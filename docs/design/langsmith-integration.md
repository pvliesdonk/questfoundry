# LangSmith Integration Design

## Problem Statement

QuestFoundry already has LangSmith credentials configured and traces are being sent, but they appear with generic labels like "model" and "tools" without semantic context about which **stage** or **phase** produced them.

Current trace structure:
```
LangGraph
├── model (ChatOllama qwen3:8b)
├── tools: list_clusters
├── model (ChatOllama qwen3:8b)
├── tools: search_corpus
└── model (ChatOllama qwen3:8b)
```

Desired structure:
```
DREAM Stage
├── Discuss Phase
│   ├── model (reasoning)
│   ├── tools: list_clusters
│   ├── model (reasoning)
│   └── tools: search_corpus
├── Summarize Phase
│   └── model (distillation)
└── Serialize Phase
    └── model (structured output)
```

## Current Architecture

### What's Already Working

1. **Environment variables** configured in `.env`:
   - `LANGSMITH_TRACING="true"`
   - `LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"`
   - `LANGSMITH_API_KEY` (set)
   - `LANGSMITH_PROJECT=questfoundry`

2. **Local logging** via `LLMLoggingCallback` in `observability/langchain_callbacks.py`

3. **Stage/phase structure** in code:
   - `DreamStage.execute()` calls three phases
   - `run_discuss_phase()` runs the agent loop
   - `summarize_discussion()` distills conversation
   - `serialize_to_artifact()` produces structured output

### What's Missing

1. **No semantic metadata** on LangChain model invocations
2. **No parent span** wrapping phase operations
3. **LangSmith logger suppressed** at line 134 of `logging.py`

## Proposed Solution

### Approach: `@traceable` Decorator + `RunnableConfig`

Use LangSmith's `@traceable` decorator to create semantic parent spans, combined with `RunnableConfig` metadata/tags that flow to child LLM calls.

### Key Mechanisms

#### 1. `@traceable` Decorator (Parent Spans)

Creates named parent runs that group related operations:

```python
from langsmith import traceable

@traceable(name="DREAM Stage", run_type="chain")
async def execute(self, model, user_prompt, ...):
    ...
```

#### 2. `RunnableConfig` (Metadata Propagation)

Pass metadata/tags through `.with_config()` or directly to `ainvoke()`:

```python
config = {
    "run_name": "Discuss Phase",
    "tags": ["dream", "discuss"],
    "metadata": {
        "stage": "dream",
        "phase": "discuss",
        "project_id": project_id,
    }
}
result = await agent.ainvoke({"messages": messages}, config=config)
```

**Key behaviors:**
- `run_name`: Names only the immediate run (not inherited)
- `tags`: Labels inherited by all sub-calls
- `metadata`: Key-value pairs inherited by all sub-calls

#### 3. Nested `@traceable` for Phase Hierarchy

```python
@traceable(name="DREAM Stage", run_type="chain", tags=["stage:dream"])
async def execute(self, model, user_prompt, ...):
    messages, _, _ = await self._run_discuss(model, tools, user_prompt)
    brief, _ = await self._run_summarize(model, messages)
    artifact, _ = await self._run_serialize(model, brief)
    return artifact

@traceable(name="Discuss Phase", run_type="chain", tags=["phase:discuss"])
async def _run_discuss(self, model, tools, user_prompt):
    ...

@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def _run_summarize(self, model, messages):
    ...

@traceable(name="Serialize Phase", run_type="chain", tags=["phase:serialize"])
async def _run_serialize(self, model, brief):
    ...
```

### Implementation Plan

#### Phase 1: Enable Basic Tracing with Semantic Structure

1. **Remove LangSmith logger suppression** in `logging.py`:
   ```python
   # Remove or change line 134:
   # logging.getLogger("langsmith").setLevel(logging.WARNING)
   ```

2. **Add `@traceable` to stage execute methods**:
   ```python
   # In dream.py
   from langsmith import traceable

   class DreamStage:
       @traceable(name="DREAM Stage", run_type="chain")
       async def execute(self, model, user_prompt, ...):
           ...
   ```

3. **Add `@traceable` to phase functions**:
   ```python
   # In discuss.py
   from langsmith import traceable

   @traceable(name="Discuss Phase", run_type="chain")
   async def run_discuss_phase(...):
       ...

   # In summarize.py
   @traceable(name="Summarize Phase", run_type="chain")
   async def summarize_discussion(...):
       ...

   # In serialize.py
   @traceable(name="Serialize Phase", run_type="chain")
   async def serialize_to_artifact(...):
       ...
   ```

4. **Pass config with tags/metadata to LLM calls**:
   ```python
   # In discuss.py
   result = await agent.ainvoke(
       {"messages": current_messages},
       config={
           "recursion_limit": max_iterations,
           "tags": ["dream", "discuss"],
           "metadata": {"stage": "dream", "phase": "discuss"},
       },
   )
   ```

#### Phase 2: Rich Metadata

Add contextual metadata for debugging and analysis:

```python
@traceable(
    name="DREAM Stage",
    run_type="chain",
    metadata={
        "stage": "dream",
        "version": "1.0",
    }
)
async def execute(self, model, user_prompt, provider_name=None, ...):
    # Add dynamic metadata
    from langsmith import get_current_run_tree
    rt = get_current_run_tree()
    rt.metadata["provider"] = provider_name
    rt.metadata["prompt_length"] = len(user_prompt)
    rt.metadata["interactive"] = interactive
    ...
```

#### Phase 3: Retry Tracking in Serialize

Track serialization retry attempts:

```python
@traceable(name="Serialize Phase", run_type="chain")
async def serialize_to_artifact(...):
    for attempt in range(1, max_retries + 1):
        with langsmith.trace(
            name=f"Serialize Attempt {attempt}",
            run_type="llm",
            metadata={"attempt": attempt, "max_retries": max_retries},
        ):
            result = await structured_model.ainvoke(messages)
            ...
```

### Expected Trace Structure After Implementation

```
DREAM Stage [chain] tags: [stage:dream]
├── metadata: {provider: "ollama/qwen3:8b", prompt_length: 150}
│
├── Discuss Phase [chain] tags: [dream, discuss]
│   ├── model [llm] (ChatOllama)
│   │   └── metadata: {stage: "dream", phase: "discuss"}
│   ├── tools: list_clusters
│   ├── model [llm] (ChatOllama)
│   └── tools: search_corpus
│
├── Summarize Phase [chain] tags: [dream, summarize]
│   └── model [llm] (ChatOllama)
│       └── metadata: {stage: "dream", phase: "summarize"}
│
└── Serialize Phase [chain] tags: [dream, serialize]
    ├── Serialize Attempt 1 [llm]
    │   └── metadata: {attempt: 1, max_retries: 3}
    └── Serialize Attempt 2 [llm]  (if retry needed)
        └── metadata: {attempt: 2, max_retries: 3}
```

## Other LangSmith Features for QuestFoundry

### 1. Datasets & Offline Evaluation

**Use Case**: Test prompt changes before deployment

Create datasets of (user_prompt, expected_genre/tone/themes) pairs and evaluate:

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create dataset
dataset = client.create_dataset("dream-prompts", description="DREAM stage test cases")
client.create_examples(
    inputs=[
        {"user_prompt": "A noir mystery set in 1940s Chicago"},
        {"user_prompt": "Epic fantasy with dragons and political intrigue"},
    ],
    outputs=[
        {"expected_genre": "mystery", "expected_tone": "dark, atmospheric"},
        {"expected_genre": "fantasy", "expected_tone": "epic, dramatic"},
    ],
    dataset_id=dataset.id,
)

# Run evaluation
def target(inputs):
    """Run DREAM stage and return artifact."""
    return run_dream_stage(inputs["user_prompt"])

def genre_match_evaluator(run, example):
    """Check if generated genre matches expected."""
    expected = example.outputs.get("expected_genre", "").lower()
    actual = run.outputs.get("genre", "").lower()
    return {"key": "genre_match", "score": 1.0 if expected in actual else 0.0}

results = evaluate(
    target,
    data="dream-prompts",
    evaluators=[genre_match_evaluator],
    experiment_prefix="dream-v1",
)
```

**Benefits**:
- Compare prompt versions (A/B testing)
- Regression testing when changing prompts
- Benchmark different models (Ollama vs OpenAI)

### 2. Pytest Integration for CI

**Use Case**: Run evaluations in CI pipeline

```python
# tests/eval/test_dream_eval.py
import pytest
from langsmith import expect

@pytest.mark.langsmith(test_suite_name="dream-quality")
def test_dream_genre_detection():
    """Test DREAM stage correctly identifies genre."""
    result = run_dream_stage("A hardboiled detective story in rainy Seattle")

    expect.embedding_distance(
        run_output=result["genre"],
        reference_output="mystery/noir",
        threshold=0.3,
    ).to_pass()

@pytest.mark.langsmith
def test_dream_output_structure():
    """Test DREAM artifact has all required fields."""
    result = run_dream_stage("Simple adventure story")

    assert result.get("genre"), "Genre should be populated"
    assert result.get("tone"), "Tone should be populated"
    assert len(result.get("themes", [])) >= 1, "At least one theme required"
```

### 3. Prompt Hub Integration

**Use Case**: Version control and A/B test prompts

```python
from langchain import hub

# Push prompt to hub
hub.push(
    "questfoundry/dream-discuss",
    discuss_prompt,
    description="Discuss phase system prompt",
)

# Pull and use in code
prompt = hub.pull("questfoundry/dream-discuss")

# Compare versions in experiments
prompt_v1 = hub.pull("questfoundry/dream-discuss:v1")
prompt_v2 = hub.pull("questfoundry/dream-discuss:v2")
```

### 4. Online Monitoring (Future)

**Use Case**: Monitor production quality in real-time

When QuestFoundry has users:
- Set up online evaluators to score outputs automatically
- Create alerts for quality degradation
- Track usage patterns and common failure modes

### 5. Annotation Queues (Future)

**Use Case**: Human review of generated artifacts

- Route traces to review queues for human feedback
- Collect annotations on artifact quality
- Use feedback to improve prompts and models

## Configuration Changes Required

### 1. Dependencies

Add `langsmith` to dependencies (if not already present):

```toml
# pyproject.toml
[project.dependencies]
langsmith = ">=0.1.141"
```

### 2. Environment Variables (Already Set)

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=questfoundry
```

### 3. Optional: Disable Local Logging When LangSmith Active

```python
# In CLI or orchestrator
if os.getenv("LANGSMITH_TRACING") == "true":
    # LangSmith handles tracing, skip local JSONL logging
    enable_llm_logging = False
```

## Migration Path

### Immediate (Low Effort)

1. Remove LangSmith logger suppression
2. Add `@traceable` to `execute()`, `run_discuss_phase()`, `summarize_discussion()`, `serialize_to_artifact()`
3. Pass tags/metadata in `RunnableConfig`

### Short-Term

4. Add dynamic metadata (provider, prompt length, interactive mode)
5. Track serialization retry attempts

### Medium-Term

6. Create evaluation datasets for DREAM stage
7. Add pytest-langsmith markers to tests
8. Push prompts to Prompt Hub

### Long-Term

9. Set up online monitoring
10. Create annotation queues for human review
11. Implement CI/CD quality gates

## Example: Minimal Implementation

```python
# src/questfoundry/pipeline/stages/dream.py
from langsmith import traceable

class DreamStage:
    name = "dream"

    @traceable(name="DREAM Stage", run_type="chain", tags=["stage:dream"])
    async def execute(
        self,
        model: BaseChatModel,
        user_prompt: str,
        provider_name: str | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], int, int]:
        # Add dynamic metadata
        from langsmith import get_current_run_tree
        if rt := get_current_run_tree():
            rt.metadata["provider"] = provider_name
            rt.metadata["prompt_length"] = len(user_prompt)
            rt.metadata["interactive"] = kwargs.get("interactive", False)

        tools = get_all_research_tools()

        # Phase 1: Discuss
        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=user_prompt,
            **kwargs,
        )

        # Phase 2: Summarize
        brief, summarize_tokens = await summarize_discussion(model, messages)

        # Phase 3: Serialize
        artifact, serialize_tokens = await serialize_to_artifact(
            model, brief, DreamArtifact, provider_name
        )

        return artifact.model_dump(), discuss_calls + 2, discuss_tokens + summarize_tokens + serialize_tokens


# src/questfoundry/agents/discuss.py
from langsmith import traceable

@traceable(name="Discuss Phase", run_type="chain", tags=["phase:discuss"])
async def run_discuss_phase(
    model: BaseChatModel,
    tools: list[BaseTool],
    user_prompt: str,
    **kwargs,
) -> tuple[list[BaseMessage], int, int]:
    agent = create_discuss_agent(model, tools)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={
            "recursion_limit": kwargs.get("max_iterations", 25),
            "tags": ["dream", "discuss"],
            "metadata": {"stage": "dream", "phase": "discuss"},
        },
    )
    ...


# src/questfoundry/agents/summarize.py
from langsmith import traceable

@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
) -> tuple[str, int]:
    ...


# src/questfoundry/agents/serialize.py
from langsmith import traceable

@traceable(name="Serialize Phase", run_type="chain", tags=["phase:serialize"])
async def serialize_to_artifact(
    model: BaseChatModel,
    brief: str,
    schema: type[T],
    provider_name: str | None = None,
    **kwargs,
) -> tuple[T, int]:
    ...
```

## References

- [LangSmith: Add metadata and tags to traces](https://docs.langchain.com/langsmith/add-metadata-tags)
- [LangSmith: Use @traceable decorator](https://docs.langchain.com/langsmith/annotate-code)
- [LangSmith: Troubleshoot trace nesting](https://docs.langchain.com/langsmith/nest-traces)
- [LangChain: RunnableConfig](https://docs.langchain.com/oss/python/langchain/models)
- [LangSmith: Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [LangSmith: pytest integration](https://docs.langchain.com/langsmith/pytest)
