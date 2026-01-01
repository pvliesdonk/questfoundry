# Pipeline Architecture

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

The pipeline orchestrator executes v5 stages sequentially with human gates and iteration control. It is the central component that coordinates artifact flow between stages.

---

## Pipeline Flow

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Pipeline Orchestrator                           │
│                                                                         │
│  ┌────────┐   ┌───────────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐│
│  │ DREAM  │ → │BRAINSTORM │ → │ SEED │ → │ GROW │ → │ FILL │ → │ SHIP ││
│  └───┬────┘   └─────┬─────┘   └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘│
│      │              │            │          │          │          │    │
│      ▼              ▼            ▼          ▼          ▼          ▼    │
│  [Gate]         [Gate]       [Gate]     [Gate]     [Gate]     [Gate]  │
│      │              │            │          │          │          │    │
│      ▼              ▼            ▼          ▼          ▼          ▼    │
│  dream.yaml   brainstorm.yaml seed.yaml  grow/*    fill/*    exports/ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Stage Definitions

### DREAM

**Purpose**: Establish the creative vision and constraints.

**Input**: User's initial prompt (free text)

**Output**: `artifacts/dream.yaml`

**LLM Calls**: 1

**Human Gate**: Optional (can auto-advance)

```yaml
# Example output
type: dream
version: 1

genre: mystery
subgenre: noir
tone: dark, atmospheric
audience: adult
themes:
  - betrayal
  - redemption
  - the cost of truth
style_notes: |
  Hard-boiled narration. Short, punchy sentences.
  Metaphors drawn from urban decay and weather.
scope:
  target_word_count: 15000
  estimated_passages: 40
  branching_depth: moderate
```

### BRAINSTORM

**Purpose**: Generate raw creative material without committing.

**Input**: `dream.yaml`

**Output**: `artifacts/brainstorm.yaml`

**LLM Calls**: 1-3 (can batch or separate characters/settings/hooks)

**Human Gate**: Optional

```yaml
# Example output
type: brainstorm
version: 1

characters:
  - id: detective_maria
    sketch: "Burned-out PI with a secret connection to the victim"
    hooks:
      - "Her scar from the unsolved case"
      - "She was the victim's AA sponsor"
  - id: captain_chen
    sketch: "By-the-book police captain, old partner of Maria"
    hooks:
      - "Owes Maria a favor from the past"

settings:
  - id: harbor_district
    sketch: "Fog-shrouded docks, abandoned warehouses"
    hooks:
      - "Secret gambling den in the old fish market"
  - id: chinatown
    sketch: "Neon signs, family-run businesses, old traditions"

what_ifs:
  - "What if the victim faked their death?"
  - "What if the killer is protecting someone?"
  - "What if Maria is being framed?"
```

### SEED

**Purpose**: Crystallize brainstorm material into committed story foundation.

**Input**: `dream.yaml` + `brainstorm.yaml`

**Output**: `artifacts/seed.yaml`

**LLM Calls**: 1

**Human Gate**: **Required** (structural commitment point)

```yaml
# Example output
type: seed
version: 1

protagonist:
  ref: brainstorm.characters.detective_maria
  name: Maria Chen
  occupation: Private investigator
  flaw: Cannot let go of cold cases
  want: Find the truth about her brother's disappearance
  need: Learn to forgive herself

setting:
  ref: brainstorm.settings.harbor_district
  time_period: 1940s
  location: San Francisco
  key_locations:
    - maria_office: "Cramped office above a laundromat"
    - docks: "Where the body surfaces"
    - chinatown: "Maria's family neighborhood"

central_tension: |
  A body surfaces that Maria recognizes from her past.
  The victim was supposed to be dead for ten years.

selected_hooks:
  - brainstorm.what_ifs[2]  # Maria is being framed
  - brainstorm.characters.detective_maria.hooks[1]  # AA sponsor
```

### GROW

**Purpose**: Build complete story topology with branching.

**Input**: `seed.yaml`

**Output**: `artifacts/grow/` (multiple files)

**LLM Calls**: Variable (see GROW specification)

**Human Gate**: **Required** after ANCHORS, optional elsewhere

See [03-grow-stage-specification.md](./03-grow-stage-specification.md) for full details.

### FILL

**Purpose**: Generate prose for each scene brief.

**Input**: `grow/briefs/*.yaml` + `seed.yaml`

**Output**: `artifacts/fill/scenes/*.yaml`

**LLM Calls**: N (one per scene brief)

**Human Gate**: Optional (batch review)

```yaml
# Example output
type: scene
version: 1

brief_ref: grow/briefs/opening_001.yaml
passage_id: opening_001

prose: |
  The fog rolled in thick that Tuesday morning, same as it did
  every morning in this part of the city. Maria Chen watched it
  crawl past her office window, coffee gone cold in her hands.

  Ten years since she'd quit the force. Ten years of small jobs
  and smaller paychecks. Ten years of trying not to think about
  Jimmy.

  The phone rang.

choices:
  - text: "Answer it"
    target: phone_call_001
  - text: "Let it ring"
    target: ignore_phone_001
    grants:
      - missed_first_call
```

### SHIP

**Purpose**: Export to playable format.

**Input**: All `fill/scenes/*.yaml` + `grow/connections.yaml`

**Output**: `exports/` (multiple formats)

**LLM Calls**: 0 (deterministic compilation)

**Human Gate**: **Required** (final review before release)

Supported formats:
- **Twee** (`.tw`) — Twine-compatible
- **JSON** — Engine-agnostic data
- **HTML** — Standalone playable
- **Ink** — Ink runtime format

---

## Human Gates

### Gate Types

| Type | Behavior |
|------|----------|
| `required` | Pipeline halts until explicit approval |
| `optional` | Auto-advances after delay; human can veto |
| `skip` | No gate (proceed immediately) |

### Gate Configuration

```yaml
# pipeline.yaml
gates:
  dream: optional
  brainstorm: optional
  seed: required        # Structural commitment
  grow.spine: optional
  grow.anchors: required  # Structure locked after this
  grow.fractures: optional
  grow.branches: optional
  grow.connections: optional
  grow.briefs: optional
  fill: optional        # Batch review possible
  ship: required        # Final release gate
```

### Gate Interface

At each gate, the human can:

1. **Approve** — Proceed to next stage
2. **Edit** — Modify artifact, then approve
3. **Regenerate** — Discard and re-run stage
4. **Abort** — Stop pipeline

```bash
# CLI example
qf review seed
# Shows: seed.yaml content
# Prompts: [A]pprove, [E]dit, [R]egenerate, A[b]ort?
```

---

## Iteration Control

### Extra Rounds

Some stages support iteration before proceeding:

```bash
qf grow --extra-round
```

This triggers the HARVEST checkpoint (see GROW specification).

### Round Limits

```yaml
# pipeline.yaml
iteration:
  grow.harvest_rounds: 1      # Default rounds
  grow.max_harvest_rounds: 3  # Maximum allowed
  fill.polish_rounds: 0       # No polish by default
  fill.max_polish_rounds: 2
```

### Iteration Flow

```
┌─────────────┐     ┌─────────────┐
│   Stage     │ ──► │    Gate     │
└─────────────┘     └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Extra round │ ◄─── qf grow --extra-round
                    │  requested? │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Under limit?│
                    └──────┬──────┘
                      yes  │  no
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │Re-run   │   │Proceed  │
              │stage    │   │to next  │
              └─────────┘   └─────────┘
```

---

## Model Routing

Different stages can use different LLM providers/models:

```yaml
# pipeline.yaml
providers:
  default: ollama/qwen3:8b

  stages:
    dream: default
    brainstorm: openai/gpt-4o        # Creative ideation
    seed: default
    grow.spine: default
    grow.anchors: default
    grow.branches: anthropic/claude-3-5-sonnet  # Complex branching
    fill: default
    ship: null                        # No LLM needed
```

### Provider Interface

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate completion from messages."""
        ...
```

Implementations required for:
- Ollama (local models)
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Generic OpenAI-compatible endpoint

---

## CLI Interface

### Stage Commands

```bash
# Run individual stages
qf dream                    # Run DREAM stage
qf brainstorm               # Run BRAINSTORM stage
qf seed                     # Run SEED stage
qf grow                     # Run all GROW layers
qf grow --layer spine       # Run specific GROW layer
qf fill                     # Run FILL stage
qf ship                     # Run SHIP stage

# Run multiple stages
qf run                      # Run entire pipeline
qf run --to seed            # Run up to SEED (inclusive)
qf run --from grow          # Run from GROW to end

# Iteration
qf grow --extra-round       # Request additional HARVEST round

# Review
qf review seed              # Review SEED artifact
qf status                   # Show pipeline state
```

### Project Commands

```bash
# Project management
qf init                     # Create new project
qf init --from template     # Create from template
qf validate                 # Validate all artifacts
qf export --format twee     # Export to format
```

### Status Output

```
$ qf status

Project: noir_mystery
Pipeline State:

  DREAM       ✓ completed   dream.yaml
  BRAINSTORM  ✓ completed   brainstorm.yaml
  SEED        ✓ approved    seed.yaml
  GROW
    spine     ✓ completed   grow/spine.yaml
    anchors   ● in_review   grow/anchors.yaml
    fractures ○ pending
    branches  ○ pending
    connections ○ pending
    briefs    ○ pending
  FILL        ○ pending
  SHIP        ○ pending

Current gate: grow.anchors (required)
Run: qf review grow.anchors
```

---

## Orchestrator Interface

```python
@dataclass
class StageResult:
    stage: str
    status: Literal["completed", "failed", "pending_review"]
    artifact_path: Path
    llm_calls: int
    tokens_used: int
    errors: list[str]

class PipelineOrchestrator:
    def __init__(self, project_path: Path, config: PipelineConfig):
        ...

    async def run_stage(
        self,
        stage_name: str,
        extra_round: bool = False
    ) -> StageResult:
        """Run a single stage."""
        ...

    async def run_to(self, target_stage: str) -> list[StageResult]:
        """Run all stages up to and including target."""
        ...

    async def run_all(self) -> list[StageResult]:
        """Run complete pipeline (stopping at required gates)."""
        ...

    def get_status(self) -> PipelineStatus:
        """Report current pipeline state."""
        ...

    def get_stage_context(self, stage_name: str) -> dict:
        """Get input context for a stage from prior artifacts."""
        ...
```

---

## Error Handling

### Stage Failures

If a stage fails (LLM error, validation failure):

1. Error logged with full context
2. Partial artifacts preserved (if any)
3. Pipeline halts
4. Human reviews error and decides:
   - **Retry** — Re-run stage
   - **Edit** — Fix artifacts manually, then proceed
   - **Abort** — Stop pipeline

### Validation Failures

Artifact validation happens before writing:

```python
async def run_stage(self, stage_name: str) -> StageResult:
    # 1. Build context from prior artifacts
    context = self.get_stage_context(stage_name)

    # 2. Compile prompt
    prompt = self.prompt_compiler.compile(stage_name, context)

    # 3. Call LLM
    response = await self.provider.complete(prompt)

    # 4. Parse response to artifact
    artifact = parse_artifact(response, stage_name)

    # 5. Validate against schema
    errors = validate_artifact(artifact, stage_name)
    if errors:
        return StageResult(
            stage=stage_name,
            status="failed",
            errors=errors
        )

    # 6. Write artifact
    write_artifact(artifact, self.project_path)

    return StageResult(stage=stage_name, status="completed", ...)
```

---

## Concurrency Model

The pipeline is **single-threaded by design**:

- One stage runs at a time
- No parallel branch generation (sequential for context coherence)
- No concurrent artifact writes

This simplicity is intentional. Parallelism introduces:
- Race conditions in artifact references
- Context coherence issues (branches need prior branches)
- Complex error recovery

---

## See Also

- [00-vision.md](./00-vision.md) — Core philosophy
- [03-grow-stage-specification.md](./03-grow-stage-specification.md) — GROW details
- [05-prompt-compiler.md](./05-prompt-compiler.md) — Prompt compilation
- [12-getting-started.md](./12-getting-started.md) — Implementation order
