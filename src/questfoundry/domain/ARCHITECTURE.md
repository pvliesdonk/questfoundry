# QuestFoundry v3: Integrated Cartridge Architecture

**Status:** Master Blueprint
**Version:** 3.1.0

---

## 1. Overview

QuestFoundry v3 replaces the layered specification model (L0-L5) with an **Integrated Domain Model** where MyST (Markedly Structured Text) documents serve as both human-readable documentation and machine-executable configuration.

### Core Principle

```
MyST Source → Compiler → Generated Code → Runtime
     ↑                                        |
     └────────── Single Source of Truth ──────┘
```

### Key Changes from v2

| Aspect | v2 | v3 |
|--------|----|----|
| Structure | 7 numbered layers | 4 vertical domain modules |
| Authoring | Prose + YAML/JSON separately | MyST (prose = config) |
| Roles | 15 roles | 8 archetypes |
| Protocol | Agent-to-agent message passing | **SR-orchestrated handoff** |
| Runtime | LangChain + LangGraph | **Handoff-based orchestration** |

### Orchestration Model

QuestFoundry uses **Showrunner-centric orchestration** where:

- **SR is the hub** — all delegation flows through the Showrunner
- **Roles are specialists** — each role has its own agent, prompt, and tools
- **Loops are guidance** — not hardcoded graphs, but heuristics SR uses to decide routing
- **Dynamic delegation** — SR decides at runtime who to delegate to based on context

```
                    ┌─────────────────┐
                    │   Showrunner    │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │ delegate_to(role, task)
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ Plotwright │    │ Lorekeeper │    │  Narrator  │
    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │ returns result + intent
                            ▼
                    ┌─────────────────┐
                    │   Showrunner    │
                    │ (decides next)  │
                    └─────────────────┘
```

This differs from static graph approaches (LangGraph, etc.) where edges are predetermined at build time.

---

## 2. Directory Structure

```
src/questfoundry/
├── domain/                 # MyST Source of Truth
│   ├── roles/              # 8 role definitions
│   ├── loops/              # Workflow guidance (heuristics for SR)
│   ├── ontology/           # Data structures
│   └── protocol/           # Communication rules
│
├── compiler/               # MyST → Generated Code
│   ├── parser/             # MyST directive extraction
│   ├── models/             # Intermediate representation
│   └── generators/         # Code generators
│
├── generated/              # Checked-in generated code
│   ├── models/             # Pydantic models (artifacts, enums)
│   └── roles/              # Role configurations
│
└── runtime/                # Execution engine
    ├── orchestrator.py     # SR-based handoff orchestration
    ├── state.py            # StudioState
    ├── roles.py            # Role agent execution
    └── providers/          # LLM SDK wrappers
```

---

## 3. The Eight Roles

### Role Roster

| # | Role | Abbr | Archetype | Agency | Mandate |
|---|------|------|-----------|--------|---------|
| 1 | **Showrunner** | SR | Product Owner | High (Strategic) | "Manage by Exception" |
| 2 | **Lorekeeper** | LK | Librarian | Medium (Consistency) | "Maintain the Truth" |
| 3 | **Narrator** | NR | Dungeon Master | High (Improvisational) | "Run the Game" |
| 4 | **Publisher** | PB | Book Binder | Zero (Deterministic) | "Assemble the Artifact" |
| 5 | **Creative Director** | CD | Visionary | High (Aesthetic) | "Ensure Sensory Coherence" |
| 6 | **Plotwright** | PW | Architect | Medium (Structural) | "Design the Topology" |
| 7 | **Scene Smith** | SS | Writer | Medium (Creative) | "Fill with Prose" |
| 8 | **Gatekeeper** | GK | Auditor | Low (Validation) | "Enforce Quality Bars" |

### Role Consolidations

| v3 Role | Merged From (v2) |
|---------|------------------|
| Lorekeeper | Lore Weaver + Codex Curator |
| Creative Director | Style Lead + Art Director + Audio Director |
| Publisher | Book Binder (refined: zero agency) |
| Narrator | Player-Narrator (refined: high agency) |

### Agency Levels

- **High**: Can deviate, improvise, make judgment calls
- **Medium**: Follows patterns but has domain discretion
- **Low**: Applies rules mechanically
- **Zero**: Purely deterministic; crashes on ambiguity

---

## 4. State Model

### StudioState (LangGraph Native)

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class StudioState(TypedDict):
    """Central state object for all graph execution."""

    # Artifact stores
    hot_store: dict[str, Artifact]      # Working drafts (mutable)
    cold_store: dict[str, Artifact]     # Committed canon (append-only)

    # Message history (LangGraph pattern)
    messages: Annotated[list, add_messages]

    # Routing
    current_role: str                   # Active role ID
    pending_intents: list[Intent]       # Queued work

    # Context
    loop_id: str                        # Current loop
    iteration: int                      # Step counter
```

### Hot vs Cold Semantics

| Store | Mutability | Purpose | Example |
|-------|------------|---------|---------|
| **hot_store** | Read/Write | Work in progress | Draft scene, proposed hooks |
| **cold_store** | Append-only | Committed canon | Published chapters, finalized lore |

### Stabilization Path

```
hot_store (draft)
    → Gatekeeper approval
    → cold_store (canon)
```

---

## 5. Protocol: SR-Orchestrated Handoff

### Delegation Model

The Showrunner (SR) is the sole orchestrator. Other roles do not call each other — they return results to SR, who decides the next action.

```python
class DelegationResult(BaseModel):
    """Result returned by a role after completing delegated work."""

    role_id: str                        # Which role executed
    status: str                         # e.g., "completed", "blocked", "needs_review"
    artifacts: list[str]                # IDs of artifacts created/modified
    message: str                        # Summary for SR
    recommendation: str | None = None   # Suggested next action
```

### Orchestration Flow

1. **SR receives request** → analyzes what needs to be done
2. **SR delegates** → `delegate_to("plotwright", task="Design topology for mystery story")`
3. **Role executes** → works independently, returns `DelegationResult`
4. **SR evaluates** → reads result, decides next action
5. **SR delegates again** → or terminates if work is complete
6. **Repeat** until SR decides to terminate

### SR's Decision Tools

SR has tools for orchestration:

```python
# Delegation
delegate_to(role: str, task: str) -> DelegationResult

# State management
read_artifact(artifact_id: str) -> Artifact
write_artifact(artifact: Artifact) -> str

# Quality gates
request_gatecheck(artifact_ids: list[str]) -> GatecheckResult

# Lifecycle
merge_to_cold(artifact_ids: list[str]) -> MergeResult
terminate(reason: str) -> None
```

### Loop Definitions as Guidance

Loop definitions (in `domain/loops/*.md`) are **not compiled to executable graphs**. They serve as:

1. **Documentation** — human-readable workflow descriptions
2. **SR guidance** — heuristics SR can reference when deciding routing
3. **Validation** — ensuring all roles/transitions are defined

SR reads loop guidance but makes dynamic decisions based on actual context.

---

## 6. MyST Directive Vocabulary

All domain knowledge is encoded in MyST files using custom directives.

### 6.1 Role Directives (`domain/roles/*.md`)

#### `{role-meta}`

Role identity and classification.

```markdown
:::{role-meta}
id: showrunner
abbr: SR
archetype: Product Owner
agency: high
mandate: "Manage by Exception"
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique role identifier |
| `abbr` | string | yes | 2-letter abbreviation |
| `archetype` | string | yes | Human-readable role type |
| `agency` | enum | yes | `high`, `medium`, `low`, `zero` |
| `mandate` | string | yes | One-line mission statement |

#### `{role-tools}`

Tools available to this role.

```markdown
:::{role-tools}
- read_state: "Read from hot_store or cold_store"
- write_state: "Write to hot_store"
- post_intent: "Declare work status"
- query_lore: "Search canon for facts"
:::
```

#### `{role-constraints}`

Hard rules the role must follow.

```markdown
:::{role-constraints}
- MUST NOT modify cold_store directly
- MUST post intent after completing work
- SHOULD escalate if blocked > 2 iterations
:::
```

#### `{role-prompt}`

The system prompt template for this role.

```markdown
:::{role-prompt}
You are the {{ role.archetype }}, responsible for {{ role.mandate }}.

## Your Tools
{% for tool in role.tools %}
- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}

## Constraints
{% for c in role.constraints %}
- {{ c }}
{% endfor %}
:::
```

---

### 6.2 Loop Directives (`domain/loops/*.md`)

#### `{loop-meta}`

Loop identity and trigger conditions.

```markdown
:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
exit_point: gatekeeper
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique loop identifier |
| `name` | string | yes | Human-readable name |
| `trigger` | string | yes | What starts this loop |
| `entry_point` | string | yes | First role to activate |
| `exit_point` | string | no | Final role (defaults to entry) |

#### `{graph-node}`

A node in the workflow graph.

```markdown
:::{graph-node}
id: plotwright
role: plotwright
timeout: 300
max_iterations: 5
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Node identifier |
| `role` | string | yes | Role to execute |
| `timeout` | int | no | Max seconds (default: 300) |
| `max_iterations` | int | no | Max LLM calls (default: 10) |

#### `{graph-edge}`

A transition between nodes.

```markdown
:::{graph-edge}
source: plotwright
target: scene_smith
condition: "intent.status == 'topology_complete'"
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | Source node ID |
| `target` | string | yes | Target node ID |
| `condition` | string | yes | Python expression on `intent` |

#### `{quality-gate}`

A quality checkpoint in the loop.

```markdown
:::{quality-gate}
before: scene_smith
role: gatekeeper
bars: [integrity, reachability]
blocking: true
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `before` | string | yes | Node this gate precedes |
| `role` | string | yes | Role performing validation |
| `bars` | list | yes | Quality bars to check |
| `blocking` | bool | yes | If true, failure halts loop |

---

### 6.3 Ontology Directives (`domain/ontology/*.md`)

#### `{artifact-type}`

Define an artifact schema.

```markdown
:::{artifact-type}
id: hook_card
name: "Hook Card"
store: hot
lifecycle: [proposed, accepted, in_progress, resolved, canonized]
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Artifact type identifier |
| `name` | string | yes | Human-readable name |
| `store` | enum | yes | `hot`, `cold`, `both` |
| `lifecycle` | list | no | Valid status values |

#### `{artifact-field}`

A field within an artifact type.

```markdown
:::{artifact-field}
artifact: hook_card
name: hook_type
type: HookType
required: true
description: "The category of change this hook represents"
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `artifact` | string | yes | Parent artifact ID |
| `name` | string | yes | Field name |
| `type` | string | yes | Type reference (primitive or enum) |
| `required` | bool | yes | Is this field mandatory? |
| `description` | string | no | Field documentation |

#### `{enum-type}`

Define an enumeration.

```markdown
:::{enum-type}
id: HookType
values:
  - narrative: "Changes to story content"
  - scene: "New or modified scenes"
  - factual: "Canon facts and lore"
  - taxonomy: "Terminology and naming"
  - structure: "Topology changes"
:::
```

#### `{enum-value}` (alternative inline form)

```markdown
:::{enum-value}
enum: HookType
value: narrative
description: "Changes to story content"
:::
```

---

### 6.4 Protocol Directives (`domain/protocol/*.md`)

#### `{intent-type}`

Define a valid intent.

```markdown
:::{intent-type}
id: handoff
description: "Role completed work, ready for next step"
fields:
  - status: string
  - artifact_ids: list[string]
:::
```

#### `{routing-rule}`

Global routing logic.

```markdown
:::{routing-rule}
match: "intent.type == 'escalation'"
target: showrunner
priority: 100
:::
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `match` | string | yes | Python expression |
| `target` | string | yes | Role to route to |
| `priority` | int | no | Higher = evaluated first |

#### `{quality-bar}`

Define a quality bar.

```markdown
:::{quality-bar}
id: integrity
name: "Integrity"
description: "No contradictions in canon"
checks:
  - "All facts traceable to sources"
  - "No circular references"
  - "Timeline consistency"
failures:
  - "Orphaned references"
  - "Contradictory statements"
:::
```

---

## 7. Generated Code Patterns

### 7.1 Pydantic Models (`generated/models/`)

From `{artifact-type}` + `{artifact-field}`:

```python
# generated/models/artifacts.py
from pydantic import BaseModel
from .enums import HookType, HookStatus

class HookCard(BaseModel):
    """Auto-generated from domain/ontology/artifacts.md"""

    id: str
    hook_type: HookType
    status: HookStatus = HookStatus.proposed
    title: str
    description: str
    owner: str | None = None
    # ... more fields
```

### 7.2 Role Configs (`generated/roles/`)

From `{role-meta}` + `{role-tools}` + `{role-constraints}` + `{role-prompt}`:

```python
# generated/roles/showrunner.py
from questfoundry.compiler.models import Agency, RoleIR, RoleToolIR

SHOWRUNNER = RoleIR(
    id="showrunner",
    abbr="SR",
    archetype="Product Owner",
    agency=Agency.HIGH,
    mandate="Manage by Exception",
    tools=[
        RoleToolIR(name="delegate_to", description="Delegate work to another role"),
        RoleToolIR(name="read_artifact", description="Read an artifact from store"),
        RoleToolIR(name="write_artifact", description="Write an artifact to hot store"),
        RoleToolIR(name="request_gatecheck", description="Request quality validation"),
        RoleToolIR(name="merge_to_cold", description="Promote artifacts to cold store"),
        RoleToolIR(name="terminate", description="End the session"),
    ],
    constraints=[
        "MUST NOT modify cold_store directly (use merge_to_cold)",
        "MUST delegate domain-specific work to specialist roles",
        "SHOULD wake only the roles needed for current work",
    ],
    prompt_template="""You are the Product Owner, responsible for: Manage by Exception.
...
""",
)
```

### 7.3 Loop Definitions (NOT compiled to graphs)

Loop definitions in `domain/loops/*.md` are **parsed but not compiled to executable code**. They serve as:

1. **Validation source** — compiler checks all referenced roles exist
2. **Documentation** — human-readable workflow guidance
3. **Runtime reference** — SR can query loop metadata for guidance

The compiler extracts loop IR for validation but does not generate graph code:

```python
# compiler/models/ir.py
@dataclass
class LoopIR:
    """Loop intermediate representation — for validation, not execution."""

    id: str
    name: str
    trigger: str
    entry_point: str
    nodes: list[GraphNodeIR]     # Roles involved
    edges: list[GraphEdgeIR]     # Possible transitions (guidance only)
    quality_gates: list[QualityGateIR]
```

At runtime, SR decides routing dynamically — loop edges are suggestions, not constraints.

---

## 8. Runtime Architecture

### 8.1 Core Components

```
┌─────────────────────────────────────────────────────┐
│                    CLI (qf)                         │
├─────────────────────────────────────────────────────┤
│                 Orchestrator                        │
│  ┌─────────────────────────────────────────────┐   │
│  │              Showrunner Agent                │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │  Tools: delegate_to, read_artifact, │    │   │
│  │  │  write_artifact, request_gatecheck, │    │   │
│  │  │  merge_to_cold, terminate           │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│     ┌───────────────┼───────────────┐              │
│     ▼               ▼               ▼              │
│  ┌──────┐       ┌──────┐       ┌──────┐           │
│  │  PW  │       │  LK  │       │  GK  │  ...      │
│  │Agent │       │Agent │       │Agent │           │
│  └──────┘       └──────┘       └──────┘           │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │              StudioState                     │  │
│  │  hot_store | cold_store | messages | ...    │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 8.2 Orchestrator

The orchestrator manages the SR agent and handles delegation to specialist roles:

```python
# runtime/orchestrator.py
class Orchestrator:
    """SR-centric orchestration engine."""

    def __init__(self, roles: dict[str, RoleIR], llm: BaseChatModel):
        self.roles = roles
        self.llm = llm
        self.sr_agent = self._create_sr_agent()

    async def run(self, request: str) -> StudioState:
        """Execute a complete session."""
        state = create_initial_state(request)

        while True:
            # SR decides what to do
            sr_response = await self.sr_agent.invoke(state)

            if sr_response.tool_call == "terminate":
                break

            if sr_response.tool_call == "delegate_to":
                role_id = sr_response.args["role"]
                task = sr_response.args["task"]
                result = await self._execute_role(role_id, task, state)
                state = self._update_state(state, result)

            # ... handle other tools

        return state

    async def _execute_role(
        self, role_id: str, task: str, state: StudioState
    ) -> DelegationResult:
        """Execute a specialist role and return result."""
        role = self.roles[role_id]
        role_agent = self._create_role_agent(role)
        return await role_agent.invoke(task, state)
```

### 8.3 Provider Abstraction

```python
# runtime/providers/base.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> CompletionResult:
        ...

# runtime/providers/ollama.py
class OllamaProvider(LLMProvider):
    """Direct Ollama SDK integration."""

    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def complete(self, messages, tools=None):
        # Direct HTTP to Ollama API
        ...
```

### 8.4 Role Execution

Each role runs as an independent agent with its own conversation history:

```python
# runtime/roles.py
class RoleAgent:
    """Agent for a specialist role."""

    def __init__(self, role: RoleIR, llm: BaseChatModel):
        self.role = role
        self.llm = llm
        self.messages: list[BaseMessage] = []

    async def invoke(self, task: str, state: StudioState) -> DelegationResult:
        """Execute the role's task and return result."""
        # Build role-specific prompt
        system_prompt = self._build_prompt(task, state)

        # Role has its own conversation history
        self.messages.append(SystemMessage(content=system_prompt))
        self.messages.append(HumanMessage(content=task))

        # Execute with role's tools
        response = await self.llm.ainvoke(
            self.messages,
            tools=self._get_role_tools(),
        )

        # Parse response into DelegationResult
        return self._parse_result(response)
```

---

## 9. Build Pipeline

### Compilation Steps

```
1. Parse MyST files (domain/**/*.md)
         ↓
2. Extract directives into IR (Intermediate Representation)
         ↓
3. Validate IR (cross-references, required fields)
         ↓
4. Generate Python code (generated/**/*)
         ↓
5. Format with Ruff
```

### CLI Commands

```bash
# Compile domain → generated
qf compile

# Compile and run a loop
qf run story-spark

# Validate domain without generating
qf validate

# Watch mode (recompile on change)
qf compile --watch
```

---

## 10. Quality Bars (v3)

| Bar | Description | Checked By |
|-----|-------------|------------|
| **Integrity** | No contradictions in canon | Lorekeeper, Gatekeeper |
| **Reachability** | All content accessible via valid paths | Plotwright, Gatekeeper |
| **Nonlinearity** | Multiple valid paths exist | Plotwright |
| **Gateways** | All gates have valid unlock conditions | Plotwright, Gatekeeper |
| **Style** | Voice and tone consistency | Creative Director |
| **Determinism** | Same inputs → same outputs (for Publisher) | Publisher, Gatekeeper |
| **Presentation** | Formatting and structure | Publisher |
| **Accessibility** | Content usable by all players | Gatekeeper |

---

## 11. Migration Notes

### What's Archived

Everything from v2 is in `_archive/`:

- `_archive/spec/` — All L0-L5 specification documents
- `_archive/lib/` — Previous runtime and compiler
- `_archive/cli/` — Previous CLI tools
- `_archive/docs/` — Previous documentation

### Reference Material

When writing v3 domain files, consult:

- `_archive/spec/00-north-star/` — Principles (many still apply)
- `_archive/spec/01-roles/charters/` — Role details to consolidate
- `_archive/spec/02-dictionary/` — Terminology to preserve
- `_archive/spec/05-definitions/` — YAML patterns to learn from

### What's NOT Migrated

- Layered structure (L0-L5)
- 15-role model
- LangChain dependencies
- Protocol envelope format
- Separate prose/config files

---

## 12. Implementation Phases

### Phase 1: Foundation ✓

- [x] `pyproject.toml` with dependencies
- [x] MyST parser for directives
- [x] `StudioState` implementation
- [x] Compiler IR models

### Phase 2: Ontology (Partial)

- [x] `domain/ontology/artifacts.md` (HookCard, Brief, Scene, etc.)
- [x] `domain/ontology/taxonomy.md` (enums)
- [x] Compiler: ontology → Pydantic models

**Migrated Artifacts (5/20+):**

- Brief, CanonEntry, GatecheckReport, HookCard, Scene

**Remaining Artifacts:**

- Act, Beat, Chapter, Character, Choice, Dialogue, Draft, Entity, Event, Fact
- Item, Location, Metadata, Moment, PlotPoint, Prose, Relationship, Timeline, World

### Phase 3: Roles ✓

- [x] All 8 role definitions in `domain/roles/`
- [x] Compiler: roles → RoleIR configurations
- [x] `domain/loops/story_spark.md` (as guidance documentation)

### Phase 4: Orchestrator ✓

- [x] `runtime/orchestrator.py` — SR-centric handoff engine
- [x] `runtime/roles.py` — Role agent execution
- [x] `runtime/executor.py` — ToolExecutor with bind_tools
- [x] SR tools: `delegate_to`, `read_artifact`, `write_artifact`, etc.
- [x] DelegationResult model
- [x] End-to-end test with SR delegating to one role

### Phase 5: Full Integration ✓

- [x] All role tools implemented (consult_*, read/write_hot_store)
- [x] Gatekeeper quality bar evaluation (8 evaluation tools)
- [x] Cold Store implementation (`runtime/cold_store.py`)
- [x] Configuration system (`runtime/config.py` with pydantic-settings)
- [x] Multi-turn delegation test (SR → PW → LK → GK → SR)

### Phase 6: Polish (Current)

- [x] CLI integration (`qf ask`, `qf doctor`, `qf config`, `qf roles`)
- [x] Ollama provider with native tool calling
- [x] Google AI Studio provider (Gemini)
- [x] OpenAI provider (GPT-4o with tool calling)
- [x] ExecutorCallbacks for streaming progress output
- [ ] State persistence/checkpointing
- [ ] Anthropic provider (Claude)

### Phase 7: Loop Migration (Remaining Work)

**Migrated Loops (1/12):**

- story_spark

**Remaining Loops:**

- hook_harvest, scene_weave, choice_tree, lore_sync, draft_review
- canon_commit, character_arc, timeline_build, world_expand
- plot_refine, quality_gate

### Phase 8: Advanced Role Configuration (Roadmap)

Future enhancements for role-specific LLM configuration:

- [ ] Different model temperatures per role (e.g., lower for GK, higher for NR)
- [ ] Different models per role (e.g., GPT-4o for SR, GPT-4o-mini for specialists)
- [ ] Different providers per role (e.g., OpenAI for SR, Ollama for specialists)

**Proposed Configuration Schema:**

```yaml
roles:
  showrunner:
    provider: openai
    model: gpt-4o
    temperature: 0.3
  narrator:
    provider: ollama
    model: qwen3:8b
    temperature: 0.9
  gatekeeper:
    provider: google
    model: gemini-2.0-flash
    temperature: 0.1
```

This allows:

- Cost optimization (expensive models for orchestration, cheap for validation)
- Creative variation (high temperature for narrative, low for quality checks)
- Provider redundancy (fallback across providers)

### Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Orchestrator | ✅ Complete | SR-centric handoff working |
| Role Agents | ✅ Complete | All 8 roles executable |
| Tool Calling | ✅ Complete | Ollama, Google, OpenAI |
| Artifacts | 🟡 Partial | 5/20+ migrated |
| Loops | 🟡 Partial | 1/12 migrated |
| CLI | ✅ Complete | `qf ask/doctor/config/roles` with -v flags |
| Logging | ✅ Complete | Rich console + structured JSONL |
| Tests | ✅ Complete | Integration + unit tests |
| Persistence | ⬜ Not Started | Checkpointing deferred |
| Role Config | ⬜ Roadmap | Per-role model/provider/temp |

---

## Appendix A: Directive Quick Reference

| Directive | Location | Purpose |
|-----------|----------|---------|
| `{role-meta}` | roles/*.md | Role identity |
| `{role-tools}` | roles/*.md | Available tools |
| `{role-constraints}` | roles/*.md | Hard rules |
| `{role-prompt}` | roles/*.md | System prompt template |
| `{loop-meta}` | loops/*.md | Loop identity |
| `{graph-node}` | loops/*.md | Workflow node |
| `{graph-edge}` | loops/*.md | Workflow transition |
| `{quality-gate}` | loops/*.md | Validation checkpoint |
| `{artifact-type}` | ontology/*.md | Data structure |
| `{artifact-field}` | ontology/*.md | Structure field |
| `{enum-type}` | ontology/*.md | Enumeration |
| `{intent-type}` | protocol/*.md | Message type |
| `{routing-rule}` | protocol/*.md | Global routing |
| `{quality-bar}` | protocol/*.md | Quality definition |

---

## Appendix B: File Naming Conventions

| Directory | Pattern | Example |
|-----------|---------|---------|
| `domain/roles/` | `{role_id}.md` | `showrunner.md` |
| `domain/loops/` | `{loop_id}.md` | `story_spark.md` |
| `domain/ontology/` | `{concept}.md` | `artifacts.md`, `taxonomy.md` |
| `domain/protocol/` | `{aspect}.md` | `intents.md`, `routing.md` |
| `generated/models/` | `{category}.py` | `artifacts.py`, `enums.py` |
| `generated/roles/` | `{role_id}.py` | `showrunner.py` |

Note: Loop definitions are NOT compiled to `generated/graphs/`. They serve as documentation and guidance only.
