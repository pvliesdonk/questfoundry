# QuestFoundry v3: Integrated Cartridge Architecture

**Status:** Master Blueprint
**Version:** 3.1.0

---

## 1. Overview

QuestFoundry v3 replaces the layered specification model (L0-L5) with an **Integrated Domain Model** where MyST (Markedly Structured Text) documents serve as both human-readable documentation and machine-executable configuration.

### Core Principle

```
MyST Source ─┬─(Compile)─→ Generated Code (Strict Constraints)
             │                       ↓
             └─(Ingest)──→ Agent Runtime (Context & Heuristics)
                                     │
                        Single Source of Truth
```

MyST files serve **two runtime purposes**:

1. **Compiled** → Python code (role configs, artifact models, enums)
2. **Ingested** → Agent context (decision heuristics, anti-patterns, prose)

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
│   ├── roles/              # 8 role definitions (Hybrid: Config + Handbook)
│   ├── loops/              # Content workflows (Hybrid: Graph + Guidance)
│   ├── playbooks/          # Operational procedures (Recovery, Setup, Git Ops)
│   ├── principles/         # Core constraints (Spoiler Hygiene, PN Safety)
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

### Loop vs Playbook Distinction

> **Rule:** If it generates Content Artifacts (Scene, Lore, Hook), it is a **Loop**.
> If it manages Studio Operations (Recovery, Setup, Git Ops), it is a **Playbook**.

| Type | Purpose | Examples |
|------|---------|----------|
| **Loops** | Content workflows that produce artifacts | story_spark, hook_harvest, scene_weave, canon_commit |
| **Playbooks** | Operational procedures for recovery/setup | emergency_retcon, gate_failure, role_stuck, hot_to_cold |

Loops absorb the "checklist" content that was in v2 playbooks — see §6.2 for the Hybrid Loop pattern.

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

Role files are **Hybrid Documents**. They contain MyST directives for the runtime (compiled to code) interleaved with prose for the Agent's system prompt (RAG/Context) and human operators.

#### Hybrid Structure Pattern

```markdown
# Showrunner

> **Mandate:** Manage by Exception. The Showrunner acts as the product owner,
> orchestrating work across specialist roles without doing domain work directly.

:::{role-meta}
id: showrunner
abbr: SR
archetype: Product Owner
agency: high
mandate: "Manage by Exception"
:::

## Operational Guidelines (Human & Agent Context)

**Decision Heuristics:**
- If a loop is stuck > 2 turns, intervene directly.
- Prefer `delegate_to` over doing work yourself.
- When in doubt, ask Gatekeeper for a quality check.

**Anti-Patterns:**
- Micro-managing prose (leave that to Scene Smith).
- Modifying `cold_store` without a Gatecheck.
- Delegating to multiple roles in parallel (causes conflicts).

**Wake Signals:**
- New user request arrives
- Delegation returns with `status: blocked`
- Gatecheck fails

**Escalation Triggers:**
- Canon contradiction detected (escalate to Lorekeeper)
- Quality bar failure (escalate to Gatekeeper)

## Configuration

:::{role-tools}
- delegate_to: "Delegate work to a specialist role"
- read_artifact: "Read an artifact from store"
- write_artifact: "Write an artifact to hot store"
- request_gatecheck: "Request quality validation"
- merge_to_cold: "Promote artifacts to cold store"
- terminate: "End the session"
:::

:::{role-constraints}
- MUST NOT modify cold_store directly (use merge_to_cold)
- MUST delegate domain-specific work to specialist roles
- SHOULD wake only the roles needed for current work
:::

:::{role-prompt}
You are the {{ role.archetype }}, responsible for {{ role.mandate }}.

Refer to "Operational Guidelines" above for decision heuristics and anti-patterns.

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

#### Directive Reference

##### `{role-meta}`

Role identity and classification.

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique role identifier |
| `abbr` | string | yes | 2-letter abbreviation |
| `archetype` | string | yes | Human-readable role type |
| `agency` | enum | yes | `high`, `medium`, `low`, `zero` |
| `mandate` | string | yes | One-line mission statement |

##### `{role-tools}`

Tools available to this role. List of `name: description` pairs.

##### `{role-constraints}`

Hard rules the role must follow. Use RFC 2119 keywords (MUST, SHOULD, MAY).

##### `{role-prompt}`

The system prompt template. Supports Jinja2 templating for dynamic content.

---

### 6.2 Loop Directives (`domain/loops/*.md`)

Loop files are **Hybrid Documents** (like Roles). They contain MyST directives for the execution graph interleaved with guidance prose that SR uses for decision-making. This consolidates the v2 "Loop + Playbook" pattern into a single source of truth.

#### Hybrid Loop Pattern

```markdown
# Story Spark Loop

> **Goal:** Create meaningful nonlinearity from a story seed.
> **Outcome:** Draft topology with scenes, choices, and quality gates.

:::{loop-meta}
id: story_spark
name: "Story Spark"
trigger: user_request
entry_point: showrunner
exit_point: gatekeeper
:::

## Guidance (Formerly Playbook Checklist)

**When to trigger:**
- New chapter/story request from user
- Fix reachability issues flagged by Gatekeeper
- Expand existing hub with new branches

**Success criteria:**
- At least 2 meaningful choices per scene
- No dead-end paths without terminal markers
- All choices have consequence text

**Common failure modes:**
- Single linear path (lacks nonlinearity)
- Orphaned scenes with no incoming edges
- Choices that don't affect downstream content

## Execution Graph

:::{graph-node}
id: plotwright
role: plotwright
timeout: 300
max_iterations: 5
:::

:::{graph-node}
id: scene_smith
role: scene_smith
timeout: 600
max_iterations: 10
:::

:::{graph-edge}
source: plotwright
target: scene_smith
condition: "intent.status == 'topology_complete'"
:::

:::{graph-edge}
source: scene_smith
target: gatekeeper
condition: "intent.status == 'scenes_drafted'"
:::

## Quality Gates

:::{quality-gate}
before: scene_smith
role: gatekeeper
bars: [reachability, nonlinearity]
blocking: true
:::
```

#### Directive Reference

##### `{loop-meta}`

Loop identity and trigger conditions.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique loop identifier |
| `name` | string | yes | Human-readable name |
| `trigger` | string | yes | What starts this loop |
| `entry_point` | string | yes | First role to activate |
| `exit_point` | string | no | Final role (defaults to entry) |

##### `{graph-node}`

A node in the workflow graph.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Node identifier |
| `role` | string | yes | Role to execute |
| `timeout` | int | no | Max seconds (default: 300) |
| `max_iterations` | int | no | Max LLM calls (default: 10) |

##### `{graph-edge}`

A transition between nodes.

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

Define a quality bar and its exception handling.

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
waiver_policy: "Allowed only for Retcons approved by SR"
:::
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique bar identifier |
| `name` | string | yes | Human-readable name |
| `description` | string | yes | What this bar validates |
| `checks` | list | yes | Conditions that must pass |
| `failures` | list | yes | Known failure modes |
| `waiver_policy` | string | no | When exceptions are allowed |

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

## 12. v2 → v3 Migration Status

> **Status:** Triaged
> **Last Updated:** 2025-12-08
> **Purpose:** Categorized inventory of v2→v3 migration status

This section documents the migration status from `_archive/spec/` (v2) to v3. Items are categorized as:

- **Fixed**: Previously blocked, now implemented
- **Intentional**: v2 patterns deliberately not carried forward (design decision)
- **Critical**: Should be addressed for v3 completeness
- **Backlog**: Important but not blocking
- **Informational**: Useful reference, low priority

---

### 12.1 Gap Summary Matrix

| Category | v2 Content | v3 Content | Triaged Status | Action |
|----------|------------|------------|----------------|--------|
| **Protocol Envelope** | 1 spec | DelegationResult + Intent | ✅ FIXED | Implemented differently |
| **YAML Definitions** | 13 loops | MyST files | ✅ INTENTIONAL | Format change to MyST |
| **Runtime Spec** | 6+ docs | 2,600 lines code | ✅ INTENTIONAL | Code is the spec |
| **Protocol Flows** | 13 flows | SR dynamic routing | ✅ INTENTIONAL | SR handles routing |
| **Role Consolidation** | 15 charters | 8 role files | ✅ INTENTIONAL | Consolidated 15→8 |
| **Role Briefs/Interfaces** | 30 docs | MyST directives | ✅ INTENTIONAL | Encoded in role files |
| **Role Depth** | Detailed prose | Hybrid pattern (§6.1) | 🟢 MITIGATED | Migrate v2 content into Hybrid files |
| **Loops + Playbooks** | 13 loops + 14 playbooks | Hybrid Loops (§6.2) | 🟢 MITIGATED | Content workflows → Loops; Ops → Playbooks |
| **Principles** | 10+ docs | 0 | 🔴 CRITICAL | Migrate 4 priority docs |
| **Quality Bars** | Detailed | Summary only | 🔴 CRITICAL | Add waiver process |
| **Artifacts** | 40+ types | 7 types | 🟡 BACKLOG | Add 4 priority types |
| **Lifecycles** | 5 lifecycles | 1 partial | 🟡 BACKLOG | Add hook lifecycle |
| **Glossary** | 200+ terms | ~20 terms | 🟡 BACKLOG | Lightweight version |
| **Protocol Intents** | 30+ intents | ~15 intents | ✅ ACCEPTABLE | Core set sufficient |

---

### 12.2 Fixed: Protocol Envelope → DelegationResult

The v2 "BLOCKER" for Protocol Envelope has been **resolved**. v3 uses a different but complete protocol model.

**v2 Approach:** Agent-to-agent messaging with envelope headers, routing, TTL, priority.

**v3 Approach:** SR-orchestrated handoff with structured models:

```python
# runtime/state.py
class DelegationResult(BaseModel):
    role_id: str           # Who did the work
    status: str            # completed | blocked | needs_review | error
    artifacts: list[str]   # IDs of artifacts created/modified
    message: str           # Summary of work done
    recommendation: str | None  # Suggested next step

class Intent(BaseModel):
    type: Literal["handoff", "escalation", "broadcast", "terminate"]
    source_role: str
    status: str
    payload: dict | None
    reason: str | None
    artifact_ids: list[str]
    timestamp: datetime
```

**Why this works:** In v3, roles don't communicate directly — they return structured results to SR, who decides next steps. The `DelegationResult` and `Intent` models provide all necessary structure for this pattern.

---

### 12.3 Intentional Design Changes

These are **not gaps** — they are deliberate v3 design decisions.

#### 12.3.1 MyST Replaces YAML

v2 used separate YAML definition files. v3 encodes the same information in MyST directives within documentation files. This is the v3 design — "documentation is configuration."

#### 12.3.2 Code Is The Spec (Runtime)

v2 had specification documents for runtime components. v3 has working implementation (~2,600 lines across 26 files) that serves as the de facto specification. Key files:

- `runtime/orchestrator.py` (400 lines) — SR hub-and-spoke pattern
- `runtime/state.py` (450 lines) — StudioState, DelegationResult, Intent
- `runtime/executor.py` (530 lines) — Tool execution loop
- `runtime/providers/` — Ollama, Google, OpenAI integrations

#### 12.3.3 SR Orchestration Replaces Agent-to-Agent Flows

v2 defined message choreography between roles. v3 uses SR as the sole orchestrator — roles don't message each other, they return to SR who decides next steps. "Flows" in v3 are emergent from SR's decision-making.

#### 12.3.4 Role Consolidation (15 → 8)

v2 had 15 roles with separate charter/brief/interface documents. v3 consolidates to 8 roles, each with a single MyST file containing `{role-meta}`, `{role-tools}`, `{role-constraints}`, `{role-prompt}` directives.

**Absorbed roles:**

- Translator → Publisher
- Researcher → Lorekeeper
- Illustrator, Audio Producer → Creative Director

#### 12.3.5 Depth Reduction (Mitigated via Hybrid Structure)

v3 role files were ~70% smaller than v2 charters. This gap is now **mitigated** by defining the Hybrid Document pattern (see §6.1).

**Solution:** v3 Role files adopt a "Hybrid" format:

- **Directives** — Compiled to Python configuration (Identity, Tools, Constraints)
- **Prose sections** — Ingested as "System Knowledge" for the Agent's context window

**Content to migrate from v2 charters:**

- Anti-patterns and example dialogues
- Escalation triggers and wake signals
- Decision heuristics
- Consultation rules and pair guides
- Failure modes and error handling guidance

**Why this works:** MyST files serve dual purposes:

1. **Machine-consumable** — Compiled to code for runtime
2. **Human-readable** — Documentation for studio operators + Agent context via RAG

**Status:** Architecture defined (§6.1). Implementation requires migrating v2 content into Hybrid role files.

---

### 12.4 Critical Gaps (Need Addressing)

#### 12.4.1 Principles & Policies

**Priority docs to migrate** (create `domain/principles/`):

| Document | v2 Path | Why Critical |
|----------|---------|--------------|
| **EVERGREEN_MANUSCRIPT.md** | `00-north-star/EVERGREEN_MANUSCRIPT.md` | Defines view/export model: "we ship views, not finals" |
| **SPOILER_HYGIENE.md** | `00-north-star/SPOILER_HYGIENE.md` | Hot/Cold separation rules — player safety |
| **PN_PRINCIPLES.md** | `00-north-star/PN_PRINCIPLES.md` | Player-Narrator interaction rules |
| **SOURCES_OF_TRUTH.md** | `00-north-star/SOURCES_OF_TRUTH.md` | Canon authority hierarchy |

**Informational docs** (useful nuggets, low priority):

- NORTH_STAR.md, INCIDENT_RESPONSE.md, TRACEABILITY.md, ACCESSIBILITY.md
- COLD_SOT_FORMAT.md, QUALITY_BARS_DETAIL.md
- PATTERNS/ and ANTI-PATTERNS/ directories
- POLICIES/ (HOOK, GATE, ESCALATION, DORMANCY)

#### 12.4.2 Loops + Playbooks (Consolidated)

**Architecture decision:** "Loops Eat Playbooks" for content workflows (see §2, §6.2).

| v2 Content | v3 Destination | Rationale |
|------------|----------------|-----------|
| new_story playbook | → `story_spark` loop | Content workflow → Hybrid Loop |
| gate_failure playbook | → `domain/playbooks/` | Operational recovery → Playbook |
| hot_to_cold playbook | → `canon_commit` loop | Content workflow → Hybrid Loop |
| emergency_retcon | → `domain/playbooks/` | Operational recovery → Playbook |
| role_stuck | → `domain/playbooks/` | Operational recovery → Playbook |

**Content workflows (become Hybrid Loops):**

- story_spark, hook_harvest, scene_weave, canon_commit
- lore_deepening, codex_expansion, style_tune_up

**Operational procedures (remain as Playbooks):**

- gate_failure, emergency_retcon, role_stuck, world_genesis

#### 12.4.3 Quality Bar Detail

v3 `quality_bars.md` covers the 8 bars but is missing:

| Component | Status | Action Needed |
|-----------|--------|---------------|
| Waiver process | ❌ Missing | Add exception handling section |
| Anti-patterns | ❌ Missing | Add common mistakes |
| Bar interaction rules | ❌ Missing | Add precedence guidance |
| Specific failure examples | ⚠️ Generic | Add real examples |

---

### 12.5 Backlog (Important but Not Blocking)

#### 12.5.1 Hybrid Loops (Content Workflows)

**Architecture:** Loops now absorb playbook "checklist" content (see §6.2). Each loop is a Hybrid Document with graph + guidance.

**Priority loops to migrate as Hybrid Documents:**

| Loop ID | Absorbs Playbook | Purpose |
|---------|------------------|---------|
| `canon_commit` | hot_to_cold | Stabilize hot → cold |
| `hook_harvest` | — | Extract change hooks during play |
| `scene_weave` | — | Compose scenes from beats |
| `quality_gate` | — | Run quality bar checks |

**Other loops** (defer until needed):

- choice_tree, lore_sync, draft_review, character_arc
- timeline_build, world_expand, plot_refine, texture_finish, lore_deepening

**Note:** v3 loops are "guidance for SR" not executable graphs. story_spark (§6.2) is the Hybrid template.

#### 12.5.2 Artifacts (35 of 40 Missing)

**Priority artifacts to add:**

| Artifact | Why Priority |
|----------|--------------|
| `Character` | Essential for narrative |
| `Location` | Essential for world-building |
| `PlotPoint` | Structural narrative |
| `Timeline` | Temporal consistency |

**Current v3 artifacts (7):** HookCard, Brief, Scene, GatecheckReport, Lore, GameState, ActualChoice

**Probably NOT needed** (v2 process overhead):

- meeting_minutes, post_mortem_report (process docs)
- style_addendum, audio_plan (absorbed role artifacts)
- section_draft (overlaps with Scene), codex_entry (overlaps with Lore)

#### 12.5.3 Lifecycles (4 of 5 Missing)

**Priority:** Add hook lifecycle (important for hook_harvest loop)

**Defer:** tu (Trace Unit), gate, view lifecycles

**Current:** `lifecycles/artifact.md` exists with states and transitions

#### 12.5.4 Glossary/Taxonomy

Create lightweight glossary with ~30 essential terms. Don't migrate all 200+.

---

### 12.6 Informational Archive

The following v2 content is available in `_archive/spec/` for reference. These contain useful nuggets but are low priority for migration.

#### 12.6.1 v2 Directory Reference

| v2 Path | Content | Notes |
|---------|---------|-------|
| `00-north-star/` | Principles, policies, patterns | See 12.4.1 for priority docs |
| `00-north-star/LOOPS/` | 13 loop definitions | See 12.5.1 for priority loops |
| `00-north-star/PLAYBOOKS/` | 14+ playbooks | See 12.4.2 for priority playbooks |
| `01-roles/charters/` | 15 role charters | Consolidated into 8 MyST role files |
| `01-roles/briefs/` | Role summaries | Encoded in `{role-meta}` directives |
| `01-roles/interfaces/` | Role APIs | Encoded in `{role-tools}` directives |
| `02-dictionary/artifacts/` | 40+ artifact templates | See 12.5.2 for priority artifacts |
| `02-dictionary/glossary.md` | 200+ terms | See 12.5.4 |
| `04-protocol/ENVELOPE.md` | Message envelope | Replaced by DelegationResult (12.2) |
| `04-protocol/FLOWS/` | Message choreography | Replaced by SR orchestration (12.3.3) |
| `04-protocol/LIFECYCLES/` | State machines | See 12.5.3 |
| `05-definitions/` | YAML definitions | Replaced by MyST (12.3.1) |
| `06-runtime/` | Runtime specs | Code is spec (12.3.2) |

#### 12.6.2 Useful Nuggets by Category

**Principles with lasting value:**

- NORTH_STAR.md — Core vision (philosophy)
- INCIDENT_RESPONSE.md — Error handling patterns
- TRACEABILITY.md — Audit trail concepts
- ACCESSIBILITY.md — Accessibility considerations

**Artifact field depth:**

v2 artifacts had 50% more fields than v3. Reference when enriching:

| Artifact | v2 Fields | v3 Fields |
|----------|-----------|-----------|
| HookCard | 15 | 8 |
| Scene | 20 | 10 |
| Brief | 12 | 6 |

**Role enrichment material:**

v2 charters included content not in v3 roles:

- Anti-patterns and example dialogues
- Escalation triggers and dormancy signals
- Per-loop participation behavior
- Consultation rules and pair guides

---

### 12.7 Resolution Strategy (Updated)

**Priority order based on triage:**

**Phase 1: Critical Gaps** (immediate value)

1. Create `domain/principles/` with 4 priority docs (SPOILER_HYGIENE, PN_PRINCIPLES, etc.)
2. Create `domain/playbooks/` with 4 operational procedures (gate_failure, emergency_retcon, role_stuck, world_genesis)
3. Add `waiver_policy` field to quality bars in `protocol/quality_bars.md`
4. Migrate v2 charter content into Hybrid role files (anti-patterns, wake signals, heuristics)

**Why content migration is Phase 1:** MyST files serve dual purposes (§1). Agents need decision heuristics in context. Human operators need prose depth.

**Phase 2: Backlog** (when needed)

1. Migrate 4 priority Hybrid Loops (canon_commit, hook_harvest, scene_weave, quality_gate) — absorbing playbook checklists
2. Add 4 priority artifacts (Character, Location, PlotPoint, Timeline)
3. Add hook lifecycle
4. Create lightweight glossary

**Phase 3: Enrichment** (polish)

1. Expand artifact field definitions
2. Migrate remaining useful v2 content as needed

**No longer blockers:**

- ~~Protocol Envelope~~ → Fixed (DelegationResult)
- ~~YAML Definitions~~ → Intentional (MyST replaces YAML)
- ~~Runtime Spec~~ → Intentional (code is spec)
- ~~Protocol Flows~~ → Intentional (SR dynamic routing)

---

## 13. Implementation Phases

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

### Phase 7: Content Migration (Critical)

**Principles:**

- [ ] Create `domain/principles/` directory
- [ ] Migrate `SPOILER_HYGIENE.md` (Critical for Gatekeeper)
- [ ] Migrate `PN_PRINCIPLES.md` (Critical for Narrator)
- [ ] Migrate `EVERGREEN_MANUSCRIPT.md` (Views/export model)
- [ ] Migrate `SOURCES_OF_TRUTH.md` (Canon authority)

**Playbooks (Operational procedures only — content workflows → Loops):**

- [ ] Create `domain/playbooks/` directory
- [ ] Migrate `gate_failure.md` (Recovery procedure)
- [ ] Migrate `emergency_retcon.md` (Canon correction)
- [ ] Migrate `role_stuck.md` (Agent recovery)
- [ ] Migrate `world_genesis.md` (Project setup)

**Hybrid Content:**

- [ ] Update `domain/roles/*.md` with Hybrid content (prose + directives)
- [ ] Add `waiver_policy` to quality bars

### Phase 8: Hybrid Loop Migration (Content Workflows)

**Note:** Loops now absorb playbook "checklist" content. See §6.2 for Hybrid Loop pattern.

**Migrated Loops (1/12):**

- story_spark (template for Hybrid pattern)

**Priority Loops (absorb playbooks):**

- canon_commit (absorbs hot_to_cold playbook)
- hook_harvest, scene_weave, quality_gate

**Remaining Loops:**

- choice_tree, lore_sync, draft_review, character_arc
- timeline_build, world_expand, plot_refine, lore_deepening

### Phase 9: Advanced Role Configuration (Roadmap)

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
| **Protocol** | 🟡 Partial | Core v3 protocol: intents, delegation, quality_bars, artifact lifecycle |
| **Quality Bars** | ✅ Complete | 8 bars defined in `protocol/quality_bars.md` |
| **Loops** | 🔴 HIGH | 1/13 migrated (story_spark only) |
| Role Agents | ✅ Complete | All 8 roles executable (depth reduced 75%) |
| Artifacts | 🟡 Partial | 5/37+ migrated |
| Orchestrator | ✅ Complete | SR-centric handoff working |
| Tool Calling | ✅ Complete | Ollama, Google, OpenAI |
| CLI | ✅ Complete | `qf ask/doctor/config/roles` with -v flags |
| Logging | ✅ Complete | Rich console + structured JSONL |
| Tests | ✅ Complete | Integration + unit tests |
| Persistence | ⬜ Not Started | Checkpointing deferred |
| Role Config | ⬜ Roadmap | Per-role model/provider/temp |

### Next Steps (Priority Order)

1. ~~**Phase A: Protocol Foundation**~~ ✅ — Core v3 protocol created (intents, delegation, quality_bars, artifact lifecycle)
2. **Phase B: Loop Migration** — Migrate remaining 12 loops from v2
3. ~~**Phase C: Quality Bars**~~ ✅ — 8 bars defined in `protocol/quality_bars.md`
4. **Phase D: Role Enrichment** — Add consultation guides, expand constraints
5. **Phase E: Ontology Completion** — Migrate remaining 32+ artifact types

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
| `domain/playbooks/` | `{procedure}.md` | `gate_failure.md`, `hot_to_cold.md` |
| `domain/principles/` | `{topic}.md` | `spoiler_hygiene.md`, `evergreen.md` |
| `domain/ontology/` | `{concept}.md` | `artifacts.md`, `taxonomy.md` |
| `domain/protocol/` | `{aspect}.md` | `intents.md`, `routing.md` |
| `generated/models/` | `{category}.py` | `artifacts.py`, `enums.py` |
| `generated/roles/` | `{role_id}.py` | `showrunner.py` |

**Notes:**

- Loop definitions are NOT compiled to `generated/graphs/`. They serve as Hybrid Documents (guidance + graph metadata).
- Playbooks are operational procedures, not compiled. They are ingested as agent context.
- Principles are reference documents, not compiled. They define policy and philosophy.
