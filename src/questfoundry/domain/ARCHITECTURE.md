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

## 12. Comprehensive Gap Inventory (v2 → v3)

> **Status:** Complete Audit
> **Last Updated:** 2025-01-20
> **Purpose:** Definitive inventory of all v2 content not migrated to v3

This section documents ALL gaps between `_archive/spec/` (v2) and `src/questfoundry/domain/` (v3). Gaps include:

- **Missing files**: Content that exists in v2 but has no v3 equivalent
- **Depth reduction**: Content that exists in v3 but with significantly less detail
- **Structural changes**: v2 patterns not carried forward to v3

---

### 12.1 Gap Summary Matrix

| Category | v2 Content | v3 Content | Gap % | Priority |
|----------|------------|------------|-------|----------|
| **Principles** | 10+ documents | 0 | 100% | 🔴 HIGH |
| **Loops** | 13 loops | 1 loop | 92% | 🔴 HIGH |
| **Playbooks** | 14+ playbooks | 0 | 100% | 🟡 MEDIUM |
| **Role Briefs** | 15 briefs | 0 | 100% | 🟡 MEDIUM |
| **Role Charters** | 15 charters | 0 | 100% | 🔴 HIGH |
| **Role Interfaces** | 15 interfaces | 0 | 100% | 🟡 MEDIUM |
| **Artifacts** | 40+ types | 5 types | 87% | 🔴 HIGH |
| **Glossary** | 200+ terms | ~20 terms | 90% | 🟡 MEDIUM |
| **Protocol Envelope** | 1 spec | 0 | 100% | 🔴 BLOCKER |
| **Protocol Intents** | 30+ intents | ~15 intents | 50% | 🟡 MEDIUM |
| **Protocol Flows** | 13 flows | 0 | 100% | 🔴 HIGH |
| **Lifecycles** | 5 lifecycles | 1 partial | 80% | 🔴 HIGH |
| **YAML Definitions** | 13 loops | 1 loop | 92% | 🔴 HIGH |
| **Runtime Spec** | 6+ docs | 0 | 100% | 🟡 MEDIUM |

---

### 12.2 Principles & Policies (00-north-star)

#### 12.2.1 Missing Principle Documents

| Document | v2 Path | Purpose | Notes |
|----------|---------|---------|-------|
| **NORTH_STAR.md** | `00-north-star/NORTH_STAR.md` | Core vision statement | Foundational—no v3 equivalent |
| **PN_PRINCIPLES.md** | `00-north-star/PN_PRINCIPLES.md` | Procedural narrative principles | Design philosophy |
| **SPOILER_HYGIENE.md** | `00-north-star/SPOILER_HYGIENE.md` | Hot/Cold information barriers | Critical for player safety |
| **INCIDENT_RESPONSE.md** | `00-north-star/INCIDENT_RESPONSE.md` | Error handling playbook | Operational guidance |
| **TRACEABILITY.md** | `00-north-star/TRACEABILITY.md` | Artifact lineage tracking | Audit trail spec |
| **ACCESSIBILITY.md** | `00-north-star/ACCESSIBILITY.md` | Accessibility requirements | Quality bar context |
| **COLD_SOT_FORMAT.md** | `00-north-star/COLD_SOT_FORMAT.md` | Cold store format spec | Serialization rules |
| **SOURCES_OF_TRUTH.md** | `00-north-star/SOURCES_OF_TRUTH.md` | Authority hierarchy | Canon precedence |
| **EVERGREEN_MANUSCRIPT.md** | `00-north-star/EVERGREEN_MANUSCRIPT.md` | Living document guidelines | Maintenance patterns |
| **QUALITY_BARS_DETAIL.md** | `00-north-star/QUALITY-BARS/*.md` | Detailed bar specifications | v3 has summary only |

#### 12.2.2 Missing Policies

| Policy | v2 Path | Purpose |
|--------|---------|---------|
| **HOOK_POLICY.md** | `00-north-star/POLICIES/HOOK_POLICY.md` | Hook creation/resolution rules |
| **GATE_POLICY.md** | `00-north-star/POLICIES/GATE_POLICY.md` | Gatecheck procedure |
| **ESCALATION_POLICY.md** | `00-north-star/POLICIES/ESCALATION_POLICY.md` | When/how to escalate to SR |
| **DORMANCY_POLICY.md** | `00-north-star/POLICIES/DORMANCY_POLICY.md` | Role sleep/wake signals |

#### 12.2.3 Missing Patterns & Anti-Patterns

v2 had `00-north-star/PATTERNS/` and `00-north-star/ANTI-PATTERNS/` directories with design guidance. v3 has none.

---

### 12.3 Loops (00-north-star/LOOPS)

#### 12.3.1 Missing Loop Definitions (12 of 13)

| Loop ID | v2 Path | Purpose | Roles Involved |
|---------|---------|---------|----------------|
| `hook_harvest` | `LOOPS/hook_harvest.md` | Extract change hooks during play | NR, LK, SR |
| `scene_weave` | `LOOPS/scene_weave.md` | Compose scenes from beats | SS, PW, CD |
| `choice_tree` | `LOOPS/choice_tree.md` | Design branching narratives | PW, LK, SR |
| `lore_sync` | `LOOPS/lore_sync.md` | Maintain canon consistency | LK, GK, SR |
| `draft_review` | `LOOPS/draft_review.md` | Edit/revise prose | SS, CD, GK |
| `canon_commit` | `LOOPS/canon_commit.md` | Stabilize hot → cold | GK, SR, LK |
| `character_arc` | `LOOPS/character_arc.md` | Track character development | LK, PW, CD |
| `timeline_build` | `LOOPS/timeline_build.md` | Construct event sequences | PW, LK |
| `world_expand` | `LOOPS/world_expand.md` | Expand setting/lore | LK, CD, PW |
| `plot_refine` | `LOOPS/plot_refine.md` | Iterate story structure | PW, SR, GK |
| `quality_gate` | `LOOPS/quality_gate.md` | Run quality bar checks | GK, SR |
| `texture_finish` | `LOOPS/texture_finish.md` | Final polish pass | CD, SS, PB |

#### 12.3.2 Depth Reduction in Existing Loop (story_spark)

**v2 `story_spark.md` (161 lines) vs v3 `story_spark.md` (176 lines)**

Despite similar line counts, v3 is ~80% less informative:

| Section | v2 | v3 |
|---------|----|----|
| **Triggers** | Detailed trigger conditions | Not specified |
| **Inputs** | Explicit input artifacts | Not specified |
| **Procedure** | 8-step detailed procedure | Only graph edges |
| **Deliverables** | Explicit output list | Implicit in artifacts |
| **Success Criteria** | Measurable criteria | Not specified |
| **Failure Modes** | Error handling guidance | Not specified |
| **RACI Matrix** | Role responsibilities | Not specified |
| **Hand-offs** | Explicit transition rules | Implicit in conditions |

---

### 12.4 Playbooks (00-north-star/PLAYBOOKS)

#### 12.4.1 Missing Playbooks (14+)

| Playbook | Purpose |
|----------|---------|
| `new_story.md` | Starting a new story project |
| `add_character.md` | Character creation workflow |
| `branch_narrative.md` | Adding choice branches |
| `resolve_contradiction.md` | Fixing canon conflicts |
| `emergency_retcon.md` | Major canon corrections |
| `gate_failure.md` | Handling gatecheck failures |
| `role_stuck.md` | Unblocking stuck roles |
| `player_feedback.md` | Incorporating player input |
| `publish_release.md` | Publishing workflow |
| `hot_to_cold.md` | Promotion process |
| `rollback.md` | Reverting changes |
| `debug_state.md` | State inspection |
| `performance_tune.md` | Optimization |
| `onboard_contributor.md` | New contributor guide |

---

### 12.5 Roles (01-roles)

#### 12.5.1 Removed Roles (4 roles)

These v2 roles have no v3 equivalent:

| Role | v2 Purpose | v3 Status |
|------|------------|-----------|
| **Translator** | Cross-format conversion | Absorbed into Publisher |
| **Researcher** | External fact lookup | Absorbed into Lorekeeper |
| **Illustrator** | Visual asset creation | Absorbed into Creative Director |
| **Audio Producer** | Sound design | Absorbed into Creative Director |

#### 12.5.2 Missing Role Documentation Types

For each of the 8 v3 roles, these v2 documents are missing:

| Document Type | v2 Location | Content | Gap Impact |
|---------------|-------------|---------|------------|
| **Brief** | `01-roles/briefs/*.md` | 1-page role summary | 87% content loss |
| **Charter** | `01-roles/charters/*.md` | Full role specification | 100% missing |
| **Interface** | `01-roles/interfaces/*.md` | API contract | 100% missing |
| **Checklist** | `01-roles/checklists/*.md` | Task verification | 100% missing |

#### 12.5.3 Depth Reduction in Existing Roles

**Example: Plotwright**

v2 Charter (187 lines) vs v3 `plotwright.md` (103 lines) — **70% depth reduction**

| Section | v2 Charter | v3 Role |
|---------|------------|---------|
| Scope Definition | ✅ Detailed | ❌ Missing |
| Inputs/Outputs | ✅ Explicit list | ❌ Missing |
| Loop Participation | ✅ Per-loop behavior | ❌ Missing |
| Hook Policy | ✅ Detailed rules | ❌ Missing |
| Consultation Rules | ✅ When to consult whom | ❌ Missing |
| Anti-patterns | ✅ What to avoid | ❌ Missing |
| Example Dialogues | ✅ Sample interactions | ❌ Missing |
| Escalation Triggers | ✅ When to escalate | ❌ Missing |
| Dormancy Signals | ✅ Sleep/wake conditions | ❌ Missing |
| Pair Guide | ✅ Working with other roles | ❌ Missing |

#### 12.5.4 Missing Cross-Role Documentation

| Document | Purpose | v3 Status |
|----------|---------|-----------|
| **RACI Matrix** | Role responsibility assignment | Missing |
| **Escalation Rules** | When each role escalates | Missing |
| **Pair Guides** | Role collaboration patterns | Missing |
| **Dormancy Signals** | Role sleep/wake conditions | Missing |

---

### 12.6 Dictionary (02-dictionary)

#### 12.6.1 Missing Artifact Types (35 of 40)

**Migrated to v3 (5):** Brief, CanonEntry, GatecheckReport, HookCard, Scene

**Missing from v3 (35):**

| Category | Missing Artifacts |
|----------|-------------------|
| **Structural** | Act, Beat, Chapter, Section, Sequence |
| **Narrative** | Choice, Dialogue, PlotPoint, Prose, Moment |
| **Entity** | Character, Entity, Item, Location, Relationship |
| **World** | World, Setting, Era, Region, Timeline |
| **Thematic** | Arc, Thread, Theme, Motif, Symbol |
| **Technical** | Draft, Event, Fact, Metadata, Texture, TU |
| **Process** | Gate, Checkpoint, Milestone, Transition, View |
| **Reference** | Codex |

#### 12.6.2 Field Depth Reduction in Existing Artifacts

| Artifact | v2 Fields | v3 Fields | Reduction |
|----------|-----------|-----------|-----------|
| HookCard | 15 fields | 8 fields | 47% |
| Scene | 20 fields | 10 fields | 50% |
| Brief | 12 fields | 6 fields | 50% |
| CanonEntry | 10 fields | 5 fields | 50% |
| GatecheckReport | 18 fields | 10 fields | 44% |

#### 12.6.3 Missing Glossary Content

| Document | v2 Terms | v3 Terms | Gap |
|----------|----------|----------|-----|
| **GLOSSARY.md** | 200+ terms | ~20 terms | 90% |
| **ACRONYMS.md** | 50+ entries | 0 | 100% |
| **TAXONOMY.md** | 30+ hierarchies | 2 | 93% |

#### 12.6.4 Missing Conventions

| Convention | Purpose | v3 Status |
|------------|---------|-----------|
| `CHOICE_INTEGRITY.md` | Choice design rules | Missing |
| `FIELD_REGISTRY.md` | Standard field definitions | Missing |
| `NAMING_CONVENTIONS.md` | ID/name patterns | Missing |
| `VERSIONING.md` | Artifact versioning rules | Missing |

---

### 12.7 Protocol (04-protocol)

#### 12.7.1 Missing Envelope Specification (BLOCKER)

v2 `ENVELOPE.md` defined the message wrapper structure:

| Component | Purpose | v3 Status |
|-----------|---------|-----------|
| **Header Schema** | Routing metadata | ❌ Missing |
| **Payload Schema** | Content structure | ❌ Missing |
| **Trace Fields** | Debugging/audit | ❌ Missing |
| **Priority Levels** | Message urgency | ❌ Missing |
| **TTL/Expiry** | Message lifetime | ❌ Missing |

**Impact:** Without envelope spec, roles cannot structure inter-role communication consistently.

#### 12.7.2 Intent Depth Reduction (50%)

**v2 `INTENTS.md` (1228 lines) vs v3 `intents.md` (348 lines)**

| Component | v2 | v3 |
|-----------|----|----|
| Intent Count | 30+ | ~15 |
| Envelope Schema | ✅ JSON examples | ❌ Missing |
| Authorization Matrix | ✅ Role × Intent | ❌ Missing |
| Error Taxonomy | ✅ Detailed error types | ⚠️ Partial (6 types) |
| Payload Examples | ✅ Full JSON | ❌ Missing |
| Validation Rules | ✅ Per-intent rules | ❌ Missing |

**Missing v2 Intents:**

- `ping`, `pong`, `heartbeat` (health check)
- `claim`, `release` (artifact locking)
- `subscribe`, `unsubscribe` (event watching)
- `snapshot`, `restore` (state management)
- `retry`, `timeout`, `cancel` (execution control)
- `merge_request`, `merge_approve` (cold store)
- `hook_propose`, `hook_accept`, `hook_reject` (hook lifecycle)
- `view_render`, `view_stale` (view lifecycle)

#### 12.7.3 Missing Flow Definitions (100%)

v2 had `FLOWS/` directory with end-to-end message choreography:

| Flow | Purpose | Messages Defined |
|------|---------|------------------|
| `story_spark_flow.md` | Story creation | 15 message types |
| `hook_harvest_flow.md` | Hook extraction | 12 message types |
| `scene_weave_flow.md` | Scene composition | 18 message types |
| `canon_commit_flow.md` | Cold promotion | 10 message types |
| `quality_gate_flow.md` | Gatecheck process | 8 message types |
| *(10 more flows...)* | | |

v3 has **no flow definitions**.

#### 12.7.4 Lifecycle Gap (80%)

**v2 Lifecycles (5) vs v3 Lifecycles (1 partial)**

| Lifecycle | v2 States | v3 Status |
|-----------|-----------|-----------|
| **Artifact** | 6 states, 12 transitions | ⚠️ Partial (artifact.md exists but missing transitions) |
| **Hook** | 5 states, 8 transitions | ❌ Missing (hooks.md) |
| **Gate** | 4 states, 6 transitions | ❌ Missing (gate.md) |
| **TU (Texture Unit)** | 5 states, 10 transitions | ❌ Missing (tu.md) |
| **View** | 4 states, 5 transitions | ❌ Missing (view.md) |

#### 12.7.5 Missing Protocol Components

| Component | Purpose | v3 Status |
|-----------|---------|-----------|
| **Authorization Matrix** | Who can send which intents | Missing |
| **PN Safety Invariant** | Player-never-sees-spoilers rule | Not documented |
| **Routing Matrix** | Intent → target role mapping | Missing |
| **Error Handling** | Recovery procedures | Partial |

---

### 12.8 YAML Definitions (05-definitions)

#### 12.8.1 Missing Loop Definitions (92%)

v2 had YAML definitions for all 13 loops. v3 has 1.

| Loop | v2 YAML | v3 Status |
|------|---------|-----------|
| `story_spark.yaml` | ✅ | ⚠️ Partial (MyST only) |
| `hook_harvest.yaml` | ✅ | ❌ Missing |
| `scene_weave.yaml` | ✅ | ❌ Missing |
| *(10 more...)* | ✅ | ❌ Missing |

#### 12.8.2 Missing Definition Components

| Component | v2 | v3 |
|-----------|----|----|
| **capabilities.yaml** | Role capability matrix | Missing |
| **protocol.yaml** | Routing rules | Missing |
| **quality_checks.yaml** | Validation logic | Missing |
| **transitions.yaml** | State machine rules | Missing |

#### 12.8.3 Role Definition Depth

v2 role YAML definitions included:

| Component | v2 | v3 |
|-----------|----|----|
| Capabilities | ✅ Detailed | ⚠️ Reduced |
| Constraints | ✅ 10-15 per role | ⚠️ 3-5 per role |
| Loop Participation | ✅ Per-loop config | ❌ Missing |
| Escalation Rules | ✅ Explicit | ❌ Missing |
| Example Prompts | ✅ Multiple examples | ❌ Missing |

---

### 12.9 Runtime Specification (06-runtime)

#### 12.9.1 Missing Runtime Specs

v2 `06-runtime/` had specifications for:

| Spec | Purpose | v3 Status |
|------|---------|-----------|
| **ARCHITECTURE.md** | Runtime component overview | ❌ Missing (runtime section in main ARCHITECTURE.md is implementation, not spec) |
| **ORCHESTRATOR.md** | Orchestrator interface spec | ❌ Missing |
| **CLI.md** | CLI command spec | ❌ Missing |
| **SHOWRUNNER_INTERFACE.md** | SR tool contract | ❌ Missing |
| **LLM_ADAPTER.md** | Provider abstraction spec | ❌ Missing |
| **TOOL_REGISTRY.md** | Tool definition spec | ❌ Missing |
| **STATE_PERSISTENCE.md** | Checkpoint/restore spec | ❌ Missing |
| **LOGGING.md** | Observability spec | ❌ Missing |

**Note:** v3 has runtime *implementation* (`runtime/*.py`) but no *specification* documents.

---

### 12.10 Quality Bar Depth Reduction

v3 `quality_bars.md` is reasonably well-migrated but missing:

| Component | v2 | v3 |
|-----------|----|----|
| Exception Handling | ✅ Waiver process | ❌ Missing |
| Anti-patterns | ✅ Common mistakes | ❌ Missing |
| Specific Failure Examples | ✅ Real examples | ⚠️ Generic only |
| Remediation Playbooks | ✅ Step-by-step | ⚠️ Brief notes only |
| Bar Interaction Rules | ✅ Precedence | ❌ Missing |

---

### 12.11 Nuanced Depth Gaps (Documents Exist But Are Insufficient)

These v3 documents exist but are significantly less informative than v2:

| Document | v2 Lines | v3 Lines | Information Loss |
|----------|----------|----------|------------------|
| `loops/story_spark.md` | 161 | 176 | ~80% (missing procedures, criteria, RACI) |
| `roles/plotwright.md` | 187 | 103 | ~70% (missing scope, anti-patterns, examples) |
| `roles/showrunner.md` | 220 | 115 | ~65% (missing decision matrix, escalation) |
| `roles/lorekeeper.md` | 195 | 98 | ~70% (missing query patterns, contradiction rules) |
| `roles/gatekeeper.md` | 180 | 95 | ~65% (missing waiver process, bar precedence) |
| `protocol/intents.md` | 1228 | 348 | ~70% (missing envelope, auth matrix, examples) |
| `ontology/artifacts.md` | 800+ | 200 | ~75% (35 of 40 types missing) |

---

### 12.12 Gap Resolution Strategy (Unchanged)

**Phase A: Protocol Foundation (Unblocks everything)**

1. Create `domain/protocol/envelope.md` with `{envelope-spec}` directive
2. Expand `domain/protocol/intents.md` with remaining 15+ intents
3. Create `domain/protocol/lifecycles/hook.md`, `tu.md`, `gate.md`, `view.md`
4. Add authorization matrix and routing rules

**Phase B: Loop Migration**

1. Migrate 12 remaining loops with full MyST directives
2. Restore procedural content (triggers, inputs, deliverables, RACI)
3. Add flow definitions per loop

**Phase C: Role Enrichment**

1. Add charter-level content to each role file
2. Restore anti-patterns, examples, escalation rules
3. Add consultation guides and pair guides

**Phase D: Ontology Completion**

1. Migrate remaining 35 artifact types
2. Restore full field definitions
3. Add glossary and conventions

**Phase E: Principles & Policies**

1. Migrate core principle documents
2. Add playbooks for common scenarios
3. Document policies and patterns

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
| `domain/ontology/` | `{concept}.md` | `artifacts.md`, `taxonomy.md` |
| `domain/protocol/` | `{aspect}.md` | `intents.md`, `routing.md` |
| `generated/models/` | `{category}.py` | `artifacts.py`, `enums.py` |
| `generated/roles/` | `{role_id}.py` | `showrunner.py` |

Note: Loop definitions are NOT compiled to `generated/graphs/`. They serve as documentation and guidance only.
