# QuestFoundry v3: Integrated Cartridge Architecture

**Status:** Master Blueprint
**Version:** 3.0.0

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
| Protocol | Agent-to-agent message passing | State-based routing (LangGraph native) |
| Runtime | LangChain + LangGraph | Pure LangGraph |

---

## 2. Directory Structure

```
src/questfoundry/
├── domain/                 # MyST Source of Truth
│   ├── roles/              # 8 role definitions
│   ├── loops/              # Workflow graphs
│   ├── ontology/           # Data structures
│   └── protocol/           # Communication rules
│
├── compiler/               # MyST → Generated Code
│   ├── parser/             # MyST directive extraction
│   ├── models/             # Intermediate representation
│   └── generators/         # Code generators
│
├── generated/              # Checked-in generated code
│   ├── models/             # Pydantic models
│   ├── roles/              # Role configurations
│   └── graphs/             # LangGraph definitions
│
└── runtime/                # Execution engine
    ├── engine.py           # Graph executor
    ├── state.py            # StudioState
    ├── router.py           # Intent-based routing
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

## 5. Protocol: System-as-Router

### Intent-Based Routing

Roles do not call each other directly. They post **Intents** to the message bus, and the runtime routes based on loop definitions.

```python
class Intent(BaseModel):
    """A role's declaration of work status."""

    type: Literal["handoff", "escalation", "broadcast", "terminate"]
    source_role: str
    status: str                         # e.g., "stabilized", "blocked"
    payload: dict[str, Any] | None = None
    reason: str | None = None           # For escalations
```

### Routing Flow

1. **Role completes work** → writes to `hot_store`
2. **Role posts Intent** → `Intent(type="handoff", status="stabilized")`
3. **Router reads loop definition** → finds edge matching intent
4. **Router activates next role** → based on `condition` match
5. **Repeat** until `Intent(type="terminate")`

### Example Routing Rules (from loop definition)

```yaml
edges:
  - source: plotwright
    target: scene_smith
    condition: "intent.status == 'topology_complete'"

  - source: plotwright
    target: showrunner
    condition: "intent.type == 'escalation'"
```

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

From `{role-meta}` + `{role-tools}` + `{role-prompt}`:

```python
# generated/roles/showrunner.py
from questfoundry.runtime.base import RoleConfig

SHOWRUNNER = RoleConfig(
    id="showrunner",
    abbr="SR",
    archetype="Product Owner",
    agency="high",
    mandate="Manage by Exception",
    tools=["read_state", "write_state", "post_intent"],
    constraints=[
        "MUST NOT modify cold_store directly",
        # ...
    ],
    prompt_template="""You are the Product Owner...""",
)
```

### 7.3 LangGraph Definitions (`generated/graphs/`)

From `{loop-meta}` + `{graph-node}` + `{graph-edge}`:

```python
# generated/graphs/story_spark.py
from langgraph.graph import StateGraph
from questfoundry.runtime.state import StudioState
from questfoundry.runtime.nodes import create_role_node

def build_story_spark_graph() -> StateGraph:
    """Auto-generated from domain/loops/story_spark.md"""

    graph = StateGraph(StudioState)

    # Nodes
    graph.add_node("showrunner", create_role_node("showrunner"))
    graph.add_node("plotwright", create_role_node("plotwright"))
    graph.add_node("scene_smith", create_role_node("scene_smith"))
    graph.add_node("lorekeeper", create_role_node("lorekeeper"))
    graph.add_node("gatekeeper", create_role_node("gatekeeper"))

    # Edges with conditions
    graph.add_conditional_edges(
        "showrunner",
        route_by_intent,
        {
            "brief_approved": "plotwright",
            "escalation": "showrunner",
        }
    )
    # ... more edges

    graph.set_entry_point("showrunner")
    return graph.compile()
```

---

## 8. Runtime Architecture

### 8.1 Core Components

```
┌─────────────────────────────────────────────────────┐
│                    CLI (qf)                         │
├─────────────────────────────────────────────────────┤
│                   Engine                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ Router  │  │ State   │  │Providers│             │
│  └────┬────┘  └────┬────┘  └────┬────┘             │
│       │            │            │                   │
│       ▼            ▼            ▼                   │
│  ┌─────────────────────────────────────┐           │
│  │        LangGraph StateGraph          │           │
│  │  ┌────┐  ┌────┐  ┌────┐  ┌────┐    │           │
│  │  │ SR │→│ PW │→│ SS │→│ GK │     │           │
│  │  └────┘  └────┘  └────┘  └────┘    │           │
│  └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### 8.2 Provider Abstraction

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

# runtime/providers/openai.py
class OpenAIProvider(LLMProvider):
    """Direct OpenAI SDK integration."""
    ...
```

### 8.3 Engine Flow

```python
# runtime/engine.py
async def run_loop(loop_id: str, initial_state: StudioState) -> StudioState:
    """Execute a complete loop."""

    # Load compiled graph
    graph = load_graph(loop_id)

    # Run to completion
    final_state = await graph.ainvoke(initial_state)

    return final_state
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

### Phase 1: Foundation

- [ ] `pyproject.toml` with pure LangGraph deps
- [ ] MyST parser for directives
- [ ] `StudioState` implementation
- [ ] Ollama provider

### Phase 2: Ontology

- [ ] `domain/ontology/artifacts.md` (HookCard, Brief)
- [ ] `domain/ontology/taxonomy.md` (enums)
- [ ] Compiler: ontology → Pydantic models

### Phase 3: Story Spark Loop

- [ ] `domain/roles/showrunner.md`
- [ ] `domain/roles/plotwright.md`
- [ ] `domain/roles/lorekeeper.md`
- [ ] `domain/roles/narrator.md`
- [ ] `domain/loops/story_spark.md`
- [ ] Compiler: roles + loops → LangGraph

### Phase 4: Execution

- [ ] Router implementation
- [ ] End-to-end Story Spark test
- [ ] OpenAI provider

### Phase 5: Remaining Roles & Loops

- [ ] Creative Director, Scene Smith, Publisher, Gatekeeper
- [ ] Additional loops

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
| `generated/graphs/` | `{loop_id}.py` | `story_spark.py` |
