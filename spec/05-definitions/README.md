# Layer 5 — Executable Definitions (The Cartridge)

> **Status:** 🚧 **IN PROGRESS** — Migration from spec/05-behavior to schema-driven definitions
>
> **Last Updated:** 2025-11-19
>
> **Migration:** This layer represents a strategic pivot to declarative, schema-driven execution. See `../MIGRATION.md` for the complete migration plan and current status.

---

## Table of Contents

- [Overview](#overview)
- [The Cartridge Philosophy](#the-cartridge-philosophy)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
- [Creating Definitions](#creating-definitions)
- [Migration Status](#migration-status)
- [References](#references)

---

## Overview

**Layer 5 Definitions** is the "executable cartridge" that drives the QuestFoundry runtime. While Layers 0-4 define **what** the studio does (roles, loops, artifacts, protocol), Layer 5 defines **how** to execute it as a LangGraph application.

**Core Principle:**
```
The Spec (spec/) is the executable source code (The Cartridge).
The Runtime (lib/runtime) is a generic player (The Console).
```

This layer contains:
- **Role Profiles**: Executable agent definitions (prompt templates, tools, I/O contracts)
- **Loop Patterns**: Graph topologies and protocol flows
- **Prompt Templates**: Jinja2 templates for role behavior
- **Transition Rules**: Lifecycle state machine definitions
- **Quality Gates**: Automated validation rules for the 8 quality bars

All definitions validate against **meta-schemas** in `spec/03-schemas/definitions/` and reference **Layer 3 artifact schemas** for strict I/O contracts.

---

## The Cartridge Philosophy

### Traditional Approach (spec/05-behavior, DEPRECATED)
- **Imperative**: Python classes hardcode role behavior
- **Tightly Coupled**: Changing a role requires code changes
- **Opaque**: Logic buried in methods, hard to inspect/test
- **Fragile**: No schema validation, runtime errors

### Cartridge Approach (spec/05-definitions, CURRENT)
- **Declarative**: YAML definitions describe behavior
- **Loosely Coupled**: Change roles without touching runtime code
- **Transparent**: All logic visible in spec files
- **Validated**: Schema-driven validation at load time
- **LangGraph Native**: Direct mapping to StateGraph constructs

### Analogy: Game Console

| Concept | Gaming | QuestFoundry |
|---------|--------|--------------|
| **Console** | Nintendo Switch | `lib/runtime` (LangGraph runtime) |
| **Cartridge** | Game ROM | `spec/05-definitions/` (role/loop definitions) |
| **Save File** | Progress data | StudioState (Hot/Cold SoT) |
| **Controller Input** | Button presses | Protocol messages (Layer 4 intents) |
| **Screen Output** | Graphics | Artifacts (validated against Layer 3 schemas) |

**Key Insight:** You can swap cartridges (definitions) without changing the console (runtime). Different projects can have different role behaviors, loop structures, and quality gates—all using the same generic runtime.

---

## Architecture

### Five-Layer Execution Model

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 0-4: Specification (What & Why)                      │
│ - Roles, Loops, Artifacts, Protocol, Quality Bars          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Definitions (How - Executable)                    │
│ - role_profile.yaml → Agent Node                           │
│ - loop_pattern.yaml → StateGraph                           │
│ - Validates against meta-schemas in 03-schemas/definitions │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Runtime: lib/runtime (Generic Interpreter)                 │
│ - NodeFactory: role_profile → LangChain Runnable           │
│ - GraphFactory: loop_pattern → LangGraph StateGraph        │
│ - SchemaRegistry: Validate artifacts against Layer 3       │
│ - State Management: StudioState (Hot/Cold SoT)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Execution: LangGraph                                        │
│ - Nodes: Role invocations                                  │
│ - Edges: Conditional routing (gatekeeper decisions, etc.)  │
│ - State: StudioState (artifacts, protocol, gates)          │
│ - Persistence: SQLite/File checkpointer                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User runs: `qf run hook_harvest`
2. Runtime loads: loops/hook_harvest.yaml
3. Runtime validates: Against loop_pattern.schema.json
4. GraphFactory builds: LangGraph StateGraph
   ├─ Loads role profiles from topology.nodes
   ├─ NodeFactory creates Runnables for each role
   └─ Adds edges (direct and conditional)
5. Runtime initializes: StudioState
   ├─ Loads Hot SoT artifacts
   ├─ Loads Cold SoT snapshots
   └─ Sets up protocol correlation
6. Graph executes: Entry node → ... → EXIT
   ├─ Each node: Reads inputs, renders prompt, calls LLM, validates output
   ├─ State updates: After each node, merged into StudioState
   └─ Edges evaluated: Conditional routing based on state
7. On exit: Runtime writes artifacts to Hot/Cold SoT
```

---

## Directory Structure

```
spec/05-definitions/
├── README.md                      # This file
├── roles/                         # Role profile definitions
│   ├── plotwright.yaml            # Plotwright agent definition
│   ├── gatekeeper.yaml            # Gatekeeper agent definition
│   ├── showrunner.yaml            # Showrunner agent definition
│   ├── scene_smith.yaml           # Scene Smith agent definition
│   ├── lore_weaver.yaml           # Lore Weaver agent definition
│   ├── codex_curator.yaml         # Codex Curator agent definition
│   ├── style_lead.yaml            # Style Lead agent definition
│   ├── book_binder.yaml           # Book Binder agent definition
│   ├── player_narrator.yaml       # Player-Narrator agent definition
│   └── ...                        # Other roles (dormant by default)
├── loops/                         # Loop pattern definitions
│   ├── hook_harvest.yaml          # Hook Harvest graph topology
│   ├── story_spark.yaml           # Story Spark graph topology
│   ├── lore_deepening.yaml        # Lore Deepening graph topology
│   ├── codex_expansion.yaml       # Codex Expansion graph topology
│   ├── gatecheck.yaml             # Gatecheck (standalone) graph topology
│   └── ...                        # Other loops
├── templates/                     # Jinja2 prompt templates
│   ├── plotwright_prompt.j2       # Plotwright system prompt
│   ├── gatekeeper_prompt.j2       # Gatekeeper system prompt
│   └── ...                        # Other role prompts
├── transitions/                   # Reusable lifecycle state machine rules
│   ├── hook_lifecycle.yaml        # Hook Card state transitions
│   ├── tu_lifecycle.yaml          # Trace Unit state transitions
│   ├── gate_lifecycle.yaml        # Gate state transitions
│   └── view_lifecycle.yaml        # View/Export state transitions
└── quality_gates/                 # Reusable quality bar validation rules
    ├── integrity.yaml             # Integrity bar validation
    ├── reachability.yaml          # Reachability bar validation
    ├── nonlinearity.yaml          # Nonlinearity bar validation
    ├── gateways.yaml              # Gateways bar validation
    ├── style.yaml                 # Style bar validation
    ├── determinism.yaml           # Determinism bar validation
    ├── presentation.yaml          # Presentation bar validation
    └── accessibility.yaml         # Accessibility bar validation
```

---

## How It Works

### 1. Role Profile Execution

**File:** `roles/plotwright.yaml`

```yaml
id: plotwright
identity:
  name: Plotwright
  abbreviation: PW
  charter_ref: spec/01-roles/charters/plotwright.md
  dormancy_policy: default_on

interface:
  inputs:
    - artifact_type: hook_card
      required: false
      state_key: artifacts.current_hook
      filter: "status == 'accepted'"
  outputs:
    - artifact_type: hook_card
      state_key: artifacts.current_hook
      validation_required: true

behavior:
  prompt:
    template: file://spec/05-definitions/templates/plotwright_prompt.j2
    context_injection:
      quality_bars: [Integrity, Reachability, Nonlinearity]
      expertise_refs:
        - spec/05-behavior/expertises/plotwright_expertise.md
  tools:
    - name: read_hot_sot
      enabled: true
    - name: write_hot_sot
      enabled: true

protocol:
  intents:
    can_send: [hook.create, hook.update_status]
    can_receive: [hook.create, tu.update]
  lifecycles:
    hook:
      can_create: true
      can_transition:
        - from_state: proposed
          to_state: accepted
          intent: hook.update_status

constraints:
  safety:
    can_see_hot: true
    can_see_spoilers: true
    pn_safe: false
  traceability:
    requires_tu_linkage: true

llm_config:
  provider: anthropic
  model: claude-3-sonnet-20240229
  temperature: 0.7
```

**Runtime Processing:**
1. `NodeFactory.load_role_profile("plotwright")` reads `plotwright.yaml`
2. Validates against `spec/03-schemas/definitions/role_profile.schema.json`
3. Loads Jinja2 template from `templates/plotwright_prompt.j2`
4. Injects context: quality bars text, expertise docs, current state
5. Creates LangChain Runnable:
   ```python
   def plotwright_node(state: StudioState) -> StudioState:
       # Filter inputs based on interface.inputs
       inputs = filter_state(state, profile.interface.inputs)

       # Render prompt
       prompt = render_template(profile.behavior.prompt.template, inputs)

       # Call LLM with tools
       response = llm.invoke(prompt, tools=profile.behavior.tools)

       # Validate output against hook_card.schema.json
       validate_artifact(response, "hook_card")

       # Update state
       state.artifacts.current_hook = response
       return state
   ```

### 2. Loop Pattern Execution

**File:** `loops/hook_harvest.yaml`

```yaml
id: hook_harvest
metadata:
  name: Hook Harvest
  type: Discovery
  loop_guide_ref: spec/00-north-star/LOOPS/hook_harvest.md
  timebox: 60 min

topology:
  entry_node: showrunner
  nodes:
    - role_id: showrunner
    - role_id: plotwright
    - role_id: gatekeeper
  edges:
    - source: showrunner
      target: plotwright
      type: direct
    - source: plotwright
      target: gatekeeper
      type: direct
    - source: gatekeeper
      target: END
      type: conditional
      condition:
        evaluator: artifact_field_match
        expression: "state.artifacts.current_gatecheck.decision"
        routes:
          pass: END
          conditional_pass: END
          block: plotwright

gates:
  gatecheck_required: true
  quality_bars: [Integrity, Reachability]

traceability:
  tu_lifecycle:
    required: true
    starts_in: hot-proposed
    must_reach: stabilizing
  produces_cold: false

execution:
  max_iterations: 5
  error_handling:
    on_bar_failure: rework
```

**Runtime Processing:**
1. `GraphFactory.load_loop_pattern("hook_harvest")` reads `hook_harvest.yaml`
2. Validates against `spec/03-schemas/definitions/loop_pattern.schema.json`
3. Creates LangGraph StateGraph:
   ```python
   from langgraph.graph import StateGraph, END

   graph = StateGraph(StudioState)

   # Add nodes
   graph.add_node("showrunner", node_factory.create_node("showrunner"))
   graph.add_node("plotwright", node_factory.create_node("plotwright"))
   graph.add_node("gatekeeper", node_factory.create_node("gatekeeper"))

   # Add edges
   graph.add_edge("showrunner", "plotwright")
   graph.add_edge("plotwright", "gatekeeper")

   # Add conditional edge
   def route_gatecheck(state):
       decision = state.artifacts.current_gatecheck.decision
       if decision in ["pass", "conditional_pass"]:
           return END
       else:
           return "plotwright"

   graph.add_conditional_edges("gatekeeper", route_gatecheck)

   # Set entry point
   graph.set_entry_point("showrunner")

   # Compile with checkpointing
   app = graph.compile(checkpointer=SqliteSaver(...))
   ```

---

## Creating Definitions

### Step 1: Create Role Profile

1. Copy template:
   ```bash
   cp spec/05-definitions/_templates/role_profile.template.yaml \
      spec/05-definitions/roles/my_role.yaml
   ```

2. Fill in fields (validate against `spec/03-schemas/definitions/role_profile.schema.json`)

3. Create prompt template:
   ```bash
   touch spec/05-definitions/templates/my_role_prompt.j2
   ```

4. Test validation:
   ```bash
   qf validate-definition roles/my_role.yaml
   ```

### Step 2: Create Loop Pattern

1. Copy template:
   ```bash
   cp spec/05-definitions/_templates/loop_pattern.template.yaml \
      spec/05-definitions/loops/my_loop.yaml
   ```

2. Define topology (nodes, edges, entry point)

3. Map to Layer 4 protocol flow (message sequences)

4. Set quality gates and traceability requirements

5. Test validation:
   ```bash
   qf validate-definition loops/my_loop.yaml
   ```

### Step 3: Test Execution

```bash
# Dry-run (no LLM calls, just graph validation)
qf run my_loop --dry-run

# Execute with mocked LLM
qf run my_loop --mock-llm

# Full execution
qf run my_loop
```

---

## Migration Status

### Phase 1: Foundation ✅ COMPLETE (2025-11-19)
- [x] Meta-schemas created in `spec/03-schemas/definitions/`
  - [x] `role_profile.schema.json`
  - [x] `loop_pattern.schema.json`
  - [x] `studio_state.schema.json`
  - [x] `transition_rule.schema.json`
  - [x] `quality_gate.schema.json`
- [x] Directory structure created (`roles/`, `loops/`, `templates/`, `transitions/`, `quality_gates/`)
- [x] README.md written (this file)

### Phase 2: Pilot Roles (IN PROGRESS)
- [ ] Showrunner role profile
- [ ] Plotwright role profile
- [ ] Gatekeeper role profile
- [ ] Scene Smith role profile
- [ ] Lore Weaver role profile
- [ ] Codex Curator role profile

### Phase 3: Pilot Loop (PENDING)
- [ ] Hook Harvest loop pattern
- [ ] Protocol flow mapping (Layer 4 → graph edges)
- [ ] Quality gate integration

### Phase 4: Runtime Integration (PENDING)
- [ ] NodeFactory implementation (`lib/runtime/core/node.py`)
- [ ] GraphFactory implementation (`lib/runtime/core/graph.py`)
- [ ] SchemaRegistry implementation (`lib/runtime/validation/registry.py`)
- [ ] CLI implementation (`lib/runtime/cli.py`)

### Phase 5: Full Migration (PENDING)
- [ ] All 15 roles migrated
- [ ] All 12 loops migrated
- [ ] All 8 quality gates migrated
- [ ] Deprecate `spec/05-behavior/` and `lib/python/`

---

## References

### Layer 0-4 Specifications
- **Layer 0:** `spec/00-north-star/` — Vision, roles, loops, quality bars
- **Layer 1:** `spec/01-roles/` — Role charters and briefs
- **Layer 2:** `spec/02-dictionary/` — Common language and artifact types
- **Layer 3:** `spec/03-schemas/` — JSON schemas for artifacts
- **Layer 4:** `spec/04-protocol/` — Protocol envelopes, intents, lifecycles

### Meta-Schemas (Layer 3)
- **Role Profile:** `spec/03-schemas/definitions/role_profile.schema.json`
- **Loop Pattern:** `spec/03-schemas/definitions/loop_pattern.schema.json`
- **Studio State:** `spec/03-schemas/definitions/studio_state.schema.json`
- **Transition Rule:** `spec/03-schemas/definitions/transition_rule.schema.json`
- **Quality Gate:** `spec/03-schemas/definitions/quality_gate.schema.json`

### Deprecated (Read-Only Reference)
- **spec/05-behavior/** — Old behavior primitives (v1 architecture)
- **lib/compiler/** — Old spec compiler (replaced by runtime interpretation)
- **lib/python/** — Old imperative runtime (replaced by lib/runtime)

### Migration Documentation
- **MIGRATION.md:** Complete migration plan and progress tracker
- **DECISIONS/:** Architecture Decision Records (ADRs) for major changes

---

## FAQ

### Why declarative instead of imperative code?

**Imperative (old):**
- Logic scattered across Python files
- Hard to understand behavior without reading code
- Difficult to test prompts in isolation
- No validation until runtime errors

**Declarative (new):**
- All behavior visible in YAML
- Easy to understand and modify
- Can test definitions without running code
- Schema validation catches errors at load time

### Can I still use the old lib/python runtime?

No. The old runtime is deprecated and will be removed. All new development uses `lib/runtime` with Layer 5 definitions.

### How do I migrate an existing role from 05-behavior?

1. Read the role's charter (`spec/01-roles/charters/<role>.md`)
2. Read the role's adapter (`spec/05-behavior/adapters/<role>.adapter.yaml`)
3. Extract prompt logic into Jinja2 template (`templates/<role>_prompt.j2`)
4. Map inputs/outputs to Layer 3 schemas
5. Define protocol permissions (Layer 4 intents)
6. Create `roles/<role>.yaml` validating against `role_profile.schema.json`

### What if a loop needs custom logic that YAML can't express?

Two options:
1. **Preferred:** Add a custom tool to the runtime tool registry, then reference it in the role profile
2. **Last resort:** Add a custom validator function to the runtime, then reference it in `loop_pattern.execution.error_handling.custom_checks`

The goal is to keep definitions declarative and push custom logic into reusable, testable tools.

---

**Built with ❤️ for the QuestFoundry Cartridge Architecture**

For questions or contributions, see `../CONTRIBUTING.md` or open an issue.
