# QuestFoundry v3: Integrated Cartridge Architecture

**Status:** Master Blueprint
**Version:** 3.3.0

---

## 1. Overview

QuestFoundry v3 replaces the layered specification model (L0-L5) with an **Integrated Domain Model** where MyST (Markedly Structured Text) documents serve as the Single Source of Truth for both machine execution and agent reasoning.

### Core Principle: The Hybrid Document

Every domain file serves two simultaneous runtimes:

```
MyST Source ─┬─(Compile)─→ Generated Code (Strict Constraints)
             │                       ↓
             └─(Ingest)──→ Agent Runtime (Context & Heuristics)
                                     │
                        Single Source of Truth
```

1. **Compiled (Directives):** Defines strict structures (schemas, graph edges, tools) for the Python runtime.
2. **Ingested (Prose):** Defines "soft" knowledge (anti-patterns, decision heuristics, checklists) for the LLM's context window (RAG).

### Orchestration Model

QuestFoundry uses **Showrunner-centric orchestration** where:

- **SR is the hub** — all delegation flows through the Showrunner
- **Roles are specialists** — each role has its own agent, prompt, and tools
- **Loops are guidance** — hybrid documents providing both graph constraints and playbook-style checklists
- **Dynamic delegation** — SR decides at runtime who to delegate to based on context

---

## 2. Directory Structure

The structure now explicitly separates **Content Workflows** (Loops) from **Operational Procedures** (Playbooks).

```
src/questfoundry/
├── domain/                 # MyST Source of Truth
│   ├── roles/              # 8 role definitions (Hybrid: Config + Handbook)
│   ├── loops/              # Content workflows (Hybrid: Graph + Guidance)
│   ├── playbooks/          # Operational procedures (Recovery, Setup, Git Ops)
│   ├── principles/         # Core constraints (Spoiler Hygiene, PN Safety)
│   ├── ontology/           # Data structures (Hybrid: Schema + Usage Guide)
│   └── protocol/           # Communication rules
│
├── compiler/               # MyST → Generated Code
│
├── generated/              # Checked-in generated code
│
└── runtime/                # Execution engine
    ├── orchestrator.py     # SR-based handoff orchestration
    └── ...
```

### Loop vs. Playbook Distinction

> **Rule:** If it generates **Content Artifacts** (Scene, Lore, Hook), it is a **Loop**.
> If it manages **Studio Operations** (Recovery, Setup, Git Ops), it is a **Playbook**.

| Type | Purpose | Runtime Behavior | Examples |
| :--- | :--- | :--- | :--- |
| **Loops** | Create/Edit Story Content | SR uses as heuristic map for delegation | `story_spark`, `hook_harvest`, `scene_weave` |
| **Playbooks** | Fix/Setup the Studio | SR reads as "Emergency Manual" | `gate_failure`, `role_stuck`, `emergency_retcon` |

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

---

## 4. State Model

### StudioState (LangGraph Native)

```python
class StudioState(TypedDict):
    hot_store: dict[str, Artifact]      # Working drafts (mutable)
    cold_store: dict[str, Artifact]     # Committed canon (append-only)
    messages: Annotated[list, add_messages]
    current_role: str
    pending_intents: list[Intent]
```

### Stabilization Path

```
hot_store (draft) → Gatekeeper approval → cold_store (canon)
```

---

## 5. Protocol: SR-Orchestrated Handoff

The Showrunner (SR) is the sole orchestrator. Roles return `DelegationResult` objects to the SR.

```python
class DelegationResult(BaseModel):
    role_id: str
    status: str                         # e.g., "completed", "blocked"
    artifacts: list[str]                # IDs of artifacts created/modified
    message: str                        # Reasoning for SR
    recommendation: str | None = None   # Suggested next action
```

---

## 6. MyST Directive Vocabulary

All domain knowledge is encoded in **Hybrid Documents** using custom directives.

### 6.1 Role Directives (`domain/roles/*.md`)

**Hybrid Pattern:** Combines machine configuration (Directives) with the "Employee Handbook" (Prose).

#### Example Hybrid Structure

```markdown
# Plotwright

> **Mandate:** Design the Topology.

:::{role-meta}
id: plotwright
abbr: PW
archetype: Architect
agency: medium
:::

## Operational Guidelines (Context for Agent & Human)

**Decision Heuristics:**
- **Reachability:** If a scene is unreachable, prioritize fixing connections over writing prose.
- **Dormancy:** Stay dormant during `style_tune_up` unless called by SR.

**Anti-Patterns (DO NOT DO):**
- **Fake Choice:** Options that differ only in wording, not consequence.
- **Meta Gates:** Locking options with "Missing Key" text instead of diegetic narration.

## Configuration (Compiled to Code)

:::{role-tools}
- create_scene: "Define a structural container for prose"
- define_gate: "Set conditions for narrative access"
:::

:::{role-constraints}
- MUST NOT write final prose (delegate to Scene Smith)
- MUST ensure every node has at least one ingress path
:::

:::{role-prompt}
You are the {{ role.archetype }}.
Refer to "Operational Guidelines" for decision logic.
...
:::
```

### 6.2 Loop Directives (`domain/loops/*.md`)

**Hybrid Pattern:** Combines the workflow graph (Directives) with the execution checklist (formerly "Playbooks").

#### Example Hybrid Structure

```markdown
# Story Spark Loop

> **Goal:** Create meaningful nonlinearity from a story seed.

:::{loop-meta}
id: story_spark
trigger: user_request
entry_point: showrunner
:::

## Guidance (Formerly Playbook Checklist)

**When to trigger:**
- New chapter/story request from user.
- Fix reachability issues flagged by Gatekeeper.

**Success criteria:**
- At least 2 meaningful choices per scene.
- No dead-end paths without terminal markers.

**Common failure modes:**
- Single linear path (lacks nonlinearity).
- Orphaned scenes with no incoming edges.

## Execution Graph (Compiled for Validation)

:::{graph-node}
id: plotwright
role: plotwright
:::

:::{graph-edge}
source: plotwright
target: scene_smith
condition: "intent.status == 'topology_complete'"
:::

## Quality Gates

:::{quality-gate}
before: scene_smith
role: gatekeeper
bars: [reachability, nonlinearity]
blocking: true
:::
```

### 6.3 Ontology Directives (`domain/ontology/*.md`)

Defines data structures. Can also be Hybrid (e.g., adding "Usage Examples" prose).

#### `{artifact-type}`

```markdown
:::{artifact-type}
id: hook_card
name: "Hook Card"
store: hot
:::
```

#### `{quality-bar}` (Updated)

Includes waiver policy for Gatekeeper reasoning.

```markdown
:::{quality-bar}
id: integrity
name: "Integrity"
checks: ["All facts traceable"]
waiver_policy: "Allowed only for Retcons approved by SR with 'emergency_retcon' playbook."
:::
```

---

## 7. Generated Code Patterns

The compiler extracts directives to generate:

1. `generated/roles/{role}.py` (Configuration)
2. `generated/models/artifacts.py` (Pydantic Models)
3. `generated/loops/{loop}.py` (Graph Metadata - optional, mostly for SR context)

The **Prose** sections are not compiled but are indexed for the Agent's RAG/Context system.

---

## 8. Runtime Architecture

*(Unchanged: SR-centric Orchestrator, LangGraph State, Tool Execution)*

---

## 9. Build Pipeline

```
1. Parse MyST files
2. Extract Directives -> Compile to Python
3. Extract Prose -> Index for Agent Context (JSON/Vector)
```

---

## 10. Quality Bars (v3)

| Bar | Description |
|-----|-------------|
| **Integrity** | No contradictions in canon |
| **Reachability** | All content accessible via valid paths |
| **Nonlinearity** | Multiple valid paths exist |
| **Gateways** | All gates have valid unlock conditions |
| **Style** | Voice and tone consistency |
| **Determinism** | Same inputs → same outputs |
| **Presentation** | Formatting and structure |
| **Accessibility** | Content usable by all players |

---

## 11. Migration Notes

**v2 Archives:** `_archive/spec/`
**Not Migrated:** Layered L0-L5 structure, 15-role model, Static Flows.

---

## 12. v2 → v3 Migration Inventory & Priorities

> **Status:** Triaged
> **Strategy:** Hybrid Consolidation

This section tracks the migration of v2 content into the v3 Hybrid structure. Items are grouped by priority to inform the Implementation Plan (Chapter 13).

### 12.1 Solved by Architecture (No Migration Needed)

| Gap | v2 Content | v3 Solution | Status |
|---|---|---|---|
| **Role Depth** | Charters (Anti-patterns, etc.) | **Hybrid Role Files** (Prose + Directives) | ✅ DEFINED |
| **Loops vs Playbooks** | Overlapping Docs | **Loops Eat Playbooks** (Content) + **Playbooks** (Ops) | ✅ DEFINED |
| **Protocol Envelope** | Header Spec | **DelegationResult** (Python Object) | ✅ FIXED |
| **Static Flows** | Message choreography | **SR Dynamic Orchestration** | ✅ FIXED |

### 12.2 Priority 1: Critical (Must Have for Functionality)

*Without these, the system is unsafe or unaware of how to fix itself.*

**1. Principles (New Directory)**
Create `domain/principles/` and migrate:

- [x] `SPOILER_HYGIENE.md` (Gatekeeper requires this to audit safety)
- [x] `PN_PRINCIPLES.md` (Narrator requires this to run the game)
- [x] `SOURCES_OF_TRUTH.md` (Defines what "Canon" actually means)

**2. Operational Playbooks (New Directory)**
Create `domain/playbooks/` for meta-processes:

- [x] `gate_failure.md` (How SR recovers when blocked by GK)
- [x] `emergency_retcon.md` (How to rewrite cold canon safely)
- [x] `role_stuck.md` (Agent reset procedure)
- [x] `world_genesis.md` (Project setup)

### 12.3 Priority 2: High (Backlog - Standard Content)

*Required for a full production studio, but not for the initial "Story Spark".*

**3. Remaining Content Loops (Consolidated)**
Migrate remaining v2 Loops + Playbook checklists into `domain/loops/`:

- [x] `hook_harvest.md` (Triage process)
- [x] `lore_deepening.md` (Canon generation)
- [x] `canon_commit.md` (The "Hot to Cold" merge process)
- [x] `scene_weave.md` (Drafting prose)
- [x] `codex_expansion.md` (Encyclopedia generation)

**4. Core Artifacts (Ontology)**
Migrate essential missing artifacts to `domain/ontology/artifacts.md`:

- [x] **Entities:** `Character`, `Location`, `Item`, `Relationship`
- [x] **Structural:** `Act`, `Chapter`, `Sequence`, `Beat`
- [x] **World:** `Timeline`, `Event`, `Fact`

### 12.4 Priority 3: Medium (Enrichment)

*Adds depth and polish.*

**5. Role Enrichment**

- [x] Manually port "Anti-patterns" and "Example Dialogues" from v2 Charters to v3 Role files.

**6. Remaining Artifacts**

- [x] `Shotlist`, `AudioPlan`, `TranslationPack` (Asset production types)

**7. Glossary**

- [x] Create `domain/ontology/glossary.md` (Lightweight version of v2 glossary).

---

## 13. Implementation Phases

This section maps the priorities from Chapter 12 into a chronological execution plan.

### Phase 1: Foundation (Completed) ✓

- [x] Runtime & Compiler basics
- [x] StudioState & Orchestrator
- [x] Basic "Story Spark" loop (Graph only)

### Phase 2: Architecture Refactor (Complete) ✓

*Goal: Prepare the directories and file structures for Hybrid content.*

- [x] Create `domain/principles/` and `domain/playbooks/` directories.
- [x] Refactor existing `domain/roles/` files to include "Operational Guidelines" headers.
- [x] Refactor `domain/loops/story_spark.md` to include "Guidance" (Playbook) headers.

### Phase 3: Critical Migration (Executing Priority 1)

*Goal: System safety and recovery.*

- [x] Populate `domain/principles/` with Spoiler Hygiene & PN Principles.
- [x] Populate `domain/playbooks/` with Emergency Retcon & Gate Failure.
- [ ] Update Gatekeeper quality bars to reference these new principles.

### Phase 4: Standard Migration (Complete) ✓

*Goal: Full content production capabilities.*

- [x] Migrate `hook_harvest`, `lore_deepening`, `canon_commit`, `scene_weave`, and `codex_expansion` loops.
- [x] Add Entity, Structural, and World artifacts to Ontology.

### Phase 5: Enrichment (Complete) ✓

*Goal: Agent intelligence and polish.*

- [x] Deep-dive copy/paste of v2 Charter prose into v3 Role files.
- [x] Add remaining asset artifacts.
- [x] Create glossary.md with essential terms.

---

## Appendix A: File Naming Conventions

| Directory | Pattern | Hybrid Content |
|-----------|---------|----------------|
| `domain/roles/` | `{role_id}.md` | Config + Handbook |
| `domain/loops/` | `{loop_id}.md` | Graph + Checklist |
| `domain/playbooks/` | `{procedure}.md` | Recovery Manuals (Prose only) |
| `domain/principles/` | `{topic}.md` | Policy (Prose only) |
| `domain/ontology/` | `{concept}.md` | Schema + Usage Guide |
