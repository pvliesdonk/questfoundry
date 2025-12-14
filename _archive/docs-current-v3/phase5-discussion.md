# Phase 5: Compiler/Runtime Adaptation

Discussion document for adapting the compiler and runtime to work with domain-v4 JSON structure.

---

## Current State Analysis

### v3 Architecture (MyST-based)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COMPILE TIME                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  domain/*.md (MyST)                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  Parser (directives.py)                                                      │
│       │ extracts :::{role-meta}, :::{role-tools}, etc.                      │
│       ▼                                                                      │
│  IR Models (ir.py)                                                           │
│       │ RoleIR, LoopIR, ArtifactTypeIR, etc.                                │
│       ▼                                                                      │
│  Generators (roles.py, loops.py, ontology.py)                               │
│       │                                                                      │
│       ▼                                                                      │
│  generated/*.py (Python source files)                                        │
│       • SHOWRUNNER = RoleIR(id="showrunner", ...)                           │
│       • ALL_ROLES = {"showrunner": SHOWRUNNER, ...}                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               RUN TIME                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Orchestrator (orchestrator.py)                                              │
│       │ imports generated/*.py                                               │
│       │ builds SR tools                                                      │
│       │ runs SR in loop, intercepts delegate_to                             │
│       │                                                                      │
│       ├─────────────────────────────────────────────────────────────────────┤
│       │                    SR-CENTRIC HUB & SPOKE                            │
│       │                                                                      │
│       │                    ┌─────────────────┐                               │
│       │                    │   Showrunner    │                               │
│       │                    │  (Orchestrator) │                               │
│       │                    └────────┬────────┘                               │
│       │                             │ delegate_to(role, task)                │
│       │           ┌─────────────────┼─────────────────┐                      │
│       │           ▼                 ▼                 ▼                      │
│       │    ┌────────────┐    ┌────────────┐    ┌────────────┐               │
│       │    │ Plotwright │    │ Lorekeeper │    │   etc...   │               │
│       │    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘               │
│       │          │                 │                 │                       │
│       │          └─────────────────┴─────────────────┘                       │
│       │                            │ returns DelegationResult                │
│       │                            ▼ SR decides next action                  │
│       │                                                                      │
│       ├─────────────────────────────────────────────────────────────────────┤
│       │  State: StudioState (hot_store, cold_store, messages)               │
│       │  Persistence: Checkpoints, ColdStore (SQLite)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### v4 Domain Structure (JSON-based)

```
domain-v4/
├── studio.json                    # Entry point, references all components
├── agents/                        # 12 agent definitions
│   ├── showrunner.json
│   ├── gatekeeper.json
│   ├── plotwright.json
│   └── ...
├── playbooks/                     # 7 playbook definitions (phases + steps)
│   ├── story_spark.json
│   ├── scene_weave.json
│   └── ...
├── artifact-types/                # 16 artifact types
├── asset-types/                   # 2 asset types (illustration, audio)
├── stores/                        # 5 store definitions
├── tools/                         # 9 tool definitions
├── governance/                    # Constitution, quality criteria
│   ├── constitution.json
│   └── quality-criteria/*.json
└── knowledge/                     # Knowledge layers
    ├── layers.json
    ├── must_know/*.json
    └── role_specific/*.json
```

---

## Conceptual Mapping: v3 → v4

| v3 Concept | v4 Concept | Key Differences |
|------------|------------|-----------------|
| **Role** | **Agent** | v4 agents have `archetypes[]` (plural), `capabilities[]`, `constraints[]` as structured objects |
| `role-meta.archetype` | `agent.archetypes` | Single string → array of archetypes |
| `role-meta.agency` | (removed) | v4 uses capabilities + constraints instead of agency levels |
| `role-meta.mandate` | `agent.description` | Embedded in description, not separate field |
| `role-tools` | `agent.capabilities` | Tools, store access, artifact actions all unified as capabilities |
| `role-constraints` | `agent.constraints` | Now structured objects with id, rule, category, enforcement, severity |
| `role-prompt` | `agent.system_prompt_template` | Optional, plus `knowledge_requirements` for dynamic injection |
| **Loop** | **Playbook** | v4 playbooks are phases with DAG steps, guidance not state machines |
| `graph-node` | `playbook.phases.steps` | Steps have depends_on for DAG, archetype references |
| `graph-edge` | `phase.on_success/on_failure` | Transitions at phase level, not step level |
| `quality-gate` | `phase.quality_checkpoint` | Now part of phase definition |
| **ArtifactType** | **artifact-type** | Similar, but v4 has richer lifecycle states |
| (embedded) | **Tool definitions** | v4 tools defined separately in `tools/*.json` |
| (prose in MyST) | **Knowledge entries** | v4 has structured knowledge layers with triggers |

---

## Key Differences to Address

### 1. Agency Model

**v3**: Single `agency` enum (HIGH/MEDIUM/LOW/ZERO) controls autonomy.

**v4**: No explicit agency. Instead:

- `capabilities[]` define what agent CAN do
- `constraints[]` define what agent CANNOT do
- `archetypes[]` define behavioral patterns
- `knowledge_requirements` control what agent knows

**Challenge**: Current runtime uses `Agency` enum. Need to derive equivalent from capabilities/constraints or remove the concept.

### 2. Tool Binding

**v3**: Tools listed inline in `role-tools` directive with descriptions.

**v4**: Tools defined separately in `tools/*.json`, agents reference via `capabilities[].tool_ref`.

**Challenge**: Need to resolve tool references and build tool instances at runtime.

### 3. System Prompts

**v3**: Single `prompt_template` (Jinja2) embedded in role definition.

**v4**:

- Optional `system_prompt_template`
- `knowledge_requirements.constitution` (bool)
- `knowledge_requirements.must_know[]` (knowledge entry IDs)
- `knowledge_requirements.role_specific[]` (knowledge entry IDs)
- `knowledge_requirements.can_lookup[]` (knowledge entry IDs)

**Challenge**: Need to assemble prompts from multiple sources, inject knowledge at runtime.

### 4. Workflows

**v3**: Loops are executable state machines with nodes, edges, conditions.

**v4**: Playbooks are guidance DAGs, not executable state machines. The runtime "follows playbook guidance to coordinate work" but playbooks don't define executable transitions.

**Challenge**: Current `LoopIR` model and runtime expect executable graphs. v4 playbooks are more advisory.

---

## Options for Phase 5

### Option A: JSON → IR Adapter

Create a new parser that reads domain-v4 JSON and produces existing IR models.

```
domain-v4/*.json → New Adapter → RoleIR, LoopIR, etc. → Existing Generators → generated/*.py
```

**Implementation**:

1. New `adapter/json_to_ir.py` that reads agent.json → RoleIR
2. Map v4 concepts to v3 IR (with some loss)
3. Preserve existing code generators and runtime

**Pros**:

- Minimal runtime changes
- Checkpoint compatibility preserved
- Incremental migration path
- Can run v3 and v4 domains during transition

**Cons**:

- Some v4 features lost in translation (knowledge layers, complex capabilities)
- IR models may need extension to capture v4 richness
- Maintaining two source formats

**Effort**: Medium

---

### Option B: New Compiler for JSON → Python

Create a new compiler pipeline that generates Python code aligned with v4's richer model.

```
domain-v4/*.json → New Parser → New IR (v4IR) → New Generators → generated/*.py (different structure)
```

**Implementation**:

1. New IR models: `AgentIR`, `PlaybookIR`, `CapabilityIR`, etc.
2. New generators producing different Python structure
3. Updated runtime to consume new generated code

**Pros**:

- Full representation of v4 concepts
- Clean break from v3 limitations
- Future-proof

**Cons**:

- Significant effort (new IR, new generators, new runtime)
- Breaking change for existing checkpoints
- Parallel maintenance during transition

**Effort**: High

---

### Option C: Direct JSON Consumption (No Compilation)

Runtime loads and validates JSON directly, no code generation.

```
domain-v4/*.json → JSON Schema validation → Runtime loads JSON at startup
```

**Implementation**:

1. Runtime loads `studio.json`, resolves all `$ref`s
2. Pydantic models generated from JSON Schema (one-time)
3. Runtime builds agent pool from loaded JSON
4. No `generated/` directory

**Pros**:

- Simplest conceptually
- Hot-reload domain changes (development convenience)
- Single source of truth
- No generated code to maintain

**Cons**:

- Lose IDE navigation into domain definitions
- Schema validation looser than Python types
- Runtime startup cost (probably negligible)
- Need to handle tool instantiation at runtime

**Effort**: Medium-Low

---

### Option D: Hybrid - JSON Schema → Pydantic (Generated Once)

Generate Pydantic models from JSON Schema, load JSON data into those models at runtime.

```
meta/schemas/*.json → datamodel-codegen → questfoundry/domain_models/*.py (one-time)
domain-v4/*.json → Load at runtime into Pydantic models
```

**Implementation**:

1. Use `datamodel-codegen` to create Pydantic models from schemas
2. Runtime loads JSON, validates against Pydantic models
3. Update runtime to work with new model structure

**Pros**:

- Type safety via Pydantic
- IDE support from generated models
- Runtime validation
- JSON remains source of truth

**Cons**:

- Schema changes require model regeneration
- Additional tooling dependency
- Still need runtime adaptation

**Effort**: Medium

---

## Recommendation

**Start with Option C (Direct JSON Consumption)**, then evolve to Option D if type safety becomes critical.

**Rationale**:

1. **Fastest path to working v4**: No compiler changes needed
2. **Validates v4 design**: See if the JSON structure works in practice before investing in compilation
3. **Reversible**: Can add compilation layer later if needed
4. **Aligns with v4 philosophy**: Playbooks are guidance, not executable state machines. Direct loading fits this model.

### Phased Implementation

**Phase 5a: Foundation**

- Load and validate studio.json and all referenced JSON files
- Create runtime loader that builds agent registry from JSON
- Preserve existing SR-centric orchestration pattern

**Phase 5b: Agent Instantiation**

- Resolve capability → tool bindings
- Build knowledge injection from knowledge_requirements
- Generate system prompts dynamically

**Phase 5c: Playbook Integration**

- Implement playbook-aware orchestration
- Phase-based work tracking
- Quality checkpoint enforcement

**Phase 5d: Optional Compilation**

- If type safety becomes an issue, add JSON Schema → Pydantic generation
- Keep as optional build step, not required for runtime

---

## Decisions Made

1. **Agency model**: **DROP**. v4 uses capabilities/constraints instead.

2. **Playbook execution**: **LOOSE TRACKING**. SR has agency to choose playbook, but runtime nudges when deliverables not made. See Runtime Nudging below.

3. **Entry agents**: **YES, separate modes**. Runtime allows choosing entry_agent, CLI will have separate commands (e.g., `qf run` vs `qf play`). Main workflow pattern identical - PN just doesn't delegate (domain-specific, not runtime-enforced).

4. **Checkpoint compatibility**: **BREAK IS OK**. Clean slate for v4.

5. **Knowledge injection**: See clarified structure below.

---

## Runtime Nudging (from meta/docs/patterns.md)

The runtime loosely tracks playbook progress and nudges agents when discrepancies are detected. Nudges are questions, not errors.

**Nudge Types**:

| Type | Example |
|------|---------|
| **Missing Output** | "According to playbook, step 'create_draft' should produce artifact Y, but none created. Did you intend to skip?" |
| **Unexpected State** | "Playbook indicates we're in 'review' phase, but you're creating new content. Intentional?" |
| **Quality Gate Reminder** | "Before proceeding to 'delivery' phase, playbook requires 'style_check' quality gate. Run check now?" |

**Runtime tracks**:

- Current playbook and phase
- Expected inputs/outputs per step
- Quality checkpoints encountered

**Key principle**: Surfaces discrepancies as questions to the orchestrator, never blocks.

---

## Knowledge Layer Architecture (RESOLVED)

### The Semantic Collision

The original design conflated **Scope** (who knows this?) with **Access Pattern** (how is it injected?):

| Layer | Implied Scope | Implied Access | Problem |
|-------|--------------|----------------|---------|
| `must_know` | Global/Essential | Always in prompt | Where do private critical rules go? |
| `role_specific` | Local to agent | ??? (Menu?) | No clear injection strategy |

**Gap**: "High-Priority Private Knowledge" (e.g., agent persona, critical operational rules) has no home:

- Put in `must_know` → sounds like "global broadcast", might leak to other agents
- Put in `role_specific` → hidden behind tool call, degrades adherence to critical rules

### The Fix: Layers as Injection Strategies

**Reframe `must_know` as "Critical Context" (injection strategy), not "Global Broadcast" (audience)**:

- `must_know` = "If you have this assigned, it is **always injected**"
- `role_specific` = "Specialist reference material, accessed via **menu/tool**"
- Scope is controlled by `applicable_to` field in knowledge entries, NOT by layer

**Runtime Prompt Assembly**:

| Zone | Source | Injection |
|------|--------|-----------|
| Zone 1 (Always) | `constitution` + agent's `must_know[]` | Full text in prompt |
| Zone 2 (Menu) | Agent's `role_specific[]` | Summaries in prompt, full via `consult()` |
| Zone 3 (Tool) | Agent's `can_lookup[]` | Only via explicit `query()` |

**Agent knowledge_requirements** (unchanged schema, clarified semantics):

```json
{
  "constitution": true,
  "must_know": [
    "spoiler_hygiene",           // Global critical - assigned to many agents
    "showrunner_prime_directive" // Private critical - applicable_to: ["showrunner"] only
  ],
  "role_specific": ["loop_guide", "pair_guides"],  // Reference material
  "can_lookup": ["artifact_templates"]              // RAG corpus
}
```

**Key insight**: An agent's `must_know[]` can include entries scoped ONLY to that agent. The runtime injects all entries in the agent's list, regardless of whether the entry is globally or privately scoped. The `applicable_to` field in knowledge entries prevents leakage.

---

## Additional Decisions

### Tool Registry: Load at Startup

Tools are loaded and validated when the studio starts:

- Parse all `tools/*.json`, instantiate tool classes
- Validate env vars, dependencies upfront
- Fail fast if tool can't be created
- Agents get pre-built tool instances

Rationale: Tools are finite (9 in domain-v4), startup cost negligible, fail-fast is valuable.

### Playbook: Consultation Model (No Formal Selection)

SR doesn't "select" a playbook. SR **consults** playbooks for guidance with full agency to adapt, combine, or deviate.

```
SR Workflow:
1. SR receives request
2. SR calls consult_playbook("story_spark") or consult_playbook("how to create story?")
3. Runtime returns playbook content as guidance (phases, steps, expected outputs)
4. SR interprets, adapts, combines as needed
5. SR delegates work freely

Runtime (background):
- Notes which playbooks SR consulted
- Observes artifacts being created
- If consulted playbook expects outputs SR hasn't made → nudge (question, not block)
```

**Key points**:

- No formal "selection" or "activation"
- SR can consult multiple playbooks, mix approaches, improvise
- `consult_playbook()` returns content for SR to read, not to "start"
- Runtime nudging is correlation-based, not enforcement
- SR is never blocked, only nudged

**Tool signature**:

```python
consult_playbook(query: str) -> PlaybookGuidance
# query can be playbook ID or natural language
# Returns: phases, steps, expected outputs, quality checkpoints
```

---

## Outstanding Tasks

Before Phase 5 implementation:

1. **Update meta/docs**: Clarify knowledge layer semantics (injection strategy vs scope)
2. **Audit domain-v4 knowledge**: Review if any `role_specific` entries should be in agents' `must_know[]`
3. **Update layers.json**: Clarify that `layer` = content classification, agent's list = injection strategy
