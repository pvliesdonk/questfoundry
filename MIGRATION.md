# QuestFoundry LangGraph Migration: "The Cartridge Pivot"

> **Migration Plan and Progress Tracker**
>
> **Started:** 2025-11-19
> **Target Completion:** TBD
> **Current Phase:** Phase 1 (Foundation) — ✅ **COMPLETE**

---

## Table of Contents

- [Overview](#overview)
- [Strategic Context](#strategic-context)
- [Migration Phases](#migration-phases)
- [Progress Tracker](#progress-tracker)
- [Architecture Decisions](#architecture-decisions)
- [Risk Register](#risk-register)
- [Next Steps](#next-steps)

---

## Overview

### What is This Migration?

This is a **clean-break migration** from an imperative Python runtime (`lib/python`) to a declarative, schema-driven LangGraph runtime (`lib/runtime`). The core philosophy:

> **"The Spec (spec/) is the executable source code (The Cartridge).
> The Runtime (lib/runtime) is a generic player (The Console)."**

### Why?

**Problems with the Old Architecture (lib/python + spec/05-behavior):**
1. **Imperative Logic**: Role behavior hardcoded in Python classes
2. **Tight Coupling**: Changing a role requires code changes
3. **Opaque Behavior**: Logic buried in methods, hard to inspect/test
4. **No Validation**: Runtime errors instead of load-time validation
5. **Difficult Testing**: Can't test prompts in isolation

**Benefits of the New Architecture (lib/runtime + spec/05-definitions):**
1. **Declarative**: All behavior in YAML, validated against schemas
2. **Loose Coupling**: Change definitions without touching runtime code
3. **Transparent**: All logic visible in spec files
4. **Schema-Driven**: Validation at load time catches errors early
5. **LangGraph Native**: Direct mapping to StateGraph constructs
6. **Testable**: Can test definitions without running code

### Scope

**IN SCOPE:**
- New meta-schemas for roles, loops, state, transitions, quality gates
- New directory: `spec/05-definitions/` with YAML definitions
- New runtime: `lib/runtime/` (LangGraph-based)
- Migration of all 15 roles to role profiles
- Migration of all 12 loops to loop patterns
- Migration of 8 quality bars to quality gates
- Complete CLI: `qf run <loop_id>`

**OUT OF SCOPE:**
- Refactoring `lib/python` (it's deprecated, not updated)
- Refactoring `spec/05-behavior` (read-only reference)
- Refactoring `lib/compiler` (replaced by runtime interpretation)
- Backward compatibility (clean break)

**DEPRECATED (Read-Only Reference):**
- `lib/python/` — Old imperative runtime
- `spec/05-behavior/` — Old behavior primitives
- `lib/compiler/` — Old spec compiler

---

## Strategic Context

### Core Constraints

1. **Clean Break**: We are NOT refactoring `lib/python`. It is dead code walking. Build `lib/runtime` from scratch.
2. **Data over Logic**: Complex logic currently hidden in Python classes must be lifted into `spec/05-definitions` (YAML).
3. **Strict L3 Compliance**: The runtime MUST refuse to process any artifact that does not validate against `spec/03-schemas`.
4. **Deprecation Policy**: `lib/compiler` and `spec/05-behavior` are read-only references. Do not modify them.

### Design Principles

1. **Specification-First**: Layers 0-4 are the source of truth. Layer 5 definitions implement them.
2. **Schema-Driven**: Every definition validates against a meta-schema.
3. **Runtime-Agnostic**: Definitions should not assume specific LangGraph versions or Python libraries.
4. **Default to Strictness**: If an input isn't clear, do not add it. Let runtime fail and force definition.
5. **No "Chat"**: If a step implies "talking about" something without an artifact, cut it. V1 is artifact-only.

---

## Migration Phases

### Phase 1: Foundation & Data Modeling ✅ COMPLETE (2025-11-19)

**Goal:** Establish strict data structures that will drive the runtime.

**Deliverables:**
- [x] Meta-schemas in `spec/03-schemas/definitions/`
  - [x] `role_profile.schema.json` — Agent node definition
  - [x] `loop_pattern.schema.json` — Graph topology definition
  - [x] `studio_state.schema.json` — LangGraph state definition
  - [x] `transition_rule.schema.json` — Lifecycle state machine rules
  - [x] `quality_gate.schema.json` — Quality bar validation rules
- [x] Directory structure: `spec/05-definitions/{roles,loops,templates,transitions,quality_gates}/`
- [x] Comprehensive README: `spec/05-definitions/README.md`
- [x] Updated documentation: `spec/03-schemas/README.md`
- [x] Migration tracking document: `MIGRATION.md` (this file)

**Commit:** `feat(schemas): Add Layer 5 meta-schemas for Cartridge Architecture`

**Key Decisions:**
1. **Five Meta-Schemas**: Comprehensive schemas capturing all Layer 0-4 semantics
2. **Directory Structure**: Separate subdirs for roles, loops, templates, transitions, quality_gates
3. **Validation-First**: All definitions must validate against meta-schemas before runtime loads them

---

### Phase 2: Pilot Role Profiles 🚧 IN PROGRESS

**Goal:** Create executable role profiles for core roles needed for Hook Harvest loop.

**Deliverables:**
- [ ] `spec/05-definitions/roles/showrunner.yaml`
  - [ ] Identity from Layer 1 charter
  - [ ] Interface: Opens TUs, coordinates loops
  - [ ] Protocol: All intents (orchestrator role)
  - [ ] Prompt template: `templates/showrunner_prompt.j2`
- [ ] `spec/05-definitions/roles/plotwright.yaml`
  - [ ] Identity from Layer 1 charter
  - [ ] Interface: Inputs (story_spark), Outputs (hook_card)
  - [ ] Protocol: `hook.create`, `hook.update_status`
  - [ ] Prompt template: `templates/plotwright_prompt.j2`
- [ ] `spec/05-definitions/roles/gatekeeper.yaml`
  - [ ] Identity from Layer 1 charter
  - [ ] Interface: Inputs (hook_card, tu_brief), Outputs (gatecheck_report)
  - [ ] Protocol: `gate.report.submit`, `gate.decision`
  - [ ] Prompt template: `templates/gatekeeper_prompt.j2`
  - [ ] Quality bars: All 8 bars
- [ ] Additional roles as needed for Hook Harvest

**Validation:**
```bash
qf validate-definition roles/showrunner.yaml
qf validate-definition roles/plotwright.yaml
qf validate-definition roles/gatekeeper.yaml
```

**Key Decisions:**
- TBD

---

### Phase 3: Pilot Loop Pattern 📋 PENDING

**Goal:** Create executable loop pattern for Hook Harvest.

**Deliverables:**
- [ ] `spec/05-definitions/loops/hook_harvest.yaml`
  - [ ] Topology: Entry node (showrunner), nodes (showrunner, plotwright, gatekeeper), edges
  - [ ] Protocol flow: Map Layer 4 FLOWS/hook_harvest.md to message sequences
  - [ ] Gates: Gatecheck required, bars (Integrity, Reachability)
  - [ ] Traceability: TU lifecycle (hot-proposed → stabilizing)
  - [ ] Execution: max_iterations=5, error_handling
- [ ] Conditional routing: gatekeeper decision → END or plotwright (rework)

**Validation:**
```bash
qf validate-definition loops/hook_harvest.yaml
qf test-loop hook_harvest --dry-run
```

**Key Decisions:**
- TBD

---

### Phase 4: Runtime Engine 📋 PENDING

**Goal:** Build the generic execution engine (`lib/runtime`).

**Deliverables:**
- [ ] Package initialization: `lib/runtime/pyproject.toml`
  - [ ] Stack: Python 3.11+, langgraph, langchain-core, pydantic, typer, jinja2
  - [ ] Structure: `src/questfoundry/{cli,core,io,validation}/`
- [ ] Component: Schema Registry (`validation/registry.py`)
  - [ ] Load L3 schemas dynamically
  - [ ] Expose `validate_artifact(type: str, data: dict) -> bool`
- [ ] Component: Node Factory (`core/node.py`)
  - [ ] Load role_profile.yaml
  - [ ] Render Jinja2 prompts with context injection
  - [ ] Bind tools
  - [ ] Return LangChain Runnable
- [ ] Component: Graph Factory (`core/graph.py`)
  - [ ] Load loop_pattern.yaml
  - [ ] Create LangGraph StateGraph
  - [ ] Add nodes via NodeFactory
  - [ ] Add edges (direct and conditional)
  - [ ] Compile with checkpointer
- [ ] Component: CLI (`cli.py`)
  - [ ] `qf run <loop_id>` command
  - [ ] Load state from `.qf/state.json`
  - [ ] Build and invoke graph
  - [ ] Stream output, save artifacts

**Validation:**
```bash
# Dry-run (no LLM calls)
qf run hook_harvest --dry-run

# Mock LLM
qf run hook_harvest --mock-llm

# Full execution
qf run hook_harvest
```

**Key Decisions:**
- TBD

---

### Phase 5: Full Migration 📋 PENDING

**Goal:** Migrate all roles, loops, and quality gates; deprecate old code.

**Deliverables:**
- [ ] All 15 roles migrated to `spec/05-definitions/roles/`
- [ ] All 12 loops migrated to `spec/05-definitions/loops/`
- [ ] All 8 quality bars migrated to `spec/05-definitions/quality_gates/`
- [ ] All Layer 4 lifecycles migrated to `spec/05-definitions/transitions/`
- [ ] Deprecation markers added to `lib/python/`, `spec/05-behavior/`, `lib/compiler/`
- [ ] Update all documentation to reference new architecture
- [ ] CI/CD updated to use `lib/runtime`

**Validation:**
```bash
# Run all loops
for loop in story_spark hook_harvest lore_deepening codex_expansion; do
  qf run $loop --dry-run
done

# Validate all definitions
qf validate-all-definitions
```

**Key Decisions:**
- TBD

---

## Progress Tracker

### Overall Progress

| Phase | Status | Started | Completed | Duration |
|-------|--------|---------|-----------|----------|
| Phase 1: Foundation | ✅ Complete | 2025-11-19 | 2025-11-19 | 1 day |
| Phase 2: Pilot Roles | 🚧 In Progress | - | - | - |
| Phase 3: Pilot Loop | 📋 Pending | - | - | - |
| Phase 4: Runtime Engine | 📋 Pending | - | - | - |
| Phase 5: Full Migration | 📋 Pending | - | - | - |

### Detailed Progress (Phase 1)

| Task | Status | Notes |
|------|--------|-------|
| Create `spec/03-schemas/definitions/` | ✅ Complete | 2025-11-19 |
| Write `role_profile.schema.json` | ✅ Complete | 625 lines, comprehensive |
| Write `loop_pattern.schema.json` | ✅ Complete | 552 lines, comprehensive |
| Write `studio_state.schema.json` | ✅ Complete | 426 lines, comprehensive |
| Write `transition_rule.schema.json` | ✅ Complete | 336 lines, comprehensive |
| Write `quality_gate.schema.json` | ✅ Complete | 354 lines, comprehensive |
| Create `spec/05-definitions/` structure | ✅ Complete | 5 subdirs: roles, loops, templates, transitions, quality_gates |
| Write `spec/05-definitions/README.md` | ✅ Complete | 560 lines, comprehensive |
| Update `spec/03-schemas/README.md` | ✅ Complete | Added meta-schema section |
| Create `MIGRATION.md` | ✅ Complete | This file |

### Detailed Progress (Phase 2)

| Task | Status | Notes |
|------|--------|-------|
| Create `roles/showrunner.yaml` | 📋 Pending | - |
| Create `roles/plotwright.yaml` | 📋 Pending | - |
| Create `roles/gatekeeper.yaml` | 📋 Pending | - |
| Create `templates/showrunner_prompt.j2` | 📋 Pending | - |
| Create `templates/plotwright_prompt.j2` | 📋 Pending | - |
| Create `templates/gatekeeper_prompt.j2` | 📋 Pending | - |

---

## Architecture Decisions

### ADR-001: Meta-Schema Design (2025-11-19)

**Context:** Need to capture all Layer 0-4 semantics in executable schemas.

**Decision:** Create 5 comprehensive meta-schemas instead of minimal 2 proposed in gist.

**Rationale:**
- Gist proposed only `role_profile` and `loop_pattern` with minimal fields
- Layers 0-4 contain far more semantics: dormancy, RACI, Hot/Cold, quality bars, protocol intents, lifecycles, traceability
- Need separate schemas for reusable components: transitions, quality gates
- Need schema for shared state to guide runtime state management

**Consequences:**
- More upfront schema design work
- Better validation and error detection
- Clearer separation of concerns
- Easier to understand what runtime needs

**Status:** Implemented

---

### ADR-002: Directory Structure (2025-11-19)

**Context:** Need to organize Layer 5 definitions logically.

**Decision:** Use 5 subdirectories in `spec/05-definitions/`:
1. `roles/` — Role profiles (agent nodes)
2. `loops/` — Loop patterns (graph topologies)
3. `templates/` — Jinja2 prompt templates
4. `transitions/` — Reusable lifecycle state machines
5. `quality_gates/` — Reusable quality bar validators

**Rationale:**
- Clear separation by artifact type
- Templates separate from profiles (reusability)
- Transitions and quality_gates reusable across roles/loops
- Maps to schema types

**Consequences:**
- Need to manage file references (role → template, loop → transition)
- Clear where to find each type of definition
- Easier to navigate and maintain

**Status:** Implemented

---

### ADR-003: Deprecation Policy (2025-11-19)

**Context:** Old code (`lib/python`, `spec/05-behavior`, `lib/compiler`) still exists.

**Decision:** Mark as read-only reference, do not modify or refactor.

**Rationale:**
- Clean break is faster and cleaner than incremental migration
- Old code has accumulated technical debt
- New architecture is fundamentally different (imperative vs declarative)
- Keeping old code allows reference during migration

**Consequences:**
- Cannot fix bugs in old code
- Must communicate deprecation clearly
- Eventually delete old code after migration complete

**Status:** Implemented (marked in README files)

---

## Risk Register

### Risk 1: Prompt Fragility

**Description:** Moving logic to YAML prompts means no unit tests for logic.

**Impact:** High (incorrect behavior hard to detect)

**Likelihood:** Medium

**Mitigation:**
- Build `qf eval` command to run test cases against prompts
- Use few-shot examples in prompt templates
- Structured output with schema validation
- Human review of generated artifacts during pilot

**Status:** Mitigated (eval command pending Phase 4)

---

### Risk 2: Context Window Management

**Description:** Naive runtime might dump too much state into prompts.

**Impact:** High (token limits exceeded, high costs)

**Likelihood:** High

**Mitigation:**
- `interface.inputs[].filter` allows JSONPath filtering in role profiles
- NodeFactory only injects declared inputs
- State pruning strategies (time-based, relevance-based)
- Monitor token usage in observability

**Status:** Mitigated (filters in schema design)

---

### Risk 3: Infinite Loops

**Description:** Gatekeeper rejection → plotwright rework can loop forever.

**Impact:** High (runtime hangs)

**Likelihood:** Medium

**Mitigation:**
- `execution.max_iterations` in loop patterns (default 5)
- LangGraph recursion limits
- Timeout configuration
- Human intervention checkpoints

**Status:** Mitigated (max_iterations in schema)

---

### Risk 4: Schema Evolution

**Description:** Changing meta-schemas breaks existing definitions.

**Impact:** Medium (need to update all definitions)

**Likelihood:** Medium (schemas will evolve)

**Mitigation:**
- Semantic versioning for schemas
- Validation at load time catches incompatibilities
- Migration scripts for schema changes
- Deprecation warnings for removed fields

**Status:** Accepted (schema versioning in metadata)

---

### Risk 5: Runtime Complexity

**Description:** NodeFactory and GraphFactory may become complex.

**Impact:** Medium (hard to maintain)

**Likelihood:** Medium

**Mitigation:**
- Keep runtime generic and definition-driven
- Push custom logic into tools, not runtime
- Comprehensive tests for runtime components
- Clear separation: runtime interprets, definitions define

**Status:** Accepted (to be monitored in Phase 4)

---

## Next Steps

### Immediate (Phase 2)

1. **Create Showrunner Role Profile**
   - Read `spec/01-roles/charters/showrunner.md`
   - Extract identity, interface, protocol permissions
   - Write `roles/showrunner.yaml`
   - Create `templates/showrunner_prompt.j2`
   - Validate against `role_profile.schema.json`

2. **Create Plotwright Role Profile**
   - Read `spec/01-roles/charters/plotwright.md`
   - Read `spec/05-behavior/adapters/plotwright.adapter.yaml` (reference)
   - Extract prompt logic into Jinja2 template
   - Write `roles/plotwright.yaml`
   - Create `templates/plotwright_prompt.j2`
   - Validate against `role_profile.schema.json`

3. **Create Gatekeeper Role Profile**
   - Read `spec/01-roles/charters/gatekeeper.md`
   - Map 8 quality bars to validation logic
   - Write `roles/gatekeeper.yaml`
   - Create `templates/gatekeeper_prompt.j2`
   - Validate against `role_profile.schema.json`

### Short-Term (Phase 3)

4. **Create Hook Harvest Loop Pattern**
   - Read `spec/00-north-star/LOOPS/hook_harvest.md`
   - Read `spec/04-protocol/FLOWS/hook_harvest.md`
   - Map nodes and edges
   - Define conditional routing (gatekeeper decision)
   - Write `loops/hook_harvest.yaml`
   - Validate against `loop_pattern.schema.json`

5. **Test Definitions**
   - Write validation script: `scripts/validate-definitions.py`
   - Validate all role profiles and loop patterns
   - Generate test reports

### Medium-Term (Phase 4)

6. **Build Runtime Components**
   - Initialize `lib/runtime` package
   - Implement SchemaRegistry
   - Implement NodeFactory
   - Implement GraphFactory
   - Implement CLI

7. **Integration Testing**
   - Dry-run Hook Harvest loop
   - Mock LLM execution
   - Full execution with real LLM
   - Validate artifacts against Layer 3 schemas

### Long-Term (Phase 5)

8. **Complete Migration**
   - Migrate remaining 12 roles
   - Migrate remaining 11 loops
   - Migrate quality gates and transitions
   - Update documentation
   - Deprecate old code
   - Celebrate! 🎉

---

## References

### Specification
- **Layer 0:** `spec/00-north-star/` — Vision, roles, loops, quality bars
- **Layer 1:** `spec/01-roles/` — Role charters and briefs
- **Layer 2:** `spec/02-dictionary/` — Common language and artifact types
- **Layer 3:** `spec/03-schemas/` — JSON schemas
- **Layer 4:** `spec/04-protocol/` — Protocol envelopes, intents, lifecycles
- **Layer 5 (OLD):** `spec/05-behavior/` — Behavior primitives (DEPRECATED)
- **Layer 5 (NEW):** `spec/05-definitions/` — Executable definitions

### Implementation
- **Runtime (NEW):** `lib/runtime/` — LangGraph-based execution engine
- **Runtime (OLD):** `lib/python/` — Imperative Python runtime (DEPRECATED)
- **Compiler (OLD):** `lib/compiler/` — Spec compiler (DEPRECATED)

### Documentation
- **Cartridge Architecture:** `spec/05-definitions/README.md`
- **Meta-Schemas:** `spec/03-schemas/README.md` (definitions/ section)
- **Migration Plan:** `MIGRATION.md` (this file)
- **Original Gist:** https://gist.github.com/pvliesdonk/8a2430c470bba3ebeccc719e9b48629e

---

**Last Updated:** 2025-11-19
**Author:** Claude (AI Assistant)
**Status:** Living Document (updated throughout migration)
