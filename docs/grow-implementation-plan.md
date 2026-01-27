# GROW Stage Implementation Plan (Milestone 3)

## Overview

GROW transforms SEED output (paths with initial beats) into a complete, validated story graph with passages, choices, codewords, and entity overlays. It has 11 phases, 9 human gates, a mix of deterministic and LLM-assisted operations, and is strictly linear in dependency.

**Total estimate:** ~4,500 lines code + tests across 12 PRs

## Architectural Decisions

### 1. Stage Protocol Compatibility

GROW's `execute()` returns a `GrowResult` dict containing summary metadata (phases completed, arcs generated, passages created). The actual work is mutations to the graph. The orchestrator already supports `has_mutation_handler()` for this; we add "grow" to `_MUTATION_STAGES`.

### 2. Intra-Stage Gates

Introduce a `PhaseGateHook` Protocol with `on_phase_complete(phase_name, phase_result) -> approve|reject`. An `AutoApprovePhaseGate` default enables non-interactive/CI execution.

### 3. Incremental Graph Mutations

Each phase reads the current graph state and adds/modifies nodes/edges. Each phase function takes a `Graph` and returns a `PhaseResult` dataclass with summary stats.

### 4. Hand-Written Pydantic Models

New node types (Arc, Passage, Codeword, Choice, EntityOverlay) and sub-phase output models (PathAgnosticAssessment, IntersectionProposal, etc.) are hand-written Pydantic models in `models/grow.py`, following the same pattern as BRAINSTORM and SEED (not the schema-first generation used by DREAM).

---

## PR Breakdown

### PR 1: GROW Models (Contract)

**Scope:** Hand-written Pydantic models for GROW data types (following BRAINSTORM/SEED pattern)

**Dependencies:** None

**Key files:**
- `src/questfoundry/models/grow.py` — Node types (Arc, Passage, Codeword, Choice, EntityOverlay, GrowResult) and sub-phase output models (PathAgnosticAssessment, IntersectionProposal, SceneTypeTag, GapProposal, OverlayProposal, ChoiceLabel)
- `src/questfoundry/models/__init__.py` — Exports
- `tests/unit/test_grow_models.py`

**Lines:** ~350-400

**Acceptance:**
- All GROW node types defined as Pydantic models
- Sub-phase output models cover all LLM-assisted phases
- Unit tests for each model class
- Models exported from `questfoundry.models`

---

### PR 2: Phase Gate Infrastructure (Contract/Protocol)

**Scope:** PhaseGateHook Protocol and GROW-specific error categories

**Dependencies:** None (parallel with PR 1)

**Key files:**
- `src/questfoundry/pipeline/gates.py` — `PhaseGateHook` Protocol, `AutoApprovePhaseGate`, `PhaseResult` dataclass
- `src/questfoundry/graph/mutations.py` — `GrowErrorCategory`, `GrowValidationError`, add "grow" to `_MUTATION_STAGES`
- `tests/unit/test_phase_gates.py`

**Lines:** ~250-300

**Acceptance:**
- `PhaseGateHook` Protocol with `on_phase_complete`
- `GrowErrorCategory` covers INNER, SEMANTIC, STRUCTURAL_ABORT
- `has_mutation_handler("grow")` returns True
- Existing tests pass (no regressions)

---

### PR 3: GROW Stage Skeleton and Phase Runner (Runner/Plumbing)

**Scope:** GrowStage class, phase dispatch, orchestrator integration

**Dependencies:** PR 1, PR 2

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — `GrowStage`, phase dispatch, snapshot mgmt
- `src/questfoundry/pipeline/stages/__init__.py` — Register grow_stage
- `src/questfoundry/pipeline/orchestrator.py` — GROW enrichment/mutation path
- `tests/unit/stages/test_grow_skeleton.py`

**Lines:** ~350-400

**Acceptance:**
- `GrowStage` conforms to `Stage` Protocol
- `execute()` can run deterministic-only phases
- Phase runner handles ordering and gate calls
- Pre-phase snapshots saved
- Orchestrator recognizes "grow" stage

---

### PR 4: Deterministic Phases 1, 5, 6 (Feature - Core Graph Ops)

**Scope:** Beat Graph Import, Arc Enumeration, Divergence Identification

**Dependencies:** PR 3

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 1, 5, 6
- `src/questfoundry/graph/grow_algorithms.py` — Topological sort, arc enumeration, divergence detection
- `tests/unit/stages/test_grow_phase{1,5,6}.py`
- `tests/unit/test_grow_algorithms.py`

**Lines:** ~500-600

**Acceptance:**
- Phase 1: Imports beats/paths/tensions from graph; validates DAG; validates commits beats exist
- Phase 5: Enumerates path combinations; topological sort; spine arc identification
- Phase 6: Computes divergence points between arc pairs
- Tests cover 2-dilemma and 3-dilemma scenarios
- Edge cases: empty beats, single path, cycles detected

**Open questions:**
- `requires` edges not created by SEED currently. Phase 1 infers order from initial_beats array position.

---

### PR 5: Deterministic Phases 7, 8a, 8b, 11 (Feature - Graph Derivation)

**Scope:** Convergence, Passage Creation, Codeword Creation, Pruning

**Dependencies:** PR 4

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phases 7, 8a, 8b, 11
- `src/questfoundry/graph/grow_algorithms.py` — Convergence detection, passage derivation, pruning
- `tests/unit/stages/test_grow_phase{7,8a,8b,11}.py`

**Lines:** ~400-500

**Acceptance:**
- Phase 7: Identifies convergence points (post-commits rejoining)
- Phase 8a: Creates passage nodes from beats (1:1, copies summary, links entities)
- Phase 8b: Creates codeword nodes from consequences; assigns grants to commits beats
- Phase 11: BFS reachability from start; deletes unreachable nodes
- Phase gates fire after 7 and 8b

---

### PR 6: LLM Phase 2 — Path-agnostic Assessment (Feature)

**Scope:** LLM assessment of beat path-agnosticism

**Dependencies:** PR 4

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 2
- `prompts/templates/grow_phase2_agnostic.yaml`
- `tests/unit/stages/test_grow_phase2.py`
- `tests/integration/test_grow_phase2_llm.py`

**Lines:** ~300-350

**Acceptance:**
- LLM receives beat summaries grouped by dilemma
- Returns path-agnostic beat IDs per dilemma
- Validation: all IDs exist in graph
- Inner retry loop (max 3) for Pydantic failures
- Phase gate with assessment results
- Beat nodes updated with `thread_agnostic_for`

---

### PR 7: LLM Phase 3 — Intersection Detection (Feature)

**Scope:** Beat clustering into intersections with compatibility checks

**Dependencies:** PR 6

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 3
- `prompts/templates/grow_phase3_knots.yaml`
- `src/questfoundry/graph/grow_algorithms.py` — Intersection compatibility checker
- `tests/unit/stages/test_grow_phase3.py`

**Lines:** ~350-400

**Acceptance:**
- LLM clusters beats by location/entity overlap
- Compatibility check: different tensions, no requires conflicts, location resolvable
- Approved: beats get multi-path assignment, locations resolved
- Merge operation creates new beat replacing both
- Rejected: beats stay separate

---

### PR 8: LLM Phases 4a-4c — Gap Detection and Scene-Type Tagging (Feature)

**Scope:** Scene-type tagging, narrative gaps, pacing gaps

**Dependencies:** PR 7

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phases 4a, 4b, 4c
- `prompts/templates/grow_phase4{a,b,c}_*.yaml`
- `tests/unit/stages/test_grow_phase4.py`

**Lines:** ~450-550

**Acceptance:**
- 4a: Each beat gets `scene_type` (scene/sequel/micro_beat)
- 4b: LLM traces paths, proposes gap beats
- 4c: LLM flags pacing issues, proposes correction beats
- Gap beats added to graph with proper path/dilemma assignments
- Three phase gates (one per sub-phase)
- Completion: each path connected, all beats tagged

**Note:** If >500 lines, split 4b/4c into separate PR.

---

### PR 9: LLM Phase 8c — Entity Overlay Creation (Feature)

**Scope:** Overlay proposals from consequences

**Dependencies:** PR 5

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 8c
- `prompts/templates/grow_phase8c_overlays.yaml`
- `tests/unit/stages/test_grow_phase8c.py`

**Lines:** ~200-250

**Acceptance:**
- LLM receives consequences with ripples and codewords
- Proposes overlays for affected entities (when/details)
- Validation: entity and codeword IDs exist
- Phase gate for human review
- Approved overlays stored on entity nodes

---

### PR 10: Phase 9 — Choice Derivation (Feature)

**Scope:** Choice edge creation with LLM-generated diegetic labels

**Dependencies:** PR 5

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 9
- `prompts/templates/grow_phase9_choices.yaml`
- `src/questfoundry/graph/grow_algorithms.py` — Successor analysis
- `tests/unit/stages/test_grow_phase9.py`

**Lines:** ~300-350

**Acceptance:**
- Deterministic: identify multi-successor divergence points
- Single-successor: implicit "continue" edges
- Multi-successor: LLM generates diegetic labels
- Choice edges with from_passage, to_passage, label, requires, grants
- Labels never generic ("Continue", "Go left")

---

### PR 11: Phase 10 — Validation (Feature)

**Scope:** Graph integrity validation

**Dependencies:** PR 10

**Key files:**
- `src/questfoundry/pipeline/stages/grow.py` — Phase 10
- `src/questfoundry/graph/grow_validation.py` — Integrity checks
- `tests/unit/stages/test_grow_phase10.py`
- `tests/unit/test_grow_validation.py`

**Lines:** ~400-450

**Acceptance:**
- Single start passage
- All passages reachable from start
- All endings reachable
- Each dilemma has commits resolved
- Gate satisfiability (required codewords obtainable)
- No cycles in requires graph
- Commits timing warnings
- Validation report returned to gate (pass/warn/fail per check)

---

### PR 12: GROW Integration and Context (Integration/Cleanup)

**Scope:** E2E integration, context formatting, valid ID injection

**Dependencies:** PR 11

**Key files:**
- `src/questfoundry/graph/context.py` — `format_grow_valid_ids_context()`, stage="grow"
- `src/questfoundry/graph/grow_context.py` — Context compression for 32k models
- `tests/integration/test_grow_e2e.py` — Full sequence with mocked LLM
- `tests/unit/test_grow_context.py`

**Lines:** ~300-350

**Acceptance:**
- `format_valid_ids_context(graph, "grow")` returns correct ID lists
- Context handles >100 beats without exceeding 32k tokens
- E2E test runs all 11 phases on fixture graph
- 2-dilemma, 4-arc fixture passes all phases

---

## Dependency Graph

```
PR 1 (Schemas) ─────┐
                     ├── PR 3 (Skeleton) ── PR 4 (Phases 1,5,6) ─┬── PR 5 (7,8a,8b,11) ─┬── PR 9 (8c)
PR 2 (Gates) ────────┘                                            │                       ├── PR 10 (9)
                                                                   │                       └── PR 11 (10) ── PR 12 (E2E)
                                                                   └── PR 6 (Phase 2) ── PR 7 (Phase 3) ── PR 8 (Phases 4a-c)
```

## Critical Path

Minimum PRs for a working GROW (deterministic-only, produces arcs/passages/codewords/choices):

**PR 1 → PR 2 → PR 3 → PR 4 → PR 5** = ~2,025 lines, 5 PRs

This gives a GROW that imports beats, enumerates arcs, creates passages/codewords/choices, validates, and prunes — all without LLM calls. LLM phases (6-10) add intelligence on top.

## Parallelization Opportunities

- PR 1 and PR 2 can be developed in parallel (no deps)
- PR 5 and PR 6 can be developed in parallel (both depend on PR 4)
- PR 9, PR 10 can be developed in parallel (both depend on PR 5)

## Human Gate Infrastructure

**Needed now (PR 2):** PhaseGateHook Protocol, AutoApprovePhaseGate (CI)

**Deferred (Slice 5 - UI):** Interactive CLI gates, "return to phase" recovery, "abort to SEED" flow

## Key Patterns to Apply

| Pattern | Where |
|---------|-------|
| Models-first | PR 1: define GrowResult and node types before implementation |
| Two-loop retry | PRs 6-10: inner Pydantic + outer semantic |
| Valid ID injection | PRs 6-10: inject manifests before each LLM call |
| Defensive prompts | PRs 6-10: GOOD/BAD examples per phase |
| Scoped IDs | All: beat::, arc::, passage::, codeword:: prefixes |
| Error classification | PR 2: GrowErrorCategory (INNER, SEMANTIC, STRUCTURAL_ABORT) |
| Token budget | PR 12: context compression for 32k models |
| Deterministic clarity | PRs 4-5: no LLM for phases 1, 5-8b, 11 |
