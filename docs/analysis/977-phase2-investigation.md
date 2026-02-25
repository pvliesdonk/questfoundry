# Phase 2 Investigation: Code Alignment with Documents 1 and 3

**Date**: 2026-02-25
**Issue**: #977 (Phase 2)
**Status**: Complete
**Branch**: `docs/977-phase2-investigation`

---

## Executive Summary

This investigation audits the entire QuestFoundry codebase against Document 3's authoritative ontology. Three parallel code audits examined: (A) DREAM/BRAINSTORM/SEED stages, (B) GROW stage's 25 phases, (C) FILL/SHIP/DRESS/CLI/cross-cutting concerns.

### Key Findings

- **12 Document 3 Appendix divergences**: All confirmed present in code
- **20 Discussion #980 gaps**: All validated; 5 additional gaps discovered
- **GROW stage split**: 13 of 25 phases move to POLISH, 12 remain (some redesigned)
- **FILL stage**: Arc-based traversal (25+ `from_beat` references in `fill_context.py`) must be rewritten
- **CLI/Orchestrator**: POLISH missing from `DEFAULT_STAGES`, `STAGE_ORDER`, and `_MUTATION_STAGE_PREREQUISITES`
- **Prompt templates**: 5 of 44 templates contain divergent terminology
- **Graph edge types**: 4 renames, 1 deletion (`arc_contains`), 2 consolidations, 2 new types needed

### Scope Estimates

| Area | Scope | Lines Affected (est.) |
|------|-------|-----------------------|
| DREAM | Trivial | <20 |
| BRAINSTORM | Small | 50-80 |
| SEED | Major | 200-400 |
| GROW | Major rewrite | 3000+ |
| FILL | Moderate | 300-500 |
| SHIP/DRESS | Small | 50-100 |
| CLI/Orchestrator | Moderate | 100-200 |
| Prompts | Small | 50-100 |
| Tests | Large (impact only) | 16,000+ lines affected |

---

## Per-Stage Audits

### DREAM (Trivial)

**Files audited:**
- `src/questfoundry/models/dream.py` — DreamArtifact model
- `src/questfoundry/pipeline/stages/dream.py` — Stage implementation

**Divergences found:** None. DREAM produces a Vision node matching Document 3 Part 1 (genre, tone, themes, audience, scope, style). No terminology mismatches.

**Estimated scope:** Trivial (<20 lines if any changes at all)

---

### BRAINSTORM (Small)

**Files audited:**
- `src/questfoundry/models/brainstorm.py`
- `src/questfoundry/graph/mutations.py` (brainstorm section)
- `src/questfoundry/graph/context.py`
- `prompts/templates/serialize_brainstorm.yaml`

**Divergences found:**

| # | Doc 3 Ref | Current Code | Required Change | File:Line | Scope |
|---|-----------|-------------|----------------|-----------|-------|
| 1 | Appendix #5 | `Answer.is_default_path: bool` | Rename to `is_canonical` | `brainstorm.py:70,78-80` | Small |
| 2 | Appendix #4 | `Dilemma.central_entity_ids: list[str]` | Create `anchored_to` edges instead | `brainstorm.py:141-144` | Moderate |

**Detail: `is_default_path` → `is_canonical`**

Currently stored as field on Answer model. Mutations at `mutations.py:1680-1699` already derive `is_canonical` on Path nodes from `is_default_path`. The rename is partially done in the graph layer but the model and prompts still use the old name.

Occurrences:
- `models/brainstorm.py:70,78-80` — Field definition and validation
- `mutations.py:1502-1503` — Reads from graph answer nodes
- `mutations.py:1680-1699` — Derives `is_canonical` from `is_default_path`
- `context.py:165-187` — `get_default_answer_from_graph()` reads `is_default_path`
- `prompts/templates/serialize_brainstorm.yaml:36` — Instructs LLM to use `is_default_path`
- Tests: ~50 occurrences across unit tests

**Detail: `central_entity_ids` → `anchored_to` edges**

Currently stored as list field on Dilemma nodes. `apply_brainstorm_mutations()` at `mutations.py:896-911` stores the list but does NOT create graph edges. Context builders read the field from node data, not via edge queries.

Downstream consumers:
- `context.py:1033-1042` — `format_interaction_candidates_context()` reads field
- `pipeline/stages/seed.py:100` — SEED reads field for entity triage
- `pipeline/stages/grow/llm_phases.py:1240` — GROW reads field during entity arcs

**Dependencies:** None — can be done independently
**Risks:** Low. Field → edge migration is additive (can keep field for backward compatibility during transition)

---

### SEED (Major)

**Files audited:**
- `src/questfoundry/models/seed.py`
- `src/questfoundry/graph/mutations.py` (seed section: lines 1058-1841)
- `src/questfoundry/graph/context.py` (seed context builders)
- `src/questfoundry/graph/seed_pruning.py`
- `prompts/templates/serialize_seed_sections.yaml`
- `prompts/templates/discuss_seed.yaml`

**Divergences found:**

| # | Doc 3 Ref | Current Code | Required Change | File:Line | Scope |
|---|-----------|-------------|----------------|-----------|-------|
| 1 | Appendix #3 | `convergence_policy: Literal["hard","soft","flavor"]` | → `dilemma_role: Literal["hard","soft"]`; remove `flavor` | `seed.py:29,261-263` | Moderate |
| 2 | Appendix #11 | `InteractionConstraint.constraint_type: Literal["shared_entity","causal_chain","resource_conflict"]` | → Pairwise edges: wraps, serial, concurrent; shared_entity derived | `seed.py:32,331-333` | Moderate |
| 3 | Gap #1 CRITICAL | `InitialBeat.paths: list[str]` | → `path_id: str` (singular) | `seed.py:221-224` | Large |
| 4 | Appendix #8 | `InitialBeat.location_alternatives: list[str]` | → Entity flexibility edges with role annotations | `seed.py:237-240` | Small-Moderate |
| 5 | Appendix #12 | No temporal hints field | Add `temporal_hint` field to InitialBeat | `seed.py:203` (missing) | Small |
| 6 | Appendix #11 | No dilemma ordering edges | Add wraps/serial/concurrent pairwise edges | `mutations.py` (missing) | Moderate |
| 7 | Gap #14 | `PathTier = Literal["major","minor"]` | No Doc 1/3 counterpart; document or remove | `seed.py:27,172` | Small |

**Detail: `convergence_policy` → `dilemma_role`**

Type alias at `seed.py:29`: `ConvergencePolicy = Literal["hard", "soft", "flavor"]`
Field at `seed.py:261-263`: `convergence_policy: ConvergencePolicy`
Graph storage at `mutations.py:1807`: stores on dilemma node

Prompt teaching at `serialize_seed_sections.yaml:679-710` explicitly teaches `flavor` classification with examples. This is ~30 lines of prompt that must be removed/rewritten.

All occurrences (9 files):
- `models/seed.py:29,261` — Type alias + field
- `graph/mutations.py:1807` — Storage
- `graph/context.py:991-992` — Context generation mentions flavor
- `graph/grow_algorithms.py` — Multiple reads (convergence logic)
- `graph/grow_validation.py` — Validation against policy
- `prompts/templates/serialize_seed_sections.yaml:679-710,814-846` — Extensive prompt teaching
- Tests: multiple files

**Detail: `InitialBeat.paths: list[str]` → singular (CRITICAL)**

Field at `seed.py:221-224`: `paths: list[str]` with `min_length=1`
Mutations at `mutations.py:1782-1784`: Creates one `belongs_to` edge per path in list
Validation at `mutations.py:1317-1326`: Validates each path in list
Prompt at `serialize_seed_sections.yaml:415`: Example shows array syntax

This is the single most critical divergence. A beat belonging to multiple paths is the structural root cause of hard-convergence violations. Document 3 requires exactly one `belongs_to` edge per beat; intersection groups handle multi-path co-occurrence separately.

**Dependencies:**
- `convergence_policy` removal depends on `flavor` deprecation mapping (#984)
- `InitialBeat.paths` singular depends on intersection group model existing (#983)
- `InteractionConstraint` redesign is independent (#985)

**Risks:**
- `InitialBeat.paths` change is structurally incompatible with old data — existing graphs need migration
- `flavor` removal breaks existing project classifications — needs deprecation mapping
- `InteractionConstraint` redesign changes how SEED prompts work — needs prompt testing

---

### GROW (Major Rewrite)

**Files audited:**
- `src/questfoundry/pipeline/stages/grow/deterministic.py` (883 lines, 13 phases)
- `src/questfoundry/pipeline/stages/grow/llm_phases.py` (2099 lines, 12 phases)
- `src/questfoundry/pipeline/stages/grow/registry.py` (phase registry)
- `src/questfoundry/graph/grow_algorithms.py` (3752 lines, ~55 functions)
- `src/questfoundry/graph/grow_validation.py` (1580 lines, ~15 checks)
- `src/questfoundry/graph/grow_routing.py` (1051 lines — entire module moves)
- `src/questfoundry/models/grow.py` (Arc, Passage, Codeword, Choice, EntityOverlay)

#### Phase Disposition Table

| Priority | Phase | File:Line | Type | Disposition |
|----------|-------|-----------|------|-------------|
| 0 | `validate_dag` | deterministic.py:32 | Det. | **STAYS** |
| 2 | `scene_types` | llm_phases.py:301 | LLM | **MOVES_TO_POLISH** |
| 3 | `narrative_gaps` | llm_phases.py:387 | LLM | **MOVES_TO_POLISH** |
| 4 | `pacing_gaps` | llm_phases.py:485 | LLM | **MOVES_TO_POLISH** |
| 5 | `atmospheric` | llm_phases.py:624 | LLM | **MOVES_TO_POLISH** |
| 6 | `path_arcs` | llm_phases.py:704 | LLM | **STAYS+REDESIGN** |
| 7 | `intersections` | llm_phases.py:56 | LLM | **STAYS+REDESIGN** |
| 8 | `entity_arcs` | llm_phases.py:853 | LLM | **STAYS+RENAME** |
| 9 | `enumerate_arcs` | deterministic.py:70 | Det. | **REFACTOR** → validation utility |
| 10 | `divergence` | deterministic.py:159 | Det. | **REFACTOR** → computed metadata |
| 11 | `convergence` | deterministic.py:232 | Det. | **REFACTOR** → state-flag derivation |
| 12 | `collapse_linear_beats` | deterministic.py:333 | Det. | **STAYS** |
| 13 | `passages` | deterministic.py:376 | Det. | **MOVES_TO_POLISH** |
| 14 | `codewords` | deterministic.py:432 | Det. | **STAYS+RENAME** → state_flags |
| 15 | `residue_beats` | llm_phases.py:1039 | LLM | **MOVES_TO_POLISH** |
| 16 | `overlays` | llm_phases.py:1174 | LLM | **STAYS+RENAME** |
| 17 | `choices` | llm_phases.py:1364 | LLM | **MOVES_TO_POLISH** |
| 18 | `fork_beats` | llm_phases.py:1668 | LLM | **MOVES_TO_POLISH** |
| 19 | `hub_spokes` | llm_phases.py:1927 | LLM | **MOVES_TO_POLISH** |
| 20 | `mark_endings` | deterministic.py:534 | Det. | **MOVES_TO_POLISH** |
| 21 | `apply_routing` | deterministic.py:566 | Det. | **MOVES_TO_POLISH** |
| 22 | `collapse_passages` | deterministic.py:642 | Det. | **MOVES_TO_POLISH** |
| 23 | `validation` | deterministic.py:689 | Det. | **STAYS+REDESIGN** (beat DAG only) |
| 25 | `prune` | deterministic.py:765 | Det. | **MOVES_TO_POLISH** |

**Summary:** 13 phases move to POLISH, 12 remain (5 unchanged, 4 renamed, 3 redesigned/refactored)

#### Key Functions in grow_algorithms.py

**Arc-dependent functions (must refactor):**
- `enumerate_arcs()` — line 1306 — Returns `list[Arc]`, creates arc nodes → becomes validation utility
- `compute_divergence_points()` — line 1458 — Takes `list[Arc]` → becomes beat DAG analyzer
- `find_convergence_points()` — line 1806 — Takes `list[Arc]` → becomes state-flag derivation
- `build_arc_codewords()` — line 841 — Maps arcs to codeword signatures → rename + refactor
- `split_ending_families()` — line 712 — Groups arcs by codewords → moves to POLISH

**Intersection functions (must redesign):**
- `apply_intersection_mark()` — line 3306 — **ROOT CAUSE of hard-convergence bug**. Currently cross-assigns `belongs_to` edges (lines 3329-3351). Must create `intersection_group` node + `PARTICIPATES_IN` edges instead.
- `build_intersection_candidates()` — line 2694 — Pre-clustering, stays
- `check_intersection_compatibility()` — line 3040 — Compatibility check, stays
- `resolve_intersection_location()` — line 3251 — Location resolution, stays

**Passage-layer functions (all move to POLISH):**
- `mark_terminal_passages()` — line 648
- `collapse_linear_passages()` — line 542
- `split_and_reroute()` — line 937
- `find_passage_successors()` — line 3603
- `create_residue_passages()` — line 2052
- `find_residue_candidates()` — line 1941

**Reference counts:**
- `Arc` model: 23 direct references across grow_algorithms.py
- `codeword` (term): 91 occurrences in grow_algorithms.py
- `sequenced_after` (edge): 6+ references in grow_algorithms.py
- `belongs_to` (edge): 15 references across 10+ functions

#### Validation Checks Disposition (grow_validation.py)

**Stay in GROW (beat DAG validation):**
- `check_single_start()` — line 142
- `check_dilemmas_resolved()` — line 274
- `check_passage_dag_cycles()` — line 449
- `check_spine_arc_exists()` — line 644 (refactored for computed arcs)
- `check_convergence_policy_compliance()` — line 1017

**Move to POLISH (passage-layer validation):**
- `check_all_passages_reachable()` — line 180
- `check_all_endings_reachable()` — line 219
- `check_gate_satisfiability()` — line 336
- `check_gate_co_satisfiability()` — line 378
- `check_commits_timing()` — line 512
- `check_codeword_gate_coverage()` — line 1142
- `check_forward_path_reachability()` — line 1189
- `check_routing_coverage()` — line 1244
- `check_prose_neutrality()` — line 1390
- `check_max_consecutive_linear()` — line 931
- `check_arc_divergence()` — line 679 (becomes validation utility)

#### Routing Module (grow_routing.py — entire module moves)

The entire `grow_routing.py` (1051 lines) implements passage-layer routing and moves to POLISH:
- `VariantPassageSpec` — line 33
- `RoutingOperation` — line 112
- `RoutingPlan` → becomes `PolishPlan`
- `compute_routing_plan()` — line 694
- `apply_routing_plan()` — line 901
- All helper functions

**Dependencies:**
- Intersection redesign (#983) must happen before or with POLISH PR 2 (#988)
- Arc removal (#989) depends on POLISH validation being in place
- Codeword → state_flag rename (#984) can happen independently

**Risks:**
- GROW is the largest module (~10,000 lines across 7 files). Phase migration must be incremental.
- `apply_intersection_mark()` redesign is the highest-risk single change — it's the root cause of the hard-convergence bug
- Arc removal has ~100 references across 5+ files — needs careful materialized_arc transition strategy
- grow_algorithms.py (3752 lines) is the single most affected file

---

### FILL (Moderate)

**Files audited:**
- `src/questfoundry/pipeline/stages/fill.py` (generation logic)
- `src/questfoundry/graph/fill_context.py` (2383 lines)
- `src/questfoundry/graph/fill_validation.py`

**Divergences found:**

| # | Doc 3 Ref | Current Code | Required Change | File:Line | Scope |
|---|-----------|-------------|----------------|-----------|-------|
| 1 | Appendix #6 | Arc-based passage ordering | Beat DAG traversal | `fill.py:881-899` | Moderate |
| 2 | Appendix #7 | `from_beat`/`from_beats` pattern | `grouped_in` edge queries | `fill_context.py`: 25+ refs | Large |
| 3 | Appendix #2 | `family_codewords` on passages | State flag terminology | `fill_context.py:1607` | Small |

**Detail: Arc-based traversal**

FILL uses `get_spine_arc_id()` (line 881) and `get_arc_passage_order(graph, arc_id)` (lines 886, 895) to determine writing order. It iterates arcs in sequence, retrieving passages per arc. With arcs becoming computed traversals, FILL must compute writing order from the passage/choice graph directly.

Key locations:
- `fill.py:881-899` — Generation order logic
- `fill.py:1115-1142` — `arc_passage_orders` dict
- `fill.py:1160-1169` — Context functions expect `arc_id` parameter

**Detail: `from_beat` references**

`fill_context.py` has 25+ references to `from_beat`/`from_beats`/`primary_beat` pattern at lines: 253, 414, 445, 628, 653, 675, 755, 1377, 1430, 1727, 2059, 2157. These assume 1:1 beat-passage mapping. Document 3 specifies 1:N via `grouped_in` edges.

**Dependencies:**
- POLISH must exist before FILL can receive its output
- Arc removal must be complete before FILL traversal can be rewritten
- `grouped_in` edge type must be created by POLISH

**Risks:**
- FILL is the prose generation stage — changes to context building affect output quality
- `fill_context.py` (2383 lines) is complex with many passage-context interactions
- Writing order change (arc-based → DAG-based) needs careful testing

---

### SHIP / DRESS / CLI (Moderate)

**Files audited:**
- `src/questfoundry/pipeline/stages/ship.py`
- `src/questfoundry/export/context.py`
- `src/questfoundry/pipeline/stages/dress.py`
- `src/questfoundry/cli.py` (2761 lines)
- `src/questfoundry/pipeline/orchestrator.py` (845 lines)
- `src/questfoundry/pipeline/config.py`

#### SHIP: State Flag Projection

Current: `export/context.py:130-136` — `_extract_codewords()` reads codeword nodes directly
Required: SHIP must decide which state flags become player-facing codewords (Document 3, Part 1)

Logic needed:
- Soft dilemma routing gates → routing codewords
- Cosmetic flavor → cosmetic codewords (deferred per Doc 3 Future Extensions)
- Otherwise: suppress from export

#### DRESS: Minimal Impact

DRESS operates on passages (not arcs directly). References to `passage_id` and `entity_id` are stable. No divergences found.

#### CLI + Orchestrator: POLISH Missing

**`config.py:17`**: `DEFAULT_STAGES = ["dream", "brainstorm", "seed", "grow", "fill", "ship"]` — missing `"polish"`
**`cli.py:101`**: `STAGE_ORDER = ["dream", "brainstorm", "seed", "grow", "fill", "dress", "ship"]` — missing `"polish"`
**`orchestrator.py:49-53`**: `_MUTATION_STAGE_PREREQUISITES` — missing entries for grow, polish, fill, dress

Required changes:
1. `config.py:17` — Insert `"polish"` between `"grow"` and `"fill"`
2. `cli.py:101` — Insert `"polish"` in STAGE_ORDER
3. `orchestrator.py:49-53` — Add polish entry and complete the chain
4. New CLI command `qf polish` with `--phase` support
5. Stage registration in `stages/__init__.py`

---

## Cross-Cutting Concerns

### Models

| Model | File | Change | Type |
|-------|------|--------|------|
| `Answer.is_default_path` | `brainstorm.py:78` | Rename → `is_canonical` | Rename |
| `Dilemma.central_entity_ids` | `brainstorm.py:141` | → `anchored_to` edges | Semantic |
| `DilemmaAnalysis.convergence_policy` | `seed.py:261` | → `dilemma_role`; remove `flavor` | Semantic + removal |
| `InteractionConstraint.constraint_type` | `seed.py:331` | Full redesign → wraps/serial/concurrent | Redesign |
| `InitialBeat.paths` | `seed.py:221` | `list[str]` → `path_id: str` | **CRITICAL** schema change |
| `InitialBeat.location_alternatives` | `seed.py:237` | → flexibility edges | Semantic |
| `InitialBeat` (missing) | `seed.py:203` | Add `temporal_hint` field | New feature |
| `Arc` | `grow.py:32-47` | Remove as stored node → computed | Removal |
| `Passage.from_beat` | `grow.py:50-60` | → `grouped_in` edges (1:N) | Semantic |
| `Codeword` | `grow.py:63-72` | Rename → `StateFlag` | Rename |
| `ConvergencePolicy` type | `seed.py:29` | Remove `flavor` value | Removal |
| `ConstraintType` type | `seed.py:32` | Replace values entirely | Redesign |
| `PathTier` type | `seed.py:27` | Document or remove | Decision needed |

### Graph Edge Types

| Current | Document 3 | Action | Locations |
|---------|-----------|--------|-----------|
| `sequenced_after` | `predecessor`/`successor` | Rename | grow_algorithms.py (6+ refs), mutations.py, tests |
| `passage_from` | `grouped_in` | Rename + semantics (1:N) | grow_algorithms.py:1225,1227 |
| `arc_contains` | (none) | Delete | deterministic.py:147 |
| `choice_from`+`choice_to` | `choice` | Consolidate | grow_algorithms.py:1009,1010 |
| `belongs_to` | `belongs_to` | No change (but enforce singular) | Multiple |
| `has_answer` | `has_answer` | No change | mutations.py:933 |
| `explores` | `explores` | No change | mutations.py:1709 |
| (missing) | `anchored_to` | Create | Brainstorm mutations |
| (missing) | `wraps`/`serial`/`concurrent` | Create | Seed mutations |
| (missing) | `participates_in` | Create | POLISH (intersection groups) |
| (missing) | `flexibility` | Create | Seed mutations |

### Prompt Templates

5 of 44 templates contain divergent terminology:

| Template | Terms Found | Lines |
|----------|------------|-------|
| `serialize_brainstorm.yaml` | `is_default_path`, `central_entity_ids` | 36, 37-38 |
| `discuss_seed.yaml` | `is_default_path`, `central_entity_ids` | 28, 32, 68, 103 |
| `serialize_seed.yaml` | `is_default_path`, `convergence_policy` | Multiple |
| `serialize_seed_sections.yaml` | `convergence_policy` (3x with context), `flavor` examples | 118, 125, 132, 679-710, 814-846, 901-913 |
| `grow_phase8d_residue.yaml` | `codeword` references | 19, 31, 50, 59 |

### Tests (Impact Assessment)

| Test File | Lines | Impact |
|-----------|-------|--------|
| `test_grow_algorithms.py` | ~6,668 | Major — phase reorg, arc removal |
| `test_grow_stage.py` | ~3,581 | Major — phase order, mock setups |
| `test_grow_validation.py` | ~2,598 | Major — checks move to POLISH |
| `test_grow_models.py` | ~767 | Moderate — Arc model removed |
| `test_fill_context.py` | ~2,798 | Moderate — arc traversal, from_beat |
| `test_mutations.py` | ~2,000 | Moderate — brainstorm+seed changes |
| Other test files | ~1,500 | Minor terminology updates |

**Total estimated test impact:** ~20,000 lines across 7+ test files

---

## Dependency Graph

```
Group 1 (docs, no code):
  #981 (Doc 1/3 fixes)
  #982 (procedures/polish.md)
  #986 (ADR updates)

Group 2 (model alignment, parallel within group):
  #983 (InitialBeat.paths singular) ─── depends on #981
  #984 (terminology renames) ─────────── depends on #981
  #985 (InteractionConstraint) ────────── depends on #981

Group 3 (POLISH, sequential):
  #987 (POLISH skeleton + Ph 1-3) ──── depends on #982, #984
  #988 (POLISH Ph 4-6) ─────────────── depends on #987, #983
  #989 (POLISH Ph 7 + GROW removal) ── depends on #988

New issues (see below):
  FILL context rewrite ──────────────── depends on #988
  CLI + orchestrator wiring ─────────── depends on #987
  Edge type renames ─────────────────── depends on #984
  SHIP state flag projection ────────── depends on #989
  Arc model removal ─────────────────── depends on #989
```

---

## Risk Assessment

### High Risk

| Change | Risk | Mitigation |
|--------|------|-----------|
| `apply_intersection_mark()` redesign | Root cause of hard-convergence bug; incorrect implementation breaks the pipeline | Incremental: create intersection group model first, then switch function |
| GROW phase migration to POLISH | 13 phases moving; registry DAG must remain valid during transition | Move phases one at a time; keep both registries valid at each step |
| `InitialBeat.paths` → singular | Incompatible with existing graph data; every test uses list | Add `path_id` field with backward-compatible `paths` property; migrate tests incrementally |
| FILL arc-based traversal rewrite | 2383 lines of context code with complex passage interactions | Write POLISH first so FILL has correct input format to test against |

### Medium Risk

| Change | Risk | Mitigation |
|--------|------|-----------|
| `convergence_policy` → `dilemma_role` | 9 files, prompt changes affect LLM behavior | `flavor` → `soft` deprecation mapping first; prompt testing |
| grow_algorithms.py changes | 3752 lines, largest single file | Rename codeword → state_flag separately from structural changes |
| grow_validation.py split | Checks interleave beat DAG and passage validation | Tag each check with layer before moving |

### Low Risk

| Change | Risk | Mitigation |
|--------|------|-----------|
| `is_default_path` → `is_canonical` | Simple rename, partially done | Mechanical, well-scoped |
| `sequenced_after` → `predecessor` | Edge type rename, DAG structure unchanged | Bulk rename + grep verification |
| Prompt terminology updates | 5 templates, limited scope | Update alongside model changes |

---

## Validation of Issues #981–#989

| Issue | Status | Scope Assessment | Recommended Changes |
|-------|--------|------------------|---------------------|
| #981 (Doc 1/3 fixes) | **Valid** | Correctly scoped | None |
| #982 (procedures/polish.md) | **Valid** | Correctly scoped | None |
| #983 (InitialBeat.paths) | **Valid, CRITICAL** | Confirmed: `seed.py:221`, `mutations.py:1782-1784`, prompts | Add note about graph migration for existing data |
| #984 (terminology) | **Valid but larger than estimated** | 91 `codeword` refs in grow_algorithms.py alone; 9 files for `convergence_policy` | Consider splitting: codeword rename separate from convergence_policy rename |
| #985 (InteractionConstraint) | **Valid** | Confirmed: `seed.py:32,331`; also `ConstraintType` and `PathTier` | Add `PathTier` decision to scope |
| #986 (ADR updates) | **Valid** | Correctly scoped | None |
| #987 (POLISH PR 1) | **Valid** | Phase disposition confirmed: scene_types, narrative_gaps, pacing_gaps, atmospheric move to POLISH (Phases 1-3 equivalent) | Confirm `compute_active_flags_at_beat()` replaces `find_residue_candidates()` at `grow_algorithms.py:1941` |
| #988 (POLISH PR 2) | **Valid, may need splitting** | Passage creation + routing = ~1500 lines moving from GROW. `grow_routing.py` entire module moves. | Explicitly note `grow_routing.py` as primary source |
| #989 (POLISH PR 3 + GROW removal) | **Valid** | 13 phases removed from GROW; validation split confirmed (~10 checks move, ~5 stay) | Add explicit phase list and validation check list |

---

## Issues to Create (Gaps Not Covered by #981–#989)

### 1. `refactor: sequenced_after → predecessor/successor edge rename`

Edge type `sequenced_after` appears in:
- `grow_algorithms.py`: lines 110, 285, 383, 508, 510, 2943, 3018, 3167, 3184
- `mutations.py`: seed mutation section
- `grow_validation.py`: multiple checks
- Tests: extensive references

Scope: Small-moderate (mechanical rename). Should happen with or before #984 (terminology).

### 2. `feat: FILL context rewrite for POLISH output`

`fill_context.py` (2383 lines) assumes arc-based traversal and 1:1 beat-passage mapping:
- Arc-based ordering: `fill.py:881-899`
- `from_beat`/`from_beats`: 25+ references in fill_context.py
- `arc_passage_orders`: `fill.py:1115-1142`

Must be rewritten after POLISH exists (#988). Depends on `grouped_in` edge type.

### 3. `feat: SHIP state-flag-to-codeword projection`

`export/context.py:130-136`: `_extract_codewords()` reads codeword nodes directly.
Document 3 requires SHIP to decide which state flags become player-facing codewords.
Depends on state_flag rename (#984) and POLISH completion (#989).

### 4. `feat: CLI qf polish command + orchestrator wiring`

Missing from:
- `config.py:17` — `DEFAULT_STAGES`
- `cli.py:101` — `STAGE_ORDER`
- `orchestrator.py:49-53` — `_MUTATION_STAGE_PREREQUISITES`
- No `qf polish` command exists

Should be created with POLISH skeleton (#987).

### 5. `refactor: Arc model removal (stored → computed)`

`Arc` model at `grow.py:32-47` with 23 references in `grow_algorithms.py`.
Functions: `enumerate_arcs()`, `compute_divergence_points()`, `find_convergence_points()`, `build_arc_codewords()`, `split_ending_families()`.
Edge type `arc_contains` created at `deterministic.py:147`.

This is implicitly part of #989 but the blast radius (~100 references, 5+ files) warrants explicit tracking.

---

## Recommended Implementation Order

1. **#981** — Doc 1/3 fixes (prerequisite for all)
2. **#982** — Write procedures/polish.md (design before code)
3. **#986** — ADR updates (docs, no code)
4. **#984** — Terminology renames (mechanical, unblocks everything)
5. **New: edge rename** — `sequenced_after` → `predecessor` (mechanical, alongside #984)
6. **#983** — InitialBeat.paths singular (CRITICAL structural fix)
7. **#985** — InteractionConstraint redesign
8. **#987 + New: CLI wiring** — POLISH skeleton + CLI (unblocks POLISH phases)
9. **#988** — POLISH Phases 4-6 (passage layer)
10. **#989 + New: Arc removal** — POLISH Phase 7 + GROW cleanup
11. **New: FILL rewrite** — FILL context builders for POLISH output
12. **New: SHIP projection** — State flag → codeword projection

---

## References

- Document 1: `docs/design/how-branching-stories-work.md`
- Document 3: `docs/design/document-3-ontology.md` (Appendix A: lines 694-793)
- Discussion #980: Design review deliberation (3 rounds)
- Issue #977: Investigation epic
- Issue #990: Implementation epic (tracks #981-#989)
