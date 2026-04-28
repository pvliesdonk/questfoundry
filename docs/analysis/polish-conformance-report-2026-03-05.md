# POLISH Stage — Design Conformance Report

**Date**: 2026-03-05
**Reviewer**: architect-reviewer subagent (adversarial review)
**Design documents reviewed**:
- `docs/design/how-branching-stories-work.md`
- `docs/design/story-graph-ontology.md`
- `docs/design/procedures/polish.md`

**Implementation reviewed**:
- `src/questfoundry/pipeline/stages/polish/`
- `src/questfoundry/graph/polish_validation.py`
- `src/questfoundry/graph/polish_context.py`
- `src/questfoundry/models/polish.py`

**Example project**: `projects/test-new/` — POLISH has never run on this project (`last_stage: "grow"`). Zero passage nodes, zero choice edges, zero character arc metadata nodes in the graph. All POLISH tests construct hand-built fixtures with no evidence they match real GROW output.

**Issues filed**: #1152–#1163

---

## Conformance Table

| # | Requirement | Source | Status | Evidence / Notes |
|---|---|---|---|---|
| 1 | Entry contract validates GROW output (6 checks) | polish.md, "Entry Contract" | CONFORMANT | `polish_validation.py:38-115` |
| 2 | Entry contract: arc traversal completeness | polish.md, Entry Contract bullet 6 | MISSING | `validate_grow_output()` does not check arc traversal completeness → **#1160** |
| 3 | Phase 1: beat reordering | polish.md, Phase 1 | CONFORMANT | `llm_phases.py:54-155` |
| 4 | Phase 1: invalid reorderings → `PolishPlan.warnings` | polish.md, Phase 1 | PARTIAL | Warnings logged but lost before PolishPlan exists (created in Phase 4) → **#1159** |
| 5 | Phase 2: pacing flags (3+ scene/sequel runs, post-commit sequel) | polish.md, Phase 2 | CONFORMANT | `llm_phases.py:645-794` |
| 6 | Phase 2: micro-beats with `role: "micro_beat"` in linear sections | polish.md, Phase 2 | CONFORMANT | `llm_phases.py:797-850` |
| 7 | Phase 3: character arc synthesis (entities in 2+ beats) | polish.md, Phase 3 | CONFORMANT | `llm_phases.py:212-288` |
| 8 | Phase 3: arc metadata annotated on entity nodes (via graph edge) | Story Graph Ontology, Part 1 | PARTIAL | Stored as orphan `character_arc_metadata::` nodes; no `has_arc_metadata` edge to entity; consumers match by string → **#1154** |
| 9 | Human gate after Phase 3 | polish.md, "Human Gates" | PARTIAL | Gate fires after every phase generically; no special post-Phase-3 behavior. AutoApprovePhaseGate auto-approves. |
| 10 | Phase 4a: beat grouping (intersection, collapse, singleton) | polish.md, Phase 4a | CONFORMANT | `deterministic.py:181-299` |
| 11 | Phase 4b: prose feasibility audit (two passes, 5 categories) | polish.md, Phase 4b | CONFORMANT | `deterministic.py:307-426` |
| 12 | Phase 4b: ambiguous cases escalate to LLM review in Phase 5 | polish.md, Phase 4b, Pass 2 | MISSING | No LLM escalation; heuristic decision only → **#1157** |
| 13 | Phase 4c: `ChoiceSpec.requires` populated (post-convergence gating) | polish.md, Phase 4c | MISSING | `compute_choice_edges` computes `grants` only; `requires` always empty list → **#1152** |
| 14 | Phase 4d: false branch candidates via actual passage adjacency | polish.md, Phase 4d | PARTIAL | Uses spec list order (acknowledged in comment line ~546-547) instead of graph adjacency → **#1161** |
| 15 | `PolishPlan.arc_traversals` populated | polish.md, PolishPlan | DEAD | Field exists, stored, but always `{}`. `compute_passage_traversals()` exists in `algorithms.py` but never called → **#1153** |
| 16 | Phase 5a: choice labels | polish.md, Phase 5 | CONFORMANT | `llm_phases.py:321-353` |
| 17 | Phase 5b: residue beat content | polish.md, Phase 5 | CONFORMANT | `llm_phases.py:355-375` |
| 18 | Phase 5c: false branch decisions | polish.md, Phase 5 | CONFORMANT | `llm_phases.py:377-411` |
| 19 | Phase 5d: variant passage summaries | polish.md, Phase 5 | CONFORMANT | `llm_phases.py:413-433` |
| 20 | Phase 6: atomic plan application | polish.md, Phase 6 | CONFORMANT | `deterministic.py:617-710` |
| 21 | Phase 6: true single-transaction atomicity (rollback on failure) | polish.md, Phase 6 | PARTIAL | No savepoint; each mutation auto-commits; `SqliteGraphStore.savepoint()` exists but unused → **#1155** |
| 22 | Phase 7: every beat in exactly one passage (including residue) | polish.md, Phase 7 | PARTIAL | `_check_beat_grouping` exempts residue beats; they SHOULD be grouped → **#1156** |
| 23 | Phase 7: every divergence has choice edges | polish.md, Phase 7 | MISSING | Not checked → **#1156** |
| 24 | Phase 7: all endings reachable | polish.md, Phase 7 | MISSING | `check_all_endings_reachable` exists but never called → **#1156** |
| 25 | Phase 7: every variant's `requires` is satisfiable | polish.md, Phase 7 | MISSING | Not checked → **#1156** |
| 26 | Phase 7: every gated choice `requires` is satisfiable | polish.md, Phase 7 | MISSING | `check_gate_satisfiability` exists but never called → **#1156** |
| 27 | Phase 7: no passage has outgoing choices with overlapping `requires` | polish.md, Phase 7 | MISSING | Not checked → **#1156** |
| 28 | Phase 7: arc completeness (every arc = complete passage sequence) | polish.md, Phase 7 | DEAD | Structurally impossible while `arc_traversals == {}` (see #15) → **#1153, #1156** |
| 29 | Phase 7: no structural split left unresolved | polish.md, Phase 7 | MISSING | Structural split warnings from 4b are not validated against phase 7 state → **#1156** |
| 30 | Residue beats: one variant per path, gated by state flag | "How Branching Stories Work", Part 4; Story Graph Ontology, Part 5 | CONFORMANT | `deterministic.py:759-802` |
| 31 | Transition guidance for collapsed passages | "How Branching Stories Work", Part 4, "Passage Collapse" | MISSING | `_merge_summaries` joins with "; " — no bridging instructions generated → **#1158** |
| 32 | Phase 2 injects micro-beats for pacing | "How Branching Stories Work", Part 4, "Pacing" | CONFORMANT | Phase 2 |
| 33 | `grouped_in` edge (beat → passage) | Story Graph Ontology, Part 9 | CONFORMANT | `deterministic.py:740` |
| 34 | `choice` edge with label, requires, grants | Story Graph Ontology, Part 9 | CONFORMANT | `deterministic.py:805-815` |
| 35 | `variant_of` edge (variant → base passage) | Story Graph Ontology, Part 9 | CONFORMANT | `deterministic.py:756` |
| 36 | POLISH audits overlay composition for prose feasibility | Story Graph Ontology, Part 6 | MISSING | Phase 4b checks state flags vs entity overlap but not multi-overlay infeasibility → **#1162** |
| 37 | POLISH may create cosmetic codewords | Story Graph Ontology, Part 1 | DEFERRED | Story Graph Ontology line 719 explicitly defers this pattern. Filed as design discussion → **#1163** |
| 38 | Phase ordering: collapse_linear_beats between character_arcs and plan_computation | Registry | CONFORMANT | `deterministic.py:66-69` |

---

## Summary

| Status | Count |
|--------|-------|
| CONFORMANT | 18 |
| PARTIAL | 7 |
| MISSING | 10 |
| DEAD | 3 |
| DEFERRED | 1 |
| **Total** | **39** |

---

## Critical Gaps

### DEAD

**#15 / #28: `arc_traversals` never computed** (`#1153`)
`PolishPlan.arc_traversals` is always `{}`. `compute_passage_traversals()` exists in `graph/algorithms.py` but is never called from POLISH. Arc completeness validation (Phase 7 check #28) is therefore structurally impossible. The entire concept of "verify every arc produces a complete passage sequence" is dead.

**#28: Arc completeness validation** is the downstream casualty of #15. Both are addressed by `#1153`.

### MISSING (most impactful)

**#13: `ChoiceSpec.requires` never populated** (`#1152`)
`compute_choice_edges` only computes `grants`. Post-convergence gated choices — a core design concept in "How Branching Stories Work" ("Gates appear after soft dilemma convergence") — are never generated. All choices are always visible to all players regardless of prior decisions. Soft dilemma convergence routing is non-functional.

**#23–#29: Phase 7 validation is a skeleton** (`#1156`)
Of 10+ required checks, only ~5 are implemented. `check_all_endings_reachable` and `check_gate_satisfiability` exist but are never called. Critical structural invariants (divergences have choices, arc completeness, no unresolved splits) are unchecked.

---

## Data Flow Breaks

| Chain | Producer | Consumer | Status |
|---|---|---|---|
| `ChoiceSpec.requires` | `compute_choice_edges` — never populates `requires` | Phase 6 `_create_choice_edge`, FILL gated routing | **Broken at producer** |
| `arc_traversals` | `phase_plan_application` — always writes `{}` | Phase 7 arc completeness check | **Broken at producer; consumer doesn't exist** |
| Character arc metadata → entity | Phase 3 creates orphan nodes | FILL `fill_context.py` (string-match lookup) | **No edge; fragile by convention** |

---

## Fixture Divergence

POLISH has never been run on the example project (`projects/test-new/graph.db` shows `last_stage: "grow"`). All POLISH test fixtures hand-construct beat DAGs, intersection groups, and state flags. Whether these match what GROW actually produces is unverified.
