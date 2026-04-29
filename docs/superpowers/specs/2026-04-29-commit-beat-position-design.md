# SEED commit-beat structural position rule

**Status:** Approved 2026-04-29 (issue #1572)
**Owner:** @pvliesdonk
**Tracking:** #1572
**Designed with:** @prompt-engineer (advisory)

## Problem

`projects/murder-haiku` (claude-haiku-4-5, 2026-04-29 fresh run) failed at GROW R-1.4 Y-fork postcondition. Diagnostic traced into `graph.db` confirmed the failure is in **SEED's LLM output**, not GROW's wiring or the mutation layer:

- Both `belongs_to` graph edges on the shared pre-commit beat exist correctly.
- Both paths have full commit-beat chains.
- The asymmetry is in `dilemma_impacts.effect` placement:

```
__executor_beat_01:    effect: commits   ← first exclusive beat (correct)
__conspirator_beat_01: effect: advances  ← first exclusive beat (BUG)
__conspirator_beat_02: effect: commits   ← out of position
```

Per `docs/design/story-graph-ontology.md` Part 1, the **commit beat IS structurally defined as the first beat exclusive to one path**. The narrative meaning ("when does the character commit?") is a separate concern from the structural slot ("which beat is first-exclusive in path P?"). The LLM treated `effect: commits` as a semantic claim and placed it on the beat that *narratively* felt most committal — on the conspirator path that was beat 2 (active suppression), not beat 1 (passive listening).

R-1.4 fires at GROW time because the validator walks predecessor edges from the shared beat and checks: for each successor, does it have `dilemma_impacts.effect == "commits"` AND `len(belongs_to) == 1`? On the executor path it finds beat_01 (commits + single path → counted). On the conspirator path it finds beat_01 (advances → not counted) → "missing commit beats."

Same root-cause class as the `also_belongs_to` murder6 bug (#1564 / #1561): small/medium models miss a **structural** constraint when it's expressed in **semantic** terms in the prompt.

## Non-goals

- Restructuring the SEED → GROW pipeline.
- Changing R-1.4's behaviour at GROW time. R-1.4 is correct; the fix prevents the violation upstream rather than relaxing the check.
- Touching DRESS / FILL / POLISH prompts. The position rule is SEED-specific.
- Adding a SEED-stage runtime validator that checks position before GROW. Filed separately if needed; the prompt fix alone should suffice for capable models.

## Design

### Spec layer (lands first per CLAUDE.md "Spec-first fix order")

**`docs/design/procedures/seed.md` R-3.11.** Currently:

> R-3.11. Every explored Path has exactly one commit beat.

Replace with:

> R-3.11. Every explored Path has exactly one commit beat, and that commit beat MUST be the first exclusive beat in the path's beat sequence. No `advances`, `reveals`, or `complicates` beat may precede it in the exclusive (post-pre-commit) region. The structural slot is fixed by the graph definition (Story Graph Ontology Part 1, "Commit beat is the first beat exclusive to one path") — narrative judgment about "when the dilemma feels most decisive" does not relocate the slot.

**Violations table.** Add a new row:

> | First exclusive beat has `effect: advances` and commit beat is at position 2+ | Commit beat placed on wrong structural slot — position is fixed by the graph definition (SGO Part 1), not by narrative judgment | R-3.11 |

The Story Graph Ontology Part 1 already describes commit beats correctly ("the first beat exclusive to one path"); no ontology change needed.

### Prompt layer (PR-B, depends on PR-A merged)

**`prompts/templates/serialize_seed_sections.yaml` `per_path_beats_prompt`** (lines ~495–671):

1. **Insert new HARD CONSTRAINT block** (after line 641, end of existing "COMMITS BEAT REQUIREMENT"):

```
## COMMIT BEAT POSITION (HARD CONSTRAINT — structural, not narrative)

The FIRST beat in your list MUST have `effect: "commits"` in `dilemma_impacts`.

WHY: In the Y-shape, the commit beat IS defined as the first beat exclusive to this
path. Its slot is fixed by graph structure, not by when the decision "feels dramatic."
Even if the first exclusive moment is quiet or passive, it is still the structural
commit beat. You prove the answer in the beats that FOLLOW it.

GOOD (structurally correct):
  Beat 1: effect: "commits"  ← first exclusive beat, regardless of dramatic weight
  Beat 2: effect: "advances" ← plays out consequences
  Beat 3: effect: "reveals"  ← proves the answer

BAD (structurally wrong — the murder-haiku failure pattern):
  Beat 1: effect: "advances"  ← "passive listening" feels less decisive
  Beat 2: effect: "commits"   ← "active suppression" feels more dramatic
  Beat 3: effect: "reveals"   ← aftermath
The beat ordering above VIOLATES the Y-fork contract. The validator checks that
the FIRST exclusive beat is the commit beat and will reject this output.
```

2. **Rewrite ARC STRUCTURE section** (lines ~629–637): drop the "Turning point" semantic label; reframe as structural slot:

```
## ARC STRUCTURE (determines beat ordering)

Beat 1 (commits): STRUCTURAL SLOT — always the first beat. The moment the
  path diverges structurally. This is NOT about dramatic weight; even a
  quiet, passive moment is the commit beat if it is the first exclusive
  beat of this path.
Beats 2–N (advances / reveals / complicates): Aftermath beats that prove
  the answer. At least one is required.

WRONG: Deciding beat 1 is "setup" and beat 2 is "commits" because beat 2
  feels more dramatically decisive. The structural slot is fixed — beat 1 IS
  the commit beat. Always.

WRONG sequence: [advances, commits] — no aftermath beat
RIGHT sequence: [commits, advances] or [commits, reveals, complicates]
```

3. **Add item 4 to FINAL VERIFICATION checklist** (after line 668):

```
4. `dilemma_impacts[0].effect` on `{path_name}_beat_01` is "commits" — the FIRST beat in your list is the commit beat.
```

4. **Add to WHAT NOT TO DO list:**

```
- Do NOT place `effect: "commits"` on beat 2 or later — it MUST be beat 1
```

**`prompts/templates/summarize_seed.yaml` VERIFY block** (after line 112), add item 5:

```
5. For each path, the FIRST post-commit beat listed under that path has
   `effect: commits` (or the beat labeled as the commit beat appears FIRST
   in the path's exclusive-beat sequence). The commit beat is ALWAYS beat 1
   of the exclusive sequence — placing it at position 2+ is a structural
   violation (seed.md R-3.11 / SGO Part 1).
```

`discuss_seed.yaml` is not touched. Creative discussion phase should not be burdened with positional slot constraints; the structural rule lives in summarize + serialize.

## File cascade

### PR-A (spec, this branch)

| # | File | Change |
|---|---|---|
| 1 | `docs/design/procedures/seed.md` | R-3.11 rewrite + new Violations row |

### PR-B (prompts, depends on PR-A merged)

| # | File | Change |
|---|---|---|
| 2 | `prompts/templates/serialize_seed_sections.yaml` | New COMMIT BEAT POSITION block + ARC STRUCTURE rewrite + FINAL VERIFICATION item 4 + WHAT NOT TO DO entry |
| 3 | `prompts/templates/summarize_seed.yaml` | VERIFY block item 5 |
| 4 | (optional) `tests/unit/test_seed_prompts.py` | Test asserting prompt template contains the position rule + GOOD/BAD examples |

## Verification

After PR-B merges:

```sh
$ rg "FIRST beat in your list MUST have" prompts/templates/serialize_seed_sections.yaml
# one hit (the new HARD CONSTRAINT block)
```

End-to-end smoke: re-run `projects/murder-haiku` against claude-haiku-4-5; confirm SEED produces symmetric Y-fork (every path's first exclusive beat has `effect: commits`).

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| LLM may write "passive listening" prose into the commit beat — the narrative slot becomes oddly low-stakes. | The prompt's GOOD example explicitly endorses this ("Even if the first exclusive moment is quiet or passive, it is still the structural commit beat"). The user-facing player experience cares about the prose, not the metadata; the metadata exists for graph correctness. If observed prose quality regresses, a separate prompt iteration can re-balance, but the structural rule cannot. |
| Spec drift: existing seed.md content references commit beat without the position clause in other places. | Cascade-nits sweep before pushing PR-A: `rg "commit beat" docs/design/procedures/seed.md` to find all references, ensure they remain consistent with the new R-3.11. |
| Prompt-only fix may not be enough for the smallest models (qwen3:4b). | Filed as a future-PR consideration: SEED-stage exit validator that fires before GROW with a clearer error attribution. The prompt fix is the first-line defense; the validator is belt-and-suspenders. |
| Bot review may find additional cascading nits in adjacent prompt sections. | Per phase3_cluster_state memory note, run `rg "Turning point|first.*beat|commit.*position" prompts/templates/` after PR-B's main edits to catch any sibling rule that should also reference the new position constraint. |

## Out of scope

- SEED-stage exit validator. Filed separately if needed.
- Cascading nit at lines 622 of `serialize_seed_sections.yaml` ("effect must be 'advances', 'reveals', 'commits', or 'complicates'") — already correct, no fix needed but flagged by audit for completeness.
- Sibling soft finding (no explicit BAD example for the double-commits pattern) — caught by the cardinality validator at runtime, lower priority than the position fix that slips through silently.

## References

- Issue #1572 — this design's tracker.
- Issue #1562 — closed-as-misdiagnosed (originally hypothesized GROW Phase 1 / mutation-layer bug; actually SEED prompt).
- `docs/design/story-graph-ontology.md` Part 1 — Commit beat structural definition.
- `docs/design/procedures/seed.md` R-3.7, R-3.11 — current commit beat rules.
- `src/questfoundry/pipeline/stages/grow/deterministic.py:241-299` — R-1.4 validator (failure surface).
- @prompt-engineer post-mortem — full audit.
