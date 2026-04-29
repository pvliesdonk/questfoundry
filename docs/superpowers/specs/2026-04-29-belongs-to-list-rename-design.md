# `belongs_to: list[str]` rename — restore symmetric path membership across spec, schema, code

**Status:** Approved 2026-04-29 (issue #1564, scope refined by user)
**Owner:** @pvliesdonk
**Tracking:** #1564
**Closes:** #1561 (superseded — prompt patches no longer needed once schema is symmetric)

## Problem

The graph layer treats path membership symmetrically: every `belongs_to` edge is equivalent. Pre-commit shared beats have **two** edges; commit/post-commit/gap beats have **one**. The schema layer (`InitialBeat` in `src/questfoundry/models/seed.py:240-285`) is asymmetric: `path_id: str` + `also_belongs_to: str | None`. The asymmetry was prescribed by the ontology document itself (`docs/design/story-graph-ontology.md` line 947) during the #1206/#1208 Y-shape ratification, was implemented in the schema, and propagated through `seed.md` R-3.6 / R-3.7 / a violation table row / a worked example, then through the SEED prompt templates and ~15-25 test fixtures.

The asymmetry is the direct cause of the murder6 production failure (qwen3:4b emits `path_id` set, `also_belongs_to` omitted entirely → 3 retry attempts → `SeedStageError`). The model treats `path_id` as singular ownership and never produces the sibling. Patching the prompt to coach the model around the asymmetry is the wrong layer; the schema asymmetry should not exist in the first place.

The original schema was `paths: list[str]` (length 1 or 2). It was correct — it matched the graph. #1206 traded that correctness for what looked like Y-shape rigidity ("exactly TWO and only TWO paths"). The trade is now visibly net-negative: small models can't comply, the spec contradicts itself across files, and the asymmetry buys nothing the list shape doesn't already enforce via `min_length=1, max_length=2` validators.

## Non-goals

- Changing the **graph** representation. Graph already uses dual `belongs_to` edges correctly. No change to `Edge.edge_type == "belongs_to"` semantics, mutation count, or query patterns.
- Touching POLISH or GROW behavior. R-1.4 Y-fork wiring (#1562) is a separate code-layer issue and not addressed here.
- Re-litigating cross-dilemma multi-`belongs_to`. Same-dilemma dual is permitted; cross-dilemma stays forbidden. The validator that enforces this stays — it just reads a list now instead of two named fields.
- Adding new beat sub-types or rules.

## Design

### Ontology change (the load-bearing decision)

`docs/design/story-graph-ontology.md` § "InitialBeat.paths — Same-Dilemma Dual belongs_to" (lines ~941-947). The "Impact" paragraph currently prescribes `also_belongs_to: str | null`. Replace it with a paragraph that prescribes the symmetric list shape:

> **Impact:** `InitialBeat.belongs_to` is `list[str]` with `min_length=1, max_length=2`. The mutation layer creates one `belongs_to` graph edge per element. Pre-commit beats supply two path IDs (one per path of their dilemma); commit, post-commit, and gap beats supply one. Cross-dilemma dual membership remains forbidden and is rejected by the mutation layer.

The section title can stay as-is (`InitialBeat.paths`). The "Current" paragraph that mentions `paths: list[str]` was correct in the original document — it's the "Impact" recommendation that misled #1206.

### Procedure spec changes

`docs/design/procedures/seed.md`:

- **R-3.6** (line 175). Currently: *"Pre-commit beats have exactly two `belongs_to` edges, both referencing Paths of the same Dilemma. In YAML form, this is represented by `path_id` (primary) and `also_belongs_to` (other path); in the graph, it is two distinct `belongs_to` edges."* → *"Pre-commit beats have exactly two `belongs_to` edges, both referencing Paths of the same Dilemma. In YAML form, `belongs_to` is a list of length 2 containing both path IDs; in the graph, two distinct `belongs_to` edges are created."*
- **R-3.7** (line 177). Currently: *"Commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains an entry with `effect: commits` naming which path locks in. In YAML, `also_belongs_to` is absent or null on commit beats."* → *"Commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains an entry with `effect: commits` naming which path locks in. In YAML, `belongs_to` contains exactly one path ID on commit beats."*
- **Violation table** (line 202). Replace the `Commit beat has also_belongs_to set` row with `Commit beat has more than one belongs_to entry` (same R-3.7 reference, same explanation logic with the new field name).
- **Worked example** (line 673). Replace `also_belongs_to: path::mentor_trust__manipulator` with the corresponding `belongs_to:` list (containing both the example's path_id and the original sibling).

### Schema changes (covered in implementation PR, not this spec PR)

`src/questfoundry/models/seed.py` `InitialBeat`:

- Drop `path_id: str` and `also_belongs_to: str | None`.
- Add `belongs_to: list[str] = Field(min_length=1, max_length=2, description=...)`.
- Drop `_migrate_paths_to_path_id` (the migration that originally went `paths → path_id + also_belongs_to`). The new schema effectively restores the original `paths` field with a renamed surface (`belongs_to`, matching the graph edge type) and explicit list-length validation.
- Replace `_also_belongs_to_differs_from_path_id` validator with one that asserts list elements are unique.
- Same-dilemma sibling check stays in the mutation layer where path→dilemma resolution is available.

### Code-side ripples (covered in implementation PR)

- `src/questfoundry/graph/mutations.py` — read `beat["belongs_to"]` (list) and create one `belongs_to` edge per element. Loop replaces the current pair-of-fields read.
- `prompts/templates/serialize_seed_sections.yaml` — `shared_beats_prompt` and `per_path_beats_prompt`: rewrite schema descriptions and GOOD/BAD examples for `belongs_to: list[str]`.
- `prompts/templates/summarize_seed_sections.yaml` — same pattern.
- All test fixtures using `path_id=` / `also_belongs_to=` on `InitialBeat` or its dict equivalent.

### Verification

```sh
$ rg "also_belongs_to" src/ tests/ docs/ prompts/
# zero hits

$ rg "\.path_id\b" src/questfoundry/models/seed.py
# zero hits in InitialBeat (Path.path_id stays — it's the Path node's own ID, not a beat's reference to it)
```

End-to-end smoke: re-run `projects/murder6` against qwen3:4b and confirm SEED `shared_beats` succeeds without retry. The list shape is unambiguous to small models — the murder6 failure mode (omitted-entirely field) is structurally impossible against `min_length=1, max_length=2`.

## PR shape

**Two PRs, in sequence, per CLAUDE.md "Spec-first fix order":**

1. **Spec PR (this design lives here, plus the doc-only edits):** `docs/design/story-graph-ontology.md` § "InitialBeat.paths" rewrite + `docs/design/procedures/seed.md` R-3.6 / R-3.7 / violation table / worked example. ~30-50 lines of doc edits. Lands first; documents the new contract before any code changes.

2. **Implementation PR (depends on spec PR being merged):** schema + mutations + prompts + tests. Atomic per CLAUDE.md "Replace directly, no compat shim." ~15-25 sites. The PR is bigger but cohesive — every site is a mechanical rename to the new field shape.

The spec PR is small enough that a single review round should converge it. The implementation PR will be the larger review surface; splitting reduces both blast radius and review fatigue.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Mid-merge state where schema is renamed but prompts still ask for old fields → broken pipeline. | Atomic implementation PR (single commit-set; replace prompts and schema and tests together). Don't split between schema and prompts. |
| The `_migrate_paths_to_path_id` was preserving backwards compatibility with on-disk graph data using the original `paths` field. Removing the migrator means snapshots predating #1206 won't load. | Project policy is "Replace directly, no compat shim." Snapshots predating #1206 are pre-Y-shape and would not load against the current schema *anyway* (the graph mutation layer changed). No new compatibility loss. |
| Cross-dilemma sibling check moves entirely to mutation layer; some tests may have been relying on the Pydantic-side validator. | The Pydantic validator could only check that the two fields differed, not that they shared a dilemma — the dilemma check was already at the mutation layer. No coverage regression. |
| Test fixtures that construct `InitialBeat` literals are pervasive. | Atomic rewrite per CLAUDE.md. The implementation PR's plan documents the inventory. |
| #1206 had design intent the rename now reverts. | Issue #1564 explicitly addresses this: the rationale was Y-shape rigidity expressed via uniform field naming. The rigidity is preserved by `min_length=1, max_length=2`. The uniform field naming claim is undermined by the very failure (#1561 / murder6) the rename eliminates. |

## Out of scope

- #1562 (GROW R-1.4 wiring code bug). The rename may make that bug easier to reason about by uniformizing the schema, but the actual fix is in `graph/mutations.py` or GROW Phase 1 edge construction — independent of this rename.
- #1563 (`grow_phase3_intersections.yaml` `entity::` prefix). Independent soft fix, unaffected by this rename.

## References

- Issue #1564 (this design's tracker; expanded scope after user pushback that the prompt-only fix in #1561 was insufficient).
- #1561 (closed as superseded — original prompt-patches design).
- #1206 / #1208 (the Y-shape ratification PRs that introduced the asymmetric form; consult their rationale during implementation; the burden is to confirm the rationale was net-negative or has expired).
- `story-graph-ontology.md` Part 8 (graph-level dual `belongs_to` rules — symmetric and unchanged).
- `seed.md` R-3.6 / R-3.7 (procedure-level rules; rewritten in this spec).
