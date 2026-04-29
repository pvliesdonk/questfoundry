# `belongs_to: list[str]` rename — Spec PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spec-PR (1 of 2) for #1564. Rewrite the ontology + procedure-spec sites that prescribe the asymmetric `path_id` + `also_belongs_to` schema, replacing them with the symmetric `belongs_to: list[str]` (length 1 or 2) shape that matches the graph.

**Architecture:** Doc-only PR. No code changes, no test changes, no schema changes. Per CLAUDE.md "Spec-first fix order" — the spec must land before the implementation PR (#1564 part 2). Two files, five edits.

**Tech Stack:** Markdown. Project rule: spec docs are authoritative; conformance with `story-graph-ontology.md` and `procedures/seed.md` drives the downstream model + code + prompts.

**Spec:** `docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md`. Closes (partially): #1564.

---

## File Structure

**Modified files (2):**

- `docs/design/story-graph-ontology.md` — § "InitialBeat.paths — Same-Dilemma Dual belongs_to" (lines ~941-947). Rewrite the "Impact" paragraph to prescribe `belongs_to: list[str]` instead of the asymmetric pair.
- `docs/design/procedures/seed.md` — four edits: R-3.6 (line ~175), R-3.7 (line ~177), violation table row (line ~202), worked example (line ~673).

No new files. No test files (this PR is doc-only). The implementation PR (separate, follows this one) will rewrite the schema, mutations, prompts, and tests.

---

## Task 1: Rewrite ontology "Impact" paragraph

**Files:**
- Modify: `docs/design/story-graph-ontology.md` line ~947 (the "Impact" paragraph inside § "InitialBeat.paths — Same-Dilemma Dual belongs_to")

- [ ] **Step 1: Read the current paragraph to confirm exact line numbers**

```bash
grep -n "also_belongs_to: str | null" docs/design/story-graph-ontology.md
```

Expected: one hit, around line 947. The full paragraph is:

```
**Impact:** `InitialBeat` needs a mechanism to signal which beats are pre-commit (dual `belongs_to`) vs post-commit (singular `belongs_to`). The recommended approach is an `also_belongs_to: str | null` field — null for post-commit beats, the other path's ID for pre-commit beats. GROW and POLISH consumers that assume one `belongs_to` per beat need auditing. See #1206.
```

- [ ] **Step 2: Replace the "Impact" paragraph**

Use `Edit` to replace the entire paragraph (the line starting with `**Impact:**` and ending with `See #1206.`):

```
**Impact:** `InitialBeat.belongs_to` is `list[str]` with `min_length=1, max_length=2`. The mutation layer creates one `belongs_to` graph edge per element. Pre-commit beats supply two path IDs (one per path of their dilemma); commit, post-commit, and gap beats supply one. Cross-dilemma dual membership remains forbidden and is rejected by the mutation layer. The asymmetric `path_id` + `also_belongs_to: str | null` form previously recommended in this section (introduced by #1206) is replaced by the list shape — see #1564.
```

The "Current" paragraph immediately above (line ~943, `InitialBeat.paths` is `list[str]` with `min_length=1`) is left intact — it correctly describes the original schema, which the rename effectively restores with explicit length bounds.

- [ ] **Step 3: Verify no other ontology mention of `also_belongs_to` remains**

```bash
rg "also_belongs_to" docs/design/story-graph-ontology.md
```

Expected: zero hits.

- [ ] **Step 4: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): replace also_belongs_to recommendation with belongs_to: list[str] (#1564)"
```

---

## Task 2: Rewrite `seed.md` R-3.6 / R-3.7 / violation row / worked example

**Files:**
- Modify: `docs/design/procedures/seed.md` four sites (lines ~175, ~177, ~202, ~673)

- [ ] **Step 1: Confirm line numbers and current wording**

```bash
rg -n "also_belongs_to" docs/design/procedures/seed.md
```

Expected hits at lines 175, 177, 202, 673.

- [ ] **Step 2: Rewrite R-3.6** (line ~175)

Find:

```
R-3.6. Pre-commit beats have exactly two `belongs_to` edges, both referencing Paths of the same Dilemma. In YAML form, this is represented by `path_id` (primary) and `also_belongs_to` (other path); in the graph, it is two distinct `belongs_to` edges.
```

Replace with:

```
R-3.6. Pre-commit beats have exactly two `belongs_to` edges, both referencing Paths of the same Dilemma. In YAML form, `belongs_to` is a list of length 2 containing both path IDs; in the graph, two distinct `belongs_to` edges are created.
```

- [ ] **Step 3: Rewrite R-3.7** (line ~177)

Find:

```
R-3.7. Commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains an entry with `effect: commits` naming which path locks in. In YAML, `also_belongs_to` is absent or null on commit beats.
```

Replace with:

```
R-3.7. Commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains an entry with `effect: commits` naming which path locks in. In YAML, `belongs_to` contains exactly one path ID on commit beats.
```

- [ ] **Step 4: Rewrite the violation table row** (line ~202)

Find:

```
| Commit beat has `also_belongs_to` set | Commit beats are exclusive to one path; `also_belongs_to` must be null | R-3.7 |
```

Replace with:

```
| Commit beat has more than one `belongs_to` entry | Commit beats are exclusive to one path; `belongs_to` must contain exactly one path ID | R-3.7 |
```

- [ ] **Step 5: Rewrite the worked example** (line ~673)

Find:

```yaml
- id: beat_mentor_warning
  summary: "Mentor delivers cryptic warning about the investigation"
  path_id: path::mentor_trust__protector
  also_belongs_to: path::mentor_trust__manipulator
  dilemma_impacts:
```

Replace with:

```yaml
- id: beat_mentor_warning
  summary: "Mentor delivers cryptic warning about the investigation"
  belongs_to:
    - path::mentor_trust__protector
    - path::mentor_trust__manipulator
  dilemma_impacts:
```

- [ ] **Step 6: Verify no `also_belongs_to` survivors and no `path_id:` followed by `also_belongs_to:` in seed.md**

```bash
rg "also_belongs_to" docs/design/procedures/seed.md
```

Expected: zero hits.

Also re-scan the worked-example block for any other shared-beat fixture using the old fields:

```bash
rg -nA1 "path_id:" docs/design/procedures/seed.md | grep -B1 "also_belongs_to:" || echo "OK: no surviving asymmetric fixtures"
```

Expected: `OK: no surviving asymmetric fixtures`.

If other shared-beat fixtures appear in the file using the old form (the grep above will surface them), rewrite each in the same `belongs_to: [...]` shape.

- [ ] **Step 7: Commit**

```bash
git add docs/design/procedures/seed.md
git commit -m "docs(seed): rewrite R-3.6/R-3.7 + violation table + worked example for belongs_to: list[str] (#1564)"
```

---

## Task 3: Final verification + draft PR

**Files:**
- N/A (verification + git only)

- [ ] **Step 1: Verify zero `also_belongs_to` survivors in spec docs**

```bash
rg "also_belongs_to" docs/design/
```

Expected: zero hits. (`docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md` and `docs/superpowers/plans/2026-04-29-belongs-to-list-rename-spec.md` may still reference it — those are the spec/plan documents themselves, intentionally describing the rename; not in `docs/design/`.)

- [ ] **Step 2: Verify the design doc and this plan still reference the correct line numbers**

If any of the seed.md edits shifted line numbers in a way that breaks references in the design doc (`docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md`), update them. The design doc's "Procedure spec changes" section cites lines ~175, ~177, ~202, ~673 — those line references should still be approximately correct after the edits (the changes are wording-only, not structural).

- [ ] **Step 3: Push branch + open draft PR**

```bash
git push -u origin refactor/1564-belongs-to-list-spec
gh pr create --draft --title "docs(spec): belongs_to: list[str] rename — ontology + seed.md (#1564 part 1/2)" --body "$(cat <<'EOF'
## Summary

Spec PR (1 of 2) for #1564. Doc-only changes restore the symmetric \`belongs_to: list[str]\` shape (length 1 or 2) that matches the graph layer, replacing the asymmetric \`path_id\` + \`also_belongs_to\` form prescribed by the ontology document during the #1206 Y-shape ratification.

The asymmetric form is the direct cause of the murder6 production failure (qwen3:4b emits \`path_id\` set, \`also_belongs_to\` omitted entirely, fails after 3 retry attempts). The list shape is structurally unambiguous: \`min_length=1, max_length=2\` is enforceable at the schema layer, and the model emits a list or doesn't.

This PR ships **doc edits only**. The follow-up implementation PR (#1564 part 2/2) covers the schema (\`models/seed.py\`), mutations (\`graph/mutations.py\`), prompts (\`serialize_seed_sections.yaml\`, \`summarize_seed_sections.yaml\`), and test fixtures.

## Spec + plan

- Design: \`docs/superpowers/specs/2026-04-29-belongs-to-list-rename-design.md\`
- This plan: \`docs/superpowers/plans/2026-04-29-belongs-to-list-rename-spec.md\`

## Cascade (5 sites in 2 files)

1. \`docs/design/story-graph-ontology.md\` line ~947 — § \"InitialBeat.paths — Same-Dilemma Dual belongs_to\" / \"Impact\" paragraph rewritten to prescribe the list shape.
2. \`docs/design/procedures/seed.md\` R-3.6 (line ~175) — drop \"path_id (primary) and also_belongs_to (other path)\" wording.
3. \`docs/design/procedures/seed.md\` R-3.7 (line ~177) — drop \"\`also_belongs_to\` is absent or null\".
4. \`docs/design/procedures/seed.md\` violation table (line ~202) — replace the \`Commit beat has also_belongs_to set\` row.
5. \`docs/design/procedures/seed.md\` worked example (line ~673) — rewrite to \`belongs_to: [...]\`.

Closes part 1 of #1564.

## Verification

- [x] \`rg \"also_belongs_to\" docs/design/\` returns zero hits.
- [x] No code changes; no test changes; CI passes trivially.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Wait for bot review loop**

This PR is doc-only; review should converge in one round. Per CLAUDE.md, gemini may be at quota; claude-review approval is sufficient. Flip ready when claude-review LGTM + all CI green + bot's review body explicitly says "Ready to merge."

---

## Self-Review Notes

Spec coverage check:

- ✅ Spec § "Ontology change" → Task 1
- ✅ Spec § "Procedure spec changes — R-3.6" → Task 2 step 2
- ✅ Spec § "Procedure spec changes — R-3.7" → Task 2 step 3
- ✅ Spec § "Procedure spec changes — Violation table" → Task 2 step 4
- ✅ Spec § "Procedure spec changes — Worked example" → Task 2 step 5
- ✅ Spec § "Verification" → Task 3 step 1

Out of scope (covered in implementation PR, not this plan):

- Schema rewrite (`models/seed.py`)
- Mutation layer changes (`graph/mutations.py`)
- Prompt template rewrites
- Test fixture updates

The plan touches **only docs** — that boundary is intentional and aligned with the spec's "PR shape" section.

No placeholders. No "implement later" / "similar to Task N" / TBD. Type consistency: every reference to the new field uses `belongs_to: list[str]` consistently across tasks.
