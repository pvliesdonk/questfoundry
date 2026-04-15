# Ontology Design Questions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the B1/B2/B3 rulings from `docs/superpowers/specs/2026-04-15-ontology-design-questions-design.md` to the story graph ontology and narrative guide, and add a regression test that locks in the existing zero-`belongs_to` handling in POLISH beat grouping.

**Architecture:** Doc-mostly change. SGO (`docs/design/story-graph-ontology.md`) gets three edits in Parts 3, 4, and 8 plus a new subsection that captures the narrative principle for determining a beat's `belongs_to`. HBSW gets a single-line forward reference. One unit test pins current zero-path-beat behavior in `compute_beat_grouping`. One docstring note on `check_no_pre_commit_intersections` cites the narrowed guard rail text. The project memory file (outside the repo) gets a one-line refresh so it stops contradicting the repo.

**Tech Stack:** Markdown (SGO, HBSW); pytest (regression test); Python docstring.

---

## File Structure

Files created or modified by this plan:

| File | Role | Type |
|---|---|---|
| `docs/design/story-graph-ontology.md` | Add narrative principle subsection in Part 8; rewrite the "same soft dilemma" constraint in Part 4; add path-termination paragraph to Part 3; rephrase guard rail 3; drop "two successors / two predecessors" divergence framing | Modify |
| `docs/design/how-branching-stories-work.md` | Add a one-sentence forward reference from the setup-beat acknowledgment to the new SGO subsection | Modify |
| `src/questfoundry/graph/grow_validation.py` | Add a docstring note on `check_no_pre_commit_intersections` citing the narrowed SGO text so the rationale is visible from the code | Modify |
| `tests/unit/test_polish_deterministic.py` | Add a regression test locking in the "two zero-`belongs_to` beats don't collapse" behavior | Modify |
| `/home/peter/.claude/projects/-mnt-code-questfoundry/memory/MEMORY.md` | One-line refresh of the Y-Shape Dilemma Model's guard rail 3 so the cached project memory matches the narrowed SGO text | Modify (outside repo) |

No code change is required to `compute_beat_grouping`. The audit confirmed it already skips zero-path beats during collapse (line 291-292 of `src/questfoundry/pipeline/stages/polish/deterministic.py`) and routes them through the singleton pass. The regression test locks that in.

---

## Task 1: B1 — Post-convergence `belongs_to` is narrative, not graph-shaped

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Part 3, Part 4

Three sub-edits: two in prose sections, one in the divergence/convergence intro in Part 3. This task does NOT add the new "Determining a beat's `belongs_to`" subsection — that lives in Task 2 alongside the zero-`belongs_to` category so both the B1 and B2 rulings are captured together and the enumeration only needs to appear once.

- [ ] **Step 1: Read SGO Part 3 "Beat Lifecycle" section**

Run:
```bash
sed -n '215,247p' docs/design/story-graph-ontology.md
```
Confirm the current paragraph about divergence/convergence mentions "The first shared beat after payoff beats IS the convergence point" at line 219 and does not yet say paths terminate at their last exclusive beat.

- [ ] **Step 2: Rewrite line 219 of SGO to clarify "convergence point" is graph-shape shorthand**

Use Edit with:

```
old_string: A beat with two successors (one per path of a dilemma) represents a **divergence**: the story splits at the commit. A beat with two predecessors (from different paths) represents a **convergence**: the storylines rejoin. These structural moments are not separate node types — they are visible in the DAG's topology. Divergence happens *between* the last shared pre-commit beat (which has one successor per path) and the per-path commit beats that follow it — each commit beat is the first beat exclusive to its path. The first shared beat after payoff beats IS the convergence point.

new_string: A beat with two successors (one per path of a dilemma) represents a **divergence**: the story splits at the commit. A beat with two predecessors (from different paths) represents a **convergence**: the storylines rejoin. These structural moments are not separate node types — they are visible in the DAG's topology. Divergence happens *between* the last shared pre-commit beat (which has one successor per path) and the per-path commit beats that follow it — each commit beat is the first beat exclusive to its path. The first beat with predecessors from both post-commit chains is the **convergence point** — a graph-shape shorthand, not a narrative category. A soft dilemma's paths terminate at their last exclusive beat; the convergence beat is outside them and does not carry `belongs_to` to either path on the converged dilemma's account. See Part 8 "Determining a beat's `belongs_to`" for the governing rule.
```

- [ ] **Step 3: Read SGO Part 4 "Intersection and Convergence Policy"**

Run:
```bash
sed -n '329,334p' docs/design/story-graph-ontology.md
```
Confirm the current text contains "beats from the same soft dilemma can intersect only if they are in the shared region (before commit or after convergence)".

- [ ] **Step 4: Rewrite the Part 4 intersection constraint to use the narrative principle**

Use Edit with:

```
old_string: Intersections must respect dilemma roles. Two beats from the same hard dilemma's paths must never be grouped into an intersection — they are mutually exclusive by definition (the player is on one path or the other, never both). Beats from different dilemmas can always intersect, regardless of those dilemmas' roles. And beats from the same soft dilemma can intersect only if they are in the shared region (before commit or after convergence).

This constraint is structural, not a guideline. Violating it produces a scene that is impossible to reach — the player cannot be on both paths of a hard dilemma simultaneously.

new_string: Intersections must respect dilemma roles. Two beats from the same hard dilemma's paths must never be grouped into an intersection — they are mutually exclusive by definition (the player is on one path or the other, never both). Beats from different dilemmas can always intersect, regardless of those dilemmas' roles. For a soft dilemma, two beats can co-occur in one intersection only when both are pre-commit beats of that dilemma — and guard rail 3 in Part 8 forbids intersection-grouping two *same-dilemma* pre-commit beats even then. Post-convergence beats are not "from" the soft dilemma (see Part 8 "Determining a beat's `belongs_to`"), so the same-dilemma constraint does not apply to them.

This constraint is structural, not a guideline. Violating it produces a scene that is impossible to reach — the player cannot be on both paths of a hard dilemma simultaneously.
```

- [ ] **Step 5: Verify the changes parse and read cleanly**

Run:
```bash
grep -n "convergence point" docs/design/story-graph-ontology.md
grep -n "same soft dilemma" docs/design/story-graph-ontology.md
```
Expected: first grep finds the rewritten line in Part 3 and nothing else. Second grep finds zero matches (we removed the "same soft dilemma" framing).

- [ ] **Step 6: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): B1 ruling — post-convergence beats are outside the converged dilemma's paths"
```

---

## Task 2: B2 — Add "Determining a beat's `belongs_to`" subsection with zero-path category

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Part 8 (just before "Guard rails")
- Modify: `docs/design/how-branching-stories-work.md` line 24

This task introduces the new subsection that lists the three beat categories (B1 + B2 rulings combined) and adds the singleton-passage grouping invariant. It also updates HBSW so the setup-beat acknowledgment links to the new subsection.

- [ ] **Step 1: Read SGO Part 8 around the "Same-dilemma pre-commit multi-`belongs_to` is permitted" paragraph**

Run:
```bash
sed -n '567,580p' docs/design/story-graph-ontology.md
```
Confirm the current text transitions from "Path Membership ≠ Scene Participation" into the "Guard rails:" numbered list at line 575.

- [ ] **Step 2: Insert the new "Determining a beat's `belongs_to`" subsection before "Guard rails"**

Use Edit with:

```
old_string: **Same-dilemma pre-commit multi-`belongs_to` is permitted.** A pre-commit beat — one that occurs before the dilemma's commit point — belongs to both paths of its own dilemma. This is structurally correct: every player experiences pre-commit beats regardless of which path they will later choose. Pre-commit beats have two `belongs_to` edges (one to each path in the dilemma); post-commit beats have exactly one. This is not the same as cross-dilemma multi-assignment, which remains forbidden.

Guard rails:

new_string: **Same-dilemma pre-commit multi-`belongs_to` is permitted.** A pre-commit beat — one that occurs before the dilemma's commit point — belongs to both paths of its own dilemma. This is structurally correct: every player experiences pre-commit beats regardless of which path they will later choose. Pre-commit beats have two `belongs_to` edges (one to each path in the dilemma); post-commit beats have exactly one. This is not the same as cross-dilemma multi-assignment, which remains forbidden.

#### Determining a beat's `belongs_to`

A beat's `belongs_to` edges are a **narrative** statement: "this beat furthers the narrative of this dilemma." They are not a graph-shape statement about which path chains reach the beat. Three beat categories cover every legal beat:

1. **Shared pre-commit** — the beat sets up the dilemma's tension for both possible answers. Two `belongs_to` edges, one to each explored path of the dilemma.
2. **Commit and exclusive post-commit** — the beat locks in or plays out one answer's consequences. One `belongs_to` edge to the answer's path.
3. **Setup, transition, and epilogue** — the beat furthers no dilemma's narrative. Zero `belongs_to` edges. Examples: a world-setup opening beat before any dilemma is introduced; a post-convergence connector beat that precedes the next dilemma's pre-commit chain; a closing epilogue after the final dilemma has committed.

Cross-dilemma co-occurrence (a scene that serves two dilemmas at once) is **not** represented as a beat belonging to two dilemmas. It is represented as two distinct beats (one per dilemma) linked by an `intersection_group`. This preserves guard rail 1 below (no cross-dilemma dual `belongs_to`).

**Grouping invariant:** a beat with zero `belongs_to` is a singleton passage during POLISH. It cannot collapse with any path-specific chain because the collapse rule requires exact path-set equality, and the empty set equals only itself, not any single-path set.

Guard rails:
```

- [ ] **Step 3: Verify the subsection inserts in the right place and all three guard rails are still below it**

Run:
```bash
sed -n '567,600p' docs/design/story-graph-ontology.md
```
Expected: you see the "Same-dilemma pre-commit multi-`belongs_to` is permitted" paragraph, then the new `#### Determining a beat's belongs_to` subsection, then "Guard rails:" and the numbered list.

- [ ] **Step 4: Read HBSW line 24**

Run:
```bash
sed -n '24p' docs/design/how-branching-stories-work.md
```
Confirm the text mentions "setup beats that establish context without directly touching the dilemma."

- [ ] **Step 5: Add a forward reference from HBSW line 24 to the new SGO subsection**

Use Edit with:

```
old_string: These describe what a beat does to the dilemma, not its full narrative purpose. When SEED scaffolds a path, it assembles beats into a complete arc — typically introducing the dilemma, developing it through advances and complications, surfacing a reveal, reaching a commit, and playing out the consequences. But this is a common shape, not a formula. A path might need multiple reveals, a try-fail cycle before the commit, emotional reaction beats that process what just happened, or setup beats that establish context without directly touching the dilemma. The scaffold should serve the story, not a template.

new_string: These describe what a beat does to the dilemma, not its full narrative purpose. When SEED scaffolds a path, it assembles beats into a complete arc — typically introducing the dilemma, developing it through advances and complications, surfacing a reveal, reaching a commit, and playing out the consequences. But this is a common shape, not a formula. A path might need multiple reveals, a try-fail cycle before the commit, emotional reaction beats that process what just happened, or setup beats that establish context without directly touching the dilemma. The scaffold should serve the story, not a template. Setup beats that serve no dilemma carry zero `belongs_to` edges in the graph; the ontology (`story-graph-ontology.md` Part 8, "Determining a beat's `belongs_to`") defines the three beat categories — shared pre-commit (dual), commit and exclusive post-commit (single), and setup / transition / epilogue (zero).
```

- [ ] **Step 6: Commit**

```bash
git add docs/design/story-graph-ontology.md docs/design/how-branching-stories-work.md
git commit -m "docs(ontology): B2 ruling — zero belongs_to is legal; add narrative principle subsection"
```

---

## Task 3: B3 — Narrow rephrase of guard rail 3

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Part 8 guard rails list

- [ ] **Step 1: Read the current guard rail 3 text**

Run:
```bash
sed -n '577,580p' docs/design/story-graph-ontology.md
```
Confirm the current text reads "**Intersection exclusion.** An intersection group must not contain two pre-commit beats from the same dilemma — they already co-occur by definition; declaring them as an intersection is redundant and creates false structural implications."

- [ ] **Step 2: Replace the guard rail 3 text with the narrow rephrase**

Use Edit with:

```
old_string: 3. **Intersection exclusion.** An intersection group must not contain two pre-commit beats from the same dilemma — they already co-occur by definition; declaring them as an intersection is redundant and creates false structural implications.

new_string: 3. **Same-dilemma pre-commit exclusion.** An intersection group must not contain two pre-commit beats of the same dilemma (identified by identical dual `belongs_to` path sets). Such beats are already sequentially ordered in the dilemma's pre-commit chain; grouping them into an intersection implies simultaneity, contradicting the chain ordering. Cross-dilemma pre-commit co-occurrence IS the intended use of intersection groups and remains allowed.
```

- [ ] **Step 3: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): B3 ruling — narrow rephrase of guard rail 3"
```

---

## Task 4: Regression test — two zero-`belongs_to` beats do not collapse

**Files:**
- Modify: `tests/unit/test_polish_deterministic.py`

The code at `src/questfoundry/pipeline/stages/polish/deterministic.py:290-298` already skips zero-path beats during the collapse pass (the `if not path_set: continue` guard). This test locks that behavior in so a future refactor cannot silently regress it.

- [ ] **Step 1: Check whether similar regression tests already exist**

Run:
```bash
grep -n "frozenset()\|zero.*belongs_to\|no belongs_to\|singleton" tests/unit/test_polish_deterministic.py | head
```
If a test already covers this case, add `# also locks in B2 ruling: zero-belongs_to beats cannot collapse` as a docstring note and skip steps 2-4. Otherwise proceed.

- [ ] **Step 2: Write the failing test — two adjacent zero-path beats produce two singleton passages**

Append to `tests/unit/test_polish_deterministic.py`. (Read the top of the file first to match the existing test fixture / import patterns. Look at an existing `compute_beat_grouping` test for the Graph-construction idiom; mirror it.)

Structure of the test (the exact idiom will depend on what's already in the file):

```python
def test_zero_belongs_to_beats_do_not_collapse() -> None:
    """Two zero-belongs_to setup beats in sequence become two singleton passages.

    Per ontology Part 8 "Determining a beat's `belongs_to`", a beat with zero
    `belongs_to` edges cannot collapse with any path-specific chain, and the
    empty path set does not match any non-empty set. Two adjacent zero-path
    beats must therefore each become their own singleton passage, not a
    two-beat collapse chain.

    This locks in the B2 ruling from issue #1237 and prevents a silent
    regression if the collapse rule is ever refactored.
    """
    graph = Graph.empty()
    # Two setup beats — no belongs_to edges, linked by predecessor edge
    graph.create_node("beat::setup_1", {"type": "beat", "summary": "A"})
    graph.create_node("beat::setup_2", {"type": "beat", "summary": "B"})
    graph.add_edge("predecessor", "beat::setup_2", "beat::setup_1")

    specs = compute_beat_grouping(graph)

    beat_to_spec = {bid: s for s in specs for bid in s.beat_ids}
    assert beat_to_spec["beat::setup_1"].passage_id != beat_to_spec["beat::setup_2"].passage_id
    assert beat_to_spec["beat::setup_1"].grouping_type == "singleton"
    assert beat_to_spec["beat::setup_2"].grouping_type == "singleton"
```

Match the imports, fixture style, and Graph construction idioms already in the file — the snippet above shows the shape, not necessarily the exact code that will compile. If existing tests use a `_build_graph` helper or a pytest fixture, use that; don't invent a new pattern.

- [ ] **Step 3: Run the new test alone to verify it passes**

Run:
```bash
uv run pytest tests/unit/test_polish_deterministic.py::test_zero_belongs_to_beats_do_not_collapse -v
```
Expected: PASS. (Not a failing TDD test — the code already handles this. The test's job is to lock in current behavior as a regression guard, per the spec's audit finding.)

- [ ] **Step 4: Run the full polish deterministic test file to confirm no interaction with other tests**

Run:
```bash
uv run pytest tests/unit/test_polish_deterministic.py -x -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_polish_deterministic.py
git commit -m "test(polish): lock in zero-belongs_to singleton-passage behavior (B2 regression guard)"
```

---

## Task 5: Docstring note on `check_no_pre_commit_intersections`

**Files:**
- Modify: `src/questfoundry/graph/grow_validation.py`

Link the code rule back to the narrowed SGO text so future readers of the validation file see the rationale without digging through the ontology.

- [ ] **Step 1: Read the current docstring of `check_no_pre_commit_intersections`**

Run:
```bash
sed -n '/^def check_no_pre_commit_intersections/,/^    """/p' src/questfoundry/graph/grow_validation.py
```
(If that range doesn't fully capture the docstring — it usually will — widen with `sed -n '/check_no_pre_commit_intersections/,+40p'`.)

- [ ] **Step 2: Add a note referencing the narrowed SGO text**

Use Edit to append a reference paragraph to the existing docstring. The exact current docstring must be read first so the new_string preserves it verbatim. The addition to append near the end of the docstring:

```
Rationale: two pre-commit beats of the same dilemma have identical dual
``belongs_to`` path sets and are sequentially ordered in the dilemma's
pre-commit chain. An intersection group implies simultaneity, contradicting
that chain ordering. Cross-dilemma pre-commit co-occurrence is not affected.
See `docs/design/story-graph-ontology.md` Part 8 guard rail 3 for the ruling.
```

Pattern for the Edit (adjust `<existing docstring tail>` to match what's actually there — read first, then do the exact-string Edit):

```
old_string: <existing docstring tail, e.g. the last sentence of the current docstring>
    """

new_string: <existing docstring tail>

    Rationale: two pre-commit beats of the same dilemma have identical dual
    ``belongs_to`` path sets and are sequentially ordered in the dilemma's
    pre-commit chain. An intersection group implies simultaneity, contradicting
    that chain ordering. Cross-dilemma pre-commit co-occurrence is not affected.
    See ``docs/design/story-graph-ontology.md`` Part 8 guard rail 3 for the ruling.
    """
```

- [ ] **Step 3: Verify the file still parses and tests still pass**

Run:
```bash
uv run ruff check src/questfoundry/graph/grow_validation.py
uv run pytest tests/unit/test_grow_validation.py -x -q
```
Expected: ruff clean; all grow_validation tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/questfoundry/graph/grow_validation.py
git commit -m "docs(validation): cite narrowed guard rail 3 rationale on check_no_pre_commit_intersections"
```

---

## Task 6: Update project memory to match narrowed guard rail 3

**Files:**
- Modify: `/home/peter/.claude/projects/-mnt-code-questfoundry/memory/MEMORY.md` (outside the repo — user's local claude memory)

This is not a git commit. Memory files are stored outside the repo and read by future claude sessions. If the memory still contains the old guard rail 3 phrasing, future sessions will reason from stale information.

- [ ] **Step 1: Check whether the memory file references guard rail 3**

Run:
```bash
grep -n "guard rail 3\|intersection_group\|Cross-dilemma co-occurrence" /home/peter/.claude/projects/-mnt-code-questfoundry/memory/MEMORY.md
```
If guard rail 3 is referenced, proceed. If not, skip this task entirely.

- [ ] **Step 2: Update the guard rail 3 line**

Read the exact current wording in MEMORY.md. Replace the guard rail 3 bullet or sentence with:

```
3. Cross-dilemma co-occurrence → `intersection_group`, not `belongs_to`. Two pre-commit beats of the SAME dilemma cannot share an intersection (sequential chain ordering; see SGO Part 8).
```

Use the Edit tool with the exact `old_string` / `new_string` — don't blindly replace.

- [ ] **Step 3: No git commit**

MEMORY.md is outside the repo. No `git add`. Move on to Task 7.

---

## Task 7: Cross-check the e2e fixture and push the branch

**Files:**
- Read-only: `tests/integration/test_y_shape_end_to_end.py`
- No new modifications expected (cross-check only)

- [ ] **Step 1: Re-read the Y-shape e2e fixture against the new three-category taxonomy**

Open `tests/integration/test_y_shape_end_to_end.py` and for each beat in the `_make_seed_output()` fixture (`shared_setup`, `commit_protector`, `post_protector`, `commit_manipulator`, `post_manipulator`), confirm which category it falls into per the new SGO Part 8 subsection:

| Beat | Category | Expected `belongs_to` |
|---|---|---|
| `shared_setup` | Shared pre-commit | Dual |
| `commit_protector` | Commit | Single |
| `post_protector` | Exclusive post-commit | Single |
| `commit_manipulator` | Commit | Single |
| `post_manipulator` | Exclusive post-commit | Single |

If every fixture beat matches its expected category, no change is needed. If a beat violates the taxonomy, stop and report BLOCKED (we would have a real inconsistency between the e2e test and the ruling; reconciling that is out of scope for this plan).

- [ ] **Step 2: Run the touched test files plus the e2e test**

Run:
```bash
uv run pytest tests/unit/test_polish_deterministic.py tests/unit/test_grow_validation.py tests/integration/test_y_shape_end_to_end.py -x -q
```
Expected: all tests pass.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin docs/ontology-design-questions
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create \
  --title "docs(ontology): apply Group B rulings from #1237 (B1/B2/B3)" \
  --body "$(cat <<'EOF'
## Summary

Applies the three design rulings from `docs/superpowers/specs/2026-04-15-ontology-design-questions-design.md` (Group B of #1237):

- **B1** — A beat's \`belongs_to\` is determined by the dilemma whose narrative it furthers, not by the graph shape. Post-convergence beats terminate the converged dilemma's paths.
- **B2** — Zero \`belongs_to\` is a legal state for setup / transition / epilogue beats. POLISH treats them as singleton passages (verified — the code already handles this; added a regression test).
- **B3** — Narrow rephrase of Part 8 guard rail 3 so it targets exactly the same-dilemma pre-commit chain collision case (\`shared_setup_01\` ∩ \`shared_setup_02\`) without overreaching.

## Changes

- \`docs/design/story-graph-ontology.md\` — Part 3 convergence-point clarification; Part 4 same-soft-dilemma rewrite; new Part 8 subsection "Determining a beat's \`belongs_to\`" with the three-category taxonomy; guard rail 3 narrow rephrase.
- \`docs/design/how-branching-stories-work.md\` — forward reference from the setup-beat acknowledgment (line 24) to the new SGO subsection.
- \`src/questfoundry/graph/grow_validation.py\` — docstring note on \`check_no_pre_commit_intersections\` linking to the narrowed SGO text.
- \`tests/unit/test_polish_deterministic.py\` — regression test for "two zero-\`belongs_to\` beats → two singleton passages" (B2 ruling, locking in existing code behavior).

## Out of scope

- Group A mechanical cleanup (#1237 Group A): entity-overlay embedded-vs-node, character arc metadata, stale Creates/Modifies tables. Separate PR.
- Group C clarifications: \`concurrent\` symmetry, \`is_canonical\` operational-privilege note. Folded into Group A or its own PR later.
- Code changes for the GROW convergence follow-ups #1221 / #1222. They consume the B1 ruling; this PR unblocks them.

## Verification

- All referenced tests pass locally.
- Every beat in \`tests/integration/test_y_shape_end_to_end.py::_make_seed_output\` maps to exactly one of the three new categories.

## Closes

Part of #1237 (Group B).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Mark this task done and report the PR URL**

Print the PR URL and confirm the plan is complete.

---

## Self-Review Notes

- **Spec coverage** — Every section of the spec (B1 doc surgery; B2 doc surgery + code audit; B3 doc rephrase + docstring note; MEMORY.md refresh; verification) is mapped to a task above. The one spec item deliberately NOT included as its own task is "code change for the compute_beat_grouping singleton rule" — the audit in Task 4 confirmed it isn't needed; a regression test replaces it.
- **Placeholder scan** — No "TBD" / "TODO" / "implement later" in the plan. Task 5 tells the implementer to read the current docstring before editing (the exact text can't be pre-drafted without knowing the current wording), but provides the full new_string content and the insertion pattern. Task 2 Step 2's Edit has the full new_string in place.
- **Type consistency** — The three beat categories (shared pre-commit / commit + exclusive post-commit / setup-transition-epilogue) and their `belongs_to` counts (dual / single / zero) are consistent across every task that references them: Task 2 Step 2, Task 2 Step 5, Task 7 Step 1.
