# Ontology Design Questions — Group B of #1237

**Status:** Approved (2026-04-15)
**Scope:** Three open design questions in `docs/design/story-graph-ontology.md` (SGO) and `docs/design/how-branching-stories-work.md` (HBSW) that were flagged in issue #1237 Group B and block implementation of the GROW convergence follow-ups (#1221, #1222).

## Context

An external audit of SGO + HBSW found three questions the docs leave genuinely open:

- **B1.** What `belongs_to` edges do post-convergence beats carry after a soft dilemma reunites its paths?
- **B2.** What `belongs_to` does a "setup beat that establishes context without touching any dilemma" (HBSW line 24) carry?
- **B3.** SGO Part 8 guard rail 3 is phrased in a way that either overreaches or does no work, depending on interpretation.

All three are doc-level rulings — no semantic change to implemented code — but each has implementation consequences we catalog below.

The brainstorm (2026-04-15 with the project owner) produced three rulings, summarized here. Related items (Group A mechanical cleanup, Group C clarifications) are **out of scope**; they get their own PRs once these rulings are stable.

---

## B1. Post-convergence `belongs_to` is narrative, not graph-shaped

### Ruling

> A beat's `belongs_to` is determined by the dilemma whose narrative it furthers, not by the graph shape. Post-convergence beats terminate the converged dilemma's paths; they themselves belong to whatever dilemma (or none) they serve.

### What this replaces

SGO currently leaves the post-convergence `belongs_to` question silent. Three interpretations were possible and none was endorsed: zero-for-this-dilemma, arbitrary-single, or dual (mirroring pre-commit). The graph-shape argument (dual, by analogy with pre-commit) was the tidiest but conflated "beat that multiple paths reach" with "beat that narratively furthers the dilemma." The ruling picks the narrative reading.

### Concrete implications

- A soft-dilemma path's beat chain terminates at its **last exclusive beat**. The convergence beat is **outside** the path — it has predecessors from both chains (graph property) but no `belongs_to` to either path (narrative property).
- The dilemma's residual influence on post-convergence narrative lives in **state flags** (already derived from consequences by GROW), **entity overlays**, and **dilemma residue weights** — not in `belongs_to` membership.
- "Convergence point" is shorthand for "first beat with predecessors from both post-commit chains." It is not a distinguished narrative category with special `belongs_to` rules.

### Doc surgery (for the Group B implementation PR)

- **SGO Part 3 "Beat Lifecycle":** add a paragraph stating that soft-dilemma paths terminate at their last exclusive beat. Clarify that "convergence point" is a graph-shape shorthand, not a narrative role.
- **SGO Part 4 / line 331** (`"beats from the same soft dilemma can intersect only if they are in the shared region (before commit or after convergence)"`): rewrite to use the narrative principle — pre-commit beats of the soft dilemma MAY intersect across dilemmas; beats post-convergence are no longer "from" the dilemma at all, so the constraint doesn't apply to them.
- **SGO line 219** (`"The first shared beat after payoff beats IS the convergence point"`): clarify "shared" means "graph-reachable from both chains", not "`belongs_to` both paths."
- **SGO Part 8:** add an explicit subsection `### Determining a beat's belongs_to` stating the narrative principle and listing the four beat categories and their `belongs_to` counts.

### Code implications

No code change required. The existing implementation (`_get_path_ids_from_beat` and the mutation layer) accepts any `path_id` / `also_belongs_to` the LLM produces; it does not enforce "post-convergence must be dual" anywhere.

The GROW convergence follow-ups **#1221** and **#1222** get an unambiguous answer: when GROW creates the predecessor edges that establish convergence topology, the post-convergence beat takes whatever `belongs_to` its own `dilemma_impacts` imply (via the narrative principle). GROW does **not** set `also_belongs_to` on convergence beats on the converged dilemma's account.

---

## B2. Zero `belongs_to` is a legal state

### Ruling

> A beat MAY have zero `belongs_to` edges when it furthers no dilemma's narrative. Setup, transition, epilogue, and pure-connector beats fall in this category.

### What this replaces

SGO implicitly assumes every beat has ≥1 `belongs_to`. The HBSW acknowledgment (line 24) of "setup beats that establish context without directly touching the dilemma" had no corresponding place in the ontology; readers could not tell whether setup beats were supposed to be smuggled into a dilemma's pre-commit (dual) or treated as first-class non-dilemma beats. The ruling legalizes the latter.

### Concrete implications

- A beat is created with `path_id` and `also_belongs_to` per the existing schema, both nullable in practice (the LLM may omit both for a setup beat that touches no dilemma).
- POLISH passage grouping: a beat with zero `belongs_to` is a **singleton passage**. It cannot collapse with any path-specific chain because path-set equality is the collapse rule, and the empty set equals only itself — not a single-path set.
- An intersection group MAY include one or more zero-`belongs_to` beats alongside beats with paths (a scene where a setup moment co-occurs with a dilemma beat). The guard rails (same-dilemma, no-commit-dual, no-pre-commit-intersection) are defined over beats that have `belongs_to`; zero-path beats are outside their scope.
- The "every beat is reachable" invariant (SGO line 250) still holds — reachability is about predecessor edges, not `belongs_to`.

### Doc surgery

- **SGO Part 8:** add the zero-`belongs_to` case to the "Determining a beat's `belongs_to`" subsection created under B1. Enumerate the three beat categories:

  1. **Shared pre-commit** — dual `belongs_to` to the dilemma's two explored paths.
  2. **Commit + exclusive post-commit** — single `belongs_to` to one path.
  3. **Shared setup / transition / epilogue** — zero `belongs_to` when the beat advances no dilemma.

  Cross-dilemma co-occurrence (a scene that serves two dilemmas at once) is **not** represented as a beat belonging to two dilemmas. It is represented as two distinct beats (one per dilemma) linked by an `intersection_group`. This preserves guard rail 1 (no cross-dilemma dual `belongs_to`).

- **SGO Part 8 grouping invariants:** add "a beat with zero `belongs_to` is a singleton passage; it cannot collapse with any path-specific chain."
- **HBSW line 24** (setup beats acknowledgment): cross-link to the new SGO subsection so the narrative mention has an ontology home.

### Code implications

Audit three places that might assume ≥1 `belongs_to` per beat:

1. `src/questfoundry/graph/algorithms.py` — `beat_to_paths: dict[str, frozenset[str]]` already handles empty frozensets (PR #1226). Verify `compute_active_flags_at_beat` does not KeyError on beats missing from `beat_to_paths`.
2. `src/questfoundry/pipeline/stages/polish/deterministic.py` — `compute_beat_grouping` collapse rule uses frozenset equality. `frozenset() == frozenset()` is True, so two zero-path beats would wrongly collapse. **Fix**: add an explicit singleton rule — a beat with `frozenset()` never collapses.
3. `src/questfoundry/graph/polish_validation.py` — `_check_divergences_have_choices` already discards `frozenset()` from `child_path_sets` (PR #1226 fix). Verify no other polish validator assumes ≥1 path.

The audit goes in the Group B implementation PR alongside the doc changes. If any real code change is needed (most likely only the `deterministic.py` collapse rule fix), it ships in the same PR; otherwise the PR is doc-only.

---

## B3. Guard rail 3 narrow rephrase

### Ruling

Replace the current SGO Part 8 guard rail 3 text with:

> *An intersection group must not contain two pre-commit beats of the same dilemma (identified by identical dual `belongs_to` path sets). Such beats are already sequentially ordered in the dilemma's pre-commit chain; grouping them into an intersection implies simultaneity, contradicting the chain ordering. Cross-dilemma pre-commit co-occurrence IS the intended use of intersection groups and remains allowed.*

### What this replaces

The current text — *"An intersection group must not contain two pre-commit beats from the same dilemma — they already co-occur by definition"* — was written to prevent `shared_setup_01` being intersection-grouped with `shared_setup_02` (same-dilemma pre-commit chain collision). The "they already co-occur by definition" framing overreaches: sequential chain beats do not co-occur; they are temporally ordered. The rephrase names the real principle (no same-dilemma pre-commit simultaneity) without the over-claim.

### Concrete implications

- Two pre-commit beats of the **same** dilemma (same dual `belongs_to` path set): illegal in one intersection group.
- Two pre-commit beats of **different** dilemmas: legal, and is the primary use case for intersection groups.
- Two beats on the same **single** path (sequential same-path chain): legal in an intersection group when they narratively co-occur (e.g., protagonist enters room → finds artifact, one scene, two beats).

### Doc surgery

- **SGO Part 8 guard rail 3:** verbatim replacement with the text above.
- **MEMORY.md** `Y-Shape Dilemma Model` section, guard rail 3 line: update to match the narrower phrasing.

### Code implications

None. PR #1219's implementation (`check_no_pre_commit_intersections`) already matches: it rejects two beats with identical `belongs_to` path sets of length ≥2. Add a docstring comment to that function in the Group B PR pointing at the narrowed SGO text so future readers understand the design rationale.

---

## Scope of the implementation PR

The Group B implementation PR is almost entirely **doc-only**:

- SGO edits in Parts 3, 4, 8, and (minor) 9
- HBSW cross-reference on line 24
- `MEMORY.md` refresh for guard rail 3
- Docstring note in `check_no_pre_commit_intersections`
- **One possible code change**: a singleton-passage rule in `compute_beat_grouping` if the zero-path case currently collapses incorrectly. If the code already handles it, the PR is doc-only.

Testing:

- No new unit tests required for the doc rulings themselves.
- If the `compute_beat_grouping` singleton rule needs adding, include a unit test covering the `frozenset()` case alongside the rule change.

## Out of scope

- **Group A mechanical cleanup** (#1237 Group A): entity overlay embedded-vs-node, character arc metadata, stale Creates/Modifies tables. Separate follow-up PR. Mechanical, no decisions needed.
- **Group C clarifications** (#1237 Group C): `concurrent` edge symmetry, `is_canonical` operational-privilege note. Each worth a paragraph; can land in Group A's PR or its own.
- **Code changes for #1221 and #1222**: the GROW convergence follow-ups. They consume the B1 ruling as a precondition but do their own work separately.
- **Revisiting guard rails 1 and 2**: they are unaffected by these rulings.

## Verification

- After the Group B PR merges, re-read SGO Part 8. The four beat categories in the new `Determining a beat's belongs_to` subsection should cover every beat a conformant SEED output can produce.
- Cross-check `tests/integration/test_y_shape_end_to_end.py` fixture against the new doc: every fixture beat should fall into one of the four categories and its `belongs_to` should match the rule.
- Eyeball `projects/*/graph.db` from the next real run (whenever test-new3 / test-new4 / etc. is run) for any beat that has `belongs_to` inconsistent with the rule. These are LLM conformance issues, not ontology bugs, but finding them this early is cheap.
