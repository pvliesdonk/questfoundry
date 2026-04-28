# prompt-engineer subagent smoke test (2026-04-25)

**Target:** `prompts/templates/serialize_seed_sections.yaml` (per-path-beats / shared-beats sections)
**Failure under test:** murder1 SEED halt — repair feedback didn't echo expected `also_belongs_to` value.

## PASS criteria

1. Identified the murder1 repair-gap with severity `hard`.
2. Proposed concrete repair-message wording that echoes the sibling path id template per beat.
3. Cited CLAUDE.md §9 or §10 AND the Y-shape Part 8 guard rail.

**Result: PASS on all three.** The subagent also surfaced 5 bonus findings (sandwich-pattern issues, schema-skew with `[bracket]` interpolation, terminology overlap between `effect: "commits"` and "commit beat") that will fold into the SEED stage audit (Task 8).

## Subagent output

### `prompts/templates/serialize_seed_sections.yaml`

**Verdict:** mixed

**Findings:**

---

- **[hard] [repair-gap]** — Validator names missing `also_belongs_to` beats but does NOT echo the expected sibling path ID value; small model retries fail 3/3
  - Where: `src/questfoundry/models/seed.py` lines 758–769 (`_require_also_belongs_to`), which feeds into `serialize.py` `_build_error_feedback` (lines 478–493). The repair feedback delivered to the model reads: *"Beats missing `also_belongs_to`: `shared_setup_dilemma_garden_key_hides_or_opens_01`, `shared_setup_dilemma_garden_key_hides_or_opens_02`. Please fix these issues and try again."* It does **not** tell the model *what value* to set.
  - Spec citation: `story-graph-ontology.md §Part 8` guard rail 2 (pre-commit only); `seed.md §R-3.6`; `CLAUDE.md §9` (never let repair feedback send generic "Missing items" — describe the specific problem and the fix); `CLAUDE.md §10` (small-model repair-loop blindness: the model does not re-read the system prompt on retry — only the new user-message is fresh context).
  - Root cause: The system prompt in `shared_beats_prompt` tells the model what `also_belongs_to` must be (`{also_belongs_to}`) near the top of the call, but the repair message delivered 3000+ tokens later (on retry) only names the field, not the value. qwen3:4b has lost the constraint-to-value mapping by that point. Every retry sees the same incomplete feedback and produces the same mistake.
  - Recommended fix — replace the body of `_require_also_belongs_to` in `src/questfoundry/models/seed.py` with a version that carries the sibling path id per-beat. Because the validator runs after construction it does not yet know the expected sibling id from the call context, so the fix must be in one of two places (pick the nearest):

    **Option A (preferred) — enrich the per-beat error at the Pydantic layer.** The validator already knows each beat's `path_id`; it should derive the expected sibling by consulting the beat's own `path_id` structural pattern and echo it. Since the validator cannot know the dilemma's two explored path IDs without extra context, pass them in via a class-var on a per-call subclass or via a closure model. The new error string for each offending beat should be:

    ```
    Beat `{beat_id}` is missing `also_belongs_to`. Set it to the sibling path of this
    dilemma. Your two explored paths for dilemma `{dilemma_id}` are:
      - primary (path_id):      `{primary_path_id}`
      - sibling (also_belongs_to): `{sibling_path_id}`
    Every shared pre-commit beat MUST have `also_belongs_to: "{sibling_path_id}"`.
    ```

    **Option B (simpler short-term) — enrich the repair feedback in the serializer.** In `_serialize_shared_beats_for_dilemma` (`serialize.py`), after `serialize_to_artifact` raises, intercept the `ValidationError`, find the `also_belongs_to`-missing beats, and re-raise with an augmented message that injects `primary_path_id` and `sibling_path_id` (which are already in scope at lines 1194–1195). The new user-message sent to the model on retry must say, verbatim per beat:

    ```
    Beat `shared_setup_dilemma_garden_key_hides_or_opens_01` is missing `also_belongs_to`.
    You MUST set it to: `path::garden_key_hides_or_opens__opens`
    (The two explored paths are `path::garden_key_hides_or_opens__hides` [path_id]
    and `path::garden_key_hides_or_opens__opens` [also_belongs_to].
    Every shared beat needs BOTH.)

    Beat `shared_setup_dilemma_garden_key_hides_or_opens_02` is missing `also_belongs_to`.
    You MUST set it to: `path::garden_key_hides_or_opens__opens`
    ```

    Either option satisfies the requirement; Option B can be shipped faster without touching the Pydantic model.

---

- **[soft] [sm-fragile]** — `per_path_beats_prompt`: the "DO NOT set `also_belongs_to`" rule for post-commit beats appears only in the "What NOT to Do" list and in the schema example (`"also_belongs_to": null`) but is **not** sandwiched — never repeated at the end of the prompt. Small models drop tail rules from a 60-line prompt.
  - Where: `per_path_beats_prompt`, section "What NOT to Do". The FINAL VERIFICATION block checks `path_id` and `dilemma_impacts` but does NOT re-check `also_belongs_to: null`.
  - Spec citation: `seed.md §R-3.7`; `CLAUDE.md §10` small-model implicit instruction loss.
  - Recommended fix: add to the FINAL VERIFICATION block:
    ```
    3. `also_belongs_to` is null for EVERY beat (these are post-commit beats,
       single-membership ONLY — never set also_belongs_to here)
    ```

---

- **[soft] [sm-fragile]** — `shared_beats_prompt`: the `also_belongs_to` constraint bullet (line 659) is the 7th bullet in Generation Requirements. Small models read early bullets more reliably than mid-list bullets.
  - Where: `shared_beats_prompt`, "Generation Requirements" section, line 659.
  - Spec citation: `CLAUDE.md §10` small-model failure modes; `seed.md §R-3.6`.
  - Recommended fix: hoist the `also_belongs_to` constraint to be bullet #1 (before `effect` restrictions) so it is the first rule encountered:
    ```
    1. MUST have `also_belongs_to` = `{also_belongs_to}` — THIS IS REQUIRED, not optional.
       Without it the beat is rejected by the Y-shape guard rail.
    ```

---

- **[soft] [schema-skew]** — `per_path_beats_prompt` schema example uses square-bracket placeholder notation (`"dilemma::[other_dilemma_id]"`) which violates `CLAUDE.md §9` ("Never interpolate Python objects… No `[…]`") and could leak as literal text if the model copies the template literally.
  - Where: `per_path_beats_prompt`, schema example temporal_hint block.
  - Spec citation: `CLAUDE.md §9` (no square-bracket interpolation); `seed.md §R-3.13`.
  - Recommended fix: replace with a format comment that makes clear it is a placeholder:
    ```json
    "temporal_hint": {
      "relative_to": "dilemma::another_dilemma_id",
      "position": "before_commit"
    }
    ```
    Add the explicit note: *"Replace `dilemma::another_dilemma_id` with an ACTUAL dilemma ID from your manifest — NOT a template placeholder."*

---

- **[soft] [schema-skew]** — `shared_beats_prompt` schema example uses `{{double-brace}}` escaping for the JSON example while the surrounding prompt text uses `{single-brace}` for template variables. Correct now under `.format()`; could break if the template engine ever changes.
  - Where: `shared_beats_prompt` schema block.
  - Spec citation: `CLAUDE.md §9` (formatting must be verified in `llm_calls.jsonl`).
  - Recommended fix: add a comment in the YAML above the block: `# Double-braces below are .format() escapes — rendered as single braces in the prompt`.

---

- **[info] [terminology]** — Word "commit" used both as field value (`"effect: commits"`) and as beat-category concept ("commit beat") in same sentences. For qwen3:4b, distinguishing field value from concept can cause confusion. Not a hard finding; flagging for completeness.
  - Where: `per_path_beats_prompt`, throughout.
  - Recommended fix: none required this round.

---

## Implementation decision

For Task 3, **Option B** is the right choice: enrich the repair feedback at the serialize.py layer where the path ids are in scope, rather than restructuring the Pydantic validator. Faster to ship, doesn't touch the model, and the regression test pins the behaviour at the boundary the model actually sees.
