# Prompt-vs-Spec Audit Report

**Date started:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-prompt-spec-audit-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-prompt-spec-audit.md`

## How to read this report

One section per stage in pipeline order. Each section was produced by
dispatching the `prompt-engineer` subagent
(`.claude/agents/prompt-engineer.md`) scoped to that stage's prompts +
procedure doc + ontology references + Pydantic models.

Findings use the audit dimensions from the spec:

- **drift** — prompt encodes outdated terminology or rule citations
- **repair-gap** — validation feedback names missing fields without
  echoing expected values (the murder1 failure shape)
- **sm-fragile** — implicit instructions, no examples, ambiguous
  phrasing, no sandwich repetition
- **schema-skew** — prompt-vs-Pydantic mismatch
- **terminology** — deprecated names (e.g. codeword vs state_flag)

Severities: **hard** (causes pipeline halt or contract violation),
**soft** (degraded output but pipeline survives), **info** (noted, no
action).

A `spec-gap` finding means the prompt encodes a constraint not in the
spec — per CLAUDE.md docs-first, the spec is updated first.

## Overall summary

(Executive summary and totals filled in by Task 14 after all 8 stage
sections land.)

| Stage | Prompts | Hard | Soft | Info | Spec gaps | Status |
|---|---|---|---|---|---|---|
| DREAM | 3 | 7 | 5 | 3 | 0 | drift |
| BRAINSTORM | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| SEED | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| GROW | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| POLISH | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| FILL | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| DRESS | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |
| SHIP | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | pending |

---

## DREAM

### `dream.yaml` — dead-code notice

**Verdict:** mixed (file is a dead artifact; findings below apply if it is ever wired in)

**Findings:**

- **[hard] [schema-skew]** — File is not loaded by any active code path; it is dead code
  - Where: entire file
  - Spec citation: `dream.md §Phase 1: Vision Capture`; `src/questfoundry/agents/prompts.py` lines 49–102 (the three functions that build DREAM prompts load `discuss`, `summarize`, and `serialize` — never `dream`)
  - Recommended fix: Either delete `dream.yaml` (it is not used) or wire it in and migrate its intent into `discuss.yaml`. Do not leave a conflicting orphan template. File a tracking issue before merging any PR that touches DREAM prompts.

- **[hard] [schema-skew]** — `pov_style` display values do not match the Pydantic `Literal` values
  - Where: `dream.yaml` lines 27–32 (`first`, `second`, `third_limited`, `third_omniscient`)
  - Spec citation: `dream.md §R-1.9`; `src/questfoundry/models/dream.py` lines 63–70 (Literal values are `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`)
  - Recommended fix: Replace the four bullet labels with the exact enum strings the Pydantic model requires. E.g., change `**first**: Intimate…` → `**first_person**: Intimate…`.

- **[hard] [schema-skew]** — `content_notes` described as a flat string list; Pydantic model has nested `includes`/`excludes` sub-object
  - Where: `dream.yaml` line 19 (`Content notes (what to include/exclude)`) — description implies a flat list
  - Spec citation: `src/questfoundry/models/dream.py` lines 16–23 (`ContentNotes` has `excludes: list[str]` and `includes: list[str]`)
  - Recommended fix: Since this file is dead, the fix priority is low, but if re-activated, replace the flat description with: "Content notes — an object with two keys: `includes` (list of things to embrace) and `excludes` (list of things to avoid)."

- **[soft] [sm-fragile]** — `{{ mode_instructions }}` and `{{ mode_reminder }}` placeholders use Jinja2 syntax; the active loader uses LangChain `{variable}` format
  - Where: `dream.yaml` lines 8, 39
  - Spec citation: `CLAUDE.md §9`; `src/questfoundry/agents/prompts.py` lines 160–168 (`PromptTemplate.from_template` uses single-brace syntax)
  - Recommended fix: If this file is ever activated, change `{{ mode_instructions }}` → `{mode_instructions}` (and same for `mode_reminder`), or confirm a Jinja2 loader is in place.

- **[info] [schema-skew]** — `protagonist_defined` field mentioned in system text but has no output instruction telling the model what JSON key or value to produce
  - Where: `dream.yaml` line 20 (`Whether a defined protagonist exists`)
  - Spec citation: `src/questfoundry/models/dream.py` lines 75–78 (`protagonist_defined: bool`, default `False`)
  - Recommended fix: If file is activated, add: "Set `protagonist_defined: true` if a specific main character is established; omit or set `false` if open."

---

### `discuss.yaml` — DREAM-specific audit

**Verdict:** mixed

**Findings:**

- **[hard] [drift]** — Scope presets listed as `vignette / short / standard / long` but the Pydantic model and `size.py` PRESETS use `micro / short / medium / long`
  - Where: `discuss.yaml` lines 118–124 (`size_presets_section`) — four bullets name `vignette` and `standard`
  - Spec citation: `dream.md §R-1.4` ("Scope is a named preset"); `src/questfoundry/models/dream.py` line 34 (`Literal["micro", "short", "medium", "long"]`); `src/questfoundry/pipeline/size.py` lines 91–208 (PRESETS dict keys)
  - Recommended fix: Replace the four bullet entries with the correct preset names and matching passage/word ranges:
    ```
    - **micro**: 5-15 passages, 2000-5000 words. Tight single-thread, minimal branching.
    - **short**: 15-30 passages, 5000-15000 words. Focused story, limited branching.
    - **medium** (default): 30-60 passages, 15000-30000 words. Full branching narrative.
    - **long**: 60-120 passages, 30000-60000 words. Expansive world, deep branching.
    ```
    Also update the fallback instruction at line 124: "If the user doesn't express a preference, default to `medium`."

- **[hard] [schema-skew]** — No instruction for R-1.2: genre and subgenre as distinct fields; discuss phase can let user (and model) conflate them
  - Where: Line 18 (`1. What genre and subgenre? (e.g., "dark fantasy mystery")`) — the example shows them merged into one string, which is exactly the R-1.2 violation pattern
  - Spec citation: `dream.md §R-1.2` and its Violations table: `Vision has genre: "cozy mystery" and no subgenre — root cause: Genre and subgenre conflated`
  - Recommended fix: Split question 1 into two and add an explicit constraint:
    ```
    1a. What is the PRIMARY genre? (one word or two: "fantasy", "mystery", "horror")
    1b. What is the SUBGENRE? (a refinement: "dark fantasy", "cozy mystery", "psychological horror")

    IMPORTANT: Genre and subgenre are SEPARATE fields. "cozy mystery" is a subgenre of "mystery".
    Do NOT merge them — always elicit both.
    ```

- **[soft] [sm-fragile]** — R-1.3 (themes must be abstract, not plot points) is not stated in the discuss prompt; the scope fence in lines 45–60 prevents carrying plot into the brief, but does not teach the model the right framing during exploration
  - Where: line 23 (`3. What themes will the story explore? (e.g., "memory vs consent")`) — good example, but no explicit rule or bad example
  - Spec citation: `dream.md §R-1.3` and Violations table: `themes: ["the mentor betrays the protagonist"]` — plot point masquerading as theme
  - Recommended fix: Add a inline constraint after question 3:
    ```
    3. What ABSTRACT THEMES will the story explore? (e.g., "memory vs consent", "the price of loyalty")
       GOOD: "forbidden knowledge", "trust and betrayal" (abstract ideas)
       BAD: "the mentor dies in chapter three", "protagonist discovers a secret letter" (plot points)
    ```

- **[soft] [sm-fragile]** — No sandwich repetition of the scope boundary constraint; the `DREAM Stage Scope (CRITICAL)` block appears only once, at the bottom of the prompt; small models completing a long discussion will have lost this by the time they make late-stage topic suggestions
  - Where: lines 32–63 (single occurrence of the scope fence)
  - Spec citation: `CLAUDE.md §10` (small-model bias); role definition §Required reading #1 (constraint-to-value mapping loss)
  - Recommended fix: Add a one-line echo of the key rule after the `{mode_section}` injection point at the very end of the template:
    ```
    REMINDER: DREAM = high-level vision only. No character names, scene descriptions, or mechanics.
    ```

- **[soft] [repair-gap]** — The `non_interactive_section` (line 67) instructs the model to "make confident, specific creative choices" but gives no fallback values for what counts as a complete vision; a model that omits `genre`, `tone`, or `themes` during autonomous mode will have its output silently accepted by the discuss phase (the serialize phase will attempt to extract what's missing)
  - Where: lines 67–71 (`non_interactive_section`)
  - Spec citation: `dream.md §R-1.8` (required fields non-empty); `CLAUDE.md §Repair-loop quality`
  - Recommended fix: Add a checklist to the non-interactive section:
    ```
    Before concluding, confirm you have established:
    - Genre (single primary category)
    - Subgenre (a refinement of genre)
    - Tone (2-4 descriptors)
    - At least 2 abstract themes
    - Target audience
    - Story size (micro/short/medium/long)
    ```

- **[info] [sm-fragile]** — `present_options` tool instruction (lines 104–112) uses "INVOKE the tool through function calling" twice; for small models this level of emphasis is appropriate, no change needed
  - Where: lines 104–112
  - Spec citation: `CLAUDE.md §10`
  - Recommended fix: None.

---

### `summarize.yaml` — DREAM-specific audit

**Verdict:** mixed

**Findings:**

- **[hard] [drift]** — Scope preset names include `vignette` and `standard`; correct names are `micro` and `medium`
  - Where: line 14 (`Story size: one of "vignette", "short", "standard", or "long"`)
  - Spec citation: `src/questfoundry/models/dream.py` line 34; `src/questfoundry/pipeline/size.py` PRESETS keys
  - Recommended fix: Change the instruction to: `Story size: one of "micro", "short", "medium", or "long" (preserve the exact keyword from the discussion)`. If the discussion used an old label (e.g., the user said "standard"), map it to "medium" here before passing to serialize.

- **[soft] [sm-fragile]** — Scope boundary section (lines 38–50) lists what to exclude but has no GOOD/BAD examples for the summarizer itself; a 4B model given a long discussion containing character names will sometimes retain them anyway
  - Where: lines 38–50
  - Spec citation: `CLAUDE.md §7` (defensive prompt patterns — every constrained section needs GOOD/BAD examples)
  - Recommended fix: Add a concrete example pair immediately after the scope exclusion list:
    ```
    GOOD summary: "Dark fantasy mystery. Themes: forbidden knowledge, trust, corruption.
    Tone: atmospheric, morally ambiguous. Audience: adult literary IF readers. Size: short."

    BAD summary (DO NOT include): "Julia the baker discovers Elias's secret at the lighthouse.
    The story has four endings. Memory resource system tracks edits."
    ```

- **[info] [schema-skew]** — `content_notes` in the summarize output format is described as a single bullet-point category with no mention of `includes`/`excludes` sub-structure; the serialize step must then infer the split
  - Where: lines 8–16 (bullet list of categories to summarize)
  - Spec citation: `src/questfoundry/models/dream.py` lines 16–23 (`ContentNotes.includes`, `ContentNotes.excludes`)
  - Recommended fix: This is low-priority because the summarize output is prose, not JSON — serialize does the structural mapping. However, guiding the summary to use "include:" / "exclude:" language would make serialize's job more reliable. Add: `- Content notes (split into: what to include/embrace vs. what to exclude/avoid)`.

---

### `serialize.yaml` — DREAM-specific audit

**Verdict:** mixed

**Findings:**

- **[hard] [drift]** — Scope preset list shows `vignette/short/standard/long`; correct names are `micro/short/medium/long`
  - Where: lines 26–32 (the scope field example block)
  - Spec citation: `src/questfoundry/models/dream.py` line 34 (`Literal["micro", "short", "medium", "long"]`); `size.py` PRESETS; serialization will produce a `ValidationError` if "vignette" or "standard" is emitted
  - Recommended fix: Update the example block and the fallback rules:
    ```
    Pick one of: "micro", "short", "medium", "long".

    Example: "scope": {"story_size": "micro"}

    If the brief says "vignette-length" or "vignette" → use "micro".
    If the brief says "standard" → use "medium".
    If the brief says "short" → use "short".
    If the brief says "long" → use "long".
    If no size is mentioned → use "medium".
    ```
    This also serves as a translation layer for any legacy labels that survived through discuss/summarize.

- **[hard] [schema-skew]** — No instruction for the `content_notes` nested structure (`includes`/`excludes` object); model will likely produce a flat list or a string, causing a Pydantic validation failure
  - Where: lines 33–43 (Scope Boundary section lists allowed fields but gives no example of `content_notes` shape)
  - Spec citation: `src/questfoundry/models/dream.py` lines 15–23 (`ContentNotes(excludes: list[str], includes: list[str])`)
  - Recommended fix: Add an explicit example to the Guidelines section:
    ```
    ## content_notes field (optional nested object)
    If content notes exist, output MUST use this structure (NOT a flat list):
    "content_notes": {
      "includes": ["single-protagonist POV", "atmospheric horror elements"],
      "excludes": ["graphic violence", "explicit magic systems"]
    }
    If no content notes were discussed, omit the field entirely.
    ```

- **[hard] [schema-skew]** — No instruction for `pov_style` valid values; model must produce exactly one of `first_person`, `second_person`, `third_person_limited`, `third_person_omniscient` or `null`; without explicit listing, small models emit free-form strings that fail the Pydantic Literal
  - Where: lines 33–43 (field list mentions `genre`, `subgenre`, `tone`, `audience`, `themes`, `style_notes`, `scope`, `content_notes` — `pov_style` is absent)
  - Spec citation: `dream.md §R-1.9`; `src/questfoundry/models/dream.py` lines 63–70
  - Recommended fix: Add to the Guidelines section:
    ```
    ## pov_style field (optional)
    If a POV preference was discussed, set pov_style to EXACTLY one of:
      "first_person", "second_person", "third_person_limited", "third_person_omniscient"
    Any other value will fail validation. If no preference was stated, omit the field.
    ```

- **[soft] [schema-skew]** — `protagonist_defined` field is present in the Pydantic model but not mentioned in the serialize prompt; the model will always emit the default value (`false`) regardless of what was discussed
  - Where: lines 7–46 (no mention of `protagonist_defined`)
  - Spec citation: `src/questfoundry/models/dream.py` lines 75–78
  - Recommended fix: Add: "Set `protagonist_defined: true` if the discussion established a specific main character; omit or set `false` if the protagonist is left open."

- **[soft] [sm-fragile]** — The instruction "Output ONLY the structured JSON" (line 22) does not say what the top-level key structure looks like; with `with_structured_output()` driving serialization, the model may wrap its output in a `{"dream": {...}}` envelope or in a plain dict depending on how the schema is presented
  - Where: lines 18–23
  - Spec citation: `CLAUDE.md §9` (never interpolate Python objects; always explicitly format)
  - Recommended fix: Add a skeleton example:
    ```
    ## Output skeleton (fill in the values — omit optional fields if absent):
    {
      "genre": "...",
      "subgenre": "...",        ← optional
      "tone": ["...", "..."],
      "themes": ["...", "..."],
      "audience": "...",
      "scope": {"story_size": "micro|short|medium|long"},
      "style_notes": "...",     ← optional
      "content_notes": {"includes": [...], "excludes": [...]},  ← optional
      "pov_style": "...",       ← optional
      "protagonist_defined": false
    }
    ```

- **[info] [schema-skew]** — `style_notes` has a `min_length=1` constraint in Pydantic; if the field is serialized as an empty string it will fail; the current prompt says "If a field is optional and not mentioned, omit it" which is correct — this is an alignment note only
  - Where: line 14 ("If a field is optional and not mentioned, omit it")
  - Spec citation: `src/questfoundry/models/dream.py` line 50 (`min_length=1`)
  - Recommended fix: None needed — "omit it" is the correct instruction.

---

### Stage summary: DREAM

DREAM's three active prompts (`discuss.yaml`, `summarize.yaml`, `serialize.yaml`) share a single critical drift that will cause runtime `ValidationError` on every run: the scope preset names `vignette` and `standard` appear in all three files but the Pydantic model and `size.py` runtime accept only `micro`, `short`, `medium`, and `long`. The serialize prompt also silently omits the two most structurally complex fields (`content_notes` nested object, `pov_style` Literal) — a small model will either guess incorrectly or omit them unpredictably. The `discuss.yaml` compounds this by using a merged genre/subgenre example that directly mirrors the R-1.2 violation pattern in the spec.

`dream.yaml` is dead code (not loaded by any active code path). Its `pov_style` display labels, Jinja2 variable syntax, and `content_notes` flat-list description are all wrong, but they have no runtime impact today. The file should be deleted or reconciled under a dedicated issue.

- **Prompts audited:** 3 (`discuss.yaml`, `summarize.yaml`, `serialize.yaml`; `dream.yaml` is dead code — not counted)
- **Hard findings:** 7
- **Soft findings:** 5
- **Info findings:** 3
- **Spec gaps surfaced:** 0
- **Recommended PR split:** per-cluster split — one PR for the `vignette`/`standard` → `micro`/`medium` rename across all three files (low-risk, mechanical); one PR for the `content_notes` structure + `pov_style` explicit listing in `serialize.yaml` (schema-completeness cluster); one PR for the discuss-phase R-1.2/R-1.3 instruction additions; one issue (not PR) to delete or reconcile `dream.yaml`.
- **Status:** drift

---

## BRAINSTORM

(Pending — Task 7.)

---

## SEED

(Pending — Task 8.)

> **Already-known finding from Phase 1:** `serialize_seed_sections.yaml`
> per-path-beats repair-loop didn't echo expected `also_belongs_to`
> value. Fixed in PR #1384. The subagent should still re-audit this
> prompt in Task 8 to catch any other findings (the smoke test
> already surfaced 5 bonus items).

---

## GROW

(Pending — Task 9.)

---

## POLISH

(Pending — Task 10.)

---

## FILL

(Pending — Task 11.)

---

## DRESS

(Pending — Task 12.)

---

## SHIP

(Pending — Task 13.)
