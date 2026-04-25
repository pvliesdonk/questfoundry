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
| BRAINSTORM | 3 | 8 | 9 | 3 | 1 | drift |
| SEED | 5 | 8 | 16 | 5 | 1 | drift |
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

### `discuss_brainstorm.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Prompt instructs model to list `central_entity_ids` inline in dilemma prose using bracket-list notation `"Central entities: [entity_id_1], [entity_id_2]"`, which violates CLAUDE.md §9 (Python list repr in LLM-facing text) and seeds the model to produce bracket lists in the summary that the summarizer echoes verbatim into the brief.
  - Where: lines 67–68; mirrored in `summarize_brainstorm.yaml` line 37.
  - Spec citation: `CLAUDE.md §9 Prompt Context Formatting (CRITICAL)`.
  - Recommended fix: Replace with backtick-separated prose: `e.g., "Central entities: \`entity_id_1\`, \`entity_id_2\`"`.

- **[hard] [schema-skew]** — `dilemma::` prefix never mentioned in discuss prompt; all GOOD examples lack the prefix. Small models produce un-prefixed IDs in summary; serializer fails `validate_dilemma_id_format` on every run.
  - Where: lines 84–99 (`## Dilemma ID Naming`).
  - Spec citation: `brainstorm.md §R-3.7`; `models/brainstorm.py:126–139`.
  - Recommended fix: Update every example ID to include `dilemma::` prefix. Add: "**Every dilemma ID MUST start with `dilemma::`** — the serializer will reject any ID that does not."

- **[hard] [schema-skew]** — Prompt prose uses "Entity Type" / "type" but field is `entity_category`. Mental-model drift in discuss propagates to serializer.
  - Where: lines 18–21, 171.
  - Spec citation: `models/brainstorm.py:41`; `brainstorm.md §R-2.2`.
  - Recommended fix: Replace "entity type" with "entity category" throughout. Add anchor sentence: "Each entity has an **entity_category** (character, location, object, or faction)".

- **[hard] [drift]** — Entity namespace (`character::mentor`, `location::archive`) required by R-2.3 but never taught in discuss prompt. Bare IDs propagate, `anchored_to` validation fails because graph stores namespaced IDs.
  - Where: entire `## Your Goal → Entity` section (lines 18–45).
  - Spec citation: `brainstorm.md §R-2.3`; `story-graph-ontology.md §Part 1 Entity`.
  - Recommended fix: Add: "**Entity IDs include their category as a namespace** (e.g., `character::mentor`, `location::archive`, `faction::conspiracy`)".

- **[soft] [sm-fragile]** — No sandwich repetition for binary-only and single-canonical constraints. They appear once mid-prompt; small models forget by output time.
  - Where: lines 63–75 (no bottom-of-prompt repetition).
  - Recommended fix: Add `## Critical Rules (Reminder)` section before `{output_language_instruction}` echoing the four critical rules.

- **[soft] [sm-fragile]** — No GOOD/BAD examples for entity ID format. Small models produce `Lady Beatrice`, `The Mentor`, `main_character_mentor`.
  - Where: entity section (lines 18–45).
  - Spec citation: `CLAUDE.md §7`.
  - Recommended fix: Add example block: GOOD `character::lady_beatrice`, BAD `Lady Beatrice` (spaces), `mentor` (missing namespace).

- **[soft] [repair-gap]** — No fallback instruction for tool errors in `research_tools_section`. Small model that gets a tool error may halt or loop.
  - Where: lines 138–167.
  - Spec citation: `CLAUDE.md §7`.
  - Recommended fix: Add: "If a tool call returns an error or no_results, note it briefly and continue. Do not retry the same query more than once."

- **[info] [schema-skew]** — `{size_entities}` and `{size_dilemmas}` correctly injected from `size_template_vars()`. No action required.

- **[info] [spec-gap]** — `brainstorm.md §R-1.1` says "typical targets are 15–25 entities and 4–8 dilemmas for a short story" but `size.py` `short` preset has 10–18 entities, 3–5 dilemmas. Update R-1.1 to reference size preset system rather than hardcoded targets.

---

### `summarize_brainstorm.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Summarizer instructed to record `central_entity_ids` as bracket-list `"Central entity IDs: [character_id], [location_id]"`. Same Python-list-repr anti-pattern; serializer may emit literal bracket strings.
  - Where: lines 36–37.
  - Spec citation: `CLAUDE.md §9`.
  - Recommended fix: Use backtick notation with real namespaced IDs: `e.g., \`character::mentor\`, \`location::archive\``.

- **[hard] [repair-gap]** — Summarizer has no instruction for handling missing `why_it_matters`. Will silently invent or omit, both of which are silent degradation.
  - Where: `## Dramatic Dilemmas` section (lines 33–39).
  - Spec citation: `brainstorm.md §R-3.1`; CLAUDE.md anti-patterns.
  - Recommended fix: Add: "If `why_it_matters` was not discussed, do NOT invent one. Write: `why_it_matters: [NOT DISCUSSED — serializer will flag this]`."

- **[soft] [sm-fragile]** — `## NO DELEGATION (CRITICAL)` block (lines 41–49) appears AFTER main instructions; small models read top-to-bottom and have already framed task as advisory by then.
  - Recommended fix: Hoist the block to be first section after the task statement, OR add one-line guard at top: "**You are writing the summary itself — NOT advising on how to write it.**"

- **[soft] [schema-skew]** — Section headers use plural `### Characters`, `### Locations`; `entity_category` Pydantic Literal accepts singular `"character"` etc. A small serializer model may emit `entity_category: "characters"`.
  - Where: lines 12–31.
  - Recommended fix: Change to `Characters (entity_category: "character")` to lock the mapping.

- **[soft] [drift]** — Summarizer template doesn't reference `{size_*}` variables but `prompts.py:218` passes them via `**size_template_vars(size_profile)` — silently ignored. Either remove the kwargs from the function or add size guidance to the template (latter is more useful).

- **[info] [schema-skew]** — Summarizer doesn't mention `name` field on entities. Drops names that were established in discussion.
  - Recommended fix: Add per-section: "If a specific name was established in discussion, include it. If only a role was discussed, omit the name — the serializer will leave it absent for SEED to generate."

---

### `serialize_brainstorm.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Table column header says "Entity Type" but actual field is `entity_category`. Small model reads column header as field name.
  - Where: lines 9–13 (`## Entity Mapping (CRITICAL)` table).
  - Spec citation: `models/brainstorm.py:41`.
  - Recommended fix: Rename column to `entity_category value` so table reads `| Brief Category | entity_category value |`.

- **[hard] [schema-skew]** — All GOOD examples in `## Dilemma ID Naming (CRITICAL)` lack the `dilemma::` prefix despite the prefix being a hard requirement. Violates §7's "GOOD example must demonstrate the rule" mandate.
  - Where: lines 40–57.
  - Spec citation: `brainstorm.md §R-3.7`; `models/brainstorm.py:126–139`; `CLAUDE.md §7`.
  - Recommended fix: Prefix every GOOD example with `dilemma::`. Add `dilemma::d1` to BAD examples to show prefix alone isn't enough.

- **[hard] [schema-skew]** — Serialize prompt has NO `### Valid Entity IDs` section, violating CLAUDE.md §6 Valid ID Injection Principle. Model must infer valid IDs from prose and will invent names like `mentor_character`, `the_mentor`.
  - Where: entire file.
  - Spec citation: `CLAUDE.md §6`; `brainstorm.md §Implementation Constraints`.
  - Recommended fix: Restructure serialize to two-pass flow: serialize entities first, extract IDs, then serialize dilemmas with `{valid_entity_ids}` injected. `## Guidelines` should state: "ONLY use entity_id values listed in `### Valid Entity IDs` above."

- **[soft] [repair-gap]** — `BrainstormMutationError._format_message()` ends with generic "Use entity_id values from the entities list" without listing them. Small model on retry has lost system prompt context.
  - Where: `mutations.py` lines 223–234.
  - Recommended fix: In `_format_message()`, append: `f"Valid entity_id values: {', '.join(f'\`{i}\`' for i in all_ids[:15])}"`.

- **[soft] [sm-fragile]** — `## Guidelines` (lines 62–68) uses passive "Extract ALL" wording that reads as advisory, buried after schema sections.
  - Recommended fix: Rewrite as `## Rules (MUST follow)` with bolded MUST/NEVER rules at top of section.

- **[soft] [schema-skew]** — Serialize prompt says `name: Optional canonical display name … Leave absent if no specific name was established`. But `Entity.name` Pydantic field is `str` with `min_length=1` (REQUIRED, not Optional). Direct contradiction — every nameless entity will fail validation.
  - Where: serialize_brainstorm.yaml lines 22–24; `models/brainstorm.py:44`.
  - Spec citation: `brainstorm.md §R-2.1`.
  - Recommended fix: Either make `Entity.name` optional in Pydantic, or update prompt to always require a name. Spec R-2.1 mandates non-empty name, so prompt fix: "If no specific name was established, generate a simple descriptive name (e.g., 'The Mentor', 'The Archive') — do not leave name absent."

- **[info] [drift]** — `## Output` block says "Return ONLY valid JSON matching BrainstormOutput schema" without showing the skeleton. Small models would benefit from explicit JSON skeleton with all top-level keys.

---

### Stage summary: BRAINSTORM

The three BRAINSTORM prompts collectively have a significant alignment gap between the discuss phase (which teaches informal notation) and the serialize phase (which enforces strict schema rules). The core failure chain: discuss teaches bare dilemma IDs and bracket-list entity references → summarizer echoes both → serializer receives ambiguous notation → emits invalid `central_entity_ids` and un-prefixed `dilemma_id`s → Pydantic + semantic validation fires → repair-loop gets generic message without the actual entity list → second attempt also fails. Breaking this chain requires four coordinated fixes (prefix propagation, ID notation, Valid ID injection, repair message).

There is one genuine spec gap: `brainstorm.md §R-1.1` cites hardcoded targets ("15–25 entities, 4–8 dilemmas") that contradict `size.py` preset values. Update spec to defer to size preset system.

The `Entity.name` required/optional contradiction between serialize prompt ("leave absent") and Pydantic (`min_length=1`) will cause repair-loop hits on every nameless entity.

- Prompts audited: 3 (`discuss_brainstorm.yaml`, `summarize_brainstorm.yaml`, `serialize_brainstorm.yaml`)
- Hard findings: 8
- Soft findings: 9
- Info findings: 3
- Spec gaps surfaced: 1 (R-1.1 abundance targets contradict size preset)
- Recommended PR split: two PRs — A (prompt fixes: ID prefix propagation, bracket notation, Valid ID injection, name required/optional, table column rename); B (repair-loop improvements: `BrainstormMutationError.to_feedback()` self-contained entity list, sandwich repetition in discuss)
- Status: drift

---

## SEED

### `discuss_seed.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Worked Example shows legacy `paths: [...]` list format for beats; pre-commit beats MUST use `path_id` + `also_belongs_to` directly per Y-shape (#1206).
  - Where: "### 4. Create Initial Beats for Each Path" section.
  - Spec citation: `seed.md §R-3.6`; `models/seed.py:283–323` (migration emits DeprecationWarning).
  - Recommended fix: Replace generic beat description with concrete Y-shape examples for shared (with `also_belongs_to`) vs per-path (with `also_belongs_to: null`) beats.

- **[hard] [sm-fragile]** — Branching constraint ("at least 2 dilemmas must have both answers explored") absent from discuss prompt; only stated in summarize. The discuss agent makes the actual exploration decisions, so the constraint must be HERE.
  - Where: "### 2. Choose Which Answers to Explore"; final paragraph contradicts: "you don't need to count arcs or worry about limits".
  - Spec citation: `seed.md §R-5.1`; `summarize_seed_sections.yaml:52–66`.
  - Recommended fix: Add MINIMUM BRANCHING REQUIREMENT block with GOOD/BAD examples; remove the contradicting "don't worry" sentence.

- **[soft] [drift]** — `is_canonical: true` terminology used in discuss but `(default)` marker used everywhere else.
  - Recommended fix: Add clarifier "the canonical answer is marked `(default)`".

- **[soft] [sm-fragile]** — Convergence target `{size_convergence_points}` doesn't note that hard dilemmas have NO convergence (paths never rejoin).
  - Recommended fix: Add caveat: "Hard dilemmas do NOT need convergence points — target applies to soft dilemmas only."

- **[soft] [schema-skew]** — Beat description omits `temporal_hint`, `location_alternatives`, `role: setup/epilogue` fields that exist in `InitialBeat`.
  - Recommended fix: Add brief note about location alternatives + temporal hints.

- **[soft] [schema-skew]** — Consequences described only as "list of consequence IDs" — `Consequence` Pydantic requires `description` (min_length=1) and `narrative_effects`.
  - Recommended fix: Expand: "consequences — what happens in the world after this path. Each has description (world state, not player action) and downstream ripples."

- **[info] [drift]** — `research_tools_section` references `list_clusters`; verify it still exists in the active corpus MCP.

---

### `summarize_seed.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — `### Initial Beats (Y-shape: …)` section uses legacy `paths: list of path IDs` field — same Y-shape gap as discuss prompt. Pre-commit beats need `path_id` + `also_belongs_to`.
  - Where: lines 84–90.
  - Spec citation: `seed.md §R-3.6`; `models/seed.py:283–323`.
  - Recommended fix: Replace beats schema with canonical Y-shape fields explicitly distinguishing shared (with `also_belongs_to`) vs per-path beats.

- **[hard] [repair-gap]** — Summarizer has detailed VERIFY checks for entity/dilemma counts but NO checks for beat counts or Y-shape completeness. Pre-commit beats without `also_belongs_to` pass through summarize silently and only fail in serialize.
  - Where: "### Initial Beats (Y-shape: …)" section.
  - Spec citation: `seed.md §R-3.6`, §R-3.10`, §R-3.12`.
  - Recommended fix: Add VERIFY block: "For EACH dilemma with 2 explored paths: at least 1 shared beat with both `path_id` AND `also_belongs_to` set; each path has exactly 1 commit beat; each path has ≥2 post-commit beats."

- **[soft] [sm-fragile]** — Location Constraint (CRITICAL) appears once at end with no top-of-prompt repetition.
  - Recommended fix: Add one-line echo at end: "REMINDER: All location IDs MUST be retained entity IDs from brainstorm."

- **[soft] [schema-skew]** — Output format prose for Entity Decisions Manifest doesn't capture `name` field needed for entities marked "(needs name)".
  - Recommended fix: Add: "If an entity needs a name: `- character::unnamed_spy: retained - name: 'Marcus Delacroix' - ...`".

- **[info] [drift]** — Path schema mentions `shadows: IDs of unexplored answers` but field is named `unexplored_answer_ids`.

---

### `summarize_seed_sections.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [sm-fragile]** — `dilemmas_system` MINIMUM BRANCHING REQUIREMENT says "GO BACK and fully explore at least 2 dilemmas" — small models can't "go back"; constraint is unactionable.
  - Where: lines 52–66 (instruction) and 143–147 (FINAL CHECK).
  - Spec citation: `seed.md §R-5.1`; `CLAUDE.md §10`.
  - Recommended fix: Replace "GO BACK" with self-sufficient OVERRIDE instruction: "For any dilemma NOT marked 'explore both' in the discussion, you MAY add the non-default answer to `explored` if total fully-explored count is still below 2."

- **[hard] [schema-skew]** — `beats_system` instructs production of beats with deprecated `paths` field — same Y-shape gap.
  - Where: lines 199–212.
  - Recommended fix: Same as `summarize_seed.yaml` — explicit `path_id` + `also_belongs_to` description with shared vs per-path distinction.

- **[soft] [sm-fragile]** — `entities_system` silent default ("if not discussed, write disposition `retained`") produces bloated retained sets violating R-1.4 spirit.
  - Recommended fix: "If an entity was not discussed, make an editorial judgment — retain if narratively necessary; cut if peripheral. Do NOT blindly retain."

- **[soft] [sm-fragile]** — `paths_system` lacks GOOD/BAD examples for path ID format (`path::dilemma_id__answer_id` with double underscore).
  - Recommended fix: Add explicit examples showing the prefix and double-underscore pattern.

- **[soft] [schema-skew]** — `convergence_system` produces convergence_points/residue_notes fields not directly mapped in any serialize schema; advisory only.
  - Recommended fix: Add note explaining how Section 7 (Dilemma Analysis) consumes this output.

- **[info] [schema-skew]** — `paths_system` doesn't mention optional `pov_character` field on `Path`.

---

### `serialize_seed.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Section 5 schema example shows beat with single-element `paths: [...]` array (legacy format) — directly contradicts Y-shape requirement. Doesn't demonstrate `also_belongs_to`.
  - Where: lines 73–97.
  - Spec citation: `seed.md §R-3.6, §R-3.7`; `models/seed.py:271–285`.
  - Recommended fix: Replace single example with TWO canonical examples (shared pre-commit with `also_belongs_to` set; per-path commit/post-commit with `also_belongs_to: null`).

- **[hard] [schema-skew]** — Investigate whether `serialize_seed.yaml` is still on any active code path. The chunked `serialize_seed_sections.yaml` is the documented path. If monolithic is dead, mark with header comment and file tracking issue. If active, must fix Y-shape examples per above.

- **[soft] [sm-fragile]** — Schema Overview says "six main sections" but only shows five. `dilemma_analyses` and `dilemma_relationships` (Sections 7+8) are absent.
  - Recommended fix: Either correct the count or add the missing sections.

- **[soft] [schema-skew]** — Path schema example shows `consequence_ids: ["host_revealed"]` implying consequences pre-exist; in chunked flow they're generated AFTER paths.
  - Recommended fix: Use `consequence_ids: []` with note: "Leave empty when generating paths — populated by Section 4."

- **[info] [terminology]** — Mapping Rules table shows only 5 of 7 SeedOutput sections.

---

### `serialize_seed_sections.yaml`

**Verdict:** mixed

**Re-confirmation of Phase 1 PR (#1384) fix (already-known finding):**

The `also_belongs_to` repair-gap is confirmed addressed via `extra_repair_hints` in serialize.py (Option B). Repair message echoes both `path_id` and `also_belongs_to` values per call. No further action needed on this hard finding.

**Other findings:**

- **[hard] [schema-skew]** — `beats_prompt` (the flat all-paths-at-once section) is REGISTERED in `_SEED_SECTION_PROMPTS` but pre-dates Y-shape refactor and has zero Y-shape awareness — no `also_belongs_to`, no shared/commit/post-commit distinction. **If pipeline can reach this prompt on a non-Y-shape code path, it produces stories with no Y-fork → POLISH Phase 4c produces 0 choices** (the known critical failure mode).
  - Where: `beats_prompt` lines 401–623.
  - Spec citation: `seed.md §R-3.10`; `story-graph-ontology.md §Part 8`.
  - Recommended fix: Investigate runtime: does pipeline use `shared_beats_prompt` + `per_path_beats_prompt` exclusively, or fall back to `beats_prompt`? If fallback path exists, this needs Y-shape rewrite. If dead, mark + tracking issue.

- **[soft] [sm-fragile]** (smoke-test bonus #2, confirmed) — `per_path_beats_prompt` FINAL VERIFICATION doesn't re-check `also_belongs_to: null` for post-commit beats.
  - Recommended fix: Add to FINAL VERIFICATION: "3. `also_belongs_to` is null for EVERY beat in this output."

- **[soft] [sm-fragile]** (smoke-test bonus #3, confirmed) — `shared_beats_prompt` `also_belongs_to` constraint is bullet #7; should be #1.
  - Recommended fix: Hoist to first bullet with REQUIRED emphasis.

- **[soft] [schema-skew]** (smoke-test bonus #4, confirmed) — `per_path_beats_prompt` schema uses `"dilemma::[other_dilemma_id]"` square-bracket placeholder violating §9.
  - Recommended fix: Replace with `"dilemma::another_dilemma_id"` + explicit "replace with actual ID — NOT a template" note.

- **[hard] [schema-skew]** — `dilemma_analyses_prompt` Self-Check says "FLAVOR count is 0-2" but `flavor` role is now rejected outright (R-7.1). Small model following Self-Check produces invalid `dilemma_role: "flavor"` → Pydantic rejection.
  - Where: Self-Check section, line 1113.
  - Spec citation: `seed.md §R-7.1`; `models/seed.py:400`.
  - Recommended fix: Remove flavor mention; replace with "Every entry has `dilemma_role` exactly `hard` or `soft` (no other values accepted)."

- **[soft] [schema-skew]** — `dilemma_relationships_prompt` requires exhaustive O(n²) pair declaration but `seed.md §R-8.2` says "Exhaustive O(n²) declaration is forbidden". Spec vs prompt direct contradiction.
  - **Spec-gap finding:** Resolution requires spec update. Current behaviour (exhaustive) is operationally sensible (missing pairs ambiguous for GROW); spec amendment recommended: "Including concurrent-default pairs for completeness is acceptable."

- **[soft] [sm-fragile]** — `per_dilemma_paths_prompt` schema example uses `{dilemma_name}__{answer_id_example}` template variables inside double-brace JSON; small model may copy literally.
  - Recommended fix: Use `path::DILEMMA_NAME__ANSWER_ID` with explicit "replace with actual values" note.

- **[info] [terminology]** (smoke-test bonus #5, confirmed) — "commit" used both as field value and concept.

---

### Stage summary: SEED

The SEED stage has the most complex prompt set in the pipeline (5 files, 8+ sections). The Y-shape refactor (#1206) was partially propagated but three concrete gaps remain: (1) deprecated `paths: [a,b]` format still in `summarize_seed.yaml` + `serialize_seed.yaml` schema examples; (2) `beats_prompt` in `serialize_seed_sections.yaml` may still be on active code path without Y-shape awareness (would cause known 0-choice POLISH failure); (3) `summarize_seed_sections.yaml` `beats_system` shows deprecated field. Phase 1 PR #1384 fix for `also_belongs_to` repair-gap is confirmed sufficient. The `flavor` role appearing in `dilemma_analyses_prompt`'s Self-Check is a Pydantic validation hazard. The minimum-branching constraint is absent from `discuss_seed.yaml` where it actually matters. Spec-gap: R-8.2 sparse vs prompt's exhaustive pair declaration.

- Prompts audited: 5 (discuss_seed, summarize_seed, summarize_seed_sections, serialize_seed, serialize_seed_sections)
- Hard findings: 8
- Soft findings: 16
- Info findings: 5
- Spec gaps surfaced: 1 (R-8.2 sparse vs exhaustive — recommend permitting exhaustive)
- Recommended PR split: PR-A (Y-shape schema examples in summarize/serialize + `also_belongs_to` sandwich + flavor removal + branching constraint in discuss_seed); PR-B (beats_prompt dead-code investigation + dilemma_relationships reconciliation pending spec update)
- Status: drift

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
