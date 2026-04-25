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

The audit covers **47 LLM-driven prompts across 7 stages** (SHIP is
deterministic). Every audited stage shows drift between its prompts and the
authoritative specs that were redrafted earlier in 2026 — confirming the
hypothesis that prompts were postponed during the doc/code overhaul.

| Stage | Prompts | Hard | Soft | Info | Spec gaps | Status |
|---|---|---|---|---|---|---|
| DREAM | 3 | 9 | 7 | 4 | 0 | drift |
| BRAINSTORM | 3 | 9 | 9 | 4 | 1 | drift |
| SEED | 5 | 10 | 16 | 5 | 0 | drift |
| GROW | 8 | 3 | 8 | 5 | 0 | drift |
| POLISH | 12 | 5 | 33 | 13 | 1 | drift |
| FILL | 8 | 3 | 10 | 8 | 2 | drift |
| DRESS | 8 | 10 | 17 | 5 | 0 | drift |
| SHIP | 0 | 0 | 0 | 0 | 0 | n/a (deterministic) |
| **Total** | **47** | **49** | **100** | **44** | **4** | — |

*Counts are mechanically derived from the per-stage finding bullets in this report (one `- **[severity]` line = one finding). Per-stage summary blocks inside each section may show slightly different totals because their authoring subagents grouped related issues into single "headline" findings; the table above is the authoritative tally.*

### Cross-cutting themes

Five patterns recur across stages and account for the bulk of the hard
findings. Fix PRs should be grouped by theme where possible to amortise
review and regression cost.

1. **Repair-loop blindness (the murder1 failure pattern).** The original
   trigger — `serialize_seed_sections.yaml` repair feedback naming
   `also_belongs_to` without echoing the expected sibling path id —
   recurs structurally across stages. SEED inherited the Phase 1 fix
   (PR #1384). DRESS shares one generic `_dress_llm_call` repair message
   across all 8 templates that strips valid IDs and value constraints.
   FILL's `_build_error_feedback` does not echo allowed enum values for
   `pov` / `voice_register` / `sentence_rhythm` failures. Every stage
   except SHIP has at least one repair-gap finding.

2. **Pydantic-Literal vs spec drift on `pov`.** `dream.yaml` (dead-code
   notice), `fill_phase0_discuss.yaml`, and `fill_phase0_voice.yaml` all
   use the short forms (`first`, `second`, `third_limited`,
   `third_omniscient`) while `dream.md §R-1.9` and `fill.md §R-1.3`
   require the long forms (`first_person`, etc.). `dream.py` already
   uses the long forms — `fill.py`'s `VoiceDocument.pov` Literal is the
   outlier. Per CLAUDE.md §Design Doc Authority the spec wins; this is
   a single coordinated fix touching the model + both phase0 prompts +
   a fill.md worked-example correction.

3. **Codeword vs state_flag terminology in DRESS.** `dress_codex.yaml`
   and `dress_codex_batch.yaml` consistently instruct the model to use
   "codeword IDs" in `visible_when` while `dress.md §R-3.7` and
   `story-graph-ontology.md §Part 8` are explicit that DRESS gates
   internally via state flag IDs. The stage code correctly reads
   `state_flag` nodes — only the prompt labels (and the Pydantic
   docstring on `CodexEntry.visible_when`) are wrong. This is the single
   largest DRESS drift and a clean ~30-line PR.

4. **POLISH migration residue from epic #1368.** Five GROW phases moved
   to POLISH (narrative_gaps→1a, pacing_gaps→2, entity_arcs→3,
   atmospheric→5e, path_arcs→5f) and the prompt files inherited their
   "GROW Phase Nf" headers, rule citations, and identity statements
   without rewriting. Twelve POLISH templates carry residue; the worst
   case (`polish_phase1a_*`) still mislabels itself as "GROW Phase 4f"
   and emits `dilemma_impacts` with `effect: commits` on post-commit
   beats — a Y-shape guard-rail violation.

5. **Bare-ID context blocks and missing GOOD/BAD examples.** A
   recurring soft pattern: context builders (`format_*_context()`) emit
   `{valid_beat_ids}` / `{entity_visuals}` / `{shadow_context}` /
   `{passages_batch}` as flat or unheaded blocks. Every CLAUDE.md §6
   /§7 /§8 review noted that small-model output quality drops sharply
   when the model cannot tell what a context block is for or which
   constraints apply. Across all stages this accounts for ~40 of the 81
   soft findings.

### Recommended PR clusters (Phase 3)

The fix work is large enough to need batching. Recommended grouping for
the demand-driven Phase 3 PRs:

- **PR cluster A — POV alignment (1 PR, narrow):** `fill.py` model
  + `fill_phase0_discuss.yaml` + `fill_phase0_voice.yaml` + `fill.md`
  worked-example fix. Touches 4 files; no behavioural change beyond
  the Literal rename.
- **PR cluster B — DRESS terminology (1 PR, narrow):** codeword →
  state_flag rename across `dress_codex.yaml`,
  `dress_codex_batch.yaml`, `dress_codex_spoiler_check.yaml`, and the
  `CodexEntry.visible_when` docstring. Add `"cover"` to the
  `IllustrationCategory` enum lists in both brief templates while in
  the same area.
- **PR cluster C — POLISH migration cleanup (1 PR per residue source,
  ~4 PRs):** rewrite phase identity statements; remove `dilemma_impacts`
  emission from post-commit-beat templates; replace "GROW Phase Nf"
  citations.
- **PR cluster D — Repair-feedback enrichment (1 PR per stage where
  applicable):** generalise the SEED Phase-1 pattern (`extra_repair_hints`
  threading) to FILL `_build_error_feedback` and DRESS `_dress_llm_call`,
  with regression tests pinning the enriched feedback wording.
- **PR cluster E — Context enrichment (per-stage, lower priority):**
  add headers to `format_*_context()` outputs; inject
  `{valid_entity_ids}` into FILL revision; add overlay enrichment to
  `format_entity_for_codex()`. This is the long tail and may be split
  into smaller follow-up PRs.

The 5 spec gaps surfaced by the audit (BRAINSTORM 1, SEED 1, POLISH 1,
FILL 2) get spec-update PRs first per CLAUDE.md §Design Doc Authority,
then the prompt fixes follow.

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

### `grow_phase3_intersections.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [schema-skew]** — `resolved_location` is `str | None` in `IntersectionProposal` but prompt says "NEVER output null"; schema-prompt contradiction. Code also guards against literal `"null"` string.
  - Recommended fix: Change `IntersectionProposal.resolved_location` to `str = Field(min_length=1, ...)`.

- **[soft] [sm-fragile]** — `valid_beat_ids` injected as flat comma-separated string (hundreds of tokens for large stories). Small models lose track.
  - Spec citation: `CLAUDE.md §6 Valid ID Injection`.
  - Recommended fix: Group by dilemma in Valid IDs section: `dilemma::mentor_trust → beat::a, beat::b | dilemma::archive → beat::c`.

- **[info] [drift]** — Spec calls this GROW Phase **2** (Intersection Detection); prompt filename and code label say "Phase 3" (pre-migration numbering).
  - Recommended fix: Rename template to `grow_phase2_intersections.yaml` (cosmetic, low priority).

---

### `grow_phase4a_scene_types.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [terminology]** — Template named `phase4a` but spec calls Scene Types Phase **4b** (4a is now Interleave, deterministic). Pydantic class is `Phase4aOutput` for scene types and `Phase4bOutput` for migrated gap proposals — both contradict spec.
  - Spec citation: `grow.md §Phase 4 / 4b — Scene Types Annotation`.
  - Recommended fix: Rename template + Pydantic class; rename old `Phase4bOutput` to reflect migrated status.

- **[soft] [sm-fragile]** — `beat_summaries` and `valid_beat_ids` flat-listed; large stories cause position-bias errors in small models.
  - Recommended fix: Add `## Path Groups` subsection grouping beats by path/dilemma. Context-builder change.

- **[info] [schema-skew]** — `exit_mood` constraints align with prompt description. Clean.

---

### `grow_phase4b_narrative_gaps.yaml`

**Verdict:** drift (DEAD CODE)

**Findings:**

- **[hard] [drift]** — File corresponds to work **migrated to POLISH Phase 1a** in epic #1368 PR #1366. Per `llm_phases.py:651-654` comment: "GROW Phase 4b (narrative_gaps) was MOVED to POLISH Phase 1a." Method `_phase_4b_narrative_gaps` does NOT exist; phase `"narrative_gaps"` registered in `_METHOD_PHASES`/`_PREDECESSOR_PHASES` but unreachable from `_phase_order()` (no `@grow_phase` decorator). **Prompt file is dead code AND `_METHOD_PHASES` has dangling entries that would AttributeError if accidentally routed**.
  - Spec citation: `grow.md §Phase 4` (no narrative_gaps sub-phase); `CLAUDE.md §Refactoring & removal discipline`.
  - Recommended fix: Delete prompt file. Remove `"narrative_gaps"` from `_METHOD_PHASES` and `_PREDECESSOR_PHASES`. Track with issue.

---

### `grow_phase4c_pacing_gaps.yaml`

**Verdict:** drift (DEAD CODE)

**Findings:**

- **[hard] [drift]** — Same situation: **migrated to POLISH Phase 2 (extended)** per `llm_phases.py:656-661`. Method `_phase_4c_pacing_gaps` doesn't exist; registry entry unreachable.
  - Recommended fix: Delete prompt file. Remove `"pacing_gaps"` from `_METHOD_PHASES`/`_PREDECESSOR_PHASES`. Track.

---

### `grow_phase4f_entity_arcs.yaml`

**Verdict:** drift (DEAD CODE)

**Findings:**

- **[hard] [drift]** — File corresponds to work **REMOVED in epic #1368 PR C**. Per `llm_phases.py:673-678`: "GROW Phase 4f (entity_arcs) was REMOVED in issue #1368 PR C." Equivalent moved to POLISH Phase 3 (`arcs_per_path`).
  - Spec citation: `grow.md §Stage Output Contract item 12` ("No … character arc metadata exists"); `story-graph-ontology.md §Character Arc Metadata` (POLISH Phase 3, not GROW).
  - Recommended fix: Delete prompt file. Remove `"entity_arcs"` from `_METHOD_PHASES`. Confirm POLISH `arcs_per_path` follow-up tracking issue exists.

---

### `grow_phase4g_transition_gaps.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — No `## Valid IDs` section. Output uses `transition_id` composite strings (`"beat::a|beat::b"`); small models may invent or corrupt these.
  - Spec citation: `CLAUDE.md §6`.
  - Recommended fix: Add `## Valid IDs` section listing valid `{transition_ids}` injected from code.

- **[soft] [repair-gap]** — No repair-loop / retry instruction. Code passes `semantic_validator` but prompt has no `{transition_feedback}` slot.
  - Recommended fix: Add `{transition_feedback}` template variable (default empty) for structured retry feedback.

- **[info] [drift]** — Spec calls this Phase **4c** (Transition Beat Insertion); template uses `4g` (pre-migration).
  - Recommended fix: Rename to `grow_phase4c_transition_gaps.yaml`.

---

### `grow_phase8c_overlays.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [schema-skew]** — Prompt uses `details: list[{key, value}]` (matching `OverlayProposal`); but graph-storage `EntityOverlay.details: dict[str, str]`. The `_phase_8c_overlays` docstring says "details: {...}" referring to dict form (wrong for the LLM model).
  - Recommended fix: Update docstring to match LLM schema; add test for `details_as_dict()` conversion.

- **[soft] [sm-fragile]** — `valid_entity_ids` and `valid_state_flag_ids` flat comma-separated strings. Consequence context block IS grouped by dilemma — Valid IDs should mirror that grouping.
  - Recommended fix: Group entity IDs by category, state flags by dilemma in Valid IDs section.

- **[info] [terminology]** — `state_flag::` prefix used correctly throughout; no codeword/state_flag drift. Clean.

---

### `grow_phase_temporal_resolution.yaml`

**Verdict:** clean

- **[info] [sm-fragile]** — Synthetic example IDs clearly marked "do not use." `{swap_pairs_context}` injection format depends on context-builder; if it mirrors example, prompt is robust. Monitor on first real run.

---

### Stage summary: GROW

The GROW audit surfaces **three dead-code prompt files** corresponding to phases migrated to POLISH or removed in epic #1368 (`narrative_gaps`, `pacing_gaps`, `entity_arcs`). These files are unreachable at runtime (registry has no entries) but their `_METHOD_PHASES` registrations point to nonexistent methods that would `AttributeError` if accidentally routed. Each is hard severity because the dangling code is a latent bug.

Live prompts (`phase3_intersections`, `phase4a_scene_types`, `phase8c_overlays`, `phase4g_transition_gaps`) have soft findings concentrated around Valid IDs formatting (unstructured flat lists vs grouped) and the repair-loop gap in `transition_gaps`. Cosmetic naming drift across multiple files where the spec's current 4a/4b/4c numbering is not reflected in template filenames.

- Prompts audited: 8
- Hard findings: 3 (all dead-code prompt files for migrated/removed phases)
- Soft findings: 8 (live-prompt quality findings)
- Info findings: 5 (numbering drift × 3 + minor)
- Spec gaps surfaced: 0
- Recommended PR split: PR-A (delete 3 dead-code prompt files + clean up `_METHOD_PHASES`/`_PREDECESSOR_PHASES`); PR-B (live-prompt soft findings: Valid IDs grouping, `transition_gaps` repair slot, `IntersectionProposal.resolved_location` type fix, template renames)
- Status: drift

---

## POLISH

### `polish_phase1_reorder.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — No GOOD/BAD examples for beat ordering rationale. The prompt has excellent rules, but small models producing a rationale string for `ReorderedSection.rationale` may emit vague text ("moved beats around") that doesn't help validation. The CLAUDE.md §7 defensive pattern requires concrete examples for every constrained field.
  - Where: lines 8–18 (Goals + Constraints) — no inline example pair for `rationale`
  - Spec citation: `CLAUDE.md §7`; `polish.md §R-1.4` (invalid proposals must log WARNING — the rationale is what reviewers inspect)
  - Recommended fix: Add:
    ```
    GOOD rationale: "Moved reflection beat before discovery to follow scene-sequel rhythm — action scene then processing beat"
    BAD rationale: "Better order" (too vague to diagnose if WARNING fires)
    ```

- **[soft] [sm-fragile]** — Valid IDs injected as flat comma-separated string `{valid_beat_ids}`. For sections with 8–15 beats, small models lose track of which IDs are legal and may introduce beat IDs from adjacent sections seen earlier in the prompt assembly. The CLAUDE.md §6 Valid ID Injection principle requires this be unambiguous.
  - Where: line 31 (`Valid beat_ids (use ONLY these, in any order): {valid_beat_ids}`)
  - Spec citation: `CLAUDE.md §6`; `polish.md §R-1.2` (reordered list MUST be same set as input)
  - Recommended fix: Inject as one-per-line bullet list:
    ```
    Valid beat_ids (use ONLY these, in any order):
    {valid_beat_ids_bulleted}
    ```
    and have the context builder format the list with leading `  - ` per ID.

- **[soft] [sm-fragile]** — `{before_context}` and `{after_context}` context blocks have no headers explaining their purpose; a 4B model may treat them as beats to include in the reordering list.
  - Where: lines 22–23 (`{before_context}`, `{after_context}`)
  - Spec citation: `CLAUDE.md §8 Context Enrichment` (every block must have a header explaining WHAT the data is and WHY it's provided)
  - Recommended fix: Wrap in labeled headers:
    ```
    ### Preceding context (NOT part of this section — do not include in output):
    {before_context}

    ### Following context (NOT part of this section — do not include in output):
    {after_context}
    ```

- **[info] [schema-skew]** — Prompt returns `reordered_sections` array with zero or one entry per call (since one section is passed per call), which aligns with `Phase1Output.reordered_sections: list[ReorderedSection]`. Clean match. No action.

---

### `polish_phase1a_narrative_gaps.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [drift]** — Gap beats are instructed to carry `dilemma_impacts` including effects `advances / reveals / commits / complicates`. This directly contradicts R-1a.2: **gap beats carry zero `dilemma_impacts`**. The schema example even demonstrates a gap beat with a `dilemma_impacts` entry. A model following this prompt produces structural-beat invariant violations on every non-empty gap-beat proposal.
  - Where: lines 62–84 (output schema describing `dilemma_impacts` field; example at lines 71–84 showing a gap beat with `dilemma_impacts`)
  - Spec citation: `polish.md §R-1a.2` ("Gap beats carry zero `dilemma_impacts`. They are structural transition beats; they MUST NOT advance any dilemma."); `story-graph-ontology.md §Part 1 Structural Beats`
  - Recommended fix: Remove the `dilemma_impacts` field from the output schema and the example entirely. Replace with an explicit prohibition:
    ```
    ## CRITICAL
    Gap beats are STRUCTURAL beats. They MUST have:
      dilemma_impacts: []   ← always empty; NEVER add dilemma impacts
    BAD: {{"dilemma_impacts": [{{"dilemma_id": "...", "effect": "advances"}}]}}
    GOOD: {{"dilemma_impacts": []}}
    ```

- **[hard] [drift]** — Phase identifier mismatch: the `description` field says "POLISH Phase 1a" which is correct, but the system prompt never identifies itself as POLISH — only as a tool "analyzing path beat sequences." An LLM receiving this in a chain may not apply POLISH-specific constraints (e.g., may apply GROW reasoning about dilemma impacts). This is absorbed-from-GROW work and the identity drift matters.
  - Where: line 1 (`description`), lines 3–4 (system header — no POLISH attribution)
  - Spec citation: `polish.md §Phase 1a: Narrative Gap Insertion` ("Absorbed from old GROW 4b per audit Q1 resolution")
  - Recommended fix: Change system opener to: "You are the POLISH stage (Phase 1a: Narrative Gap Insertion), analyzing path beat sequences to find narrative gaps. This is prose-craft work, NOT structural dilemma construction."

- **[hard] [schema-skew]** — The output schema omits R-1a.1's required gap-beat annotation fields: `is_gap_beat: True`, `role: "gap_beat"`, and `created_by: "POLISH"`. These are set by code post-validation, but the LLM schema should not carry fields that contradict the spec (specifically `dilemma_impacts` — see above) while silently omitting the ones that ARE required. At minimum the prompt should note what the code adds.
  - Where: lines 54–84 (output format section)
  - Spec citation: `polish.md §R-1a.1`; `story-graph-ontology.md §Part 1 Gap beat`
  - Recommended fix: Add a note block after the output format:
    ```
    Note: Code automatically adds `is_gap_beat: true`, `role: "gap_beat"`, `created_by: "POLISH"`.
    You do NOT need to include these — but you MUST NOT include `dilemma_impacts`.
    ```

- **[soft] [sm-fragile]** — Valid IDs injected as flat comma-separated strings for all three ID types (`valid_path_ids`, `valid_beat_ids`, `valid_dilemma_ids`). Three separate flat lists that reference each other create high cross-list confusion risk for small models.
  - Where: lines 48–51 (Valid IDs section)
  - Spec citation: `CLAUDE.md §6`
  - Recommended fix: Group beat IDs by path and list dilemma IDs separately; makes the path→beat relationship explicit:
    ```
    Valid path_ids and their beats:
      path::dilemma_a__answer_x → beat::ax1, beat::ax2, beat::ax3
      path::dilemma_b__answer_y → beat::by1, beat::by2
    Valid dilemma_ids (for path→dilemma mapping ONLY):
      dilemma::a, dilemma::b
    ```

- **[soft] [sm-fragile]** — Per-path cap R-1a.4 (max 2 gap beats per path) is mentioned in the checklist (line 31: "Maximum 2 gap beats per path") and in the user turn reminder (line 91: "Maximum 2 gap beats per path"), but there is no GOOD/BAD example showing what happens when 3 are proposed. The CLAUDE.md §7 defensive pattern requires an inline example pair for every hard constraint.
  - Where: line 31 and line 91
  - Spec citation: `polish.md §R-1a.4`; `CLAUDE.md §7`
  - Recommended fix: Add:
    ```
    GOOD (within cap): Two gap beats on path::dilemma__answer — one bridging setup→escalation, one bridging escalation→climax
    BAD (exceeds cap): Three gap beats on the same path — reduce to 2 most impactful
    ```

- **[info] [schema-skew]** — `scene_type` field in output schema allows `"micro_beat"` as a value. The ontology defines `scene_type ∈ {scene, sequel, micro_beat}` for annotation purposes (Part 1: Beat Annotations), but gap beats created by Phase 1a should have `scene_type` set to `scene` or `sequel` — the spec notes "default to sequel type for gap beats." The prompt's checklist (line 28) says the same. The schema allowing `micro_beat` is marginally confusing but not a functional problem. No action required beyond noting.

---

### `polish_phase2_pacing.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [drift]** — Pacing-run detection (R-2.6/R-2.7) is entirely absent from this prompt. The spec (R-2.6) requires Phase 2 to "detect runs of 3+ consecutive same-`scene_type` beats per path and insert correction beats of the opposite type." The prompt only handles pacing-flags (consecutive_scene, consecutive_sequel, no_sequel_after_commit) as injected detected issues — it does not instruct the LLM about run detection at all. If run detection is expected to happen in this LLM call, the prompt will not produce it.
  - Where: lines 22–31 (Guidelines section — no mention of R-2.6 run detection or R-2.7 correction-beat `is_gap_beat: True`)
  - Spec citation: `polish.md §R-2.6`; `polish.md §R-2.7`; `polish.md §Phase 2 §Pacing-Run Detection`
  - Recommended fix (pending clarification): If run detection is deterministic (code-side detection, not LLM) and only the correction beat *content* is LLM-generated via the same call, the `{pacing_issues}` context should include `type: "pacing_run"` entries alongside `consecutive_scene` entries, and the Guidelines should add: "For `pacing_run`: insert a correction beat of the OPPOSITE `scene_type` to break the run." If run detection is entirely code-side and content is not LLM-generated, the prompt is fine as-is (info only). Clarify which path is intended, then update accordingly.

- **[soft] [sm-fragile]** — `{pacing_issues}` placeholder has no schema description — the LLM receives injected text with unknown structure. Small models that receive an unexpected pacing issue format will mis-classify or ignore entries.
  - Where: line 20 (`## Pacing Issues Detected` section — only `{pacing_issues}` with no schema example)
  - Spec citation: `CLAUDE.md §8`; `CLAUDE.md §9` (every context block must have a header explaining what the data is)
  - Recommended fix: Add a schema note:
    ```
    Each issue follows this format:
    - type: consecutive_scene | consecutive_sequel | no_sequel_after_commit
    - after_beat: <beat_id after which to insert the micro-beat>
    - context: <brief description of surrounding beats>
    ```

- **[soft] [sm-fragile]** — No GOOD/BAD examples for the `summary` field of micro-beats. The prompt provides three illustrative examples (lines 11–14) but they are under the `## What Are Micro-beats?` header, not under a GOOD/BAD label. Small models conflate illustrative text with rules. The CLAUDE.md §7 pattern requires explicit GOOD/BAD labeling.
  - Where: lines 11–14 (example micro-beats)
  - Spec citation: `CLAUDE.md §7`
  - Recommended fix: Relabel as:
    ```
    GOOD (environment-focused, one sentence):
    - "A moment of silence falls over the study"
    - "The sound of distant thunder rolls across the valley"
    BAD (advances dilemma or adds characters):
    - "Mentor reveals they have been watching protagonist all along" (plot advancement)
    - "A stranger enters the room and speaks" (new entity — not from surrounding beats)
    ```

- **[soft] [repair-gap]** — No repair-loop slot. Phase 2 can produce micro-beats with invalid `entity_ids` (not in `valid_entity_ids`). There is no `{pacing_feedback}` template variable for retry error messages. A failed validation on retry will only have the generic system prompt as context, which is suboptimal for small models.
  - Where: entire file — no `{pacing_feedback}` variable
  - Spec citation: `CLAUDE.md §5 Repair-loop blindness` (repair feedback must be self-contained); `CLAUDE.md §Validation & Repair Loop`
  - Recommended fix: Add `{pacing_feedback}` slot to the user turn: "If this is a retry, correction needed: `{pacing_feedback}`". Default to empty string when no retry.

- **[info] [schema-skew]** — Pydantic `MicroBeatProposal` has no `scene_type` field, only `after_beat_id`, `summary`, `entity_ids`. The prompt also omits `scene_type` from the output. This is internally consistent. No action.

---

### `polish_phase3_arcs.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{anchored_dilemmas}` and `{overlay_data}` context variables are interpolated with no structural schema hint. Per CLAUDE.md §9, every injected variable must be explicitly formatted; per §8, every context block must explain what the data is and why it is provided. A Python list repr or dataclass repr leaking here would silently harm context quality.
  - Where: lines 12–15 (`Entity` section, `{anchored_dilemmas}` and `{overlay_data}`)
  - Spec citation: `CLAUDE.md §8`, `CLAUDE.md §9`
  - Recommended fix: Add inline schema stubs:
    ```
    Central to dilemmas (format: "dilemma_id: question"):
    {anchored_dilemmas}

    Entity overlays (format: "when [flag_id]: [property changes]"):
    {overlay_data}
    ```
    and confirm the context builder emits human-readable text in these shapes.

- **[soft] [sm-fragile]** — `{path_ids}` listed as a bare variable with the label "## Paths in This Story" — no context explaining what paths ARE relative to this entity. Small models receiving a list of path IDs without knowing which paths this entity appears on may write arcs for all paths even when the entity is absent from some.
  - Where: line 22 (`{path_ids}`)
  - Spec citation: `CLAUDE.md §8 Context Enrichment` (include all ontologically relevant fields)
  - Recommended fix: Have the context builder filter to paths where the entity has beat appearances, and label them:
    ```
    Paths on which this entity appears (generate arcs ONLY for these):
    {entity_relevant_path_ids}
    ```

- **[soft] [sm-fragile]** — `{beat_appearances}` header says "in story order" but gives no structural schema hint for what each entry looks like. R-3.5 requires "full context for each entity: beat summaries in order, dilemma questions, path descriptions, overlay details." If the context builder emits bare beat IDs, R-3.5 is violated.
  - Where: line 18 (`## Beats Featuring This Entity (in story order)`)
  - Spec citation: `polish.md §R-3.5`; `CLAUDE.md §8`
  - Recommended fix: Add schema stub: `Format per beat: "[beat_id] ([path_id]) — [summary] — scene_type: [scene|sequel] — exit_mood: [mood]"`

- **[soft] [sm-fragile]** — The constraint that `pivot_beat` MUST equal the matching `pivots[path_id].beat_id` (R-3.8) is stated in the "What NOT to Do" section (line 92) but the working example in the output format section (lines 49–87) already correctly demonstrates this. However, the rule reads as a negative prohibition rather than a positive contract. Small models weight positive examples over negative prohibitions.
  - Where: line 92 (`- Do NOT use a different pivot_beat than the matching pivots[path_id] entry`)
  - Spec citation: `polish.md §R-3.8`
  - Recommended fix: Move the consistency constraint to the `## Arc Structure` section before the example as a positive rule: "**Consistency (REQUIRED):** For each path, `pivots[*].beat_id` and `arcs_per_path[*].pivot_beat` MUST be the SAME beat_id. Pick one beat per path and use it in both fields."

- **[info] [schema-skew]** — `arc_type` placeholder approach (`"_set_by_code"`) is clearly documented in the prompt (line 39) and the Pydantic model `PerPathArc` docstring. The model accepts any string (code overwrites). Clean alignment. No action.

- **[info] [sm-fragile]** — Per-entity call pattern (one entity per call) matches the procedure spec's "constrained ~32k context" note. The scoped approach avoids context overflow. No action.

---

### `polish_phase5a_choice_labels.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — No `### Valid IDs` section. The LLM must produce `from_passage` and `to_passage` IDs that exactly match the `ChoiceSpec.from_passage` / `ChoiceSpec.to_passage` values computed by Phase 4c. Without an explicit valid passage ID list, small models invent or mangle passage IDs, causing Phase 6 to fail to wire choice edges.
  - Where: entire prompt — no valid passage ID section
  - Spec citation: `CLAUDE.md §6 Valid ID Injection Principle`; `polish.md §Implementation Constraints §Valid ID Injection`
  - Recommended fix: Add a `## Valid Passage IDs` section injected from the ChoiceSpec list:
    ```
    ## Valid Passage IDs (use ONLY these exact strings)
    {valid_from_passage_ids}
    {valid_to_passage_ids}
    ```
    and update "What NOT to Do" to include: "Do NOT invent or modify passage IDs — use only the IDs listed above."

- **[soft] [sm-fragile]** — `{story_context}` is injected without a schema header explaining what it contains. CLAUDE.md §8 requires every context block to have an explanatory header.
  - Where: line 16 (`{story_context}`)
  - Spec citation: `CLAUDE.md §8`
  - Recommended fix: Replace `{story_context}` with:
    ```
    ## Story Background (for tone and diegetic voice reference)
    {story_context}
    ```

- **[soft] [sm-fragile]** — `{choice_details}` contains the ChoiceSpec data but the prompt gives no schema description. The LLM doesn't know if `from_passage` is a passage summary or an ID. If `choice_details` contains bare passage IDs, the model has insufficient context (violating R-5.4: "full context for each choice: source passage summary, target passage summary, surrounding beat summaries, active state flags, relevant dilemma question").
  - Where: lines 17–18 (`{choice_details}`)
  - Spec citation: `polish.md §R-5.4`; `CLAUDE.md §8`
  - Recommended fix: Add a schema stub:
    ```
    ## Choices to Label (one entry per choice)
    Format: "from_passage [passage_id]: [passage_summary] → to_passage [passage_id]: [passage_summary]"
    Active flags: [flag list]
    Dilemma question: [question text]
    {choice_details}
    ```
    Confirm the context builder emits this rich format.

- **[soft] [repair-gap]** — No repair-loop slot for label validation failures (e.g., non-diegetic label, labels not distinct within source passage per R-5.2). Retry will only have the static system prompt.
  - Where: entire file — no `{choice_label_feedback}` variable
  - Spec citation: `CLAUDE.md §Validation & Repair Loop`; `polish.md §R-5.1`, `§R-5.2`
  - Recommended fix: Add `{choice_label_feedback}` to user turn (default empty).

- **[info] [schema-skew]** — Output schema uses `from_passage` and `to_passage` matching `ChoiceLabelItem` Pydantic fields exactly. Clean. No action.

---

### `polish_phase5b_residue.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{residue_details}` injected without a schema header. The model does not know what fields each residue entry carries (passage summary, flag context, path description). R-5.5 requires one variant per path; the LLM cannot produce path-specific content without knowing which paths / flags map to which residue.
  - Where: line 18 (`{residue_details}`)
  - Spec citation: `polish.md §R-5.5`; `CLAUDE.md §8`
  - Recommended fix: Add schema stub:
    ```
    ## Residue Beats to Write
    Format per residue:
    residue_id: [id]
    target_passage: [summary of the shared scene that follows]
    active_flag: [state_flag_id] (this path's flag — set if player chose [path_description])
    other_flag: [sibling_flag_id] (absent for this variant)
    {residue_details}
    ```

- **[soft] [sm-fragile]** — No GOOD/BAD example for `content_hint` length. R-5.6 requires "brief — a mood-setter, not a full scene." The prompt says "1-2 sentences max" which is a correct constraint, but CLAUDE.md §7 requires an explicit GOOD/BAD example pair.
  - Where: lines 7–14 (Goals section)
  - Spec citation: `polish.md §R-5.6`; `CLAUDE.md §7`
  - Recommended fix: Add:
    ```
    GOOD: "You enter the vault with quiet confidence, the mentor's warning still fresh"
    BAD: "You spent years preparing for this moment. The vault door is cold to the touch. Behind you, the city sleeps, unaware of what you're about to uncover..." (too long — full passage, not mood-setter)
    ```

- **[soft] [sm-fragile]** — No repair-loop slot. `mapping_strategy` validation failure (model produces a value outside `{residue_passage_with_variants, parallel_passages}`) has no mechanism for targeted feedback on retry.
  - Where: entire file — no `{residue_feedback}` variable
  - Spec citation: `CLAUDE.md §Validation & Repair Loop`
  - Recommended fix: Add `{residue_feedback}` to user turn (default empty). Include in repair: "`mapping_strategy` must be exactly `residue_passage_with_variants` or `parallel_passages`."

- **[info] [schema-skew]** — `mapping_strategy` Literal values in the prompt match the `ResidueContentItem.mapping_strategy` Pydantic Literal exactly. `residue_id` output field matches `ResidueContentItem.residue_id`. Clean. No action.

---

### `polish_phase5c_false_branches.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{candidate_details}` injected without schema description. The model doesn't know what a candidate stretch looks like (passage IDs, beat summaries, pacing context). R-4d.3 requires "surrounding context: passage IDs, beat summaries, entity references, pacing annotations." Without this schema hint, the model cannot make an informed skip/diamond/sidetrack decision.
  - Where: line 16 (`{candidate_details}`)
  - Spec citation: `polish.md §R-4d.3`; `CLAUDE.md §8`
  - Recommended fix: Add schema stub:
    ```
    ## Candidate Stretches (one entry per linear passage run)
    Format:
    Candidate [N]: passages [id1 → id2 → id3], [beat count] beats total
    Beat summaries: [brief per-beat summaries]
    Entity context: [entities in these passages]
    Pacing annotation: [exit_mood transitions]
    {candidate_details}
    ```

- **[soft] [sm-fragile]** — No constraint preventing false branches at dilemma commit beats (R-5.12). This is a critical rule: "False branches never affect dilemma-driven branching — they sit within linear sections or as cosmetic fork-rejoin structures." A model unaware of this could propose a diamond or sidetrack at a real branching point.
  - Where: "What NOT to Do" section (lines 48–52) — no mention of R-5.12
  - Spec citation: `polish.md §R-5.12`
  - Recommended fix: Add to "What NOT to Do":
    ```
    - Do NOT propose diamond or sidetrack at a passage that ends with a real dilemma choice
      (the candidate_details will mark dilemma-choice passages — never place false branches there)
    ```

- **[soft] [sm-fragile]** — No GOOD/BAD examples for the `skip` decision. The prompt says "Prefer 'skip' unless the stretch genuinely needs more player engagement" (user turn, line 56), but provides no example of what makes a stretch worth skipping vs. acting on. Without a concrete bad example, small models default to over-applying diamond/sidetrack.
  - Where: user turn lines 54–58
  - Spec citation: `CLAUDE.md §7`
  - Recommended fix: Add in the user turn:
    ```
    GOOD reason to skip: "Passage run is naturally tense — forcing a choice here would interrupt the momentum"
    GOOD reason for diamond: "Introductory scene works equally well from two sensory angles"
    GOOD reason for sidetrack: "Stretch is 5 passages with no narrative texture — a brief encounter adds atmosphere"
    ```

- **[info] [schema-skew]** — `candidate_index` (0-based) in `FalseBranchDecisionItem` matches the prompt's "Candidate [N]" format. The off-by-one (0-based vs 1-based labeling) is a minor risk but manageable if context builder emits 0-based labels. Worth verifying at integration time.

---

### `polish_phase5d_variants.yaml`

**Verdict:** clean

**Findings:**

- **[soft] [sm-fragile]** — `{variant_details}` injected without schema description. The model needs to know what active state flags are associated with each variant to write meaningfully different summaries. Without the flag context, all variants risk being generic.
  - Where: line 18 (`{variant_details}`)
  - Spec citation: `polish.md §R-5.13` (distinct summary reflecting its flag combination); `CLAUDE.md §8`
  - Recommended fix: Add schema stub:
    ```
    ## Variants to Summarize
    Format per variant:
    variant_id: [id]
    base_passage: [base passage summary]
    active_flags: [flag_id_1] (player took [path description])
    {variant_details}
    ```

- **[info] [schema-skew]** — `variant_id` and `summary` output fields match `VariantSummaryItem` Pydantic exactly. No action.

---

### `polish_phase5e_atmospheric.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{narrative_frame}` injected without a schema label or header. If this contains the story's genre/tone as a string, a small model may conflate it with the beat list that follows.
  - Where: line 8 (`{narrative_frame}`)
  - Spec citation: `CLAUDE.md §8`, `CLAUDE.md §9`
  - Recommended fix: Add header: `## Story Frame (genre and tone, for sensory register reference)` before the variable.

- **[soft] [sm-fragile]** — Partial coverage detection (R-5e.1) cannot happen from this prompt alone — the prompt instructs the model to produce details for "ALL beats" but doesn't explain what the model should do if it is uncertain about a beat. The contract between LLM output and code-side WARNING detection should be clarified.
  - Where: user turn line 38 (`Every beat needs an atmospheric_detail`) — no fallback instruction for uncertain beats
  - Spec citation: `polish.md §R-5e.1` (partial coverage MUST log a WARNING)
  - Recommended fix: Add to user turn: "If you are uncertain about the sensory environment for a beat, write a generic sensory anchor rather than omitting it (e.g., 'Dim ambient light and the faint sound of settling wood'). Omitting a beat is worse than a generic detail."

- **[soft] [sm-fragile]** — No GOOD/BAD example differentiating environment from interiority (R-5e.2). The illustrative examples (lines 13–16) are good but are not labeled GOOD. The prohibition "Write ENVIRONMENT, not character emotion" (line 17) is present but lacks a BAD counterexample.
  - Where: lines 13–17
  - Spec citation: `polish.md §R-5e.2`; `CLAUDE.md §7`
  - Recommended fix: Add:
    ```
    BAD: "A sense of dread hangs in the air" (character interiority, not environment)
    BAD: "The protagonist feels cold and alone" (character emotion)
    GOOD: "The smell of wet earth and rust" (environment)
    GOOD: "Cold steel handrails vibrating faintly underfoot" (physical sensation, environment)
    ```

- **[info] [schema-skew]** — Output format returns `details: [{beat_id, atmospheric_detail}]` array. The Pydantic model for Phase 5e output is not defined in `polish.py` as a named model — but the schema-level check (beat IDs exist, no extra IDs) is handled by code. The prompt matches the expected JSON shape. No action.

---

### `polish_phase5e_feasibility.yaml`

**Verdict:** clean

**Findings:**

- **[soft] [sm-fragile]** — `{case_details}` injected without a schema description. The model doesn't know what the `[0]`, `[1]` labels represent without an explicit schema header.
  - Where: line 22 (`{case_details}`)
  - Spec citation: `CLAUDE.md §8`
  - Recommended fix: Add schema stub:
    ```
    ## Cases to Resolve
    Format per case:
    Passage: [passage_id] — [passage_summary]
    Entities in passage: [entity list]
    Flags to decide:
      [0] flag_id: [flag description] — weight: heavy|light — dilemma: [question]
      [1] flag_id: [flag description] — weight: heavy|light — dilemma: [question]
    {case_details}
    ```
    The `[0]`, `[1]` label convention is referenced in the output format (`flag_index`) but only explained in the output schema. Move the explanation to the cases section.

- **[soft] [sm-fragile]** — Decision values `"variant" / "residue" / "irrelevant"` are stated in the task description but the user-turn reminder (lines 50–55) doesn't echo them. Small models completing a long list of flags may drift toward free-form values.
  - Where: user turn lines 50–55
  - Spec citation: `CLAUDE.md §10`
  - Recommended fix: Add to user turn: "REMINDER: each decision MUST be exactly one of: `variant`, `residue`, or `irrelevant` — no other values accepted."

- **[info] [schema-skew]** — `FeasibilityDecisionItem` Pydantic model has `decision: str` (not a Literal). The code validates against `{"variant", "residue", "irrelevant"}` post-parse. Prompt correctly names the three values. Alignment is good.

---

### `polish_phase5f_path_thematic.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{entity_arcs}` injected without a schema label clarifying it contains POLISH Phase 3 output. A small model may treat it as general character descriptions rather than POLISH-synthesized arc metadata. The context builder must ensure R-3.6 `arcs_per_path` data is in this injection.
  - Where: line 19 (`{entity_arcs}`)
  - Spec citation: `polish.md §R-5f.2` (LLM consumes full beat sequence with summaries, scene types, narrative functions, exit moods); `CLAUDE.md §8`
  - Recommended fix: Relabel: `## Character Arcs on This Path (synthesized by POLISH Phase 3)` and add: "Use these to understand how character trajectories influence the path's emotional through-line."

- **[soft] [sm-fragile]** — `{beat_sequence}` injected without a schema description. R-5f specification says the LLM should consume "the full beat sequence with their summaries, scene types, narrative functions, and exit moods." If the context builder emits bare beat IDs or only summaries, the rich context required by the spec is missing.
  - Where: line 22 (`{beat_sequence}`)
  - Spec citation: `polish.md §R-5f` (operations description); `CLAUDE.md §8`
  - Recommended fix: Add schema stub: `Format per beat: "[beat_id] — [summary] — scene_type: [scene|sequel] — narrative_function: [introduce|develop|...] — exit_mood: [mood]"`

- **[soft] [drift]** — Phase identity: this is POLISH Phase 5f but the system prompt says "narrative architect analyzing a single path" with no POLISH attribution. Unlike the Phase 1a case this is less critical (the instructions are narrow and focused), but adding POLISH attribution aligns with the absorbed-from-GROW traceability intent.
  - Where: lines 3–4 (system header)
  - Spec citation: `polish.md §Phase 5f — Path Thematic Annotation` ("Absorbed from old GROW 4e per audit Q1")
  - Recommended fix: Add: "You are the POLISH stage (Phase 5f: Path Thematic Annotation), synthesizing the emotional through-line of a single story path."

- **[info] [schema-skew]** — Output fields `path_id`, `path_theme` (10–200 chars), `path_mood` (2–50 chars) match `story-graph-ontology.md §Part 1 Path Annotations` and `polish.md §R-5f.1`/`R-5f.2` exactly. The length constraints are enforced in the prompt with explicit character examples and a "CRITICAL: count characters manually" warning. This is the best small-model-resilience implementation in all 12 POLISH prompts.

---

### `polish_phase5f_transitions.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — `{collapsed_passage_details}` injected without a schema description. The model needs to know what "beat boundaries" look like — how they're listed in the injected context — to count them correctly. If the context builder emits passage summaries without explicit boundary markers, the model will miscount.
  - Where: line 36 (`{collapsed_passage_details}`)
  - Spec citation: `CLAUDE.md §8`; `polish.md §Phase 5f` (transitions sub-task)
  - Recommended fix: Add schema stub:
    ```
    ## Collapsed Passages to Process
    Format per passage:
    passage_id: [id] — [N] beats → [N-1] boundaries needed
    Beat 1: [summary]
    Boundary 1 ←
    Beat 2: [summary]
    Boundary 2 ←
    Beat 3: [summary]
    {collapsed_passage_details}
    ```

- **[soft] [sm-fragile]** — The transition count constraint ("exactly N-1 transitions for N beats") is stated at lines 10–13 and in the user turn (line 56), but the BAD examples (lines 30–33) include "Transition smoothly" and "Write a transition here" without showing an example of wrong-count output. A small model producing too many or too few transitions is a common failure mode that benefits from a count-violation example.
  - Where: lines 30–33 (Bad Transition Examples)
  - Spec citation: `CLAUDE.md §7`
  - Recommended fix: Add to "What NOT to Do":
    ```
    - Do NOT provide 3 transitions for a 3-beat passage (3 beats = 2 boundaries = 2 transitions, not 3)
    GOOD (2-beat passage = 1 transition): {{"transitions": ["The silence stretches until the protagonist turns away."]}}
    BAD (2-beat passage = 2 transitions — too many): {{"transitions": ["...", "..."]}}
    ```

- **[soft] [schema-skew]** — `polish_phase5f_transitions.yaml` is named "5f_transitions" but in the POLISH procedure spec, transition guidance is part of Phase 5f alongside path thematic annotation, not a separate phase. The Pydantic model is `Phase5fOutput` with `TransitionGuidanceItem` — which correctly lumps them together. The prompt file name implies a separate sub-phase that doesn't exist in the spec. This causes navigational confusion when reading the procedure doc alongside the prompt file.
  - Where: prompt `name` field (line 1: `polish_phase5f_transitions`) vs `polish.md §Phase 5` (5f is Path Thematic Annotation; transitions are a sub-task of Phase 5)
  - Spec citation: `polish.md §Phase 5f` and `§Phase 5 Output Contract item 6`; `src/questfoundry/models/polish.py:420–435`
  - Recommended fix: The filename is `5f_transitions` and the other file is `5f_path_thematic`. This correctly splits two LLM calls within Phase 5f. The disconnect is that `polish.md §Phase 5f` does not mention transition guidance as a sub-task — it is implied by `PassageSpec.transition_guidance` and the output contract. No file rename needed; spec should be updated to explicitly list transition guidance as a Phase 5f sub-task.
  - **spec-gap flag:** `polish.md §Phase 5f` should be updated to include transition guidance as a named sub-task alongside path thematic annotation. Recommend spec edit before any code change.

- **[info] [schema-skew]** — `TransitionGuidanceItem.transitions: list[str]` with `min_length=1` in Pydantic. A single-boundary passage returns one-element list; the prompt correctly handles this. Output field `passage_id` and `transitions` match Pydantic exactly. No action.

---

### Stage summary: POLISH

POLISH is the largest prompt set (12 files covering 9 distinct phases/sub-phases), and it absorbed 5 phases from GROW in epic #1368. That migration created two classes of problems:

**Class 1 — Migration residue (absorbed-from-GROW prompts):** `phase1a_narrative_gaps` still teaches the model to produce `dilemma_impacts` in gap beats, directly violating R-1a.2 (structural-beat invariant). This is the single hardest finding in the stage. `phase2_pacing` is silent on pacing-run detection (R-2.6/R-2.7), leaving it ambiguous whether run correction content is LLM-generated via this call or entirely code-side. `phase3_arcs` handles the `arcs_per_path` consolidation correctly (the `_set_by_code` placeholder pattern is well-documented), but the context-enrichment deficits in `{anchored_dilemmas}` and `{beat_appearances}` risk violating R-3.5.

**Class 2 — Valid ID gaps across Phase 5 prompts:** `phase5a_choice_labels` has no `### Valid IDs` section at all, violating the CLAUDE.md §6 mandate and opening the door to phantom passage IDs in Phase 6's atomic application. All four remaining Phase 5 prompts (5b, 5c, 5d, 5e) lack schema descriptions for their main context injection variables (`{residue_details}`, `{candidate_details}`, `{variant_details}`, `{beat_summaries}`), making R-5.4/R-5.5/R-5.13 context-enrichment requirements effectively unverifiable without inspecting the context builders.

**Migrated phases that are correct:** `phase5f_path_thematic` is the best-written prompt in POLISH — explicit length constraints with character counts, concrete bad examples, and CRITICAL emphasis. `phase1_reorder` and `phase5d_variants` are mostly clean with only soft findings. `phase5e_feasibility` has a modest schema description gap but its decision-value enumeration and example are correct.

**Spec gap:** `polish.md §Phase 5f` does not name transition guidance as a sub-task; the `phase5f_transitions.yaml` prompt exists and the Pydantic output model is `Phase5fOutput`, but the procedure doc needs updating to make this explicit.

- Prompts audited: 12
- Hard findings: 4 (`phase1a` dilemma_impacts violation ×2, `phase1a` phase identity drift, `phase5a` missing Valid IDs)
- Soft findings: 22 (context injection schema gaps across Phase 5; repair-loop slot absences; GOOD/BAD example gaps; pacing-run detection gap in Phase 2; identity drift in Phase 1a and Phase 5f)
- Info findings: 7
- Spec gaps surfaced: 1 (`polish.md §Phase 5f` missing transition guidance sub-task — recommend spec edit before PR)
- Recommended PR split: PR-A (hard findings: `phase1a` dilemma_impacts removal + phase identity fix + `phase5a` Valid IDs injection); PR-B (Phase 5 context enrichment: schema stubs for `{residue_details}`, `{candidate_details}`, `{variant_details}`, `{beat_summaries}` + repair-loop slots); PR-C (soft/info: GOOD/BAD examples, sandwich repetitions, R-5.12 false-branch constraint, pacing-run R-2.6 clarification after code investigation)
- Status: drift

---

## FILL

### `prompts/templates/fill_phase0_discuss.yaml`

**Verdict:** mixed

**Findings:**

- **[info] [terminology]** — Pipeline stage counter says "5 of 6: DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP"; the pipeline has 8 stages and POLISH (which precedes FILL) is omitted from the chain.
  - Where: system block, line 10
  - Spec citation: `CLAUDE.md §Architecture` (six stages listed as DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP, but POLISH and DRESS are authoritative stages per `docs/design/procedures/`); `fill.md §Stage Input Contract` ("Must match POLISH §Stage Output Contract exactly")
  - Recommended fix: Update to "Stage: FILL (6 of 8: DREAM → BRAINSTORM → SEED → GROW → POLISH → FILL → DRESS → SHIP)"

- **[soft] [schema-skew]** — POV option labels use the short forms `first, second, third_limited, third_omniscient` (matching `VoiceDocument.pov` Pydantic Literal) but the spec uses the long forms `first_person, second_person, third_person_limited, third_person_omniscient` (fill.md R-1.3) — and the downstream `fill_phase0_voice.yaml` also uses the short forms. This creates an inconsistency that will confuse anyone reading spec vs. prompt, and it means the spec violation goes undetected because the Pydantic model itself uses the short forms.
  - Where: system block, line 51 ("1. **POV**: first, second, third_limited, or third_omniscient?")
  - Spec citation: `fill.md §R-1.3` (`pov` ∈ {`first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`}); `story-graph-ontology.md §Part 9` ("Voice Document | FILL | No | Prose contract: POV, tense, register, rhythm")
  - Recommended fix: Either (a) update `VoiceDocument.pov` in `fill.py` to use the canonical long-form Literals matching `dream.py` and fill.md R-1.3, then update both FILL prompts accordingly, **or** (b) update `fill.md` R-1.3 to reflect the short-form values the model actually validates. Per CLAUDE.md §Design Doc Authority, the spec wins — the Pydantic model should be updated. Note: `dream.py` already uses long forms (`first_person`, etc.) which confirms the canonical choice.

- **[info] [sm-fragile]** — The `search_corpus` requirement is stated twice in slightly different ways (lines 59-74 and later in "Craft Corpus Research (REQUIRED)"), which is good sandwich repetition, but the two calls to `search_corpus` lack explicit topic examples tailored to the stage. The corpus search instruction ("at least ONE of these topics") could be strengthened by requiring all three rather than one.
  - Where: system block, lines 63-75
  - Spec citation: `fill.md §Implementation Constraints §Small Model Prompt Bias`; `CLAUDE.md §10`
  - Recommended fix: Minor — change "at least ONE of these topics" to "at least two of these topics" or restructure into a numbered sequence. Low priority.

---

### `prompts/templates/fill_phase0_voice.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — POV Literal values in the schema description (`first, second, third_limited, third_omniscient`) diverge from the authoritative spec (`first_person, second_person, third_person_limited, third_person_omniscient` per `fill.md §R-1.3`). The model currently validates against the Pydantic model's short forms, so this is only `hard` if the Pydantic model is corrected per the spec (which it should be per CLAUDE.md §Design Doc Authority). If the spec value is corrected in the Pydantic model first, this prompt will produce values that fail validation against the updated model.
  - Where: system block, line 28 (`- **pov**: Point of view. One of: first, second, third_limited, third_omniscient`) and line 77 (`- Match POV/tense to genre conventions (fantasy → third_limited past; horror → second present)`)
  - Spec citation: `fill.md §R-1.3` (`pov` ∈ {`first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`})
  - Recommended fix: Once `VoiceDocument.pov` is corrected in the model, update line 28 to:
    ```
    - **pov**: Point of view. One of: first_person, second_person, third_person_limited, third_person_omniscient
    ```
    And update line 77's examples accordingly: `fantasy → third_person_limited past; horror → second_person present`.

- **[soft] [spec-gap]** — `pov_character` description says "required for first/third_limited, empty for omniscient/second." The spec (fill.md R-1.3) says "When `pov` is limited, `pov_character` names the POV entity" — which implies first-person also needs it (the POV character IS the first-person narrator). The description is ambiguous: does "first" (first-person) require it or not?
  - Where: system block, line 29
  - Spec citation: `fill.md §R-1.3`
  - Recommended fix: Update the spec to be explicit: "When `pov` is `first_person` or `third_person_limited`, `pov_character` names the POV character." Then update the prompt: "required for first_person and third_person_limited; empty string for second_person and third_person_omniscient."

- **[soft] [repair-gap]** — The repair loop (`_build_error_feedback`) that fires on `VoiceDocument` validation failure sends a generic "validation error" message plus the expected field paths, but does NOT echo the valid enum values for `pov` or `voice_register` or `sentence_rhythm`. A 4B model that outputs `pov: "third person limited"` or `voice_register: "terse"` will see "voice.pov: value is not a valid enumeration member" with no list of what IS valid.
  - Where: `src/questfoundry/pipeline/stages/fill.py` lines 678–689 (`_build_error_feedback`)
  - Spec citation: `CLAUDE.md §Repair-loop quality` ("repair feedback must be self-contained"); role file §Repair-loop blindness
  - Recommended fix: In `_build_error_feedback`, when `failure_type == "content"` and the field is an enum, include the allowed values. Example for a content retry message targeting `pov`:
    ```
    Field `voice.pov` has an invalid value.
    Valid values are: first_person, second_person, third_person_limited, third_person_omniscient
    ```
    Either enhance `_build_error_feedback` generically using Pydantic's JSON schema for Literal fields, or add a post-validation check that echoes allowed values for enum violations.

- **[info] [schema-skew]** — The worked example in `fill.md` line 451 shows `register 'atmospheric-terse'`, which is not a valid `voice_register` Literal (`formal | conversational | literary | sparse`). This is a spec inconsistency, not a prompt issue — the prompt correctly lists the Pydantic values. Flag for spec cleanup.
  - Where: `fill.md` Worked Example §Phase 1
  - Spec citation: `src/questfoundry/models/fill.py` line 44 (`voice_register: Literal["formal", "conversational", "literary", "sparse"]`)
  - Recommended fix: Update the worked example to use a valid value, e.g. `register: literary`.

---

### `prompts/templates/fill_phase1_expand.yaml`

**Verdict:** clean

**Findings:**

- **[info] [sm-fragile]** — `{craft_constraint_instruction}` is a variable injected at runtime for rule 5, which is good — but the user-message recap at lines 48–52 hard-codes "craft_constraint (copy from the passage details above, or empty string)" without echoing what constraint is active. For a small model, by the time it reads the user turn, the constraint was in rule 5 of the system turn (which may be far back in context).
  - Where: user block, line 51
  - Spec citation: `CLAUDE.md §10 §Small Model Prompt Bias`; role file §Required reading §Constraint-to-value mapping loss
  - Recommended fix: Change the user-turn reminder to: "craft_constraint: use the constraint from Rule 5 above (or empty string if none specified)" — this keeps the instruction forward-pointing without needing to re-echo the full constraint.

---

### `prompts/templates/fill_phase1_extract.yaml`

**Verdict:** clean

**Findings:**

- **[info] [sm-fragile]** — The distinction between universal and path-dependent details is explained in prose ("Only extract UNIVERSAL details — things true regardless of which path the player takes") with one good/bad example pair. A second concrete example contrasting a universal vs path-dependent edge case (e.g., physical injury vs injury on a specific path) would reduce false-positives on path-dependent extraction.
  - Where: system block, lines 23–26
  - Spec citation: `fill.md §R-2.12 / R-2.13` (micro-details must be universal; FILL cannot modify overlays); `CLAUDE.md §7 Defensive Prompt Patterns`
  - Recommended fix: Add one more GOOD/BAD pair:
    ```
    GOOD: "walks with a limp" (physical trait, arc-independent)
    BAD: "lost their weapon in the ambush" (event from a specific path)
    ```

---

### `prompts/templates/fill_phase1_prose.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [sm-fragile]** — "## Narrative Function Guidance" lists all five guidance entries (introduce/develop/complicate/confront/resolve) as static text, but the current passage's `narrative_function` is never explicitly injected as a variable. The model must infer which guidance applies from the `{beat_summary}` content. Small models often apply all five entries superficially or default to the first. The narrative function value is computed at runtime (`beat.get("narrative_function", "develop")`) but is not in the context dict passed to the template.
  - Where: system block, lines 37–48; context dict in `fill.py` lines 1345–1382 (no `narrative_function` key)
  - Spec citation: `fill.md §R-2.7` ("FILL generates prose per scene-type guidance: `scene` / `sequel` / `micro_beat`") — spec also references narrative function implicitly via ontology; `story-graph-ontology.md §Beat Annotations §Narrative function` ("Consumed by FILL for prose pacing")
  - Recommended fix: Add `narrative_function` to the context dict and inject it at the point of use:
    ```yaml
    ## Narrative Function: {narrative_function}
    ```
    Place this immediately after "**Scene Type:** {scene_type}" so both values are co-located. The full Narrative Function Guidance section remains below but the model now knows which entry applies.

- **[soft] [sm-fragile]** — The `{shadow_context}` placeholder is present (line 70), but the phrase "shadows (non-chosen answers)" is used in the spec Implementation Constraints without any setup in the prompt explaining what shadows are. If the context is non-empty, the model may not understand why it's receiving "shadow" content.
  - Where: system block, line 70 (`{shadow_context}`)
  - Spec citation: `fill.md §Implementation Constraints §Context Enrichment` ("shadows (non-chosen answers)")
  - Recommended fix: Add a header line before the placeholder:
    ```yaml
    ## Shadows (Non-Chosen Alternatives)
    {shadow_context}
    ```
    And format the shadow context with a brief header explaining "These are the paths the player did NOT take — use as subtext, not explicit content."

- **[info] [schema-skew]** — The output format description (lines 127–133) describes `spoke_labels` items as having `choice_id` and `label` with a note "(may omit 'choice::' prefix)". The `SpokeLabelUpdate` Pydantic model has `max_length=80` for `label` but the prompt says "3-60 characters". The documentation says 3-60 but the model allows up to 80.
  - Where: system block, lines 131–133
  - Spec citation: `src/questfoundry/models/fill.py` lines 85–89 (`SpokeLabelUpdate.label max_length=80`, description "3-60 chars")
  - Recommended fix: The Pydantic model's description already says "3-60 chars (e.g., 'Examine the sketch')" but `max_length=80` is the validator. Either tighten the validator to 60, or update the prompt description to say "3-80 characters (aim for 3-60)". The discrepancy is benign but misleading.

---

### `prompts/templates/fill_phase1_prose_only.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [spec-gap]** — `{shadow_context}` is completely absent from this template. The spec Implementation Constraints require shadows as mandatory context for prose generation: "shadows (non-chosen answers)" must be included. In two-step mode (`_two_step=True`), `fill_phase1_prose_only` is used and shadows are silently omitted even though they are in the context dict.
  - Where: entire template — no `{shadow_context}` placeholder
  - Spec citation: `fill.md §Implementation Constraints §Context Enrichment` ("shadows (non-chosen answers)" listed as mandatory); `fill.md §R-2.4` ("each LLM call receives full entity details — not just names but … active overlay state")
  - Recommended fix: Add the shadow section (matching `fill_phase1_prose.yaml` line 70):
    ```yaml
    ## Shadows (Non-Chosen Alternatives)
    {shadow_context}
    ```
    Place it after `## Path Arcs` and before the output section.

- **[soft] [sm-fragile]** — `fill_phase1_prose_only.yaml` lacks the "Narrative Function Guidance" section entirely (compare to `fill_phase1_prose.yaml` lines 37–48). Since two-step mode is used for prose generation (arguably the more demanding task), the model gets less structural guidance about dramatic purpose.
  - Where: entire template — compare with `fill_phase1_prose.yaml` lines 37–48
  - Spec citation: `story-graph-ontology.md §Beat Annotations §Narrative function` ("Consumed by FILL for prose pacing")
  - Recommended fix: Add the Narrative Function Guidance section (same content as `fill_phase1_prose.yaml`) and inject `{narrative_function}` as a variable pointing to the current passage's function value.

- **[soft] [sm-fragile]** — The user-message CRITICAL REMINDERS section says "Return ONLY the prose text. No JSON, no commentary, no preamble." but does not repeat the voice document constraints. The `fill_phase1_prose.yaml` user turn repeats "Follow the voice document EXACTLY — POV, tense, register, rhythm, tone." The prose-only variant omits this critical reminder, increasing voice drift risk.
  - Where: user block, lines 92–96
  - Spec citation: `fill.md §R-2.3` ("Each LLM call receives the Voice Document as mandatory context"); `CLAUDE.md §10 §Small Model Prompt Bias`
  - Recommended fix: Add to the user CRITICAL REMINDERS:
    ```
    - Follow the voice document EXACTLY — POV, tense, register, rhythm, tone
    - Match scene type guidance for structure and length
    - Cover beat summary content — do NOT invent major plot points
    ```

---

### `prompts/templates/fill_phase2_review.yaml`

**Verdict:** mixed

**Findings:**

- **[info] [schema-skew]** — The user-message reminder lists 7 valid `issue_type` values but the `ReviewFlag.issue_type` Pydantic Literal has 10 values (adding `near_duplicate`, `opening_trigram`, `low_vocabulary`). This is intentional — the three additional types are generated by the deterministic quality gate phase, not by this LLM prompt. However, if the deterministic flags ever flow through the LLM review path (e.g., combined batch), the model would not know these types exist.
  - Where: user block, line 47–48
  - Spec citation: `src/questfoundry/models/fill.py` lines 178–190 (`ReviewFlag.issue_type` Literal)
  - Recommended fix: Add a comment in the prompt: "Note: `near_duplicate`, `opening_trigram`, and `low_vocabulary` are assigned by automated checks — do not use them in LLM review output." This prevents accidental usage if the prompt is ever repurposed.

- **[soft] [sm-fragile]** — The review prompt receives `{passages_batch}` and `{voice_document}` but no passage summaries or beat context. A small model reviewing for `summary_deviation` needs the beat summary to compare against the prose — without it, the model can only detect gross deviations (e.g., completely wrong characters). The review is less actionable without the ground truth.
  - Where: system block, lines 7–9; missing beat summary context
  - Spec citation: `fill.md §R-3.2` ("Flags name specific issues: … summary deviation"); `fill.md §Implementation Constraints §Context Enrichment`
  - Recommended fix: The `{passages_batch}` context (built by `format_passages_batch`) should include the beat summary alongside the prose. Check `fill_context.py`'s `format_passages_batch` implementation to verify whether beat summaries are included; if not, add them.

- **[info] [sm-fragile]** — The review criteria table (lines 18–28) has no GOOD/BAD examples for what a flag's `issue` description should look like. A model might write vague flags like "voice issues noted" which violates R-3.2 ("flags name specific issues"). The user prompt asks for a "clear description of the problem" but doesn't model what that looks like.
  - Where: system block, lines 30–33 ("Guidelines"); user block, line 42
  - Spec citation: `fill.md §R-3.2`; `CLAUDE.md §7 Defensive Prompt Patterns`
  - Recommended fix: Add to Guidelines:
    ```
    BAD: issue: "voice issues noted"
    GOOD: issue: "Register shifts to conversational ('gonna', 'yeah') while voice document specifies literary"
    ```

---

### `prompts/templates/fill_phase3_revision.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — The revision prompt's output format includes `entity_updates` in the `passage` field, but the context dict injected at runtime (fill.py lines 1879–1891) contains no `entity_states` or `valid_entity_ids`. The model must generate entity update IDs from memory of the passage's prose, with no valid entity ID list. This violates CLAUDE.md §6 (Valid ID Injection) and creates phantom ID risk — the escalation handler at fill.py lines 1951–1963 escalates phantom IDs at stage exit, but the phantom ID was preventable.
  - Where: system block, lines 62–66 (output format shows `entity_updates`); `fill.py` lines 1879–1891 (context dict missing `entity_states`, `valid_entity_ids`)
  - Spec citation: `CLAUDE.md §6 Valid ID Injection Principle` ("Always provide an explicit Valid IDs section listing every ID the model is allowed to use. Never assume the model will correctly infer IDs from prose."); `fill.md §R-2.12` ("entity updates are additive only"); `fill.md §R-2.15` ("Micro-detail updates must not contradict existing Entity state")
  - Recommended fix: Add `entity_states` and `valid_entity_ids` to the revision context dict (reuse the same `format_entity_states` call used in the prose context) and add a `{valid_entity_ids}` section in the template:
    ```yaml
    ## Valid Entity IDs (use ONLY these for entity_updates)
    {valid_entity_ids}
    ```

- **[soft] [sm-fragile]** — The revision prompt has no sandwich repetition for the voice document constraint. The system block starts with "Follow the voice document exactly" (line 4) but the user-message CRITICAL REMINDERS (lines 70–75) say "Follow the voice document EXACTLY" — good, but omit the specific constraint echoing (POV, tense, register) that would anchor a 4B model after reading a long system prompt.
  - Where: user block, lines 70–75
  - Spec citation: `CLAUDE.md §10 §Small Model Prompt Bias`; `fill.md §R-4.1` ("Revision uses the same rules as Phase 2 generation plus the issue description")
  - Recommended fix: Extend the user-message reminder:
    ```
    - Follow the voice document EXACTLY — POV: {pov_hint}, tense: {tense_hint}, register: {register_hint}
    ```
    Or at minimum: "Follow the voice document EXACTLY — POV, tense, register, rhythm, tone are binding."

- **[soft] [repair-gap]** — The `issues_list` format is `"{i+1}. [{issue_type}] {issue_text}"` (fill.py line 1874). The revision guidance section maps each `issue_type` to a fix strategy, which is excellent. However, the feedback does not echo the specific passage's beat summary, which is needed for `summary_deviation` fixes. The model must infer what the summary said from its context window.
  - Where: fill.py lines 1873–1877 (issues_list construction); template line 7–10 (`{issues_list}`)
  - Spec citation: `fill.md §R-4.1` ("Revision uses Phase 2 rules plus the issue description"); `fill.md §Phase 3: Revision §Operations §What` ("extended context: Voice Document, issue description, extended sliding window, and relevant lookahead/continuity passages")
  - Recommended fix: Add `beat_summary` to the revision context dict and inject it in the template:
    ```yaml
    ## Beat Summary (Reference for summary_deviation fixes)
    {beat_summary}
    ```
    This is already known at call time (from the passage node) and makes the revision self-contained for the `summary_deviation` case.

---

### Stage summary: FILL

- Prompts audited: 8
- Hard findings: 3 (POV schema gap latent in `fill_phase0_voice.yaml` triggered on spec fix; shadow omission in `fill_phase1_prose_only.yaml`; missing Valid Entity IDs in `fill_phase3_revision.yaml`)
- Soft findings: 10
- Info findings: 8
- Spec gaps surfaced: 2 (`pov_character` applicability for first-person unclear in spec; `fill.md` worked example uses invalid `voice_register` value)
- Recommended PR split: Two clusters — (1) POV Literal alignment (`fill.py` model + both phase0 prompts + fill.md spec fix, one PR), (2) context enrichment fixes (`fill_phase1_prose.yaml` + `fill_phase1_prose_only.yaml` shadow + narrative_function, revision context) as a second PR. The Valid Entity IDs fix in `fill_phase3_revision.yaml` is one-line and can go in either cluster.
- Status: drift

---

## DRESS

### `prompts/templates/dress_discuss.yaml`

**Verdict:** mixed

**Findings:**

- **[soft] [schema-skew]** — `entity_list` injected via code uses Python f-string interpolation of raw entity dict values; small models see `entity_visual::kael_vex: character — ` with the scoped prefix in the template, while `dress_serialize` later expects raw IDs without prefix. The entity category prefix is visible in the list (e.g., `entity::mentor`) but the discuss prompt's `{entity_list}` block is built in the stage code (line 442–445) with the full scoped node ID, while the Guidelines section tells the model to think in terms of appearance but never tells it which ID format to use in output. This creates constraint-to-value mapping ambiguity when the model begins proposing visual IDs.
  - Where: `dress_discuss.yaml` lines 27–29 (`{entity_list}` variable); `dress.py` lines 442–445 (builder code)
  - Spec citation: `CLAUDE.md §6` (Valid ID Injection); `CLAUDE.md §9` (never interpolate Python objects)
  - Recommended fix: In the prompt, add an explicit instruction: "Entity IDs to use: see `{entity_list}` above. Use the **raw ID** (without `entity::` prefix) in all visual references." The builder code should also strip the scope prefix for the entity list shown in discuss, matching what `dress_serialize` accepts.

- **[soft] [sm-fragile]** — The `reference_prompt_fragment` constraint (appearance only, no camera angles/poses/actions) is stated in the Guidelines section but not reinforced at the bottom of the prompt (no sandwich). For a 4B model in a long multi-turn discussion, this rule will be forgotten before the model finalizes its entity profile proposals.
  - Where: `dress_discuss.yaml` lines 35–36 (first and only occurrence)
  - Spec citation: `CLAUDE.md §10` (small-model bias); role file §Required reading #1 (constraint-to-value mapping loss)
  - Recommended fix: Add a closing reminder after `{mode_section}`:
    ```
    REMINDER: reference_prompt_fragment describes APPEARANCE ONLY — clothing, colors, build,
    distinguishing features. No camera angles, poses, or actions.
    ```

- **[soft] [sm-fragile]** — The `non_interactive_section` instructs the model to "make confident visual decisions" but provides no completeness checklist. A model that produces visual profiles for only 3 of 8 entities will have its discuss output forwarded to summarize with gaps. The summarize prompt does check for missed entities ("flag any that were missed") but the model generates this warning to itself — no gate catches a missing entity before serialization.
  - Where: `dress_discuss.yaml` lines 88–93 (`non_interactive_section`)
  - Spec citation: `dress.md §R-1.3` (every entity with `appears` edge gets EntityVisual); `CLAUDE.md §10`
  - Recommended fix: Add an entity completeness check to the non_interactive_section:
    ```
    Before concluding, verify you have proposed a visual profile for EVERY entity in the
    Entities Requiring Visual Profiles section above. Missing entities will cause validation failure.
    ```

- **[info] [drift]** — The `research_tools_section` lists `web_search` and `web_fetch` as available tools. These are valid for interactive mode but `get_all_research_tools()` in the stage code conditionally includes them only when the tool list is non-empty. No issue — the section is only injected when tools exist — but the tool names appear in the prompt without a check against what tools are actually wired. If the tool list changes, the prompt's enumeration becomes stale.
  - Where: `dress_discuss.yaml` lines 73–76 (tool list)
  - Spec citation: `CLAUDE.md §7` (defensive prompt patterns)
  - Recommended fix: Consider abstracting tool availability into the injected `{research_tools_section}` variable so the prompt never hardcodes tool names.

---

### `prompts/templates/dress_summarize.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [repair-gap]** — The `NO DELEGATION` section forbids the model from saying "consider using…" but there is no mention of what to do when an entity's visual profile was not discussed at all. The summarize prompt checks for gaps ("flag any that were missed") but only logs them as "not specified" — it doesn't trigger a re-discuss loop or fail. The stage code (`summarize_discussion()` → `serialize_to_artifact()`) accepts whatever summary is produced; if the summary says "Aldric: not specified" the serialize phase will receive that as an entity visual description. The Pydantic model requires `min_length=1` on `description` and `reference_prompt_fragment`, so serialization will fail, but by then the feedback loop is entirely gone — no repair prompt is generated for summarize phase output failures.
  - Where: `dress_summarize.yaml` lines 36–37 ("If something wasn't discussed, note it as 'not specified'"); `dress.py` lines 528–560 (serialize path — no retry if summarize emits a gap)
  - Spec citation: `dress.md §R-1.3` and §R-1.4 (every entity must have EntityVisual with non-empty `reference_prompt_fragment`); `CLAUDE.md §repair-loop quality`
  - Recommended fix: Change the "not specified" fallback to a hard instruction: "If an entity's visual profile is incomplete, you MUST invent plausible visual details consistent with the story's genre and tone — do NOT write 'not specified'. Every entity requires a complete visual profile." This moves the gap-filling responsibility to summarize rather than silently propagating None into serialize.

- **[soft] [sm-fragile]** — No output structure is prescribed for the per-entity sections. The summarize prompt says "provide a structured summary with a global art direction section and per-entity visual sections" but gives no heading template. The serialize phase uses `{art_brief}` as a free-text input; if the model arranges entity sections inconsistently the serialize phase's ability to extract each entity's data degrades on small models.
  - Where: `dress_summarize.yaml` lines 39–41 (Output Format section)
  - Spec citation: `CLAUDE.md §7` (GOOD/BAD examples); `CLAUDE.md §10`
  - Recommended fix: Add a minimal heading template for the per-entity section:
    ```
    ### Entity: {entity_raw_id}
    - Description: ...
    - Distinguishing features: ...
    - Color associations: ...
    - Reference prompt fragment: ...
    ```

- **[soft] [sm-fragile]** — The `reference_prompt_fragment` constraint (appearance only, no camera angles) is mentioned only in line 18, and that occurrence is in a prose list item. Small models summarizing a long discussion will default to the last thing that was said about an entity and may include camera/pose language in the fragment if it appeared late in the discuss conversation.
  - Where: `dress_summarize.yaml` line 18
  - Spec citation: `CLAUDE.md §10`; `dress.md §R-1.4`
  - Recommended fix: Repeat the constraint explicitly in the per-entity section template (see above fix), e.g., "Reference prompt fragment: [appearance only — clothing, colors, build, features. NO camera angles, poses, or actions]".

---

### `prompts/templates/dress_serialize.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — `palette` is described as `list of dominant color names` — correct — but the `entity_visuals` schema block says `color_associations: Colors tied to this entity (can be empty)`. The Pydantic model has `color_associations: list[str] = Field(default_factory=list)` — so "can be empty" is correct. However, `distinguishing_features` is described as "List of key visual identifiers" with no `min_length` hint. The Pydantic model has `min_length=1` on that field, so an empty list will fail validation. The prompt provides no GOOD/BAD example to prevent an empty list.
  - Where: `dress_serialize.yaml` lines 27–29 (entity_visuals schema block)
  - Spec citation: `src/questfoundry/models/dress.py` line 111–113 (`distinguishing_features: list[str] = Field(min_length=1)`); `CLAUDE.md §7`
  - Recommended fix: Add a constraint note: "distinguishing_features: List of key visual identifiers — MUST have at least one item. GOOD: `['scar across left cheek', 'silver hair']`. BAD: `[]`."

- **[hard] [repair-gap]** — The repair loop in `_dress_llm_call` (dress.py lines 682–689) emits: `"Your response failed validation:\n{e}\n\nExpected fields: {', '.join(expected)}\nPlease fix the errors and try again."` This is the generic repair message for ALL DRESS templates, including `dress_serialize`. For serialization failures this message echoes the field names but not the valid entity IDs — so a "entity_id not in valid list" Pydantic error will be sent back to the model without repeating the valid IDs, allowing the model to invent a different wrong ID on retry.
  - Where: `dress.py` lines 682–689 (generic repair message); `dress_serialize.yaml` lines 42 (`CRITICAL: Use ONLY entity IDs from the valid list`)
  - Spec citation: `CLAUDE.md §6` (Valid ID Injection Principle); `CLAUDE.md §repair-loop quality` (repair feedback must echo the expected value)
  - Recommended fix: The generic repair message should re-inject the valid entity IDs on retry for `dress_serialize` calls, e.g., append: `"Valid entity IDs: {entity_ids}\n"`. Alternatively, the template's user turn (which already has the CRITICAL reminder) should be preserved in the messages list so it's present when the error feedback is appended.

- **[soft] [schema-skew]** — The prompt describes `entity_id` as "Must be one of the valid entity IDs listed above (raw ID without scope prefix)". The `entity_ids` variable injected into the template (dress.py line 519–521) is built from `edata.get('raw_id', strip_scope_prefix(eid))` — correct. However, `EntityVisualWithId.entity_id` has `min_length=1` but no further validation that the value matches an existing entity — so the Pydantic model accepts any non-empty string. The only cross-reference check is the template's instruction. If the model produces a scope-prefixed ID (e.g., `entity::aldric` instead of `aldric`), Pydantic passes but the graph lookup will fail silently.
  - Where: `dress_serialize.yaml` line 25; `src/questfoundry/models/dress.py` line 131
  - Spec citation: `dress.md §R-1.3`; `CLAUDE.md §6`
  - Recommended fix: Add a defensive GOOD/BAD example: "GOOD: `aldric`, BAD: `entity::aldric`". The mutations layer should also strip prefix before lookup.

- **[info] [schema-skew]** — The prompt's `art_direction` section lists `aspect_ratio` but says `Default ratio (e.g. "16:9")`. The stage code in `_parse_aspect_ratio()` tolerates freeform strings like `"16:9 (story panels), 4:5 (character plates)"` and extracts the first valid ratio. The prompt's example is clean, but a small model that produces `"16:9 for action, 4:3 for portraits"` will have its output silently reduced to `"16:9"`. This is handled gracefully in code; the prompt could add: "Use only one ratio string."
  - Where: `dress_serialize.yaml` line 22; `dress.py` lines 121–135
  - Spec citation: `dress.md §R-1.2`
  - Recommended fix: Add: `aspect_ratio: Exactly one ratio (e.g. "16:9"). Do NOT include multiple ratios or explanatory text.`

---

### `prompts/templates/dress_brief.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — The `category` field lists `"scene", "portrait", "vista", "item_detail"` (4 values) but the Pydantic `IllustrationCategory` Literal includes `"cover"` as a fifth value. The prompt never mentions `"cover"` so models can never produce it intentionally for a cover-image brief.
  - Where: `dress_brief.yaml` line 43 (`category: One of "scene", "portrait", "vista", "item_detail"`)
  - Spec citation: `src/questfoundry/models/dress.py` line 22 (`IllustrationCategory = Literal["scene", "portrait", "vista", "item_detail", "cover"]`)
  - Recommended fix: Update the line to: `category: One of "scene", "portrait", "vista", "item_detail", "cover"`. The same fix applies to `dress_brief_batch.yaml` (identical omission at line 39).

- **[soft] [sm-fragile]** — `{entity_visuals}` is listed in the system template but its content (from `format_all_entity_visuals()`) only includes entities that have an `entity_visual` node — i.e., entities for which Phase 0 created a visual profile. If Phase 0 produced EntityVisuals for 6 of 8 entities, the brief prompt silently omits the other 2. The model has no signal that some entities lack a visual fragment and may generate references to them anyway, producing inconsistent illustration specs.
  - Where: `dress_brief.yaml` line 12 (`{entity_visuals}`); `dress_context.py` lines 379–408 (`format_all_entity_visuals`)
  - Spec citation: `dress.md §R-2.6` (LLM receives EntityVisuals for every appearing entity); `CLAUDE.md §8` (context enrichment)
  - Recommended fix: The context builder should note missing EntityVisuals explicitly: "Note: the following entities appear in passages but have no visual profile yet and should not be depicted with specific features: `aldric`, `archive`." Without this signal the model has no way to adapt.

- **[soft] [repair-gap]** — The repair loop for brief failures (same generic `_dress_llm_call` path) does not distinguish between a `priority` out-of-range failure and a missing `caption`. The error message echoes field names but not the valid range for `priority` (1–3) or the caption format. A 4B model receiving "priority: validation error" on retry has no guidance on what value to use.
  - Where: `dress.py` lines 682–689 (generic repair message)
  - Spec citation: `CLAUDE.md §repair-loop quality`
  - Recommended fix: The repair message for brief templates should echo: "priority must be 1, 2, or 3 — not 0, not 4, not a string. caption must be 10-60 characters in format '[Subject] [action/state]'." This is prompt-level guidance that should appear in the repair feedback.

- **[soft] [sm-fragile]** — The `{priority_context}` variable appears in the system template but is NOT present in `dress_brief_batch.yaml`. The batch version receives per-passage context in the user message via `format_passages_batch_for_briefs()` which calls `describe_priority_context()` per passage and appends it inline. This asymmetry means the single-passage prompt has priority context in the system (shared, reusable) while the batch has it per-passage in the user turn. No immediate bug, but the two prompts diverge in how structural score is communicated, making future changes error-prone.
  - Where: `dress_brief.yaml` line 18 (`{priority_context}`) vs `dress_brief_batch.yaml` (no `priority_context` variable)
  - Spec citation: `dress.md §R-2.5` (priority scoring rules)
  - Recommended fix: This is a design note rather than a bug. Document the intentional difference in a code comment in `_phase_1_briefs`.

- **[info] [drift]** — `{output_language_instruction}` is present but appears at a mid-point in the schema block (line 58), sandwiched between the schema description and the guidelines. For small models this placement may cause the language instruction to be treated as part of the schema. Move it to after all guidelines, immediately before the user turn.
  - Where: `dress_brief.yaml` line 58
  - Spec citation: `CLAUDE.md §9`
  - Recommended fix: Move `{output_language_instruction}` to the last line of the system block, after all Guidelines content.

---

### `prompts/templates/dress_brief_batch.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [schema-skew]** — Same `category` omission as `dress_brief.yaml`: lists only 4 values, missing `"cover"`.
  - Where: `dress_brief_batch.yaml` line 39
  - Spec citation: `src/questfoundry/models/dress.py` line 22
  - Recommended fix: Add `"cover"` to the category options list.

- **[soft] [sm-fragile]** — The batch version's `{entity_visuals}` section at line 9 receives ALL entity visuals for the batch (not per-passage), so a 5-passage batch covering different locations and characters gets one combined visual reference block. If 3 of 5 passages share characters and 2 do not, the model may incorrectly assign entity references. The prompt gives no instruction to use `entities` field matching.
  - Where: `dress_brief_batch.yaml` line 9 (`{entity_visuals}`); `dress_context.py` lines 379–408
  - Spec citation: `dress.md §R-2.6`; `CLAUDE.md §8`
  - Recommended fix: In the batch user message (the `passages_batch` block), each passage's section already lists "Entities present:" (from `format_passage_for_brief`). Add a note in the system prompt: "Only include entity visual references in a brief's `entities` field if that entity appears in THAT passage's 'Entities present' list."

- **[soft] [sm-fragile]** — The batch user turn repeats the passage count in three places (`{passage_count}` appears twice in `user:` and once as a count in the header). For a small model, this is effective emphasis. However, the user turn has no per-passage ID reminder — it relies on the passage section headers from `format_passages_batch_for_briefs()`. If the passage raw_id contains characters that make it visually ambiguous (e.g., `passage::intro_a` vs `passage::intro_b`), the model may mix up which `passage_id` belongs to which brief.
  - Where: `dress_brief_batch.yaml` lines 83–89
  - Spec citation: `dress.md §R-2.1`
  - Recommended fix: Confirm the brief batch response validator checks `passage_id` in the returned briefs against the expected set and logs WARNING for mismatches.

- **[info] [drift]** — Same `{output_language_instruction}` placement issue as `dress_brief.yaml` — appears mid-schema at line 54. Low priority given the batch context is already long.
  - Where: `dress_brief_batch.yaml` line 54
  - Spec citation: `CLAUDE.md §9`
  - Recommended fix: Move to end of system block.

---

### `prompts/templates/dress_codex.yaml`

**Verdict:** broken

**Findings:**

- **[hard] [terminology]** — The prompt uses "codewords" throughout (`## Available Codewords`, `visible_when: List of codeword IDs`, `Codeword IDs in visible_when MUST be from the available codewords list`) but the authoritative spec says DRESS gates internally via **state flag** IDs, not codewords. R-3.7 is explicit: "CodexEntry gating uses state flag IDs, not codewords. SHIP projects a subset of state flags as player-facing codewords; DRESS gates internally via state flags." The `CodexEntry.visible_when` Pydantic model's docstring says "Codeword IDs that must all be present to unlock this tier" — which is itself a terminology error matching this prompt.
  - Where: `dress_codex.yaml` lines 14, 18, 24, 33, 34, 41 (all "codeword" occurrences); `src/questfoundry/models/dress.py` line 91 (docstring)
  - Spec citation: `dress.md §R-3.7`; `story-graph-ontology.md §Part 8: Codewords ≠ State Flags`
  - Recommended fix: Replace all occurrences of "codeword" / "Codewords" with "state flag" / "State Flags" in this template. Update the Pydantic model docstring too. The `{codewords}` variable injected from the stage code (dress.py line 919) is built from `state_flag` nodes — the data is correct, only the label is wrong.

- **[hard] [repair-gap]** — The `dress_codex` template is used during single-entity regeneration after spoiler detection (`_regenerate_codex_for_entity`). The repair context appended to `entity_details` (dress.py line 1122–1123) echoes the specific leaked content. However, the codex template's system prompt does NOT mention spoiler-direction rules at all — only the repair context injected into `entity_details_with_warning` carries this constraint. A 4B model on retry may follow the regeneration prompt without understanding WHY rank-1 must be vague: the system prompt gives no rule about "lower tier must not disclose what higher tier reveals."
  - Where: `dress_codex.yaml` lines 28–34 (Rules section — no mention of spoiler direction); `dress.py` lines 1112–1128 (repair injection)
  - Spec citation: `dress.md §R-3.6`; `CLAUDE.md §repair-loop quality`
  - Recommended fix: Add to the Rules section:
    ```
    - Spoiler direction: rank 1 must NOT disclose content whose reveal is gated in rank 2+.
      GOOD rank 1: "A traveling scholar who offers guidance to weary travelers."
      BAD rank 1: "A mysterious scholar who secretly knows your true identity." (leaks rank 2 content)
    ```

- **[soft] [sm-fragile]** — The `{entity_details}` variable is populated by `format_entity_for_codex()`, which does include concept, entity type, visual profile, and related state flags. However, the base entity's overlay descriptions are NOT included — only the base `concept` field. The spec (R-3.8) requires "full Entity description (base + overlays)". For entities with path-specific overlays (e.g., mentor_aligned overlay vs mentor_hostile overlay), the model generates codex entries without knowing the full entity arc.
  - Where: `dress_context.py` lines 186–252 (`format_entity_for_codex` — no overlay retrieval); `dress_codex.yaml` line 15 (`{entity_details}`)
  - Spec citation: `dress.md §R-3.8`
  - Recommended fix: `format_entity_for_codex()` should also retrieve and format entity overlays (nodes connected to the entity by overlay edges), providing the full entity arc that the codex is supposed to reflect.

- **[soft] [schema-skew]** — The `{codewords}` variable is formatted by the stage code as: `- \`{sf_raw}\`: {trigger}` (dress.py line 897). The `trigger` field on a state_flag node is the narrative trigger (e.g., "when player chooses to trust the mentor"). This is the right field for communicating what the state flag means, but the format does not include the full `state_flag_id` that `visible_when` must use. Since `sf_raw` IS the raw ID (it's what gets stored in `visible_when`), the mapping is correct. However, the prompt says "Codeword IDs in visible_when MUST be from the available codewords list" — if the model copies `trigger` text instead of `sf_raw` into `visible_when`, validation will fail with no hint about which IDs are valid.
  - Where: `dress_codex.yaml` lines 24, 34; `dress.py` lines 896–899
  - Spec citation: `CLAUDE.md §6`
  - Recommended fix: After fixing the codeword→state_flag terminology: "The ID to use in `visible_when` is the backtick-wrapped value at the start of each line, e.g., `met_aldric`. Do NOT use the description text as an ID."

---

### `prompts/templates/dress_codex_batch.yaml`

**Verdict:** broken

**Findings:**

- **[hard] [terminology]** — Same codeword vs state flag terminology error as `dress_codex.yaml`. The batch version uses "codewords" in the same positions: `## Available Codewords`, `visible_when: List of codeword IDs`, `Codeword IDs in visible_when MUST be from the available codewords list`.
  - Where: `dress_codex_batch.yaml` lines 12, 22, 29, 33, 41
  - Spec citation: `dress.md §R-3.7`; `story-graph-ontology.md §Part 8: Codewords ≠ State Flags`
  - Recommended fix: Same as `dress_codex.yaml`: replace all "codeword" → "state flag" throughout.

- **[hard] [repair-gap]** — The batch template omits the spoiler-direction rule entirely (same gap as `dress_codex.yaml`). The batch version also has no per-entity spoiler instruction, meaning the first-pass batch generation for all entities has NO guidance about rank-1 spoiler avoidance. The spoiler check is a separate post-hoc LLM call — it catches violations after generation — but the generation prompt gives the model no reason to avoid the violation in the first place.
  - Where: `dress_codex_batch.yaml` lines 28–35 (Rules — missing spoiler-direction rule)
  - Spec citation: `dress.md §R-3.6`; `CLAUDE.md §sm-fragile`
  - Recommended fix: Add to Rules:
    ```
    - Spoiler direction: rank 1 must NOT disclose content whose reveal is gated in rank 2+.
      GOOD rank 1: "A traveling scholar who offers guidance."
      BAD rank 1: "A mysterious scholar who secretly knows your true identity." (leaks rank 2)
    ```
    This reduces spoiler-check retry load.

- **[soft] [sm-fragile]** — No per-entity entity detail in system prompt; the `{entities_batch}` is injected in the user message. For a batch of 4 entities (default `_CODEX_BATCH_SIZE = 4`), the model must maintain separate codex tiers for 4 different entities simultaneously. The user message's entity sections are produced by `format_entities_batch_for_codex()` → `format_entity_for_codex()` per entity, which includes concept, type, and related state flags. This is adequate, but the model has no summary of how many entries are expected per entity and no guidance on tier count. Models routinely produce only 1 tier per entity in batch mode.
  - Where: `dress_codex_batch.yaml` lines 15–24 (output schema — no tier count guidance)
  - Spec citation: `dress.md §Phase 3: Codex Generation` (Plan tier structure)
  - Recommended fix: Add: "Aim for 2-3 tiers per entity when multiple relevant state flags exist; 1 tier when no state flags are related. Do not produce only 1 tier for an entity that has multiple related state flags — that wastes the codex system's spoiler-graduation capability."

- **[soft] [sm-fragile]** — Same overlay enrichment gap as `dress_codex.yaml`: entity details passed to the batch prompt do not include overlay descriptions (R-3.8 violation at the context builder level).
  - Where: `dress_context.py` lines 411–427 (`format_entities_batch_for_codex`) → `format_entity_for_codex` (no overlays)
  - Spec citation: `dress.md §R-3.8`
  - Recommended fix: Same as `dress_codex.yaml` — fix `format_entity_for_codex()` to include overlays; the batch function inherits the fix.

---

### `prompts/templates/dress_codex_spoiler_check.yaml`

**Verdict:** mixed

**Findings:**

- **[hard] [terminology]** — The template's description says "gated by state flags" (line 10) but the examples section (lines 22–28) uses `gated by \`met_aldric\`` without clarifying whether `met_aldric` is a state flag ID or a codeword. Given the ambiguity in the sibling codex templates, a model that has generated entries using codeword IDs (due to the `dress_codex` terminology bug) will be checked by a spoiler-check prompt that accepts those IDs as natural. The spoiler-check prompt should assert that `visible_when` values are state flag IDs to be consistent when the codex templates are fixed.
  - Where: `dress_codex_spoiler_check.yaml` line 10 ("gated by state flags") vs rest of prompt
  - Spec citation: `dress.md §R-3.7`
  - Recommended fix: Minor — add one line to clarify: "State flags are the IDs listed in `visible_when` (e.g., `met_aldric`, `chose_betrayal`)." This is a low-risk clarification since the spoiler-check prompt doesn't generate `visible_when` values itself.

- **[soft] [repair-gap]** — The spoiler-check prompt is clean about its detection task but the `reason` field is the only feedback mechanism for the downstream regeneration. The `_regenerate_codex_for_entity` function formats the leak list as: `"rank {lower} leaked content gated behind rank {higher}: {leaked_content}"`. The regeneration then calls `dress_codex` (single-entity template) with this appended to `entity_details`. If the spoiler-check's `leaked_content` field is terse or inaccurate (e.g., "scholar knows secret"), the regeneration prompt may not identify WHICH sentence in rank 1 to remove. The spoiler check prompt doesn't instruct the model to quote the problematic phrase verbatim.
  - Where: `dress_codex_spoiler_check.yaml` lines 33–38 (output schema for `leaked_content`); `dress.py` lines 1112–1123 (repair injection)
  - Spec citation: `CLAUDE.md §repair-loop quality`
  - Recommended fix: In the schema description for `leaked_content`, change: "short paraphrase of what was leaked" → "exact quote or close paraphrase of the specific phrase in the lower-ranked entry that prematurely reveals the higher-tier content (this will be shown to the author for revision)". This guides the model to produce a more precise quote that makes repair feedback actionable.

- **[soft] [sm-fragile]** — The `{entries_block}` user-turn variable is formatted by `_format_entries_for_spoiler_check()` (stage code). The check is: does a LOWER rank entry disclose what a HIGHER rank was meant to reveal? The prompt's "No leak" example is well-chosen and correct (lines 29–31). However, there's no GOOD/BAD example for the output JSON itself — a 4B model may produce `has_leak: true` with an empty `leaks` list (contradictory), or produce `has_leak: false` with a non-empty `leaks` list. The Pydantic model does not validate this consistency.
  - Where: `dress_codex_spoiler_check.yaml` lines 32–39 (output schema — no consistency example)
  - Spec citation: `CLAUDE.md §7` (GOOD/BAD examples); `src/questfoundry/models/dress.py` lines 268–277 (`SpoilerCheckResult`)
  - Recommended fix: Add a schema note: "`has_leak` and `leaks` must be consistent: if `has_leak: true` then `leaks` must be non-empty; if `has_leak: false` then `leaks` must be empty." Add the same invariant as a `model_validator` in `SpoilerCheckResult` to catch it at validation time.

- **[info] [schema-skew]** — The prompt schema describes `lower_rank (int ≥ 1)` and `higher_rank (int ≥ 2)`. The Pydantic model has `lower_rank: int = Field(ge=1)` and `higher_rank: int = Field(ge=2)` and a `model_validator` that enforces `lower_rank < higher_rank`. The prompt does NOT mention the ordering constraint (`lower_rank must be strictly less than higher_rank`). This is caught at validation time but would reduce retries if stated explicitly.
  - Where: `dress_codex_spoiler_check.yaml` lines 34–36
  - Spec citation: `src/questfoundry/models/dress.py` lines 258–264
  - Recommended fix: Add: "`lower_rank` must be strictly less than `higher_rank` (e.g., lower_rank=1, higher_rank=2 is valid; lower_rank=2, higher_rank=1 is not)."

---

### Stage summary: DRESS

- Prompts audited: 8
- Hard findings: 10
- Soft findings: 17
- Info findings: 5
- Spec gaps surfaced: 0
- Recommended PR split: see Cross-cutting issues below — terminology fix (codeword → state_flag) is one narrow PR; repair-feedback enrichment, entity-overlay enrichment, and spoiler-direction additions can each be their own follow-up.
- Status: drift

**Cross-cutting issues:**

1. **codeword vs state_flag terminology** (`dress_codex.yaml`, `dress_codex_batch.yaml`, model docstring) — the single largest drift in the DRESS stage. Two templates consistently instruct the model to reference "codeword IDs" in `visible_when` but R-3.7 requires state flag IDs. The stage code correctly reads `state_flag` nodes; only the prompt labels are wrong.
2. **Generic repair message** (`_dress_llm_call`) — all DRESS templates share one repair message that echoes field names but not valid ID lists or field value constraints. For `dress_serialize`, `dress_brief`, and `dress_codex`, this causes blind retries.
3. **Entity overlay enrichment missing** (`format_entity_for_codex()`) — both codex templates receive entity context without overlays, violating R-3.8.
4. **Spoiler-direction rule absent from generation prompts** — `dress_codex.yaml` and `dress_codex_batch.yaml` have no R-3.6 instruction; spoiler avoidance is entirely reactive (post-hoc check) rather than proactive (instruction at generation time).

---

## SHIP

**Verdict:** n/a — no LLM prompts to audit.

SHIP is the only stage in the pipeline that does not invoke an LLM. It is a
deterministic technical transformation: it reads the completed story graph and
emits playable formats (Twee, HTML, JSON, Gamebook PDF). Confirmed by:

- `prompts/templates/ship_*.yaml` — no files exist.
- `src/questfoundry/models/ship*.py` — no SHIP-specific Pydantic artifact models exist.
- `src/questfoundry/pipeline/stages/ship.py` — module docstrings explicitly state
  "SHIP is deterministic (no LLM)" and "Unlike other stages, SHIP does not use
  an LLM. It reads the graph".
- `docs/design/procedures/ship.md §Overview` — "It is a read-only technical
  transformation: SHIP does not mutate the graph."

The five audit dimensions (spec accuracy, repair-loop quality, small-model
resilience, schema alignment, drift markers) presuppose an LLM-driven prompt
to evaluate. With no prompts, none of the dimensions apply. Any SHIP-related
correctness work is non-prompt engineering and falls outside this audit's scope.

The codeword projection rules (R-1.1 through R-1.7), persistent/working
boundary, and export determinism are validated by code and tests, not by
prompt review. Tracking issues for those concerns belong in the existing
SHIP cluster from the spec-compliance audit (memory: 8 clusters remaining).
