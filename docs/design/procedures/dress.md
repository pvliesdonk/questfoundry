# DRESS — Visual identity, illustrations, and codex

## Overview

DRESS adds visual identity and reference material to a completed story. It generates an ArtDirection node, per-entity visual profiles, illustration briefs with priority scoring, generated image assets, and tiered diegetic codex entries. DRESS is **optional** — the story is playable without it — and it does NOT modify prose, beats, passages, choices, entities, or any narrative structure. It reads and decorates.

## Stage Input Contract

*Must match FILL §Stage Output Contract exactly.*

1. Voice Document singleton exists with all required fields populated.
2. Every Passage has non-empty `prose`.
3. Entity base-state enriched with zero or more universal micro-details (additive only).
4. No Passage, Choice, beat, Entity, Dilemma, Path, Consequence, or State Flag nodes created or deleted by FILL.
5. No overlay modifications.
6. Character arc metadata unchanged (consumed as context, not mutated).
7. At most 2 review+revision cycles were run.

---

## Phase 1: Art Direction

**Purpose:** Establish the story's visual identity — ArtDirection singleton plus one EntityVisual per Entity appearing in passages. Governs all downstream illustration generation, analogous to how the Voice Document governs prose.

### Input Contract

1. Stage Input Contract satisfied.
2. `appears` edges exist from Entities to Passages (GROW/POLISH output; derivable from beat references if missing).

### Operations

#### Art Direction Discussion and Serialization

**What:** Uses the three-phase pattern (discuss / summarize / serialize). LLM and human explore visual style; LLM distills to a narrative art brief; structured output produces the ArtDirection node and per-entity EntityVisual nodes with `describes_visual` edges.

**Rules:**

R-1.1. Exactly one ArtDirection singleton node is created. Retries replace the previous node.

R-1.2. ArtDirection fields include `style`, `medium`, `palette`, `composition_notes`, `style_exclusions`, `aspect_ratio`.

R-1.3. Every Entity with at least one `appears` edge to a Passage gets an EntityVisual node with a `describes_visual` edge.

R-1.4. Every EntityVisual has a non-empty `reference_prompt_fragment` — the text that will be injected into image prompts when this entity appears.

R-1.5. Human approval at Gate 1 is required before proceeding to Phase 2.

R-1.6. DRESS does not modify the Entity's base-state. EntityVisual is a separate node, not an annotation on the Entity.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two ArtDirection nodes exist | Retry duplicated | R-1.1 |
| Entity appearing in 3 passages has no EntityVisual | Phase 1 skipped this entity | R-1.3 |
| EntityVisual has `reference_prompt_fragment: ""` | Required field empty | R-1.4 |
| DRESS writes to `character::mentor.description` | Entity base-state mutated | R-1.6 |
| Phase 2 starts without Gate 1 approval | Human gate skipped | R-1.5 |

### Output Contract

1. Exactly one ArtDirection node exists.
2. Every Entity with ≥1 `appears` edge has an EntityVisual node connected by `describes_visual`.
3. Every EntityVisual has a non-empty `reference_prompt_fragment`.
4. Human approval of Gate 1 recorded.

---

## Phase 2: Illustration Brief Generation

**Purpose:** For each Passage, generate a structured IllustrationBrief with priority score, subject, composition, mood, category, caption, and negative prompt. Briefs are candidates; Phase 4 renders the selected subset.

### Input Contract

1. Phase 1 Output Contract satisfied.

### Operations

#### Per-Passage Brief Generation

**What:** For each Passage, read the prose, the `appears` entities, and the Passage's position in the graph (spine/branch/ending). Compute a structural priority base score (spine +3, branch opening +2, ending +2, climax +2, new location +1, transition/micro-beat −1). LLM generates the narrative brief and an adjustment score (visually striking +1, emotionally pivotal +1, dialogue-heavy −1). Compute final priority.

**Rules:**

R-2.1. Every Passage gets an IllustrationBrief node with a `targets` edge to the Passage. Briefs that score ≤ 0 are still created but marked `priority: skip` (not rendered by default).

R-2.2. Briefs are created for Passages, not for beats.

R-2.3. Each Brief has non-empty `subject`, `composition`, `mood`, `category`, `caption`, plus `negative` and optional `style_overrides`.

R-2.4. Caption must be diegetic — written in the story's voice. "The bridge where loyalties shatter" (good) vs "An illustration of two characters confronting each other on a bridge" (bad).

R-2.5. Priority ∈ {1 (must-have, score ≥ 5), 2 (important, 3–4), 3 (nice-to-have, 1–2), skip (≤ 0)}.

R-2.6. LLM receives full Passage prose + ArtDirection + EntityVisuals for every entity in `appears` — not just IDs or summaries.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Brief has `caption: "Illustration of Kay entering the archive"` | Non-diegetic | R-2.4 |
| Brief has `targets` edge to a beat | Briefs target passages | R-2.2 |
| Brief has empty `subject` | Required field missing | R-2.3 |
| Brief LLM call receives bare passage IDs | Context enrichment missing | R-2.6 |

### Output Contract

1. Every Passage has exactly one IllustrationBrief node with a `targets` edge.
2. Every Brief has all required fields and a priority score.
3. Captions are diegetic.

---

## Phase 3: Codex Generation

**Purpose:** For each Entity, generate tiered diegetic codex entries gated by state flags. Tier 1 is always visible; higher tiers are gated by the state flags that track entity-affecting choices — enabling spoiler graduation as the player progresses.

### Input Contract

1. Phase 1 Output Contract satisfied (Phase 3 may run in parallel with Phase 2; it only depends on Phase 1's Entity/EntityVisual outputs and the FILL-era state flags).

### Operations

#### Per-Entity Codex Entry Generation

**What:** For each Entity, identify which state flags relate to it (from overlay `when` clauses and from Consequences of paths involving the entity). Plan tier structure: rank 1 visible always, higher ranks gated by state flags. One LLM call per entity produces all tiers at once, with diegetic-voice instructions and self-containment requirements.

**Rules:**

R-3.1. Every Entity has at least one CodexEntry with `HasEntry` edge to the Entity.

R-3.2. Tier 1 (rank = 1) has `visible_when: []` — always visible from the start.

R-3.3. Higher-tier entries (rank ≥ 2) are gated by `visible_when` referencing valid state flag IDs.

R-3.4. Codex entries are diegetic — written in the story's voice, not as technical documentation. "A traveling scholar who offers guidance" (good) vs "Aldric is a character who serves as the protagonist's mentor" (bad).

R-3.5. Each entry is self-contained — readable without prior tiers.

R-3.6. A lower-ranked (earlier-visible) entry must not prematurely disclose content whose reveal is gated behind a higher-ranked tier. The direction of the spoiler is what matters: rank 1 may be deliberately vague and rank 2+ may fully reveal, but rank 1 must not leak information that rank 2+ is supposed to reveal. LLM validation checks for this; violations trigger retry (max 2 per entity).

R-3.7. CodexEntry gating uses **state flag** IDs, not codewords. SHIP projects a subset of state flags as player-facing codewords; DRESS gates internally via state flags. → ontology §Part 8: Codewords ≠ State Flags.

R-3.8. LLM call per entity receives: full Entity description (base + overlays), related state flags with their narrative meanings, tier plan, diegetic-voice instructions, Vision metadata.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Entity has no CodexEntry | Phase 3 skipped | R-3.1 |
| Rank-1 entry has `visible_when: [state_flag::mentor_hostile]` | Rank 1 must be unconditional | R-3.2 |
| Rank-1 entry: "A mysterious scholar who secretly knows your true identity." Rank 2 (gated behind `state_flag::met_aldric`): "Aldric is the traitor who betrayed your mentor." | Rank 1 prematurely disclosed the deception angle — that reveal was supposed to be gated behind rank 2. Direction of spoiler is low tier → high tier content | R-3.6 |
| `visible_when` references `codeword::met_aldric` | Should be a state flag ID; codewords are SHIP's projection | R-3.7 |
| Entry: "Aldric is a character who serves as the protagonist's mentor" | Non-diegetic voice | R-3.4 |

### Output Contract

1. Every Entity has ≥1 CodexEntry via `HasEntry` edge.
2. Rank 1 of every Entity's entries is always visible.
3. Higher tiers gate by valid state flag IDs.
4. All entries are diegetic and self-contained.
5. No lower-tier spoilers.

---

## Phase 4: Human Review (Gate 2)

**Purpose:** Review briefs and codex, set the image generation budget (which briefs to render).

### Input Contract

1. Phases 2 and 3 Output Contracts satisfied.

### Operations

#### Review and Budget Setting

**What:** Present illustration briefs sorted by priority and codex entries grouped by entity. Human approves / edits / skips briefs, sets a rendering budget (e.g., "render all priority 1 and 2," "render top 10," "skip all images"). Codex entries may be edited, added, or removed.

**Rules:**

R-4.1. Human explicitly sets the rendering budget. Silent defaults are forbidden — DRESS does not decide which briefs to render without human input.

R-4.2. Edits to briefs are captured in the brief node. Skipped briefs have `priority: skip` or a corresponding marker.

R-4.3. Codex edits are persisted to the CodexEntry nodes.

R-4.4. Human approval recorded.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| DRESS proceeds to Phase 5 with no recorded budget | Silent default applied | R-4.1 |
| Codex edits not reflected in graph | Edits lost | R-4.3 |

### Output Contract

1. Set of approved IllustrationBrief IDs for rendering (possibly empty — user may skip images entirely).
2. CodexEntry edits persisted.
3. Human approval recorded.

---

## Phase 5: Image Generation

**Purpose:** Render selected briefs into image assets via the configured image provider. Produces Illustration nodes and asset files.

### Input Contract

1. Phase 4 Output Contract satisfied with a non-empty rendering budget. If budget is empty, Phase 5 is skipped.

### Operations

#### Prompt Assembly and Image Generation

**What:** For each selected brief, assemble a prompt from ArtDirection + EntityVisual fragments (for appearing entities) + brief's subject/composition/mood + negative prompt. Generate one sample image first for style confirmation; on approval, render the rest.

**Rules:**

R-5.1. Prompt assembly injects `reference_prompt_fragment` for every Entity in the brief's appearing set so characters/locations look consistent across images.

R-5.2. Sample image is generated for the highest-priority brief first. Human confirms style fit before batch rendering.

R-5.3. On sample rejection, the user may adjust ArtDirection or the brief and re-run — batch does not proceed without confirmation.

R-5.4. On per-brief generation failure, log ERROR with the brief ID and provider error, skip the brief, continue the batch. No automatic retry (image generation is expensive).

R-5.5. Generated images are stored with SHA-256 hash-prefix filenames in `projects/<name>/assets/` enabling deduplication.

R-5.6. Each successfully rendered image produces an Illustration node + `Depicts` edge to the target passage + `from_brief` edge to the source IllustrationBrief.

R-5.7. No orphan assets: every asset file has a corresponding Illustration node, and every Illustration node has an existing asset file.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Character appearance inconsistent across images | `reference_prompt_fragment` not injected | R-5.1 |
| Batch renders without sample confirmation | Gate skipped | R-5.2 |
| Asset file exists without Illustration node | Orphan — either log-and-clean or node creation failed | R-5.7 |
| Illustration node with no corresponding asset file | Broken reference | R-5.7 |

### Output Contract

1. For every successfully rendered brief: an Illustration node + `Depicts` edge + `from_brief` edge + asset file on disk.
2. For every failed brief: ERROR logged; brief marked as not-rendered; no Illustration node.
3. No orphan asset files.

---

## Stage Output Contract

1. Exactly one ArtDirection node exists.
2. Every Entity with ≥1 `appears` edge has an EntityVisual with non-empty `reference_prompt_fragment`.
3. Every Passage has an IllustrationBrief with a `targets` edge.
4. Every Brief has all required fields and a priority score; captions are diegetic.
5. Every Entity has ≥1 CodexEntry with `HasEntry` edge.
6. CodexEntry rank 1 is always visible; higher ranks gated by state flag IDs.
7. All codex entries are diegetic and self-contained; no lower-tier spoilers.
8. Selected Briefs have corresponding Illustration nodes with assets on disk; `Depicts` and `from_brief` edges wired.
9. No prose, passage, choice, beat, entity-core, or state-flag mutations.
10. DRESS may be skipped entirely — no required outputs if human opts out.

## Implementation Constraints

- **Context Enrichment:** Brief generation (Phase 2) must receive full Passage prose + ArtDirection + EntityVisuals for appearing entities. Codex generation (Phase 3) must receive full entity descriptions + related state flag meanings + Vision. Bare IDs produce generic output. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Prompt Context Formatting:** All context blocks (entity visuals, state flag lists, brief lists during review) use human-readable formatting. No Python repr. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Valid ID Injection:** Codex entries' `visible_when` fields must use valid state flag IDs. Provide the full list when calling the LLM for codex generation. → CLAUDE.md §Valid ID Injection Principle
- **Silent Degradation:** Phase 5 per-brief failures log ERROR and skip the brief — the batch does not halt silently. The absence of an Illustration node for a selected brief must be logged. Phase 3 retry exhaustion produces a minimal rank-1-only codex with a WARNING, not silent gaps. → CLAUDE.md §Silent Degradation
- **Small Model Prompt Bias:** If briefs consistently non-diegetic or codex entries consistently bland, fix the prompt first. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- DRESS narrative concept → how-branching-stories-work.md §Part 6: Illustration and Export
- ArtDirection, EntityVisual, IllustrationBrief, Illustration, CodexEntry schemas → story-graph-ontology.md §Part 9: Node Types
- `describes_visual`, `targets`, `from_brief`, `HasEntry`, `Depicts` edges → story-graph-ontology.md §Part 9: Edge Types
- State flags vs codewords distinction → story-graph-ontology.md §Part 8: Codewords ≠ State Flags
- Diegetic voice constraint (codex in-world voice) → how-branching-stories-work.md §Illustration and Reference (DRESS)
- Previous stage → fill.md §Stage Output Contract
- Next stage → ship.md §Stage Input Contract

## Rule Index

R-1.1: Exactly one ArtDirection singleton; retries replace.
R-1.2: ArtDirection has style, medium, palette, composition_notes, style_exclusions, aspect_ratio.
R-1.3: Every entity with `appears` edge has an EntityVisual.
R-1.4: Every EntityVisual has non-empty `reference_prompt_fragment`.
R-1.5: Gate 1 approval required before Phase 2.
R-1.6: DRESS does not mutate Entity base-state.
R-2.1: Every Passage has an IllustrationBrief via `targets` edge.
R-2.2: Briefs target Passages, not beats.
R-2.3: Briefs have all required fields.
R-2.4: Captions are diegetic.
R-2.5: Priority ∈ {1, 2, 3, skip}.
R-2.6: Brief LLM calls receive full Passage prose + ArtDirection + EntityVisuals.
R-3.1: Every Entity has ≥1 CodexEntry via `HasEntry`.
R-3.2: Rank 1 is always visible (`visible_when: []`).
R-3.3: Higher tiers gated by valid state flag IDs.
R-3.4: Entries are diegetic.
R-3.5: Entries are self-contained.
R-3.6: Lower tiers do not contain higher-tier spoilers; retries up to 2 per entity.
R-3.7: Gating uses state flag IDs, not codewords.
R-3.8: Codex LLM calls receive full entity + state flag meanings + Vision.
R-4.1: Human explicitly sets rendering budget; no silent defaults.
R-4.2: Brief edits persisted.
R-4.3: Codex edits persisted.
R-4.4: Gate 2 approval recorded.
R-5.1: Prompt assembly injects `reference_prompt_fragment` per appearing entity.
R-5.2: Sample image first for style confirmation.
R-5.3: Sample rejection blocks batch.
R-5.4: Per-brief failure logs ERROR, skips, continues batch.
R-5.5: Asset files hash-prefixed in projects/<name>/assets/.
R-5.6: Successful renders produce Illustration + Depicts + from_brief.
R-5.7: No orphan assets; every asset has a node, every node has an asset.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Art Direction (Gate 1) | Required — approve ArtDirection + EntityVisuals |
| 4 | Review (Gate 2) | Required — set rendering budget, approve codex |
| 5 | Sample image | Required — confirm style before batch; can loop back to Phase 1 or brief edit |

**Skipping DRESS entirely:** The human may set the Phase 4 rendering budget to empty and opt out of codex edits. The story remains playable — SHIP handles the absence gracefully.

## Iteration Control

**Forward flow:** 1 → 2 + 3 (parallel) → 4 → 5 (if budget non-empty).

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| Gate 1 | Phase 1 discuss | Art direction rejected |
| Gate 2 | Phase 2 or 3 | Briefs or codex need regeneration |
| Phase 5 sample | Phase 1 or brief edit | Style mismatch on first image |

**Maximum iterations:**

- Phase 1: up to 3 validation retries (standard three-phase pattern).
- Phase 2: no retry loop; failed briefs get minimal-subject fallback.
- Phase 3: max 2 retries per entity if validation fails; fallback to rank-1-only codex.
- Phase 5: no automatic retry on API failure; user may re-run Phase 5 for unrendered briefs.

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Style rejected at Gate 1 | Human review | Re-enter discuss; full Phase 1 re-run |
| 2 | Brief validation fails | Validator check | Minimal brief (subject only) with WARNING |
| 3 | Spoiler detected in low tier | Validator / LLM | Up to 2 retries; fallback to rank-1-only with WARNING |
| 5 | Image provider API failure | Provider response | Log ERROR, skip brief, continue batch |
| 5 | Sample image rejected | Human review | Adjust ArtDirection or brief; re-run |

## Context Management

**Standard (≥128k context):** Full Vision + entity list + per-phase context comfortable in context window. Entity count is typically ~10–20; fits in a single prompt.

**Constrained (~32k context):** Phase 1 entity visuals may be generated in batches (5–10 entities per call). Phase 2 per-passage briefs run one call at a time (already bounded). Phase 3 per-entity codex runs one call at a time.

## Worked Example

### Starting Point (FILL output)

- 12 Passages with prose
- 8 Entities (4 characters, 3 locations, 1 object)
- Voice Document, state flags, overlays all in place

### Phase 1

Human + LLM discuss visual style. Final:

```yaml
art_direction:
  style: "ink-wash with muted color tint"
  medium: "digital painting mimicking traditional ink"
  palette: "deep blues and greys, warm amber accents"
  composition_notes: "dramatic lighting, low-angle compositions for tension"
  # Story-tone visual prohibitions only — renderer-quality fillers (blurry, watermark, etc.) are auto-injected at render time.
  style_exclusions: "no photorealism, no cartoon styling, no modern clothing"
  aspect_ratio: "3:2"
```

8 EntityVisual nodes created with reference_prompt_fragments. Gate 1 approved.

### Phase 2 + 3 (parallel)

12 IllustrationBriefs generated, 4 priority-1, 5 priority-2, 3 priority-3. Sample caption: "Where the archive keeps its oldest silence."

8 Entities get tiered codex entries (mostly 2–3 tiers each).

### Phase 4 (Gate 2)

Human reviews. Budget: "render all priority-1 and priority-2" (9 images). Codex approved with minor edits to 2 entries.

### Phase 5

Sample image for highest-priority brief generated — approved. Remaining 8 rendered in batch. All 9 Illustration nodes + assets created.

DRESS complete.
