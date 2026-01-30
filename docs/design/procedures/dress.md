# QuestFoundry v5 — DRESS Algorithm Specification

**Status:** Specification Complete
**Parent:** docs/design/00-spec.md
**Purpose:** Detailed specification of the DRESS stage mechanics

---

## Summary

**Purpose:** Generate the presentation layer — art direction, illustrations, and codex — for a completed story. DRESS operates on finished prose and entities, adding visual and encyclopedic content without modifying narrative structure.

**Input artifacts:**
- Completed story graph (all passages with prose, entities with details)
- Vision metadata (genre, tone, themes)
- Codewords (for codex visibility gating)

**Output artifacts:**
- ArtDirection node (global visual identity)
- EntityVisual nodes (per-entity appearance profiles)
- IllustrationBrief nodes (structured image prompts with priorities)
- Illustration nodes (generated image assets)
- CodexEntry nodes (diegetic encyclopedia entries, spoiler-graduated)
- Depicts edges (illustration → passage)
- HasEntry edges (codex_entry → entity)
- Asset files (`projects/<name>/assets/`)

**Mode:** LLM-heavy generation (art direction, briefs, codex) + image provider generation (illustrations). Two human gates. Optional stage — story is playable without DRESS.

---

## Prerequisites

### Required Input State

| Data | Required State |
|------|----------------|
| Vision | Complete (approved) |
| Passages | All have prose (FILL complete) |
| Entities | Complete with details (FILL updates applied) |
| Codewords | All defined (GROW complete) |
| Appears edges | Present (entity-to-passage links from GROW) |

### Required Human Decisions from Prior Stages

- FILL approved (all prose reviewed)
- All prior human gates passed

### Required Configuration

| Config | Purpose |
|--------|---------|
| `providers.image` | Image generation provider (e.g., `openai/gpt-image-1`) |
| `OPENAI_API_KEY` | Required if using OpenAI image provider |

---

## Core Concepts

### Art Direction Document

The art direction document captures the visual identity of the story. It governs all illustration generation, analogous to how FILL's voice document governs all prose generation.

Created through the standard discuss/summarize/serialize pattern, where user and LLM collaborate on visual style before any images are generated.

**What it contains:**

| Field | Purpose |
|-------|---------|
| `style` | Art style (watercolor, digital painting, ink sketch, pixel art) |
| `medium` | What it looks like it was made with |
| `palette` | Dominant colors and mood |
| `composition_notes` | Framing preferences |
| `negative_defaults` | Things to avoid across all images |
| `aspect_ratio` | Default image dimensions |

### Entity Visual Profiles

Entities (characters, locations, objects) must look consistent across all illustrations. EntityVisual nodes store per-entity appearance descriptions and a `reference_prompt_fragment` that is injected into every image prompt featuring that entity.

The Appears edges from GROW tell DRESS which entities appear in each passage. For each illustration brief, DRESS:
1. Reads the passage's Appears edges to find entities
2. Loads each entity's EntityVisual node
3. Injects `reference_prompt_fragment` into the image prompt

This ensures a character looks the same whether they appear in the opening scene or the climax.

### Diegetic Constraint

Both illustration captions and codex entries must be **diegetic** — written in the story's voice, as if they are part of the world.

**Captions:**
- Good: "The bridge where loyalties shatter"
- Bad: "An illustration of two characters confronting each other on a bridge"

**Codex:**
- Good: "A traveling scholar who offers guidance to those in need."
- Bad: "Aldric is a character who serves as the protagonist's mentor."

This constraint is enforced through prompt instructions and validated during human review at Gate 2.

### Illustration Priority Scoring

Not all passages need illustrations. Image generation is expensive, so DRESS assigns a priority score to each passage using a hybrid approach: structural rules provide a stable base, LLM judgment adds narrative sensitivity.

**Structural base score:**

| Factor | Score |
|--------|-------|
| Spine passage | +3 |
| Branch opening (first passage after a choice) | +2 |
| Ending passage (no outgoing choices) | +2 |
| Climax scene_type | +2 |
| New location introduction | +1 |
| Transition / micro_beat scene_type | -1 |

**LLM adjustment (applied per passage):**

| Factor | Score |
|--------|-------|
| Visually striking scene | +1 |
| Emotionally pivotal moment | +1 |
| Dialogue-heavy / static scene | -1 |

**Priority mapping:**

| Total Score | Priority | Label |
|-------------|----------|-------|
| ≥ 5 | 1 | Must-have |
| 3–4 | 2 | Important |
| 1–2 | 3 | Nice-to-have |
| ≤ 0 | — | Skip (no brief generated) |

### Cumulative Codex Model

Each entity gets multiple codex entries, ranked and gated by codewords. This enables spoiler graduation — players learn more about entities as they progress through the story.

**Key properties:**
- **Cumulative:** SHIP shows all unlocked entries, not just the highest
- **Ranked:** Entries display in rank order (1 = base knowledge)
- **Self-contained:** Each entry is readable without prior tiers
- **Diegetic:** Written in the story's voice

**Tier 1** (rank=1) always has `visible_when: []` — it's visible from the start. Higher tiers are gated by codewords earned through gameplay.

The LLM generates all tiers at once, with instruction to minimize redundancy across tiers while keeping each self-contained.

### Image Provider Abstraction

DRESS uses a provider-independent interface for image generation. LangChain Python has no `BaseImageModel` — only a DALL-E tool wrapper — so QuestFoundry defines its own thin protocol.

```python
class ImageProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        aspect_ratio: str = "1:1",
        quality: str = "standard",
    ) -> ImageResult: ...

@dataclass
class ImageResult:
    image_data: bytes
    content_type: str       # "image/png", "image/webp"
    provider_metadata: dict  # provider-specific info (model, seed, etc.)
```

Single provider per project. The prompt assembly layer translates structured IllustrationBrief data into a provider-specific prompt string.

---

## Algorithm Phases

### Phase 0: Art Direction

**Purpose:** Establish the story's visual identity through collaborative exploration.

**Input:**
- Vision metadata (genre, tone, themes)
- Entity list (all entities with their base descriptions)

**Operations:**

Uses the standard **discuss/summarize/serialize** pattern:

1. **Discuss:** High-temperature exploration of visual style. LLM proposes styles, user reacts. Tools available for style research (genre conventions, art references). Discussion covers:
   - Overall art style and medium
   - Color palette and mood
   - Composition preferences
   - Per-entity appearance (characters, key locations, significant objects)
   - What to avoid (negative prompt defaults)

2. **Summarize:** Distill discussion into a narrative art brief. Captures the agreed visual direction in prose form.

3. **Serialize:** Convert to structured output:
   - One ArtDirection node (global style)
   - One EntityVisual node per entity (appearance profile + prompt fragment)
   - `describes_visual` edges linking each EntityVisual to its entity

**Output:**
- ArtDirection node
- EntityVisual[] nodes
- `describes_visual` edges

**Human Gate:** YES — **Gate 1**
- Review global art direction document
- Review entity visual profiles (especially `reference_prompt_fragment`)
- Ensure consistency between entity descriptions and art style
- Can request regeneration or manual edits

**LLM Involvement:** Generate + Validate (standard three-phase)

**Completion Criteria:**
- ArtDirection node created with all fields populated
- EntityVisual node created for every entity
- Each EntityVisual has a non-empty `reference_prompt_fragment`
- Human approval at Gate 1

---

### Phase 1: Illustration Brief Generation

**Purpose:** Generate structured image prompts for all passages, with priority scoring.

**Input:**
- All passages (with prose)
- ArtDirection node
- EntityVisual[] nodes
- Appears edges (entity → passage)

**Operations:**

For each passage:

1. **Read context:** passage prose, scene_type (from beat), position in graph (spine/branch/ending)
2. **Identify entities:** Follow Appears edges to find all entities in this passage
3. **Load visuals:** Read EntityVisual for each appearing entity
4. **Compute structural score:** Apply base scoring rules (see Core Concepts)
5. **Generate brief:** LLM call with passage context, entity visuals, and art direction. LLM produces:
   - `subject`: What the image depicts
   - `composition`: Framing and camera notes
   - `mood`: Emotional tone
   - `category`: scene | portrait | vista | item_detail
   - `caption`: Diegetic caption
   - `negative`: Image-specific things to avoid
   - `style_overrides`: Deviations from global art direction (usually empty)
   - LLM priority adjustment (visually striking, emotionally pivotal, etc.)
6. **Compute final priority:** Combine structural base + LLM adjustment
7. **Create IllustrationBrief node** with `targets` edge to passage

**Batching:** Passages can be processed in parallel batches. Brief generation is cheap (text LLM calls).

**Output:**
- IllustrationBrief[] nodes (one per passage)
- `targets` edges (brief → passage)

**Human Gate:** NO (deferred to Gate 2)

**LLM Involvement:** Generate (one call per passage)

**Completion Criteria:**
- IllustrationBrief created for every passage
- All briefs have valid priority scores
- All briefs have non-empty subject, composition, and caption

---

### Phase 2: Codex Generation

**Purpose:** Generate diegetic, spoiler-graduated encyclopedia entries for all entities.

**Input:**
- All entities (with base descriptions and overlays)
- All codewords (with `tracks` links to consequences)
- Vision metadata (genre, tone for voice)

**Operations:**

For each entity:

1. **Analyze codeword gates:** Identify which codewords relate to this entity:
   - Codewords that appear in the entity's overlay `when` clauses
   - Codewords tracked by consequences involving this entity's paths
2. **Determine tier structure:** Plan which knowledge is revealed at each gate:
   - Rank 1: Always visible — basic introduction (who/what is this?)
   - Rank 2+: Gated by codewords — progressively deeper knowledge
   - Number of tiers depends on entity complexity and available codewords
3. **Generate entries:** Single LLM call per entity produces all tiers. The prompt includes:
   - Entity base description and overlays (full knowledge)
   - Available codewords with their narrative meanings
   - Tier plan (which codewords gate which rank)
   - Diegetic constraint instructions
   - Vision metadata for voice consistency
4. **Validate:** Each entry is self-contained, diegetic, and doesn't spoil content from higher tiers
5. **Create CodexEntry nodes** with `HasEntry` edges to entity

**Parallelism:** Codex generation runs in parallel with Phase 1 (illustration briefs). Both read from the same graph state and create different node types.

**Output:**
- CodexEntry[] nodes (multiple per entity)
- `HasEntry` edges (codex_entry → entity)

**Human Gate:** NO (deferred to Gate 2)

**LLM Involvement:** Generate (one call per entity)

**Completion Criteria:**
- Every entity has at least one codex entry (rank 1, always visible)
- All entries are diegetic (in-world voice)
- Tier codeword gates are valid (codewords exist in graph)
- No spoilers in lower-ranked tiers

---

### Phase 3: Human Review (Gate 2)

**Purpose:** Review all generated content and set the image generation budget.

**Input:**
- IllustrationBrief[] (sorted by priority)
- CodexEntry[] (grouped by entity, sorted by rank)

**Operations:**

Present to user for review:

1. **Illustration briefs:**
   - Displayed sorted by priority (must-have first)
   - Show: subject, composition, mood, category, caption, priority score
   - User can: approve, edit, skip, reprioritize
   - User sets **image generation budget**: number of images or priority cutoff
     (e.g., "render all priority 1+2", "render top 10", "skip all")

2. **Codex entries:**
   - Displayed grouped by entity, sorted by rank within each group
   - Show: entity name, rank, visible_when, content
   - User can: approve, edit, add/remove tiers, adjust codeword gates

**Output:**
- Set of selected brief IDs (approved for rendering)
- Approved codex entries (may be edited)

**Human Gate:** YES — **Gate 2**

**LLM Involvement:** None

**Completion Criteria:**
- User has explicitly approved or skipped image generation
- Codex entries reviewed (at minimum, spot-checked)

---

### Phase 4: Image Generation

**Purpose:** Render selected illustration briefs into image assets.

**Input:**
- Selected IllustrationBrief[] (from Gate 2)
- ArtDirection node
- EntityVisual[] nodes (for prompt assembly)

**Operations:**

1. **Prompt assembly:** For each selected brief:
   - Assemble entity descriptions: join `reference_prompt_fragment` for each entity in `entities[]`
   - Combine with action: append brief's `subject`, `composition`, and `mood`
   - Apply global style: append ArtDirection's `style`, `medium`, and `palette`
   - Apply overrides: append any `style_overrides` from the brief
   - Set negative prompt: combine brief's `negative` with ArtDirection's `negative_defaults`
   - Format for provider (e.g., DALL-E expects a single text prompt; A1111 expects separate positive/negative)

2. **Sample generation:**
   - Generate ONE image from the highest-priority brief
   - Present to user for style confirmation
   - If rejected: user can adjust art direction or brief, then retry
   - If approved: proceed to batch

3. **Batch generation:**
   - Generate remaining selected briefs
   - Track progress (X of Y complete)
   - On failure: log error, skip brief, continue batch
   - No retry by default (image generation is expensive; user can re-run)

4. **Asset storage:**
   - Compute SHA-256 of image data
   - Store as `projects/<name>/assets/<sha256_prefix>.<ext>`
   - Hash-based naming enables deduplication

5. **Graph updates:**
   - Create Illustration node (id, asset path, caption from brief, category from brief)
   - Create Depicts edge (illustration → passage, via brief's targets edge)
   - Create from_brief edge (illustration → illustration_brief)

**Output:**
- Illustration[] nodes
- Asset files on disk
- Depicts edges (illustration → passage)
- from_brief edges (illustration → illustration_brief)

**Human Gate:** Implicit (sample confirmation before batch)

**LLM Involvement:** None (image provider only)

**Completion Criteria:**
- All selected briefs either rendered or logged as failed
- Assets stored with correct hash-based naming
- Illustration nodes and Depicts edges created for all successful renders
- No orphan assets (every file has a corresponding Illustration node)

---

## Human Gates Summary

| Gate | After Phase | Reviews | Blocking? | Can Loop Back? |
|------|-------------|---------|-----------|----------------|
| **Gate 1** | Phase 0 (Art Direction) | Global style + entity visuals | Yes | To Phase 0 discuss |
| **Gate 2** | Phase 1+2 (Briefs + Codex) | Priority list + budget + codex | Yes | To Phase 1 or 2 |
| **Sample** | Phase 4 first image | Style confirmation | Yes | To Phase 0 or brief edit |

**Skipping DRESS entirely:** User can skip DRESS without affecting story playability. SHIP handles the absence of illustrations and codex gracefully.

---

## Iteration Control

### Art Direction (Phase 0)
- Standard discuss/summarize/serialize with max 3 validation retries
- If art direction is rejected at Gate 1, re-enter discuss phase (full re-run)

### Brief Generation (Phase 1)
- No retry loop — briefs are generated once per passage
- If a brief fails validation, log warning and create a minimal brief (subject only)

### Codex Generation (Phase 2)
- Max 2 retries per entity if validation fails (e.g., spoiler detected in low tier)
- If retries exhausted, create minimal codex (rank 1 only, basic description)

### Image Generation (Phase 4)
- No automatic retry (expensive operation)
- On API failure: log error, mark brief as "not rendered", continue batch
- User can re-run Phase 4 with remaining unrendered briefs

---

## Context Management

### Phase 0: Art Direction

| Context | Tokens (est.) |
|---------|---------------|
| Vision metadata | ~200 |
| Entity list with descriptions | ~500–2,000 (depends on entity count) |
| Style discussion history | ~2,000–5,000 |
| Output buffer | ~500 |

### Phase 1: Illustration Brief Generation (per passage)

| Context | Tokens (est.) |
|---------|---------------|
| Art direction document | ~300 |
| Passage prose | ~300–500 |
| Entity visuals (for appearing entities) | ~200–500 |
| Scene type and graph position | ~100 |
| Output buffer | ~300 |

Fits comfortably in modern context windows. For stories with many entities per passage, entity visuals may be summarized.

### Phase 2: Codex Generation (per entity)

| Context | Tokens (est.) |
|---------|---------------|
| Entity full description (base + overlays) | ~200–500 |
| Related codewords with meanings | ~200–400 |
| Vision metadata (for voice) | ~200 |
| Output buffer (all tiers) | ~500–1,000 |

---

## Worked Example

### Setup

A minimal story with 3 passages and 2 entities:

- **Passages:** opening → bridge_confrontation → ending
- **Entities:** protagonist (character), mentor_aldric (character)
- **Codewords:** met_aldric, discovered_betrayal

### Phase 0: Art Direction

Discuss produces:
```yaml
art_direction:
  id: art_direction::main
  style: "ink wash with watercolor accents"
  medium: "traditional Japanese sumi-e"
  palette: ["deep indigo", "rust orange", "ash grey"]
  composition_notes: "Wide shots for landscapes, close-up for emotional moments"
  negative_defaults: "photorealistic, text, modern clothing, bright colors"
  aspect_ratio: "16:9"
```

Entity visuals:
```yaml
entity_visual:
  id: ev::protagonist
  description: "Young woman in her 20s, short dark hair, determined expression"
  distinguishing_features: ["jade pendant", "ink-stained fingers"]
  color_associations: ["deep indigo", "jade green"]
  reference_prompt_fragment: "young woman, short dark hair, jade pendant, ink-stained fingers, determined expression"

entity_visual:
  id: ev::mentor_aldric
  description: "Tall gaunt man in his 60s, silver-streaked beard, scar over left eye"
  distinguishing_features: ["scar across left eyebrow", "worn leather satchel"]
  color_associations: ["deep burgundy", "aged gold"]
  reference_prompt_fragment: "tall gaunt elderly man, silver beard, scar over left eye, burgundy robes, leather satchel"
```

### Phase 1: Illustration Briefs

For `bridge_confrontation` (spine passage, climax scene_type):
- Structural score: spine(+3) + climax(+2) = 5
- LLM adjustment: emotionally pivotal(+1) = +1
- Total: 6 → **Priority 1 (must-have)**

```yaml
illustration_brief:
  id: ib::bridge_confrontation
  priority: 1
  category: scene
  subject: "Protagonist and Aldric face each other on the crumbling stone bridge at twilight"
  entities: [entity::protagonist, entity::mentor_aldric]
  composition: "wide shot, two figures silhouetted against sunset, bridge spanning a gorge"
  mood: "tense, bittersweet"
  style_overrides: {}
  negative: "modern architecture, happy expressions"
  caption: "The bridge where loyalties shatter"
```

### Phase 2: Codex

For `mentor_aldric`:
```yaml
codex_entry:
  id: codex::aldric_basic
  rank: 1
  visible_when: []
  content: "A traveling scholar who offers guidance to those in need."

codex_entry:
  id: codex::aldric_background
  rank: 2
  visible_when: [met_aldric]
  content: "Claims to be a former court advisor, exiled for speaking truth to power."

codex_entry:
  id: codex::aldric_truth
  rank: 3
  visible_when: [discovered_betrayal]
  content: "His exile was self-imposed — he left the court after orchestrating the king's downfall."
```

### Phase 3: Gate 2

User reviews briefs (sorted by priority) and sets budget: "render all priority 1" (1 image).
User reviews codex entries and approves.

### Phase 4: Image Generation

1. Assemble prompt for `ib::bridge_confrontation`:
   ```
   young woman, short dark hair, jade pendant, ink-stained fingers, determined expression
   and tall gaunt elderly man, silver beard, scar over left eye, burgundy robes, leather satchel
   face each other on the crumbling stone bridge at twilight,
   wide shot, two figures silhouetted against sunset, bridge spanning a gorge,
   tense, bittersweet,
   ink wash with watercolor accents, traditional Japanese sumi-e style,
   deep indigo, rust orange, ash grey palette
   ```
   Negative: `modern architecture, happy expressions, photorealistic, text, modern clothing, bright colors`

2. Generate sample → user confirms style
3. (Only one image in budget, so batch = done)
4. Store: `assets/a3f8c2...png`
5. Create Illustration node + Depicts edge to `bridge_confrontation`

---

## Failure Modes

### Image Provider Failure

| Failure | Recovery |
|---------|----------|
| API timeout | Log, skip brief, continue batch |
| Rate limiting | Back off, retry once, then skip |
| Content policy rejection | Log prompt, mark brief as "blocked", continue |
| Authentication failure | Abort batch, report error |

**No automatic retry for cost reasons.** User can re-run Phase 4 targeting only unrendered briefs.

### Art Direction Too Vague

If serialization produces an ArtDirection with empty or generic fields (e.g., `style: "good"`), the validation loop retries with feedback specifying which fields need concrete values. Max 3 retries.

### Entity Visual Inconsistency

Gate 1 is the primary defense. If entity visuals are inconsistent (e.g., protagonist described differently from their entity base description), the user catches this during review and either edits directly or requests re-generation.

### Codex Spoiler Leak

If a low-ranked codex entry reveals information gated by codewords in a higher-ranked entry, validation flags it. Recovery: regenerate the entity's codex entries with explicit spoiler ordering instructions. Max 2 retries.

### Budget Exceeded

If the user-selected budget would exceed a configured cost limit (if set), warn before proceeding. The user can reduce the selection or override.

---

## Design Principle: Visual Identity Serves the Narrative

Art direction is a **constraint document**, not creative freedom. Like the voice document in FILL, the art direction document exists to ensure consistency, not to showcase art.

Every visual element must serve the story:
- **Illustrations** show what the player is experiencing, not abstract beauty
- **Captions** are part of the world, not commentary on it
- **Codex entries** are the world speaking to the player, not the author explaining the world
- **Entity visuals** exist for consistency, not for their own sake

DRESS is optional. A story without illustrations is still a complete story. DRESS adds atmosphere, not structure.

---

## Output Checklist

Before DRESS is complete:

- [ ] ArtDirection node exists with all fields populated
- [ ] EntityVisual exists for every entity with non-empty `reference_prompt_fragment`
- [ ] IllustrationBrief exists for every passage with valid priority
- [ ] Illustration exists for every selected (rendered) brief
- [ ] Every Illustration has a Depicts edge to its passage
- [ ] Every Illustration has a from_brief edge to its IllustrationBrief
- [ ] Every Illustration has a diegetic caption
- [ ] Every entity has at least one CodexEntry (rank 1, visible_when: [])
- [ ] Every CodexEntry has a HasEntry edge to its entity
- [ ] All codex content is diegetic
- [ ] No spoiler leaks in low-ranked codex entries
- [ ] All assets stored in `projects/<name>/assets/` with hash-based naming
- [ ] Human gates passed (Gate 1 + Gate 2 + sample confirmation)

---

## Summary

| Phase | Purpose | Creates | Cost | Gate? |
|-------|---------|---------|------|-------|
| 0: Art Direction | Visual identity | ArtDirection, EntityVisual[] | Cheap (LLM) | Gate 1 |
| 1: Illustration Briefs | Image prompts | IllustrationBrief[] | Cheap (LLM) | — |
| 2: Codex | Encyclopedia | CodexEntry[] | Cheap (LLM) | — |
| 3: Human Review | Budget + approval | (edits only) | Free | Gate 2 |
| 4: Image Generation | Render assets | Illustration[] | Expensive (image API) | Sample |
