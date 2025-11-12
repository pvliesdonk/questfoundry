# Layer 0 — North Star (Navigator)

Layer 0 is the **human map** of QuestFoundry: what this studio is, how it thinks, and how the parts
talk without ever leaking spoilers to players. If you only read one folder before touching anything
else, read this one.

> Scope: concepts, roles, loops, safety bars, and operating model.  
> Not here: schemas, protocol messages, prompts, code. Those live in later layers.

---

## How to read this layer (recommended path)

1. **Working model** — the studio at a glance  
   → `WORKING_MODEL.md`
2. **Sources of Truth** — Hot vs Cold, snapshots, views  
   → `SOURCES_OF_TRUTH.md` and `EVERGREEN_MANUSCRIPT.md`
3. **Roles** — who does what, dormancy policy  
   → `ROLE_INDEX.md`
4. **Safety rails** — Quality Bars, PN boundaries, spoiler & accessibility rules  
   → `QUALITY_BARS.md`, `PN_PRINCIPLES.md`, `SPOILER_HYGIENE.md`,
   `ACCESSIBILITY_AND_CONTENT_NOTES.md`
5. **Doing the work** — targeted loops & playbooks  
   → `LOOPS/README.md` and `PLAYBOOKS/README.md`
6. **Traceability** — TUs, merges, views  
   → `TRACEABILITY.md`

---

## The studio in one paragraph

The **Showrunner** coordinates **targeted loops** (small, focused passes). Makers propose **hooks**
in **Hot**; the **Lore Weaver** turns accepted hooks into canon; the **Codex Curator** publishes
player-safe summaries; the **Scene Smith** writes prose to the **Plotwright**’s topology; the
**Style Lead** keeps voice tight; **Art/Audio Directors** may add plans or assets
(optional/dormant); the **Translator** maintains language slices. The **Gatekeeper** runs **Quality
Bars** before anything merges to **Cold**. A **Book Binder** cuts **views** from Cold; the
**Player-Narrator (PN)** performs them **in-world** and spoiler-safe.

---

## Sources of Truth (SoT)

- **Hot** — discovery space: drafts, hooks, canon notes, plans.
- **Cold** — curated: canon & player-safe surfaces approved by Gatekeeper.
- **Snapshots** — immutable tags of Cold; exports are **views** on snapshots.

Read: `SOURCES_OF_TRUTH.md` and `EVERGREEN_MANUSCRIPT.md`.

---

## Roles (canon names, dormancy)

- Always on: **Showrunner**, **Gatekeeper**
- Default on: **Plotwright**, **Scene Smith**, **Style Lead**, **Lore Weaver**, **Codex Curator**
- Optional/dormant: **Researcher**, **Art Director**, **Illustrator**, **Audio Director**, **Audio
  Producer**, **Translator**
- Consumers (downstream): **Book Binder**, **Player-Narrator (PN)**

Details: `ROLE_INDEX.md` and `PN_PRINCIPLES.md`.

---

## Quality Bars (what must be true to merge)

- **Integrity** (no dead refs), **Reachability** (keystones reachable),
- **Nonlinearity** (hubs/loops matter), **Gateways** (coherent diegetic checks),
- **Style** (voice/register/motifs), **Determinism** (when promised for assets),
- **Presentation** (no spoilers or technique on surfaces; accessibility).

See: `QUALITY_BARS.md`, `SPOILER_HYGIENE.md`, `ACCESSIBILITY_AND_CONTENT_NOTES.md`.

---

## Targeted loops (how work gets done)

Each loop is a short, goal-driven pass with clear hand-offs. Start from the **index**:

- Index: `LOOPS/README.md`
- Full guides:
  - `LOOPS/full_production_run.md`
  - `LOOPS/story_spark.md`
  - `LOOPS/hook_harvest.md`
  - `LOOPS/lore_deepening.md`
  - `LOOPS/codex_expansion.md`
  - `LOOPS/style_tune_up.md`
  - `LOOPS/art_touch_up.md`
  - `LOOPS/audio_pass.md`
  - `LOOPS/translation_pass.md`
  - `LOOPS/binding_run.md`
  - `LOOPS/narration_dry_run.md`

Prefer the **Playbooks** for a one-page, checkbox-ready version: `PLAYBOOKS/README.md`.

---

## Traceability (TU-first changes)

Every meaningful change has a **Trace Unit (TU)** with: purpose, inputs, bar pressure, hand-offs,
and merge status. Gatekeeper passes → **Showrunner** merges to **Cold** → a Binder can cut a
**view**.

Read: `TRACEABILITY.md`.  
Repo-wide contribution flow: `/CONTRIBUTING.md`. Structural changes? Use ADRs in `/DECISIONS`.

---

## Spoiler hygiene & PN boundaries

- Player surfaces never show **codeword names**, **gate logic**, **seeds/models**, or canon twists.
- PN enforces gates **diegetically** (token/reputation/knowledge/physical), never with mechanics
  talk.

Details live in `SPOILER_HYGIENE.md` and `PN_PRINCIPLES.md`.

---

## Accessibility

Alt text, descriptive links, high contrast, motion/audio safety, and localized text equivalents are
**baseline**, not extras. See `ACCESSIBILITY_AND_CONTENT_NOTES.md`.

---

## Quick start (15 minutes)

1. Skim `WORKING_MODEL.md`.
2. Open a TU for a tiny **Story Spark** on a test chapter.
3. Run `Hook Harvest` → `Lore Deepening` → `Codex Expansion`.
4. Gatecheck, then **Binding Run** a view.
5. Do a **Narration Dry-Run** and file follow-up TUs.

---

## What this layer is **not**

- A schema or protocol. Those are Layers **3–4**.
- An AI prompt kit. That’s Layer **5**.
- Code or UI. Layers **6–7**.  
  We keep them visible as first-class directories so readers see the runway, but Layer 0 stays human
  and implementation-agnostic.

---

## Directory (Layer 0)

```

00-north-star/
README.md
WORKING_MODEL.md
ROLE_INDEX.md
PN_PRINCIPLES.md
SOURCES_OF_TRUTH.md
QUALITY_BARS.md
TRACEABILITY.md
EVERGREEN_MANUSCRIPT.md
SPOILER_HYGIENE.md
ACCESSIBILITY_AND_CONTENT_NOTES.md
LOOPS/
README.md
*.md
PLAYBOOKS/
README.md
playbook_*.md

```

Layer 0 is the compass. When in doubt, come back here, pick a loop, and move one clean step forward.
