# Layer 1 — Roles (Charters, Briefs, and Interfaces)

Layer 1 defines **who does what** in QuestFoundry. It’s human-readable, implementation-agnostic, and
deliberately light on restrictions so creators can breathe. Each role gets a **charter** (mission,
scope, authorities, inputs/outputs) and an **agent brief** (practical heuristics). A few **interface
notes** describe how key pairs collaborate.

> No schemas. No prompts. No message formats. Those belong to later layers. Layer 1 must stand on
> its own and point back to Layer-0 policies when needed.

---

## What’s in this layer

```

01-roles/
README.md
_templates/
ROLE_CHARTER.template.md    # Meta-templates for creating new roles
AGENT_BRIEF.template.md
charters/
showrunner.md
gatekeeper.md
plotwright.md
scene_smith.md
lore_weaver.md
codex_curator.md
style_lead.md
researcher.md            # optional/dormant
art_director.md          # optional/dormant
illustrator.md           # optional/dormant
audio_director.md        # optional/dormant
audio_producer.md        # optional/dormant
translator.md            # optional/dormant
book_binder.md
player_narrator.md
briefs/
<one file per role, mirrors charters/>
interfaces/
pair_guides.md
dormancy_signals.md
escalation_rules.md

```

- **Charters** say _why the role exists, what's in/out of scope, and what it hands off_.
- **Briefs** give _how-to heuristics_ for day-to-day execution (still human, not prompts).
- **Interfaces** describe _the handshake_ between specific roles without dictating wire formats.

> **Note on work artifact structures:** Each role produces specific artifacts (hooks, TUs, canon
> packs, etc.). The **structure** of these artifacts (what fields they have) lives in Layer 2
> (`../02-dictionary/artifacts/`). Role charters/briefs reference which artifacts each role
> produces, but defer to Layer 2 for the actual structure definitions.

---

## Normative references (from Layer 0)

Every charter/brief **MUST** link back to these:

- **Quality Bars** — `00-north-star/QUALITY_BARS.md`
- **PN Principles** — `00-north-star/PN_PRINCIPLES.md`
- **Spoiler Hygiene** — `00-north-star/SPOILER_HYGIENE.md`
- **Accessibility & Content Notes** — `00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- **Sources of Truth** (Hot/Cold) — `00-north-star/SOURCES_OF_TRUTH.md`
- **Traceability** (TUs, snapshots, views) — `00-north-star/TRACEABILITY.md`

Layer-1 documents **SHOULD NOT** restate these policies; they **SHOULD** reference them.

---

## Canon roles (scope in a sentence)

- **Showrunner** — scopes work, wakes dormant roles, sequences loops, merges, and exports.
- **Gatekeeper** — enforces the Quality Bars before anything touches Cold.
- **Plotwright** — designs hubs/loops/gateways; maintains topology intent and briefs.
- **Scene Smith** — writes/rewrites section prose to briefs and style.
- **Lore Weaver** — turns accepted hooks into spoiler-level canon and downstream notes.
- **Codex Curator** — publishes player-safe entries and crosslinks (never invents canon).
- **Style Lead** — maintains voice/register/motifs and phrasing patterns.
- **Researcher** _(optional)_ — corroborates facts; marks uncertainty when dormant.
- **Art Director / Illustrator** _(optional)_ — plan and/or render illustrations.
- **Audio Director / Producer** _(optional)_ — plan and/or produce sound.
- **Translator** _(optional)_ — builds language packs; localized surfaces.
- **Book Binder** — assembles export views from Cold snapshots.
- **Player-Narrator (PN)** — performs the book in-world; enforces gates diegetically.

Dormancy policy and wake signals are documented in `interfaces/dormancy_signals.md`.

---

## Creative freedom (important)

- Charters describe the **edges of the sandbox**, not your every move.
- Briefs use **heuristics** (“prefer small, testable steps”, “keep choices contrastive”) rather than
  rules you could trip over.
- Any role **MAY** propose **hooks**; the Showrunner’s **Hook Harvest** will triage them.
- If a better idea appears mid-loop, **SHOULD** capture it as a hook and keep the current TU small.

---

## How to use Layer 1

1. **Start with charters** for the roles you’ll actually wake this sprint.
2. **Skim the matching briefs** to align on working heuristics.
3. **Check pair guides** before heavy hand-offs (Plotwright↔Scene, Lore↔Codex,
   Style↔PN/Translator, Directors↔Producers, Binder↔PN).
4. Keep a TU open in Hot; when done, go through Gatekeeper and merge to Cold.

---

## Drafting order (recommended)

1. Showrunner → 2) Gatekeeper → 3) Plotwright → 4) Scene Smith → 5) Style Lead →
2. Lore Weaver → 7) Codex Curator → 8) Book Binder → 9) Player-Narrator →
3. Optional roles: Researcher, Art/Audio (Director/Producer), Translator →
4. Interfaces (pair guides, dormancy signals, escalation rules)

We’ll draft these one by one to keep quality high.

---

## Contribution & traceability

- Use a **TU** (`tu-<topic>-<date>`) for each charter/brief you add or revise.
- If you propose renaming a role or changing its authority, that’s an **ADR** (`/DECISIONS`), not
  just a TU.
- Keep examples **player-safe** unless the doc explicitly lives in Hot.

---

## What Layer 1 is **not**

- Not schemas (Layer 3), not protocol (Layer 4), not prompts (Layer 5), not code (Layer 6–7).
- It also isn’t a dumping ground for lore or prose—keep those in the studio loops under Layer 0
  rules.

---

## Index

- `charters/showrunner.md`
- `charters/gatekeeper.md`
- `charters/plotwright.md`
- `charters/scene_smith.md`
- `charters/style_lead.md`
- `charters/lore_weaver.md`
- `charters/codex_curator.md`
- `charters/researcher.md` _(optional)_
- `charters/art_director.md` _(optional)_
- `charters/illustrator.md` _(optional)_
- `charters/audio_director.md` _(optional)_
- `charters/audio_producer.md` _(optional)_
- `charters/translator.md` _(optional)_
- `charters/book_binder.md`
- `charters/player_narrator.md`

Briefs mirror the same names under `briefs/`.
