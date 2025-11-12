# Codex Expansion — Publish Player-Safe Knowledge

**Purpose**  
Turn canon (often spoiler-heavy) into **player-safe** codex pages with clear cross-references.
Improve comprehension without leaking twists or internal plumbing.

**Outcome**  
Codex entries and crosslink maps (in **Hot**) derived from **canon** and ready to merge to **Cold**
after gatecheck. Taxonomy/clarity hooks may also be addressed here.

---

## 1) Triggers (Showrunner)

- After **Lore Deepening** produces new/updated canon.
- When Story Spark/Scene Smith introduce terms repeatedly.
- On player-comprehension concerns (PN/Binder feedback).
- To resolve **taxonomy/clarity hooks** (coverage gaps, red-links).

**Activation**  
Showrunner opens/updates a **Trace Unit (TU)**: `tu-codex-<topic-or-batch>`.

---

## 2) Inputs

- Canon entries from **Lore Deepening** (spoiler-level answers).
- Style guardrails (tone, register, motif vocabulary).
- Existing codex pages (Cold) for alignment.
- Accepted **taxonomy/clarity hooks** (from Hook Harvest).
- PN feedback and Binder UX notes.

---

## 3) Roles & Responsibilities

- **Codex Curator (R)**
  - Author **player-safe** entries; create cross-refs (“See also”), disambiguation notes, and
    glossary stubs.
  - May originate **taxonomy/clarity hooks**; does **not** invent deep lore.
- **Lore Weaver (C)**
  - Ensures summaries accurately reflect canon; marks spoilers to avoid.
- **Style Lead (C)**
  - Enforces voice clarity and reading level; motif consistency.
- **Gatekeeper (C)**
  - Checks Presentation Safety (no spoilers), Integrity (links resolve), and Style.
- **Showrunner (A)**
  - Confirms scope/priority; sequences merges.

_(Researcher may be consulted if factual claims appear; otherwise factual content should be neutral
or cite canon as in-world belief.)_

---

## 4) Player-Safe Pattern

**Entry anatomy (human-level, not a schema):**

- **Title** — term/name players see in manuscript.
- **Overview (2–4 sentences)** — neutral, spoiler-safe description; avoid causal reveals.
- **Usage in the book** — how/why the player might encounter the term.
- **Context** — high-level setting notes (political, technical, cultural) without twist details.
- **See also** — 3–5 related entries. Prefer breadth over recursion.
- **Notes** — accessibility or localization hints (e.g., pronunciation; units).
- **Lineage** — `Lineage: TU …` (traceability); **no spoilers** in lineage text itself.
- _(Optional)_ **Player warnings** — diegetic safety content (“Clipped gravity decks require
  mag-soles”).

**Never include:** hidden gate conditions, codeword names, internal IDs, seed/model info, twist
explanations.

---

## 5) Procedure

1. **Select Topics**
   - From canon deltas, term frequency in manuscript, and taxonomy hooks.
   - Prioritize **player-value**: comprehension bottlenecks first.

2. **Draft Entries (Curator)**
   - Write overview and context in **Style Lead**’s register.
   - Add **See also** list that improves navigation (avoid self-loops).

3. **Spoiler Sweep (Lore Weaver)**
   - Compare against the spoiler-level canon; mask revelations.
   - If masking makes an entry misleading, add a **neutral phrasing** workaround or defer until a
     later publication window.

4. **Style Pass (Style Lead)**
   - Ensure clarity, consistent terminology, motif harmonization, and reading level.

5. **Link Audit (Curator)**
   - Check that every cross-reference resolves (or create the stub if approved).
   - Add disambiguation if a term is overloaded.

6. **Gatekeeper Pre-Check**
   - Presentation Safety, Integrity, Style. Flag any gateway logic leaks.

7. **Package & Handoff**
   - Produce a **Codex Pack** (entries + crosslink map) and attach to TU.
   - Notify **Binder** and **PN** that new player-safe surfaces will land after merge.

---

## 6) Deliverables (Hot)

- **Codex Pack** (human text):
  - Entries (Title, Overview, Usage, Context, See also, Notes, Lineage).
  - **Crosslink Map** (simple list is fine) ensuring navigability.
- **Coverage Report**
  - What new terms now covered; remaining red-links (with hooks if needed).
- **Spoiler Hygiene Note**
  - Summary of masked details; any entries deferred due to spoil risk.

---

## 7) Merge Path (summary)

- Gatekeeper runs full **Presentation Safety** + **Integrity** + **Style** checks.
- **Showrunner** merges Pack to **Cold** on pass.
- Binder/PN can now consume the **Cold** codex pages for exports/play.

---

## 8) Success Criteria

- High-frequency manuscript terms have matching codex entries.
- No spoilers; PN can reference entries safely.
- All **See also** links resolve; no dead ends.
- Reading level and tone align with Style Lead; localization notes present where needed.
- Traceability present (`TU-ID` lineages).

---

## 9) Failure Modes & Remedies

- **Accidental spoilers** → Move detail back to canon notes; rewrite with neutral phrasing.
- **Over-technical voice** → Style Lead simplifies; add examples.
- **Link rot** → Add stubs (with plan) or reduce See also fan-out.
- **Taxonomy creep into canon** → Escalate to Lore Weaver; Curator does not invent backstory.

---

## 10) RACI (quick)

| Task          | R             | A          | C           | I          |
| ------------- | ------------- | ---------- | ----------- | ---------- |
| Draft entries | Codex Curator | Showrunner | Lore, Style | Gatekeeper |
| Spoiler sweep | Lore Weaver   | Showrunner | Curator     | Gatekeeper |
| Style pass    | Style Lead    | Showrunner | Curator     | Gatekeeper |
| Pre-check     | Gatekeeper    | Showrunner | —           | All        |
| Merge         | Showrunner    | Showrunner | Gatekeeper  | Binder, PN |

---

## 11) Example (miniature)

**Title**: Dock 7  
**Overview**: A cargo and repairs quay on the station’s shadow side, known for low-bid maintenance
and odd-hour shifts.  
**Usage**: Early chapters reference Dock 7 for side-jobs and parts salvage.  
**Context**: Security patrols are thin; rumor credits a refinery incident years back with today’s
strict fire doors.  
**See also**: “Wormhole Tolls”, “Salvage Permits”, “Shadow Decks”, “Station Security”  
**Notes**: “Dock” vs “Berth” distinction maintained; local slang prefers “D7”.  
**Lineage**: TU tu-codex-docks-batch-1

_(Canon’s detailed cause of the “refinery incident” stays in spoiler notes, not here.)_

---

**TL;DR**  
Publish knowledge the player can safely read: clear, helpful, spoiler-tight. Let lore be deep; let
codex be kind.
