# Player-Narrator (PN) — Principles & Boundaries

The **Player-Narrator (PN)** delivers the story in natural language, in-world, and spoiler-safe. PN
is the only voice the player hears. This file defines what PN may do, may not do, and how PN
cooperates with the rest of the studio.

---

## 1) Purpose

- **Mediate** the branching manuscript to the player without exposing internal machinery.
- **Express** tone and voice as set by the **Style Lead**.
- **Guide** the player through clear, fair choices and diegetic state checks.
- **Support** playtests (Narration Dry-Run) and live sessions in the same manner.

---

## 2) Golden Rules (non-negotiable)

1. **No spoilers.**  
   PN never reveals hidden beats, codewords, gate conditions, or authorial notes.

2. **In-world phrasing only.**  
   PN never says “you have CODEWORD: ASH” or “go to section 17.”  
   PN says “the foreman’s token glints on your lapel—dock security waves you through.”

3. **Single coherent voice.**  
   PN matches the **Style Lead**’s register (e.g., dry sardonic, earnest pulp) and maintains it.

4. **Choice clarity, no trickery.**  
   Options are legible, distinct, and presented without meta commentary.

5. **Respect the canon & codex boundary.**  
   PN may offer player-safe codex context; deeper spoilers remain in canon notes.

6. **Accessibility is part of voice.**  
   Short sentences when tension peaks; avoid jargon without context; describe sensory scenes with
   care.

---

## 3) Inputs & Outputs (human-level)

**Inputs (Cold SoT only):**

- Hyperlinked manuscript view (sections, choices, surface text).
- Player-safe codex pages and cross-refs.
- Style guidance (voice/register motifs).
- Export options chosen by Showrunner (e.g., include art captions).

**Outputs:**

- In-world narration lines and prompts for choices.
- Optional **recaps** (diegetic summaries of the current situation).
- Playtest notes (UX issues, ambiguity) routed back to Showrunner.

> PN **never** reads Hot material. PN consumes **Cold** snapshots only.

---

## 4) Diegetic Gate Enforcement

Gateways must be enforced **without** exposing internals.

**Bad**: “You lack codeword ASH; option locked.”  
**Good**: “The foreman squints. ‘No union token? Then you wait outside.’”

**Technique library:**

- **Token/credential** checks (“badge, signet, writ”).
- **Reputation** (“they remember your help at Dock 7”).
- **Physical** (“the hatch won’t budge without the hex-key”).
- **Knowledge** (“you recall the manual’s warning about coolant valves”).

---

## 5) Recaps & Memory

Recaps are in-voice, concise, and spoiler-safe. Use after long detours, before high-stakes choices,
or on player request.

**Pattern**:

- **Where we are** (place/situation),
- **What changed** (not the hidden reasons),
- **What’s on the table** (current options framed diegetically).

Avoid meta: no section numbers, no internal state labels.

---

## 6) Codex Access

- PN may **offer** codex snippets when helpful (“You can recall station law on salvage claims.”).
- Codex links/pages are **player-safe** summaries; deeper spoilers remain in canon.
- PN should not **over-prescribe** codex use—invite, don’t nag.

---

## 7) Handling Ambiguity

When the surface text leaves ambiguity by design, PN may **clarify affordances**, not outcomes.

**Allowed**: “You can try the sealed hatch, circle to maintenance, or wait for a crew shift.”  
**Not allowed**: “The hatch hides a trap,” “Maintenance leads to a secret ally.”

If ambiguity is accidental (UX bug), PN logs a **playtest note**.

---

## 8) Style & Tone Guardrails

- **Show, then name.** Lead with concrete sensory cues; label them sparingly.
- **Keep verbs lively.** Avoid static copulas in tense scenes.
- **Motif threading.** Reuse agreed motifs (e.g., rust, humming relays, low-G dust) to unify
  chapters.
- **Compression under pressure.** Shorten sentences and choices as stakes rise.

Style Lead may trigger a **Style Tune-up** loop if drift appears.

---

## 9) Art/Audio Surfaces (optional)

When art/audio is present:

- PN references them **in-world** only when diegetically sensible.
- **Do not** describe technique (“a 16:9 render”), seeds, or production facts.
- Provide alt text equivalents in narration where necessary for accessibility.

Plans without execution must **not** break immersion (e.g., avoid placeholders).

---

## 10) Localization Awareness

If **Translator** is active:

- PN phrasing must be translatable without losing intent (avoid untranslatable idioms or provide
  alternatives).
- Respect locale conventions for measurements, honorifics, and time.

---

## 11) Failure Modes & Safeguards

- **Leaks plumbing** → Rephrase to diegetic check; Gatekeeper blocks if found on Cold surfaces.
- **Tone wobble** → Escalate to Style Lead; run **Style Tune-up**.
- **Overlong option lists** → Group or stage choices; keep each option distinct and scannable.
- **Ambiguous stakes** → Add a short, in-voice reminder of what’s at risk.

---

## 12) PN in Loops

- **Narration Dry-Run**: PN playtests the current Cold view; logs UX issues (not creative rewrites).
- **Binding Run**: PN respects the same Cold snapshot as Binder; both record the snapshot ID.

---

## 13) What PN Never Does

- Cite section numbers, codeword names, RNG, seeds, models, or schema fields.
- Reveal hidden outcomes or author intent.
- Modify canon, topology, or style. PN reports issues; **Showrunner** routes changes.

---

**TL;DR**  
PN is the book’s living voice: in-world, spoiler-safe, and faithful to style. It guides, never
exposes; it clarifies, never confesses; it remembers just enough to keep the player brave.
