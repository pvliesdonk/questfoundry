# Accessibility & Content Notes — Player-First Surfaces

Player-facing surfaces must be **readable, navigable, and safe**. This doc sets the **baseline
requirements** for manuscript, PN lines, codex entries, captions/alt text, and audio descriptions
across all exports. It’s not a style suggestion; it’s a merge bar.

> Scope: human-level rules for Layer 0. Tools and format-specific details (EPUB/PDF/HTML) are
> implementation choices later; the **requirements stay the same**.

---

## 1) Principles

- **Perceivable** — Provide text equivalents and clear structure.
- **Operable** — Navigation is predictable; links and anchors work.
- **Understandable** — Language is clear; reading order makes sense; no hidden mechanics.
- **Robust** — Surfaces degrade gracefully across formats (MD/HTML/EPUB/PDF).
- **Kind by default** — Content warnings and sensory safety are normal, not special.

---

## 2) What’s in scope

- **Manuscript** (sections, choices, headings)
- **PN surfaces** (narration phrases, gate phrasing)
- **Codex** (entries, crosslinks, glossary)
- **Art** (captions + **alt text**)
- **Audio** (captions/**text equivalents**, loudness safety)
- **Translation slices** (localized versions of all the above)

---

## 3) Global requirements (apply everywhere)

1. **Headings & structure**
   - Logical hierarchy (`#`, `##`, `###`); no orphan headings.
   - One H1 per document; sections start with an H2 or higher.

2. **Links & anchors**
   - Descriptive link text (“See Salvage Permits”) — never “click here”.
   - Anchors resolve in **all export formats** (Gatekeeper checks Integrity).

3. **Contrast & color**
   - Don’t rely on color alone to convey state or choice grouping.
   - Print-friendly exports use high contrast by default.

4. **Language & register**
   - Plain, direct sentences. Avoid jargon unless the codex covers it.
   - Use numerals and units consistently; SI units with localized variants allowed.

5. **Keyboard and reading order (for HTML/EPUB)**
   - Natural DOM order; headings outline matches visual order.
   - Focusable elements (if any in UI later) must be reachable without a mouse.

6. **No motion or flashing surprises**
   - We don’t ship animation in Layer 0 exports; in later UIs, avoid strobing/flashing and provide
     “reduce motion” options.

7. **Content notes (CW)**
   - Put **brief, non-spoiling** content warnings in front matter (see §8).
   - Keep CW phrasing neutral; no plot reveals.

---

## 4) Manuscript (sections & choices)

- **Section intro**: short lead that orients time/place if needed.
- **Choices**: concise, contrastive; avoid near-synonyms.
  - Example: “Slip through maintenance.” / “Face the foreman.”
- **Codeword & gate masking**: phrase **diegetically** (see PN Principles).
- **Tables & lists**: prefer lists; if tables are used, include header cells and simple structure.

**Checklist**

- [ ] Headings reflect narrative hierarchy.
- [ ] Choices are distinct and unambiguous.
- [ ] No internal labels (flags/IDs) on the surface.
- [ ] Long paragraphs broken into readable lengths.

---

## 5) PN surfaces

- **In-world phrasing** only; never expose mechanics.
- **Recaps**: allowed and encouraged after long detours (2–3 lines max).
- **Gate phrasing**: offer neutral refusals or alternatives (“The foreman scans your lapel…”).

**Checklist**

- [ ] No “developer voice” or schema talk.
- [ ] Clear, neutral recaps where readers commonly lose context.
- [ ] Gate enforcement reads like natural interaction.

---

## 6) Codex entries

- **Anatomy**: Title • Overview (2–4 sentences) • Usage • Context • See also • Notes • Lineage (TU
  ID only).
- **Crosslinks**: 3–5 useful “See also” items that actually resolve.
- **Readability**: keep sentences shorter than manuscript prose; avoid lore dumps.

**Checklist**

- [ ] Links resolve; no loops that trap the reader.
- [ ] No spoilers; summaries aid comprehension, not revelation.
- [ ] Pronunciation/diacritics added where helpful.

---

## 7) Art (captions & alt text)

- **Caption (player-safe)**: mood/affordance, not a twist.
- **Alt text (required)**: one sentence with **concrete nouns/relations**; avoid “image of…”.
  - Good: “A scarred bulkhead with spent extinguishers; a foreman’s shadow reaches across the deck.”
- **Determinism/technique**: keep in logs, not on surfaces.
- **Decorative images**: if truly decorative in HTML/EPUB, mark as such (`role="presentation"` or
  empty alt), but prefer meaningful alt text.

**Checklist**

- [ ] Every image has alt text or is explicitly decorative.
- [ ] Caption is spoiler-safe and in register.
- [ ] No seeds/models/technical notes on the surface.

---

## 8) Audio (captions, text equivalents, safety)

- **Text equivalents**: concise, descriptive lines for ambience/foley/stingers/VO.
  - Example: “[Low mechanical hum; distant footsteps on metal grating.]”
- **Loudness & dynamics**: avoid startle peaks; provide fade/ramp guidance in plan; normalize assets
  in later layers.
- **Safety notes**: call out harsh/interruptive cues in front matter (“Contains intermittent alarm
  tones.”)
- **VO**: avoid dialect stereotyping; coordinate with Translator for register mapping.

**Checklist**

- [ ] Text equivalents present for each cue.
- [ ] Safety notes listed in front matter without spoilers.
- [ ] No technique terms (DAW/plugins) on the surface.

---

## 9) Translations (localized slices)

- **Glossary & register map first**; lock T/V, honorifics, and idiom policy.
- **Anchors**: preserve IDs; crosslinks resolve in target language.
- **PN phrasing**: diegetic and natural in target language; no internal labels.

**Checklist**

- [ ] Coverage % computed and shown in export.
- [ ] Links & anchors work in the slice.
- [ ] Idioms translated functionally, not literally, when needed.

---

## 10) Front matter & CW policy

Include a small **Accessibility & Content Notes** block in every export view:

- **Accessibility**: alt text present; high contrast; descriptive links; captions for audio cues; no
  motion surprises.
- **Content notes** (non-spoiling): e.g., _violence (industrial accident), confinement, alarm
  sounds_.
- **Languages**: list translation coverage (`complete`/`incomplete`).
- **Options included**: art plans/renders, audio plans/assets (for reader expectations).

Keep CW **broad**; do not name perpetrators or reveal twists.

---

## 11) Gatekeeper review (Presentation bar)

Before merging to **Cold** or shipping a view:

- [ ] **Perceivable**: alt text/captions present; headings/links coherent.
- [ ] **Operable**: anchors resolve; navigation consistent across formats.
- [ ] **Understandable**: no meta/technique; codex helps, not spoils.
- [ ] **Kind**: content notes present; audio safety addressed.
- [ ] **Localization**: slices preserve links and diegesis; coverage labeled.

---

## 12) Failure modes & quick remedies

- **Alt text spoils** → Move detail to canon notes; rewrite alt text to mood/affordance.
- **Broken links after export** → Re-run anchor sync; treat as Integrity failure.
- **Choice ambiguity** → Compress wording; add micro-context to the section lead.
- **Onomatopoeia spam in captions** → Replace with concise descriptions (“alarm rings twice,
  distant”).
- **Localization breaks anchors** → Preserve IDs; add mapping table if the title changes.

---

## 13) Ownership

- **Gatekeeper**: enforces this doc as part of **Presentation** checks.
- **Style Lead**: sanity on tone and readability.
- **Binder**: verifies multi-format navigation and front-matter notes.
- **Curator/PN/Translator**: each owns their surfaces meeting this bar.

---

**TL;DR**  
Write surfaces people can **read, navigate, and trust**: clear headings, descriptive links, alt text
and captions, safe audio, diegetic gates, neutral content notes, and zero spoilers or technique. If
a surface confuses a screen reader—or spoils a twist—it doesn’t ship.
