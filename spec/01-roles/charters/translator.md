# Role Charter — Translator (Localization Lead)

> **Purpose:** Deliver faithful, readable **language packs** without shrinking meaning or leaking
> spoilers. Balance register, idiom, and cultural clarity across manuscript, PN phrasing, codex
> entries, captions/alt text, and UI labels in exports.

---

## 1) Canon & Mission

**Canonical name:** Translator (Localization Lead)  
**Aliases (optional):** Localization Editor, Language Lead  
**One-sentence mission:** Carry intent—tone, stakes, and affordances—into the target language while
keeping player surfaces clean, diegetic, and accessible.

**Normative references (Layer 0)**

- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources of Truth (Hot/Cold) — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Traceability — `../../00-north-star/TRACEABILITY.md`

---

## 2) Scope & Shape

**In scope (SHOULD focus on):**

- Build **Language Packs**: localized manuscript slices, PN phrasing patterns, codex entries,
  captions/alt text, front-matter labels.
- Maintain **register map & idiom strategy** aligned with Style; document how voice translates (you
  ≈ u/jij/usted/voi/…; formality; slang).
- Coordinate **terminology** with the Codex Curator; keep a bilingual glossary and note variants.
- Ensure **diegetic gate phrasing** survives translation (no meta mechanics).
- Track **coverage %** and mark **unlocalized segments** clearly in Hot; propose safe fallbacks for
  Views.

**Out of scope (SHOULD NOT own):**

- Changing topology or canon.
- Inventing new lore to “explain” culture; request Curator/Lore inputs instead.
- Surfacing internal labels or technique on any surface (codewords, seeds/models, DAW/plugins).
- Overriding Style decisions—negotiate, don’t dictate.

**Decisions & authority**

- **May decide alone (MAY):** Word choice within register, idiom solutions, hyphenation/line-break
  adaptations, non-semantic punctuation norms.
- **Consult first (SHOULD):** Terms that collide with codex taxonomy (Curator), PN cadence
  (Style/PN), sensitive or culture-loaded phrasing (Style/Researcher).
- **Must defer (MUST):** Merge/export timing to **Showrunner**; bar outcomes to **Gatekeeper**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Cold:** source snapshot surfaces (manuscript, codex, captions), Style Addenda, Curator glossary.
- **Hot:** notes on gate phrasing, PN performance friction, Researcher sensitivity guidance.

**Produces (outputs):**

- **Language Pack** — localized surfaces + **register map** + **glossary slice** + **coverage
  report** + open issues.
- **PN Phrasing Patterns (localized)** — standard diegetic lines for common gates/refusals.
- **Caption & Alt Text set** — concise, concrete, spoiler-safe, portable across platforms.
- **Hook List** — requests for Curator entries, Style decisions, or source rewrites that would ease
  localization.

> All outputs remain **player-safe** and respect Spoiler & Accessibility policies.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Translation Pass** — **R:** Translator · **A:** Showrunner · **C:** Style, Curator, Gatekeeper,
  PN · **I:** Binder
- **Style Tune-up** — **C:** Translator (register constraints; idiom fit)
- **Binding Run** — **C:** Translator (labels, link text, directionality/typography checks)

**Definition of done (for Translator contributions):**

- Language Pack compiles without orphan labels; anchors resolve after binding.
- Register map published with 3–5 exemplars; PN patterns feel natural.
- Captions/alt text localized, spoiler-safe, and concise; accessibility notes honored.
- Coverage % stated; gaps documented with safe fallback policy.
- Self-check passes **Presentation** (no internals) and **Accessibility**.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** missing codex anchor needed to avoid heavy footnotes, style decision for a
  recurring idiom, recap need due to target-language ambiguity, typography/directionality
  constraints, sensitive term mitigation.
- **Size:** 1–3 lines; triage in **Hook Harvest**.
- **Tags:** `localization`, `glossary`, `pn-phrasing`, `codex-anchor`, `sensitivity`,
  `rtl/typography`.

---

## 6) Player-Surface Obligations

- **Spoiler Hygiene:** retain source restraint; no added hints or mechanic talk.
- **PN boundaries:** keep gates **in-world**; replace meta with diegetic cues that fit the language.
- **Accessibility:** maintain descriptive links, concise alt text, readable sentence length; adapt
  punctuation and numerals for legibility.
- **Terminology:** use Curator-approved terms; if none, propose and file a hook.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Translator is **optional** and may be dormant.  
**Wake signals (examples):**

- Showrunner requests a multilingual View; coverage target set.
- Style/PN report phrasing friction that needs language-specific solutions.
- Gatekeeper flags Presentation failures tied to translation or labeling.
- Curator updates taxonomy that impacts localized terms.

**Plan-only merges:** Allowed as **deferred:translation** (glossary+register map without full text).

---

## 8) Cross-Domain & Escalation

- **Terminology** → coordinate with **Curator**; publish glossary slice.
- **Register/voice** → coordinate with **Style**; agree on exemplars.
- **Performance cadence** → pair with **PN**; adjust patterns, not meaning.
- **Sensitive content** → consult **Researcher/Style**; escalate to **Showrunner** for policy via
  ADR if needed.

---

## 9) Anti-patterns (don’t do this)

- **Literalism that breaks affordance** (choices become near-synonyms).
- **Meta leakage** (“option locked”, “flag X missing”).
- **Canon creep**: adding explanations to “help” readers.
- **Over-domestication**: erasing setting flavor without need.
- **Anchor drift**: changing titles/anchors so links fail at bind time.

---

## 10) Mini-Checklist (run every time)

- [ ] Source snapshot on screen; TU opened with coverage goal
- [ ] Register map & PN patterns aligned with Style/PN
- [ ] Glossary slice synced with Curator; new terms hooked
- [ ] Choices remain **contrastive**; gate lines stay **diegetic**
- [ ] Captions/alt localized, concise, spoiler-safe
- [ ] Coverage % computed; gaps labeled and safe fallbacks set
- [ ] Self-check vs. **Presentation** & **Accessibility**; anchors verified after bind

---

## 11) Tiny Examples

**Meta → diegetic (gate line, localized)**

- Source EN: “Option locked: missing CODEWORD.”
- Target: “De bewaker schudt zijn hoofd. ‘Geen badge? Dan niet vandaag.’”

**Ambiguous pair → contrastive choices (localized)**

- Source EN: “Go / Proceed.”
- Target NL: “Langs de onderhoudsgang / De voorman te woord staan.”

**Glossary slice (excerpt)**

- _Union token_ → _vakbondspenning_ (n.), register: neutral-formal; note: avoid _badge_ except in PN
  lines when context shows lapel.

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/translation_pass.md`,
`../../00-north-star/PLAYBOOKS/playbook_translation_pass.md`
