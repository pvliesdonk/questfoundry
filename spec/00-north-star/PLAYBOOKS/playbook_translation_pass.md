# Playbook — Translation Pass

**Use when:** You need a **target-language slice** (manuscript/codex/PN surfaces) that preserves
voice and diegesis. Works even if you only ship the **glossary/style kit** this round.

**Outcome:** A **Language Pack** (glossary, register map, motif equivalence, idiom strategy,
localized surfaces, coverage %, open issues) ready for gatecheck and Cold merge. Exports can include
complete or partial translations flagged accordingly.

---

## 1) One-minute scope (Showrunner)

- [ ] Choose **language & slice** (full book, act, codex subset).
- [ ] Decide **kit-only** vs **kit+surfaces**.
- [ ] Mark roles **active/dormant** (Translator active; PN/Curator/Style consult; others optional).
- [ ] Open TU: `tu-translation-<lang>-<scope>`; set coverage target & timebox.

---

## 2) Inputs you need on screen

- **Cold snapshot** (player-safe surfaces only).
- **Style guardrails** (voice/register/motifs).
- **PN Principles** (diegetic gate phrasing patterns).
- Existing **glossary/terminology** (if any).
- Known **untranslatables** / culture notes.

---

## 3) Do the thing (compact procedure)

**Translator (R)**

1. **Glossary first** — lock key terms; build **Register Map** (pronouns, formality, honorifics).
2. **Motif equivalence** — decide how house motifs render (e.g., “low-G dust”).
3. **Idiom strategy** — list idioms → functional equivalents; mark avoid/replace.
4. **Localize surfaces** — manuscript choices/paragraphs, codex titles/summaries, captions/alt text;
   **preserve anchors & links**.
5. **Open issues** — record untranslatables, cultural cautions, glossary gaps.

**Style Lead (C)** 6. Tone & register pass; flag wording that fights house voice.

**PN (C)** 7. Check **diegetic gate phrasing** reads natural; no internal labels.

**Codex Curator (C)** 8. Cross-ref audit; ensure **See also** links resolve in target language.

**Gatekeeper (C)** 9. **Pre-gate**: Presentation Safety, Integrity (links), Style.

**Translator (R)** 10. Package **Language Pack**; compute **coverage %** (sections & codex entries).

---

## 4) Deliverables (Hot)

- **Language Pack** for `<lang>`
  - Glossary (term → approved translation + notes)
  - Register Map (pronouns, honorifics, formality policy)
  - Motif Equivalence (rendering patterns)
  - Idiom Strategy (source → equivalent/rewrites)
  - Localized Surfaces (manuscript/codex/captions/alt text)
  - Coverage % (by sections & codex entries)
  - Open Issues (with suggested upstream rewrites if needed)
  - Traceability (TU-ID, snapshot ID)
- **Pre-gate note** (Gatekeeper): pass/fail + remediation.

---

## 5) Hand-offs

- → **Binding Run**: include language slice with `complete`/`incomplete` flags.
- → **Style Tune-up**: if recurring register mismatches appear.
- → **PN**: update phrasing bank patterns for the language.
- → **Codex Expansion**: terms that need new or renamed entries.

---

## 6) Definition of “done” (for this play)

- [ ] Glossary & Register Map approved by Style Lead.
- [ ] Motifs carry over; idioms handled or listed with alternatives.
- [ ] Localized surfaces keep **anchors/links** intact; choice labels remain distinct.
- [ ] PN phrasing enforces gates **diegetically**.
- [ ] Gatekeeper **pre-gate green** (Presentation/Integrity/Style).
- [ ] Coverage % computed; **open issues** documented.

---

## 7) Fast heuristics

- Prefer **functional equivalents** over literalism when tone would suffer.
- If a reveal depends on wording nuance, **mask** in the target language too.
- Keep choice labels **short, contrastive**; avoid near-synonyms.
- When in doubt, add a **glossary usage example**.

---

## 8) RACI (micro)

| Task                          | R          | A          | C                  | I          |
| ----------------------------- | ---------- | ---------- | ------------------ | ---------- |
| Build glossary & register map | Translator | Showrunner | Style Lead         | Gatekeeper |
| Localize surfaces             | Translator | Showrunner | PN, Curator        | Gatekeeper |
| Style/voice pass              | Style Lead | Showrunner | Translator         | Gatekeeper |
| Pre-gate                      | Gatekeeper | Showrunner | Style Lead         | All        |
| Merge & export opts           | Showrunner | Showrunner | Gatekeeper, Binder | PN         |

---

## 9) Templates

**Glossary entry**

```

Source: <term>   POS: <noun/verb/...>
Target: <approved translation>
Notes: <usage, do-not-translate cases, variants>
Example: "<short sentence in target>” (maps to: "<source>")

```

**Register Map (snippet)**

```

Second person: <T/V policy>   Formal address: <rules>
Honorifics: <list/usage>       Swears: <allowed/avoid + substitutes>

```

**Localized choice (pattern)**

```

EN: "Slip through maintenance, or face the foreman." <LANG>: "<concise option A>, of <concise option B>."
Anchors: <keep original IDs/links>

```

---

## 10) Anti-patterns

- Translating **internal labels** (codeword names, schema fields) onto the surface.
- “Faithful” translations that wreck **register** or **motifs**.
- Breaking anchors/links during line-wrap or punctuation changes.
- Over-localizing proper nouns against the glossary.

---

**Cheat line (TU note):**  
“Translation Pass (NL, Act I): glossary+register map locked; motifs mapped; 74% surfaces localized;
PN diegetic gates verified; 5 open idioms; pre-gate Presentation/Integrity/Style green.”
