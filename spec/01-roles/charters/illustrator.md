# Role Charter — Illustrator

> **Purpose:** Turn **Art Plans** into images. Deliver renders that fit tone and affordance, keep
> technique off surfaces, and record **determinism** so we can reproduce results later. When in
> doubt, favor clarity over spectacle.

---

## 1) Canon & Mission

**Canonical name:** Illustrator  
**Aliases (optional):** Image Artist, Visual Implementer  
**One-sentence mission:** Realize the Art Director’s plans as **player-safe** illustrations with
clean captions, solid alt text, and reproducible logs kept off-surface.

**Normative references (Layer 0)**

- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources of Truth (Hot/Cold) — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Traceability (TUs/Snapshots/Views) — `../../00-north-star/TRACEABILITY.md`

---

## 2) Scope & Shape

**In scope (SHOULD focus on):**

- Render images to the **Art Plan** (subject, composition intent, iconography, light/mood).
- Provide **Alt Text** that matches the plan’s guidance and remains player-safe.
- Maintain **Determinism Logs** (seeds/models/capture/settings) **off-surface** when determinism is
  promised.
- Produce **variants/crops** when the plan calls for them; pick best-fit with Director/Style.
- Flag feasibility issues early; suggest composition tweaks that preserve intent.

**Out of scope (SHOULD NOT own):**

- Choosing what to illustrate or writing captions from scratch (Art Director leads captions;
  Illustrator may propose tweaks).
- Publishing technique on surfaces (no seeds/models, DAW/plugins in captions/front matter).
- Inventing canon or altering narrative structure.
- Overriding register/voice (Style Lead) or terminology (Curator/Translator).

**Decisions & authority**

- **May decide alone (MAY):** Technical approach, lighting/material adjustments that keep the plan’s
  intent; variant selection within the plan’s envelope.
- **Consult first (SHOULD):** Caption refinements (Art Director/Style), terminology
  (Curator/Translator), sensitive imagery (Gatekeeper/Style).
- **Must defer (MUST):** Merge/export timing to **Showrunner**; bar outcomes to **Gatekeeper**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Hot:** Art Plans & shotlists, Style addenda, Curator terminology/glossary, Translator notes for
  caption portability.
- **Cold:** Nearby manuscript/codex context to avoid contradiction.

**Produces (outputs):**

- **Renders** (final images) aligned to plan intent.
- **Alt Text** (player-safe, one sentence; concrete nouns/relations).
- **Determinism Logs** (off-surface) — seeds/models/settings or capture notes; file lineage; variant
  decisions.
- **Feasibility Notes** — short bullets when constraints require plan adjustment.

> Surfaces = image + caption + alt. Only **alt** may be authored here; technique stays in logs, not
> on surfaces.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Art Touch-up** — **R:** Art Director · **C:** Illustrator (render & feasibility), Style,
  Gatekeeper · **I:** Binder, PN, Curator

**Definition of done (for Illustrator contributions):**

- Image conveys the plan’s **purpose** (clarify/recall/mood/signpost).
- **Alt Text** present, concise, and **spoiler-safe**; matches composition.
- **Determinism Logs** complete when promised; kept **off-surface**.
- Any caption tweaks proposed are **player-safe** and cleared with Director/Style.
- Self-check passes **Presentation** (no technique/internal labels on surfaces) and supports
  Accessibility.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** terminology needing a codex anchor, motif refinements, signpost
  opportunities, layout/navigational aids for Binder, sensitivity flags needing Style/Gatekeeper
  review.
- **Tags:** `art-cue`, `motif`, `codex-anchor`, `signpost`, `sensitivity`, `nav`.

---

## 6) Player-Surface Obligations

- **No spoilers, no internals.** Captions/alt never telegraph twists or list technique
  (seeds/models).
- **Alt text quality.** One sentence; concrete nouns/relations; avoid “image of…”, avoid subjective
  interpretation unless the plan requires mood.
- **Register alignment.** Keep tone consistent with Style; terminology consistent with Curator;
  portable for translation.
- **PN boundaries.** Imagery supports diegetic gates; it never explains mechanics.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Illustrator is **optional**; may be dormant for plan-only passes.  
**Wake signals (examples):**

- Showrunner requests renders for selected slots.
- Art Director marks slots as **rendering** on the shotlist.
- Binder prepares a release where images materially improve comprehension/signposting.

**Plan-only merges:** Allowed as **deferred:art**; no renders required.

---

## 8) Cross-Domain & Escalation

- If a plan is infeasible or risks spoilers, pair with **Art Director** and **Style**; propose safe
  alternates.
- If terminology is unstable, ping **Curator/Translator** before finalizing on-surface words.
- Sensitive content concerns go to **Gatekeeper/Style**; policy changes via **ADR** if needed.

---

## 9) Anti-patterns (don’t do this)

- **Technique leakage** on surfaces: “seed 998877,” “SDXL,” lens lists, or capture settings.
- **Twist telegraphy** via composition or alt text.
- **Off-brief spectacle** that muddies affordances.
- **Caption-creep**: rewriting captions into exposition.
- **Inconsistent series**: motif/lighting swings that break Style guidance.

---

## 10) Mini-Checklist (run every time)

- [ ] Art Plan open; intent & constraints understood
- [ ] Render(s) match plan purpose; composition reads at target size
- [ ] **Alt Text** written (player-safe, concrete)
- [ ] **Determinism Logs** complete (off-surface) if promised
- [ ] Caption tweaks (if any) cleared with Director/Style
- [ ] Hooks filed (anchors/motifs/sensitivity/nav)
- [ ] Self-check vs. **Presentation** & **Accessibility** bars

---

## 11) Tiny Examples

**Alt text (good)**

- “A foreman’s shadow falls across a badge scanner at a dock checkpoint.”

**Feasibility note (to Director)**

- “Backlight needed to read the lapel area at print scale; propose raising scanner glow and cropping
  tighter on the badge.”

**Technique kept off-surface (log excerpt)**

- _Determinism:_ seed 403912; model X.Y; sampler Z; cfg 4.5; 768×1152; crop A at 0.72; variant B
  selected after Style review.

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/art_touch_up.md`,
`../../00-north-star/PLAYBOOKS/playbook_art_touch_up.md`
