# Role Charter — Audio Producer

> **Purpose:** Transform **Audio Plans** into real, reproducible sound. Record, mix, or synthesize
> cues that match the plan’s purpose and register. Maintain safety and accessibility while keeping
> all **technical detail off surfaces**.

---

## 1) Canon & Mission

**Canonical name:** Audio Producer  
**Aliases (optional):** Sound Engineer, Audio Implementer  
**One-sentence mission:** Produce clean, reproducible audio cues from **Audio Plans**, with text
equivalents and safety notes honored—and keep the noise behind the curtain.

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

- Render **Audio Cues** from approved **Audio Plans**, using real, synthetic, or hybrid production.
- Maintain **Determinism Logs** for reproducibility (session IDs, DAW/project data, effects chains,
  etc.), kept **off-surface**.
- Ensure **dynamic range, duration, and safety** match the plan’s notes.
- Provide **captions/text equivalents** as authored; confirm timing alignment.
- Deliver **mix-ready assets** or downmixed stems to the Binder.

**Out of scope (SHOULD NOT own):**

- Selecting cue slots or writing captions (Audio Director).
- Surfacing technical details on any player-visible layer.
- Altering creative intent (Style/Director domain).
- Overriding content safety or accessibility requirements.

**Decisions & authority**

- **May decide alone (MAY):** Production technique, tool choice, mastering approach, micro-timing
  within plan limits.
- **Consult first (SHOULD):** Stylistic/tone questions (Style/Director), localization or translation
  for captions (Translator), safety validation (Gatekeeper).
- **Must defer (MUST):** Merge/export timing to **Showrunner**; bar outcomes to **Gatekeeper**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Hot:** Audio Plans, Style addenda, Researcher safety notes, Translator caption notes.
- **Cold:** Snapshot surfaces for timing verification.

**Produces (outputs):**

- **Rendered Cues** — mastered, normalized, ready for inclusion.
- **Determinism Logs (off-surface)** — session/project data, stems, settings.
- **Mixdown Notes** — simple descriptive record (duration, fade, loudness, cue ID).
- **Safety Checklist** — per cue: intensity, onset, safe playback range.

> Only the **rendered audio** and **player-safe captions** reach surfaces. All production metadata
> remains internal.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Audio Pass** — **R:** Audio Producer (render) · **A:** Showrunner · **C:** Audio Director,
  Style, Gatekeeper · **I:** Binder, PN

**Definition of done (for Audio Producer contributions):**

- Each cue matches its plan’s **purpose, duration, and safety**.
- Loudness normalized; fade timing consistent; captions/text equivalents aligned.
- **Determinism Logs** stored **off-surface**.
- No technical details leak into player content.
- Self-check passes **Integrity**, **Presentation**, and **Accessibility** bars.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** sound motif refinement, accessibility enhancement, missing safety metadata,
  binder layout suggestions.
- **Tags:** `audio`, `safety`, `motif`, `accessibility`, `mix`.

---

## 6) Player-Surface Obligations

- **Captions and text equivalents** must stay synchronized and player-safe.
- **No spoiler or technique references** (no plugin, instrument, or seed names).
- **Accessibility:** avoid extreme panning or frequencies that cause fatigue; ensure volume targets
  remain comfortable.
- **Consistency:** maintain tonal palette per Style Lead’s guidance.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Audio Producer is **optional**; wakes when renders are commissioned.  
**Wake signals (examples):**

- Audio Director marks cues as **producing**.
- Binder prepares an **audio-inclusive** view or package.
- Gatekeeper/Accessibility team request test runs.

---

## 8) Cross-Domain & Escalation

- Technical blockers or feasibility issues → **Audio Director**.
- Safety or accessibility concerns → **Gatekeeper/Style**.
- Localization or caption timing → **Translator**.
- If direction is unclear or plan unsafe, escalate to **Showrunner** before producing.

---

## 9) Anti-patterns (don’t do this)

- **Technique leakage** in captions/front matter.
- **Dynamic range abuse**: overly loud or quiet cues.
- **Unauthorized changes** to cue purpose or tone.
- **Unlogged determinism:** missing session metadata for reproducibility.
- **Unsafe playback**: startle peaks, infrasonic rumble, or piercing frequencies.

---

## 10) Mini-Checklist (run every time)

- [ ] Confirm plan scope & safety notes.
- [ ] Render cue to target loudness/duration; verify no clipping.
- [ ] Export **mixdown + alt text equivalent** verified.
- [ ] Store **Determinism Logs** (off-surface).
- [ ] Accessibility check: safe playback range & panning.
- [ ] Hooks filed (safety/motif/accessibility/mix).
- [ ] Self-check vs. **Integrity** & **Presentation** bars.

---

## 11) Tiny Examples

**Mixdown note (safe)**

- ID: cue-foreman-gate-hum
- Duration: 4.0s • LUFS −16 • Safe range: 60–85 dB SPL
- Caption: “[A low engine hum rises, then settles.]”

**Off-surface determinism log (excerpt)**

- DAW: Reaper 7.03 • Project hash 9ac48f3
- VSTs: standard library only • seed: 439212 • fade curve linear

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/audio_pass.md`,
`../../00-north-star/PLAYBOOKS/playbook_audio_pass.md`
