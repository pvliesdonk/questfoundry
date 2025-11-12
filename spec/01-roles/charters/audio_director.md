# Role Charter — Audio Director

> **Purpose:** Decide **what to hear and why**. Author spoiler-safe **Audio Plans**—cues, placement,
> intensity, and text equivalents—so sound enhances comprehension, mood, and signposting without
> leaking internals. Plans may ship **without** assets (plan-only), enabling prose-only books today
> and audio-ready tomorrow.

---

## 1) Canon & Mission

**Canonical name:** Audio Director  
**Aliases (optional):** Sound Director, Audio Lead  
**One-sentence mission:** Choose moments where sound clarifies, paces, or signals—and specify them
in player-safe plans with captions/text equivalents and safety notes.

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

- Select **cue slots** and state purpose: **clarify / recall / mood / signpost / pace**.
- Write **Audio Plans**: cue description, placement (before/after/under line), **intensity &
  duration**, text equivalents/captions, **safety notes**, inclusion criteria.
- Define **leitmotif/use policy** (non-spoiling) and align with Style Lead’s register.
- Set **reproducibility expectations** (if promised) for off-surface DAW/session notes to be
  maintained by the Audio Producer.
- Coordinate with Translator for caption portability and with PN for in-world fit.

**Out of scope (SHOULD NOT own):**

- Producing/recording/mixing assets (Audio Producer).
- Publishing technique on surfaces (plugins, DAW versions, stems).
- Inventing canon or altering topology.
- Using leitmotifs to hint at hidden allegiances (spoiler risk).

**Decisions & authority**

- **May decide alone (MAY):** Which moments get cues; plan content; caption phrasing that remains
  player-safe.
- **Consult first (SHOULD):** Register/idiom (Style/Translator); content sensitivity
  (Gatekeeper/Style); PN cadence impacts.
- **Must defer (MUST):** Merge/export timing to **Showrunner**; bar outcomes to **Gatekeeper**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Hot:** section drafts, Style addenda, Curator terminology, PN friction tags, Researcher safety
  notes.
- **Cold:** current snapshot surfaces to avoid contradiction; learn typical section cadence.

**Produces (outputs):**

- **Audio Plan** per slot — _purpose • cue description • placement • intensity/duration • text
  equivalents/captions • safety notes • inclusion criteria • reproducibility expectation
  (off-surface)_.
- **Cue List** — ordered map of slots with owners/status (`planned | producing | deferred`).
- **Hook List** — requests for Curator anchors, Style patterns, PN cadence adjustments, or
  Researcher checks.

> Captions and text equivalents are **player-safe**. Repro notes live **off-surface**.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Audio Pass** — **R:** Audio Director · **A:** Showrunner · **C:** Style, Gatekeeper, PN,
  Translator · **I:** Audio Producer, Binder

**Definition of done (for Audio Director contributions):**

- Cues **serve purpose** (clarify/recall/mood/signpost/pace) and do not telegraph spoilers.
- Each plan includes **text equivalents/captions** and **safety notes** (e.g., intensity, startle
  risk).
- Style/register aligned; captions portable for translation.
- If reproducibility is promised, expectations stated for **off-surface** DAW/session logging.
- Self-check passes **Presentation** and **Accessibility** bars.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** PN cadence fix, needed codex anchor for a procedure/alarm, sensitivity
  mitigation, motif refinement, layout hints for Binder.
- **Tags:** `audio-cue`, `pace`, `signpost`, `safety`, `pn-phrasing`, `localization`,
  `codex-anchor`.

---

## 6) Player-Surface Obligations

- **Text equivalents/captions:** concise, **non-technical**, and evocative; avoid plugin names or
  levels.
- **Safety:** mark startle/intensity risks in plan notes; keep captions player-safe (e.g., “[A short
  alarm chirps twice, distant.]”).
- **Spoiler Hygiene:** no leitmotif-as-spoiler; no internal state hints.
- **PN boundaries:** cues **support** in-world delivery; never explain mechanics.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Audio Director is **optional** and may be dormant.  
**Wake signals (examples):**

- PN/Style report **cadence** or **comprehension** gaps.
- Accessibility goals call for **text equivalents** beyond prose.
- Showrunner wants plan-only audio coverage for a release.

**Plan-only merges:** Allowed as **deferred:audio**; assets can be produced later.

---

## 8) Cross-Domain & Escalation

- Pair with **Audio Producer** for feasibility; reproducibility tracked **off-surface**.
- Coordinate with **Style** for caption phrasing and with **Translator** for portability.
- If a cue would imply canon/topology, escalate via **Showrunner**; request Lore/Plot guidance.

---

## 9) Anti-patterns (don’t do this)

- **Jump-scare stingers** that create startle risk without safety notes.
- **Technique leakage** in captions (“low-pass with 12 dB res,” “VST X”).
- **Spoiler leitmotifs** that signal hidden allegiances.
- **Sound wallpaper**: constant bed that muddies reading cadence.
- **UI/meta labels** (“SFX_Alarm_03.wav”, “TRACK-2”) on surfaces.

---

## 10) Mini-Checklist (run every time)

- [ ] TU scoped; slice & purpose chosen for each cue
- [ ] Audio Plans include **text equivalents + safety notes**
- [ ] Style/Translator consulted for register & portability
- [ ] Repro expectations (if any) noted **off-surface** for Producer
- [ ] Hooks filed (PN cadence, anchors, sensitivity)
- [ ] Self-check vs. **Presentation** & **Accessibility** bars

---

## 11) Tiny Examples

**Bad → Good (caption)**

- Bad: “ALARM SFX plays (Limiter -1dB).”
- Good: “[A short alarm chirps twice, distant.]”

**Audio Plan (excerpt, player-safe)**

- _Purpose:_ **Pace** — Brief tension lift after inspection scene
- _Cue:_ Low, steady engine hum swells, then eases
- _Placement:_ Under final two lines of section; fade by choice list
- _Intensity/Duration:_ Soft; 3–4 seconds
- _Text equivalent:_ “[A low engine hum rises, then settles.]”
- _Safety:_ Avoid sudden onset; no sharp transients
- _Inclusion:_ Only when section mentions freight engines or dock machinery
- _Repro:_ Off-surface: “log session tempo, key bed, and gain automation notes”

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/audio_pass.md`,
`../../00-north-star/PLAYBOOKS/playbook_audio_pass.md`
