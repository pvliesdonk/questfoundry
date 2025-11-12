# Spoiler Hygiene — Keep Player Surfaces Clean

Players read **surfaces** (manuscript, PN lines, codex, captions, localized text). Everything else
can be messy, clever, or mechanical—but **surfaces must never leak** twists, internals, or
technique. This document is the single rulebook.

> Surfaces = **Cold** only. Spoiler work happens in **Hot** (canon notes, plans, hooks).

---

## 1) Golden Rules

1. **No reveals on surfaces.**  
   If knowing it would change how a first-time reader perceives a choice or scene, it stays off the
   surface.

2. **No internals.**  
   Never print: codeword names, gate logic, RNG, seeds/models, DAW/plugins, schema fields, file
   paths, IDs.

3. **Diegetic gates only.**  
   PN and manuscript phrase access as **in-world conditions** (token/reputation/knowledge/tool), not
   mechanics.

4. **Canon ≠ Codex.**  
   Canon (spoiler-level truth) lives in Hot. Codex publishes **player-safe summaries** only.

5. **Caption ≠ Explanation.**  
   Art/audio captions set mood or clarify affordances; they don’t foreshadow twists or explain
   technique.

---

## 2) Leak Taxonomy (recognize and block)

| Leak type               | Example (bad)                                       | Fix (good)                                                         |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------------------------ |
| **Twist telegraph**     | “The foreman is secretly Syndicate.”                | “The foreman eyes your badge; his smile doesn’t reach his eyes.”   |
| **Gate logic**          | “Option locked: missing CODEWORD: ASH.”             | “No union token on your lapel; the foreman waves you back.”        |
| **Technique**           | “Rendered in SDXL, seed 1234; stinger at -10 LUFS.” | _(omit; technique recorded in logs, not on surface)_               |
| **Canon dump in codex** | “Dock 7 fire was sabotage by Toll Syndicate.”       | “A refinery accident years ago reshaped safety drills.”            |
| **Meta UI**             | “Click here to set FLAG_X.”                         | “Show your salvage permit.”                                        |
| **Localization leak**   | “Choice A is better for Act II.”                    | Keep choices neutral; translator preserves contrast, not outcomes. |

---

## 3) What each role must do

- **Showrunner** — Enforce this doc as a non-negotiable bar. Route cross-domain debates through
  loops, not ad-hoc fixes.
- **Gatekeeper** — Block merges with **Presentation** failures. Run spot checks on exports.
- **Plotwright/Scene Smith** — Write choices that are **distinct and fair** without hinting at
  hidden logic.
- **Lore Weaver** — Quarantine spoilers in canon notes; provide **player-safe summaries** for
  Curator.
- **Codex Curator** — Publish comprehension, not revelation. Mask timeline/causality that would
  spoil.
  - **⚠️ Critical Risk:** The Curator manually "strips" spoilers from `canon_pack` (Hot) to create
    `codex_entry` (Cold). This is a **high-risk manual step** with no automated guardrails. The
    Gatekeeper MUST validate every `codex_entry` for spoiler leaks before merging to Cold. See
    `01-roles/charters/codex_curator.md` for detailed guidance.
- **Style Lead** — Nudge wording away from meta/technique. Keep voice tight under pressure.
- **Art/Audio Directors** — Author **surface captions** (mood/affordance). Keep technique in logs.
- **Illustrator/Audio Producer** — Record determinism/repro notes off-surface.
- **Translator** — Preserve diegesis and neutrality; avoid meta hints.
- **PN** — Enforce gates **diegetically**; never say the quiet part (no codewords, no checks by
  name).

---

## 4) Redlines (never allowed on surfaces)

- Codeword names or gate logic (`CODEWORD: …`, `flag=true`, DC checks, RNG).
- Determinism or production details (seeds, models, DAW, plugins, sampler settings).
- Canon twists, hidden allegiances, behind-the-scenes causes.
- Internal identifiers (schema field names, section IDs as canonical truth, file paths).
- Reviewer/debug marks (“TODO”, “FIXME”, “dev note”).
- Out-of-world directives (“click”, “flag”, “roll”, “invoke schema”).

---

## 5) Phrasing patterns (copy-paste safe)

**Gate enforcement (PN/manuscript)**

- “The foreman scans your lapel. No union token—he shakes his head.”
- “You recite the salvage clause; the clerk’s shoulders drop. ‘Fine—through you go.’”
- “You’ve seen this panel before; two quick twists, and the hatch yields.”

**Choice clarity**

- “Slip through maintenance.”
- “Face the foreman.”

**Codex voice**

- **Overview (2–4 sentences), usage, context, see-also**; no causality reveals.

**Captions**

- “Shadow-side quay under sodium lamps; freight nets breathe.” _(mood/affordance)_

---

## 6) Review checklist (Gatekeeper & owners)

- [ ] No twist-revealing statements on manuscript/PN/codex/captions/alt text.
- [ ] No internals (codewords, gate logic, IDs, seeds, model names, DAW/plugins).
- [ ] Gate phrasing is **diegetic** and natural.
- [ ] Codex entries resolve links and aid comprehension; **no canon dumps**.
- [ ] Captions and audio text equivalents are atmospheric/functional, not explanatory.
- [ ] Localization preserves neutrality; no future-knowledge hints.
- [ ] Accessibility text (alt/captions) does not introduce spoilers.

---

## 7) Handling inevitable edge cases

- **Necessary ambiguity**: If masking a detail risks confusion, add a **neutral workaround** (e.g.,
  imply stakes without naming causes) or defer entry until after the reveal.
- **Research dormant**: Keep factual claims neutral on surfaces; mark `uncorroborated:<risk>` in Hot
  and schedule Researcher.
- **Topology friction**: If surface clarity requires structural change, route a **Story Spark**;
  don’t smuggle logic via wording.

---

## 8) Workflow hooks (where this applies)

- **Lore Deepening** → produce **player-safe summaries** alongside canon.
- **Codex Expansion** → run **spoiler sweep** vs canon.
- **Art/Audio Pass** → captions/text equivalents checked here.
- **Binding Run** → Gatekeeper spot-checks **Presentation** before export.
- **Translation Pass** → translator reviews for meta leakage in target language.
- **PN Dry-Run** → tag `leak-risk` issues and route back to loops.

---

## 9) Examples (miniatures)

**Bad PN**: “Access denied without CODEWORD: ASH.”  
**Good PN**: “The scanner blinks red. ‘Union badge?’ the guard asks.”

**Bad codex**: “The Dock-7 accident was deliberate sabotage by the Toll Syndicate.”  
**Good codex**: “A refinery mishap years ago reshaped safety drills and shifted cargo routes.”

**Bad caption**: “Rendered at seed 998877; the saboteur is off-frame.”  
**Good caption**: “Weld scars ladder the bulkhead; extinguishers hang spent.”

---

## 10) Enforcement & remediation

- Gatekeeper flags a **Presentation** failure → work returns to Hot with a note citing this doc.
- If the leak already reached a release view, Binder adds a **View Log** note and we cut a patched
  view from the same snapshot after fixing surfaces (no new spoilers added).

---

**TL;DR**  
Keep wonder on the page and machinery off it. If a player could infer a twist or see the gears, it
doesn’t belong on the surface. Put it in Hot, or phrase it in-world.
