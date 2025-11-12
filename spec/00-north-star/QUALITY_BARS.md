# Quality Bars — What Must Be True Before We Merge to Cold

These are human-level checks the **Gatekeeper** applies before any Hot change merges to Cold. No
tooling required (yet). Bars are written so creators can self-check; the Gatekeeper formalizes the
pass/fail with notes.

> Cold = export/play-safe. If a bar fails, we don’t merge.

---

## The Bars (overview)

1. **Integrity** — References resolve; no unintended dead ends.
2. **Reachability** — Critical beats can be reached via at least one viable path.
3. **Nonlinearity** — Hubs/loops/gateways/codewords are deliberate and meaningful.
4. **Gateways** — Conditions are consistent, enforceable, and spoiler-safe on the surface.
5. **Style** — Voice, register, motifs, and visual guardrails hold.
6. **Determinism** — Promised assets are reproducible from recorded parameters.
7. **Presentation** — Player-facing surfaces (manuscript/codex/PN) reveal no spoilers or internals.
8. **Accessibility** — Navigation is clear; alt text present; sensory overload avoided.

Each bar has quick checks, common failures, and remediation hints.

---

## 1) Integrity

**What it means**

- All links/section references resolve.
- No accidental dead ends (only intentional terminals).
- Codeword/state effects don’t create contradictions downstream.

**Quick checks**

- “Every choice leads somewhere” scan.
- “Any section referenced must exist” scan.
- If terminal: marked clearly as _The End_ (or equivalent).

**Common failures & fixes**

- _Dangling link_: create the section or retarget the choice.
- _Forgotten return from a loop_: add a return path or mark deliberately terminal.

---

## 2) Reachability

**What it means**

- “Must-see” beats (plot keystones, required items) are reachable from at least one path without
  paradoxes.

**Quick checks**

- List the keystones → demonstrate at least one viable route to each.
- If a gateway is required, ensure at least one route to obtain its condition.

**Common failures & fixes**

- _Locked content with no key_: add an alternative key path or remove the lock.
- _Unrealistic chain_: reduce dependency depth or provide a second route.

---

## 3) Nonlinearity

**What it means**

- Hubs, loops, and gateways exist where intended and matter to play (not decorative).

**Quick checks**

- Identify planned hubs/loops/gateways → show them in the current topology.
- For loops: show at least one meaningful “return with difference” (usually via codeword/state).
- First-choice integrity: avoid early funnels where sibling choices are functionally equivalent. If
  convergence is necessary, the next scene’s first paragraph must reflect the chosen path (lexical,
  behavioral, or situational).

**Common failures & fixes**

- _Cosmetic loop_: have the return alter stakes, tone, or options.
- _Hub too thin_: add branches with distinct outcomes or merge if unnecessary.

---

## 4) Gateways

**What it means**

- Conditions (codewords, items, states) are consistent across sections and can be enforced by the PN
  without leaks.

**Quick checks**

- Each gateway states a single clear condition.
- There exists a plausible in-world way the PN can enforce it without exposing internals.
- Negative conditions (“unless you have…”) don’t contradict positive checks elsewhere.

**Common failures & fixes**

- _Ambiguous condition wording_: rewrite condition and surface phrasing.
- _Hidden unwinnable state_: add catch-up or fail-forward beats.

---

## 5) Style

**What it means**

- Prose voice, register, motifs align with the **Style Lead**’s guardrails.
- Visual/aesthetic cues (captions, composition intent) align with style.
- PN remains in-voice.

**Quick checks**

- Read a random triad of sections aloud: does the voice shift?
- Captions feel like the same book?
- PN phrasing matches the agreed register (e.g., dry, sardonic, earnest).

**Common failures & fixes**

- _Tone wobble_: run a **Style Tune-up** loop.
- _Over-exposition_: move detail to codex; keep player surface lean.

---

## 6) Determinism (when promised)

**What it means**

- Visual/audio assets can be reproduced from recorded parameters (seed, prompt version, model,
  aspect, chain).
- If determinism is _not_ promised, this bar is N/A.

**Quick checks**

- Parameters logged and sufficient to re-render/re-generate.
- If plan-only (no execution yet), marked `deferred:<asset>` with constraints reviewed.
- When branches converge, the next scene’s first paragraph reflects the entering state; at least one
  affordance is conditioned by that state.

**Common failures & fixes**

- _Missing seeds/versions_: regenerate with logging or accept as non-deterministic (explicitly).
- _Plan contradicts style_: adjust constraints with Style Lead.

---

## 7) Presentation

**What it means**

- Player-facing surfaces (manuscript, codex, PN) contain **no spoilers** or internal plumbing.
- All visible text respects spoiler hygiene and uses in-world language.

**Quick checks**

- Manuscript choices are clear, not meta.
- Codex omits hidden gate logic, reveals only player-safe summaries.
- PN lines never mention codewords/state directly; uses in-world wording.
- No technique talk (seeds, models, DAW sessions) on player surfaces.
- Diegetic transit present on anchor jumps (avoid “teleporting”).

**Common failures & fixes**

- _Spoiler in codex_: move detail to canon notes; keep player-safe summary.
- _PN leaks plumbing_: rephrase to diegetic checks ("If the foreman vouched for you, proceed…").
- _Internal labels visible_: replace with player-safe refs or in-world terms.

---

## 8) Accessibility

**What it means**

- Navigation is clear and predictable.
- Alt text present for all images.
- Sensory considerations respected (avoid flashing, provide content notes where appropriate).

**Quick checks**

- Alt text present for images; describes composition/intent without spoilers.
- Headings and link text are informative (not "click here").
- Navigation breadcrumbs or section markers help orient readers.
- Content notes flag potentially distressing material.

**Common failures & fixes**

- _Missing alt text_: add descriptive captions (coordinate with Art Director).
- _Ambiguous navigation_: revise link text; add section headers or breadcrumbs.
- _Sensory overload risk_: add content note or adjust composition plan.

---

## Gatekeeper's Checklist (per change/batch)

- [ ] **Integrity**: No unresolved refs; terminals are intentional.
- [ ] **Reachability**: Keystone beats reachable; gateway keys obtainable.
- [ ] **Nonlinearity**: Planned hubs/loops/gateways are present and alter outcomes.
- [ ] **Gateways**: Conditions clear, consistent, enforceable by PN.
- [ ] **Style**: Voice/guardrails hold; captions aligned; PN in-voice.
- [ ] **Determinism** _(if promised)_: Params logged; plan-only items deferred & reviewed.
- [ ] **Presentation**: No spoilers; codex/manuscript/PN player-safe; no internal plumbing visible.
- [ ] **Accessibility**: Alt text present; navigation clear; sensory considerations respected.

**Outcome**: `pass` or `fail` with **specific remediation notes**. The Showrunner can only merge on
`pass` (or on an explicitly documented exception).

---

## Exceptions (rare, explicit)

- A bar may be **temporarily waived** only if:
  - The Showrunner records a rationale and time-boxed remediation plan, **and**
  - The Gatekeeper logs the exact scope and player impact, **and**
  - The waiver is visible in the change record and scheduled for a follow-up loop.

Examples:

- Temporarily shipping a plan-only art caption while Illustrator is dormant.
- Carrying `uncorroborated:low` factual detail until Researcher returns.

---

## Anti-patterns (don’t)

- Treating Cold as “staging”. Cold is **ship-ready**.
- Hiding spoilers in captions or PN side comments.
- Using codeword names in text. Use in-world phrasing.
- Bundling unrelated changes; large changes should be split so each bar can be judged cleanly.

---

**TL;DR**  
If it’s going to **Cold**, it must be reachable, intentional, consistent, in-voice, spoiler-safe,
and—when we promise it—deterministic. The Gatekeeper guards the door so the player never sees the
duct tape.
