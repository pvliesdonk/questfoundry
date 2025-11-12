# Playbook — Art Touch-up

**Use when:** You want to **plan and/or add illustrations** that clarify, anchor recall, or set
mood—without leaking spoilers. Works even if the **Illustrator** is dormant (plan-only merge).

**Outcome:** An **Art Plan** (subjects, purpose, composition intent, spoiler-safe captions,
constraints) and, if active, **renders** with determinism logs + alt text. Ready for gatecheck and
Cold merge (plans may land as **deferred:art**).

---

## 1) One-minute scope (Showrunner)

- [ ] Define **slice** (which chapters/sections/scenes).
- [ ] Decide **plan-only** vs **plan+render**.
- [ ] Mark roles **active/dormant** (Art Director, Illustrator).
- [ ] Open TU: `tu-art-<scope>` and state **why** these images exist
      (clarity/recall/mood/signposting).

---

## 2) Inputs you need on screen

- **Cold snapshot** (canon, codex, style guardrails).
- Target **sections** (Hot drafts acceptable for planning).
- **PN Principles** (no plumbing on surfaces).
- Accessibility requirements (alt text patterns).

---

## 3) Do the thing (compact procedure)

**Art Director (R)**

1. **Select subjects** (per image): section anchor, **purpose** (clarify/recall/mood/signpost),
   **spoiler risk** (low/med/high).
2. Write **Composition Intent** (framing, focal points, motion cues); add **constraints** (aspect,
   palette/motif hooks, avoid clichés).
3. Draft **Caption** (player-safe, atmospheric; no twist reveals).
4. Add **Accessibility Notes** (alt-text guidance).

**Style Lead (C)** 5. Tune plan to visual guardrails; ensure motif alignment; veto drift.

**Illustrator (R, optional)** 6. If active, **render** candidates; iterate to intent.  
7. Create **Determinism Log** (if promised): seed, prompt+version, model/build hash, sampler/steps,
aspect, CFG/controls, post-process chain.  
8. Provide **Alt Text** (succinct, descriptive, spoiler-safe).

**Gatekeeper (C)** 9. **Pre-gate**: Style, Presentation Safety (no spoilers/technique-talk),
Determinism (if promised).

**Art Director (R)** 10. **Package** plan (+ renders/logs if any) into the TU.

---

## 4) Deliverables (Hot)

- **Art Plan** (per image)
  - Subject • Section anchor • **Purpose** • **Composition intent** • **Caption (player-safe)**
  - **Constraints** (aspect/palette/motif hooks/negative constraints)
  - **Accessibility notes** (alt-text guidance) • **Spoiler risk**
- **Renders** (optional): image files + **Determinism Log** (if promised) + **Alt Text**
- **Pre-gate note** (Gatekeeper): pass/fail + remediations

---

## 5) Hand-offs

- → **Binding Run**: inclusion options (plan, renders, both).
- → **Style Tune-up**: if captions clash with register.
- → **PN**: confirm diegetic references when images are mentioned in-world.

---

## 6) Definition of “done” (for this play)

- [ ] Every image has a **clear narrative purpose**.
- [ ] **Caption** is spoiler-safe and in register.
- [ ] **Alt text** exists (or guidance provided).
- [ ] If renders included and determinism is promised, **logs are complete**.
- [ ] Gatekeeper pre-gate **green** (Style/Presentation; Determinism if applicable).
- [ ] TU updated; plan can merge as **deferred:art** if Illustrator dormant.

---

## 7) Fast heuristics

- Prefer images that **disambiguate space**, **anchor faces/props**, or **telegraph affordances**
  (without reveals).
- If **spoiler risk ≥ med**, ship **plan-only** now; render later.
- Captions **evoke**; they don’t **explain** twists or technique.
- One strong motif per image beats five scattered ones.

---

## 8) RACI (micro)

| Task                   | R            | A          | C                        | I           |
| ---------------------- | ------------ | ---------- | ------------------------ | ----------- |
| Choose subjects & plan | Art Director | Showrunner | Style Lead               | Curator, PN |
| Style alignment        | Style Lead   | Showrunner | Art Director             | Gatekeeper  |
| Render & log           | Illustrator  | Showrunner | Art Director, Style Lead | Gatekeeper  |
| Pre-gate               | Gatekeeper   | Showrunner | Style Lead               | All         |
| Merge / export opts    | Showrunner   | Showrunner | Gatekeeper, Binder       | PN          |

---

## 9) Templates

**Art Plan (per image)**

```

Title: <concise label>
Section: <anchor / label>
Purpose: <clarify | recall | mood | signpost>
Composition intent: <framing, focal points, motion/leading lines>
Caption (player-safe): <1–2 lines, atmospheric>
Constraints: <aspect, palette/motif hooks, avoid…>
Accessibility: <alt-text guidance>
Spoiler risk: <low | med | high>  (if med/high: consider plan-only)
Lineage: TU <id>

```

**Determinism Log (if promised)**

```

Seed: <int>
Prompt: <text>   Prompt-Version: <vX.Y>
Model: <name/build-hash>   Sampler/Steps: <…>
Aspect: <WxH or ratio>   CFG/Controls: <…>
Post: <chain: upscaler vN, color grade preset, etc.>
Notes: <variants, negative prompts, known quirks>

```

**Alt Text**

```

<One sentence, specific nouns/relations, no spoilers.>

```

---

## 10) Anti-patterns

- “Helpful” captions that **explain the twist**.
- Technique on surface: “a 16:9 render, SDXL seed 1234.”
- Mood pieces that don’t help **comprehension** or **recall**.
- Plans that ignore **Style** motifs or PN boundaries.

---

**Cheat line (TU note):**  
“Art Touch-up (Act I): 5 images (Dock 7 hub, foreman gate, medbay recall, shadow-side berth,
wormhole toll queue). Captions spoiler-safe; 2 renders with full logs + alt text; 3 **deferred:art**
plans; Gatekeeper pre-gate green.”
