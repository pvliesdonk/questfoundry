# Section Brief — Writing Instructions (Layer 2)

> **Use:** Clear instructions from Plotwright to Scene Smith for drafting a specific section. Defines
> goal, beats, choice intents, and expected outcomes—enough detail that Scene can write without
> guessing.
>
> **Producer:** Plotwright
> **Consumer:** Scene Smith (primary), Style Lead, Lore Weaver (context)

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md`
- Role charters: `../../01-roles/charters/plotwright.md` · `../../01-roles/charters/scene_smith.md`
- **Layer 2 references:** `../glossary.md` (Section, Choice, Hub, Gateway) · `../conventions/choice_integrity.md`

---

## Structure

### Header

```
Section Brief — <section title>
ID: SB-<NNN>
For Anchor: anchor<NNN>
Topology Context: <Hub | Loop | Gateway | Linear>
Author: Plotwright
TU: <tu-id>
Edited: <YYYY-MM-DD>
Status: draft | ready | implemented
```

---

## 1) Goal

> What does this section accomplish narratively?

**Example:**
"Player discovers the lighthouse is locked and must choose between finding a key or seeking an
alternative route."

---

## 2) Key Beats

> 2-4 story moments that must appear in the prose (sequence matters).

**Format:**
1. [Opening beat - continuity/context]
2. [Development beat - tension/discovery]
3. [Optional: complication beat]
4. [Setup for choice block]

**Example:**
1. Chase and crew arrive at lighthouse door
2. Door is locked; key symbol visible but unfamiliar
3. Skye spots alternate path around the rocks (risky)
4. Player must decide approach

---

## 3) Choice Intents (Contrastive)

> Describe what each choice represents, with expected targets and differentiating outcomes.

**Format per choice:**
- **Intent:** <clear verb + object>
- **Target:** anchor<NNN>
- **Outcome difference:** <how this path differs from siblings>
- **Gate (if any):** <diegetic condition; empty if ungated>

**Example:**
- **Intent:** Search for the key
  - **Target:** anchor005
  - **Outcome:** Slower but safer; leads to foreman encounter
  - **Gate:** None

- **Intent:** Climb around the rocks
  - **Target:** anchor006
  - **Outcome:** Faster but risky; bypasses foreman, different discovery
  - **Gate:** None (but PN will note weather/waves)

---

## 4) Expected Outcomes (Player-Safe)

> What should the player understand after reading this section? No spoilers.

**Example:**
"Player knows: lighthouse is locked, key is missing, alternate route exists but looks dangerous.
Foreman may have information."

---

## 5) Open Questions / Dependencies

> What needs clarification from Lore, Style, or Curator before Scene drafts?

**Example:**
- [ ] Lore: What's the key symbol meaning? (Canonical or local legend?)
- [ ] Curator: Should "lighthouse keeper" get a codex anchor?
- [ ] Style: Tone for risk descriptions (playful vs cautionary)?

---

## 6) Constraints & Notes

Optional section for:
- **Continuity:** References to prior sections/choices
- **Accessibility:** Specific needs (alt text opportunities, content warnings)
- **Art/Audio cues:** Moments that benefit from illustration/sound
- **Micro-context:** When convergence requires immediate reflection

**Example:**
- If arriving from anchor003 (rushed path), open with breathlessness/urgency
- Art opportunity: lighthouse door close-up with key symbol
- Audio cue: waves crashing (intensity based on weather state)

---

## Hot vs Cold

### Hot only

Section briefs remain in **Hot** as working documents:
- Guide for Scene Smith during drafting
- Reference for Style/Lore when reviewing sections
- Not exported to players or Cold

---

## Lifecycle

1. **Plotwright** creates brief during Story Spark or topology update
2. **Scene Smith** drafts section using brief as guide
3. **Style/Lore** consult brief if reviewing section
4. **Brief archived** after section merges to Cold (retained for traceability)

---

## Example (complete)

```markdown
Section Brief — The Lighthouse Lock
ID: SB-005
For Anchor: anchor005
Topology Context: Hub (3 exits)
Author: Plotwright
TU: TU-2025-11-24-PW02
Edited: 2025-11-24
Status: ready

---

## Goal

Establish lighthouse as locked hub; present key search vs bypass choice; set up foreman encounter
or alternate discovery path.

## Key Beats

1. Chase and Skye arrive at lighthouse; door is heavy oak with brass lock
2. Unusual key symbol etched above lock (anchor for future codex entry)
3. Skye reports seeing movement around rocks (alternate route)
4. Player chooses: methodical search or risky climb

## Choice Intents

**Choice A: Search the area for clues**
- Target: anchor006
- Outcome: Slower, leads to foreman who knows key location
- Gate: None

**Choice B: Follow Skye's sighting around the rocks**
- Target: anchor007
- Outcome: Faster, bypasses foreman, discovers oil leak first
- Gate: None (weather affects difficulty, PN will phrase)

**Choice C: Try to force the door**
- Target: anchor008
- Outcome: Fails but triggers helpful NPC (maintenance worker)
- Gate: None

## Expected Outcomes (Player-Safe)

Player understands:
- Lighthouse is locked and key is special/symbolic
- Multiple approaches exist (thorough vs direct vs forceful)
- Choice affects encounter order (foreman vs discovery vs maintenance)

## Open Questions

- [x] Lore: Key symbol meaning — ✅ Maritime guild tradition (canon-pack-017)
- [ ] Curator: Need codex entry for "Maritime Guild"?
- [ ] Style: Skye's voice - playful or focused here?

## Constraints

- If player came from anchor003 (rushed), start with urgency/breathing
- Art cue: Close-up of door + key symbol (reference maritime tradition)
- Audio cue: Wind + distant gulls (calm establishes tone)
- Micro-context: If player talked to Mayor earlier, Chase remembers mention of "old keys"
```

---

## Validation checklist

Before Scene Smith drafts:

- [ ] Goal is clear and achievable in one section
- [ ] Beats are specific enough to guide prose
- [ ] Choice intents are contrastive (different outcomes, not synonyms)
- [ ] All open questions are resolved or explicitly deferred
- [ ] Continuity noted if this section converges from multiple paths
- [ ] Player-safe language throughout (no spoilers in outcomes)

---

**Created:** 2025-11-24
**Status:** Initial template
