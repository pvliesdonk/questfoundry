---
procedure_id: voice_register_coherence
name: Voice & Register Coherence
description: Define/maintain register & motif kit; provide exemplars across manuscript, PN, codex, captions
roles: [style_lead]
references_schemas:
  - style_addendum.schema.json
references_expertises:
  - style_lead_voice
quality_bars: [style]
---

# Voice & Register Coherence

## Purpose
Establish and maintain consistent voice, register, and motif usage across all player-facing surfaces (manuscript, PN phrasing, codex, captions).

## Components

### Voice
**The narrative personality** (perspective, tone, distance)

Examples:
- Close 3rd person ("You slip through the hatch")
- Industrial noir (terse, mechanical, shadow-side)
- Procedural documentary (clinical, observational)

### Register
**Formality level and diction choices**

Spectrum:
- Formal: "Proceed to engineering via maintenance corridor"
- Neutral: "Head to engineering through the maintenance tunnel"
- Informal: "Slip through maintenance, hit engineering"

### Motif Kit
**Recurring imagery and phrasing patterns**

Examples (Industrial Noir):
- "relay hum" (sound motif)
- "shadow-side neon" (visual motif)
- "low-G dust" (environmental motif)
- "hex-key" (object motif)

## Steps

### 1. Define Current Register
- Extract voice/register from existing Cold surfaces
- Note formality level, POV, distance, tone
- Identify what works consistently

### 2. Build Motif Palette
- List recurring imagery (sight, sound, texture)
- Note phrases that reinforce setting
- Identify object/place naming conventions

### 3. Create Exemplars
- "Before/After" pairs showing voice consistency
- Cross-surface examples (manuscript, PN, codex, captions)
- Counter-examples (what NOT to do)

### 4. Document Style Addendum
- Voice definition
- Register spectrum (with examples)
- Motif kit (approved terms/phrases)
- Banned phrases/patterns

### 5. Validate Across Surfaces
- Manuscript: Does prose match register?
- PN Phrasing: Do gate lines fit voice?
- Codex: Do entries maintain tone?
- Captions: Do descriptions align?

## Example Style Addendum

```yaml
voice:
  perspective: "Close 3rd person present"
  tone: "Industrial noir (terse, mechanical, shadow-side)"
  distance: "Player-adjacent (you observe, you act)"
  
register:
  formality: "Neutral to informal"
  examples:
    - correct: "Slip through maintenance"
    - avoid: "Proceed to the maintenance corridor"
  sentence_rhythm: "Short under pressure (1-2 sentences). Longer in reflection (3)."
  
motif_kit:
  sound:
    - "relay hum"
    - "hydraulic hiss"
    - "PA crackle"
  sight:
    - "shadow-side neon"
    - "dim LEDs"
    - "frost-webbed viewports"
  texture:
    - "low-G dust"
    - "deck plate thrums"
    - "cold bulkheads"
  objects:
    - "hex-key" (not "wrench")
    - "datachip" (not "USB drive")
    - "union token" (not "ID card")
    
banned_phrases:
  - "You feel..."  (tells not shows)
  - "Suddenly..." (lazy tension)
  - "Click here" (meta)
  - Modern slang (breaks setting)
```

## Outputs
- `style_addendum` - Voice, register, motif kit documentation
- Exemplars (before/after pairs)
- Cross-surface validation notes

## Quality Bars Pressed
- **Style:** Voice/register/motif consistency

## Handoffs
- **To All Roles:** Provide style addendum for reference
- **From Scene Smith/Codex/PN:** Receive drafts for voice validation

## Common Issues
- **Drift:** Voice shifts gradually over time (regular audits needed)
- **Cross-Surface Mismatch:** Manuscript formal, codex informal
- **Motif Inconsistency:** Same concept described different ways
- **Banned Phrase Creep:** Prohibited patterns slip back in
