---
snippet_id: register_alignment
name: Register Alignment
description: Tone consistent with Style; terminology consistent with Curator; portable for translation
applies_to_roles: [illustrator, audio_producer, translator, style_lead, codex_curator]
quality_bars: [style, accessibility]
---

# Register Alignment

## Core Principle

All player-facing content (prose, alt text, captions, codex) must maintain consistent register/tone as defined by Style Lead.

## What is Register?

**Register:** Formality level and diction choices

Examples:

- Formal: "Proceed to engineering via maintenance corridor"
- Neutral: "Head to engineering through maintenance"
- Informal: "Slip through maintenance, hit engineering"

**For Cold (example: industrial noir):**

- Register: Neutral to informal
- Tone: Terse, mechanical, shadow-side
- Rhythm: Short under pressure, longer in reflection

## Register in Different Surfaces

### Manuscript Prose (Scene Smith)

```
✓ "The relay hum thrums through deck plates."
(Terse, mechanical, fitting register)

✗ "The lovely ambient machinery creates a beautiful soundscape."
(Flowery, breaks register)
```

### Alt Text (Illustrator)

```
✓ "Frost patterns web the airlock glass"
(Terse, concrete, fitting register)

✗ "A stunningly beautiful display of intricate crystalline formations"
(Overly formal, flowery)
```

### Audio Captions (Audio Producer)

```
✓ "[Relay hum thrums through bulkheads]"
(Mechanical, fitting register)

✗ "[Delightful mechanical ambience fills the space]"
(Subjective, breaks register)
```

### Codex Entries (Codex Curator)

```
✓ "Relay Hum: Constant mechanical sound from station power relays"
(Neutral, informative, fitting register)

✗ "Relay Hum: A fascinating auditory phenomenon created by..."
(Too formal/academic)
```

## Terminology Consistency

Use Curator-approved terms across ALL surfaces:

**Approved term:** "union token"

```
✓ Prose: "Your union token gets you past the foreman"
✓ Alt text: "A union token lying on the desk"
✓ Caption: "[Scanner beeps—union token accepted]"
✓ Codex: "Union Token: Physical ID marking union membership"
```

**Inconsistent (forbidden):**

```
❌ Prose: "union token"
❌ Alt text: "ID badge"
❌ Caption: "worker card"
❌ Codex: "membership credential"
```

Each surface uses same approved term.

## Style Lead Responsibilities

Define register in Style Addendum:

```yaml
voice:
  perspective: "Close 3rd person present"
  tone: "Industrial noir (terse, mechanical, shadow-side)"
  distance: "Player-adjacent"
  
register:
  formality: "Neutral to informal"
  examples:
    correct: "Slip through maintenance"
    avoid: "Proceed to maintenance corridor"
  sentence_rhythm: "Short under pressure (1-2). Longer in reflection (3)."
  
banned_phrases:
  - "You feel..." (tells not shows)
  - "Suddenly..." (lazy tension)
  - Modern slang (breaks setting)
```

Provide to all content creators.

## Codex Curator Responsibilities

Maintain terminology glossary:

```yaml
term: "hex-key"
definition: "Standard six-sided maintenance tool"
register_note: "Informal/neutral; avoid formal 'hexagonal wrench'"
approved_usage: "hex-key" (consistent across all surfaces)
```

Supply to all roles for consistency.

## Illustrator Responsibilities

Write alt text matching register:

```yaml
style_register: "Neutral to informal, terse, industrial"

✓ "Cargo bay with damaged crates stacked high"
(Terse, concrete, fitting)

✗ "An atmospheric image depicting a cargo storage facility"
(Formal, verbose, breaks register)
```

Coordinate with Style Lead if uncertain.

## Audio Producer Responsibilities

Write captions matching register:

```yaml
style_register: "Terse, mechanical, industrial"

✓ "[Hydraulic hiss as airlock seals]"
(Terse, mechanical)

✗ "[The airlock creates a pleasant hissing sound as it seals shut]"
(Verbose, subjective, breaks register)
```

Use Style motif kit (e.g., "relay hum", "PA crackle").

## Translator Responsibilities

Maintain register in target language:

```yaml
source_register: "Neutral to informal"
target_language: es
target_register: "tú (informal) for consistency"

source: "Slip through maintenance"
target: "Ve a mantenimiento" (informal, maintains register)
NOT: "Diríjase a mantenimiento" (formal, breaks register)
```

Adapt formality to target language norms while preserving tone.

## Portability for Translation

Write content that translates cleanly:

### Good (Portable)

```
"The foreman blocks the door"
→ Translates cleanly, register maintainable
```

### Poor (Portability Issues)

```
"The foreman's like, blocking the door, y'know?"
→ Slang/colloquialisms hard to translate
```

Style Lead notes portable vs. challenging phrases:

```yaml
motif: "relay hum"
portability: high
guidance: "Mechanical sound, translatable"

motif: "shadow-side neon"
portability: medium
guidance: "Noir imagery; adapt to target culture's noir conventions"
```

## Gatekeeper Validation

Pre-gate checks:

- [ ] Register matches Style Addendum
- [ ] Terminology matches Curator glossary
- [ ] Tone consistent (not formal then informal)
- [ ] Banned phrases absent
- [ ] Portable for translation (if localization planned)

**Block if:**

- Register drift (formal where informal expected)
- Terminology inconsistent (different terms for same concept)
- Banned phrases present
- Tone wobble (shifts mid-section)

## Common Issues

### Register Drift

```
Section starts:
✓ "The relay hum thrums. Deck plates vibrate."

Section ends:
❌ "You proceed with great alacrity toward the engineering facility."
(Drifted to formal register)

Fix:
✓ "You head to engineering."
```

### Terminology Inconsistency

```
❌ Paragraph 1: "hex-key"
❌ Paragraph 2: "allen wrench"
❌ Alt text: "maintenance tool"

Fix:
✓ All use: "hex-key" (Curator-approved term)
```

### Tone Wobble

```
❌ "The cargo bay's dim and grimy. It's such a lovely space, really quite charming."
(Starts industrial, becomes flowery)

Fix:
✓ "The cargo bay's dim and grimy. Stacks of damaged crates reach three stories high."
(Maintains industrial tone)
```

### Portability Issues

```
❌ "The foreman's totally not having it, y'know?"
(Slang doesn't translate)

Fix:
✓ "The foreman refuses."
(Simple, portable)
```

## Validation Across Surfaces

**Scene Smith prose:**

- Matches Style register? ✓

**Illustrator alt text:**

- Matches Style register? ✓
- Uses Curator terminology? ✓

**Audio Producer captions:**

- Matches Style register? ✓
- Uses Style motif kit? ✓

**Codex Curator entries:**

- Matches Style register? ✓
- Defines Curator terminology? ✓

**Translator localization:**

- Maintains register in target language? ✓
- Uses approved term translations? ✓

All surfaces aligned = register coherence achieved.
