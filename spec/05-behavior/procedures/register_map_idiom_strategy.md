---
procedure_id: register_map_idiom_strategy
name: Register Map & Idiom Strategy
description: Maintain register map aligned with Style; document how voice translates (you/formality/slang); keep idiom portable
roles: [translator]
references_schemas:
  - translation_pack.schema.json
references_expertises:
  - translator_localization
quality_bars: [style, accessibility]
---

# Register Map & Idiom Strategy

## Purpose
Document how source register/voice translates to target language, ensuring tone and intent carry over without literal translation problems.

## Register Mapping

### Formality Levels
**Map source formality to target equivalent**

Example (English → Spanish):
- Source Formal: "Proceed to engineering"
- Target Formal: "Diríjase a ingeniería"
- Source Informal: "Hit engineering"
- Target Informal: "Ve a ingeniería"

### "You" Forms
**Handle formal/informal address**

Languages with T-V distinction (tu/vous, tú/usted, etc.):
- Document which to use consistently
- Note exceptions (authority figures, strangers)

Example:
- English "you" (neutral) → Spanish "tú" (informal, consistent with register)
- Exception: Guard dialogue uses "usted" (formal authority)

### Slang & Colloquialisms
**Map source slang to target equivalent (or remove)**

Approaches:
- **Equivalent Slang:** Find target language equivalent
- **Neutral Replacement:** Use standard term
- **Localize:** Adapt to target culture

Example:
- Source: "That's a raw deal" (English slang)
- Literal: "Eso es un trato crudo" (nonsense in Spanish)
- Equivalent: "Eso es injusto" (neutral replacement)
- Localized: "Te están timando" (Spanish colloquial equivalent)

## Idiom Strategy

### Identify Untranslatable Idioms
**Flag phrases that don't translate literally**

Examples:
- "Bite the bullet" (English)
- "It's raining cats and dogs"
- "Break a leg"

### Resolution Strategies

#### 1. Find Equivalent Idiom
**Target language has similar saying**

Example:
- Source: "Spill the beans" (reveal secret)
- Spanish Equivalent: "Soltar la sopa"
- French Equivalent: "Vendre la mèche"

#### 2. Neutral Replacement
**Use literal meaning**

Example:
- Source: "Bite the bullet"
- Neutral: "Face the difficult situation"
- Spanish: "Enfrentar la situación difícil"

#### 3. Localize
**Adapt to target culture**

Example:
- Source: "The whole nine yards" (everything)
- Localize: Use culture-appropriate expression for "everything"

### Document Decisions

```yaml
idiom_id: spill_beans
source_phrase: "spill the beans"
source_meaning: "reveal secret information"

target_language: es
resolution_strategy: equivalent_idiom
target_phrase: "soltar la sopa"
rationale: "Direct equivalent exists; maintains informal tone"

target_language: fr
resolution_strategy: equivalent_idiom
target_phrase: "vendre la mèche"
rationale: "Common expression; tone matches"
```

## POV Distance
**Handle narrative distance in translation**

English uses "you" neutrally. Some languages require choice:
- Close: tú/tu (informal)
- Distant: usted/vous (formal)

Decision factors:
- Source register (formal/informal)
- Genre conventions
- Target audience expectations

## Motif Preservation

### Recurring Phrases
**Keep motifs recognizable across translation**

Example:
- English motif: "relay hum" (recurring sound)
- Spanish: "zumbido del relé" (consistent across all instances)
- NOT: "zumbido", "ruido del relé", "sonido" (inconsistent)

### Object/Place Names
**Transliterate vs Translate**

Rules:
- Proper nouns: Transliterate ("Station Alpha" → "Estación Alpha")
- Generic objects: Translate ("maintenance key" → "llave de mantenimiento")
- Technical terms: Use target language standard ("airlock" → "esclusa de aire")

## Outputs
- `register_map` - Formality levels, you-forms, slang handling
- `idiom_strategy` - Untranslatable phrases with resolutions
- `motif_preservation` - Recurring phrase translations

## Quality Bars Pressed
- **Style:** Register aligns with source
- **Accessibility:** Readable, natural in target language

## Handoffs
- **From Style Lead:** Receive source register definition
- **To Gatekeeper:** Submit for Style validation

## Common Issues
- **Literal Translation:** Idioms that don't work
- **Formality Mismatch:** Informal source, formal target (or vice versa)
- **Motif Inconsistency:** Same phrase translated multiple ways
- **Cultural Mismatch:** Reference doesn't exist in target culture
