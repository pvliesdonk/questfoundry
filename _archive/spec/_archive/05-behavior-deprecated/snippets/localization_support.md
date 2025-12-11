---
snippet_id: localization_support
name: Localization Support
description: Supply glossary and register notes without prescribing translation solutions; note variants and cultural portability
applies_to_roles: [codex_curator, style_lead, translator]
quality_bars: [accessibility, style]
---

# Localization Support

## Core Principle

Curator and Style Lead prepare content for translation without dictating solutions. Provide context and constraints; let Translator determine target-language approach.

## Curator Responsibilities

### Glossary Preparation

Supply bilingual glossary foundations:

```yaml
term: "union token"
definition: "Physical ID marking union membership"
usage_context: "social gates, identity checks"
cultural_notes: "Labor union context; may need cultural adaptation"
portability: medium
```

### Register Notes

Document formality and tone:

```yaml
register: "neutral to informal"
formality_examples:
  - avoid: "Proceed to engineering"
  - prefer: "Head to engineering"
tone: "industrial, terse, working-class"
```

### Cultural Portability Assessment

Flag elements needing translation strategy:

- **High portability:** Universal concepts (airlocks, tools)
- **Medium portability:** Context-dependent (union membership)
- **Low portability:** Culture-specific (US labor law references)

### Variants by Region

Note regional differences if applicable:

```yaml
term: "wrench"
variants:
  US: "wrench"
  UK: "spanner"
localization_note: "Use target-appropriate tool terminology"
```

## Style Lead Responsibilities

### Voice Portability

Document voice elements for translation:

```yaml
voice:
  perspective: "Close 3rd person present"
  distance: "Player-adjacent"
  tone: "Industrial noir"
translation_guidance:
  - "Maintain terse sentence rhythm"
  - "Preserve mechanical/shadow-side imagery"
  - "Adapt formality to target T-V distinction"
```

### Motif Kit for Translation

Identify portable vs. language-specific motifs:

```yaml
motifs:
  - phrase: "relay hum"
    portability: high
    guidance: "Mechanical sound, translatable"
  - phrase: "shadow-side neon"
    portability: medium
    guidance: "Noir imagery; adapt to target culture's noir conventions"
```

### Banned Phrases (Portable)

Flag phrases to avoid universally:

```yaml
banned_phrases:
  - pattern: "You feel..."
    reason: "Tells not shows (universal)"
  - pattern: "Suddenly..."
    reason: "Lazy tension (universal)"
  - pattern: "Click here"
    reason: "Meta (universal)"
```

## Translator Coordination

### Glossary Feedback Loop

1. Curator supplies English glossary + notes
2. Translator proposes target equivalents
3. Curator reviews for consistency
4. Approved equivalents added to bilingual glossary

### Cultural Adaptation Proposals

When cultural portability low:

```yaml
source_element: "Union token gates"
portability: medium
translator_proposal:
  language: es
  adaptation: "Worker credential system"
  rationale: "Union context varies; broader 'worker credential' more portable"
curator_approval: pending
```

### Register Mapping

Translator documents how source register maps to target:

```yaml
source_register: "neutral to informal"
target_language: fr
target_register: "tu (informal) for consistency"
exceptions:
  - context: "Authority figures"
    register: "vous (formal)"
rationale: "Matches industrial working-class setting"
```

## What NOT to Prescribe

### ❌ Don't Dictate Translation

```yaml
term: "union token"
DO NOT: "Translate as 'tarjeta sindical'"
```

### ✓ Provide Context Instead

```yaml
term: "union token"
definition: "Physical ID marking union membership"
usage: "Social gates, identity checks"
cultural_context: "Labor union setting"
portability: medium
translator_notes: "Adapt to target labor culture norms"
```

### ❌ Don't Prescribe Grammar

```yaml
DO NOT: "Use subjunctive mood here"
```

### ✓ Provide Intent Instead

```yaml
intent: "Express uncertainty without revealing outcome"
source_example: "The foreman might help"
translator_guidance: "Maintain speculative tone appropriate to target language"
```

## Portability Flags

### High Portability

- Universal concepts (technical equipment, basic emotions)
- Direct translation usually works
- Minimal cultural adaptation needed

### Medium Portability

- Context-dependent (labor relations, social structures)
- May need cultural adaptation
- Equivalent concepts exist but differ by culture

### Low Portability

- Culture-specific (legal systems, historical references)
- Require localization strategy
- Direct translation may not convey meaning

## Validation

**Curator checks:**

- Glossary complete for key terms
- Cultural portability assessed
- Variants documented
- No prescriptive translation dictates

**Style Lead checks:**

- Register guidance provided
- Motif portability assessed
- Voice elements documented
- Banned phrases flagged universally

**Translator checks:**

- Context sufficient for decisions
- Cultural notes helpful
- Register guidance clear
- Freedom to adapt maintained

## Examples

### Good Localization Support

```yaml
term: "hex-key"
definition: "Six-sided maintenance tool, standard for station equipment"
usage_context: "Technical gates, tool inventory"
cultural_notes: "Generic tool; adapt to target tool terminology norms"
portability: high
visual_reference: "Allen wrench / Allen key"
translator_freedom: "Use culturally appropriate tool name"
```

### Poor Localization Support

```yaml
term: "hex-key"
translation: "llave hexagonal"
```

(Prescriptive; doesn't let Translator assess cultural fit)

### Good Register Guidance

```yaml
formality: "neutral to informal"
examples:
  - avoid: "Proceed to the maintenance corridor"
  - prefer: "Head to maintenance"
translator_guidance: "Adapt formality to target T-V distinction norms"
```

### Poor Register Guidance

```yaml
formality: "Use 'tú' in Spanish, 'tu' in French"
```

(Prescriptive; doesn't account for Translator's cultural expertise)
