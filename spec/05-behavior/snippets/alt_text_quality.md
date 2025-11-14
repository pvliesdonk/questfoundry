---
snippet_id: alt_text_quality
name: Alt Text Quality
description: One sentence, concrete nouns/relations, avoid "image of...", avoid subjective interpretation unless plan requires mood
applies_to_roles: [illustrator, art_director, gatekeeper]
quality_bars: [accessibility, presentation]
---

# Alt Text Quality

## Core Principle

Alt text makes images accessible. It must be concise, concrete, and free of technique or spoilers.

## Requirements

### One Sentence

- Single sentence preferred
- Two sentences maximum if complexity requires
- Avoid multi-sentence descriptions

### Concrete Nouns and Relations

- Describe what's visible, not interpretations
- Use specific objects, not generic categories
- Describe spatial relationships

### Avoid "Image of..."

- Screen readers already announce "image"
- Start directly with description
- Skip meta framing

### Avoid Subjective Interpretation

- Don't describe mood unless art plan specifies
- No "beautiful", "mysterious", "ominous" unless intentional
- Stick to observable elements

## Examples

### ✓ Good Alt Text

```
"Cargo bay with damaged crates stacked three stories high"
```

- Concrete: cargo bay, damaged crates
- Spatial: three stories high
- No meta framing
- Objective description

```
"Frost patterns web the airlock glass"
```

- Concrete: frost patterns, airlock glass
- Relation: patterns web (cover) the glass
- Evocative but not subjective

```
"The foreman's desk, cluttered with datachips and tools"
```

- Concrete: desk, datachips, tools
- State: cluttered (objective)
- Establishes setting

### ✗ Bad Alt Text

```
"Image of a cargo bay"
```

- Says "image of" (redundant)
- Generic, not specific
- Lacks detail

```
"A beautiful and mysterious scene"
```

- Subjective: "beautiful", "mysterious"
- Vague: no concrete objects
- No useful description

```
"This foreshadows the betrayal"
```

- Spoiler (forbidden)
- Interpretive, not descriptive
- Breaks presentation bar

```
"Generated with DALL-E using seed 1234"
```

- Technique leak (forbidden)
- Not descriptive
- Should be in off-surface log

## Role-Specific Applications

### Illustrator (Author)

- Write alt text for every image
- Keep to one sentence
- Use concrete nouns
- Match tone to Style register
- Avoid technique references

### Art Director (Guidance)

- Provide alt text guidance in art plans
- Specify when mood needed (rare)
- Note key elements to include
- Flag spoiler risks

### Gatekeeper (Validation)

- Check all images have alt text
- Validate concreteness (no vague descriptions)
- Block technique leakage
- Block spoiler content
- Check accessibility bar

## When to Include Mood

**Most of the time: Objective description**

```
✓ "Frost patterns web the airlock glass"
(Describes what's visible)
```

**Rare exception: Art plan specifies mood**

```
Art plan: "Emphasize ominous mood"
✓ "The airlock glass webbed with frost, shadows beyond unmoving"
(Mood justified by plan)
```

If unsure, stay objective.

## Spoiler Hygiene in Alt Text

Never include:

- Twist reveals ("The traitor's hidden emblem visible on badge")
- Behind-the-scenes info ("Foreshadowing the betrayal")
- Character secrets ("Her true allegiance visible in reflection")
- Gate logic ("Image shows the required hex-key")

Keep alt text player-safe.

## Technique Off-Surface

Never include:

- Model names (DALL-E, Midjourney, Stable Diffusion)
- Seeds or parameters
- Generation details
- Processing steps

Store in determinism logs off-surface.

## Terminology Consistency

Use Curator-approved terms:

```
✓ "hex-key" (approved term)
✗ "allen wrench" (not in glossary)

✓ "union token" (approved term)
✗ "ID badge" (generic)
```

Coordinate with Codex Curator for terminology.

## Register Alignment

Match Style register:

```
Industrial noir register:
✓ "The bay's dim LEDs stripe the bulkheads"
(Terse, mechanical, fitting register)

✗ "Lovely ambient lighting illuminates the beautiful industrial space"
(Flowery, breaks register)
```

Alt text is player-facing content; maintain voice.

## Validation Checklist

For each image:

- [ ] Alt text present (not empty)
- [ ] One sentence (two max)
- [ ] Concrete nouns/relations used
- [ ] No "image of..." framing
- [ ] Objective description (or mood justified by plan)
- [ ] No spoilers
- [ ] No technique (seeds, models, tools)
- [ ] Terminology matches Curator glossary
- [ ] Register matches Style guidance
- [ ] Portable for translation

## Common Fixes

**Generic → Specific:**

```
Before: "A room"
After: "Cargo bay with damaged crates stacked high"
```

**Subjective → Objective:**

```
Before: "A mysterious and beautiful scene"
After: "Frost patterns web the airlock glass"
```

**Technique Leak → Removed:**

```
Before: "Industrial viewport (DALL-E, seed 1234)"
After: "Frost patterns web the viewport"
(Technique moved to off-surface log)
```

**"Image of..." → Direct:**

```
Before: "Image of a foreman's desk"
After: "The foreman's desk, cluttered with datachips"
```

## Translation Considerations

Alt text must be translatable:

- Use simple sentence structure
- Avoid idioms unless essential
- Use Curator-approved terminology
- Coordinate with Translator for cultural portability

Illustrator provides English alt text; Translator adapts to target language with same quality standards.
