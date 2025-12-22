---
title: Localization Considerations for Interactive Fiction
summary: Writing IF for translation and cultural adaptation with localization-friendly text design and practices.
topics:
  - localization
  - translation
  - cultural-adaptation
  - string-externalization
  - text-expansion
  - pluralization
  - date-formatting
  - global-audiences
cluster: audience-and-access
---

# Localization Considerations for Interactive Fiction

Craft guidance for writing IF that can be translated and culturally adapted—text design, cultural assumptions, and localization-friendly practices.

---

## Why Localization Matters

### Reaching Global Audiences

- English represents ~25% of internet users
- Growing IF markets worldwide
- Translation opens significant audiences
- Cultural adaptation beyond word-for-word

### Planning for Localization

**Early Planning Benefits:**

- Cheaper than retrofitting
- Better quality translations
- Fewer structural changes needed
- Smoother localization process

**Costs of Ignoring:**

- Text embedded in code
- Hardcoded assumptions
- Untranslatable constructs
- Cultural issues in content

---

## Text Design for Translation

### String Externalization

**Principle:** All player-facing text should be separate from code.

**Good:**

```
dialogue.meeting_stranger = "Hello, I don't believe we've met."
```

**Bad:**

```
print("Hello, I don't believe we've met.")
```

### Avoiding Concatenation

**The Problem:**

Different languages have different word orders. Concatenated strings break.

**Bad:**

```
"You have " + count + " apples."
```

In German: "Sie haben 5 Äpfel." (works)
In Polish: "Masz 5 jabłek." (word order differs)

**Good:**

```
"You have {count} apples."
// Translators can reorder: "{count} jabłek masz."
```

### Placeholder Guidelines

- Use named placeholders, not positional
- Allow translators to reorder
- Document what each placeholder contains
- Provide context for all strings

### Text Expansion

Translations often expand or contract text:

| Language | Expansion vs English |
|----------|---------------------|
| German | +30% |
| French | +15-20% |
| Spanish | +20-25% |
| Japanese | -10-50% |
| Chinese | -30-50% |

**Implications:**

- UI must accommodate longer text
- Buttons need flexible sizing
- Text areas should scroll or wrap
- Test with expanded text

### Pluralization

**The Problem:**

Languages have different plural rules.

**English:** 1 apple, 2 apples (singular, plural)
**Polish:** 1 jabłko, 2 jabłka, 5 jabłek (singular, few, many)
**Arabic:** Six forms for different quantities

**Solution:**

Use pluralization systems that handle language-specific rules:

```
{count, plural,
  one {# apple}
  other {# apples}
}
```

### Gender Agreement

Many languages have grammatical gender affecting multiple words:

**English:** "The player chose the sword."
**Spanish:** "El jugador eligió la espada." (masculine player)
          "La jugadora eligió la espada." (feminine player)

**Solutions:**

- Allow gender selection where appropriate
- Use gender-neutral constructions when possible
- Provide variants for gendered languages
- Document which strings need gender variants

---

## Cultural Considerations

### Assumptions to Question

**Date and Time:**

- MM/DD/YYYY vs DD/MM/YYYY vs YYYY-MM-DD
- 12-hour vs 24-hour clock
- Week starting day (Sunday vs Monday)

**Numbers:**

- Decimal separators (. vs ,)
- Thousands separators
- Number formats

**Currency:**

- Symbol placement
- Decimal conventions
- Currency-appropriate amounts

**Measurements:**

- Metric vs imperial
- Temperature scales
- Distance units

### Cultural References

**What Translates Poorly:**

- Idioms ("raining cats and dogs")
- Sports metaphors
- Pop culture references
- Political references
- Holidays and celebrations
- Food and customs

**Solutions:**

- Use universal concepts where possible
- Provide translator notes for references
- Allow localized equivalents
- Consider cultural adaptation, not just translation

### Names and Terms

**Character Names:**

- Pronounceable across languages?
- Offensive meanings in other languages?
- Consider localized name variants

**Fictional Terms:**

- Can they be translated or should they remain?
- Are they pronounceable?
- Do they carry unintended meanings?

### Visual and Symbolic

**Consider:**

- Color symbolism varies by culture
- Gestures mean different things
- Religious symbols sensitive
- Direction (left-to-right vs right-to-left)

---

## Writing for Translation

### Clear, Simple Prose

**Helps Translation:**

- Direct sentence structure
- Common vocabulary
- Explicit subjects (not just pronouns)
- Consistent terminology

**Hinders Translation:**

- Complex nested clauses
- Ambiguous pronouns
- Idiomatic expressions
- Wordplay and puns

### Terminology Consistency

Use the same term for the same concept throughout:

**Bad:**

- "sword" in chapter 1
- "blade" in chapter 3
- "steel" in chapter 5
- (All referring to same object)

**Good:**

- "sword" consistently
- Or establish pattern: "sword" in narration, "blade" in character's voice

### Avoiding Untranslatables

**Wordplay:**

If meaning depends on word sounds or spellings, it may not translate.

**Option A:** Accept loss in translation
**Option B:** Provide translator note with intended effect
**Option C:** Allow localized equivalent jokes

**Cultural Specifics:**

References meaningful only to source culture may need adaptation or explanation.

### Context for Translators

**Provide:**

- Speaker information
- Scene context
- Emotional tone
- Where text appears (UI, dialogue, narration)
- Related strings that should match

---

## Technical Considerations

### Text Direction

**Left-to-Right (LTR):** English, Spanish, French, German, etc.
**Right-to-Left (RTL):** Arabic, Hebrew, Persian, Urdu

**Requirements for RTL:**

- UI must mirror
- Text alignment reverses
- Punctuation positioning
- Mixed LTR/RTL content handling

### Character Sets

**Support:**

- Unicode (UTF-8) throughout
- Extended Latin (accents, diacritics)
- Non-Latin scripts (Cyrillic, Arabic, Asian)
- Special characters and symbols

**Testing:**

- Test with actual translated text
- Check character rendering
- Verify font support
- Test text input

### Font Considerations

**Requirements:**

- Font must support target languages
- Character coverage varies by font
- Some languages need specific fonts
- Size may need adjustment by language

---

## Localization-Friendly IF Structures

### Choice Text

**Consider:**

- Choices must make sense in translation
- Context preserved across languages
- Pronoun references clear
- Cultural appropriateness

### Variable Text

**Challenge:**

Dynamic text insertion complicates translation.

**Example:**

```
"You picked up the {item}."
```

In gendered languages, article and verb may need to agree with item gender.

**Solutions:**

- Provide gender metadata for items
- Use flexible translation systems
- Allow multiple item description patterns
- Simplify where possible

### Branching and State

**Document for Translators:**

- What state affects which text
- How choices change dialogue
- Which variations exist
- Conditions for seeing text

---

## Localization Process

### Preparation

1. Externalize all strings
2. Document context
3. Create translation memory terms
4. Establish style guide
5. Choose localization management tool

### Translation

1. Professional translators (not just bilingual speakers)
2. IF/game localization experience preferred
3. Provide context and reference materials
4. Allow questions and clarification
5. Review and QA process

### Integration

1. Import translated strings
2. Test all languages
3. Check text fit and display
4. Verify consistency
5. User testing with native speakers

### Maintenance

1. Track string changes
2. Update translations for changes
3. Maintain translation memory
4. Document decisions

---

## Common Mistakes

### Hardcoded Text

Text embedded in code rather than externalized.

**Fix:** All player-facing text in resource files.

### Concatenation

Building sentences from fragments.

**Fix:** Complete sentences with placeholders.

### Assuming English Rules

Plural, gender, word order assumptions.

**Fix:** Use localization-aware systems.

### No Context

Translators working blind.

**Fix:** Provide context for every string.

### Machine Translation Only

Google Translate as final product.

**Fix:** Professional translation with proper review.

### Ignoring Cultural Issues

Direct translation without cultural consideration.

**Fix:** Cultural adaptation alongside translation.

### No Budget/Time for Localization

Treating localization as free afterthought.

**Fix:** Budget time and money from project start.

---

## Quick Reference

| Area | Key Practice |
|------|--------------|
| Text | Externalize, no concatenation, named placeholders |
| Expansion | Design for 30% longer text |
| Plurals | Language-specific plural rules |
| Gender | Support grammatical gender where needed |
| Culture | Question assumptions, provide context |
| References | Document or avoid culture-specific content |
| Technical | Unicode, RTL support, font coverage |
| Process | Professional translators, review, testing |
| Maintenance | Track changes, update translations |
