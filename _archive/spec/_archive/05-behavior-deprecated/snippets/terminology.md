---
snippet_id: terminology
name: Terminology
description: Use Curator-approved terms; if none exist, propose and file hook; maintain glossary consistency
applies_to_roles: [translator, codex_curator, scene_smith, lore_weaver, style_lead]
quality_bars: [integrity, accessibility]
---

# Terminology

## Core Principle

Use Curator-approved terminology consistently across all surfaces. When new terms needed, propose via Hook Harvest rather than inventing ad-hoc.

## Workflow

### When Term Exists (Curator Glossary)

1. Search Curator glossary
2. Use approved term exactly
3. Maintain consistency across surfaces

### When Term Doesn't Exist

1. **Do NOT invent term on the spot**
2. Propose term via Hook Harvest
3. Include context and usage example
4. Wait for Curator approval
5. Use approved term once available

### For Translation

1. Check Curator glossary for source term
2. Use Translator-provided equivalent
3. If no equivalent exists, propose to Curator
4. Coordinate bilingual glossary updates

## Glossary Structure

Curator maintains:

```yaml
term: "union token"
definition: "Physical ID card marking union membership"
usage_context: "gates, social standing"
variants: []
translations:
  es: "ficha sindical"
  fr: "jeton syndical"
approved_by: codex_curator
status: approved
```

## Role-Specific Applications

**Translator:**

- Use Curator-approved terms
- If none exist, propose and file hook
- Coordinate bilingual glossary
- Never invent translations ad-hoc

**Codex Curator:**

- Maintain canonical glossary
- Approve new term proposals
- Supply register notes without prescribing translation
- Note variants and cultural portability

**Scene Smith:**

- Use approved terminology in prose
- Flag terms needing codex anchor
- Avoid synonyms for established terms

**Lore Weaver:**

- Ensure canon uses approved terms
- Propose new terms via Hook Harvest
- Maintain terminology consistency in canon packs

**Style Lead:**

- Include approved terms in motif kit
- Flag terminology drift in review
- Coordinate with Curator for register alignment

## Common Issues

### Ad-Hoc Invention

❌ Scene Smith writes "badge" for concept not yet in glossary
✓ Scene Smith files hook proposing "union token" with definition

### Synonym Drift

❌ Same concept called "badge", "token", "card" across sections
✓ Single approved term "union token" used consistently

### Translation Mismatch

❌ Translator invents "tarjeta de unión" without Curator coordination
✓ Translator uses Curator-approved "ficha sindical"

### Unclear Scope

❌ "Relay" used for both machinery and communication protocol
✓ Curator defines "relay (mechanical)" vs "relay (comms)" as distinct terms

## Hook Harvest Integration

When proposing new term:

```yaml
hook_type: terminology_proposal
term: "hex-key"
definition: "Standard maintenance tool for station equipment"
usage_context: "technical gates, tool inventory"
example_sentence: "The panel requires a hex-key you don't have"
proposer: scene_smith
```

Curator reviews, approves, adds to glossary.

## Glossary Accessibility

**Curator responsibilities:**

- Descriptive headings for glossary sections
- Plain language definitions
- Avoid circular definitions
- Provide usage examples
- Note pronunciation if non-obvious

**Example entries:**
✓ "Hex-key: Standard six-sided maintenance tool"
❌ "Hex-key: Tool of hexagonal configuration"

## Localization Support

**Curator provides:**

- Cultural portability notes
- Register guidance (formal/informal)
- Variants by region if applicable
- Sound-alike warnings (false friends)

**Example:**

```yaml
term: "union token"
translations:
  es: "ficha sindical"
  fr: "jeton syndical"
localization_notes:
  es: "Avoid 'tarjeta' (suggests credit card)"
  fr: "Maintain 'jeton' (coin-like object, fitting register)"
```

## Validation

**Gatekeeper checks:**

- Terms used match Curator glossary
- No ad-hoc invented terminology
- Consistent usage across surfaces
- Translations align with approved equivalents

**Curator audits:**

- Regular glossary coverage review
- Identify terminology gaps
- Coordinate with Scene Smith/Lore for proposals
- Update glossary as canon expands

## Integration with Codex

When term appears in glossary AND needs player-facing explanation:

- Curator creates codex entry (player-safe)
- Entry cross-references gameplay relevance
- Avoid spoiling gate logic ("what it does" not "how it's checked")
