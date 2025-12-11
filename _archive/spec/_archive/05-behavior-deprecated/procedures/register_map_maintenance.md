---
procedure_id: register_map_maintenance
description: Maintain register mapping for translations preserving formality and cultural tone
version: 1.0.0
references_expertises:
  - translator_terminology
  - style_lead_voice
references_schemas:
  - translation_pack.schema.json
references_roles:
  - translator
  - style_lead
tags:
  - translation
  - register
  - style
---

# Register Map Maintenance Procedure

## Overview

Establish and maintain register mapping between source and target languages, preserving formality levels, pronoun systems, honorifics, and cultural tone equivalents.

## Source

Extracted from v1 `spec/05-prompts/loops/translation_pass.playbook.md` Step 2: "Glossary First"

## Steps

### Step 1: Decide Register System

Coordinate with Style Lead to establish target language register choices:

- Pronoun system (T/V distinction, formal/informal "you")
- Honorifics and titles appropriate for setting
- Dialect and regional variation strategy

### Step 2: Map Formality Levels

Create equivalence mapping for formality contexts:

- Dialogue vs narration registers
- Character-to-character relationships (power dynamics, intimacy)
- Setting-specific formality markers

### Step 3: Lock Decisions

Document register choices in translation_pack register_map field:

- Pronoun choices with usage notes
- Honorific system
- Tone equivalents (e.g., swear policy, endearments)
- Examples for each register level

### Step 4: Coordinate Edge Cases

Escalate ambiguous cases to Style Lead:

- Context-dependent formality shifts
- Motif-related register changes
- Idioms requiring functionally equivalent tone

### Step 5: Update Register Map

Maintain register_map deltas as translation progresses:

- New character relationship patterns
- Setting-specific register discoveries
- Corrections from Style Lead feedback

## Output

Updated register_map in translation_pack documenting all register decisions and usage patterns.

## Quality Criteria

- Register feels native to target language
- Formality levels match source intent
- Consistency across all translated surfaces
- Style Lead approval on ambiguous cases
