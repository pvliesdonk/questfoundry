---
procedure_id: terminology_coordination
description: Coordinate bilingual glossary and term consistency across roles
version: 1.0.0
references_expertises:
  - translator_terminology
  - codex_curator_terminology
  - style_lead_voice
references_schemas:
  - bilingual_glossary.schema.json
  - translation_pack.schema.json
references_roles:
  - translator
  - style_lead
  - codex_curator
tags:
  - translation
  - terminology
  - glossary
---

# Terminology Coordination Procedure

## Overview

Create and maintain bilingual glossary with Style Lead and Codex Curator, ensuring term consistency, avoiding false friends, and locking translation decisions.

## Source

Extracted from v1 `spec/05-prompts/loops/translation_pass.playbook.md` Step 2 and translator system prompt

## Steps

### Step 1: Identify Terms for Glossary

Extract terminology requiring consistent translation:

- World-specific terms (places, factions, artifacts)
- Motif-carrying words
- Technical/domain terms
- Potentially ambiguous terms (false friends)

### Step 2: Coordinate with Style Lead

For each term, determine approved translation with Style Lead:

- Part of speech and grammatical notes
- Register level (formal/informal/archaic)
- Motif resonance in target language
- Cultural adaptation strategy

### Step 3: Create Glossary Entries

Document each term in bilingual_glossary:

- Source term → approved translation
- Part of speech
- Usage notes and context
- Do-not-translate list (proper names, motifs requiring preservation)
- Example sentences showing usage

### Step 4: Coordinate with Codex Curator

Ensure glossary aligns with codex terminology:

- Cross-reference consistency
- Spoiler-safe definitions
- Player-facing vs internal term distinctions

### Step 5: Lock and Maintain

Mark glossary as stable; batch-fix any inconsistencies:

- Update translation_pack glossary field
- Flag any required upstream changes for untranslatables
- Track glossary gaps and additions

## Output

Bilingual glossary with approved translations, usage notes, and Style Lead sign-off.

## Quality Criteria

- All key terms have locked translations
- No false friends or ambiguities
- Style Lead approval on motif-carrying terms
- Codex Curator confirms consistency
- Usage examples provided for context-dependent terms
