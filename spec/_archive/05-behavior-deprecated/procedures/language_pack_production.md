---
procedure_id: language_pack_production
description: Assemble complete language pack with localized surfaces, glossary, and coverage metrics
version: 1.0.0
references_expertises:
  - translator_terminology
references_schemas:
  - translation_pack.schema.json
  - coverage_report.schema.json
  - bilingual_glossary.schema.json
references_roles:
  - translator
tags:
  - translation
  - packaging
  - deliverables
---

# Language Pack Production Procedure

## Overview

Package complete translation deliverable including glossary, register map, localized surfaces, coverage metrics, and open issues.

## Source

Extracted from v1 `spec/05-prompts/loops/translation_pass.playbook.md` Step 8: "Package"

## Steps

### Step 1: Assemble Core Components

Gather all translation artifacts:

- Bilingual glossary (term → translation with usage notes)
- Register map (pronoun system, honorifics, tone equivalents, swear policy)
- Motif equivalence table (how house motifs render in target language)
- Idiom strategy (literal vs functional equivalents)

### Step 2: Package Localized Surfaces

Include all translated player-facing content:

- Manuscript sections and choice labels
- Codex titles and summaries
- Captions and alt text
- UI labels and link text

### Step 3: Compute Coverage Metrics

Calculate and document translation completeness:

- Coverage percentage by section count
- Coverage percentage by codex entries
- Scope completeness (full book, acts, subset)
- Mark partial outputs as `incomplete` with coverage flags

### Step 4: Document Open Issues

List remaining work and blockers:

- Untranslatables requiring upstream rewrite
- Glossary gaps requiring decision
- Cultural cautions or adaptation notes
- Deferred sections and reasons

### Step 5: Add Traceability

Include provenance and version metadata:

- TU-ID for this translation pass
- Source snapshot ID
- Target language code
- Translation date and translator role

### Step 6: Package Language Pack

Assemble final translation_pack artifact with all components and validate against schema.

## Output

Complete language_pack ready for Gatekeeper pre-gate and merge to Cold, with coverage flags and traceability.

## Quality Criteria

- All required components present
- Coverage metrics accurate
- Open issues clearly documented
- Traceability complete (TU-ID, snapshot ID)
- Schema validation passes
