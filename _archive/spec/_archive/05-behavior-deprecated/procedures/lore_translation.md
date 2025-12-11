---
procedure_id: lore_translation
name: Lore Translation
description: Translate Lore Weaver's player-safe summaries into entries; never add hidden causality or invent canon
roles: [codex_curator]
references_schemas:
  - codex_entry.schema.json
  - canon_summary.schema.json
references_expertises:
  - codex_curator_publication
  - lore_weaver_summarization
quality_bars: [presentation, integrity]
---

# Lore Translation

## Purpose

Transform Lore Weaver's player-safe summaries into structured, accessible codex entries without adding hidden causality, inventing canon, or revealing spoilers.

## Core Principles

- **Source Fidelity**: Only publish what Lore Weaver provided in player-safe summaries
- **No Canon Invention**: Never guess or fill gaps—request summaries if needed
- **Spoiler Preservation**: Maintain Lore's spoiler boundaries exactly
- **Structural Enhancement**: Add clarity and organization without adding content
- **Verification**: Confirm summaries are actually player-safe before publishing

## Steps

1. **Receive Summary from Lore Weaver**: Get player-safe abstracts
   - Brief, non-spoiling summaries
   - Outcomes only, not mechanisms or causes
   - Neutral phrasing without foreshadowing
   - Clear boundary: what's safe to publish

2. **Verify Player Safety**: Confirm summary is safe
   - No twist causality or hidden allegiances
   - No gate logic or internal state
   - No secret motivations or future reveals
   - If questionable, coordinate with Lore to confirm boundaries

3. **Extract Core Concepts**: Identify what needs entries
   - Terms requiring definition
   - Places/characters needing context
   - Cultural/historical background
   - In-world systems or rules

4. **Structure as Entry**: Apply codex format
   - Overview: Brief definition from summary
   - Usage: How it functions (from summary content)
   - Context: Background provided by Lore
   - See also: Crosslinks to related entries
   - Notes: Clarifications (if needed)
   - Lineage: Reference to Lore's summary TU

5. **Add NO New Canon**: Strict constraint
   - Don't extrapolate beyond summary
   - Don't fill gaps with guesses
   - Don't add "likely" interpretations
   - If information missing, create hook for Lore

6. **Coordinate Terminology**: Ensure consistency
   - Use glossary terms
   - Align with Style Lead phrasing
   - Coordinate with Translator for portability

7. **Cross-Reference Manuscript**: Check for contradiction
   - Ensure entry doesn't conflict with published narrative
   - If gap between Lore summary and manuscript, flag for resolution
   - Don't resolve contradictions yourself—escalate

## Outputs

- **Codex Entries**: Structured entries derived from Lore summaries
- **Gap Hooks**: Requests for additional Lore summaries when information insufficient
- **Validation Notes**: Confirmation of player-safety boundaries
- **Lineage Links**: Traceability to source Lore summaries

## Quality Checks

- Entry content derived from Lore Weaver's player-safe summary only
- No canon invention or gap-filling by Curator
- No hidden causality or twist mechanisms added
- Spoiler boundaries from Lore preserved exactly
- Terminology consistent with glossary
- No contradiction with published manuscript
- Entry structure applied without altering content
- Lineage documents source summary
- Gaps flagged via hooks, not filled with guesses
- Lore Weaver confirms summary was player-safe if uncertain
