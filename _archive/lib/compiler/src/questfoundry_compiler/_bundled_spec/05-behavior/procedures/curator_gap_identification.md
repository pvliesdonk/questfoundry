---
procedure_id: curator_gap_identification
name: Curator Gap Identification
description: Identify missing anchors, ambiguous terms; propose hooks for clarification or new entries
roles: [codex_curator]
references_schemas:
  - codex_entry.schema.json
  - hook.schema.json
references_expertises:
  - codex_curator_publication
quality_bars: [integrity, accessibility]
---

# Curator Gap Identification

## Purpose

Proactively identify missing codex anchors, ambiguous terminology, and clarity gaps—then propose hooks for clarification or new entries without guessing or inventing canon.

## Core Principles

- **Reader Advocacy**: Identify what players need to act with confidence
- **Gap Detection**: Find missing definitions, unclear references, broken crosslinks
- **Hook Creation**: Request clarification from appropriate roles, don't guess
- **No Invention**: Never fill gaps with curator assumptions or made-up canon
- **Preventive**: Catch issues before they reach players

## Types of Gaps

1. **Missing Anchors**: Terms used in manuscript/choices lacking codex entries
2. **Ambiguous Terms**: Words with unclear in-world meaning
3. **Broken Crosslinks**: "See also" references to non-existent entries
4. **Orphaned Entries**: Codex entries unreferenced and unreachable
5. **Terminology Conflicts**: Same concept called different things
6. **Context Gaps**: Entry exists but lacks necessary context
7. **Structural Gaps**: Topology introduces concepts needing explanation

## Steps

1. **Scan Manuscript**: Look for undefined terms
   - Terms used in choices or gates
   - World-specific vocabulary
   - Technical/specialized language
   - Character titles or roles
   - Place names without context

2. **Review PN Feedback**: Check dry-run reports
   - Terms PN struggled to phrase diegetically
   - Concepts lacking clear in-world explanation
   - Gate phrasing friction due to missing context

3. **Audit Crosslinks**: Check reference integrity
   - "See also" links to missing entries
   - Related concepts not cross-referenced
   - One-way links that should be bidirectional

4. **Check Topology Notes**: Identify structural needs
   - Plotwright briefs mentioning new concepts
   - Gateway conditions requiring world knowledge
   - Loop-with-difference justifications needing explanation

5. **Coordinate with Scene Smith**: Surface terminology issues
   - Ambiguous word choices in manuscript
   - Terms needing diegetic grounding
   - Concepts appearing without introduction

6. **Coordinate with Translator**: Find portability gaps
   - Terms with no clear translation
   - Cultural concepts needing explanation
   - Idioms requiring codex support

7. **Create Hooks**: Propose solutions for gaps
   - **To Lore Weaver**: Request player-safe summaries for canon-dependent entries
   - **To Style Lead**: Request terminology decisions for ambiguous terms
   - **To Plotwright**: Request structural anchors for topology-introduced concepts
   - **To Scene Smith**: Suggest micro-context for ambiguous usage
   - Hooks specify gap type and suggested approach

8. **Prioritize Gaps**: Triage urgency
   - Critical: Blocks player comprehension (choice ambiguity, gate confusion)
   - High: Reduces confidence (unclear world rules, missing context)
   - Medium: Exploratory support (crosslink enrichment, optional depth)
   - Low: Nice-to-have (minor variants, edge cases)

9. **Track Gap Status**: Monitor resolution
   - Hooks accepted → entry planned
   - Hooks deferred → note wake conditions
   - Gaps filled → verify quality
   - Recurring gaps → pattern suggests process issue

## Outputs

- **Gap Report**: Identified clarity issues with:
  - Gap type (missing anchor, ambiguous term, etc.)
  - Location (manuscript section, codex entry, PN phrasing)
  - Player impact (critical, high, medium, low)
  - Suggested approach
- **Hooks**: Requests to appropriate roles for gap resolution
  - To Lore: Canon-dependent entries
  - To Style: Terminology decisions
  - To Plotwright: Structural anchors
  - To Scene: Micro-context additions
- **Priority Queue**: Ordered list of gaps by urgency
- **Pattern Notes**: Recurring gap types suggesting systemic issues

## Quality Checks

- Gaps identified before reaching players
- Each gap has clear impact assessment
- Hooks target appropriate role (don't ask Scene for canon)
- No curator gap-filling (create hooks, don't guess)
- Critical gaps escalated promptly
- Crosslink integrity maintained
- Terminology conflicts surfaced and routed to Style Lead
- PN friction patterns trigger hooks
- Gap tracking shows resolution progress
- Recurring patterns reported to Showrunner for process review
