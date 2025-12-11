---
procedure_id: research_memo_creation
description: Create structured research memo with findings, posture, and creative implications
version: 1.0.0
references_expertises:
  - researcher_fact_checking
references_schemas:
  - research_memo.schema.json
  - uncertainty_assessment.schema.json
references_roles:
  - researcher
tags:
  - research
  - documentation
  - handoff
---

# Research Memo Creation Procedure

## Overview

Produce concise, actionable research memo documenting findings, confidence level, player-safe phrasing, and creative implications for requesting roles.

## Source

Extracted from v1 `spec/05-prompts/researcher/system_prompt.md` "Operating Model" section

## Steps

### Step 1: Frame the Question

Define research scope player-safe:

- Question in surface language (no internal mechanics)
- Context: where it appears, why it matters
- Stakeholders: which roles need the answer
- Blocking vs nice-to-have priority

### Step 2: Gather Sources

Collect 2-5 relevant sources:

- Prioritize quality and relevance
- Note contradictions or gaps
- Assess currency and domain expertise

### Step 3: Assess Posture

Apply uncertainty posture taxonomy:

- Classify as corroborated / plausible / disputed / uncorroborated
- Justify posture with evidence summary
- Cite source relevance plainly

### Step 4: Write Short Answer

Provide concise answer to the question:

- 1-3 sentences
- Match posture (don't overclaim)
- Flag uncertainties or alternatives

### Step 5: Craft Neutral Phrasing

Suggest player-safe surface language:

- 2-3 example phrasings for PN/codex use
- In-world terminology (no internal mechanics)
- Spoiler-free (keep canon details in Hot)
- Natural and diegetic

### Step 6: List Creative Implications

Document impact by role:

- **For Lore Weaver**: Canon decisions this enables/constrains
- **For Scene Smith**: Prose opportunities or limitations
- **For Plotwright**: Story structure implications
- **For other roles**: Relevant creative impacts

### Step 7: Note Risks and Mitigations

Identify potential issues:

- Spoiler risks if used player-facing
- Contradictions with existing canon
- Cultural sensitivity concerns
- Accessibility considerations

### Step 8: Package Research Memo

Assemble complete research_memo artifact:

- Question framing
- Short answer with posture
- Source summaries with relevance
- Neutral phrasing examples
- Creative implications by role
- Risks and mitigations
- Revisit criteria (when to research again)

### Step 9: Emit Checkpoint

Send tu.checkpoint with research_memo delivery:

- Notify requesting role
- Mark ready for gate and handoff
- Include any proposed hooks if scope grew

## Output

Complete research_memo (Hot) with findings, posture, player-safe phrasing, and actionable implications.

## Quality Criteria

- Question framed clearly and player-safe
- Posture accurately reflects evidence
- Short answer is concise and actionable
- Neutral phrasing is diegetic and spoiler-free
- Creative implications specific to roles
- Risks identified with mitigations
- Memo validates against schema
