---
procedure_id: uncertainty_posture_assessment
description: Assess and communicate confidence level for research findings using posture taxonomy
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
  - uncertainty
  - evidence
---

# Uncertainty Posture Assessment Procedure

## Overview

Apply structured posture taxonomy to research findings, communicating confidence level and evidence quality for informed creative decisions.

## Source

Extracted from v1 `spec/05-prompts/researcher/system_prompt.md` "Evidence & Posture" section

## Steps

### Step 1: Gather Evidence

Collect sources for the research question:

- 2-5 relevant sources
- Assess source quality and relevance
- Note agreement or contradictions between sources

### Step 2: Apply Posture Taxonomy

Classify confidence level using standard taxonomy:

- **corroborated**: Multiple reliable sources agree
- **plausible**: Reasonable but not definitively confirmed
- **disputed**: Sources conflict or present contradictory evidence
- **uncorroborated:low**: Single source, low confidence
- **uncorroborated:medium**: Single source, moderate confidence
- **uncorroborated:high**: Single source, strong indicators

### Step 3: Justify Posture Assignment

Document reasoning for posture choice:

- Number and quality of sources
- Level of agreement between sources
- Recency and relevance of evidence
- Domain expertise of sources

### Step 4: Cite Source Relevance

Summarize each source contribution:

- What the source says
- Why it's relevant to the question
- How it supports or contradicts other sources
- Any limitations or caveats

### Step 5: Avoid Overclaiming

Respect posture boundaries:

- Don't present "plausible" as "corroborated"
- Don't hide contradictions or uncertainties
- Make limitations explicit
- Acknowledge gaps in evidence

### Step 6: Communicate Implications

Explain what posture means for creative work:

- **corroborated**: Safe to treat as established fact
- **plausible**: Can use but note as interpretation
- **disputed**: Creative choice required, document alternatives
- **uncorroborated**: Speculative, requires explicit framing

### Step 7: Include in Research Memo

Document posture in research_memo:

- Posture classification
- Evidence summary
- Justification
- Creative implications by role

## Output

Uncertainty assessment with clear posture classification and evidence justification in research_memo.

## Quality Criteria

- Posture accurately reflects evidence strength
- Source relevance clearly explained
- No overclaiming beyond evidence
- Contradictions acknowledged
- Creative implications clear for each posture level
- Justification traceable to sources
