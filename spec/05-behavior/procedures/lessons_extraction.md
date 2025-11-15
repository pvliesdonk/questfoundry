---
procedure_id: lessons_extraction
description: Extract actionable lessons and patterns from completed work for process improvement
version: 1.0.0
references_expertises:
  - gatekeeper_quality_bars
references_schemas:
  - post_mortem_report.schema.json
references_roles:
  - showrunner
  - gatekeeper
  - all
tags:
  - retrospective
  - learning
  - process
---

# Lessons Extraction Procedure

## Overview

Systematically identify successes, failures, surprising discoveries, and process patterns from completed milestones to inform future improvements.

## Source

Extracted from v1 `spec/05-prompts/loops/post_mortem.playbook.md` Step 4: "Retrospective Session"

## Steps

### Step 1: Gather Retrospective Data

Collect qualitative and quantitative inputs:

- Gate pass / conditional pass / block rates per loop
- Most common quality bar failures
- Rework cycles per artifact type
- Time from TU open to Cold merge
- Hook acceptance / deferral / rejection patterns
- Gatecheck reports, playtest notes, incident logs

### Step 2: What Went Well

Celebrate successes and effective practices (3-5 items):

- Processes that worked smoothly
- Quality bar improvements
- Effective collaborations
- Successful innovations or experiments

### Step 3: What Went Poorly

Identify pain points, blockers, inefficiencies (3-5 items):

- Recurring quality bar failures
- Process bottlenecks or delays
- Communication breakdowns
- Tool or workflow frustrations

### Step 4: Surprising Discoveries

Note unexpected insights and emergent patterns (2-3 items):

- Unanticipated workflow synergies
- Emergent quality issues
- Unexpected dependencies
- Novel solutions or workarounds

### Step 5: Identify Improvement Opportunities

For each issue area, propose specific improvements:

- Process changes to try
- Tool or workflow enhancements
- Communication or coordination improvements
- Documentation or training needs

### Step 6: Use "5 Whys" for Root Causes

For significant issues, dig deeper:

- Ask "What process allowed this?" not "Who made the mistake?"
- Seek systemic root causes
- Focus on preventable patterns
- Propose process changes that would prevent recurrence

## Output

Structured lessons in post_mortem_report documenting what worked, what didn't, and why.

## Quality Criteria

- Blameless culture maintained (systems focus, not individuals)
- All participating roles contribute
- Root causes identified for significant issues
- Concrete improvement opportunities proposed
- Patterns documented for future reference
