---
procedure_id: hook_triaging
description: Annotate and prioritize hooks with triage tags, uncertainty levels, and next-step assignments
version: 1.0.0
references_expertises:
  - lore_weaver_expertise
references_schemas:
  - hook.schema.json
  - harvest_sheet.schema.json
references_roles:
  - showrunner
  - lore_weaver
  - plotwright
  - scene_smith
tags:
  - hooks
  - triage
  - prioritization
---

# Hook Triaging Procedure

## Overview

Annotate proposed hooks with triage tags, uncertainty assessments, dependencies, and decide which to accept, defer, or reject for advancement.

## Source

Extracted from v1 `spec/05-prompts/loops/hook_harvest.playbook.md` Steps 4-5: "Annotate" and "Decide"

## Steps

### Step 1: Review Hook Clusters

Examine clustered hooks by theme and type:

- Narrative hooks (story events, character arcs)
- Scene hooks (specific moments, dialogue beats)
- Factual hooks (research questions, worldbuilding facts)
- Taxonomy hooks (codex entries, glossary terms)

### Step 2: Assign Triage Tags

For each hook, add triage classification:

- `quick-win`: Can advance immediately with minimal research/work
- `needs-research`: Requires Researcher validation or fact-checking
- `structure-impact`: Affects story topology, requires Plotwright review
- `style-impact`: Affects voice/tone, requires Style Lead coordination
- `deferred`: Not ready yet, needs future consideration
- `reject`: Duplicate, out-of-scope, or conflicts with canon

### Step 3: Assess Uncertainty (Factual Hooks)

For factual hooks, add uncertainty posture:

- `corroborated`: Multiple reliable sources agree
- `plausible`: Reasonable but not confirmed
- `disputed`: Sources conflict
- `uncorroborated:low/med/high`: Single source with confidence level
- Include any citations or source notes

### Step 4: Identify Dependencies

Note what must happen before hook can advance:

- Upstream references: other hooks or canon that must exist first
- Dormant role wake: roles that must activate (e.g., Researcher)
- Gatecheck blockers: quality bars likely to fail
- Coordination needs: roles that must consult

### Step 5: Decide Status

Mark each hook with final decision:

- **Accepted**: Advance to next loop (specify which: Lore Deepening, Story Spark, etc.)
- **Deferred**: Not ready; note wake condition and revisit criteria
- **Rejected**: Provide 1-line reason; link to surviving duplicate if applicable

### Step 6: Assign Next Steps

For accepted hooks:

- Specify next loop (e.g., Lore Deepening, Story Spark)
- Assign responsible (R) role
- Assign accountable (A) role (usually Showrunner)
- Set due window or priority

### Step 7: Package Decisions

Document triage outcomes in harvest_sheet:

- Accepted hooks with next loop + owner + timeline
- Deferred hooks with reason + wake condition
- Rejected hooks with reason + duplicate links

## Output

Triaged hooks with status, tags, uncertainty levels, dependencies, and next-step assignments ready for handoff.

## Quality Criteria

- All hooks have triage tag assigned
- Factual hooks have uncertainty assessment
- Dependencies identified and documented
- Clear accept/defer/reject decision with justification
- Accepted hooks have assigned next loop and owner
- Deferred hooks have revisit criteria
- Rejected hooks have reason and duplicate links
