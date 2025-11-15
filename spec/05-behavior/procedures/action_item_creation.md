---
procedure_id: action_item_creation
description: Create specific, owned, dated action items from retrospective improvements
version: 1.0.0
references_schemas:
  - action_items.schema.json
  - post_mortem_report.schema.json
references_roles:
  - showrunner
  - all
tags:
  - retrospective
  - action-items
  - accountability
---

# Action Item Creation Procedure

## Overview

Transform retrospective improvement proposals into specific, actionable, trackable items with clear owners, deadlines, and success criteria.

## Source

Extracted from v1 `spec/05-prompts/loops/post_mortem.playbook.md` Step 5: "Identify Action Items"

## Steps

### Step 1: Review Improvement Proposals

Examine all "Improvements to try" from retrospective:

- Process changes
- Tool or workflow enhancements
- Communication improvements
- Documentation or training needs

### Step 2: Create Action Item Structure

For each improvement, define complete action item:

- **Description**: Specific action to take (not vague goal)
- **Owner**: Role responsible for implementation (single owner)
- **Target**: Completion date or milestone
- **Success Criteria**: How we'll know it worked (measurable)
- **Priority**: High / Medium / Low

### Step 3: Ensure Specificity

Verify each action item is actionable:

- **Bad**: "Improve gate pass rate"
- **Good**: "Add Style Lead to all pre-gate sessions"
- **Bad**: "Better communication"
- **Good**: "Create template for TU handoff messages with required fields"

### Step 4: Assign Clear Ownership

Every action item must have single owner:

- Owner commits to implementation
- Owner may delegate but retains accountability
- Multiple contributors OK, but one owner

### Step 5: Set Realistic Targets

Assign completion dates:

- Near-term (within 1-2 weeks): High priority
- Medium-term (within milestone): Medium priority
- Long-term (next quarter): Low priority

### Step 6: Define Success Criteria

Make success measurable:

- Quantitative when possible (e.g., "Style bar pass rate >90%")
- Observable behaviors when qualitative (e.g., "All TUs include handoff template")
- Verifiable outcomes (e.g., "Documentation updated and reviewed")

### Step 7: Prioritize and Limit

Focus on high-impact actions:

- Limit to 3-5 high-priority actions per post-mortem
- Archive lower-priority items for future consideration
- Avoid action item overload

### Step 8: Document and Track

Add action items to post_mortem_report and create tracking:

- Assign action items to relevant upcoming TUs
- Review completion status in next post-mortem
- Update status: pending / in-progress / completed / deferred

## Output

Action items table in post_mortem_report with description, owner, target date, success criteria, and priority.

## Quality Criteria

- All action items are specific and actionable
- Every item has single clear owner
- Target dates are realistic
- Success criteria are measurable
- Priority reflects impact and urgency
- Limited to 3-5 high-priority items
