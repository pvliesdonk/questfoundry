---
procedure_id: alt_text_creation
description: Create succinct, descriptive, spoiler-safe alt text for visual accessibility
version: 1.0.0
references_expertises:
  - art_director_planning
references_schemas:
  - art_render.schema.json
references_roles:
  - illustrator
  - art_director
tags:
  - accessibility
  - art
  - presentation
---

# Alt Text Creation Procedure

## Overview

Write effective alternative text for images that conveys visual content to screen reader users while maintaining spoiler safety and narrative consistency.

## Source

Extracted from v1 `spec/05-prompts/illustrator/system_prompt.md` and `spec/05-prompts/loops/art_touch_up.playbook.md`

## Steps

### Step 1: Identify Visual Purpose

Understand why the image exists:

- Narrative purpose (clarity, mood, foreshadowing)
- Key visual elements supporting story
- Emotional or atmospheric intent

### Step 2: Describe What's Visible

Write objective description of image content:

- Subject and focal points
- Composition and framing
- Key visual details
- Setting or environment

### Step 3: Keep It Succinct

Aim for 1-2 sentences:

- Describe what matters for narrative understanding
- Omit excessive detail
- Focus on story-relevant elements
- ~40-150 characters ideal, max ~250

### Step 4: Ensure Spoiler Safety

Maintain presentation quality bar:

- No twist reveals or future plot points
- Avoid internal terminology or mechanics
- Keep diegetic and player-appropriate
- Coordinate with caption for consistency

### Step 5: Make It Descriptive

Use concrete, sensory language:

- **Bad**: "An interesting scene"
- **Good**: "A cloaked figure stands in a fog-shrouded alley, streetlamp casting long shadows"

### Step 6: Avoid Technique Talk

Keep alt text player-facing:

- No mention of rendering methods
- No model names or generation parameters
- No artistic technique jargon (unless diegetically relevant)

### Step 7: Coordinate with Caption

Ensure alt text and caption work together:

- Alt text describes visual content
- Caption may provide context or narrative framing
- Avoid redundancy while maintaining accessibility

### Step 8: Test for Accessibility

Verify alt text serves screen reader users:

- Read aloud to check natural flow
- Confirm conveys necessary visual information
- Ensure doesn't spoil or confuse
- Check against PN diegetic standards

## Output

Succinct, descriptive, spoiler-safe alt text for each image in art_render.

## Quality Criteria

- 1-2 sentences describing visual content
- Story-relevant details included
- Spoiler-safe and player-appropriate
- Natural language (reads well aloud)
- Coordinates with caption
- No technique or internal terminology
- Serves accessibility needs
