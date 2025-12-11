---
procedure_id: leitmotif_management
description: Select and manage recurring musical motifs for narrative signposting and emotional resonance while maintaining spoiler safety
version: 1.0.0
references_expertises:
  - audio_director_planning
  - style_lead_voice
references_schemas:
  - audio_plan.schema.json
  - style_addendum.schema.json
references_roles:
  - audio_director
  - style_lead
tags:
  - audio
  - motif
  - style
quality_bars:
  - style
  - presentation
---

# Leitmotif Management Procedure

## Overview

Choose and track recurring musical themes (leitmotifs) that support narrative structure, signal character/location/emotional states, and maintain stylistic coherence WITHOUT telegraphing spoilers.

## Source

Extracted from v1 `spec/05-prompts/loops/audio_pass.playbook.md` Steps 1-2: Audio Director selecting cues with motif ties

Merged with leitmotif_use_policy for spoiler safety governance

## Steps

### Step 1: Identify Motif Candidates

Determine recurring narrative elements warranting musical signposting:

- Key characters or factions
- Important locations or settings
- Emotional states or story beats
- Thematic concepts (e.g., hope, danger, mystery)

### Step 2: Select Cues with Motif Ties

For each audio cue, specify motif connections:

- Which house motif(s) the cue threads
- How the cue reinforces or varies the motif
- Relationship to other motif occurrences

### Step 3: Coordinate with Style Lead

Ensure motif language aligns with overall style:

- Motif resonance in musical form
- Consistency with visual and prose motif usage
- Cultural and tonal appropriateness

### Step 4: Document Motif Patterns

Maintain motif inventory in audio_plan:

- Motif name and narrative purpose
- Musical characteristics (instrumentation, tempo, key, mood)
- Variation strategy (when to repeat vs transform)
- Placement pattern (entry points, reprises)

### Step 5: Track Motif Usage

Monitor motif deployment across scenes:

- Avoid over-repetition (motif fatigue)
- Ensure strategic placement (narrative payoff)
- Coordinate variations for story progression

### Step 6: Establish Spoiler Boundaries

Document what leitmotifs must NOT signal:

- Hidden character allegiances or secret identities
- Future plot twists or outcomes
- Internal game state, flags, or codewords
- Gate logic or mechanical conditions
- Player-invisible narrative tracking

### Step 7: Update Style Guide

Feed successful motif patterns back to style documentation for consistency.

## Output

- Leitmotif inventory in audio_plan with motif ties documented for each cue
- Use policy documenting when and how to use specific leitmotifs
- Spoiler boundaries explicitly listing forbidden signaling patterns
- Style coordination notes confirming alignment with register/tone

## Quality Criteria

- Motifs serve clear narrative purpose
- Style Lead approval on motif resonance
- Balanced repetition (recognition without fatigue)
- Strategic placement supports story structure
- Motif patterns documented for reuse
- **NO leitmotif-as-spoiler patterns** (signaling hidden information)
- Themes consistently applied across all audio cues
- Diegetic compatibility maintained (supports in-world delivery)
