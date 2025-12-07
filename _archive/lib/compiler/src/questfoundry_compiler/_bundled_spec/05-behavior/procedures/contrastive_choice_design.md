---
procedure_id: contrastive_choice_design
name: Contrastive Choice Design
description: End sections with choices that communicate intent; avoid near-synonyms; add micro-context where needed
roles: [scene_smith, plotwright, style_lead]
references_schemas:
  - section.schema.json
  - choice.schema.json
references_expertises:
  - scene_smith_prose_craft
  - plotwright_topology
  - style_lead_voice
quality_bars: [nonlinearity, style, accessibility]
---

# Contrastive Choice Design

## Purpose

Craft choice labels that clearly communicate distinct player intents, avoiding near-synonyms and ambiguity. Ensure choices are contrastive (meaningfully different), concise, and player-safe.

## Inputs

- Section context and beats
- Choice intents from Plotwright brief
- Style guardrails for choice phrasing
- PN phrasing patterns (diegetic language)

## Core Principles

- **Contrastive:** Choices must differ in verb OR object (not just synonyms)
- **Concise:** Keep labels short (2-8 words typically)
- **Clear Intent:** Player should understand what each choice represents
- **No Spoilers:** Avoid revealing outcomes or hidden information
- **Diegetic:** Use in-world language, no meta phrasing

## Steps

### 1. Identify Choice Intents

- Extract intended player actions from Plotwright brief
- Note expected divergence (different destinations OR different opening beats)
- Check for anti-funneling: choices must NOT be functionally equivalent

### 2. Draft Choice Labels

- Use distinct verbs for different actions ("Slip through" vs "Face the foreman")
- Use distinct objects for same action type ("Take the maintenance hex-key" vs "Take the foreman's token")
- Avoid near-synonyms ("Go" vs "Proceed" - FORBIDDEN)

### 3. Add Micro-Context If Needed

- If choice labels alone are ambiguous, add 1-2 lines of clarifying prose before choices
- Micro-context explains stakes or affordances WITHOUT spoiling outcomes
- Example: "The maintenance hex-key opens crew passages. The foreman's token grants access to secure zones."

### 4. Check Contrast

- Read choices aloud - do they sound meaningfully different?
- Would a player understand the distinction?
- Do they align with diegetic gate phrasing patterns?

### 5. Validate Diegetic Language

- No meta phrasing: "Attempt X" ❌ → "Do X" ✓
- No mechanics exposure: "Roll for stealth" ❌ → "Slip through quietly" ✓
- In-world objects/actions only

## Anti-Patterns to Avoid

### Near-Synonyms (FORBIDDEN)

- "Go / Proceed"
- "Enter / Go in"
- "Accept / Agree"
- "Leave / Depart"

### Meta Phrasing (FORBIDDEN)

- "Attempt to X"
- "Try to Y"
- "Choose to Z"
- "Option A"

### Spoiler Leaks (FORBIDDEN)

- "Betray the crew" (reveals twist)
- "Use the hidden passage" (reveals secret)
- "Trust the saboteur" (reveals allegiance)

## Valid Contrast Examples

### Different Verbs

- "Slip through maintenance" vs "Face the foreman"
- "Negotiate with the captain" vs "Steal the keycard"
- "Study the map" vs "Ask for directions"

### Different Objects

- "Take the hex-key" vs "Take the token"
- "Board the shuttle" vs "Board the cargo hauler"
- "Contact engineering" vs "Contact security"

### Different Outcomes (Signaled, Not Spoiled)

- "Move quickly" vs "Move carefully" (pace difference clear)
- "Speak openly" vs "Speak guardedly" (tone difference clear)
- "Continue alone" vs "Wait for backup" (resources difference clear)

## Outputs

- Contrastive choice labels embedded in section draft
- Micro-context prose (if needed) preceding choices
- Hooks for clarity issues requiring structural fixes

## Quality Bars Pressed

- **Nonlinearity:** Choices meaningfully differentiate
- **Style:** Labels match voice/register
- **Accessibility:** Clear, readable choice language

## Handoffs

- **To Style Lead:** Receive phrasing patterns for common choice types
- **To Plotwright:** Flag when choice ambiguity requires structural fix (not just rewording)
- **To Gatekeeper:** Submit for Nonlinearity and Presentation checks

## Common Issues

- **Ambiguity:** Add micro-context, don't just reword
- **Near-Synonyms:** Change verb OR object to create contrast
- **Over-Clarification:** Trust player intelligence; don't over-explain
- **Spoiler Risk:** Move revealing details to outcomes, not choice labels
