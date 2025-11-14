---
snippet_id: research_posture
name: Research Posture
description: When Researcher dormant - mark claims uncorroborated:<risk> and keep surfaces neutral; when active - coordinate validation
applies_to_roles: [lore_weaver, scene_smith, researcher, showrunner]
quality_bars: [integrity, presentation]
---

# Research Posture

## Core Principle

Handle factual claims differently based on Researcher activation state. Always maintain Integrity bar regardless of posture.

## When Researcher Dormant

### Mark Claims with Risk Level

- `uncorroborated:low` — Minor detail, unlikely to break immersion if wrong
- `uncorroborated:med` — Notable claim, could affect plausibility
- `uncorroborated:high` — Critical to plot, must validate before release

### Keep Surfaces Neutral

Instead of specific claims, use general phrasing:

**Specific (needs validation):**
❌ "Six months in low-G causes severe bone density loss"

**Neutral (safe without Researcher):**
✓ "Long-term low-G takes a toll on the bones"

### Schedule Research TU

If risk ≥ medium AND release approaching:

- Showrunner schedules Research TU
- Activate Researcher for validation
- Resolve before shipping to players

### Document Uncertainty

In Hot notes, mark:

```yaml
claim: "Low-G causes bone density loss over 6-12 months"
research_posture: dormant
risk_level: medium
surface_phrasing: "Long-term low-G affects bone density"
research_needed_before: release
```

## When Researcher Active

### Request Validation

Roles submit research requests:

```yaml
question: "Can low-gravity environments cause long-term bone density loss?"
context: "Station workers been in low-G for years; want medical gate"
requested_by: lore_weaver
risk_level: medium
```

### Receive Research Memo

Researcher provides:

- Validation level (corroborated / plausible / disputed / uncorroborated)
- Citations (2-5 sources)
- Caveats
- Creative implications (enables/forbids)
- Suggested phrasing (if needed)

### Incorporate Findings

**If Corroborated:**

- Use claim confidently in canon/prose
- Cite as justification for gates/plot points
- No caveats needed on surfaces

**If Plausible:**

- Use with neutral phrasing
- Avoid overly specific claims
- Safe to use as background logic

**If Disputed:**

- Use very neutral phrasing
- Avoid taking sides
- Consider alternative plot mechanisms

**If Uncorroborated:**

- Reassess risk level
- If low: keep neutral phrasing
- If med/high: consider alternative approach or activate deeper research

## Role-Specific Applications

**Lore Weaver:**

- When Researcher dormant: mark claims `uncorroborated:<risk>`, keep summaries neutral
- When Researcher active: coordinate fact validation, cite sources in Hot notes
- Always: maintain plausibility, avoid breaking immersion with errors

**Scene Smith:**

- When dormant: use neutral phrasing for uncertain claims
- When active: incorporate research memo findings into prose
- Always: prioritize narrative over specificity

**Researcher:**

- When dormant: not consulted
- When active: validate claims, provide citations, suggest safe phrasing
- Always: maintain Integrity bar

**Showrunner:**

- Track uncorroborated claims and risk levels
- Schedule Research TU when risk ≥ med before release
- Activate Researcher with clear research questions
- Ensure findings incorporated before shipping

## Risk Assessment

### Low Risk

- Background detail
- Flavor text
- Generic claims
- Easy to fix if wrong

**Example:**
"The relay hums with electrical current"
→ If technically inaccurate, not immersion-breaking

### Medium Risk

- Plot-relevant detail
- Gate justification
- Character expertise
- Noticeable if wrong

**Example:**
"Airlocks require EVA certification for safety"
→ If wrong, players might question plausibility

### High Risk

- Critical plot mechanism
- Major gate dependency
- Expert character knowledge
- Immersion-breaking if wrong

**Example:**
"Decompression causes nitrogen narcosis symptoms within 30 seconds"
→ If factually wrong, breaks medical expert character credibility

## Validation Workflow

1. **Claim Identified** (Lore or Scene drafts content with factual claim)
2. **Risk Assessment** (Author evaluates: low/med/high)
3. **Check Researcher Status:**
   - If dormant: Use neutral phrasing, mark for later
   - If active: Submit research request
4. **Researcher Validates** (if active)
5. **Author Incorporates** (adjust phrasing based on validation level)
6. **Showrunner Reviews** (before release, ensure risk ≥ med claims validated)

## Examples

### Dormant Posture

```markdown
Hot note: "Claim: Coriolis effect affects projectile trajectory in rotating station"
Risk: medium
Research posture: dormant
Surface phrasing: "Firing a weapon in a spinning station has complications"
```

### Active Posture

```markdown
Research memo received:
Validation: Corroborated
Citations: [NASA, ESA rotating hab studies]
Finding: "Yes, Coriolis force affects trajectories measurably in large rotating stations"

Updated surface phrasing: "The station's spin curves projectile paths—aim carefully"
```

## Quality Bar Connection

**Integrity Bar:**

- Maintained regardless of posture
- Dormant: neutral phrasing avoids false claims
- Active: validated claims maintain plausibility
- Never sacrifice Integrity for specificity
