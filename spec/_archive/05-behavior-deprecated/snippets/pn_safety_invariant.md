---
snippet_id: pn_safety_invariant
name: PN Safety Invariant
description: CRITICAL - Block any content to PN that violates safety triple (hot_cold=cold AND player_safe=true AND spoilers=forbidden)
applies_to_roles: [gatekeeper, showrunner, book_binder]
quality_bars: [presentation, integrity]
criticality: CRITICAL
---

# PN Safety Invariant

## Core Rule (CRITICAL)

**NEVER route Hot content to Player-Narrator**

The PN Safety Invariant is a business-critical rule that protects player experience by ensuring Player-Narrator only receives spoiler-safe, player-facing content.

## Safety Triple

When `receiver.role = player_narrator`, ALL three conditions MUST be true:

1. `hot_cold = "cold"` — Content from Cold (stable, player-safe) not Hot (work-in-progress)
2. `player_safe = true` — Content approved for player visibility
3. `spoilers = "forbidden"` — No twists, codewords, or behind-the-scenes information

**AND** `snapshot` must be present (specific Cold snapshot ID)

## Violation Handling

**Gatekeeper:**

- Block any message to PN violating safety triple
- Report violation as `business_rule_violation`
- Rule ID: `PN_SAFETY_INVARIANT`
- Do NOT attempt heuristic fixes
- Escalate to Showrunner immediately

**Showrunner:**

- Enforce safety triple when receiver.role = PN
- Violation is CRITICAL ERROR
- Do not proceed with workflow until resolved
- Coordinate with Binder for proper snapshot sourcing

**Book Binder:**

- NEVER export from Hot
- NEVER mix Hot & Cold sources
- Single snapshot source for entire view
- Validate safety triple before delivering to PN

## Why This Matters

**Player Experience:**

- PN performance is player-facing
- Spoilers in PN output ruin narrative discovery
- Hot content may contain incomplete/contradictory information

**Production Safety:**

- Hot workspace contains spoilers, internals, technique
- PN has no context to filter unsafe content
- Violation breaks immersion irreparably

**Business Risk:**

- Spoiled players cannot "unsee" reveals
- Lost narrative value cannot be recovered
- Reputation damage from poor player experience

## Validation Points

**Pre-Gate (Gatekeeper):**

- Check all PN inputs for safety triple
- Block on violation before PN receives content

**View Export (Binder):**

- Verify snapshot source is Cold
- Validate all included content marked player_safe
- Ensure no Hot contamination

**TU Orchestration (Showrunner):**

- Enforce safety triple when routing to PN
- Double-check snapshot ID present
- Never wake PN for Hot-only content

## Common Violations

**Hot Content Leak:**

- Accidental inclusion of Hot files in view
- Mixed Hot/Cold sources in export
- Missing snapshot validation

**Spoiler Contamination:**

- Codewords visible in gate text
- Twist causality in summaries
- Internal labels in navigation

**Missing Snapshot:**

- PN invoked without snapshot ID
- Attempting to perform from working draft
- No stable Cold source identified

## Recovery

If violation detected:

1. STOP workflow immediately
2. Do not deliver to PN
3. Report to Showrunner with violation details
4. Identify source of contamination
5. Re-export from valid Cold snapshot
6. Re-validate safety triple
7. Resume workflow only after confirmation
