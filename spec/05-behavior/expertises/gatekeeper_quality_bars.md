# Gatekeeper Quality Bars Expertise

## Mission

Enforce Quality Bars before Hot→Cold merges; provide actionable remediation.

## The 8 Quality Bars

All Cold merges must pass these criteria (from Layer 0):

1. **Integrity** - No dead references
2. **Reachability** - Critical beats accessible
3. **Nonlinearity** - Meaningful branching
4. **Gateways** - Clear diegetic conditions
5. **Style** - Consistent voice/register
6. **Determinism** - Reproducible assets
7. **Presentation** - Spoiler-free surfaces
8. **Accessibility** - Navigation and alt text

## Bar 1: Integrity

**Validates:** Referential integrity and internal consistency

**Checks:**

- All choice `target_id` references resolve to existing sections
- All gateway conditions reference defined state variables
- All canon references point to existing canon entries
- All codex references resolve to published entries
- No orphaned artifacts or dangling pointers
- Timeline anchors are consistent
- Entity state transitions are coherent

**Common Violations:**

- Deleted sections still referenced in choices
- Gateway checks variables that don't exist
- Scene callbacks to non-existent canon

**Remediation:** List all broken references with file paths and line numbers

## Bar 2: Reachability

**Validates:** All critical story beats are accessible through valid player paths

**Checks:**

- Start section reachable from initialization
- All mandatory beats have at least one path from start
- No story-critical content behind impossible gateways
- Loop returns don't create dead-end cycles
- Hub diversity supports multiple valid paths

**Common Violations:**

- Unreachable sections due to impossible gateway combinations
- Required story beats only accessible through single fragile path
- Dead-end loops with no exit conditions

**Remediation:** Path analysis showing which beats are unreachable and suggested gateway adjustments

## Bar 3: Nonlinearity

**Validates:** Player choices have meaningful consequences

**Checks:**

- Multiple viable paths exist through story
- Choices are contrastive (not cosmetic)
- Loop-with-difference: repeat visits show meaningful changes
- State effects create narrative branching
- Hub returns reflect prior player decisions

**Common Violations:**

- All choices converge immediately (illusory branching)
- Loop returns are identical regardless of state
- Choices with no narrative consequence

**Remediation:** Identify cosmetic choices and suggest state-based differentiation

## Bar 4: Gateways

**Validates:** Choice availability uses clear diegetic conditions

**Checks:**

- Gateway reasons are world-based, not meta
- Conditions are comprehensible to players through story
- PN-safe phrasing (no codewords or state leaks)
- Consistency: same condition should gate similar choices
- No player-hostile hidden gates

**Common Violations:**

- Meta conditions ("if flag X is set")
- Incomprehensible requirements
- Arbitrary restrictions without story justification

**Remediation:** Suggest diegetic phrasings that align with canon

## Bar 5: Style

**Validates:** Consistent voice, register, and prose quality

**Checks:**

- Register matches style guide
- Voice consistent across sections
- Diction appropriate for setting
- Motif usage aligned with Style Lead direction
- No anachronisms or register breaks
- Paragraph rhythm maintained

**Common Violations:**

- Register shifts between sections
- Inconsistent character voice
- Modern idioms in historical settings
- Purple prose mixed with minimalism

**Remediation:** Highlight inconsistencies with style guide references

## Bar 6: Determinism

**Validates:** Reproducible asset generation

**Checks:**

- All images have generation parameters logged
- All audio has production metadata
- Asset manifests include checksums
- Generation prompts are version-controlled
- Provider/model versions recorded
- Seed values documented for regeneration

**Common Violations:**

- Assets without generation parameters
- Missing checksums or file paths
- Undocumented manual edits to generated assets
- Provider version not recorded

**Remediation:** List assets missing determinism metadata

## Bar 7: Presentation

**Validates:** Spoiler hygiene and player safety

**Checks:**

- No spoilers in player-facing surfaces
- Canon Pack remains in Hot only
- Codex entries are player-safe
- Gateway phrasings don't leak mechanics
- Choice text doesn't preview outcomes
- Section titles avoid spoilers

**Common Violations:**

- Canon details in codex entries
- Choice text that reveals consequences
- Section titles that spoil twists
- Gateway text exposing state variables

**Remediation:** Flag specific spoiler leaks with suggested neutral phrasing

## Bar 8: Accessibility

**Validates:** Navigation and inclusive design

**Checks:**

- All images have alt text
- Navigation is clear and consistent
- Choice presentation is accessible
- No reliance on color alone for meaning
- Text contrast meets standards
- Screen reader compatibility

**Common Violations:**

- Missing alt text for images
- Unclear navigation between sections
- Color-only state indicators
- Tiny or low-contrast text

**Remediation:** List accessibility issues with WCAG references

## Validation Audit Protocol

For each TU submitted for gatecheck:

1. **Enumerate artifacts:** List all artifacts in submission
2. **Schema validation:** Verify all artifacts pass JSON schema validation
3. **Bar-by-bar audit:** Check each of 8 bars systematically
4. **Collect violations:** Document specific failures with evidence
5. **Determine severity:** Critical (must fix) vs Minor (should fix)
6. **Issue decision:** `pass` or `fail` with remediation guidance

## Decision Framework

**PASS criteria:**

- All 8 bars pass
- No critical violations
- Minor issues documented but non-blocking
- All artifacts have valid schemas

**FAIL criteria:**

- Any bar has critical violations
- Schema validation failures
- Spoiler leaks in player surfaces
- Broken references or unreachable content

## Remediation Guidance

For each violation:

- **Location:** File path, line number, artifact ID
- **Bar violated:** Which of the 8 bars
- **Issue:** Specific problem description
- **Severity:** Critical (must fix) vs Minor (should fix)
- **Suggested fix:** Actionable remediation
- **Assigned role:** Who should fix it

## Enforcement

**Hard gates (no exceptions):**

- Schema validation failures
- Spoiler leaks in Cold
- Broken references (Integrity Bar)
- Unreachable critical beats (Reachability Bar)

**Escalate to Showrunner when:**

- Remediation requires cross-role coordination
- Multiple bars fail indicating systemic issues
- Same violations recur across TUs
- Human decision needed on trade-offs
