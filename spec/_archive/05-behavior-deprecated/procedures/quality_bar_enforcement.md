---
procedure_id: quality_bar_enforcement
name: Quality Bar Enforcement
description: Evaluate all 8 quality bars and deliver pass/conditional pass/block decisions with smallest viable fixes
roles: [gatekeeper]
references_schemas:
  - gatecheck_report.schema.json
references_expertises:
  - gatekeeper_quality_bars
quality_bars: [all]
---

# Quality Bar Enforcement

## Purpose

Systematically evaluate work against all 8 quality bars, delivering clear pass/conditional pass/block decisions with actionable remediation guidance.

## The 8 Quality Bars

### 1. Integrity

**Definition:** Referential consistency, timeline coherence, link resolution

**Checks:**

- All references resolve (canon, codex, schema)
- Timeline events don't contradict
- Crosslinks land on valid targets
- No orphan artifacts

### 2. Reachability

**Definition:** Critical beats reachable, no dead ends

**Checks:**

- All keystones accessible via at least one path
- No sections that block progress permanently
- Redundancy around single-point-of-failure bottlenecks

### 3. Nonlinearity

**Definition:** Hubs/loops intentional and meaningful

**Checks:**

- Hubs offer distinct experiences (not decorative)
- Loops provide return-with-difference (not empty)
- Choices have consequence (not cosmetic)

### 4. Gateways

**Definition:** Gateway conditions enforceable, diegetic

**Checks:**

- Conditions are in-world (not meta)
- PN can enforce without exposing internals
- Fair signposting (player can anticipate)
- Acquisition paths exist

### 5. Style

**Definition:** Voice/register/motif consistency

**Checks:**

- Voice matches Style Lead guidance
- Register consistent across surfaces
- Motifs used consistently
- Choice labels contrastive

### 6. Determinism

**Definition:** Reproducible when promised (seeds/logs off-surface)

**Checks:**

- When reproducibility promised, logs exist
- Logs are complete (seeds, models, settings)
- Logs are off-surface (not player-visible)

### 7. Presentation

**Definition:** Spoiler safety; no internals on player surfaces

**Checks:**

- No spoilers in codex/captions/choice labels
- No codewords on surfaces
- No technique details visible (seeds, tools)
- PN boundaries respected

### 8. Accessibility

**Definition:** Alt text, descriptive links, readable pacing

**Checks:**

- Alt text present for images
- Links descriptive ("See Salvage Permits", not "click here")
- Sentence structure readable
- Audio has captions/text equivalents

## Steps

### 1. Receive Submission

- Owner signals work ready for gatecheck
- Collect artifacts (canon, sections, codex, plans, etc.)
- Note prior pre-gate feedback

### 2. Evaluate Each Bar

For each of the 8 bars:

- Review against bar criteria
- Assign status: GREEN (pass), YELLOW (fixable), RED (blocks)
- Document specific findings

### 3. Determine Decision

- **PASS:** All bars green
- **CONDITIONAL PASS:** Yellow bars with fixes identified
- **BLOCK:** Red bars (critical failures)

### 4. Identify Smallest Viable Fixes

For each yellow/red bar:

- What's the minimal change to pass?
- Who owns the fix?
- Can it be fixed in current TU or requires escalation?

### 5. Document Report

Create gatecheck report with:

- Bar-by-bar status (green/yellow/red)
- Decision (pass/conditional/block)
- Specific findings for each bar
- Smallest viable fixes with owners

### 6. Deliver Decision

- Send report to Showrunner
- If conditional: send fixes to owners
- If block: escalate for resolution

## Decision Framework

### PASS (All Green)

- All 8 bars meet criteria
- Minor issues documented but non-blocking
- Ready for merge approval

### CONDITIONAL PASS (Yellows, No Reds)

- Yellow bars have fixes identified
- Owner can address within current TU
- No hard-gate violations
- Fixes are surgical and scoped

### BLOCK (Any Red)

- Critical failures on one or more bars
- Hard-gate violations (spoilers in Cold, broken refs)
- Fixes require cross-loop coordination or major rework

## Hard Gates (Automatic Block)

These ALWAYS result in BLOCK, no exceptions:

- Schema validation failures
- Spoiler leaks in Cold
- Broken references (Integrity Bar)
- Unreachable critical beats (Reachability Bar)
- PN Safety Triple violated

## Smallest Viable Fix Principles

- **Minimal:** Smallest change to pass
- **Surgical:** Localized fix, not systemic rework
- **Scoped:** Can complete in current TU
- **Owned:** Clear role assignment

## Example Findings

### Green (Pass)

```
Bar: Integrity
Status: GREEN
Findings: All references resolve. Timeline coherent. No orphans.
```

### Yellow (Conditional)

```
Bar: Presentation
Status: YELLOW
Findings:
  - Line 47: Choice label "Take the sabotage chip" reveals twist
  - Line 92: Caption mentions "seed 1234" (technique leak)
Smallest Viable Fixes:
  - Line 47: Rephrase as "Take the datachip" (Scene Smith)
  - Line 92: Remove seed reference from caption (Art Director)
```

### Red (Block)

```
Bar: Reachability
Status: RED
Findings:
  - Section "Engineering Access" is unreachable (no path from hub)
  - Keystone "Discover evidence" blocked by impossible gate
Remediation: Requires topology rework (coordinate with Plotwright)
```

## Outputs

- `gatecheck_report` - Complete report with bar statuses and decision
- Remediation guidance for yellow/red bars

## Quality Bars Pressed

- All (this procedure validates all bars)

## Handoffs

- **To Showrunner:** Deliver decision and coordinate next steps
- **To Owners:** Route remediation fixes for conditional pass
- **From All Roles:** Receive artifact submissions

## Common Issues

- **Inconsistent Bar Application:** Apply same standards across all loops
- **Vague Findings:** Specific line numbers and examples required
- **Missing Fixes:** Every yellow/red needs smallest viable fix
- **Scope Creep:** Fixes should be surgical, not rewrites
