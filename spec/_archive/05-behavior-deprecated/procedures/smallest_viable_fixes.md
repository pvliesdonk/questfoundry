---
procedure_id: smallest_viable_fixes
name: Smallest Viable Fixes
description: Identify minimal, surgical changes to pass quality bars without scope creep or rewrites
roles: [gatekeeper]
references_schemas:
  - gatecheck_report.schema.json
references_expertises:
  - gatekeeper_quality_bars
quality_bars: [all]
---

# Smallest Viable Fixes

## Purpose

For each quality bar failure (yellow/red), identify the minimal change required to pass—avoiding scope creep, rewrites, or systemic changes.

## Principles

### Minimal

**The smallest change that resolves the issue**

Example:

- ❌ "Rewrite entire section"
- ✓ "Change line 47 choice label from 'Take sabotage chip' to 'Take datachip'"

### Surgical

**Localized fix, not systemic rework**

Example:

- ❌ "Redesign entire topology"
- ✓ "Add one path from Hub B to Section 'Engineering'"

### Scoped

**Can complete in current TU**

Example:

- ❌ "Requires new Lore Deepening TU for canon backfill"
- ✓ "Use existing canon from TU-2024-10-12"

### Owned

**Clear role assignment**

Example:

- ❌ "Someone should fix this"
- ✓ "Scene Smith to revise lines 45-47"

## Steps

### 1. Identify Root Cause

- What exactly is failing the bar?
- Is it a single line, a pattern, or structural?

### 2. Determine Minimal Change

- What's the smallest edit that resolves this?
- Line-level change? Single artifact? Structural adjustment?

### 3. Validate Scope

- Can this be done in current TU?
- Does it require coordination with other roles?
- Does it cascade into other changes?

### 4. Assign Owner

- Which role owns this artifact?
- Can they make this change quickly?

### 5. Document Fix

- Specific location (file, line number, artifact ID)
- Exact change required
- Owner assignment
- Estimated effort (quick/moderate/requires-coordination)

## Fix Patterns by Bar

### Integrity Fixes

**Pattern:** Update references, fix broken links

Examples:

- "Line 23: Update reference from TU-OLD to TU-2024-10-15"
- "Section 'Cargo Bay': Add anchor for crosslink target"
- "Canon Pack: Add timeline entry for 'Station Collapse'"

### Reachability Fixes

**Pattern:** Add paths, remove dead ends

Examples:

- "Add path from Section 'Hub A' to Section 'Engineering'"
- "Section 'Dead End': Add exit choice to 'Return to Cargo Bay'"
- "Gateway 'secure_access': Add second acquisition path"

### Nonlinearity Fixes

**Pattern:** Add distinction, remove decorative branches

Examples:

- "Hub 'Choose Path': Differentiate Branch A outcome from Branch B"
- "Loop 'Return to Office': Add changed element (desk empty)"
- "Remove decorative Hub C (all branches identical)"

### Gateways Fixes

**Pattern:** Add signposting, make conditions diegetic

Examples:

- "Section 'Cargo Bay': Add mention of 'maintenance hex-key'"
- "Gateway 'engineering_access': Change condition from 'completed X' to 'has hex-key'"
- "Section 'Office': Make hex-key visible on desk"

### Style Fixes

**Pattern:** Adjust phrasing, match register

Examples:

- "Line 47: Change 'proceed' to 'slip through' (matches register)"
- "Section 'Chase': Reduce sentence length (3 → 2 per paragraph)"
- "Choices: Make contrastive ('Go'/'Proceed' → 'Move quickly'/'Move carefully')"

### Determinism Fixes

**Pattern:** Create logs, move technique off-surface

Examples:

- "Create determinism_log for Art Render #12 (seed + model)"
- "Caption line 5: Remove 'generated with DALL-E' mention"
- "Audio Note: Move DAW details to off-surface log"

### Presentation Fixes

**Pattern:** Remove spoilers, hide internals

Examples:

- "Line 47: Change 'sabotage chip' to 'datachip' (spoiler)"
- "Caption: Remove 'this foreshadows the betrayal'"
- "Choice: Remove codeword reference 'FLAG_UNION_MEMBER'"

### Accessibility Fixes

**Pattern:** Add alt text, make links descriptive

Examples:

- "Image #3: Add alt text 'Cargo bay with damaged crates'"
- "Link line 23: Change 'click here' to 'See Salvage Permits entry'"
- "Audio Cue #5: Add text equivalent '[Distant alarm chirps twice]'"

## Fix Documentation Template

```yaml
bar: presentation
status: YELLOW
finding: "Line 47: Choice label reveals plot twist"
smallest_viable_fix:
  location: "section_cargo_bay.md line 47"
  current: "Take the sabotage chip"
  required: "Take the datachip"
  owner: scene_smith
  effort: quick
  rationale: "Remove spoiler word 'sabotage', use neutral term"
```

## When Fix is NOT Viable

### Escalate to Showrunner when:

- Fix requires new canon (needs Lore Deepening TU)
- Fix requires structural change (needs Story Spark TU)
- Fix requires cross-loop coordination
- Fix is too large for current TU

### Example Escalation:

```yaml
bar: reachability
status: RED
finding: "Section 'Engineering' is unreachable from any hub"
remediation: "ESCALATE: Requires topology rework in Story Spark TU"
reason: "Not a surgical fix; needs structural redesign"
```

## Effort Estimates

### Quick (< 10 minutes)

- Single line changes
- Reference updates
- Link fixes

### Moderate (10-30 minutes)

- Multiple line changes in one artifact
- Small structural adjustments (add one path)
- Caption/alt text creation

### Requires Coordination (30+ minutes)

- Changes affecting multiple roles
- Cascading updates
- Structural adjustments requiring validation

## Outputs

- Detailed fix specifications for each yellow/red bar
- Owner assignments
- Effort estimates

## Handoffs

- **To Owners:** Deliver specific fix requirements
- **To Showrunner:** Escalate fixes that exceed viable scope

## Common Pitfalls

- **Scope Creep:** "Fix this" → "Rewrite everything"
- **Vague Fixes:** "Make it better" instead of specific change
- **Missing Owner:** No role assigned to implement
- **Unrealistic Scope:** Fixes that can't complete in current TU
