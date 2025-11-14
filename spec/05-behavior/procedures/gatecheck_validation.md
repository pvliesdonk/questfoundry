---
procedure_id: gatecheck_validation
description: Gatekeeper's comprehensive validation against 8 quality bars
version: 2.0.0
references_expertises:
  - gatekeeper_quality_bars
references_schemas:
  - gatecheck_report.schema.json
references_roles:
  - gatekeeper
  - showrunner
tags:
  - validation
  - quality-bars
  - gatecheck
---

# Gatecheck Validation Procedure

## Overview

Comprehensive validation of artifacts against all 8 quality bars before Hot→Cold merge. This is a hard gate enforced by Gatekeeper.

## The 8 Quality Bars

1. **Integrity** - No dead references
2. **Reachability** - Critical beats accessible
3. **Nonlinearity** - Meaningful branching
4. **Gateways** - Clear diegetic conditions
5. **Style** - Consistent voice/register
6. **Determinism** - Reproducible assets
7. **Presentation** - Spoiler-free surfaces
8. **Accessibility** - Navigation and alt text

## Prerequisites

- TU artifacts submitted for gatecheck
- All artifacts have passed schema validation (validation_report.json present)
- Showrunner `gate.submit` request with TU context

## Step 1: Enumerate Artifacts

List all artifacts in the TU submission.

**Actions:**

1. Review `gate.submit` payload for artifact list
2. Verify each artifact file exists
3. Check each artifact has `"$schema"` field
4. Locate corresponding `validation_report.json` for each

**Output:** Complete artifact inventory

**If missing artifacts or validation reports:**

- BLOCK immediately
- Report missing files to Showrunner
- Do not proceed with gatecheck

## Step 2: Schema Validation Audit (Mandatory Bar)

Verify all artifacts passed schema validation.

**For each artifact:**

1. **Locate `validation_report.json`:**
   - Example: `hook_card.json` → `hook_card_validation_report.json`

2. **Verify report structure:**

   ```json
   {
     "artifact_path": "out/hook_card.json",
     "schema_id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
     "schema_sha256": "abc123...",
     "valid": true,
     "errors": [],
     "timestamp": "2025-11-06T12:00:00Z",
     "validator": "jsonschema-python-4.20"
   }
   ```

3. **Check validation status:**
   - `"valid": true` required
   - `"errors": []` must be empty
   - Schema SHA-256 must match SCHEMA_INDEX.json

**If ANY artifact fails schema validation:**

- **BLOCK merge** (hard gate)
- List ALL artifacts with validation issues
- Provide remediation for each:

  ```
  Artifact 'hook_card.json' failed validation.
  Errors:
  - $.hook_id: Required property missing
  - $.tags: Expected array, got string

  Producer role must fix and re-validate before resubmission.
  ```

- Escalate to Showrunner with role assignments

**Integration with Determinism Bar:**
Schema validation is prerequisite for Determinism. Invalid schemas make determinism checks irrelevant.

## Step 3: Bar-by-Bar Audit

Systematically check each of the 8 quality bars.

### Bar 1: Integrity

**Validates:** Referential integrity and internal consistency

**Checks:**

- All choice `target_id` resolve to existing sections
- All gateway conditions reference defined state variables
- All canon references point to existing entries
- All codex references resolve
- No orphaned artifacts
- Timeline anchors consistent
- Entity state transitions coherent

**Example violations:**

- Choice points to deleted section
- Gateway checks undefined variable
- Scene callbacks non-existent canon

**Remediation:** List broken references with file:line

### Bar 2: Reachability

**Validates:** All critical beats accessible via valid paths

**Checks:**

- Start section reachable
- All mandatory beats have path from start
- No impossible gateway combinations blocking critical content
- Loop returns have exit conditions
- Hub diversity supports multiple paths

**Example violations:**

- Keystone beat behind impossible gateways
- Required content only via single fragile path
- Dead-end loops with no escape

**Remediation:** Path analysis showing unreachable beats, suggested fixes

### Bar 3: Nonlinearity

**Validates:** Meaningful choice consequences

**Checks:**

- Multiple viable paths exist
- Choices are contrastive (not cosmetic)
- Loop-with-difference shows state changes
- State effects create branching
- Hub returns reflect decisions

**Example violations:**

- All choices converge immediately
- Loop returns identical regardless of state
- Choices with no narrative consequence

**Remediation:** Identify cosmetic choices, suggest state-based differentiation

### Bar 4: Gateways

**Validates:** Clear diegetic choice conditions

**Checks:**

- Gateway reasons are world-based (not meta)
- Conditions comprehensible through story
- PN-safe phrasing (no codeword leaks)
- Consistency across similar choices
- No player-hostile hidden gates

**Example violations:**

- Meta conditions ("if flag_x == true")
- Incomprehensible requirements
- Arbitrary restrictions without story justification

**Remediation:** Suggest diegetic phrasings aligned with canon

### Bar 5: Style

**Validates:** Consistent voice and register

**Checks:**

- Register matches style guide
- Voice consistent across sections
- Diction appropriate for setting
- Motif usage aligned
- No anachronisms or register breaks
- Paragraph rhythm maintained

**Example violations:**

- Register shifts between sections
- Inconsistent character voice
- Modern idioms in historical settings

**Remediation:** Highlight inconsistencies with style guide refs

### Bar 6: Determinism

**Validates:** Reproducible asset generation

**Checks:**

- All images have generation parameters logged
- All audio has production metadata
- Asset manifests include checksums
- Generation prompts version-controlled
- Provider/model versions recorded
- Seeds documented for regeneration

**Example violations:**

- Assets without generation parameters
- Missing checksums
- Undocumented manual edits
- Provider version not recorded

**Remediation:** List assets missing determinism metadata

### Bar 7: Presentation

**Validates:** Spoiler hygiene and player safety

**Checks:**

- No spoilers in player surfaces
- Canon stays Hot only
- Codex entries player-safe
- Gateway phrasings don't leak mechanics
- Choice text doesn't preview outcomes
- Section titles avoid spoilers

**Example violations:**

- Canon details in codex
- Choice text reveals consequences
- Section titles spoil twists
- Gateway text exposes state variables

**Remediation:** Flag spoiler leaks with neutral phrasing suggestions

### Bar 8: Accessibility

**Validates:** Navigation and inclusive design

**Checks:**

- All images have alt text
- Navigation clear and consistent
- Choice presentation accessible
- No reliance on color alone
- Text contrast meets standards
- Screen reader compatible

**Example violations:**

- Missing alt text
- Unclear navigation
- Color-only indicators
- Low contrast text

**Remediation:** List accessibility issues with WCAG refs

## Step 4: Collect Violations

Aggregate all violations across bars.

**Actions:**

1. Create violation list per bar
2. Classify severity: Critical vs Minor
3. Assign remediation responsibility (which role fixes)
4. Provide specific, actionable fixes

**Output:** Structured violation report

**Example:**

```yaml
violations:
  bar_1_integrity:
    - location: "section_42.md:line 15"
      issue: "Choice target '#deleted-section' not found"
      severity: "critical"
      assigned_role: "scene_smith"
      fix: "Update choice target to '#alternative-path'"

  bar_7_presentation:
    - location: "codex_kestrel.json:field 'backstory'"
      issue: "Canon spoiler in codex entry: reveals betrayal"
      severity: "critical"
      assigned_role: "codex_curator"
      fix: "Replace with player-safe summary: 'mysterious past'"
```

## Step 5: Determine Severity

Grade overall submission.

**Critical violations (must fix):**

- Schema validation failures
- Spoiler leaks in Cold
- Broken references (Integrity)
- Unreachable critical beats (Reachability)

**Major violations (should fix):**

- Gateway clarity issues
- Style inconsistencies
- Missing determinism params
- Accessibility gaps

**Minor violations (polish):**

- Style refinements
- Optional accessibility improvements

**Decision:**

- **PASS:** No critical, few/no major
- **FAIL:** Any critical OR multiple major

## Step 6: Issue Decision

Generate gatecheck report and decision.

**If PASS:**

```json
{
  "intent": "gate.decision",
  "sender": "GK",
  "receiver": "SR",
  "context": {
    "tu": "TU-2025-11-06-LW01",
    "correlation_id": "msg-gate-submit"
  },
  "payload": {
    "decision": "pass",
    "bars_checked": ["integrity", "reachability", "nonlinearity", "gateways", "style", "determinism", "presentation", "accessibility"],
    "critical_violations": 0,
    "major_violations": 0,
    "minor_violations": 2,
    "notes": "All critical and major bars passed. Minor style polish recommended but non-blocking.",
    "minor_issues": [
      {"bar": "style", "suggestion": "Consider motif tie-in at section 15"},
      {"bar": "accessibility", "suggestion": "Alt text for image_03 could be more descriptive"}
    ],
    "authorization": "merge_to_cold_approved"
  }
}
```

**If FAIL:**

```json
{
  "intent": "gate.decision",
  "sender": "GK",
  "receiver": "SR",
  "context": {
    "tu": "TU-2025-11-06-LW01"
  },
  "payload": {
    "decision": "fail",
    "bars_checked": [...],
    "critical_violations": 3,
    "major_violations": 5,
    "blocking_issues": [
      {
        "bar": "integrity",
        "location": "section_42.md:15",
        "issue": "Broken choice reference",
        "severity": "critical",
        "assigned_role": "scene_smith",
        "fix": "Update target to valid section"
      },
      {
        "bar": "presentation",
        "location": "codex_kestrel.json",
        "issue": "Spoiler leak in codex",
        "severity": "critical",
        "assigned_role": "codex_curator",
        "fix": "Redact canon details, use player-safe summary"
      }
    ],
    "remediation_required": true,
    "resubmit_after_fixes": true
  }
}
```

## Step 7: Coordinate Remediation (If Failed)

Work with Showrunner to fix issues.

**Actions:**

1. Showrunner assigns fixes to appropriate roles
2. Roles address violations systematically
3. Artifacts re-validated (schema + quality)
4. Resubmit to Gatekeeper when ready

**Tracking:** Update TU with remediation status

## Enforcement

**Hard gates (no exceptions):**

- Schema validation failures
- Spoiler leaks in Cold
- Broken references
- Unreachable critical beats

**Escalate to Showrunner when:**

- Remediation requires cross-role coordination
- Multiple bars fail (systemic issues)
- Same violations recur
- Human decision needed on trade-offs

## Summary Checklist

- [ ] All artifacts enumerated
- [ ] Schema validation audit complete (all valid)
- [ ] 8 bars checked systematically
- [ ] Violations collected and classified
- [ ] Severity determined
- [ ] Decision issued (pass/fail)
- [ ] If fail: remediation coordinated
- [ ] Authorization granted or withheld

**Gatecheck is the final quality gate before Hot→Cold merge. No exceptions to hard gates.**
