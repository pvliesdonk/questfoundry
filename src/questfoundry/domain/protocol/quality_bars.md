# Protocol: Quality Bars

> **Status:** Normative
> **Version:** 3.0.0

This document defines the **8 Quality Bars** that artifacts must pass before canonization.

---

## Overview

Quality Bars are the gatekeeper's criteria for evaluating artifacts. Each bar has:

- **Definition:** What it measures
- **Checks:** Specific validations
- **Failures:** What causes red/yellow status
- **Remediation:** How to fix issues

---

## The 8 Quality Bars

:::{quality-bar}
id: integrity
name: "Integrity"
description: "No contradictions in canon"
owner: lorekeeper
checks:

- "All facts traceable to sources"
- "No circular references"
- "Timeline consistency"
- "Character facts consistent"
- "Location facts consistent"
failures:
- "Orphaned references to non-existent entities"
- "Contradictory statements about same entity"
- "Timeline paradoxes"
- "Dead links to removed content"
remediation:
- "Trace contradiction to source"
- "Determine canonical truth"
- "Update or remove conflicting content"
:::

:::{quality-bar}
id: reachability
name: "Reachability"
description: "All content accessible via valid paths"
owner: plotwright
checks:

- "Every section reachable from start"
- "No orphaned content"
- "All paths lead somewhere"
- "No dead ends without explicit design"
failures:
- "Section with no incoming paths"
- "Branch that leads nowhere"
- "Content referenced but not created"
- "Locked content with impossible unlock"
remediation:
- "Add missing path connections"
- "Remove or integrate orphaned content"
- "Fix broken references"
:::

:::{quality-bar}
id: nonlinearity
name: "Nonlinearity"
description: "Multiple valid paths exist"
owner: plotwright
checks:

- "Player has meaningful choices"
- "Different paths lead to different experiences"
- "Choices have consequences"
- "No false choices (all options identical)"
failures:
- "Single linear path through content"
- "Choices that don't matter"
- "Illusion of choice (converge immediately)"
- "Missing alternative routes"
remediation:
- "Add genuine branching points"
- "Differentiate path outcomes"
- "Ensure choice consequences are visible"
:::

:::{quality-bar}
id: gateways
name: "Gateways"
description: "All gates have valid unlock conditions"
owner: plotwright
checks:

- "Every gate has defined unlock criteria"
- "Unlock criteria are achievable"
- "No impossible gates"
- "Gate states properly tracked"
failures:
- "Gate with no unlock condition"
- "Unlock requires impossible combination"
- "Circular unlock dependencies"
- "Gate state not persisted"
remediation:
- "Define unlock criteria"
- "Verify criteria achievability"
- "Break circular dependencies"
:::

:::{quality-bar}
id: style
name: "Style"
description: "Voice and tone consistency"
owner: creative_director
checks:

- "Consistent narrative voice"
- "Tone matches genre expectations"
- "Character voices distinct"
- "No jarring style shifts"
failures:
- "Mixed point of view"
- "Inconsistent tone (comedy in horror)"
- "Character voice drift"
- "Unexplained style changes"
remediation:
- "Establish style guide"
- "Review for voice consistency"
- "Smooth transitions between tones"
:::

:::{quality-bar}
id: determinism
name: "Determinism"
description: "Same inputs produce same outputs"
owner: publisher
checks:

- "Reproducible builds"
- "Consistent formatting"
- "Stable output ordering"
- "No random variations without seed"
failures:
- "Different output on rebuild"
- "Non-deterministic ordering"
- "Unstable formatting"
- "Unseeded randomness"
remediation:
- "Add deterministic seeds"
- "Sort collections before output"
- "Pin format versions"
:::

:::{quality-bar}
id: presentation
name: "Presentation"
description: "Formatting and structure correct"
owner: publisher
checks:

- "Valid markdown/markup"
- "Consistent heading hierarchy"
- "Proper list formatting"
- "Images/media properly linked"
failures:
- "Broken markup syntax"
- "Inconsistent heading levels"
- "Malformed lists"
- "Missing media files"
remediation:
- "Fix syntax errors"
- "Standardize heading structure"
- "Verify media paths"
:::

:::{quality-bar}
id: accessibility
name: "Accessibility"
description: "Content usable by all players"
owner: gatekeeper
checks:

- "Alt text for images"
- "Color contrast sufficient"
- "Screen reader compatible"
- "Keyboard navigable"
failures:
- "Images without alt text"
- "Low contrast text"
- "Audio without transcript"
- "Click-only interactions"
remediation:
- "Add alt text descriptions"
- "Improve color contrast"
- "Provide transcripts"
- "Add keyboard alternatives"
:::

---

## Bar Status Levels

| Status | Meaning | Action |
|--------|---------|--------|
| **Green** | Passes all checks | Ready for canonization |
| **Yellow** | Minor issues | Can proceed with notes |
| **Red** | Blocking issues | Must fix before proceeding |

---

## Gatecheck Decision Matrix

| Bar Status | Decision | Next Step |
|------------|----------|-----------|
| All green | **Pass** | Approve for Cold |
| Some yellow, no red | **Conditional Pass** | Approve with notes |
| Any red | **Block** | Return for rework |

---

## Role Responsibilities

| Bar | Primary Owner | Backup |
|-----|---------------|--------|
| Integrity | Lorekeeper | Gatekeeper |
| Reachability | Plotwright | Gatekeeper |
| Nonlinearity | Plotwright | Gatekeeper |
| Gateways | Plotwright | Gatekeeper |
| Style | Creative Director | Gatekeeper |
| Determinism | Publisher | Gatekeeper |
| Presentation | Publisher | Gatekeeper |
| Accessibility | Gatekeeper | - |

---

## Gatecheck Report Format

:::{gatecheck-report-format}
fields:

- title: "Artifact or TU being evaluated"
- checked: "Date of evaluation"
- gatekeeper: "Evaluator name/agent"
- mode: "pre-gate | full-gate"
- bars:
  - bar: "Bar name"
        status: "green | yellow | red"
        notes: "Specific findings"
        fixes: "Required remediation"
- decision: "pass | conditional | block"
- handoffs: "Remediation assignments"
:::

---

## Cross-References

- `domain/roles/gatekeeper.md` — GK role definition
- `domain/protocol/lifecycles/artifact.md` — Artifact lifecycle
- `runtime/tools.py` — Gatecheck tool implementations
