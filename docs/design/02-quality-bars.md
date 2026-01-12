# Quality Bars

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

Quality bars define validation criteria for story content. Every story must pass all bars before SHIP. Bars are organized by automation potential and validation stage.

---

## The Eight Quality Bars

### 1. Integrity

**Definition**: Structural completeness and internal consistency.

**Checks**:
- All references resolve (passage IDs, codewords, items exist)
- No unintended dead ends (passages with no outgoing choices)
- State consistency (codewords don't create contradictions)
- Choice targets exist

**Automation**: Fully automatable

**Stage**: CONNECTIONS (pre-FILL)

```yaml
# Validation output
integrity:
  status: pass
  checks:
    - name: reference_resolution
      status: pass
      details: "All 47 passage references resolve"
    - name: dead_ends
      status: pass
      details: "3 intentional endings, 0 unintended dead ends"
    - name: state_consistency
      status: pass
      details: "No contradictory state paths detected"
```

### 2. Reachability

**Definition**: Critical content is accessible via viable player paths.

**Checks**:
- All passages reachable from start
- All endings reachable
- Gate conditions have obtainable paths
- No orphaned content

**Automation**: Fully automatable (graph traversal)

**Stage**: CONNECTIONS

```yaml
reachability:
  status: warn
  checks:
    - name: passage_reachability
      status: pass
      details: "All passages reachable from opening_001"
    - name: ending_reachability
      status: warn
      details: "Ending 'redemption' requires rare state combination"
```

### 3. Nonlinearity

**Definition**: Branching is meaningful, not decorative.

**Checks**:
- Hubs provide genuine choice diversity
- Loops add new content or state on return
- Gateways gate content worth gating
- Branch points have substantive differences

**Automation**: Partial (structure automatable, meaning requires LLM)

**Stage**: CONNECTIONS + FILL review

```yaml
nonlinearity:
  status: pass
  checks:
    - name: hub_diversity
      status: pass
      details: "Office hub offers 4 distinct investigation paths"
    - name: loop_progression
      status: pass
      details: "Chinatown loop adds new dialogue on each return"
    - name: branch_substance
      status: warn
      details: "Branches at morgue differ only in dialogue tone"
```

### 4. Gateways

**Definition**: Gate conditions are enforceable, fair, and spoiler-safe.

**Checks**:
- Gate conditions achievable before gate encountered
- Gates don't reference hidden implementation details
- Gate feedback is diegetic (in-world), not mechanical

**Automation**: Partial

**Stage**: CONNECTIONS

```yaml
gateways:
  status: pass
  checks:
    - name: achievability
      status: pass
      details: "All gate conditions have prior paths"
    - name: diegetic_feedback
      status: pass
      details: "Gate messages describe in-world barriers"
```

### 5. Style

**Definition**: Voice and register consistency across all content.

**Checks**:
- POV consistency (first person maintained)
- Tense consistency (past tense maintained)
- Character voice consistency
- Tone matches dream.yaml specification

**Automation**: LLM-based

**Stage**: FILL

```yaml
style:
  status: pass
  checks:
    - name: pov_consistency
      status: pass
      details: "First person maintained in all 47 passages"
    - name: voice_consistency
      status: warn
      details: "Passage chinatown_005 tone shifts unexpectedly"
```

### 6. Determinism

**Definition**: Story is reproducible from recorded parameters.

**Checks**:
- Same inputs produce same outputs
- Random elements are seeded and logged
- No hidden dependencies on external state

**Automation**: Fully automatable (with recorded seeds)

**Stage**: SHIP

```yaml
determinism:
  status: pass
  checks:
    - name: reproducibility
      status: pass
      details: "Re-generation with same seed produces identical output"
```

### 7. Presentation

**Definition**: Player-facing content is spoiler-safe and polished.

**Checks**:
- No internal IDs exposed to player (passage_047, codeword_03)
- No implementation language ("if you have the badge codeword")
- No debugging artifacts
- Choice text is polished and meaningful

**Automation**: Partial (pattern matching + LLM review)

**Stage**: FILL + SHIP

```yaml
presentation:
  status: pass
  checks:
    - name: id_exposure
      status: pass
      details: "No internal IDs in player-facing text"
    - name: implementation_language
      status: pass
      details: "No mechanical language detected"
    - name: choice_quality
      status: pass
      details: "All choice texts are distinct and meaningful"
```

### 8. Accessibility

**Definition**: Content is accessible to diverse players.

**Checks**:
- Navigation is clear (player knows available actions)
- Sensory descriptions don't exclude (visual-only, audio-only)
- Reading level appropriate for audience
- Content warnings present where needed

**Automation**: Partial (readability automatable, inclusion requires review)

**Stage**: FILL + SHIP

```yaml
accessibility:
  status: pass
  checks:
    - name: navigation_clarity
      status: pass
      details: "All passages have clear choice text"
    - name: sensory_balance
      status: pass
      details: "Multi-sensory descriptions throughout"
    - name: reading_level
      status: pass
      details: "Flesch-Kincaid grade level: 8 (target: 6-10)"
```

---

## Validation Stages

### Pre-Gate Validation (Automated)

Quick structural checks before expensive LLM validation:

| Bar | Pre-Gate Checks |
|-----|----------------|
| Integrity | Reference resolution, dead end detection |
| Reachability | Graph traversal, orphan detection |
| Gateways | Condition achievability |
| Determinism | Seed logging |

### Full-Gate Validation (LLM + Human)

Comprehensive validation including semantic checks:

| Bar | Full-Gate Checks |
|-----|-----------------|
| Nonlinearity | Branch meaning, loop value |
| Style | Voice, tone, consistency |
| Presentation | Spoiler safety, polish |
| Accessibility | Inclusion, clarity |

---

## Validation Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ CONNECTIONS │ ──► │  Pre-Gate   │ ──► │  FILL       │
│ Complete    │     │  Validation │     │  Stage      │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐
                    │   Pass?     │     │  Full-Gate  │
                    └──────┬──────┘     │  Validation │
                      yes  │  no        └──────┬──────┘
                    ┌──────┴──────┐            │
                    ▼             ▼     ┌──────▼──────┐
              Continue       Fix issues │   Pass?     │
                                        └──────┬──────┘
                                          yes  │  no
                                        ┌──────┴──────┐
                                        ▼             ▼
                                      SHIP        Rework
```

---

## Validation Reports

Each validation produces a structured report:

```yaml
# validation_report.yaml

timestamp: 2026-01-01T10:30:00Z
project: noir_mystery
stage: full_gate

overall_status: warn  # pass | warn | fail

bars:
  integrity:
    status: pass
    checks: [...]

  reachability:
    status: pass
    checks: [...]

  nonlinearity:
    status: warn
    checks:
      - name: branch_substance
        status: warn
        location: grow/branches/morgue_path.yaml
        details: "Options differ only in dialogue tone"
        suggestion: "Add distinct consequences or revelations"

  # ... remaining bars

summary:
  passed: 6
  warnings: 2
  failed: 0

recommendations:
  - "Review morgue_path branches for substantive differences"
  - "Consider adding state effect to distinguish chinatown options"
```

---

## Per-Bar Configuration

Some bars can be configured per project:

```yaml
# project.yaml

quality:
  bars:
    accessibility:
      reading_level:
        target: 8
        range: [6, 10]
      content_warnings: required

    style:
      pov: first_person
      tense: past
      voice_reference: seed.protagonist.voice_notes

    determinism:
      enabled: true
      seed_logging: required
```

---

## Automation Matrix

| Bar | Automated | LLM Review | Human Review |
|-----|-----------|------------|--------------|
| Integrity | 100% | 0% | 0% |
| Reachability | 100% | 0% | 0% |
| Gateways | 80% | 20% | 0% |
| Determinism | 100% | 0% | 0% |
| Nonlinearity | 30% | 50% | 20% |
| Style | 20% | 60% | 20% |
| Presentation | 50% | 40% | 10% |
| Accessibility | 40% | 30% | 30% |

---

## Failure Handling

### Blocking Failures

Some failures block progression:

| Bar | Blocking? |
|-----|-----------|
| Integrity | Yes — broken references crash export |
| Reachability | Yes — orphan content wastes work |
| Gateways | Yes — impossible gates frustrate players |
| Others | Advisory — human can override |

### Override Protocol

For non-blocking warnings, human can override:

```yaml
# override.yaml

overrides:
  - bar: nonlinearity
    location: grow/branches/morgue_path.yaml
    check: branch_substance
    reason: "Dialogue tone difference is intentional characterization"
    approved_by: author
    date: 2026-01-01
```

---

## Integration with Pipeline

### CONNECTIONS Stage

```bash
qf validate --pre-gate
# Runs: Integrity, Reachability, Gateways (partial), Determinism
```

### FILL Stage

```bash
qf validate --full-gate
# Runs: All bars including LLM-based checks
```

### SHIP Stage

```bash
qf validate --final
# Runs: All bars + presentation + export-specific checks
```

---

## See Also

- [03-grow-stage-specification.md](./03-grow-stage-specification.md) — CONNECTIONS validation
- [07-design-principles.md](./07-design-principles.md) — Underlying principles
- [08-research-foundation.md](./08-research-foundation.md) — Research basis for bars
