---
procedure_id: fact_corroboration
name: Fact Corroboration
description: Validate claims affecting plot feasibility (physics, law, policy, technology, medicine, history, language, culture)
roles: [researcher]
references_schemas:
  - research_memo.schema.json
references_expertises:
  - researcher_verification
quality_bars: [integrity]
---

# Fact Corroboration

## Purpose

Validate real-world factual claims in canon/prose to maintain plausibility and avoid errors that break immersion or credibility.

## Scope

### In Scope

- Physics/engineering feasibility
- Medical/biological accuracy
- Legal/policy frameworks
- Historical/cultural accuracy
- Linguistic accuracy
- Technology plausibility

### Out of Scope

- Canon consistency (handled by Lore Weaver)
- Style preferences (handled by Style Lead)
- Speculative fiction elements (OK to bend rules with justification)

## Validation Levels

### Corroborated

**Multiple reliable sources confirm**

- Mark: ✓ Corroborated
- Citations: 2-5 sources
- No caveats needed in prose

### Plausible

**Reasonable but not directly confirmed**

- Mark: ⚠ Plausible
- Note: "No direct sources but consistent with known principles"
- Safe to use with neutral phrasing

### Disputed

**Conflicting evidence or expert disagreement**

- Mark: ⚠ Disputed
- Note: "Sources conflict; provide multiple perspectives"
- Recommend neutral wording or avoid specifics

### Uncorroborated (Low/Med/High Risk)

**No evidence found**

- Mark: ⚠ Uncorroborated:low/med/high
- Provide safe neutral phrasing
- Schedule research TU if risk ≥ med

## Steps

### 1. Receive Research Request

- Extract specific claim to validate
- Note context (why this matters to plot)

### 2. Conduct Research

- Search 2-5 reliable sources
- Note expert consensus or disagreement
- Identify caveats or edge cases

### 3. Assign Validation Level

- Corroborated / Plausible / Disputed / Uncorroborated
- If uncorroborated, assess risk level

### 4. Provide Creative Implications

- What does this enable? (affordances)
- What does this forbid? (constraints)
- Suggest plot/gate/canon opportunities

### 5. Document Research Memo

- Question asked
- Answer (with validation level)
- Citations (2-5)
- Caveats
- Creative implications

### 6. Suggest Neutral Phrasing (If Needed)

- For disputed/uncorroborated claims
- Keep surfaces safe without specifics

## Research Memo Template

```yaml
question: "Can low-gravity environments cause long-term bone density loss?"

answer: "Yes, corroborated by multiple space medicine studies."

validation_level: corroborated

citations:
  - "NASA Human Research Program, 2018"
  - "ESA Bone Loss Study, 2020"
  - "Journal of Space Medicine, Vol 45"

caveats:
  - "Timeline varies by individual (6-12 months typically)"
  - "Countermeasures exist (exercise, medication)"

creative_implications:
  enables:
    - "Long-term station workers have visible health impacts"
    - "Medical checkups/treatment as plot points"
    - "Gates based on medical clearance"
  forbids:
    - "Instant bone loss (requires extended stay)"

suggested_phrasing:
  - "Years in low-G take a toll on the bones"
  - "The medic checks your bone density scan"
```

## Outputs

- `research.memo` - Question, answer, citations, caveats, implications
- `research.posture` - Validation level
- `research.phrasing` - Neutral alternatives (if needed)

## Quality Bars Pressed

- **Integrity:** Factual accuracy maintained

## Handoffs

- **To Lore Weaver:** Provide constraints (not outcomes)
- **To Plotwright:** Suggest plausible mechanisms
- **To Style Lead:** Flag terminology/sensitivity issues

## Common Issues

- **Speculation Presented as Fact:** Mark clearly as plausible/uncorroborated
- **Anachronisms:** Modern tech in historical setting
- **Cultural Stereotypes:** Flag for sensitivity review
- **Over-Certainty:** Claim absolute when evidence disputed
