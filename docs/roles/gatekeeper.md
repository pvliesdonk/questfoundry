# Gatekeeper (GK)

The Gatekeeper is the **quality enforcer** of the QuestFoundry studio.

## Profile

| Attribute | Value |
|-----------|-------|
| Abbreviation | GK |
| Archetype | Auditor |
| Agency | Low |
| Mandate | Enforce Quality Bars |

## Responsibilities

- Evaluate content against quality bars
- Create gatecheck reports
- Block promotion of low-quality content
- Provide actionable feedback

## The 8 Quality Bars

| Bar | Type | Description |
|-----|------|-------------|
| Integrity | Tool | Structural completeness |
| Reachability | Tool | All nodes accessible |
| Nonlinearity | LLM | Meaningful choices |
| Gateways | Tool | Gate conditions valid |
| Style | LLM | Narrative consistency |
| Determinism | Tool | Reproducible outcomes |
| Presentation | Tool | Content completeness |
| Accessibility | LLM | Player accessibility |

## Tools

- `evaluate_integrity` - Check structural completeness
- `evaluate_reachability` - Verify node accessibility
- `evaluate_nonlinearity` - Assess choice meaningfulness
- `evaluate_gateways` - Validate gate conditions
- `evaluate_style` - Check narrative consistency
- `evaluate_determinism` - Verify reproducibility
- `evaluate_presentation` - Check content completeness
- `evaluate_accessibility` - Assess player accessibility
- `create_gatecheck_report` - Generate evaluation report

## Intent Protocol

### Topology Validation

- **topology_passed**: Structure is valid
- **topology_failed**: Structure has issues

### Prose Validation

- **prose_passed**: Content is ready for canon
- **prose_failed**: Content needs revision

## See Also

- [Domain Definition](https://github.com/pvliesdonk/questfoundry/blob/main/src/questfoundry/domain/roles/gatekeeper.md)
