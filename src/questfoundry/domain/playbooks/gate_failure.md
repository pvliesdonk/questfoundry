# Gate Failure Recovery

> When Gatekeeper blocks, this playbook gets you moving again.

This playbook defines how Showrunner recovers when work is blocked by a Gatekeeper quality gate failure.

## When to Use

Invoke this playbook when:

- Gatekeeper returns `status: failed` on a quality gate
- Work cannot proceed due to bar violations
- Multiple revision attempts have failed
- Team needs guidance on remediation path

## Diagnosis Steps

### 1. Read the GatecheckReport

Identify from the report:

- **Which bars failed**: integrity, reachability, nonlinearity, gateways, style, etc.
- **Specific issues**: What exactly violated each bar
- **Recommendations**: Gatekeeper's suggested fixes
- **Target artifact**: What was being validated

### 2. Classify the Failure

| Failure Type | Symptoms | Likely Fix |
|--------------|----------|------------|
| **Structural** | reachability, nonlinearity, gateways | Plotwright revision |
| **Content** | integrity, style | Lorekeeper or Scene Smith revision |
| **Presentation** | spoiler leaks, formatting | Scene Smith or role that created content |
| **Ambiguity** | Gatekeeper couldn't determine compliance | Clarify requirements, then re-gate |

### 3. Identify Root Cause

- Is this a one-time mistake or systemic issue?
- Did the Brief specify the bar that failed?
- Was the bar appropriate for this work?

## Recovery Options

### Option 1: Revise and Resubmit (Default)

**When:** Clear issue with clear fix.

1. Route work back to responsible role with GatecheckReport
2. Role makes specific fixes addressing each issue
3. Role resubmits for gatecheck
4. Gatekeeper re-validates

**Anti-pattern:** Vague "try again" without specific guidance.

### Option 2: Escalate for Waiver

**When:** Bar is inappropriate for this work, or strict compliance would harm the project.

1. Showrunner reviews the failure and Gatekeeper's reasoning
2. Showrunner decides if waiver is justified
3. If approved: Document waiver rationale and accepted risks
4. Gatekeeper records waiver in GatecheckReport
5. Work proceeds with waiver notation

**Waiver conditions:**

- Must be Showrunner-approved (no self-waivers)
- Must document rationale and risks
- Must be exception, not pattern

### Option 3: Roll Back

**When:** Work has drifted too far from viable path.

1. Identify last known-good state
2. Discard or archive failed work
3. Return to known-good state
4. Restart with clearer Brief

**When to use:** Multiple failed attempts, fundamental approach is wrong.

### Option 4: Scope Reduction

**When:** Failure is due to over-ambitious scope.

1. Identify minimum viable scope that can pass gates
2. Create new Brief with reduced scope
3. Defer cut content to future work
4. Proceed with reduced scope

## Waiver Process

### Requesting a Waiver

1. Gatekeeper submits `status: waiver_requested` with:
   - Which bar would be waived
   - Why strict compliance is problematic
   - Risks of proceeding without compliance
   - Proposed mitigations

2. Showrunner evaluates:
   - Is the bar appropriate for this work?
   - What are the actual risks?
   - Are mitigations sufficient?

3. Decision:
   - **Approve**: Document and proceed
   - **Deny**: Must fix before proceeding
   - **Modify**: Adjust scope or approach

### Waiver Anti-Patterns

- Waiving bars to meet deadlines (bars exist for reasons)
- Serial waivers on same bar (fix the systemic issue)
- Waiving without documentation (future problems)
- Self-approving waivers (must be Showrunner)

## Repeated Failures

If the same artifact fails multiple times:

1. **Stop and diagnose** — Something systemic is wrong
2. **Review the Brief** — Was the work well-defined?
3. **Review the approach** — Is the method fundamentally flawed?
4. **Consider reset** — Sometimes starting fresh is faster

## Success Criteria

Recovery is complete when:

- [ ] Gatekeeper returns `status: passed` OR
- [ ] Showrunner approves documented waiver
- [ ] Root cause is understood (even if waived)
- [ ] Lessons are captured for future work

## Summary

Gate failures are information, not punishment. Diagnose carefully, fix specifically, and escalate thoughtfully. Waivers are valid tools but require justification and documentation.
