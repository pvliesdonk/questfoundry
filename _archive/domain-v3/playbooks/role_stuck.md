# Role Stuck Recovery

> When an agent can't proceed, this playbook gets the workflow moving again.

This playbook defines the procedure for recovering when a role becomes stuck and cannot complete its work.

## When to Use

Invoke this playbook when:

- Role exceeds timeout without producing output
- Role enters a loop (same action repeated without progress)
- Role reports an error it cannot resolve
- Role escalates repeatedly without resolution
- Workflow is blocked waiting for a role

## Detection Signals

### Timeout

- Role exceeds `max_iterations` defined in loop
- Role exceeds `timeout` without meaningful progress
- No intent posted after extended period

### Loop Detection

- Same intent posted multiple times
- Identical artifacts created repeatedly
- No state change between iterations

### Error State

- Role posts `escalation` intent with unresolvable error
- Tool calls fail repeatedly
- Invalid output that fails validation

## Diagnosis Steps

### 1. Check Last Intent

What was the role trying to do?

- Intent type (handoff, escalation, terminate)
- Intent status (what state were they reporting)
- Message/reasoning (what were they thinking)

### 2. Check Pending Artifacts

What was being worked on?

- Are there incomplete artifacts in hot_store?
- Did a tool call fail mid-operation?
- Is there corrupted or invalid state?

### 3. Check Context

What led to this state?

- What was the Brief or triggering request?
- What did upstream roles hand off?
- Is the task actually completable with available tools?

### 4. Identify Root Cause

| Symptom | Likely Cause | Recovery Path |
|---------|--------------|---------------|
| Repeated escalations | Unclear requirements | Clarify Brief |
| Tool failures | Missing data or permissions | Fix preconditions |
| Loop behavior | Conflicting constraints | Simplify task |
| Timeout | Task too complex | Break into subtasks |
| Invalid output | Misunderstood format | Provide examples |

## Recovery Options

### Option 1: Retry with Simplified Context

**When:** Role was overwhelmed or confused.

1. Identify the core task that needs completing
2. Strip unnecessary context
3. Provide clear, minimal instructions
4. Retry with focused scope

### Option 2: Escalate to Showrunner

**When:** Role genuinely cannot proceed without decision.

1. Document the blocker clearly
2. Present options if any exist
3. Showrunner makes decision or reassigns
4. Resume with guidance

### Option 3: Reset Role State

**When:** Role has corrupted or inconsistent state.

1. Save any salvageable work from hot_store
2. Clear role's pending state
3. Re-initialize role context
4. Restart task from clean state

### Option 4: Delegate to Different Role

**When:** Wrong role for the task, or role lacks necessary capabilities.

1. Identify which role should handle this
2. Hand off with context from stuck role
3. Original role returns to dormant
4. New role proceeds

### Option 5: Human Intervention

**When:** Systemic issue that agents cannot resolve.

1. Document the situation fully
2. Flag for human operator
3. Human provides guidance or fixes underlying issue
4. Resume workflow

## Prevention Guidelines

### Clear Briefs

- Specify exactly what the role should produce
- Define success criteria
- Note relevant constraints and context

### Appropriate Scope

- Don't ask one role to do too much
- Break complex tasks into subtasks
- Match task complexity to role capabilities

### Explicit Handoffs

- Include all necessary context in handoffs
- Don't assume roles remember previous context
- Verify receiving role has what it needs

### Timeout Tuning

- Set realistic timeouts for task complexity
- Allow buffer for exploration and retry
- Don't set timeouts so tight they trigger false positives

## Stuck Role Checklist

When a role is stuck:

- [ ] Identified which role is stuck
- [ ] Checked last intent and artifacts
- [ ] Diagnosed root cause
- [ ] Chose recovery option
- [ ] Executed recovery
- [ ] Verified workflow resumed
- [ ] Documented what happened and why

## Anti-Patterns

- **Infinite retry**: Retrying without changing anything
- **Ignoring errors**: Hoping problems resolve themselves
- **Blame game**: Focusing on which role failed vs. fixing the issue
- **Scope expansion**: Adding more to a stuck role's plate
- **Context flooding**: Sending entire history instead of relevant info

## Success Criteria

Recovery is complete when:

- [ ] Stuck role is either functional or replaced
- [ ] Workflow is proceeding normally
- [ ] No artifacts left in corrupted state
- [ ] Root cause understood (even if not fully fixed)
- [ ] Prevention measures identified for future

## Summary

Stuck roles happen. Diagnose quickly, recover efficiently, and learn from each incident. The goal is workflow progress, not perfect execution. Sometimes the right answer is to simplify, reassign, or ask for help.
