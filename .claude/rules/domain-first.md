# Domain Knowledge is Source of Truth

## The Problem This Solves

LLMs (including you) have a pattern of treating existing code as authoritative over domain documentation. This leads to:

- Perpetuating bugs in code that doesn't match design intent
- "Fixing" code to match other broken code
- Ignoring domain specs when they conflict with implementation

## The Rule

When reasoning about how something **should** work:

1. **FIRST** consult domain knowledge (`domain-v4/`)
2. **THEN** check if code matches domain intent
3. **IF mismatch** - the code is wrong, not the domain

## Domain Knowledge Locations

| Question | Where to Look |
|----------|---------------|
| What is the contract? | `meta/schemas/core/*.schema.json` |
| What artifacts exist? | `domain-v4/artifact-types/*.json` |
| What do agents do? | `domain-v4/agents/*.json` |
| What stores exist? | `domain-v4/stores/*.json` |
| How do workflows flow? | `domain-v4/playbooks/*.json` |
| What knowledge is required? | `domain-v4/knowledge/` |
| Studio configuration? | `domain-v4/studio.json` |

## Example: Store Schema

**Wrong approach:** "I'll read _archive/runtime-v3/stores/cold_store.py to understand the schema"
**Right approach:** "I'll check meta/schemas/core/store.schema.json for the contract, then domain-v4/stores/ for instances"

## When Code and Domain Disagree

The domain is correct. The code has a bug. Fix the code to match domain intent.

Do NOT:

- Assume code is correct because it exists
- Update domain to match buggy code
- Treat your own previous code as authoritative
