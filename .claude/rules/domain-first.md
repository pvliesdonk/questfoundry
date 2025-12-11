# Domain Knowledge is Source of Truth

## The Problem This Solves

LLMs (including you) have a pattern of treating existing code as authoritative over domain documentation. This leads to:

- Perpetuating bugs in code that doesn't match design intent
- "Fixing" code to match other broken code
- Ignoring domain specs when they conflict with implementation

## The Rule

When reasoning about how something **should** work:

1. **FIRST** consult domain knowledge (`src/questfoundry/domain/`)
2. **THEN** check if code matches domain intent
3. **IF mismatch** - the code is wrong, not the domain

## Domain Knowledge Locations

| Question | Where to Look |
|----------|---------------|
| What artifacts exist? | `domain/ontology/artifacts.md` |
| What do roles do? | `domain/roles/*.md` |
| How do workflows flow? | `domain/loops/*.md` |
| What are valid states? | `domain/ontology/enums.md` |

## Example: Cold Store Schema

**Wrong approach:** "I'll read cold_store.py to understand the schema"
**Right approach:** "I'll check domain/ontology/artifacts.md for store: fields, then verify cold_store.py matches"

## When Code and Domain Disagree

The domain is correct. The code has a bug. Fix the code (or compiler) to match domain intent.

Do NOT:

- Assume code is correct because it exists
- Update domain to match buggy code
- Treat your own previous code as authoritative
