---
name: questfoundry-stage-implementation
description: Use when adding a new QuestFoundry pipeline stage, modifying an existing stage's structure, working on `ConversationRunner`, `with_structured_output()`, tool response shapes, validation/repair loops, or anywhere in `src/questfoundry/pipeline/stages/` or `src/questfoundry/conversation/`.
---

# QuestFoundry Stage Implementation

## Overview

Every pipeline stage (DREAM, BRAINSTORM, SEED, GROW, FILL, SHIP, plus DRESS/POLISH) follows the same scaffolding. The DREAM stage is the reference implementation (see ADR-009). This skill is the playbook for either adding a new stage or correctly modifying an existing one.

**Authoritative spec for stage *behaviour* lives in `docs/design/procedures/<stage>.md`** — that defines what the stage must produce. This skill defines *how* the stage is wired up.

## Three-phase pattern (Discuss → Summarize → Serialize)

Every stage uses this scaffold:

1. **Discuss** — explore / refine via dialogue and research tools (using `langchain.agents`).
2. **Summarize** — distill into concise narrative (direct model call).
3. **Serialize** — convert to validated YAML / structured output via `with_structured_output()`.

The `ConversationRunner` orchestrates all three phases and handles:

- Tool execution and result collection
- Validation and repair loops (max 3 retries)
- Message history management
- Token counting and logging

**Do not** invent a new orchestrator for a new stage. Use `ConversationRunner`.

## Provider strategies for structured output

When serializing structured output (phase 3), all providers use the same strategy:

```python
structured_model = model.with_structured_output(
    schema=BrainstormOutput,
    method="json_schema",  # native JSON mode
)
```

**Why JSON_MODE for all providers?**

- Works reliably with complex nested schemas (`BrainstormOutput`, `SeedOutput`).
- The TOOL strategy (`function_calling`) was tried and returns `None` for complex schemas on Ollama.
- Native JSON mode is efficient and consistent across providers.

The two available strategies (for reference):

- `method="json_schema"` (JSON_MODE) — uses provider's native JSON mode to constrain output.
- `method="function_calling"` (TOOL) — creates a fake tool with the schema, forces model to call it.

Strategy selection lives in `providers/structured_output.py`.

## Prompt management

Use `ChatPromptTemplate`:

- LangChain's `ChatPromptTemplate` for variable injection.
- Template stored externally in `prompts/templates/<stagename>.md`.
- Separated from serialization logic.

When building prompts for stages:

1. Create `prompts/templates/<stagename>.md` with template text.
2. Use `ChatPromptTemplate.from_template()` to load and parameterize.
3. Pass variables dict to template compilation.
4. **Never hardcode prompts in Python — always externalize.**

For prompt review (any new template, any modification to context formatting), dispatch the `@prompt-engineer` subagent.

## Tool response format

All tool results MUST return structured JSON (per ADR-008):

```python
# Success
{
    "result": "success",
    "data": { ... },
    "action": "Use this information to inform decisions.",
}

# No results
{
    "result": "no_results",
    "query": "...",
    "action": "No matching guidance found. Proceed with your instincts.",
}

# Error
{
    "result": "error",
    "error": "Connection timeout",
    "action": "Tool unavailable. Continue without this information.",
}
```

This prevents infinite loops where the LLM follows "try again" instructions repeatedly.

## Validation & repair loop

When structured output validation fails, return errors in a shape the LLM can act on:

```python
def _validate_dream(self, data: dict[str, Any]) -> ValidationResult:
    """Validate using Pydantic model.

    Returns structured errors that guide LLM correction.
    """
    try:
        DreamArtifact.model_validate(data)
        return ValidationResult(valid=True, data=...)
    except ValidationError as e:
        return ValidationResult(
            valid=False,
            errors=pydantic_errors_to_details(e.errors()),
            expected_fields=get_all_field_paths(DreamArtifact),
        )
```

The error format tells the LLM exactly what's wrong and what to fix, enabling recovery in 1–2 retries. Generic "validation failed" feedback does not converge.

**Repair-loop routing (avoid bug class):** when constructing the corrective user-message, the `field_path` prefix MUST match the prompt phase that produced the field — otherwise feedback gets routed to the wrong serialize prompt and never converges. See #1243/#1244 and #1246/#1247 for prior occurrences.

## Ontology → Pydantic → Graph

Every new stage model traces back to the ontology:

```
docs/design/story-graph-ontology.md   ← Ontology definition (node types, edges, invariants)
docs/design/how-branching-stories-work.md ← Narrative model the ontology serves
        ↓
src/questfoundry/models/*.py          ← Hand-written Pydantic models (validate LLM output)
        ↓
graph/mutations.py                    ← Semantic validation against graph state
        ↓
graph.db                              ← Runtime source of truth (SQLite, nodes + edges)
```

When adding a new stage's models:

1. Read the ontology for the node types and edges your stage produces or consumes.
2. Create / update Pydantic models in `src/questfoundry/models/`.
3. Add semantic validation in `graph/mutations.py` if the rule is graph-relative.
4. Export from `models/__init__.py`.

**Optional vs Nullable (semantic distinction):**

- **Optional** (not in `required`): field may be absent → Pydantic defaults to `None`.
- **Nullable**: field explicitly accepts `null` as a value.
- LLMs often send `null` for optional fields; use `strip_null_values()` before validation to treat `null` as absent.

## Pre-implementation analysis (do this before writing code)

Before implementing a stage or a non-trivial change to one, enumerate:

- **Empty inputs** — empty strings, empty lists, empty dicts, `None`.
- **Missing fields** — what if optional fields are absent? required ones?
- **Invalid types** — wrong type passed, type not in supported set.
- **Special characters** — quotes, newlines, tabs, unicode in string inputs.
- **Name collisions** — what if two inputs produce the same output name?
- **Error paths** — what happens when external calls fail?

Then write the edge-case tests *first* (TDD). Trace every default and fallback — silent fallbacks like `items.get("type", "Any")` are bugs, not features. Prefer raising `ValueError` with a helpful message over returning a default.

For TDD discipline itself, use the global `superpowers:test-driven-development` skill.

## What this skill is NOT

- It is not a procedure spec. The stage's actual behaviour rules live in `docs/design/procedures/<stage>.md`.
- It is not a prompt-writing guide. For prompt content, dispatch `@prompt-engineer`.
- It is not an LLM-debugging skill. For "why is this stage's output wrong", see `questfoundry-llm-debugging`.
