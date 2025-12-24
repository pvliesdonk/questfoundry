# Issue 261: Refactor must_know to Structured Knowledge

## Overview

Replace inline markdown in `must_know` knowledge entries with structured JSON validated against a new schema. Per discussion, **deprecate `inline` entirely** from v4.

## Current State Analysis

### Meta Layer

- `meta/schemas/knowledge/knowledge-entry.schema.json` supports `content.type` in `{inline, file_ref, structured, corpus}`
- `content.data` for structured type is unconstrained (`{}`)
- No `structured-content.schema.json` exists

### Domain Layer (11 must_know entries)

| Entry | Type | Content Patterns |
|-------|------|------------------|
| `quality_bars_overview` | inline | Numbered list of principles + role section |
| `delegation_protocol` | inline | Critical rules + examples (code blocks) + anti-patterns |
| `runtime_guidelines` | inline | Output rules + examples + stop conditions |
| `lifecycle_contract` | inline | State table + transitions + do/don't lists |
| `topology_patterns` | inline | Pattern definitions with purpose/requirements/examples |
| `choice_integrity` | inline | Principle + contrastive examples + test |
| `diegetic_gates` | inline | Gate types + good/bad phrasings + PN enforcement |
| `spoiler_hygiene` | inline | Leak taxonomy table + canon/codex separation |
| `sources_of_truth` | inline | Hot/Cold definitions + workflow steps |
| `runtime_guidelines_core` | inline | Core rules + anti-pattern |
| `tool_response_interpretation` | file_ref | Tables + examples + lifecycle feedback |

### Runtime Layer

- `agent/knowledge.py:244-257`: Handles structured via `json.dumps(content.data)`
- `tools/consult_knowledge.py:123-125`: Same handling
- `models/base.py:363-374`: `KnowledgeContent` model with all type variants

## Proposed Structured Content Schema

### Design Philosophy: Semantics over Format

Types describe **what the knowledge IS**, not how it's displayed. Format (tables, bullets, code blocks) is a rendering concern, not a schema concern.

### Semantic Types

| Type | Purpose | Key Distinction |
|------|---------|-----------------|
| `rule` | Unilateral constraint | Imposed, no negotiation |
| `contract` | Multi-party agreement | Between agents, handoff obligations |
| `criterion` | Quality judgment standard | What pass/fail means |
| `heuristic` | Contextual guidance | "When X, prefer Y" |
| `definition` | Vocabulary establishment | What a term means here |
| `procedure` | Ordered steps | How to accomplish something |
| `warning` | Failure mode to avoid | Consequence if violated |

### Common Attributes

All semantic types can have:

- `examples: string[]` - How to do it right
- `counter_examples: string[]` - What NOT to do
- `reasoning: string` - Why this matters

### Schema Definition

```json
{
  "$defs": {
    "rule": {
      "type": "object",
      "required": ["statement"],
      "properties": {
        "statement": { "type": "string", "description": "The constraint" },
        "enforcement": { "enum": ["runtime", "llm"], "default": "llm" },
        "severity": { "enum": ["critical", "error", "warning"], "default": "error" },
        "reasoning": { "type": "string" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "contract": {
      "type": "object",
      "required": ["parties", "obligations"],
      "properties": {
        "name": { "type": "string" },
        "parties": { "type": "array", "items": { "type": "string" }, "minItems": 2 },
        "obligations": { "type": "array", "items": { "$ref": "#/$defs/obligation" } },
        "reasoning": { "type": "string" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "obligation": {
      "type": "object",
      "required": ["party", "must"],
      "properties": {
        "party": { "type": "string", "description": "Who is obligated" },
        "must": { "type": "string", "description": "What they must do" },
        "when": { "type": "string", "description": "Trigger condition" }
      }
    },
    "criterion": {
      "type": "object",
      "required": ["name", "pass_condition"],
      "properties": {
        "name": { "type": "string" },
        "pass_condition": { "type": "string", "description": "What 'pass' means" },
        "fail_indicators": { "type": "array", "items": { "type": "string" } },
        "reasoning": { "type": "string" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "heuristic": {
      "type": "object",
      "required": ["guidance"],
      "properties": {
        "context": { "type": "string", "description": "When this applies" },
        "guidance": { "type": "string", "description": "What to prefer/do" },
        "reasoning": { "type": "string" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "definition": {
      "type": "object",
      "required": ["term", "meaning"],
      "properties": {
        "term": { "type": "string" },
        "meaning": { "type": "string" },
        "distinguishing_features": { "type": "array", "items": { "type": "string" } },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "procedure": {
      "type": "object",
      "required": ["goal", "steps"],
      "properties": {
        "goal": { "type": "string", "description": "What this accomplishes" },
        "preconditions": { "type": "array", "items": { "type": "string" } },
        "steps": { "type": "array", "items": { "type": "string" } },
        "postconditions": { "type": "array", "items": { "type": "string" } },
        "reasoning": { "type": "string" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "warning": {
      "type": "object",
      "required": ["failure_mode"],
      "properties": {
        "failure_mode": { "type": "string", "description": "What goes wrong" },
        "consequence": { "type": "string", "description": "Why it matters" },
        "detection": { "type": "string", "description": "How to recognize it" },
        "prevention": { "type": "string", "description": "How to avoid it" },
        "examples": { "type": "array", "items": { "type": "string" } },
        "counter_examples": { "type": "array", "items": { "type": "string" } }
      }
    },
    "structured_entry": {
      "type": "object",
      "description": "Top-level container mixing semantic types as needed",
      "properties": {
        "rules": { "type": "array", "items": { "$ref": "#/$defs/rule" } },
        "contracts": { "type": "array", "items": { "$ref": "#/$defs/contract" } },
        "criteria": { "type": "array", "items": { "$ref": "#/$defs/criterion" } },
        "heuristics": { "type": "array", "items": { "$ref": "#/$defs/heuristic" } },
        "definitions": { "type": "array", "items": { "$ref": "#/$defs/definition" } },
        "procedures": { "type": "array", "items": { "$ref": "#/$defs/procedure" } },
        "warnings": { "type": "array", "items": { "$ref": "#/$defs/warning" } }
      }
    }
  }
}
```

### Mapping Existing Content to Semantic Types

| Entry | Primary Types |
|-------|---------------|
| `quality_bars_overview` | criteria (8 bars) + rules (Gatekeeper role) |
| `delegation_protocol` | contract (agent↔orchestrator handoff) + procedure |
| `lifecycle_contract` | contract (state machine) + definitions (states) |
| `runtime_guidelines` | rules + procedure + warnings |
| `topology_patterns` | definitions (Hub/Loop/Gateway) + heuristics |
| `choice_integrity` | criterion + heuristics + warnings |
| `diegetic_gates` | definitions + rules + warnings |
| `spoiler_hygiene` | rules + criterion + warnings (with counter_examples) |
| `sources_of_truth` | definitions (Hot/Cold) + contract + procedure |
| `runtime_guidelines_core` | rules + warnings |
| `tool_response_interpretation` | definitions (taxonomy) + procedure |

## Implementation Plan

### Phase 1: Meta Schema Creation

**Files**: `meta/schemas/knowledge/structured-content.schema.json`

1. Create `structured-content.schema.json` with `$defs` for:
   - `principle` - statement/reasoning/examples
   - `pattern` - name/definition/purpose/requirements/example
   - `checklist` - label/items
   - `antipattern` - context/bad/good
   - `table` - caption/columns/rows
   - `code_example` - description/language/code
   - `section` - title/content (containing above)
   - `structured_entry` - top-level wrapper with sections array

2. Update `knowledge-entry.schema.json`:
   - Remove `inline` from `content.type` enum
   - Remove `content.text` field
   - Add `$ref` to `structured-content.schema.json#/$defs/structured_entry` for `content.data` when `type: structured`
   - Keep `file_ref` for backward compat (convert `tool_response_interpretation.md` → structured later)
   - Keep `corpus` unchanged

### Phase 2: Domain Migration

**Files**: All `domain-v4/knowledge/must_know/*.json`

For each entry:

1. Analyze markdown structure to identify sections/principles/patterns/etc.
2. Convert to structured JSON with `content.type: structured`
3. Preserve semantic content, improve clarity
4. Validate against new schema

Migration order (by complexity):

1. `runtime_guidelines_core` - Simple (4 rules + anti-pattern)
2. `sources_of_truth` - Medium (2 sections + workflow)
3. `quality_bars_overview` - Medium (8 numbered items + role)
4. `topology_patterns` - Medium (3 pattern definitions)
5. `choice_integrity` - Medium (principle + examples + test)
6. `lifecycle_contract` - Medium (states + transitions + do/don't)
7. `delegation_protocol` - Complex (rules + code examples + patterns)
8. `runtime_guidelines` - Complex (format + examples + conditions)
9. `diegetic_gates` - Complex (types + examples + PN rules)
10. `spoiler_hygiene` - Complex (taxonomy table + canon/codex)
11. `tool_response_interpretation` - Convert from file_ref to structured

### Phase 3: Runtime Updates

**Files**:

- `src/questfoundry/runtime/models/base.py`
- `src/questfoundry/runtime/agent/knowledge.py`
- `src/questfoundry/runtime/tools/consult_knowledge.py`

1. **Update `KnowledgeContent` model** (`models/base.py:363-374`):
   - Remove `inline` from `type` Literal
   - Remove `text` field
   - Add Pydantic models for structured content types (optional, for validation)

2. **Update knowledge context builder** (`agent/knowledge.py:235-257`):
   - Remove `inline` handling
   - Enhance structured rendering to produce readable markdown from structured data
   - Add renderer functions for each content type (principle → markdown, etc.)

3. **Update consult_knowledge tool** (`tools/consult_knowledge.py:108-144`):
   - Remove `inline` handling
   - Add structured content rendering (same renderers as above)
   - Support section extraction from structured content (by title)

4. **Add structured content renderers**:
   - `render_principle(p) → "**{statement}**\n\n{reasoning}\n\nExamples:\n- ..."`
   - `render_pattern(p) → "### {name}\n\n**Definition**: ...\n**Purpose**: ..."`
   - `render_checklist(c) → "**{label}**:\n- [ ] item1\n- [ ] item2"`
   - `render_antipattern(a) → "❌ {bad}\n✅ {good}"`
   - `render_table(t) → markdown table format`
   - `render_code_example(e) → "```{language}\n{code}\n```"`
   - `render_section(s) → "## {title}\n\n{rendered_content}"`
   - `render_structured_entry(e) → joined sections`

### Phase 4: Testing & Validation

1. **Schema validation**: Ensure all migrated entries pass JSON Schema
2. **Rendering tests**: Verify structured → markdown produces equivalent output
3. **Integration tests**: Knowledge context builds correctly with new format
4. **E2E test**: Agent workflows function with structured knowledge

## Example Migration: `runtime_guidelines_core`

### Before (inline markdown)

```json
{
  "content": {
    "type": "inline",
    "format": "markdown",
    "text": "## Core Orchestrator Rules\n\n1. **All output via tools** - Use delegate()..."
  }
}
```

### After (structured with semantic types)

```json
{
  "content": {
    "type": "structured",
    "format": "json",
    "schema_ref": "knowledge/structured-content.schema.json#/$defs/structured_entry",
    "data": {
      "rules": [
        {
          "statement": "All output via tools",
          "enforcement": "llm",
          "severity": "critical",
          "reasoning": "Orchestrators coordinate, they don't create",
          "examples": ["delegate(to_agent='plotwright', task='...')"],
          "counter_examples": ["Writing a paragraph of story prose directly"]
        },
        {
          "statement": "Never generate story prose",
          "enforcement": "llm",
          "severity": "critical",
          "reasoning": "If writing narrative, STOP and delegate to specialist",
          "counter_examples": ["The detective entered the dimly lit room..."]
        },
        {
          "statement": "Coordinate specialists, don't create",
          "enforcement": "llm",
          "severity": "error"
        },
        {
          "statement": "Consult knowledge for details",
          "enforcement": "llm",
          "severity": "warning",
          "examples": ["consult_knowledge(entry_id='delegation_protocol')"]
        }
      ],
      "heuristics": [
        {
          "context": "Deciding when to terminate",
          "guidance": "Terminate when all artifacts validated and promoted to canon",
          "counter_examples": ["Terminating while content awaits validation"]
        }
      ],
      "warnings": [
        {
          "failure_mode": "Response contains more than 2-3 sentences that aren't tool calls",
          "consequence": "You're doing the work instead of delegating",
          "prevention": "Delegate to specialist agents instead"
        }
      ]
    }
  }
}
```

## Design Decisions

1. **Deprecate `inline` entirely** - Per issue comment, no mixed world needed
2. **Keep `file_ref`** - For truly large external content (rare)
3. **Semantic types over format types** - Types describe what knowledge IS, not how it's displayed
4. **Flat structure, not nested sections** - Single `structured_entry` with arrays of semantic types
5. **Examples + counter_examples as attributes** - They belong to something, not standalone
6. **Renderers produce markdown** - Agents consume markdown; structured is source of truth
7. **No backwards compatibility** - v4 is a clean break

## Semantic Type Benefits

| Benefit | How |
|---------|-----|
| **Queryable** | "Give me all rules with severity=critical" |
| **Composable** | Combine rules from multiple entries |
| **Validatable** | Schema enforces structure (rule needs statement) |
| **Renderable** | Each type has natural markdown representation |
| **Extensible** | Add new types without breaking existing |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Content loss during migration | Diff before/after rendered output |
| Schema too rigid | Allow `reasoning` as escape hatch for nuance |
| Wrong type chosen | Clear guidance in schema descriptions |
| Runtime rendering bugs | Comprehensive unit tests for each renderer |
| Authors find structured harder | Provide templates and validation tooling |

## Resolved Questions

### 1. Markdown within strings

**Decision**: Yes, allow markdown in `statement`, `meaning`, `guidance`, etc.

**Rationale**: Emphasis (`**bold**`) and inline code add clarity. Content is already written this way. Rendering target is markdown anyway.

**Guideline**: Keep it minimal - emphasis and inline code only. If you need headers or lists, add another semantic type instead.

### 2. Code examples in `examples` array

**Decision**: Yes, as strings with embedded markdown code blocks.

**Rationale**: Simpler than typed examples. Renderer already handles markdown.

```json
"examples": ["```json\n{\"summary\": \"Created 3 sections\"}\n```"]
```

### 3. Handling `tool_response_interpretation` (file_ref entry)

**Decision**: Migrate to structured, but do it last.

**Rationale**: Content maps well to `definitions` (taxonomy) + `procedure` + `warnings`. Good test case for schema. Migrate after schema is proven with simpler entries.

### 4. Migration approach

**Decision**: LLM-assisted with human review.

**Process**:

1. Feed original markdown to LLM with schema and semantic type definitions
2. LLM outputs structured JSON with semantic classification
3. Validate against JSON Schema
4. Render back to markdown
5. Human reviews: diff original vs rendered, check semantic accuracy
6. Iterate if needed

**Rationale**: LLMs are good at semantic classification. Schema validation catches structural errors. Human review catches semantic edge cases. Faster than fully manual for 11 entries.

### 5. State machines in knowledge

**Decision**: Model `lifecycle_contract` using `contract` + `rules` + `definitions` types.

**Rationale**: Knowledge explains behavior; meta defines structure. The lifecycle state machine is defined in `meta/schemas/core/artifact-type.schema.json`. Knowledge teaches agents how to work within it, not duplicate the definition.

**Follow-up**: See #262 for broader discussion on whether lifecycle definition and explanation should be co-located.
