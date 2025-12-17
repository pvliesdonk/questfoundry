# Knowledge Patterns

This document describes patterns for managing knowledge in agent prompts, with emphasis on budget management and the menu+consult pattern.

## The Problem: Context Explosion

LLM agents have limited context windows. If we inline all knowledge into every prompt:

- Prompts become bloated (10k+ tokens just for knowledge)
- Critical guidance gets lost in noise
- Token costs increase significantly
- Model performance degrades with excessive context

## The Solution: Layered Knowledge with Budget Management

Knowledge is stratified by **layer**, which determines the default injection strategy:

| Layer | Default Strategy | When to Use |
|-------|------------------|-------------|
| `constitution` | Always inline | Inviolable principles, safety rules |
| `must_know` | Inline up to budget | Critical operational guidance |
| `should_know` | Menu only | Important but not critical |
| `role_specific` | Menu only | Specialist knowledge |
| `lookup` | Never inline | Reference material, queried on demand |

---

## The Menu+Consult Pattern

**Principle**: Show summaries in the prompt (the "menu"), retrieve full content on demand (via "consult").

### Prompt Structure

```text
┌────────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                                   │
│                                                                 │
│ ## Constitution                                                 │
│ [Full content - always present]                                 │
│                                                                 │
│ ## Critical Guidance                                            │
│ [Full content of must_know entries, up to budget]               │
│                                                                 │
│ ## Knowledge Menu                                               │
│ Use `consult_knowledge(entry_id)` for full details.             │
│                                                                 │
│ - **spoiler_hygiene**: Never reveal future plot points          │
│ - **delegation_patterns**: When and how to delegate work        │
│ - **quality_bars_overview**: The 8 quality criteria             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Agent sees menu**: Brief summaries of available knowledge
2. **Agent decides**: "I need more detail on spoiler hygiene"
3. **Agent calls tool**: `consult_knowledge("spoiler_hygiene")`
4. **Runtime returns**: Full content with examples and procedures
5. **Agent proceeds**: With detailed guidance in context

---

## Knowledge Entry Structure

Each knowledge entry has two key text fields:

| Field | Purpose | When Used |
|-------|---------|-----------|
| `summary` | Brief description (1-3 sentences) | Menu display, search results |
| `content` | Full detailed content | Retrieved via `consult_knowledge` |

### Example Entry

```json
{
  "id": "spoiler_hygiene",
  "name": "Spoiler Hygiene",
  "layer": "must_know",
  "summary": "Never reveal future plot points. Describe current state only. Gate reveals behind player actions.",
  "content": {
    "type": "inline",
    "text": "## Spoiler Hygiene\n\nThe cardinal rule of interactive fiction...\n\n### Examples\n\nWRONG: The dragon will attack the village in Act 3...\nCORRECT: The village seems peaceful, but rumors speak of an ancient threat...\n\n[... 500 more words of detailed guidance ...]"
  }
}
```

**Key insight**: The `summary` is what appears in the menu. The full `content` is only loaded when explicitly requested.

---

## Budget Management

The runtime tracks token usage when building prompts:

### Budget Allocation

```python
class KnowledgeContextBuilder:
    def __init__(self, config: KnowledgeBudgetConfig):
        self.total_budget = config.total_prompt_budget_tokens  # e.g., 4000
        self.constitution_budget = config.constitution_tokens   # e.g., 500
        self.must_know_budget = config.must_know_tokens         # e.g., 1500
        self.menu_budget = config.menu_tokens                   # e.g., 500
```

### Injection Algorithm

```python
def build_knowledge_context(self, agent: Agent, studio: Studio) -> str:
    sections = []
    used_tokens = 0

    # 1. Constitution - always inline (fixed budget)
    constitution = self.get_constitution(studio)
    sections.append(constitution)
    used_tokens += self.count_tokens(constitution)

    # 2. Must-know - inline up to budget, overflow to menu
    menu_items = []
    for entry_id in agent.knowledge_requirements.must_know:
        entry = studio.knowledge[entry_id]
        entry_tokens = self.count_tokens(entry.content)

        if used_tokens + entry_tokens <= self.must_know_budget:
            # Fits in budget - inline it
            sections.append(self.format_inline(entry))
            used_tokens += entry_tokens
        else:
            # Over budget - add to menu instead
            menu_items.append(self.format_menu_item(entry))

    # 3. Should-know and role-specific - menu only
    for entry_id in agent.knowledge_requirements.should_know:
        entry = studio.knowledge[entry_id]
        menu_items.append(self.format_menu_item(entry))

    # 4. Build menu section
    if menu_items:
        sections.append(self.format_menu(menu_items))

    return "\n\n".join(sections)
```

### Budget Targets

| Agent Type | Target Prompt Size | Knowledge Budget |
|------------|-------------------|------------------|
| Orchestrator | < 4,000 tokens | ~2,000 tokens |
| Creator | < 6,000 tokens | ~3,000 tokens |
| Validator | < 4,000 tokens | ~2,000 tokens |

---

## The `consult_knowledge` Tool

Agents retrieve full knowledge content via this tool.

### Tool Definition

```json
{
  "id": "consult_knowledge",
  "name": "Consult Knowledge",
  "description": "Retrieve detailed guidance from the studio knowledge base.",
  "input_schema": {
    "type": "object",
    "properties": {
      "entry_id": {
        "type": "string",
        "description": "The knowledge entry ID"
      },
      "section": {
        "type": "string",
        "description": "Optional: specific section within the entry"
      }
    },
    "required": ["entry_id"]
  }
}
```

### Usage Examples

**Full entry retrieval**:

```json
{
  "entry_id": "spoiler_hygiene"
}
```

**Section-specific retrieval** (for large entries):

```json
{
  "entry_id": "runtime_guidelines",
  "section": "Tool Usage"
}
```

### Return Value

```json
{
  "entry_id": "spoiler_hygiene",
  "name": "Spoiler Hygiene",
  "layer": "must_know",
  "content": "## Spoiler Hygiene\n\n[Full content...]",
  "related_entries": ["sources_of_truth", "diegetic_gating"]
}
```

---

## Agent Knowledge Requirements

Each agent declares what knowledge it needs:

```json
{
  "knowledge_requirements": {
    "constitution": true,
    "must_know": ["spoiler_hygiene", "runtime_guidelines"],
    "should_know": ["delegation_patterns", "quality_bars_overview"],
    "role_specific": ["showrunner_operating_principles"]
  }
}
```

### Override Behavior

An agent's list can **override** the entry's default layer:

| Entry Layer | In Agent's List | Result |
|-------------|-----------------|--------|
| `should_know` | `must_know[]` | Inlined (budget permitting) |
| `must_know` | `should_know[]` | Menu only |
| `role_specific` | `must_know[]` | Inlined (budget permitting) |

This allows agents to promote important entries to always-inline or demote less critical ones to menu-only.

---

## Authoring Guidelines

When creating knowledge entries:

### Summary Best Practices

- Keep under 100 characters
- Focus on the "what" not the "how"
- Make it scannable in a list
- Include key action verb

**Good**: "Never reveal future plot points. Gate reveals behind player actions."

**Bad**: "This entry contains guidance about how to handle information that the player character doesn't know yet, including examples of correct and incorrect approaches."

### Content Structure

- Start with a clear heading
- Lead with the most important rule/principle
- Put examples after the core guidance
- Use WRONG/CORRECT comparisons
- Keep total content under 1000 tokens when possible

### When to Split Entries

Split an entry if:

- Content exceeds 1500 tokens
- It covers multiple distinct topics
- Different agents need different parts

---

## The Reference Shelf Pattern

For external reference material (genre guides, craft books), use the `corpus` content type:

```json
{
  "id": "genre_conventions",
  "layer": "lookup",
  "summary": "Genre tropes and conventions. Search with consult_corpus tool.",
  "content": {
    "type": "corpus",
    "corpus_ref": {
      "store_ref": "reference_library",
      "path_pattern": "genres/**"
    }
  }
}
```

**Key distinction**:

| Type | Use For | Access Pattern |
|------|---------|----------------|
| `inline`/`file_ref` | Studio rules, brand voice | `consult_knowledge(id)` |
| `corpus` | External guides, general craft | `consult_corpus(id, query)` |

This prevents confusion between internal rules and external suggestions.

---

## Monitoring and Tuning

### Metrics to Track

- **Consult frequency**: Which entries are accessed most?
- **Prompt size**: Are budgets being respected?
- **Overflow rate**: How often do must_know entries go to menu?

### Tuning Process

1. Monitor which entries are consulted frequently
2. Consider promoting frequently-consulted entries to `must_know`
3. Consider demoting rarely-consulted `must_know` entries
4. Adjust budgets based on agent performance

### Signs of Problems

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Agent ignores guidance | Entry not in `must_know` | Promote entry |
| Agent over-consults | Entry content too brief | Expand summary |
| Prompt too large | Too many `must_know` | Demote or split entries |
| Agent confused | Conflicting entries | Consolidate or clarify |

---

## Summary

| Pattern | Purpose |
|---------|---------|
| Layered knowledge | Control injection strategy per entry |
| Budget management | Prevent context explosion |
| Menu+consult | Show summaries, retrieve on demand |
| Summary/content split | Enable efficient menu display |
| Agent overrides | Per-agent customization of defaults |
| Reference shelf | Separate internal rules from external guides |

This architecture keeps prompts lean while ensuring agents can access detailed guidance when needed.
