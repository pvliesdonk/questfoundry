# Design Patterns

This document describes key patterns for using the meta-model effectively.

## consult-schema Pattern

**Problem**: LLMs struggle with complex schemas when creating artifacts.

**Solution**: Agent requests schema + documentation before creating.

### Flow

```
Agent: "I need to create a [ArtifactType]. What fields does it need?"

Runtime: Returns:
  1. JSON Schema for the artifact type
  2. Field descriptions and examples
  3. Current lifecycle state requirements
  4. Related quality criteria

Agent: Creates artifact following the guidance
```

### Implementation

The runtime provides a tool like `consult_schema(artifact_type_id)` that returns:

```json
{
  "artifact_type": "scene",
  "required_fields": ["id", "title", "content"],
  "optional_fields": ["setting", "characters", "notes"],
  "field_guidance": {
    "title": "Brief, evocative title for the scene",
    "content": "Full narrative content of the scene"
  },
  "lifecycle": {
    "initial_state": "draft",
    "current_valid_states": ["draft"]
  },
  "quality_criteria": ["style_check", "continuity_check"]
}
```

---

## validate-with-feedback Pattern

**Problem**: Validation failures should be actionable, not just rejections.

**Solution**: Return specific, field-level feedback that agents can act on.

### Flow

```
Agent: Creates artifact

Runtime: Validates against schema
  If valid: Accepts artifact
  If invalid: Returns structured feedback

Feedback format:
{
  "valid": false,
  "errors": [
    {
      "field": "content",
      "error": "Required field is empty",
      "guidance": "Provide the full narrative content"
    }
  ],
  "warnings": [
    {
      "field": "word_count",
      "warning": "Unusually short for a scene",
      "suggestion": "Consider expanding if this is a full scene"
    }
  ]
}

Agent: Corrects issues based on feedback
Agent: Resubmits
```

---

## menu+consult Pattern

**Problem**: LLMs have limited context; can't include all knowledge.

**Solution**: Include summaries (menu) in prompt, retrieve details (consult) on demand.

### Knowledge Stratification

```
System Prompt (always present):
┌─────────────────────────────────────────────┐
│ CONSTITUTION                                │
│ - Principle 1: ...                          │
│ - Principle 2: ...                          │
│                                             │
│ AVAILABLE KNOWLEDGE (menu):                 │
│ - style_guide: Writing style guidelines     │
│ - character_bible: Character definitions    │
│ - world_lore: World building details        │
│                                             │
│ Use consult(id) to retrieve full details.   │
└─────────────────────────────────────────────┘

On demand (via tool call):
Agent: consult("character_bible")
Runtime: Returns full character bible content
```

### Layer Access Patterns (Defaults)

| Layer | Default In Prompt | Via Tool |
|-------|-------------------|----------|
| constitution | Full content | - |
| must_know | Full content | - |
| should_know | Summary only | Full via `consult()` |
| role_specific | Summary only | Full via `consult()` |
| lookup | Not included | Via `query()` with search |

**Note**: These are *defaults*. An agent's `knowledge_requirements` lists can override injection strategy. For example, an agent can place a `role_specific` entry in their `must_know[]` list to always inject it. The entry's `applicable_to` field controls scope (who can reference it), independent of injection strategy.

---

## Runtime Nudging Pattern

**Problem**: LLM agents may skip steps or forget outputs.

**Solution**: Runtime detects discrepancies and nudges agents.

### Nudge Types

**Missing Output Nudge**:

```
Runtime: "According to the playbook, step 'create_draft' should produce
         a 'draft' artifact, but none was created. Did you intend to
         skip this, or should you create the draft now?"
```

**Unexpected State Nudge**:

```
Runtime: "The playbook indicates we're in the 'review' phase, but I see
         you're creating new content rather than reviewing. Is this
         intentional, or should we return to drafting first?"
```

**Quality Gate Reminder**:

```
Runtime: "Before proceeding to 'delivery' phase, the playbook requires
         passing the 'style_check' quality gate. Would you like me to
         run that check now?"
```

### Implementation

The runtime tracks:

- Current playbook and phase
- Expected inputs/outputs per step
- Quality checkpoints encountered

When it detects a discrepancy, it surfaces this to the active agent (usually the orchestrator) as a question, not an error.

---

## Self-Organizing Team Pattern

**Problem**: Some work requires multiple agents collaborating.

**Solution**: Define a team with roles and let them self-coordinate.

### Team Definition

```yaml
team:
  roles:
    - archetype: researcher
      responsibility: "Gather background information and references"
    - archetype: creator
      responsibility: "Draft the content using research"
    - archetype: validator
      responsibility: "Review for accuracy and consistency"
  coordination: self_organizing
  lead: creator
```

### Coordination Modes

| Mode | Behavior |
|------|----------|
| `sequential` | Roles execute in order listed |
| `parallel` | Roles work simultaneously, sync at end |
| `self_organizing` | Lead coordinates, team decides order |

### Self-Organizing Flow

1. Lead receives the delegation
2. Lead consults team members' expertise
3. Lead proposes work distribution
4. Team members execute their responsibilities
5. Lead integrates results
6. Lead reports completion

---

## Rework Loop Pattern

**Problem**: Quality gates may reject work, requiring revision.

**Solution**: Structured rework with feedback preservation.

### Playbook Structure

```yaml
phases:
  - id: create
    # ... creation steps ...
    on_completion: { next_phase: validate }

  - id: validate
    # ... validation steps ...
    quality_checkpoint:
      criteria: [quality_check]
    on_completion: { message: "Approved" }  # Ends playbook
    on_failure: { next_phase: revise, message: "Needs revision" }

  - id: revise
    # ... revision steps ...
    on_completion: { next_phase: validate }  # Loop back!
```

### Feedback Preservation

When delegating for rework, include previous attempts:

```json
{
  "context": {
    "previous_attempts": [
      {
        "attempt_number": 1,
        "outcome": "rejected",
        "feedback": "Inconsistent character voice in dialogue",
        "artifacts_produced": [
          { "type": "draft", "id": "draft_v1" }
        ]
      }
    ]
  }
}
```

### Rework Limits

Playbooks define `max_rework_cycles` (default: 3). After exceeding:

- Escalate to orchestrator
- Orchestrator may: adjust requirements, change agent, or report to customer

---

## Subprocess Delegation Pattern

**Problem**: Complex work may require invoking another playbook.

**Solution**: Steps can delegate to subprocesses.

### Step with Delegation

```yaml
- id: research_background
  action: "Research relevant background for this piece"
  delegation:
    playbook: research_workflow
    input_mapping:
      query: "{{ task.topic }}"
      depth: "comprehensive"
    output_mapping:
      findings: research_results
    wait: true
```

### Input/Output Mapping

Maps current context to subprocess inputs, and subprocess outputs back:

```
Current context → input_mapping → Subprocess inputs
Subprocess outputs → output_mapping → Current context
```

### Wait Behavior

| Wait | Behavior |
|------|----------|
| `true` | Block until subprocess completes |
| `false` | Continue; results arrive later |

For `wait: false`, the orchestrator receives a notification when the subprocess completes.
