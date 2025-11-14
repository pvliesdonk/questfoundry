---
procedure_id: human_question
description: Protocol for escalating questions to human customer
version: 2.0.0
references_expertises:
  - showrunner_orchestration
references_schemas:
  - message_envelope.schema.json
references_roles:
  - all
tags:
  - escalation
  - human-interaction
  - protocol
---

# Human Question Procedure

## Overview

Formal protocol for escalating questions to the human customer when agent needs clarification, approval, or decision on ambiguous matters.

## Hard Rule

**NEVER invent your own escalation format.** Always use the `human.question` protocol intent defined in Layer 4.

## When to Ask Human

**Appropriate triggers:**

- Ambiguity that blocks progress (tone, stakes, constraints unclear)
- Forking choices that change scope or style
- Trade-offs requiring creative judgment
- Facts best provided by author (character motivations, world rules)
- Policy uncertainty or conflicting quality bars
- Major changes affecting published content

**Do NOT ask for:**

- Routine decisions covered by existing specs
- Technical implementation details
- Process questions (consult specs/documentation)
- Preference without material impact

## Step 1: Identify Question

Formulate specific, answerable question.

**Good questions:**

- "Should Kestrel's backstory reveal happen in Chapter 2 or defer to Chapter 4?"
- "Which tone for this scene: horror or mystery?"
- "Is this spoiler acceptable in codex entry, or too revealing?"

**Poor questions:**

- "What should I do?" (too vague)
- "Is this good?" (seeking validation, not decision)
- "How do I implement X?" (technical, not creative)

**Actions:**

1. Identify specific decision point
2. Frame question clearly
3. Determine if answerable with options

## Step 2: Prepare Context

Provide minimal but sufficient context for human to decide.

**Context structure:**

- **What changed:** Trigger for this question
- **What's needed:** Specific decision required
- **Why it matters:** Impact on story/quality/scope

**Example:**

```
Context: Kestrel's backstory canon is ready. Decision needed on reveal timing.

Impact: Chapter 2 reveal supports early character depth but risks pacing.
        Chapter 4 reveal maintains mystery longer but delays payoff.

Recommendation: Chapter 4 for stronger dramatic timing.
```

**Keep it concise:** 2-4 sentences max.

## Step 3: Provide Options

Offer 2-4 concrete choices when possible.

**Option structure:**

- **Key:** Short identifier (A, B, C or 1, 2, 3)
- **Label:** Clear, descriptive text
- **Implication:** Brief consequence note

**Example:**

```json
"options": [
  {
    "key": "A",
    "label": "Reveal in Chapter 2 (early character depth)",
    "implication": "Supports player connection but reduces mystery"
  },
  {
    "key": "B",
    "label": "Defer to Chapter 4 (maintain mystery)",
    "implication": "Stronger dramatic timing, longer payoff"
  },
  {
    "key": "C",
    "label": "Progressive hints (Chapter 2) + full reveal (Chapter 4)",
    "implication": "Best of both, but requires additional writing"
  }
]
```

**Include:**

- Safe default option
- Free text option if appropriate: `{"key": "other", "label": "Specify custom approach"}`

**For open-ended questions:**

- Provide empty `options: []` array
- Expect free-text response

## Step 4: Construct Protocol Envelope

Build valid `human.question` message.

**Envelope structure:**

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-YYYYMMDD-HHMMSS-<role><nnn>",
  "time": "2025-11-06T10:30:00Z",
  "sender": "<role_abbreviation>",
  "receiver": "human",
  "intent": "human.question",
  "context": {
    "tu": "TU-2025-11-06-LW01",
    "loop": "lore_deepening"
  },
  "safety": {
    "player_safe": false,
    "sot": "hot"
  },
  "payload": {
    "type": "question",
    "data": {
      "question_text": "<your question>",
      "context_summary": "<brief context>",
      "options": [ /* array of option objects */ ],
      "recommendation": "<your suggested choice, if any>"
    }
  }
}
```

**Required fields:**

- `protocol`, `id`, `time`, `sender`, `receiver`, `intent`
- `payload.type = "question"`
- `payload.data.question_text`

**Optional but recommended:**

- `context.tu` - Link to active trace unit
- `payload.data.context_summary` - Background
- `payload.data.options` - Suggested answers
- `payload.data.recommendation` - Your preference

## Step 5: Pause and Wait

Stop current work and wait for human response.

**Actions:**

1. Emit `human.question` envelope
2. Pause task execution
3. Do NOT proceed with guesses or assumptions
4. Do NOT emit placeholder acknowledgments

**System behavior:**

- Intercepts JSON envelope
- Presents question to human
- Returns `human.response` when answered

## Step 6: Receive and Apply Response

Process `human.response` and continue work.

**Response structure:**

```json
{
  "intent": "human.response",
  "sender": "human",
  "receiver": "<original_sender>",
  "context": {
    "reply_to": "msg-YYYYMMDD-HHMMSS-<role><nnn>",
    "correlation_id": "<original_message_id>"
  },
  "payload": {
    "type": "answer",
    "data": {
      "choice": "B",
      "free_text": "Actually, let's do progressive hints in Ch2, full reveal in Ch4"
    }
  }
}
```

**Interpretation priority:**

1. **If `choice` present:** Use selected option
2. **If `free_text` present:** Interpret custom answer
3. **If both:** Prefer `choice` unless `choice = "other"` then use `free_text`

**Actions:**

1. Apply answer immediately to current work
2. If answer changes scope, emit `tu.update`
3. Continue with updated direction
4. No need for explicit acknowledgment (just proceed)

## Question Batching

When multiple questions arise, batch efficiently.

**Batching strategy:**

1. **Independent questions:** Ask separately (parallel is fine)
2. **Dependent questions:** Ask first, then ask follow-ups based on answer
3. **Related questions:** Combine into single question with compound options

**Example of combining:**
Instead of:

- Q1: "Which tone: horror or mystery?"
- Q2: "Should we reveal backstory early or late?"

Combine:

- Q: "Tone and reveal timing?"
  - A: "Horror tone, early reveal"
  - B: "Horror tone, late reveal"
  - C: "Mystery tone, early reveal"
  - D: "Mystery tone, late reveal"

**Avoid:** Overwhelming human with 5+ questions at once.

## Escalation Levels

Different levels for different severity.

**L1: Clarification (minor)**

- Single question
- No artifact blockage
- Prefer `human.question` with options
- Example: "Which phrasing do you prefer?"

**L2: Artifact Risk (moderate)**

- Quality bar could slip
- Notify Showrunner
- May need specialist role wake
- Example: "Style inconsistency detected, coordinate with Style Lead?"

**L3: Blocker (major)**

- Cannot proceed
- Request Gatekeeper review
- Include `tu.checkpoint` summary
- Example: "Canon contradiction blocks Lore Deepening, need resolution"

## Timeout Handling

Set reasonable expectations for response time.

**Fast track (minutes):** Simple preference questions
**Normal (hours-days):** Creative decisions, scope changes
**Slow track (days-weeks):** Major retcons, controversial changes

**If timeout concerns:**

- Note in context: "Time-sensitive: affects current session"
- Or provide fallback: "Will defer to Chapter 4 if no response by EOD"

## Common Patterns

### Tone Ambiguity

```json
{
  "question_text": "This scene feels ambiguous. Horror or mystery tone?",
  "options": [
    {"key": "horror", "label": "Horror (dread, visceral imagery)"},
    {"key": "mystery", "label": "Mystery (intrigue, puzzle focus)"},
    {"key": "both", "label": "Blend both tones"}
  ]
}
```

### Scope Decision

```json
{
  "question_text": "Canonizing this hook revealed it needs new location. Expand scope or defer?",
  "context_summary": "Guild Hall mentioned but not in topology",
  "options": [
    {"key": "expand", "label": "Add Guild Hall to current TU"},
    {"key": "defer", "label": "Note for future Story Spark"},
    {"key": "remove", "label": "Revise canon to avoid new location"}
  ],
  "recommendation": "defer"
}
```

### Trade-Off

```json
{
  "question_text": "Quality vs speed trade-off for this loop?",
  "options": [
    {"key": "quality", "label": "Full quality pass (2-3 hours)"},
    {"key": "speed", "label": "Fast iteration, address in Style Tune-up later"},
    {"key": "balanced", "label": "Core quality now, polish later"}
  ]
}
```

## Integration with Showrunner

If you're not Showrunner, your `human.question` goes to SR first:

**Your envelope:**

```json
{
  "sender": "LW",
  "receiver": "SR",  // NOT "human" directly
  "intent": "human.question",
  // ... rest of envelope
}
```

**Showrunner responsibilities:**

- Review question for clarity
- Add additional context if needed
- Forward to human with `sender: "SR"`, original question in payload
- Route response back to you with `correlation_id`

## Summary Checklist

- [ ] Question is specific and answerable
- [ ] Context is minimal but sufficient
- [ ] 2-4 concrete options provided (or open-ended justified)
- [ ] Recommendation included if you have preference
- [ ] Protocol envelope properly formatted
- [ ] Work paused until response received
- [ ] Response applied immediately upon receipt
- [ ] Scope changes documented via `tu.update` if needed

**Human questions enable collaborative decision-making while maintaining formal protocol.**
