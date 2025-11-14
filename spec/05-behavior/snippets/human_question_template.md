# Human Question Template

Use this structure when asking human for decisions:

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-YYYYMMDD-HHMMSS-<role><nnn>",
  "time": "2025-11-06T10:30:00Z",
  "sender": "<role_abbreviation>",
  "receiver": "human",
  "intent": "human.question",
  "context": {
    "tu": "TU-YYYY-MM-DD-<ROLE><NN>",
    "loop": "<loop_name>"
  },
  "safety": {
    "player_safe": false,
    "sot": "hot"
  },
  "payload": {
    "type": "question",
    "data": {
      "question_text": "<specific question>",
      "context_summary": "<brief 2-3 sentence context>",
      "options": [
        {
          "key": "A",
          "label": "<option 1 description>"
        },
        {
          "key": "B",
          "label": "<option 2 description>"
        }
      ],
      "recommendation": "<your suggested choice, if any>"
    }
  }
}
```

**When to use:**

- Ambiguity blocks progress
- Creative decisions requiring author input
- Trade-offs needing human judgment

**Do NOT invent your own escalation format.** Always use this protocol.

**Refer to:** `@procedure:human_question` for complete guidance.
