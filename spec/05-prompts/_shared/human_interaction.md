# Human Interaction Protocol

When you (as an agent) need to escalate to the human customer, you must use the single, formal
protocol intent: `human.question`.

This protocol is defined in `04-protocol/INTENTS.md` and MUST be used for all escalations, whether
for clarification or approval.

## How to Escalate

**NEVER** invent your own escalation format (like "Waiting for user input...").

Instead, you **MUST** pause your current task and output a full, valid protocol envelope (as a JSON
object) with the following structure.

### Example: Showrunner needs to clarify intent

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-20251112-093000-sr123",
  "time": "2025-11-12T09:30:00Z",
  "sender": "SR",
  "receiver": "human",
  "intent": "human.question",
  "context": {
    "tu_id": "TU-20251112-SR01"
  },
  "safety": {
    "player_safe": false,
    "sot": "hot"
  },
  "payload": {
    "type": "question",
    "data": {
      "question_text": "I see you want to work on 'worldbuilding.' Which did you mean?",
      "options": [
        {
          "key": "A",
          "label": "Proactively build a new part of the world (World Genesis)"
        },
        {
          "key": "B",
          "label": "Fleshen out an existing idea (Lore Deepening)"
        }
      ]
    }
  }
}
```

## Payload Schema

The payload `data` object for a `human.question` intent is defined in `04-protocol/INTENTS.md` as:

- `question_text` (string, required): The question you are asking.
- `options` (array, optional): A list of suggested answers.
  - `key` (string, required): A short key (e.g., "A", "B", "1").
  - `label` (string, required): The descriptive text for the option.

If you are asking an open-ended question, provide an empty `options` array.

The system will intercept this JSON envelope, present the question to the human, and send their
answer back to you under the `human.response` intent.

## When to Ask

- Ambiguity that blocks progress (tone, stakes, constraints).
- Forking choices that change scope or style.
- Facts best provided by the author.

## How to Ask

- One specific question per envelope.
- Include terse context (what changed, what's needed) and 2–4 concrete options.
- Offer a "free text" option and a safe default.

## Interpreting Answers

- If `choice` is present, prefer it over free text.
- Apply the answer immediately and emit a checkpoint if it changes scope.
- Use `ack` sparingly; prefer making progress with the new information.

## References

- `04-protocol/INTENTS.md` — Formal intent definitions
- `04-protocol/EXAMPLES/human.question.json` — Example envelope
- `04-protocol/EXAMPLES/human.response.json` — Example response
