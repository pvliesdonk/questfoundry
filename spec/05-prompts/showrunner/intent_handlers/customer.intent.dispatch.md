# Intent Handler — customer.intent.dispatch

**Layer:** Prompt-level (Layer 5) — This is NOT a Layer 4 protocol intent. This handler is internal
to the Showrunner prompt and is used to route customer commands to the appropriate loop playbook.

## Purpose

To receive a freeform text message from the human customer, interpret their intent, and select the
correct `loop_id` and `config` parameters to execute.

## Inputs

- Freeform text message from the human customer (e.g., "Write the next scene", "Build the book as
  epub")
- Current project state (TUs, hooks, manifest)
- Full knowledge of all available studio playbooks in `00-north-star/PLAYBOOKS/` and
  `05-prompts/loops/`

## Preconditions

- You have access to the project's current state (TUs, hooks) to make informed decisions.
- You MUST adhere to the `_shared/human_interaction.md` guide for all escalations.

## Process

1. **Analyze Intent:** Read the customer's message and determine what action they want to perform.
2. **Map to Loop:** Select the _one_ `loop_id` that best fulfills the request.
   - Use `snake_case` for standard loops (e.g., `story_spark`, `art_touch_up`)
   - Use `dotted.notation` for special canon workflows (e.g., `canon.genesis.create`,
     `canon.transfer.import`, `canon.transfer.export`)
3. **Extract Parameters:** Parse the message for any config parameters (e.g., "build the book as an
   epub" → `{"format": "epub"}`).
4. **Check for Ambiguity:**
   - **If Ambiguous:** DO NOT guess. You MUST escalate by outputting a **full JSON envelope** with
     the `human.question` intent (see Outputs section below).
   - **If Clear:** Proceed to step 5.
5. **Formulate Response:** Respond with a **JSON object** (not an envelope) containing the `loop_id`
   to run and any extracted `config` parameters.

## Outputs

### Success (Clear Intent)

Return a JSON object (not an envelope) with the dispatch decision:

```json
{
  "decision": "dispatch_loop",
  "loop_id": "story_spark",
  "config": {
    "customer_intent": "Write the next scene",
    "auto": false
  }
}
```

**Fields:**

- `decision` (string, required): Always `"dispatch_loop"`
- `loop_id` (string, required): The loop to execute (snake_case or dotted.notation)
- `config` (object, required): Configuration parameters for the loop
  - `customer_intent` (string, required): Original customer message for traceability
  - `auto` (boolean): Set to `false` by default unless customer explicitly requests automated
    execution
  - Additional fields as needed (e.g., `format`, `scope`, `target_hook`)

### Success (Canon Workflow)

For special canon workflows, use dotted notation:

```json
{
  "decision": "dispatch_loop",
  "loop_id": "canon.genesis.create",
  "config": {
    "customer_intent": "Start a new fantasy project by building the world first.",
    "auto": true,
    "concept": "A new fantasy project, world-first."
  }
}
```

### Ambiguous (Must Escalate)

When the intent is ambiguous, output a full JSON envelope with `human.question` intent:

```json
{
  "protocol": "questfoundry/1.0.0",
  "id": "msg-20251112-093000-sr123",
  "time": "2025-11-12T09:30:00Z",
  "sender": "SR",
  "receiver": "human",
  "intent": "human.question",
  "context": {
    "tu_id": null
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

The system will intercept this envelope, present the question to the user, and send back a
`human.response` with their choice.

## Loop Selection Guidelines

Use these guidelines to map customer intent to the correct loop:

- **Plot/Story work** → `story_spark`
  - "Write the next scene", "Continue the story", "Add a new chapter"
- **World expansion (existing)** → `lore_deepening`
  - "Expand on the Shadow Syndicate hook", "Deepen the magic system"
- **World creation (new project)** → `canon.genesis.create`
  - "Start a new project with world-first approach", "Build the world before the plot"
- **Book building/export** → `binding_run`
  - "Build the book", "Export as epub", "Generate the final manuscript"
- **Style/prose refinement** → `style_tune_up`
  - "Improve the writing style", "Make it more dramatic", "Adjust tone"
- **Art/visual work** → `art_touch_up`
  - "Add some art", "Create illustrations", "Design cover art"
- **Audio work** → `audio_pass`
  - "Add narration", "Create soundscapes", "Generate audio"
- **Review/triage hooks** → `hook_harvest`
  - "Review the hooks", "Triage story ideas", "See what's pending"
- **Player-safe documentation** → `codex_expansion`
  - "Add a codex entry", "Document the world for players", "Create a glossary"
- **Quality checks** → `gatecheck`
  - "Run quality check", "Verify the artifact", "Check for issues"
- **Narration testing** → `narration_dry_run`
  - "Test the narration", "Playtest this scene", "Check player experience"
- **Translation** → `translation_pass`
  - "Translate to Spanish", "Localize for French", "Create language pack"
- **Archival** → `archive_snapshot`
  - "Create a snapshot", "Archive the current state", "Save a checkpoint"
- **Retrospective** → `post_mortem`
  - "Run a retrospective", "Review what worked", "Post-mortem analysis"

## Ambiguity Examples

When the customer's intent could map to multiple loops, escalate with `human.question`:

- **"worldbuilding"** → Could be `canon.genesis.create` OR `lore_deepening`
  - Ask: "Do you want to proactively build a new part of the world (World Genesis) or expand on an
    existing idea (Lore Deepening)?"
- **"improve the writing"** → Could be `style_tune_up` OR `story_spark` (rewrite)
  - Ask: "Do you want to refine the prose style (Style Tune-up) or rewrite content (Story Spark)?"
- **"add content"** → Could be `story_spark`, `lore_deepening`, OR `codex_expansion`
  - Ask: "What type of content? Story (new scenes), World (lore), or Documentation (codex)?"

## Errors

- **Unknown loop requested:** If the customer mentions a loop name that doesn't exist, escalate with
  `human.question` to clarify.
- **Missing context:** If you need project state but it's unavailable, emit
  `error(validation_error)` with details.

## Notes

- Always include the original `customer_intent` in the `config` for traceability.
- Set `auto: false` by default unless the customer explicitly requests automated execution.
- For special canon workflows (e.g., `canon.genesis.create`), refer to
  `00-north-star/LAYER6_7_CANON_IMPACT.md` for additional context.

## References

- Loop playbooks: `05-prompts/loops/*.playbook.md`
- North Star playbooks: `00-north-star/PLAYBOOKS/playbook_*.md`
- Human interaction pattern: `05-prompts/_shared/human_interaction.md`
- Layer 4 protocol: `04-protocol/INTENTS.md`
