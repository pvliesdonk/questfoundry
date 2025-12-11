# Scene Weave Loop

> **Goal:** Transform structural topology into living narrative prose.

The **Scene Weave** loop handles the prose drafting phase where Plotwright's structural shells become player-facing narrative. It integrates canon references, presents choices contrastively, and maintains voice consistency.

:::{loop-meta}
id: scene_weave
name: "Scene Weave"
trigger: topology_ready
entry_point: showrunner
version: 1
:::

## Guidance

This section provides operational context for executing the Scene Weave workflow.

### When to Trigger

Invoke the Scene Weave loop when:

- **Post-topology**: Plotwright has completed structural design (scenes, gates, choices)
- **Prose gaps**: Existing scenes need content filled in
- **Style revision**: Prose needs voice/register adjustment
- **Canon integration**: New lore needs to be woven into existing scenes

Do NOT invoke when:

- Structure isn't defined yet (use Story Spark first)
- Only canon facts needed (use Lore Deepening)
- Only player-safe summaries needed (use Codex Expansion)
- Major topology changes needed (go back to Story Spark)

### Success Criteria

The loop succeeds when:

- [ ] All structural shells have prose content
- [ ] Choices are contrastive and clearly communicated
- [ ] Voice matches style guide for the project
- [ ] Canon references are naturally integrated
- [ ] No spoilers in player-facing content
- [ ] Gateway conditions are phrased diegetically

### Common Failure Modes

**Cosmetic choices**

- Symptom: Options differ in wording but lead to same outcome
- Fix: Make choices genuinely different in consequence
- Prevention: Verify each choice maps to distinct state changes

**Voice drift**

- Symptom: Prose tone changes mid-scene or between scenes
- Fix: Audit against style guide; revise for consistency
- Prevention: Reference style addenda before each drafting session

**Thin scenes**

- Symptom: Scenes are too short, lacking sensory grounding
- Fix: Add lead image, goal/friction, choice setup (3-paragraph minimum)
- Prevention: Use paragraph cadence checklist

**Meta phrasing**

- Symptom: Choices read like UI instructions, not in-world actions
- Fix: Rewrite as diegetic actions the character would take
- Prevention: Self-check for "click", "select", "choose" language

**Gateway leaks**

- Symptom: Internal state variables appear in player text
- Fix: Rephrase using in-world reasoning
- Prevention: Review all conditional text for PN-safety

**Over-exposition**

- Symptom: Prose explains lore that should live in Codex
- Fix: Remove explanation; trust Codex for world facts
- Prevention: Check if detail serves story or just informs

## Loop Participants

The roles that participate in this loop with their operational parameters.

:::{loop-participants}
showrunner:
  timeout: 300
  max_iterations: 5
scene_smith:
  timeout: 900
  max_iterations: 15
creative_director:
  timeout: 300
  max_iterations: 5
lorekeeper:
  timeout: 300
  max_iterations: 5
gatekeeper:
  timeout: 300
  max_iterations: 3
:::

## Routing Rules

Decision table for Showrunner: after a role completes work, these rules
describe when to delegate to the next role.

### After Showrunner

:::{routing-rule}
after: showrunner
when: brief_created
delegate_to: scene_smith
description: SR creates prose brief; SS drafts content
:::

:::{routing-rule}
after: showrunner
when: terminate
delegate_to: END
description: Scene weaving complete; end the loop
:::

### After Scene Smith

:::{routing-rule}
after: scene_smith
when: needs_canon
delegate_to: lorekeeper
description: SS needs canon facts for accurate prose
:::

:::{routing-rule}
after: scene_smith
when: needs_style_check
delegate_to: creative_director
description: SS has voice questions; CD provides guidance
:::

:::{routing-rule}
after: scene_smith
when: draft_complete
delegate_to: gatekeeper
description: Prose drafted; GK validates style and safety
:::

:::{routing-rule}
after: scene_smith
when: escalation
delegate_to: showrunner
description: SS encounters blocking issue requiring SR decision
:::

### After Lorekeeper

:::{routing-rule}
after: lorekeeper
when: canon_provided
delegate_to: scene_smith
description: Canon callbacks ready; SS continues prose work
:::

### After Creative Director

:::{routing-rule}
after: creative_director
when: style_guidance_provided
delegate_to: scene_smith
description: Style guidance given; SS applies to prose
:::

:::{routing-rule}
after: creative_director
when: escalation
delegate_to: showrunner
description: CD encounters voice conflict requiring SR decision
:::

### After Gatekeeper

:::{routing-rule}
after: gatekeeper
when: failed
delegate_to: scene_smith
description: Validation failed; SS revises prose
:::

:::{routing-rule}
after: gatekeeper
when: passed
delegate_to: showrunner
description: All bars passed; SR approves for merge
:::

## Quality Gates

### Pre-Approval Validation

:::{quality-gate}
before: gatekeeper
role: gatekeeper
bars:

- style
- presentation
- accessibility
blocking: true
:::

## Expected Flow

```text
Topology Ready
    |
[Showrunner] -> creates prose brief
    |
[Scene Smith] -> drafts prose
    | (if canon needed)
[Lorekeeper] -> provides callbacks
    | (if style questions)
[Creative Director] -> provides guidance
    |
[Gatekeeper] -> validates prose
    | (if passed)
[Showrunner] -> approves for merge
```

## Artifacts Produced

- **Scene** (updated): Prose content added to structural shell
- **Edit Notes**: Documentation of changes for revision tracking
- **HookCard**: Hooks for gaps, callbacks, or art/audio cues

## Prose Cadence Guidelines

### Default Scene Structure (3+ paragraphs)

1. **Lead image + motion**: Opening sensory details and action
2. **Goal/vector + friction**: Character intent and obstacles
3. **Choice setup**: Context for upcoming decision

### Micro-beats

Transit-only passages between scenes may be 1 paragraph if explicitly designated. The next full scene must then carry reflection and affordances.

### Self-Check Before Handoff

- [ ] Voice/register matches style guide
- [ ] Paragraph consistency (voice doesn't waver)
- [ ] Choices are contrastive (meaningfully different)
- [ ] No meta phrasing in choices
- [ ] Gateway hints are PN-safe (no codewords)
- [ ] Sensory details present for grounding

## Handoffs

After Scene Weave:

- **Canon Commit**: Approved prose ready for cold_store
- **Publisher**: Prose ready for export pipelines
- **Codex Expansion**: Terms in prose may need codex entries
