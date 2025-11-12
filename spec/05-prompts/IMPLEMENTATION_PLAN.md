# Layer 5 Implementation Plan

**Target Agent:** GPT-5 (primary) — Claude Sonnet 3.5+ acceptable **Repository:**
`pvliesdonk/questfoundry` (spec repo, `05-prompts/` directory) **Output:** Markdown prompt files +
JSON examples

## Overview

Layer 5 requires sophisticated prompt engineering and understanding of:

- Role charters and mission statements (Layer 1)
- Artifact structures and validation rules (Layer 2/3)
- Protocol communication patterns (Layer 4)
- Quality bars and safety principles (Layer 0)

This is high-stakes work requiring deep context and nuanced understanding - perfect for Claude.

---

## Epic 1: Shared Foundations

**Goal:** Create shared prompt patterns used by all roles.

**Dependencies:** Layer 0-4 complete ✅

**Estimated Effort:** 3-4 days

### Features

#### 1.1: Context Management Pattern

**File to create:** `05-prompts/_shared/context_management.md`

**Content to include:**

- How agents track current TU
- How to maintain awareness of hot vs cold state
- How to track active snapshot
- How to maintain conversation history
- When to reference previous artifacts
- How to handle context window limits

**Acceptance Criteria:**

- Clear instructions for maintaining context
- Examples of referencing TU context
- Guidance on conversation history management

**AI Agent Instructions:**

```
Reference Layer 4 protocol context fields.
Study how TUs link artifacts.
Review Layer 0 traceability principles.
Create comprehensive guidance for any role agent.
```

---

#### 1.2: Safety Protocol Pattern

**File to create:** `05-prompts/_shared/safety_protocol.md`

**Content to include:**

- PN boundary enforcement rules
- Spoiler hygiene requirements
- Hot vs Cold awareness
- When to flag player_safe=false
- How to avoid codewords/mechanics in player surfaces
- Diegetic enforcement principles

**Acceptance Criteria:**

- Clear safety rules
- Examples of safe vs unsafe content
- Guidance for all player-facing artifacts

**AI Agent Instructions:**

```
Study Layer 0 PN_PRINCIPLES.md carefully.
Reference Layer 0 QUALITY_BARS.md (Spoiler Hygiene bar).
Review Layer 4 envelope safety fields.
Create zero-ambiguity safety rules.
```

---

#### 1.3: Escalation Rules Pattern

**File to create:** `05-prompts/_shared/escalation_rules.md`

**Content to include:**

- When to wake Showrunner
- When to request human intervention
- When to signal dormancy
- When to escalate to Gatekeeper
- Error handling patterns

**Acceptance Criteria:**

- Clear escalation triggers
- Examples of escalation messages
- Decision trees for complex situations

**AI Agent Instructions:**

```
Reference Layer 1 role interfaces.
Study Layer 0 RACI matrices for escalation paths.
Review Layer 4 intent patterns.
```

---

#### 1.4: Human Interaction Pattern

**File to create:** `05-prompts/_shared/human_interaction.md`

**Content to include:**

- When to ask human questions
- How to phrase questions (clear, actionable)
- How to provide context with questions
- How to offer suggestions
- How to handle free-form answers
- When NOT to ask (avoid chatty agents)

**Acceptance Criteria:**

- Guidelines for question frequency
- Examples of good vs bad questions
- Patterns for interpreting answers

**AI Agent Instructions:**

```
Create guidance for collaborative AI-human workflow.
Balance autonomy with human input.
Examples of good question framing.
```

---

## Epic 2: Orchestration Roles

**Goal:** Implement Showrunner and Gatekeeper - the coordination layer.

**Dependencies:** Epic 1

**Estimated Effort:** 5-6 days

### Features

#### 2.1: Showrunner System Prompt

**File to create:** `05-prompts/showrunner/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Coordinate loops, wake roles, manage TUs
- How to parse loop definitions
- How to sequence role actions
- How to create TUs
- How to handle checkpoints
- When to request human approval
- Error recovery strategies

**Acceptance Criteria:**

- Complete system prompt that implements charter
- References all shared patterns
- Clear loop orchestration logic

**AI Agent Instructions:**

```
Study Layer 1 showrunner charter deeply.
Reference all 13 loops from Layer 0 LOOPS/.
Review Layer 4 FLOWS for message sequences.
This is the "conductor" role - needs big picture view.
Create comprehensive orchestration instructions.
```

---

#### 2.2: Showrunner Intent Handlers

**Files to create:**

```
05-prompts/showrunner/intent_handlers/
  tu.open.md
  tu.close.md
  role.wake.md
  role.dormant.md
  tu.checkpoint.md
  human.question.md
```

**Content per file:**

- How to handle specific intent
- Input validation
- Processing logic
- Output generation
- Error cases

**Acceptance Criteria:**

- All Showrunner intents covered
- Clear processing logic per intent

**AI Agent Instructions:**

```
Reference Layer 4 INTENTS.md for Showrunner intents.
Each handler should be clear step-by-step logic.
Include validation and error handling.
```

---

#### 2.3: Showrunner Examples

**Files to create:**

```
05-prompts/showrunner/examples/
  hook_harvest_coordination.json
  gatecheck_orchestration.json
  checkpoint_approval.json
```

**Content:**

- Multi-message conversation examples
- Show full envelope sequences
- Demonstrate decision-making

**Acceptance Criteria:**

- Realistic conversation examples
- Valid Layer 4 envelopes
- Cover common and edge cases

---

#### 2.4: Gatekeeper System Prompt

**File to create:** `05-prompts/gatekeeper/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Enforce quality bars
- How to run each quality bar check
- How to generate gatecheck reports
- When to block hot→cold promotion
- How to provide actionable feedback

**Acceptance Criteria:**

- Complete system prompt
- Integration with 8 quality bars
- Clear pass/fail logic

**AI Agent Instructions:**

```
Study Layer 1 gatekeeper charter.
Master all 8 quality bars from Layer 0 QUALITY_BARS.md.
Reference Layer 4 gatecheck flow.
This role is the "quality guardian" - must be rigorous.
```

---

#### 2.5: Gatekeeper Quality Bar Checks

**Files to create:**

```
05-prompts/gatekeeper/quality_bars/
  integrity.md
  reachability.md
  nonlinearity.md
  gateways.md
  style.md
  determinism.md
  presentation.md
  spoiler_hygiene.md
```

**Content per file:**

- What to check
- How to validate
- Common violations
- How to report findings

**Acceptance Criteria:**

- All 8 bars have detailed check logic
- Clear, actionable violation reports

**AI Agent Instructions:**

```
Each quality bar from Layer 0 needs checking logic.
Be specific about what passes/fails.
Provide helpful guidance for fixing violations.
```

---

#### 2.6: Gatekeeper Examples

**Files to create:**

```
05-prompts/gatekeeper/examples/
  passing_gatecheck.json
  failing_integrity.json
  failing_style.json
```

**Content:**

- Example gatecheck conversations
- Show report generation
- Demonstrate feedback

---

## Epic 3: Core Content Roles

**Goal:** Implement Lore Weaver and Scene Smith - the content creators.

**Dependencies:** Epic 2

**Estimated Effort:** 5-6 days

### Features

#### 3.1: Lore Weaver System Prompt

**File to create:** `05-prompts/lore_weaver/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Turn hooks into canon
- How to analyze hooks
- How to create canon packs
- How to maintain continuity
- How to identify implications
- When to coordinate with Codex Curator
- How to handle stakes

**Acceptance Criteria:**

- Complete system prompt
- Clear canonization process
- Continuity checking logic

**AI Agent Instructions:**

```
Study Layer 1 lore_weaver charter and brief.
Understand hook_card and canon_pack artifacts deeply.
Reference Layer 4 lore_deepening flow.
This role builds the story universe - needs creative depth.
```

---

#### 3.2: Lore Weaver Intent Handlers

**Files to create:**

```
05-prompts/lore_weaver/intent_handlers/
  hook.canonize.md
  canon.create.md
  canon.revise.md
```

**Content:**

- How to process each intent
- Artifact transformation logic
- Validation steps

---

#### 3.3: Lore Weaver Examples

**Files to create:**

```
05-prompts/lore_weaver/examples/
  hook_to_canon.json
  continuity_check.json
  stakes_analysis.json
```

**Content:**

- Example canonization conversations
- Show artifact creation
- Demonstrate continuity reasoning

---

#### 3.4: Scene Smith System Prompt

**File to create:** `05-prompts/scene_smith/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Write section prose
- How to interpret TU briefs
- How to maintain voice/register
- How to integrate canon
- How to handle player choices
- When to coordinate with Style Lead
- Prose quality standards

**Acceptance Criteria:**

- Complete system prompt
- Clear prose generation process
- Style consistency guidance

**AI Agent Instructions:**

```
Study Layer 1 scene_smith charter and brief.
Understand tu_brief artifact structure.
This role is the "prose writer" - needs literary skill.
Reference Layer 0 QUALITY_BARS.md Style bar.
```

---

#### 3.5: Scene Smith Intent Handlers

**Files to create:**

```
05-prompts/scene_smith/intent_handlers/
  scene.write.md
  scene.revise.md
```

**Content:**

- Scene generation process
- Revision handling
- Quality self-checks

---

#### 3.6: Scene Smith Examples

**Files to create:**

```
05-prompts/scene_smith/examples/
  scene_creation.json
  style_consistency.json
  revision_request.json
```

**Content:**

- Example scene writing conversations
- Show prose generation
- Demonstrate style adherence

---

## Epic 4: Structure & Discovery Roles

**Goal:** Implement Plotwright and Codex Curator - topology and player knowledge.

**Dependencies:** Epic 3

**Estimated Effort:** 4-5 days

### Features

#### 4.1: Plotwright System Prompt

**File to create:** `05-prompts/plotwright/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Design topology (hubs/loops/gateways)
- How to create nonlinear structures
- How to define gateway conditions
- How to ensure reachability
- How to balance linearity vs freedom
- Integration with hooks and canon

**Acceptance Criteria:**

- Complete system prompt
- Topology design guidance
- Gateway definition logic

**AI Agent Instructions:**

```
Study Layer 1 plotwright charter.
Master Layer 0 Nonlinearity and Gateways quality bars.
This role designs the "story graph" - needs systemic thinking.
```

---

#### 4.2: Plotwright Intent Handlers & Examples

**Files to create:**

```
05-prompts/plotwright/intent_handlers/
  topology.design.md
  gateway.define.md
05-prompts/plotwright/examples/
  hub_design.json
  gateway_definition.json
```

---

#### 4.3: Codex Curator System Prompt

**File to create:** `05-prompts/codex_curator/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Publish player-safe lore entries
- How to transform canon into codex entries
- How to enforce player-safe boundaries
- How to handle progressive reveal
- How to avoid spoilers
- When entries unlock

**Acceptance Criteria:**

- Complete system prompt
- Player-safety rigor
- Spoiler prevention logic

**AI Agent Instructions:**

```
Study Layer 1 codex_curator charter.
Master spoiler hygiene principles from Layer 0.
This role protects player experience - zero tolerance for leaks.
```

---

#### 4.4: Codex Curator Intent Handlers & Examples

**Files to create:**

```
05-prompts/codex_curator/intent_handlers/
  codex.create.md
  codex.revise.md
05-prompts/codex_curator/examples/
  canon_to_codex.json
  spoiler_strip.json
```

---

## Epic 5: Style & Quality Roles

**Goal:** Implement Style Lead and supporting QA roles.

**Dependencies:** Epic 4

**Estimated Effort:** 3-4 days

### Features

#### 5.1: Style Lead System Prompt

**File to create:** `05-prompts/style_lead/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Maintain voice/register/motifs
- How to define style guides
- How to audit for consistency
- How to provide style feedback
- How to track motifs
- Register mapping

**Acceptance Criteria:**

- Complete system prompt
- Style analysis guidance
- Feedback patterns

**AI Agent Instructions:**

```
Study Layer 1 style_lead charter.
Reference Layer 0 Style quality bar.
Understand register_map artifact.
This role is the "voice guardian" - needs literary sensitivity.
```

---

#### 5.2: Style Lead Intent Handlers & Examples

**Files to create:**

```
05-prompts/style_lead/intent_handlers/
  style.audit.md
  style.guide.md
05-prompts/style_lead/examples/
  style_audit.json
  register_guidance.json
```

---

#### 5.3: Researcher System Prompt (Optional Role)

**File to create:** `05-prompts/researcher/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Verify facts
- When to activate (dormancy signals)
- How to research claims
- How to document findings
- When to go dormant

**Acceptance Criteria:**

- Complete system prompt
- Dormancy logic clear
- Research methodology

**AI Agent Instructions:**

```
Study Layer 1 researcher charter.
This role is often dormant - needs clear wake/sleep signals.
Focus on fact-checking workflow.
```

---

## Epic 6: Asset Roles

**Goal:** Implement Art Director, Illustrator, Audio Director, Audio Producer.

**Dependencies:** Epic 5

**Estimated Effort:** 5-6 days

### Features

#### 6.1: Art Director System Prompt

**File to create:** `05-prompts/art_director/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Plan visual assets
- How to create shotlists from scenes
- How to define visual style
- How to specify composition, mood, style
- How to maintain visual consistency
- Integration with art_plan artifact

**Acceptance Criteria:**

- Complete system prompt
- Shotlist creation logic
- Visual planning guidance

**AI Agent Instructions:**

```
Study Layer 1 art_director charter.
Understand shotlist artifact structure.
This role plans visuals - needs visual thinking.
```

---

#### 6.2: Art Director Intent Handlers & Examples

**Files to create:**

```
05-prompts/art_director/intent_handlers/
  shotlist.create.md
  art_plan.update.md
05-prompts/art_director/examples/
  scene_to_shotlist.json
```

---

#### 6.3: Illustrator System Prompt

**File to create:** `05-prompts/illustrator/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Generate images from shotlists
- How to interpret shotlist specifications
- How to craft effective image prompts
- How to handle provider-specific parameters
- How to review generated images
- Quality criteria

**Acceptance Criteria:**

- Complete system prompt
- Prompt engineering guidance
- Quality assessment logic

**AI Agent Instructions:**

```
Study Layer 1 illustrator charter.
Understand shotlist → image workflow.
This role executes visual generation - needs prompt engineering skill.
```

---

#### 6.4: Illustrator Intent Handlers & Examples

**Files to create:**

```
05-prompts/illustrator/intent_handlers/
  image.generate.md
  image.revise.md
05-prompts/illustrator/examples/
  shotlist_to_prompt.json
```

---

#### 6.5: Audio Director System Prompt

**File to create:** `05-prompts/audio_director/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Plan audio assets
- How to create cuelists from scenes
- How to define audio style (music, SFX, voice)
- How to specify mood, instrumentation
- How to maintain audio consistency
- Integration with audio_plan artifact

**Acceptance Criteria:**

- Complete system prompt
- Cuelist creation logic
- Audio planning guidance

**AI Agent Instructions:**

```
Study Layer 1 audio_director charter.
Understand cuelist artifact structure.
This role plans audio - needs audio design thinking.
```

---

#### 6.6: Audio Director & Audio Producer Intent Handlers & Examples

**Files to create:**

```
05-prompts/audio_director/intent_handlers/
  cuelist.create.md
05-prompts/audio_director/examples/
  scene_to_cuelist.json

05-prompts/audio_producer/system_prompt.md
05-prompts/audio_producer/intent_handlers/
  audio.generate.md
05-prompts/audio_producer/examples/
  cuelist_to_audio.json
```

---

## Epic 7: Publication Roles

**Goal:** Implement Translator, Book Binder, Player-Narrator.

**Dependencies:** Epic 6

**Estimated Effort:** 4-5 days

### Features

#### 7.1: Translator System Prompt (Optional Role)

**File to create:** `05-prompts/translator/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Localization
- How to translate while preserving style
- How to handle cultural adaptation
- How to maintain terminology consistency
- When to activate (dormancy signals)
- language_pack artifact structure

**Acceptance Criteria:**

- Complete system prompt
- Translation workflow
- Dormancy logic

**AI Agent Instructions:**

```
Study Layer 1 translator charter.
This role is often dormant.
Focus on localization best practices.
```

---

#### 7.2: Book Binder System Prompt

**File to create:** `05-prompts/book_binder/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Export views
- How to assemble snapshots into views
- How to render to various formats (HTML, Markdown, PDF)
- How to ensure player-safe content only
- How to handle front_matter
- How to create view_log

**Acceptance Criteria:**

- Complete system prompt
- Export format logic
- Player-safety enforcement

**AI Agent Instructions:**

```
Study Layer 1 book_binder charter.
Understand snapshot → view workflow.
This role publishes final output - needs format knowledge.
```

---

#### 7.3: Book Binder Intent Handlers & Examples

**Files to create:**

```
05-prompts/book_binder/intent_handlers/
  view.export.md
  format.render.md
05-prompts/book_binder/examples/
  html_export.json
  markdown_export.json
```

---

#### 7.4: Player-Narrator System Prompt

**File to create:** `05-prompts/player_narrator/system_prompt.md`

**Content to include:**

- Role charter from Layer 1
- Mission: Perform book in-world
- How to present narration
- How to enforce gateway conditions diegetically
- How to respond to player choices
- How to access codex
- How to maintain spoiler boundaries
- How to track player state

**Acceptance Criteria:**

- Complete system prompt
- Performance guidelines
- Gateway enforcement logic

**AI Agent Instructions:**

```
Study Layer 1 player_narrator charter deeply.
Master Layer 0 PN_PRINCIPLES.md.
This is the player-facing role - CRITICAL for spoiler safety.
Diegetic enforcement is key.
```

---

#### 7.5: Player-Narrator Intent Handlers & Examples

**Files to create:**

```
05-prompts/player_narrator/intent_handlers/
  narration.perform.md
  gateway.check.md
  choice.present.md
05-prompts/player_narrator/examples/
  narration_session.json
  gateway_enforcement.json
  player_choice.json
```

---

## Epic 8: Testing & Validation

**Goal:** Validate all prompts and create conversation test suites.

**Dependencies:** Epic 7

**Estimated Effort:** 3-4 days

### Features

#### 8.1: Prompt Validation Suite

**Files to create:**

```
05-prompts/tests/
  validate_prompts.py      # Script to validate all prompts
  test_coverage.py         # Check all intents covered
  test_references.py       # Check Layer 0-4 references valid
```

**Tasks:**

- Check all roles have system prompts
- Verify all required intents have handlers
- Validate references to Layer 0-4 documents
- Check shared pattern usage
- Ensure safety guidelines present

**Acceptance Criteria:**

- All prompts structurally valid
- No broken references
- All intents covered

---

#### 8.2: Conversation Test Fixtures

**Files to create:**

```
05-prompts/tests/fixtures/
  hook_harvest_flow.json
  lore_deepening_flow.json
  gatecheck_flow.json
  [... all loops from Layer 4 FLOWS]
```

**Content:**

- Complete conversation sequences
- Valid Layer 4 envelopes
- Expected agent responses
- Cover all 13 loops

**Acceptance Criteria:**

- All loops have test fixtures
- Fixtures follow Layer 4 FLOWS exactly
- Realistic agent behavior

---

#### 8.3: Role Interaction Matrix

**File to create:** `05-prompts/tests/role_interactions.md`

**Content:**

- Document which roles interact
- Common message sequences
- Handoff patterns
- Conflict resolution

**Acceptance Criteria:**

- All role pairs documented
- Clear interaction patterns

---

## Epic 9: Documentation & Examples

**Goal:** Comprehensive documentation for prompt usage and customization.

**Dependencies:** Epic 8

**Estimated Effort:** 2-3 days

### Features

#### 9.1: Prompt Engineering Guide

**File to create:** `05-prompts/PROMPT_ENGINEERING.md`

**Content:**

- How prompts are structured
- How to customize for different LLMs
- How to add new intents
- How to modify behavior
- Best practices
- Common pitfalls

**Acceptance Criteria:**

- Clear customization guide
- Examples of modifications

---

#### 9.2: Role Comparison Matrix

**File to create:** `05-prompts/ROLE_MATRIX.md`

**Content:**

- All 14 core roles in table (translator optional)
- Inputs, outputs, triggers
- Dormancy conditions
- LLM requirements
- Quick reference

**Acceptance Criteria:**

- Complete role overview
- Easy to scan

---

#### 9.3: Migration Guide

**File to create:** `05-prompts/MIGRATION.md`

**Content:**

- How to update prompts without breaking projects
- Versioning strategy
- Backward compatibility
- Deprecation policy

**Acceptance Criteria:**

- Clear migration path
- Version management strategy

---

## Implementation Order Summary

**Phase 1: Foundations**

1. Epic 1: Shared Foundations (all roles depend on this)

**Phase 2: Coordination Layer** 2. Epic 2: Orchestration Roles (Showrunner, Gatekeeper)

**Phase 3: Core Content** 3. Epic 3: Core Content Roles (Lore Weaver, Scene Smith) 4. Epic 4:
Structure & Discovery Roles (Plotwright, Codex Curator)

**Phase 4: Quality & Enhancement** 5. Epic 5: Style & Quality Roles (Style Lead, Researcher)

**Phase 5: Media** 6. Epic 6: Asset Roles (Art Director, Illustrator, Audio roles)

**Phase 6: Publication** 7. Epic 7: Publication Roles (Translator, Book Binder, Player-Narrator)

**Phase 7: Validation** 8. Epic 8: Testing & Validation 9. Epic 9: Documentation & Examples

---

## AI Agent Instructions (for Claude)

### Context to Provide

When working with Claude on Layer 5:

**Essential Context:**

```
You are creating AI agent prompts for the QuestFoundry system.

Background:
- QuestFoundry is a system for creating nonlinear interactive narratives
- Layer 0: North Star (vision, quality bars, loops)
- Layer 1: 14 role charters (human-level descriptions)
- Layer 2: Common language (artifacts, terminology)
- Layer 3: JSON schemas (validation rules)
- Layer 4: Protocol (message envelopes, intents, flows)
- Layer 5 (THIS LAYER): AI prompts that implement roles

Your task: Transform role charters into executable AI agent prompts.

Key principles:
- Protocol-first: All communication via Layer 4 envelopes
- Safety-first: PN boundaries are sacred
- Canon-first: Validate against Layer 3 schemas
- Traceable: Link everything to TUs
- Collaborative: Support human-AI partnership
```

**Before each epic:**

```
Current Epic: [Epic Name]
Reference documents:
- Layer 1: [relevant role charters]
- Layer 0: [relevant quality bars, loops]
- Layer 4: [relevant flows, intents]

Your goal: [epic goal]

Focus on:
1. Faithfully implementing the role charter
2. Clear, actionable instructions for LLM
3. Safety and validation at every step
4. Helpful examples
```

### Quality Checklist

For each prompt, Claude should verify:

- [ ] Implements role charter faithfully
- [ ] References relevant Layer 0-4 documents
- [ ] Includes safety guidelines
- [ ] Includes context management
- [ ] Includes escalation rules
- [ ] Provides clear intent handlers
- [ ] Includes realistic examples
- [ ] Written for LLM audience (not human)
- [ ] Actionable and specific (not vague)
- [ ] Handles error cases

### Best Practices for Claude

**When creating prompts:**

1. **Study the charter deeply:** Read Layer 1 charter multiple times
2. **Understand the artifacts:** Know the schemas inside-out
3. **Follow the flows:** Layer 4 FLOWS show real interactions
4. **Think about edge cases:** What could go wrong?
5. **Be specific:** Avoid "consider doing X" - say "do X when Y"
6. **Provide examples:** Show don't tell
7. **Safety first:** When in doubt, more safety checks

**Prompt structure pattern:**

```markdown
# [Role Name] System Prompt

## Role Identity

[From Layer 1 charter]

## Mission

[Clear, specific mission statement]

## Context Awareness

[How to track TU, snapshot, hot/cold, etc.] [Reference _shared/context_management.md]

## Safety Protocol

[Role-specific safety rules] [Reference _shared/safety_protocol.md]

## Core Responsibilities

[Detailed responsibilities with examples]

## Message Handling

[How to process intents] [Reference intent_handlers/]

## Quality Standards

[Self-check before sending] [Relevant quality bars]

## Escalation

[When to escalate] [Reference _shared/escalation_rules.md]

## Human Collaboration

[When to ask questions] [Reference _shared/human_interaction.md]

## Dormancy Signals

[When to go dormant, if applicable]

## Examples

[Link to examples/ directory]
```

### Measuring Success

Each role prompt is complete when:

- [ ] System prompt written
- [ ] All required intent handlers created
- [ ] At least 2 realistic examples provided
- [ ] References to Layers 0-4 correct
- [ ] Safety guidelines included
- [ ] Claude can role-play the agent successfully
- [ ] Human reviewer approves

---

## Testing Strategy

### Manual Testing

**For each role:**

1. Have Claude role-play the agent
2. Send it Layer 4 envelopes
3. Verify responses conform to protocol
4. Check artifacts validate against schemas
5. Verify safety boundaries respected

### Automated Testing

**Validation scripts:**

- Check all files exist
- Validate markdown structure
- Verify Layer 0-4 references resolve
- Check intent coverage
- Validate example envelopes

### Human Review Checklist

- [ ] Prompt faithful to charter?
- [ ] Clear and actionable?
- [ ] Safety guidelines adequate?
- [ ] Examples realistic?
- [ ] Would work with real LLM?

---

## Success Criteria

Layer 5 is complete when:

- [ ] All 9 epics implemented
- [ ] All 14 core roles have complete prompts (translator optional)
- [ ] All shared foundations documented
- [ ] All intents from Layer 4 handled
- [ ] At least 2 examples per role
- [ ] All prompts validated
- [ ] Human review passed
- [ ] Test conversation fixtures created
- [ ] Documentation complete
- [ ] Ready for Layer 6 integration

---

## Estimated Timeline

**With Claude Sonnet 3.5+ (focused work):**

- Aggressive: 3-4 weeks
- Moderate: 5-6 weeks
- Conservative: 8-10 weeks

Assumes:

- Claude working with human oversight
- Iterative refinement
- Human review at epic boundaries
- Testing between epics

---

## Dependencies for Layer 6 Integration

When Layer 5 is complete, Layer 6 needs:

- All prompt files in `05-prompts/`
- `prompt_loader.py` utility (Layer 6 responsibility)
- Bundling process in Layer 6 build

Layer 6 Epic 7 (Role Execution) directly depends on Layer 5 completion.

---

## Notes for Human Reviewers

**What to look for:**

- Are prompts faithful to Layer 1 charters?
- Are safety boundaries clear and enforced?
- Would an LLM understand these instructions?
- Are examples realistic and helpful?
- Is the tone appropriate (instructive, not conversational)?
- Are there any ambiguities that could lead to unsafe behavior?

**Common issues to watch for:**

- Vague instructions ("consider", "try to")
- Missing safety checks
- Insufficient examples
- Too much or too little autonomy
- Unclear escalation paths
- Overly chatty agent behavior

**Best prompts have:**

- Crystal clear instructions
- Specific examples
- Multiple safety layers
- Clear success criteria
- Graceful error handling
- Appropriate human collaboration
