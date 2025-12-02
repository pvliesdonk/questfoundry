---
description: Search and summarize content from the QuestFoundry specification (spec/)
allowed-tools: Glob, Grep, Read, Task
---

# Spec Reader

You are answering a question about the QuestFoundry specification. Search and summarize relevant content from `spec/` to answer: **$ARGUMENTS**

## Spec Structure (use this to find relevant files)

```
spec/
├── 00-north-star/     # Vision, principles, loops, quality bars, playbooks
│   ├── WORKING_MODEL.md      # Hot/Cold model, roles overview
│   ├── QUALITY_BARS.md       # 8 quality bars (Integrity, Reachability, etc.)
│   ├── SOURCES_OF_TRUTH.md   # Hot vs Cold storage explained
│   ├── LOOPS/                # 12 production loops (Story Spark, Hook Harvest, etc.)
│   └── PLAYBOOKS/            # Detailed loop execution guides
├── 01-roles/          # 15 role definitions
│   ├── charters/      # Role responsibilities and authorities
│   ├── briefs/        # Agent implementation briefs
│   └── interfaces/    # Role-to-role interaction patterns
├── 02-dictionary/     # Common language
│   ├── artifacts/     # 22 artifact type definitions (hook_card, tu_brief, etc.)
│   ├── glossary.md    # Terminology
│   └── taxonomy/      # Classification systems
├── 03-schemas/        # JSON Schema specs (28 schemas, Draft 2020-12)
├── 04-protocol/       # Messaging protocol
│   ├── ENVELOPE.md    # Message envelope format
│   ├── INTENTS.md     # Message intent catalog (hook.create, tu.merge, etc.)
│   ├── LIFECYCLES/    # State machines (hooks, TU, gate, view)
│   ├── FLOWS/         # Message sequence diagrams per loop
│   └── EXAMPLES/      # Example messages
├── 05-definitions/    # Executable definitions (Cartridge Architecture)
│   ├── expertises/    # Domain knowledge per role
│   ├── procedures/    # Reusable workflow steps
│   ├── snippets/      # Small text blocks
│   ├── playbooks/     # Loop YAML definitions
│   └── adapters/      # Role YAML configurations
└── 06-runtime/        # Runtime behavior specs
```

## Key Concepts Quick Reference

- **Hot/Cold**: Hot = discovery/draft space, Cold = curated canon (approved by Gatekeeper)
- **Trace Unit (TU)**: Work order tracking changes through states: hot-proposed → stabilizing → gatecheck → cold-merged
- **Hook Card**: Small traceable follow-up items discovered during work
- **Quality Bars**: 8 criteria (Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility)
- **Roles**: 15 roles (Showrunner, Gatekeeper, Plotwright, Scene Smith, Lore Weaver, Codex Curator, etc.)
- **Envelope**: Protocol message wrapper with sender, receiver, intent, context, safety flags, payload
- **PN (Player-Narrator)**: Downstream role that only sees Cold/player-safe content

## Instructions

1. Use Glob/Grep to find relevant files based on the question
2. Read the most relevant sections (be surgical - don't read everything)
3. Provide a concise summary with:
   - Clear explanation (bullets preferred)
   - File references for further reading
   - Related concepts if relevant

Keep the summary focused and concise - the user can ask follow-up questions.
