# Spec Query Template

Use this with `Task` tool (subagent_type: `Explore`) to query the spec without filling main context.

## Spec Structure

```
spec/
├── 00-north-star/     # Vision, principles, loops, quality bars
│   ├── WORKING_MODEL.md, QUALITY_BARS.md, SOURCES_OF_TRUTH.md
│   ├── LOOPS/         # 12 loops (Story Spark, Hook Harvest, etc.)
│   └── PLAYBOOKS/     # Execution guides
├── 01-roles/          # 15 roles (charters/, briefs/, interfaces/)
├── 02-dictionary/     # artifacts/, glossary.md, taxonomy/
├── 03-schemas/        # 28 JSON schemas
├── 04-protocol/       # ENVELOPE.md, INTENTS.md, LIFECYCLES/, FLOWS/
├── 05-definitions/    # expertises/, procedures/, snippets/, playbooks/, adapters/
└── 06-runtime/        # Runtime behavior specs
```

## Key Concepts

- **Hot/Cold**: Hot=drafts, Cold=approved canon
- **TU (Trace Unit)**: Work order (hot-proposed→stabilizing→gatecheck→cold-merged)
- **Hook Card**: Follow-up items
- **Quality Bars**: 8 criteria (Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility)
- **Envelope**: Protocol message wrapper
- **PN**: Player-Narrator (only sees Cold/player-safe)
