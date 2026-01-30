# QuestFoundry Roadmap

Implementation roadmap for QuestFoundry v5 pipeline.

## Overview

QuestFoundry is built incrementally in **slices** — each slice delivers working software, not stubs. Complete each slice before starting the next.

```
Slice 1 → Slice 2 → Slice 3 → Slice 4 → Polish → Slice 5 (DRESS)
```

For detailed specifications, see [docs/design/12-getting-started.md](docs/design/12-getting-started.md).

---

## Slice 1: DREAM Only

**Goal**: Single stage working end-to-end.

**Milestone**: [Slice 1: DREAM Only](https://github.com/pvliesdonk/questfoundry/milestone/1)

| Deliverable | Issue | Status |
|-------------|-------|--------|
| Pipeline orchestrator skeleton | #2 | Open |
| Prompt compiler (basic) | #3 | Open |
| DREAM artifact schema | #4 | Open |
| Ollama provider | #5 | Open |
| DREAM stage implementation | #6 | Open |
| CLI `qf dream` command | #7 | Open |
| CI pipeline setup | #8 | Open |

**Test checkpoint**:
```bash
qf dream "A noir detective story in 1940s Los Angeles"
cat artifacts/dream.yaml
```

**Design reference**: [docs/design/12-getting-started.md#slice-1-dream-only](docs/design/12-getting-started.md)

---

## Slice 2: DREAM → SEED

**Goal**: Multi-stage pipeline with context passing and human gates.

**Milestone**: [Slice 2: DREAM → SEED](https://github.com/pvliesdonk/questfoundry/milestone/2)

**Epic**: #9

| Deliverable | Status |
|-------------|--------|
| BRAINSTORM stage | To be detailed |
| SEED stage | To be detailed |
| Context injection | To be detailed |
| Human gates | To be detailed |
| CLI: `qf run --to`, `qf review` | To be detailed |

**Test checkpoint**:
```bash
qf run --to brainstorm
qf review brainstorm
qf seed
qf review seed  # Required gate
```

**Design reference**: [docs/design/12-getting-started.md#slice-2-dream--seed](docs/design/12-getting-started.md)

---

## Slice 3: Full GROW Decomposition

**Goal**: Complete GROW stage with all six layers.

**Milestone**: [Slice 3: Full GROW](https://github.com/pvliesdonk/questfoundry/milestone/3)

**Epic**: #10

| Layer | Purpose | Status |
|-------|---------|--------|
| SPINE | Core emotional arc | To be detailed |
| ANCHORS | Hubs, gates, bottlenecks | To be detailed |
| FRACTURES | Divergence points | To be detailed |
| BRANCHES | Path expansion | To be detailed |
| CONNECTIONS | Topology validation | To be detailed |
| BRIEFS | Scene specifications | To be detailed |

**Test checkpoint**:
```bash
qf grow --layer spine
qf grow --layer anchors
qf review grow.anchors  # Required gate
qf grow  # Complete remaining layers
qf validate --pre-gate
```

**Design reference**: [docs/design/03-grow-stage-specification.md](docs/design/03-grow-stage-specification.md)

---

## Slice 4: FILL and SHIP

**Goal**: Complete pipeline producing playable output.

**Milestone**: [Slice 4: FILL and SHIP](https://github.com/pvliesdonk/questfoundry/milestone/4)

**Epic**: #11

| Deliverable | Status |
|-------------|--------|
| FILL stage (prose) | To be detailed |
| SHIP stage (export) | To be detailed |
| Twee format | To be detailed |
| HTML format | To be detailed |
| JSON format | To be detailed |
| Full-gate validation | To be detailed |

**Test checkpoint**:
```bash
qf fill
qf validate --full-gate
qf ship --format twee
# Open exports/story.tw in Twine
```

**Design reference**: [docs/design/12-getting-started.md#slice-4-fill-and-ship](docs/design/12-getting-started.md)

---

## Post-Slice 4: Polish

After core pipeline works:

| Area | Examples |
|------|----------|
| Polish | Better error messages, progress display |
| Optimization | Token budget tuning, caching |
| Providers | More LLM providers |
| Formats | Ink export, more HTML themes |
| Validation | Enhanced accessibility checks |

---

## Slice 5: DRESS (Art Direction, Illustrations, Codex)

**Goal**: Presentation layer — visual identity, illustrations, and player-facing codex.

**Epic**: #414

| Deliverable | Status |
|-------------|--------|
| Ontology + procedure spec | To be detailed |
| DRESS Pydantic models | To be detailed |
| Graph mutations for DRESS | To be detailed |
| Image provider abstraction | To be detailed |
| Art direction sub-stage | To be detailed |
| Illustration brief generation | To be detailed |
| Codex generation | To be detailed |
| Image generation orchestration | To be detailed |
| Asset storage + manifest | To be detailed |
| SHIP updates for DRESS export | To be detailed |
| CLI: `qf dress` | To be detailed |
| Prompt templates | To be detailed |

**Test checkpoint**:
```bash
qf dress --project myproject
# Gate 1: review art direction + entity visuals
# Gate 2: review briefs + codex, set budget
# Sample confirmation, then batch generation
ls projects/myproject/assets/
qf ship --format html  # verify illustrations + codex in export
```

**Design reference**: [docs/design/procedures/dress.md](docs/design/procedures/dress.md)

---

## Deferred

| Feature | Reason |
|---------|--------|
| HARVEST iteration | Optional complexity; add if needed |
| Web UI | CLI-first; UI is separate concern |

---

## Links

- **Design documentation**: [docs/design/](docs/design/)
- **Architecture documentation**: [docs/architecture/](docs/architecture/)
- **All milestones**: [GitHub Milestones](https://github.com/pvliesdonk/questfoundry/milestones)
- **All issues**: [GitHub Issues](https://github.com/pvliesdonk/questfoundry/issues)
