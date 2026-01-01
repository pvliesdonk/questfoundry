# Getting Started

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

This guide describes the recommended implementation order for a v5 cleanroom build. Follow these slices to build incrementally with working software at each step.

---

## Prerequisites

Before starting:

1. Read [00-vision.md](./00-vision.md) — Understand what v5 is
2. Read [01-pipeline-architecture.md](./01-pipeline-architecture.md) — Understand the flow
3. Set up development environment (see [11-proposed-dependencies.md](./11-proposed-dependencies.md))

---

## Implementation Slices

### Slice 1: DREAM Only

**Goal**: Single stage working end-to-end.

**Deliverables**:
- [ ] Project initialization (`qf init`)
- [ ] DREAM stage implementation
- [ ] Artifact writing (YAML)
- [ ] Basic CLI (`qf dream`)
- [ ] Schema validation for dream.yaml

**What to Build**:

```
src/questfoundry/
├── __init__.py
├── cli.py                 # qf command
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py    # Minimal: run_stage("dream")
│   └── stages/
│       ├── __init__.py
│       └── dream.py       # DREAM stage
├── prompts/
│   ├── compiler.py        # Basic template loading
│   └── templates/
│       └── dream.yaml     # DREAM prompt template
├── artifacts/
│   ├── __init__.py
│   ├── schemas/
│   │   └── dream.schema.json
│   └── writer.py          # Write YAML artifacts
└── providers/
    ├── __init__.py
    └── base.py            # LLM provider interface
```

**Test**:
```bash
qf init my_story
qf dream
cat my_story/artifacts/dream.yaml
```

### Slice 2: DREAM → SEED

**Goal**: Multi-stage pipeline with context passing.

**Deliverables**:
- [ ] BRAINSTORM stage
- [ ] SEED stage
- [ ] Context injection (prior artifacts → prompts)
- [ ] Human gates (review command)
- [ ] Run-to command (`qf run --to seed`)

**What to Add**:

```
src/questfoundry/
├── pipeline/
│   ├── orchestrator.py    # Add: run_to(), gate handling
│   └── stages/
│       ├── brainstorm.py
│       └── seed.py
├── prompts/
│   ├── compiler.py        # Add: context injection
│   └── templates/
│       ├── brainstorm.yaml
│       └── seed.yaml
├── artifacts/
│   ├── schemas/
│   │   ├── brainstorm.schema.json
│   │   └── seed.schema.json
│   └── reader.py          # Read prior artifacts for context
└── cli.py                 # Add: qf review, qf run --to
```

**Test**:
```bash
qf run --to brainstorm
qf review brainstorm  # Approve
qf seed
qf review seed        # Required gate
```

### Slice 3: Full GROW Decomposition

**Goal**: Complete GROW stage with all six layers.

**Deliverables**:
- [ ] SPINE layer
- [ ] ANCHORS layer (with state definitions)
- [ ] FRACTURES layer
- [ ] BRANCHES layer (sequential generation)
- [ ] CONNECTIONS layer (topology validation)
- [ ] BRIEFS layer
- [ ] HARVEST checkpoint support
- [ ] Layer-specific commands (`qf grow --layer spine`)

**What to Add**:

```
src/questfoundry/
├── pipeline/
│   └── stages/
│       └── grow/
│           ├── __init__.py
│           ├── spine.py
│           ├── anchors.py
│           ├── fractures.py
│           ├── branches.py
│           ├── connections.py
│           └── briefs.py
├── validation/
│   ├── __init__.py
│   ├── topology.py        # Graph validation
│   └── state.py           # State consistency
├── prompts/
│   └── templates/
│       ├── grow_spine.yaml
│       ├── grow_anchors.yaml
│       ├── grow_fractures.yaml
│       ├── grow_branch.yaml
│       └── grow_brief.yaml
└── artifacts/
    └── schemas/
        ├── grow_spine.schema.json
        ├── grow_anchors.schema.json
        └── ...
```

**Test**:
```bash
qf grow --layer spine
qf grow --layer anchors
qf review grow.anchors  # Required gate
qf grow  # Complete remaining layers
qf validate --pre-gate  # Check topology
```

### Slice 4: FILL and SHIP

**Goal**: Complete pipeline from DREAM to playable output.

**Deliverables**:
- [ ] FILL stage (scene prose generation)
- [ ] SHIP stage (export to formats)
- [ ] Twee export
- [ ] HTML export
- [ ] Full-gate validation
- [ ] Quality bars integration

**What to Add**:

```
src/questfoundry/
├── pipeline/
│   └── stages/
│       ├── fill.py
│       └── ship/
│           ├── __init__.py
│           ├── twee.py
│           ├── html.py
│           └── json.py
├── validation/
│   ├── quality_bars.py
│   └── full_gate.py
└── artifacts/
    └── schemas/
        ├── scene.schema.json
        └── manifest.schema.json
```

**Test**:
```bash
qf fill
qf validate --full-gate
qf ship --format twee
# Open my_story/exports/story.tw in Twine
```

---

## Testing Strategy

### Per-Slice Tests

Each slice should have:

| Test Type | Coverage |
|-----------|----------|
| Unit | Individual functions |
| Integration | Stage → artifact flow |
| E2E | CLI → output |

### Slice 1 Tests

```python
# tests/unit/test_dream_stage.py
def test_dream_produces_valid_artifact():
    ...

# tests/integration/test_dream_pipeline.py
def test_dream_writes_artifact():
    ...

# tests/e2e/test_dream_cli.py
def test_qf_dream_command():
    ...
```

### Cumulative Tests

As slices complete, add regression tests:

```python
# tests/e2e/test_full_pipeline.py
def test_dream_to_ship():
    """Complete pipeline produces valid export."""
    ...
```

---

## Checkpoints

### After Slice 1

You should have:
- Working CLI (`qf init`, `qf dream`)
- YAML artifact output
- Schema validation
- Basic prompt compilation

### After Slice 2

You should have:
- Multi-stage execution
- Context passing between stages
- Human gate enforcement
- Review workflow

### After Slice 3

You should have:
- Complete GROW decomposition
- Sequential branch generation
- Topology validation
- State management

### After Slice 4

You should have:
- Complete pipeline
- Playable output
- Full validation
- Export formats

---

## Common Pitfalls

### Pitfall: Building Everything at Once

**Problem**: Trying to implement all stages before testing any.

**Solution**: Complete each slice before starting the next. Have working software at each step.

### Pitfall: Skipping Validation

**Problem**: Writing stages without schema validation.

**Solution**: Validate artifacts immediately after generation. Catch errors early.

### Pitfall: Hardcoding Prompts

**Problem**: Embedding prompts in Python code.

**Solution**: Use prompt templates from the start. The compiler pays off quickly.

### Pitfall: Ignoring Human Gates

**Problem**: Building fully automated pipeline, adding gates later.

**Solution**: Gates are architectural. Build them into orchestrator from Slice 2.

### Pitfall: Parallel Branch Generation

**Problem**: Generating all branches simultaneously for speed.

**Solution**: Sequential generation is required for context coherence. Don't optimize prematurely.

---

## Development Workflow

### Daily Cycle

```bash
# 1. Run tests
uv run pytest tests/

# 2. Make changes

# 3. Run affected tests
uv run pytest tests/unit/test_<changed>.py -v

# 4. Run integration tests
uv run pytest tests/integration/ -v

# 5. Manual test
qf dream
qf review dream
```

### Commit Strategy

Commit at meaningful checkpoints:
- Stage implementation complete
- Tests passing
- Schema defined and validated

---

## Next Steps After Slice 4

Once the core pipeline works:

1. **Polish**: Better error messages, progress display
2. **Optimization**: Token budget tuning, caching
3. **Providers**: Add more LLM providers
4. **Export Formats**: Ink, JSON, more HTML themes
5. **Validation**: Research posture, enhanced accessibility

---

## See Also

- [01-pipeline-architecture.md](./01-pipeline-architecture.md) — Pipeline design
- [11-proposed-dependencies.md](./11-proposed-dependencies.md) — Setup
- [13-project-structure.md](./13-project-structure.md) — Where code goes
