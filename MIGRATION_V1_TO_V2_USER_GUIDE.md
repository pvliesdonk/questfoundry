# QuestFoundry v1 → v2 Migration Guide

**For Users of the QuestFoundry Library**

This guide helps you migrate your code from QuestFoundry v1 to v2.

## Overview

QuestFoundry v2 introduces a major architectural change:

**v1:** Hardcoded loop classes with pre-assembled prompts
**v2:** Generic `PlaybookExecutor` with compiled behavior manifests

## Breaking Changes

### 1. Loop Execution

**v1 (Deprecated):**

```python
from questfoundry.loops.lore_deepening import LoreDeepeningLoop

loop = LoreDeepeningLoop(workspace, roles)
result = loop.execute()
```

**v2 (New):**

```python
from questfoundry.execution import PlaybookExecutor

# Option 1: Load from manifest directory
executor = PlaybookExecutor(
    playbook_id="lore_deepening",
    manifest_dir="dist/compiled/manifests"
)

# Option 2: Load from manifest file directly
executor = PlaybookExecutor(
    manifest_path="dist/compiled/manifests/lore_deepening.manifest.json"
)

# Execute with roles
result = executor.execute_full_loop(roles, artifacts, workspace, project_metadata)
```

### 2. Loop Discovery

**v1 (Deprecated):**

```python
from questfoundry.loops.registry import LoopRegistry

registry = LoopRegistry()
loop_metadata = registry.get_loop_metadata("lore_deepening")

# Instantiate specific loop class
from questfoundry.loops.lore_deepening import LoreDeepeningLoop
loop = LoreDeepeningLoop(...)
```

**v2 (New):**

```python
from questfoundry.loops.registry import LoopRegistry

# Automatically discovers loops from compiled manifests
registry = LoopRegistry(
    manifest_dir="dist/compiled/manifests",
    use_manifests=True  # Default
)

# Get executor directly
executor = registry.get_executor("lore_deepening")

# Or list available loops
loops = registry.list_loops()
```

### 3. Role Interface

Roles now support both v1 and v2 interfaces:

**v1 (Still Supported):**

```python
from questfoundry.roles.base import RoleContext

context = RoleContext(task="my_task", artifacts=[])
result = role.execute_task(context)
```

**v2 (Preferred):**

```python
from questfoundry.roles.base import RoleContext

# PlaybookExecutor automatically provides procedure content
context = RoleContext(
    task="my_task",
    artifacts=[],
    additional_context={
        "procedure": "Compiled procedure content from manifest",
        "step_id": "frame_questions",
        "playbook_id": "lore_deepening"
    }
)

# Both methods work
result = role.execute(context)  # v2
result = role.execute_task(context)  # v1 (internally calls execute)
```

### 4. Resource Bundling

**v1:**

- Prompts bundled from `spec/05-prompts/`
- No compilation step

**v2:**

- Behavior primitives compiled from `spec/05-behavior/`
- Manifests generated at build time
- Run `python scripts/bundle_resources.py` or `hatch run bundle`

## Compilation Workflow

### Build-Time Compilation

Add to your build process:

```bash
# 1. Compile behavior primitives to manifests
python -m questfoundry.compiler.cli \
    --spec-dir spec/ \
    --output dist/compiled/

# 2. Bundle resources (includes compiled manifests)
cd lib/python
python scripts/bundle_resources.py
```

### Runtime Usage

```python
from questfoundry.execution import PlaybookExecutor, ManifestLoader

# Option 1: Direct playbook execution
executor = PlaybookExecutor(playbook_id="lore_deepening")
results = executor.execute_full_loop(roles)

# Option 2: List available playbooks first
from questfoundry.execution import ManifestLoader
loader = ManifestLoader("dist/compiled/manifests")
available = loader.list_available_manifests()
print(f"Available playbooks: {available}")

# Then execute
executor = PlaybookExecutor(playbook_id=available[0])
```

## New Features in v2

### 1. Generic Execution

No need to import specific loop classes - one executor handles all playbooks:

```python
from questfoundry.execution import PlaybookExecutor

# Execute any playbook
for playbook_id in ["lore_deepening", "hook_harvest", "story_spark"]:
    executor = PlaybookExecutor(playbook_id=playbook_id)
    results = executor.execute_full_loop(roles)
```

### 2. Manifest Introspection

Access playbook metadata at runtime:

```python
executor = PlaybookExecutor(playbook_id="lore_deepening")

# Get RACI matrix
raci = executor.get_raci()
print(f"Responsible: {raci['responsible']}")
print(f"Consulted: {raci['consulted']}")

# Get quality bars
quality_bars = executor.get_quality_bars()
print(f"Quality bars: {quality_bars}")

# Get source files used in compilation
sources = executor.get_source_files()
print(f"Compiled from: {sources}")
```

### 3. Step-by-Step Execution

Execute individual steps with full control:

```python
executor = PlaybookExecutor(playbook_id="lore_deepening")

# Execute steps one at a time
result1 = executor.execute_step("frame_questions", roles, artifacts)
print(f"Step 1 result: {result1.output}")

# Results are available to subsequent steps
result2 = executor.execute_step("draft_canon", roles, artifacts)
print(f"Step 2 result: {result2.output}")
```

### 4. Atomic Behavior Primitives

Extend or customize by editing atomic primitives:

- **Expertises**: `spec/05-behavior/expertises/`
- **Procedures**: `spec/05-behavior/procedures/`
- **Snippets**: `spec/05-behavior/snippets/`
- **Playbooks**: `spec/05-behavior/playbooks/`
- **Adapters**: `spec/05-behavior/adapters/`

Changes to primitives are automatically included on next compilation.

## Backward Compatibility

v2 maintains compatibility during transition:

- Old loop classes still exist (but deprecated)
- v1 prompts still bundled from `spec/05-prompts/`
- `execute_task()` still works on all roles
- Registry works with both manifest-based and legacy loops

## Migration Checklist

- [ ] Update loop instantiation to use `PlaybookExecutor`
- [ ] Replace `LoopRegistry` usage to use manifests
- [ ] Add compilation step to your build process
- [ ] Test with compiled manifests
- [ ] Remove imports of specific loop classes (optional)
- [ ] Update role execution to use `execute()` if desired (optional)

## Common Issues

### Manifest Not Found

**Error:** `FileNotFoundError: Manifest not found`

**Solution:** Ensure manifests are compiled:

```bash
python -m questfoundry.compiler.cli --spec-dir spec/ --output dist/compiled/
```

### Validation Errors During Compilation

**Error:** Missing expertises or procedures

**Solution:** Check that all referenced primitives exist:

```bash
python -m questfoundry.compiler.cli --validate-only
```

Review validation output and ensure all `@expertise:`, `@procedure:`, and `@snippet:` references resolve.

### Missing Manifest Directory

**Error:** `Manifest directory not found`

**Solution:** Specify manifest directory explicitly:

```python
executor = PlaybookExecutor(
    playbook_id="lore_deepening",
    manifest_dir="/path/to/dist/compiled/manifests"
)
```

Or use default location: `dist/compiled/manifests/`

## Getting Help

- **Validation issues:** Run `python -m questfoundry.compiler.cli --validate-only`
- **Compilation issues:** Check `spec/05-behavior/` structure
- **Runtime issues:** Verify manifest files exist in `dist/compiled/manifests/`
- **Documentation:** See `spec/05-behavior/README.md`

## Version

This guide applies to QuestFoundry **v2.0.0** and later.
