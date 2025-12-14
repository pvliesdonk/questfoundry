# Mystery Manor Example

A complete QuestFoundry example project demonstrating the story_spark workflow.

## Story Summary

A short mystery story set in an old manor, featuring:

- 6 scenes with branching narrative
- A mysterious artifact (McGuffin)
- Suspects gathering for revelations

## Generation Details

- **Provider**: OpenAI
- **Workflow**: story_spark loop
- **Roles involved**: Showrunner, Plotwright, Scene Smith, Gatekeeper, Lorekeeper

### Workflow Sequence

1. **Plotwright** designed the narrative topology (acts, chapters, scenes)
2. **Scene Smith** filled scenes with prose content
3. **Gatekeeper** validated style + presentation bars
4. **Lorekeeper** promoted content to cold_store (canon)

## Files

| File | Description |
|------|-------------|
| `project.qfdb` | SQLite database with cold_store canon |
| `checkpoints.db` | Checkpoint history for resumption |
| `assets/` | Generated assets (if any) |

## Inspecting the Cold Store

```bash
# List all sections
sqlite3 project.qfdb "SELECT anchor, title FROM sections;"

# Read a scene's content
sqlite3 project.qfdb "SELECT content FROM sections WHERE anchor='scene_1';"

# Get full scene with metadata
sqlite3 project.qfdb "SELECT * FROM sections WHERE anchor='scene_1';"
```

## Resuming the Project

```bash
cd examples/mystery_manor
uv run qf ask "continue the story with act 2"
```

## Scene Overview

| Scene | Title |
|-------|-------|
| scene_1 | The Discovery |
| scene_2 | The McGuffin |
| scene_3 | The Suspects Gather |
| scene_4 | First Impressions |
| scene_5 | Whispered Confessions |
| scene_6 | The Artifact's Shadow |
