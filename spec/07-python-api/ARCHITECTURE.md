# Phase 6: Python API Layer Architecture

**Version**: 1.0.0
**Date**: 2025-11-20
**Status**: Specification

---

## Purpose

The Python API layer provides a **high-level, author-friendly interface** to the QuestFoundry runtime. It wraps the LangGraph-based runtime (Phase 5) with intuitive Python classes and methods.

---

## Design Principles

1. **Simple is better than complex** - Hide runtime complexity
2. **Explicit is better than implicit** - Clear method names and parameters
3. **Pythonic** - Follow Python conventions and idioms
4. **Type-safe** - Full type hints for IDE support
5. **Runtime-agnostic** - Could swap runtime implementations

---

## Architecture Layers

```
┌─────────────────────────────────────┐
│   Author's Python Script            │  (User code)
│   from questfoundry import Story    │
│   story = Story.new("My Novel")     │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   Python API Layer (Phase 6)        │  ← THIS LAYER
│   ├── Story class                   │
│   ├── Scene class                   │
│   ├── Codex class                   │
│   ├── Project class                 │
│   └── Session class                 │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   Runtime Layer (Phase 5)           │
│   ├── Showrunner                    │
│   ├── GraphFactory                  │
│   ├── NodeFactory                   │
│   └── StateManager                  │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   LangGraph StateGraphs             │
│   (10 loops executing roles)        │
└─────────────────────────────────────┘
```

---

## Core Classes

### 1. **Project** - Project management

```python
from questfoundry import Project

# Create new project
project = Project.new(
    path="./my-novel",
    name="Dragon's Legacy",
    author="Alice",
    genre="fantasy"
)

# Open existing project
project = Project.open("./my-novel")

# Project operations
project.save()
project.export("epub")
project.status()  # Show stats
```

### 2. **Story** - Story management

```python
from questfoundry import Story

# Access story from project
story = project.story

# Write scenes
scene = story.write("A tense cargo bay confrontation")

# Review and refine
story.review()

# Get scenes
scenes = story.scenes
latest = story.latest_scene
```

### 3. **Scene** - Individual scene

```python
# Scene operations
scene.content  # Full text
scene.preview()  # First 200 chars
scene.metadata  # TU info, quality bars
scene.edit("Update this part")
scene.delete()
```

### 4. **Codex** - Lore and worldbuilding

```python
codex = project.codex

# Add lore
entry = codex.add("Fuel theft protocols in the Alliance")

# Query
results = codex.search("fuel")
entry = codex.get("fuel-theft-protocols")

# Categories
codex.categories()  # ["technology", "culture", etc.]
```

### 5. **Session** - Interactive mode

```python
from questfoundry import Session

# Start interactive session
with Session(project) as session:
    session.write("New scene")
    session.review()
    session.add_lore("Topic")
```

---

## API Examples

### Example 1: Complete Workflow

```python
from questfoundry import Project

# 1. Create project
project = Project.new("./my-novel", name="Space Opera")

# 2. Write scenes
scene1 = project.story.write("Captain discovers missing fuel")
scene2 = project.story.write("Confrontation with pilot")

# 3. Review
project.story.review()

# 4. Add lore
project.codex.add("Alliance fuel protocols")

# 5. Export
project.export("epub", output="./output/book.epub")
```

### Example 2: Scene Operations

```python
# Create scene
scene = project.story.write(
    "A tense moment in the cargo bay",
    mode="workshop"  # or "production"
)

# Check status
print(scene.status)  # "hot-proposed"
print(scene.quality)  # Dict of quality bars

# Preview
print(scene.preview())  # First 200 chars

# Edit
scene.edit("Make it more tense")

# Approve
scene.approve()  # Moves to stabilizing → gatecheck
```

### Example 3: Codex Management

```python
codex = project.codex

# Add entries
entry1 = codex.add("The Alliance formed in 2247")
entry2 = codex.add("Fuel is precious in deep space")

# Search
results = codex.search("Alliance")
for entry in results:
    print(entry.title, entry.content)

# Get specific entry
entry = codex.get_by_id("LORE-001")
entry.expand()  # Run lore_deepening loop
```

### Example 4: Project Export

```python
# Export formats
project.export("epub")        # E-book
project.export("html")        # Web page
project.export("markdown")    # Plain text
project.export("pdf")         # PDF (requires pandoc)

# With options
project.export("epub",
    output="./dist/book.epub",
    cover_image="./cover.png",
    metadata={
        "author": "Alice",
        "title": "Dragon's Legacy"
    }
)
```

---

## Class Responsibilities

### Project Class

**Responsibilities**:
- Project initialization and configuration
- File system management (.questfoundry/ directory)
- Access to story, codex, session
- Export orchestration

**Methods**:
```python
class Project:
    @classmethod
    def new(cls, path: str, name: str, **kwargs) -> "Project"

    @classmethod
    def open(cls, path: str) -> "Project"

    def save(self) -> None
    def export(self, format: str, **kwargs) -> Path
    def status(self) -> Dict[str, Any]

    @property
    def story(self) -> Story

    @property
    def codex(self) -> Codex
```

### Story Class

**Responsibilities**:
- Scene creation (story_spark loop)
- Scene management and querying
- Review orchestration (hook_harvest loop)
- Style tuning

**Methods**:
```python
class Story:
    def write(self, text: str, mode: str = "workshop") -> Scene
    def review(self) -> ReviewResult
    def tune_style(self) -> TuneResult

    @property
    def scenes(self) -> List[Scene]

    @property
    def latest_scene(self) -> Optional[Scene]

    def get_scene(self, tu_id: str) -> Optional[Scene]
```

### Scene Class

**Responsibilities**:
- Scene content access
- Scene metadata and quality
- Scene editing
- Scene lifecycle management

**Methods**:
```python
class Scene:
    @property
    def content(self) -> str

    @property
    def status(self) -> str

    @property
    def quality(self) -> Dict[str, str]

    def preview(self, length: int = 200) -> str
    def edit(self, instruction: str) -> "Scene"
    def approve(self) -> None
    def delete(self) -> None
```

### Codex Class

**Responsibilities**:
- Lore entry creation (lore_deepening loop)
- Lore querying and search
- Entry expansion and refinement

**Methods**:
```python
class Codex:
    def add(self, topic: str) -> CodexEntry
    def search(self, query: str) -> List[CodexEntry]
    def get_by_id(self, entry_id: str) -> Optional[CodexEntry]
    def categories(self) -> List[str]
```

### Session Class

**Responsibilities**:
- Interactive REPL-like interface
- Command history
- Context management
- Multi-command workflows

**Methods**:
```python
class Session:
    def __enter__(self) -> "Session"
    def __exit__(self, *args) -> None

    def write(self, text: str) -> Scene
    def review(self) -> ReviewResult
    def add_lore(self, topic: str) -> CodexEntry
    def export(self, format: str) -> Path
```

---

## Runtime Integration

The API layer calls the runtime via Showrunner:

```python
# In Story.write()
def write(self, text: str, mode: str = "workshop") -> Scene:
    # 1. Create ParsedIntent
    intent = ParsedIntent(
        action="write",
        args=[text],
        flags={"mode": mode},
        loop_id="story_spark"
    )

    # 2. Execute via Showrunner
    from questfoundry.runtime.cli.showrunner import Showrunner
    showrunner = Showrunner()
    result = showrunner.execute_request(f"write {text}", intent)

    # 3. Wrap in Scene object
    return Scene(
        tu_id=result.tu_id,
        content=result.artifacts["scene_output"]["content"],
        status=result.quality_status,
        project=self.project
    )
```

---

## Storage Layer

The API maintains a simple storage layer:

```
project-dir/
├── .questfoundry/
│   ├── config.yaml          # Project config
│   ├── storage.db           # SQLite database
│   └── cache/               # Cached artifacts
├── scenes/                  # Scene files (optional)
│   ├── TU-2025-042.md
│   └── TU-2025-043.md
└── codex/                   # Lore files (optional)
    ├── LORE-001.md
    └── LORE-002.md
```

**Database Schema**:
```sql
CREATE TABLE trace_units (
    tu_id TEXT PRIMARY KEY,
    loop_id TEXT,
    status TEXT,
    created_at TIMESTAMP,
    content TEXT,
    metadata JSON
);

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    tu_id TEXT,
    artifact_type TEXT,
    content TEXT,
    metadata JSON,
    FOREIGN KEY (tu_id) REFERENCES trace_units(tu_id)
);

CREATE TABLE quality_bars (
    tu_id TEXT,
    bar_name TEXT,
    status TEXT,
    score INTEGER,
    details JSON,
    PRIMARY KEY (tu_id, bar_name),
    FOREIGN KEY (tu_id) REFERENCES trace_units(tu_id)
);
```

---

## Implementation Phases

### Phase 6A: Foundation (Priority 1)
1. Project class with initialization
2. Storage layer (SQLite + file system)
3. Basic Story class with write()
4. Scene class with content access

### Phase 6B: Core Features (Priority 2)
5. Story.review() integration
6. Codex class with add() and search()
7. Scene editing and lifecycle
8. Export basic formats

### Phase 6C: Advanced Features (Priority 3)
9. Session class for interactive mode
10. Advanced export formats
11. Caching and optimization
12. Error recovery

---

## Success Criteria

Phase 6 is complete when:

1. ✅ Project.new() creates valid project structure
2. ✅ Story.write() executes story_spark loop and returns Scene
3. ✅ Scene.content returns full scene text
4. ✅ Story.review() executes hook_harvest loop
5. ✅ Codex.add() executes lore_deepening loop
6. ✅ Project.export("markdown") generates output file
7. ✅ All classes have full type hints
8. ✅ API is intuitive and Pythonic
9. ✅ Storage persists across sessions

---

## Next Steps

After Phase 6:
- **Phase 7**: CLI tool that uses the Python API
- **Phase 8**: Web interface (optional)
- **Phase 9**: Advanced features (audio, art, translation)

---

**References**:
- Phase 5 Runtime: `spec/06-runtime/`
- Showrunner: `spec/06-runtime/components/showrunner_agent.md`
- Loop Definitions: `spec/05-definitions/loops/`
