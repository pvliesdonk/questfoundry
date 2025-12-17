# QuestFoundry

**QuestFoundry** is an AI-powered interactive fiction studio. A multi-agent runtime orchestrates specialized agents (writers, validators, archivists) to collaboratively create and maintain story content.

> **Status:** v4 runtime — functional alpha with full delegation support

## Features

- **12 Specialized Agents** — Each with specific responsibilities (Showrunner orchestrates, Scene Smith writes prose, Gatekeeper validates quality, Lorekeeper manages canon)
- **Playbook-Driven Workflows** — Declarative JSON playbooks define multi-phase workflows with quality checkpoints
- **Domain-Aligned Loop Termination** — Rework budgets per playbook, not simplistic depth limits
- **Artifact Versioning** — Full version history with hot/cold store semantics
- **Interactive & Batch Modes** — REPL for exploration, single-shot for automation
- **Checkpoint & Resume** — Save session state and resume from any checkpoint
- **Provider Flexibility** — Works with Ollama (local) or OpenAI

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/pvliesdonk/questfoundry.git
cd questfoundry
uv sync

# Verify installation
uv run qf version
```

### Basic Usage

```bash
# Check system health and provider availability
uv run qf doctor

# Create a new story project
uv run qf projects create "My Story"

# Start an interactive session with the Showrunner
uv run qf ask my-story

# Single-shot query (non-interactive)
uv run qf ask my-story "Create a mystery story outline set in a lighthouse"
```

### Interactive Session Example

```bash
$ uv run qf ask my-story
QuestFoundry Interactive Session
Project: my-story
Agent: Showrunner (showrunner)
Type 'exit' or Ctrl+D to quit

> Create a short mystery story set in an abandoned theater

[Showrunner activates, delegates to Plotwright for outline,
Scene Smith drafts prose, Gatekeeper validates quality...]

The dusty velvet curtains swayed in an impossible breeze...
```

### Using Different Providers

```bash
# Use Ollama (default, requires ollama serve running)
uv run qf ask my-story --provider ollama --model qwen3:8b

# Use OpenAI (requires OPENAI_API_KEY)
uv run qf ask my-story --provider openai --model gpt-4o

# Check which providers are available
uv run qf doctor
```

### Project Management

```bash
# List all projects
uv run qf projects list

# Show project details
uv run qf projects info my-story

# List artifacts in a project
uv run qf artifacts list my-story

# Show a specific artifact
uv run qf artifacts show my-story artifact-id
```

### Checkpoints and Resume

```bash
# Sessions auto-checkpoint after each turn
# List checkpoints
uv run qf checkpoints list my-story

# Resume from the latest checkpoint
uv run qf ask my-story --from-checkpoint latest

# Resume from a specific checkpoint
uv run qf ask my-story --from-checkpoint cp-abc123
```

### Verbose Mode

```bash
# Show token counts
uv run qf ask my-story "Hello" -v

# Show timing and debug info
uv run qf ask my-story "Hello" -vv

# Show full prompts (caution: very verbose)
uv run qf ask my-story "Hello" -vvv
```

### Event Logging

```bash
# Enable JSONL event logging to project directory
uv run qf ask my-story --log

# Events are written to projects/my-story/logs/events.jsonl
```

## The 12 Agents

| Agent | Archetype | Key Responsibility |
|-------|-----------|-------------------|
| **Showrunner** | Orchestrator | Hub-and-spoke delegation, manages workflows |
| **Lorekeeper** | Librarian | Canon management, lifecycle transitions |
| **Plotwright** | Architect | Story structure and outlines |
| **Scene Smith** | Author | Prose writing and drafting |
| **Gatekeeper** | Validator | Quality enforcement and rework triggers |
| **Researcher** | Fact Checker | Plausibility and consistency checking |
| **Style Lead** | Curator | Aesthetic coherence and voice |
| **Lore Weaver** | Synthesizer | Canon deepening and connections |
| **Codex Curator** | Documentarian | Player-safe entries and summaries |
| **Art Director** | Planner | Visual asset planning |
| **Audio Director** | Planner | Audio asset planning |
| **Book Binder** | Publisher | Static export and publishing |

## Repository Structure

```text
meta/                   # Domain-agnostic schemas (the contract)
├── schemas/core/       # Agent, Store, Tool, Artifact schemas
├── schemas/governance/ # Quality criteria schemas
└── docs/               # Meta-model documentation

domain-v4/              # QuestFoundry domain instances (JSON)
├── studio.json         # Main studio configuration
├── agents/             # 12 agent definitions
├── stores/             # Store definitions (canon, workspace, etc.)
├── tools/              # Tool definitions
├── playbooks/          # Workflow definitions
└── knowledge/          # Knowledge base entries

src/questfoundry/
├── runtime/            # V4 runtime implementation
│   ├── agent/          # Agent execution and showrunner loop
│   ├── delegation/     # Async delegation with rework budgets
│   ├── messaging/      # Inter-agent communication
│   ├── providers/      # LLM provider integrations
│   ├── storage/        # Project and artifact storage
│   └── tools/          # Tool implementations
└── cli.py              # Command-line interface
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/

# Format code
uv run ruff format src/
```

## Configuration

QuestFoundry reads configuration from environment variables:

```bash
# Ollama (default provider)
OLLAMA_HOST=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-...

# LangSmith tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=questfoundry
```

## License

MIT — See [`LICENSE`](LICENSE) for details.
