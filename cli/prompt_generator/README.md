# QuestFoundry Prompt Generator

**Generate monolithic web agent prompts from QuestFoundry behavior primitives**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The QuestFoundry Prompt Generator (`qf-generate`) is a CLI tool that assembles monolithic, context-optimized prompts for "web agent" simulations. These prompts enable you to simulate the entire QuestFoundry studio (or specific roles) in third-party chat UIs like ChatGPT, Claude, or Gemini.

This tool is part of the QuestFoundry mono-repo at `cli/prompt_generator/` and solves a key usability problem: enabling web-based AI assistants to understand and execute QuestFoundry workflows without requiring direct file access to the specification.

## Installation

From the `cli/prompt_generator` directory:

```bash
uv pip install -e .
```

Or install with dependencies from the project root:

```bash
cd cli/prompt_generator
uv sync
```

## Quick Start

### Generate a Loop Prompt

Generate a complete prompt for the "Lore Deepening" loop:

```bash
qf-generate --loop lore_deepening --output sim_prompt.md
```

This creates a monolithic prompt containing:

- Showrunner expertise
- All participating role expertises (Lore Weaver, Researcher, etc.)
- All procedures for the loop
- Safety protocols
- Execution steps
- RACI matrix

### Generate a Role Prompt

Generate a prompt for specific roles:

```bash
qf-generate --role lore_weaver --role plotwright --output roles_prompt.md
```

With standalone mode (includes all procedures from loops they participate in):

```bash
qf-generate --role lore_weaver --standalone --output standalone_prompt.md
```

### Interactive Mode

Run without arguments to enter interactive mode:

```bash
qf-generate
```

This will prompt you to:

1. Choose between loop or role prompt
2. Select the specific loop or roles
3. Configure options (e.g., standalone mode)

### List Available Options

List all available loops:

```bash
qf-generate list-loops
```

List all available roles:

```bash
qf-generate list-roles
```

## Usage Examples

### Web Agent Simulation

1. Generate a loop prompt:

   ```bash
   qf-generate --loop lore_deepening --output lore_deepening_sim.md
   ```

2. Upload `lore_deepening_sim.md` to your web agent (ChatGPT, Claude, etc.)

3. Start the simulation:

   ```
   You: "I have three accepted hooks about time magic: [hook descriptions].
        Please run the Lore Deepening loop."

   Agent: [Acts as Showrunner, coordinates Lore Weaver, Researcher, etc.]
   ```

### Role-Specific Testing

Test a specific role in isolation:

```bash
qf-generate --role gatekeeper --output gatekeeper_test.md
```

Then upload to test Gatekeeper's expertise and procedures.

### Multi-Role Collaboration

Simulate multiple roles working together:

```bash
qf-generate --role plotwright --role scene_smith --role style_lead \
    --standalone --output writing_team.md
```

## CLI Reference

### `qf-generate`

Main command to generate prompts.

**Options:**

- `--loop, -l <id>` - Loop/playbook ID (e.g., `lore_deepening`)
- `--role, -r <id>` - Role/adapter ID (can be repeated)
- `--standalone, -s` - Include all loop procedures for roles
- `--output, -o <path>` - Output file (default: stdout)
- `--spec-dir <path>` - Spec root directory (default: `spec/`)
- `--spec-source <auto|bundled|release>` - Choose how the spec is resolved. `auto`
  searches the repo tree then falls back to the bundled minimal spec, `bundled`
  forces the bundled copy, and `release` downloads the latest GitHub release
  into `~/.cache/questfoundry/spec/`.
- `--verbose, -v` - Show detailed progress

**Examples:**

```bash
# Single loop
qf-generate --loop lore_deepening -o prompt.md

# Multiple roles
qf-generate -r lore_weaver -r researcher -o prompt.md

# Standalone role prompt
qf-generate -r lore_weaver --standalone -o prompt.md

# Output to stdout
qf-generate --loop story_spark

# Verbose mode
qf-generate --loop hook_harvest --verbose -o prompt.md
```

### `qf-generate list-loops`

List all available loops/playbooks.

**Options:**

- `--spec-dir <path>` - Spec root directory
- `--spec-source <auto|bundled|release>` - Same semantics as the main command.

**Example:**

```bash
qf-generate list-loops
```

### `qf-generate list-roles`

List all available roles/adapters.

**Options:**

- `--spec-dir <path>` - Spec root directory
- `--spec-source <auto|bundled|release>` - Same semantics as the main command.

**Example:**

```bash
qf-generate list-roles
```

## Architecture

The tool uses the `questfoundry-compiler` library to:

1. Load all behavior primitives from `../../spec/05-behavior/`
2. Parse RACI assignments from `../../spec/01-roles/raci/by_loop.md`
3. Resolve all `@type:id` references
4. Assemble content into a single monolithic markdown file
5. Include safety protocols and validation requirements

The generated prompts are optimized for:

- **Token efficiency** - Only includes relevant content
- **Context preservation** - Clear structure with sections
- **Self-contained** - No external file dependencies
- **LLM-friendly** - Formatted for optimal LLM comprehension

## Spec Sources & Releases

`qf-generate` bundles a minimal snapshot of the spec for offline use. With
`--spec-source auto` (the default) the CLI searches for a local `spec/`
directory by walking up from the current working directory and falls back to
the bundled snapshot when nothing is found.

Use `--spec-source release` to download the latest published spec release from
GitHub (trusted per QuestFoundry policy). Releases are cached under
`~/.cache/questfoundry/spec/<tag>` so subsequent invocations reuse them without a
network call. You can also force the bundled snapshot via `--spec-source
bundled` when you explicitly want the paired minimal spec.

## Development

### Run Tests

```bash
uv run pytest tests/
```

### Run Linters

```bash
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
```

### Run All Checks

```bash
uv run hatch run all-checks
```

## Dependencies

- **questfoundry-compiler** - Core spec compilation library (from `../../lib/compiler/`)
- **typer** - CLI framework
- **rich** - Terminal formatting
- **questionary** - Interactive prompts

## Related Tools

- **[questfoundry-compiler](../../lib/compiler/)** - Spec compilation engine
- **[questfoundry-py](../../lib/python/)** - Python runtime library
- **[QuestFoundry Spec](../../spec/)** - Complete specification (Layers 0-5)

## License

MIT License - see LICENSE file for details.
