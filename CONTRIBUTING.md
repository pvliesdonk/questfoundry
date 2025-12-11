# Contributing to QuestFoundry

Thank you for your interest in contributing to QuestFoundry! This document provides
guidelines for contributing to the v3 architecture.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Essential Reading

Start with these foundational documents:

- [README.md](README.md) — Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) — Complete v3 system design
- [src/questfoundry/domain/](src/questfoundry/domain/) — MyST source of truth

### Key Concepts

- **MyST as Source of Truth**: Domain knowledge lives in MyST files with custom directives
- **8 Roles**: Showrunner, Lorekeeper, Narrator, Publisher, Creative Director, Plotwright,
  Scene Smith, Gatekeeper
- **Hot/Cold Stores**: Hot = working drafts, Cold = committed canon
- **System-as-Router**: Roles post intents, runtime routes based on loop definitions

## Project Structure

QuestFoundry v3 uses an integrated domain model:

```text
src/questfoundry/
├── domain/           # MyST source of truth (EDIT HERE)
│   ├── roles/        # Role definitions
│   ├── loops/        # Workflow graphs
│   ├── ontology/     # Artifacts, enums
│   └── protocol/     # Intents, routing rules
├── compiler/         # MyST → Python code generator
├── generated/        # Auto-generated (DO NOT EDIT)
└── runtime/          # LangGraph execution engine

tests/                # Test suites
examples/             # Example projects
_archive/             # v2 reference (read-only)
```

### Critical Rule: Never Edit `generated/`

The `generated/` directory is auto-generated from `domain/` files. Always:

1. Edit files in `domain/`
2. Run `qf compile` to regenerate
3. Never manually edit `generated/` files

## Development Setup

### 1. Install Dependencies

```bash
# Clone repository
git clone https://github.com/pvliesdonk/questfoundry.git
cd questfoundry

# Install with uv
uv sync

# Verify installation
uv run qf version
```

### 2. Install Pre-Commit Hooks

```bash
pre-commit install
```

This enables automatic quality checks:

- Markdown linting
- Python linting and formatting (Ruff)
- Type checking (mypy)
- Generated file protection

### 3. Run Quality Checks

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Test
uv run pytest
```

## Development Workflow

### Working with Domain Files

1. **Edit domain files** in `src/questfoundry/domain/`
2. **Compile** to regenerate code:

   ```bash
   qf compile
   ```

3. **Test** your changes:

   ```bash
   uv run pytest
   ```

### Adding a New Role

1. Create `domain/roles/new_role.md` with required directives
2. Run `qf compile`
3. Implement runtime tools in `runtime/tools/` if needed
4. Add tests

### Adding a New Loop

1. Create `domain/loops/new_loop.md` with graph definitions
2. Run `qf compile`
3. Test the workflow

## Code Standards

### Python Guidelines

- **Style**: Ruff for linting and formatting
- **Types**: Full type hints, mypy strict mode
- **Docstrings**: Google-style docstrings for all public APIs
- **Tests**: pytest with good coverage

### Docstring Example

```python
def promote_to_canon(self, artifact: Artifact) -> PromoteResult:
    """Promote a hot_store artifact to cold_store canon.

    Args:
        artifact: The artifact to promote. Must pass all quality bars.

    Returns:
        PromoteResult with success status and canon_id if successful.

    Raises:
        QualityBarError: If artifact fails any quality check.
    """
```

### Markdown Guidelines

- All `.md` files must pass markdownlint
- Use consistent heading hierarchy
- Include code blocks with language specifiers

## Submitting Changes

### Conventional Commits

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): subject
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`

**Scopes**: `domain`, `compiler`, `runtime`, `cli`

**Examples**:

```text
feat(domain): add new quality bar for accessibility
fix(runtime): handle empty scene content in gatekeeper
docs(readme): update quick start instructions
```

### Pull Request Process

1. Create a feature branch from `feat/v3` (or `main` after merge)
2. Make changes following the workflow above
3. Ensure all CI checks pass
4. Create PR with clear description
5. Address review feedback

### Definition of Done

Before submitting:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

## Questions?

- Open an issue for bugs or feature requests
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design questions
- Review domain files for role/loop specifications

Thank you for contributing to QuestFoundry!
