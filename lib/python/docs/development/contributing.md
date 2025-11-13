# Contributing to QuestFoundry

Thank you for your interest in contributing to QuestFoundry!

## Development Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/pvliesdonk/questfoundry.git
   cd questfoundry
   ```

2. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Set up the Python environment**:

   ```bash
   cd lib/python
   uv sync
   ```

4. **Install pre-commit hooks**:

   ```bash
   cd ../..  # Back to repo root
   pip install pre-commit
   pre-commit install
   ```

5. **Bundle resources**:

   ```bash
   cd lib/python
   uv run hatch run bundle
   ```

## Development Workflow

### Running Tests

```bash
cd lib/python
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=questfoundry --cov-report=term
```

### Linting and Formatting

```bash
# Check linting
uv run ruff check src tests

# Auto-fix linting issues
uv run ruff check --fix src tests

# Format code
uv run ruff format src tests
```

### Type Checking

```bash
uv run mypy src
```

### Run All Checks

```bash
uv run hatch run all-checks
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feat/my-feature
```

Use conventional commit prefixes:

- `feat/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `refactor/` for refactoring
- `test/` for test changes

### 2. Make Your Changes

- Follow the coding guidelines in the repository
- Add tests for new functionality
- Update documentation as needed
- Bundle resources if you modified schemas or prompts

### 3. Commit Your Changes

Use conventional commits:

```bash
git commit -m "feat(validators): add new validation function"
```

Format: `type(scope): description`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

### 4. Run Pre-commit Hooks

Pre-commit hooks will run automatically on commit. To run manually:

```bash
pre-commit run --all-files
```

### 5. Push and Create PR

```bash
git push origin feat/my-feature
```

Then create a pull request on GitHub.

## Project Structure

```text
lib/python/
├── src/questfoundry/        # Main package
│   ├── utils/               # Utilities (resource loading, etc.)
│   ├── validators/          # Validation logic
│   ├── providers/           # LLM providers
│   ├── lifecycles/          # Lifecycle implementations
│   └── resources/           # Bundled resources (auto-generated)
├── tests/                   # Test suite
├── scripts/                 # Build and utility scripts
├── docs/                    # Documentation
├── pyproject.toml           # Project configuration
└── mkdocs.yml              # Documentation configuration
```

## Coding Standards

- Python 3.11+ features
- Full type annotations
- Follow Ruff linting rules
- Use modern Python idioms

## Questions?

- Open a [Discussion](https://github.com/pvliesdonk/questfoundry/discussions)
- Create an [Issue](https://github.com/pvliesdonk/questfoundry/issues)
