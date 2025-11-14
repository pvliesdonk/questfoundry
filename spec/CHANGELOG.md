## spec-v0.1.0 (2025-11-13)

### Feat

- **workflow**: enhance version bumping process and update configuration for commitizen
- **workflow**: update workspace configuration and bump command for spec versioning
- **docs**: update GitHub Actions workflow and add mkdocs configuration for documentation deployment
- **docs**: add mkdocs configuration and CNAME for site deployment
- add configuration for Commitizen and update workflows for documentation and versioning
- **mono-repo**: complete infrastructure setup with bundling, CI/CD, and documentation
- **mono-repo**: initial mono-repo setup with de-duplicated architecture

### Fix

- **spec**: update version format in SPEC_VERSION.txt for consistency
- **ci**: resolve setup errors by removing 'uv' cache and updating dependency installation steps
- **ci**: streamline Python setup and remove obsolete mkdocs configuration
- **state**: add **del** to SQLiteStore for proper resource cleanup
- **test**: improve error handling in threaded test
- **ci**: remove Python 3.14 from test matrix
- **lint**: resolve markdownlint errors in spec documentation
- **lint**: update markdownlint configuration to enforce ordered list style
- **lint**: enable markdownlint for spec/ directory with lenient config
- **ci**: add uv to pre-commit job and exclude docs from markdownlint
- configure mypy/ruff to work consistently in all environments
- correct mypy type ignore comments for workflow compatibility
- **pre-commit**: resolve all pre-commit hook failures
- **mypy**: correct type ignore comments for Google library imports
- **ci**: use setup-python for pre-commit job
- **ci**: resolve ruff format and mypy type errors
- **ci**: install dev dependencies in all workflow jobs
- remove unused Path imports from test files
- resolve MkDocs build errors and markdownlint issues
- address PR review comments

### Refactor

- **workflow**: rename bump version step and streamline version extraction
