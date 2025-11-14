## spec-v0.7.0 (2025-11-14)

### Feat

- **phase3**: version bump to 2.0.0 with migration guide
- **phase3**: update resource bundling for v2 architecture
- **phase3**: implement PlaybookExecutor and manifest-based loop registry
- **phase2**: add manifest JSON schemas
- **phase2**: add CLI entry point for spec compiler
- **phase2**: implement spec compiler core
- **behavior**: add cross-reference validation script
- **behavior**: complete snippet extraction (29 total)
- **behavior**: complete procedure extraction (50 total)
- **behavior**: extract Scene Smith and Plotwright procedures (batch 2)
- **behavior**: extract art and audio procedures (batch 1)
- **behavior**: extract art and audio procedures (batch 1)
- **behavior**: convert final 4 playbooks to YAML
- **behavior**: convert playbooks 7-9 to YAML
- **behavior**: convert playbooks 4-6 to YAML
- **behavior**: convert first 3 playbooks to YAML
- **behavior**: convert all 15 role adapters to YAML
- **behavior**: add core reusable snippets
- **behavior**: add coordination and interaction procedures
- **behavior**: add quality and canon procedures
- **behavior**: add core validation and workflow procedures
- **behavior**: extract asset and support role expertises
- **behavior**: extract export, verification, performance expertises
- **behavior**: extract topology, publication, voice expertises
- **behavior**: extract core role expertises
- **behavior**: create Layer 5 v2 architecture foundation
- **migration**: add comprehensive v1 to v2 migration instructions

### Fix

- **lib/python**: bump version to 0.6.0
- **lib**: set version to 0.2.0 and dedupe resources
- **lib**: stabilize manifest runtime
- **phase3**: address code review feedback
- **phase2**: address linting and test issues

### Refactor

- **phase3**: remove v1 deprecated code (backed up in v1-archive tag)
- **compiler**: address PR #4 code review comments

## spec-v0.6.0 (2025-11-14)

### Fix

- align release numbering after the rollback so future spec tags resume at `spec-v0.6.0`

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
