## compiler-v0.6.0 (2025-11-17)

### Feat

- Add spec_fetcher module for downloading QuestFoundry specs from GitHub

### Fix

- Ensure UTF-8 encoding when writing prompt files

## compiler-v0.5.0 (2025-11-17)

### Feat

- **compiler-cli**: integrate logging with compiler CLI
- **compiler**: add comprehensive logging at multiple levels
- **webui**: complete final 5% - polish, utilities, and offline support
- **webui**: wire up pages with component library and add mobile nav
- **webui**: complete PWA component library and production setup
- **webui**: implement Phase 6 CI/CD workflows
- **webui**: implement Phase 5.1-5.2 PWA foundation (setup + API client + core pages)
- **webui**: complete Phase 4 database and deployment validation
- **webui**: implement Phase 3.2 artifact operations endpoints
- **webui**: implement Phase 3 API endpoints
- **webui**: implement Phase 2 API server core components
- **webui**: implement ValkeyStore with full StateStore protocol support
- **webui**: implement PostgresStore with full StateStore protocol support
- **webui**: add scaffolding for multi-tenant web API and PWA

### Fix

- enhance CLI options for spec directory and verbosity, and improve spec fetching logic
- improve error logging for CompilationError and standardize UTF-8 encoding
- update questfoundry-compiler version to 0.4.0
- update reference resolution for showrunner expertise and specify UTF-8 encoding for YAML file reading
- enhance error logging for CompilationError in assemblers
- specify UTF-8 encoding when reading markdown files
- **pwa**: satisfy Biome type import rules
- **pwa**: convert React/typed imports to import type
- **pwa**: resolve biome lint failures
- **webui**: align valkey error message
- **webui**: stabilize auth/project tests
- **webui**: expose PostgresStore for project tests
- **webui**: use valid encryption key in CI
- **webui**: satisfy ruff line length
- **ci**: ignore node modules during linting
- **ci**: skip node_modules json checks
- **ci**: limit markdownlint scope
- **webui**: unblock docker build and postgres tests
- **webui**: align artifact deps with local questfoundry lib
- **webui**: align mypy and tests with local library
- **webui**: address lint regressions
- address PR review comments - security, performance, and deprecation fixes
- **ci**: install dev dependencies in webui-ci workflow

## compiler-v0.4.0 (2025-11-16)

### Feat

- **compiler**: prep 0.5.0 release
- **cli,compiler**: add prompt profiles and controllers
- **cli**: support multi-loop bundles
- **prompt-generator**: add spec release fetcher

### Fix

- **compiler**: bundle spec data in hook
- **compiler**: add hatch spec copy hook
- address review feedback
- **cli**: handle duplicate abbreviations
- **prompt-generator**: satisfy mypy checks
- **prompt-generator**: fallback to bundled spec

## compiler-v0.3.1 (2025-11-16)

### Fix

- **compiler**: not exportong some needed classes

## compiler-v0.3.0 (2025-11-15)

### Feat

- **cli**: add prompt generator CLI tool (qf-generate)
- **compiler**: add PromptAssembler for web agent prompt generation
- **spec**: batch-reference all 23 orphaned snippets in adapters
- **spec**: reference 6 specialized procedures in adapters/playbooks
- **spec**: reference 16 high-value procedures in playbooks
- **orphans**: Priority 2 complete - merged/deleted 7 duplicate procedures
- **orphans**: delete 4 duplicate procedures (merges 3-6)
- **orphans**: delete binder_integrity_enforcement (duplicate of integrity_enforcement)
- **orphans**: merge leitmotif_use_policy → leitmotif_management
- **orphans**: Priority 1 complete - reference 6 new expertises in adapters

### Fix

- **pre-commit**: sync ruff versions with CI and skip cli mypy
- correct formatting and mypy issues
- **ci**: resolve CI failures from merge
- address code review feedback
- **cli**: improve spec directory path resolution
- **spec**: correct binder_integrity_enforcement expertise reference
- **spec**: complete spec validation fixes - all 157 errors resolved

## compiler-v0.2.0 (2025-11-15)

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
- **workflow**: enhance version bumping process and update configuration for commitizen
- **workflow**: update workspace configuration and bump command for spec versioning
- **docs**: update GitHub Actions workflow and add mkdocs configuration for documentation deployment
- **docs**: add mkdocs configuration and CNAME for site deployment
- add configuration for Commitizen and update workflows for documentation and versioning
- **mono-repo**: complete infrastructure setup with bundling, CI/CD, and documentation
- **mono-repo**: initial mono-repo setup with de-duplicated architecture

### Fix

- **compiler**: address PR review comments
- **compiler**: correct test fixture paths after package extraction
- **lib/python**: bump version to 0.6.0
- **lib**: set version to 0.2.0 and dedupe resources
- **lib**: stabilize manifest runtime
- **phase3**: address code review feedback
- **phase2**: address linting and test issues
- **spec**: update version format in SPEC_VERSION.txt for consistency
- **ci**: resolve setup errors by removing 'uv' cache and updating dependency installation steps
- **ci**: streamline Python setup and remove obsolete mkdocs configuration
- **state**: add `__del__` to SQLiteStore for proper resource cleanup
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

- **compiler**: extract spec compiler into standalone package
- **phase3**: remove v1 deprecated code (backed up in v1-archive tag)
- **compiler**: address PR #4 code review comments
- **workflow**: rename bump version step and streamline version extraction
