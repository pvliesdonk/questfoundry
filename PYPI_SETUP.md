# PyPI Publishing Setup Guide

This document explains what you need to configure to publish the QuestFoundry packages to PyPI.

## Overview

The monorepo contains two Python packages that can be published to PyPI:

1. **questfoundry-compiler** (`lib/compiler`) - Spec compiler for QuestFoundry
2. **questfoundry-py** (`lib/python`) - Python library for QuestFoundry protocol

Each package has its own GitHub Actions workflow for automated publishing.

## PyPI Account Setup

### 1. Create PyPI Accounts

You'll need accounts on both:

- **PyPI (Production)**: <https://pypi.org>
- **TestPyPI (Testing)**: <https://test.pypi.org>

### 2. Register Package Names

Before you can publish, you need to register the package names:

1. Build the packages locally:

   ```bash
   # For compiler
   cd lib/compiler
   uv build
   
   # For python library
   cd lib/python
   uv build
   ```

2. Manually upload the first version to claim the package name:

   ```bash
   # Install twine
   pip install twine
   
   # Upload to TestPyPI first (recommended for testing)
   twine upload --repository testpypi dist/*
   
   # Then upload to PyPI (production)
   twine upload dist/*
   ```

## GitHub Repository Configuration

### 3. Configure GitHub Environments

You need to create GitHub Environments for trusted publishing. Go to your repository settings:

**Settings → Environments → New environment**

Create the following environments:

#### For questfoundry-compiler:

1. **Environment: `pypi-compiler`**
   - No environment protection rules needed for basic setup
   - Used for publishing to PyPI production

2. **Environment: `testpypi-compiler`**
   - No environment protection rules needed for basic setup
   - Used for publishing to TestPyPI

#### For questfoundry-py:

1. **Environment: `pypi`**
   - No environment protection rules needed for basic setup
   - Used for publishing to PyPI production

2. **Environment: `testpypi`**
   - No environment protection rules needed for basic setup
   - Used for publishing to TestPyPI

### 4. Configure PyPI Trusted Publishing

Trusted Publishing uses OpenID Connect (OIDC) to securely publish packages without requiring long-lived API tokens.

#### For questfoundry-compiler:

1. Go to <https://pypi.org/manage/account/publishing/> (or <https://test.pypi.org/manage/account/publishing/> for TestPyPI)
2. Add a new pending publisher with these details:
   - **PyPI Project Name**: `questfoundry-compiler`
   - **Owner**: `pvliesdonk` (your GitHub username/org)
   - **Repository name**: `questfoundry`
   - **Workflow name**: `publish-compiler.yml`
   - **Environment name**: `pypi-compiler` (or `testpypi-compiler` for TestPyPI)

#### For questfoundry-py:

1. Go to <https://pypi.org/manage/account/publishing/>
2. Add a new pending publisher with these details:
   - **PyPI Project Name**: `questfoundry-py`
   - **Owner**: `pvliesdonk`
   - **Repository name**: `questfoundry`
   - **Workflow name**: `publish-python.yml`
   - **Environment name**: `pypi` (or `testpypi` for TestPyPI)

**Note**: You must register on both PyPI and TestPyPI separately if you want to publish to both.

## Publishing Packages

Once configured, you can publish packages using GitHub Actions:

### Manual Workflow Dispatch

1. Go to **Actions** tab in GitHub
2. Select the workflow:
   - `Publish Compiler` for questfoundry-compiler
   - `Publish Python Library` for questfoundry-py
3. Click **Run workflow**
4. Choose options:
   - **Destination**: `pypi` (production) or `testpypi` (testing)
   - **Bump version**: Check to auto-increment version (default: true)

### Version Bumping

When "bump version" is enabled, the workflow will:

1. Use Commitizen to determine the next version based on commit messages
2. Update version in `pyproject.toml`
3. Create a git tag (format: `compiler-v0.1.1` or `v0.6.1`)
4. Push the tag to GitHub
5. Build and publish the package
6. Create a GitHub Release with the built artifacts

### Testing Before Production

**Always test on TestPyPI first:**

1. Run workflow with destination `testpypi`
2. Verify the package installs correctly:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ questfoundry-compiler
   # or
   pip install --index-url https://test.pypi.org/simple/ questfoundry-py
   ```

3. Once verified, run workflow with destination `pypi`

## Troubleshooting

### "Package name already taken"

You need to manually upload the first version to claim the name (see step 2).

### "Invalid or non-existent authentication"

Ensure Trusted Publishing is configured correctly on PyPI with matching workflow and environment names.

### "Environment not found"

Create the required GitHub Environments in your repository settings.

### "Permission denied"

Ensure the workflow has `id-token: write` permission (already configured in the workflows).

## Commitizen Configuration

Both packages use Commitizen for version management:

- **Compiler**: Tag format `compiler-v$version` (e.g., `compiler-v0.1.1`)
- **Python Library**: Tag format `v$version` (e.g., `v0.6.1`)

Commit messages must follow Conventional Commits format for automatic version bumping:

- `feat:` - Increments minor version (0.1.0 → 0.2.0)
- `fix:` - Increments patch version (0.1.0 → 0.1.1)
- `BREAKING CHANGE:` or `!` - Increments major version (0.1.0 → 1.0.0)

## Summary Checklist

- [ ] Create PyPI and TestPyPI accounts
- [ ] Build packages locally
- [ ] Manually upload first version to claim package names
- [ ] Create GitHub Environments (pypi-compiler, testpypi-compiler, pypi, testpypi)
- [ ] Configure Trusted Publishing on PyPI for questfoundry-compiler
- [ ] Configure Trusted Publishing on TestPyPI for questfoundry-compiler
- [ ] Configure Trusted Publishing on PyPI for questfoundry-py
- [ ] Configure Trusted Publishing on TestPyPI for questfoundry-py
- [ ] Test publish to TestPyPI
- [ ] Verify installation from TestPyPI
- [ ] Publish to production PyPI

## Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [Commitizen Documentation](https://commitizen-tools.github.io/commitizen/)
- [Conventional Commits](https://www.conventionalcommits.org/)
