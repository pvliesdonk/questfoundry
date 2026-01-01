"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def design_docs_path(project_root: Path) -> Path:
    """Return the path to design documentation."""
    return project_root / "docs" / "design"
