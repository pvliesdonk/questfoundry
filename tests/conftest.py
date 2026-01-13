"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def disable_langsmith_tracing() -> None:
    """Disable LangSmith tracing during test runs.

    This prevents test runs from cluttering the LangSmith dashboard.
    Set LANGSMITH_TEST_TRACING=true to override for debugging.
    """
    if os.environ.get("LANGSMITH_TEST_TRACING", "").lower() != "true":
        os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def design_docs_path(project_root: Path) -> Path:
    """Return the path to design documentation."""
    return project_root / "docs" / "design"
