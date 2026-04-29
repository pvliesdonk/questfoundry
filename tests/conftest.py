"""Pytest configuration and shared fixtures."""

import os
from collections.abc import Generator
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


@pytest.fixture(autouse=True)
def _reset_max_concurrency_override() -> Generator[None, None, None]:
    """Reset the process-global concurrency override around every test.

    The orchestrator constructor mutates this global from project.yaml. Without
    a reset, a leak from one test's orchestrator would change the result of
    ``get_model_info()`` in unrelated tests run later in the same session.
    Pinned at the suite level (#1581 review) rather than per-file so the
    invariant doesn't depend on which file owns the override.
    """
    from questfoundry.providers.model_info import set_max_concurrency_override

    set_max_concurrency_override(None)
    yield
    set_max_concurrency_override(None)


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def design_docs_path(project_root: Path) -> Path:
    """Return the path to design documentation."""
    return project_root / "docs" / "design"
