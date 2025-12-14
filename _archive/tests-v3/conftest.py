"""Pytest configuration and fixtures.

NOTE: Some tests in this suite make actual LLM API calls (via langchain).
Integration tests (tests/integration/) are particularly slow as they exercise
full orchestrator workflows with multiple LLM calls per test.

Running tips:
- Use `pytest tests/unit/` for fast tests only
- Use `pytest -m "not slow"` to skip slow tests (when markers are added)
- Use `pytest --durations=10` to see slowest tests
"""

from __future__ import annotations

import pytest


def pytest_report_header(config: pytest.Config) -> list[str]:  # noqa: ARG001
    """Add custom header to pytest output."""
    return [
        "",
        "NOTE: Integration tests make real LLM calls and may be slow.",
        "      Use `pytest tests/unit/` for fast unit tests only.",
        "      Use `pytest -x` to stop on first failure.",
        "",
    ]


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (making LLM API calls)",
    )
    config.addinivalue_line(
        "markers",
        "llm: marks tests that require LLM API access",
    )
