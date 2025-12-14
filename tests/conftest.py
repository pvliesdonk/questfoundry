"""
Pytest configuration and shared fixtures for QuestFoundry tests.
"""

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Root of the questfoundry repository."""
    return Path(__file__).parent.parent


@pytest.fixture
def domain_v4_path(repo_root: Path) -> Path:
    """Path to domain-v4/ studio definition."""
    return repo_root / "domain-v4"


@pytest.fixture
def meta_schemas_path(repo_root: Path) -> Path:
    """Path to meta/schemas/ directory."""
    return repo_root / "meta" / "schemas"


@pytest.fixture
def studio_json_path(domain_v4_path: Path) -> Path:
    """Path to domain-v4/studio.json."""
    return domain_v4_path / "studio.json"
