"""Shared fixtures for runtime tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def make_mock_playbook(
    playbook_id: str,
    phases: dict[str, dict[str, Any]],
    max_rework_cycles: int = 3,
) -> MagicMock:
    """
    Create a mock playbook for testing.

    Args:
        playbook_id: Playbook ID
        phases: Dictionary of phase_id -> phase definition
        max_rework_cycles: Maximum rework cycles for this playbook

    Returns:
        Mock playbook object
    """
    playbook = MagicMock()
    playbook.id = playbook_id
    playbook.name = playbook_id.replace("_", " ").title()
    playbook.phases = phases
    playbook.max_rework_cycles = max_rework_cycles
    playbook.entry_phase = next(iter(phases.keys())) if phases else None
    return playbook


@pytest.fixture
def mock_playbook_factory():
    """Factory fixture for creating mock playbooks."""
    return make_mock_playbook
