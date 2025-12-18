"""
Playbook loader for delegation system.

Provides convenient access to playbook metadata required for delegation,
particularly rework budgets and phase properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.models.base import Playbook


class PlaybookLoader:
    """
    Loads and provides access to playbook metadata for delegation.

    This class bridges the domain model (Playbook objects) with the delegation
    system's needs for rework budget information and phase properties.

    The delegation system needs:
    - max_rework_cycles: Per-playbook ceiling on how many times phases can be reworked
    - is_rework_target: Per-phase marker indicating if re-entry counts against budget

    Usage:
        # Create from loaded Studio
        loader = PlaybookLoader.from_playbooks(studio.playbooks)

        # Get rework budget for a playbook
        budget = loader.get_max_rework_cycles("story_spark")  # Returns 3

        # Check if a phase is a rework target
        is_rework = loader.is_rework_target("story_spark", "brief_creation")  # True
    """

    def __init__(self, playbooks: dict[str, dict[str, Any]]) -> None:
        """
        Initialize the loader with playbook data.

        Args:
            playbooks: Dictionary of playbook_id -> playbook definition dict
        """
        self._playbooks = playbooks

    @classmethod
    def from_playbooks(cls, playbooks: list[Playbook]) -> PlaybookLoader:
        """
        Create a PlaybookLoader from a list of Playbook model objects.

        This is the preferred way to create a loader from domain data loaded
        via load_studio().

        Args:
            playbooks: List of Playbook model objects from Studio.playbooks

        Returns:
            PlaybookLoader instance
        """
        playbook_dict = {
            pb.id: {
                "id": pb.id,
                "name": pb.name,
                "phases": pb.phases,
                "max_rework_cycles": pb.max_rework_cycles,
                "entry_phase": pb.entry_phase,
            }
            for pb in playbooks
        }
        return cls(playbook_dict)

    def get_playbook(self, playbook_id: str) -> dict[str, Any] | None:
        """
        Get the raw playbook definition.

        Args:
            playbook_id: ID of the playbook

        Returns:
            Playbook definition dict, or None if not found
        """
        return self._playbooks.get(playbook_id)

    def get_max_rework_cycles(self, playbook_id: str) -> int:
        """
        Get the maximum rework cycles for a playbook.

        This is the per-playbook ceiling on how many times phases marked
        as rework targets can be re-entered before escalation.

        Args:
            playbook_id: ID of the playbook

        Returns:
            Maximum rework cycles, defaults to 3 if playbook not found
        """
        result: int = self._playbooks.get(playbook_id, {}).get("max_rework_cycles", 3)
        return result

    def is_rework_target(self, playbook_id: str, phase_id: str) -> bool:
        """
        Check if a phase is marked as a rework target.

        Phases with is_rework_target=true have their re-entries counted
        against the playbook's rework budget. When the budget is exhausted,
        re-entering these phases triggers escalation.

        Args:
            playbook_id: ID of the playbook
            phase_id: ID of the phase

        Returns:
            True if the phase is a rework target, False otherwise
        """
        playbook = self._playbooks.get(playbook_id, {})
        phases = playbook.get("phases", {})
        phase = phases.get(phase_id, {})
        result: bool = phase.get("is_rework_target", False)
        return result

    def get_phase(self, playbook_id: str, phase_id: str) -> dict[str, Any] | None:
        """
        Get a specific phase definition.

        Args:
            playbook_id: ID of the playbook
            phase_id: ID of the phase

        Returns:
            Phase definition dict, or None if not found
        """
        result: dict[str, Any] | None = (
            self._playbooks.get(playbook_id, {}).get("phases", {}).get(phase_id)
        )
        return result

    def get_entry_phase(self, playbook_id: str) -> str | None:
        """
        Get the entry phase for a playbook.

        Args:
            playbook_id: ID of the playbook

        Returns:
            Entry phase ID, or None if not found/not specified
        """
        playbook = self._playbooks.get(playbook_id)
        if not playbook:
            return None
        return playbook.get("entry_phase")

    def get_rework_target_phases(self, playbook_id: str) -> list[str]:
        """
        Get all phases marked as rework targets in a playbook.

        Args:
            playbook_id: ID of the playbook

        Returns:
            List of phase IDs that are rework targets
        """
        phases = self._playbooks.get(playbook_id, {}).get("phases", {})
        return [
            phase_id for phase_id, phase in phases.items() if phase.get("is_rework_target", False)
        ]

    def list_playbooks(self) -> list[str]:
        """
        List all loaded playbook IDs.

        Returns:
            List of playbook IDs
        """
        return list(self._playbooks.keys())

    def get_playbook_dict(self) -> dict[str, dict[str, Any]]:
        """
        Get the raw playbook dictionary.

        This can be passed directly to PlaybookNudger which expects
        the same dict[str, dict[str, Any]] format.

        Returns:
            Dictionary of playbook_id -> playbook definition
        """
        return self._playbooks
