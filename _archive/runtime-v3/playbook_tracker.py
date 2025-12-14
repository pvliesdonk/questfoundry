"""Playbook tracker - tracks playbook context for runtime nudging.

This module provides:
- PlaybookTracker: Tracks consulted playbooks and expected outputs
- Nudge generation when outputs are missing
- Integration with orchestrator for runtime guidance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.runtime.domain.models import Playbook, PlaybookPhase

logger = logging.getLogger(__name__)


@dataclass
class PhaseProgress:
    """Tracks progress through a playbook phase."""

    phase_id: str
    phase_name: str
    completed_steps: set[str] = field(default_factory=set)
    expected_outputs: set[str] = field(default_factory=set)
    produced_outputs: set[str] = field(default_factory=set)
    is_complete: bool = False


@dataclass
class PlaybookProgress:
    """Tracks progress through a playbook."""

    playbook_id: str
    playbook_name: str
    current_phase: str | None = None
    phases_entered: set[str] = field(default_factory=set)
    phases_completed: set[str] = field(default_factory=set)
    expected_outputs: set[str] = field(default_factory=set)
    produced_outputs: set[str] = field(default_factory=set)
    is_complete: bool = False


class PlaybookTracker:
    """Tracks playbook context for nudging.

    The tracker monitors:
    - Which playbooks have been consulted
    - What phase we're in (if following a playbook)
    - What outputs are expected vs produced
    - When nudges are needed
    """

    def __init__(self) -> None:
        """Initialize the playbook tracker."""
        self.consulted_playbooks: dict[str, PlaybookProgress] = {}
        self.active_playbook_id: str | None = None
        self.produced_artifacts: set[str] = set()
        self._nudge_count = 0
        self._max_nudges_per_playbook = 3

    def on_playbook_consulted(self, playbook_id: str, playbook: Playbook) -> None:
        """Called when SR consults a playbook.

        Args:
            playbook_id: The playbook ID
            playbook: The loaded playbook definition
        """
        if playbook_id not in self.consulted_playbooks:
            progress = PlaybookProgress(
                playbook_id=playbook_id,
                playbook_name=playbook.name,
                current_phase=playbook.entry_phase,
            )

            # Extract expected outputs from playbook
            for output in playbook.outputs.required_artifacts:
                progress.expected_outputs.add(output.type)

            # Also gather outputs from all phases
            for phase_id, phase in playbook.phases.items():
                for step_id, step in phase.steps.items():
                    for step_output in step.outputs:
                        if step_output.artifact_type:
                            progress.expected_outputs.add(step_output.artifact_type)

            self.consulted_playbooks[playbook_id] = progress
            logger.info(
                f"Playbook '{playbook.name}' consulted. "
                f"Expected outputs: {progress.expected_outputs}"
            )

        # Set as active playbook
        self.active_playbook_id = playbook_id

    def on_phase_entered(self, playbook_id: str, phase_id: str) -> None:
        """Called when SR enters a phase in a playbook.

        Args:
            playbook_id: The playbook ID
            phase_id: The phase ID being entered
        """
        progress = self.consulted_playbooks.get(playbook_id)
        if progress:
            progress.current_phase = phase_id
            progress.phases_entered.add(phase_id)
            logger.debug(f"Entered phase '{phase_id}' in playbook '{playbook_id}'")

    def on_phase_completed(self, playbook_id: str, phase_id: str) -> None:
        """Called when a phase is completed.

        Args:
            playbook_id: The playbook ID
            phase_id: The completed phase ID
        """
        progress = self.consulted_playbooks.get(playbook_id)
        if progress:
            progress.phases_completed.add(phase_id)
            logger.debug(f"Completed phase '{phase_id}' in playbook '{playbook_id}'")

    def on_artifact_created(self, artifact_type: str) -> None:
        """Called when an artifact is written.

        Args:
            artifact_type: The type of artifact created
        """
        self.produced_artifacts.add(artifact_type)

        # Update all active playbook progress
        for progress in self.consulted_playbooks.values():
            if artifact_type in progress.expected_outputs:
                progress.produced_outputs.add(artifact_type)
                logger.debug(
                    f"Artifact type '{artifact_type}' produced for playbook "
                    f"'{progress.playbook_name}'"
                )

    def on_playbook_completed(self, playbook_id: str) -> None:
        """Called when a playbook is completed.

        Args:
            playbook_id: The completed playbook ID
        """
        progress = self.consulted_playbooks.get(playbook_id)
        if progress:
            progress.is_complete = True
            logger.info(f"Playbook '{progress.playbook_name}' marked complete")

            # Clear active if this was the active playbook
            if self.active_playbook_id == playbook_id:
                self.active_playbook_id = None

    def get_nudge(self) -> str | None:
        """Check if a nudge is needed.

        Returns:
            A nudge message if outputs are missing, None otherwise
        """
        if not self.active_playbook_id:
            return None

        progress = self.consulted_playbooks.get(self.active_playbook_id)
        if not progress or progress.is_complete:
            return None

        # Check for missing outputs
        missing = progress.expected_outputs - progress.produced_outputs
        if not missing:
            return None

        # Rate limit nudges
        self._nudge_count += 1
        if self._nudge_count > self._max_nudges_per_playbook:
            return None

        return (
            f"According to the '{progress.playbook_name}' playbook, "
            f"these outputs are expected but not yet produced: {', '.join(sorted(missing))}. "
            f"Is this intentional, or should work continue?"
        )

    def get_phase_guidance(self) -> str | None:
        """Get guidance for the current phase.

        Returns:
            Phase guidance message if in a phase, None otherwise
        """
        if not self.active_playbook_id:
            return None

        progress = self.consulted_playbooks.get(self.active_playbook_id)
        if not progress or not progress.current_phase:
            return None

        return (
            f"Currently in phase '{progress.current_phase}' of "
            f"'{progress.playbook_name}' playbook."
        )

    def get_progress_summary(self) -> str:
        """Get a summary of all playbook progress.

        Returns:
            Human-readable progress summary
        """
        if not self.consulted_playbooks:
            return "No playbooks consulted yet."

        lines = ["## Playbook Progress"]
        for playbook_id, progress in self.consulted_playbooks.items():
            status = "Complete" if progress.is_complete else "In Progress"
            produced = len(progress.produced_outputs)
            expected = len(progress.expected_outputs)

            lines.append(f"\n### {progress.playbook_name} [{status}]")
            lines.append(f"- Current phase: {progress.current_phase or 'None'}")
            lines.append(f"- Outputs: {produced}/{expected}")

            missing = progress.expected_outputs - progress.produced_outputs
            if missing:
                lines.append(f"- Missing: {', '.join(sorted(missing))}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the tracker state."""
        self.consulted_playbooks.clear()
        self.active_playbook_id = None
        self.produced_artifacts.clear()
        self._nudge_count = 0
