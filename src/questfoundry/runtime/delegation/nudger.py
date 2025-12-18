"""
Playbook-aware nudging.

Observes agent behavior against playbook expectations and generates
advisory nudge messages when deviations are detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from questfoundry.runtime.delegation.tracker import PlaybookInstance
from questfoundry.runtime.messaging import Message, create_nudge


@dataclass
class NudgeContext:
    """Context for nudge generation."""

    playbook_id: str
    instance_id: str
    phase_id: str
    turn: int
    agent_id: str


class PlaybookNudger:
    """
    Generates advisory nudges based on playbook expectations.

    The nudger compares actual agent behavior against playbook definitions
    and produces gentle reminders when deviations are detected. Nudges are
    advisory - agents can ignore them if they have good reason.

    Nudge types used:
    - missing_output: Expected artifact from phase not produced
    - unexpected_state: Phase/step state doesn't match expectations
    - quality_gate_reminder: Upcoming quality checkpoint
    - budget_warning: Rework budget running low or exhausted
    """

    def __init__(self, playbooks: dict[str, dict[str, Any]]) -> None:
        """
        Initialize the nudger with playbook definitions.

        Args:
            playbooks: Dictionary of playbook_id -> playbook definition
        """
        self._playbooks = playbooks

    def get_playbook(self, playbook_id: str) -> dict[str, Any] | None:
        """Get playbook definition by ID."""
        return self._playbooks.get(playbook_id)

    def check_phase_outputs(
        self,
        ctx: NudgeContext,
        artifacts_produced: list[str],
        artifact_types: dict[str, str] | None = None,
    ) -> list[Message]:
        """
        Check if expected outputs from a phase were produced.

        Args:
            ctx: Nudge context with playbook/phase info
            artifacts_produced: List of artifact IDs produced
            artifact_types: Optional mapping of artifact_id -> artifact_type

        Returns:
            List of nudge messages for missing outputs
        """
        playbook = self._playbooks.get(ctx.playbook_id)
        if not playbook:
            return []

        # Phases is a dict with phase_id as key
        phases = playbook.get("phases", {})
        phase = phases.get(ctx.phase_id)
        if not phase:
            return []

        nudges: list[Message] = []

        # Steps is a dict with step_id as key
        steps = phase.get("steps")
        if not steps:
            # A phase without steps has no expected outputs
            return []

        # Get expected outputs for this phase
        for _step_id, step in steps.items():
            expected_outputs = step.get("outputs", [])
            for output in expected_outputs:
                output_type = output.get("artifact_type")
                if not output_type:
                    continue

                # Check if any produced artifact matches the expected type
                type_found = False
                if artifact_types:
                    type_found = any(
                        artifact_types.get(aid) == output_type for aid in artifacts_produced
                    )
                else:
                    # Without type info, check for exact type match in artifact ID
                    # Use underscore/hyphen boundaries to avoid false positives
                    type_lower = output_type.lower()
                    for aid in artifacts_produced:
                        aid_lower = aid.lower()
                        # Check for exact match or bounded by common delimiters
                        if (
                            aid_lower == type_lower
                            or aid_lower.startswith(f"{type_lower}_")
                            or aid_lower.startswith(f"{type_lower}-")
                            or f"_{type_lower}_" in aid_lower
                            or f"-{type_lower}-" in aid_lower
                            or aid_lower.endswith(f"_{type_lower}")
                            or aid_lower.endswith(f"-{type_lower}")
                        ):
                            type_found = True
                            break

                if not type_found:
                    nudge = create_nudge(
                        from_agent="runtime",
                        to_agent=ctx.agent_id,
                        nudge_type="missing_output",
                        message=f"Phase '{ctx.phase_id}' expects output of type '{output_type}' "
                        f"which hasn't been produced yet.",
                        playbook_id=ctx.playbook_id,
                        playbook_instance_id=ctx.instance_id,
                        phase_id=ctx.phase_id,
                        expected_output=output_type,
                        turn_created=ctx.turn,
                    )
                    nudges.append(nudge)

        return nudges

    def check_quality_checkpoint(
        self,
        ctx: NudgeContext,
    ) -> Message | None:
        """
        Generate a reminder if the current phase has a quality checkpoint.

        Args:
            ctx: Nudge context with playbook/phase info

        Returns:
            Nudge message if quality checkpoint exists, None otherwise
        """
        playbook = self._playbooks.get(ctx.playbook_id)
        if not playbook:
            return None

        # Phases is a dict with phase_id as key
        phases = playbook.get("phases", {})
        phase = phases.get(ctx.phase_id)
        if not phase:
            return None

        quality_checkpoint = phase.get("quality_checkpoint")
        if not quality_checkpoint:
            return None

        validator = quality_checkpoint.get("validator")
        criteria = quality_checkpoint.get("criteria", [])

        if not validator or not criteria:
            return None

        criteria_summary = ", ".join(criteria[:3])
        if len(criteria) > 3:
            criteria_summary += f" (+{len(criteria) - 3} more)"

        return create_nudge(
            from_agent="runtime",
            to_agent=ctx.agent_id,
            nudge_type="quality_gate_reminder",
            message=f"Phase '{ctx.phase_id}' has a quality checkpoint validated by '{validator}'. "
            f"Criteria: {criteria_summary}",
            playbook_id=ctx.playbook_id,
            playbook_instance_id=ctx.instance_id,
            phase_id=ctx.phase_id,
            turn_created=ctx.turn,
        )

    def check_rework_budget_warning(
        self,
        ctx: NudgeContext,
        instance: PlaybookInstance,
    ) -> Message | None:
        """
        Generate a warning if rework budget is running low.

        Args:
            ctx: Nudge context
            instance: Playbook instance with budget info

        Returns:
            Nudge if budget is low (<=1 remaining), None otherwise
        """
        remaining = instance.max_rework_cycles - instance.rework_count
        if remaining > 1:
            return None

        if remaining == 1:
            message = (
                f"Rework budget warning: Only 1 rework cycle remaining for playbook "
                f"'{ctx.playbook_id}'. Current rework count: {instance.rework_count}."
            )
        else:
            message = (
                f"Rework budget critical: No rework cycles remaining for playbook "
                f"'{ctx.playbook_id}'. Further rework will trigger escalation."
            )

        return create_nudge(
            from_agent="runtime",
            to_agent=ctx.agent_id,
            nudge_type="budget_warning",
            message=message,
            playbook_id=ctx.playbook_id,
            playbook_instance_id=ctx.instance_id,
            phase_id=ctx.phase_id,
            current_state=f"rework_count={instance.rework_count}, max={instance.max_rework_cycles}",
            turn_created=ctx.turn,
        )

    def check_phase_consistency(
        self,
        ctx: NudgeContext,
        previous_phase: str | None,
    ) -> Message | None:
        """
        Check if phase transition follows expected flow.

        Args:
            ctx: Nudge context
            previous_phase: ID of the previous phase (None if first)

        Returns:
            Nudge if transition seems unexpected, None otherwise
        """
        playbook = self._playbooks.get(ctx.playbook_id)
        if not playbook or not previous_phase:
            return None

        # Phases is a dict with phase_id as key
        phases = playbook.get("phases", {})
        prev = phases.get(previous_phase)
        if not prev:
            return None

        # Check if current phase is an expected successor
        # on_success/on_failure contain next_phases arrays
        on_success = prev.get("on_success", {})
        on_failure = prev.get("on_failure", {})
        expected_next = set(on_success.get("next_phases", []) + on_failure.get("next_phases", []))

        if ctx.phase_id not in expected_next and expected_next:
            return create_nudge(
                from_agent="runtime",
                to_agent=ctx.agent_id,
                nudge_type="unexpected_state",
                message=f"Transition from '{previous_phase}' to '{ctx.phase_id}' "
                f"not in expected successors: {expected_next}",
                playbook_id=ctx.playbook_id,
                playbook_instance_id=ctx.instance_id,
                phase_id=ctx.phase_id,
                current_state=f"from={previous_phase}, to={ctx.phase_id}",
                turn_created=ctx.turn,
            )

        return None

    def generate_phase_entry_nudges(
        self,
        ctx: NudgeContext,
        instance: PlaybookInstance,
        previous_phase: str | None = None,
    ) -> list[Message]:
        """
        Generate all relevant nudges when entering a phase.

        Args:
            ctx: Nudge context
            instance: Playbook instance
            previous_phase: Previous phase ID (for consistency check)

        Returns:
            List of nudge messages
        """
        nudges: list[Message] = []

        # Check quality checkpoint
        quality_nudge = self.check_quality_checkpoint(ctx)
        if quality_nudge:
            nudges.append(quality_nudge)

        # Check rework budget
        budget_nudge = self.check_rework_budget_warning(ctx, instance)
        if budget_nudge:
            nudges.append(budget_nudge)

        # Check phase consistency
        consistency_nudge = self.check_phase_consistency(ctx, previous_phase)
        if consistency_nudge:
            nudges.append(consistency_nudge)

        return nudges
