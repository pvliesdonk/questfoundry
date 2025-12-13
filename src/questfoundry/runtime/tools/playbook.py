"""Playbook consultation tools for domain-v4.

This module provides the v4-compatible playbook consultation tool
that integrates with PlaybookTracker for nudging.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ConsultPlaybookV4(BaseTool):
    """Look up a playbook definition to understand workflow guidance.

    This v4 version works with domain-v4 Studio models and integrates
    with PlaybookTracker for output tracking and nudging.

    Use this to understand:
    - What phases and steps a workflow contains
    - What artifacts are expected as outputs
    - What quality criteria apply
    """

    name: str = "consult_playbook"
    description: str = (
        "Look up a playbook to understand its workflow, phases, steps, and "
        "expected outputs. Input: playbook_id (e.g., 'story_spark', 'scene_weave')"
    )

    # Injected by registry - use Any to avoid Pydantic forward reference issues
    studio: Any = None
    tracker: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str) -> str:
        """Look up playbook definition.

        Args:
            query: Playbook ID or search term

        Returns:
            Formatted playbook guidance
        """
        if not self.studio:
            return "Error: Studio not configured. Cannot look up playbooks."

        # Normalize query
        query = query.lower().replace(" ", "_").replace("-", "_")

        # Try exact match first
        playbook = self.studio.playbooks.get(query)

        # If no exact match, try partial match
        if playbook is None:
            for playbook_id, pb in self.studio.playbooks.items():
                if query in playbook_id.lower() or query in pb.name.lower():
                    playbook = pb
                    query = playbook_id
                    break

        if playbook is None:
            # List available playbooks
            available = list(self.studio.playbooks.keys())
            if available:
                return (
                    f"Playbook '{query}' not found.\n\n"
                    f"Available playbooks: {', '.join(sorted(available))}"
                )
            return f"Playbook '{query}' not found and no playbooks available."

        # Notify tracker
        if self.tracker:
            self.tracker.on_playbook_consulted(query, playbook)

        # Return formatted guidance
        return self._format_playbook(query, playbook)

    def _format_playbook(self, playbook_id: str, playbook: "Playbook") -> str:
        """Format playbook as readable guidance for agents."""
        lines = [
            f"# Playbook: {playbook.name}",
            "",
            f"**ID**: {playbook_id}",
            f"**Version**: {playbook.version}",
            "",
            "## Purpose",
            playbook.purpose,
            "",
        ]

        # Description if present
        if playbook.description:
            lines.extend(["## Description", playbook.description, ""])

        # Triggers
        if playbook.triggers:
            lines.append("## When to Use (Triggers)")
            for trigger in sorted(playbook.triggers, key=lambda t: t.priority):
                lines.append(f"- [{trigger.priority}] {trigger.condition}")
            lines.append("")

        # Expected Inputs
        if playbook.inputs.required_artifacts or playbook.inputs.optional_artifacts:
            lines.append("## Inputs")
            if playbook.inputs.required_artifacts:
                lines.append("**Required:**")
                for inp in playbook.inputs.required_artifacts:
                    desc = f" - {inp.description}" if inp.description else ""
                    lines.append(f"- {inp.type}{desc}")
            if playbook.inputs.optional_artifacts:
                lines.append("**Optional:**")
                for inp in playbook.inputs.optional_artifacts:
                    desc = f" - {inp.description}" if inp.description else ""
                    lines.append(f"- {inp.type}{desc}")
            lines.append("")

        # Expected Outputs
        if playbook.outputs.required_artifacts or playbook.outputs.optional_artifacts:
            lines.append("## Expected Outputs")
            if playbook.outputs.required_artifacts:
                lines.append("**Required:**")
                for out in playbook.outputs.required_artifacts:
                    desc = f" - {out.description}" if out.description else ""
                    lines.append(f"- {out.type}{desc}")
            if playbook.outputs.optional_artifacts:
                lines.append("**Optional:**")
                for out in playbook.outputs.optional_artifacts:
                    desc = f" - {out.description}" if out.description else ""
                    lines.append(f"- {out.type}{desc}")
            lines.append("")

        # Phases
        lines.append("## Phases")
        lines.append(f"**Entry Point**: {playbook.entry_phase}")
        lines.append("")

        for phase_id, phase in playbook.phases.items():
            entry_marker = " [ENTRY]" if phase_id == playbook.entry_phase else ""
            lines.append(f"### {phase.name}{entry_marker}")
            lines.append(f"**Purpose**: {phase.purpose}")

            if phase.depends_on:
                lines.append(f"**Depends on**: {', '.join(phase.depends_on)}")

            # Steps
            if phase.steps:
                lines.append("")
                lines.append("**Steps:**")
                for step_id, step in phase.steps.items():
                    agent = step.specific_agent or f"archetype:{step.agent_archetype}"
                    lines.append(f"1. **{step_id}** ({agent}): {step.action}")
                    if step.guidance:
                        lines.append(f"   - _Guidance_: {step.guidance}")

            # Completion criteria
            if phase.completion_criteria:
                lines.append("")
                lines.append("**Completion Criteria:**")
                for criterion in phase.completion_criteria:
                    lines.append(f"- {criterion}")

            # Quality checkpoint
            if phase.quality_checkpoint:
                qc = phase.quality_checkpoint
                lines.append("")
                lines.append(
                    f"**Quality Checkpoint**: {', '.join(qc.criteria)} "
                    f"(threshold: {qc.pass_threshold or 'default'})"
                )

            # Transitions
            if phase.on_success:
                if phase.on_success.end_playbook:
                    lines.append(f"**On Success**: End playbook")
                elif phase.on_success.next_phases:
                    lines.append(
                        f"**On Success**: -> {', '.join(phase.on_success.next_phases)}"
                    )
                if phase.on_success.message:
                    lines.append(f"   _{phase.on_success.message}_")

            if phase.on_failure:
                if phase.on_failure.next_phases:
                    lines.append(
                        f"**On Failure**: -> {', '.join(phase.on_failure.next_phases)}"
                    )
                if phase.on_failure.message:
                    lines.append(f"   _{phase.on_failure.message}_")

            lines.append("")

        # Quality criteria
        if playbook.quality_criteria:
            lines.append("## Quality Criteria")
            for criterion in playbook.quality_criteria:
                lines.append(f"- {criterion}")
            lines.append("")

        lines.append(f"**Max Rework Cycles**: {playbook.max_rework_cycles}")

        return "\n".join(lines)


def create_consult_playbook_tool(
    studio: Any,
    tracker: Any = None,
) -> ConsultPlaybookV4:
    """Create a ConsultPlaybook tool configured for a studio.

    Args:
        studio: The loaded studio
        tracker: Optional playbook tracker for nudging

    Returns:
        Configured ConsultPlaybookV4 tool
    """
    tool = ConsultPlaybookV4()
    tool.studio = studio
    tool.tracker = tracker
    return tool
