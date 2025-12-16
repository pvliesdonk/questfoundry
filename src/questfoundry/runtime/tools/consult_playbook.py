"""
Consult Playbook tool implementation.

Returns full playbook details including phases, steps, agents, and completion criteria.
Follows the menu+consult pattern from meta/docs/patterns.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.models import Playbook


@register_tool("consult_playbook")
class ConsultPlaybookTool(BaseTool):
    """
    Retrieve full details for a playbook/workflow.

    Use this to understand:
    - The sequence of phases and steps
    - Which agents handle each step
    - Input/output artifacts
    - Completion criteria
    - Quality checkpoints
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute playbook lookup."""
        playbook_id = args.get("playbook_id")

        if not playbook_id:
            # List available playbooks with directive message
            available = [pb.id for pb in self._context.studio.playbooks]
            return ToolResult(
                success=True,
                data={
                    "message": (
                        "You called consult_playbook without specifying a playbook_id. "
                        "For new story requests, call again with playbook_id='story_spark'. "
                        "Available playbooks:"
                    ),
                    "available_playbooks": available,
                    "next_action": "Call consult_playbook(playbook_id='story_spark') for story creation workflow",
                },
            )

        # Find playbook in studio
        playbook = None
        for pb in self._context.studio.playbooks:
            if pb.id == playbook_id:
                playbook = pb
                break

        if not playbook:
            available = [pb.id for pb in self._context.studio.playbooks]
            return ToolResult(
                success=False,
                data={"available_playbooks": available},
                error=f"Playbook not found: {playbook_id}",
            )

        # Build detailed response
        result_data = self._format_playbook(playbook)

        # Add clear next-action guidance
        if playbook.entry_phase and playbook.phases:
            entry_phase_data = playbook.phases.get(playbook.entry_phase, {})
            steps = entry_phase_data.get("steps", {})
            if steps:
                first_step = list(steps.values())[0] if steps else {}
                first_agent = first_step.get("specific_agent") or first_step.get("agent_archetype")
                if first_agent:
                    result_data["next_action"] = (
                        f"Now delegate to '{first_agent}' to begin the workflow. "
                        f"Use: delegate(to_agent='{first_agent}', task='...')"
                    )

        return ToolResult(success=True, data=result_data)

    def _format_playbook(self, playbook: Playbook) -> dict[str, Any]:
        """Format playbook with full details."""
        result: dict[str, Any] = {
            "id": playbook.id,
            "name": playbook.name or playbook.id,
            "purpose": playbook.purpose or "",
            "description": playbook.description or "",
        }

        # Triggers
        if playbook.triggers:
            result["triggers"] = [
                t.get("condition", str(t)) if isinstance(t, dict) else str(t)
                for t in playbook.triggers
            ]

        # Inputs/Outputs
        if playbook.inputs:
            result["inputs"] = self._format_io(playbook.inputs)
        if playbook.outputs:
            result["outputs"] = self._format_io(playbook.outputs)

        # Entry phase
        if playbook.entry_phase:
            result["entry_phase"] = playbook.entry_phase

        # Phases with full details
        if playbook.phases:
            result["phases"] = self._format_phases(playbook.phases)

        # Quality criteria
        if playbook.quality_criteria:
            result["quality_criteria"] = playbook.quality_criteria

        # Rework limits
        if playbook.max_rework_cycles:
            result["max_rework_cycles"] = playbook.max_rework_cycles

        # Build human-readable summary
        result["summary"] = self._build_summary(playbook)

        return result

    def _format_io(self, io_spec: Any) -> dict[str, Any]:
        """Format input/output specification."""
        if isinstance(io_spec, dict):
            return {
                "required_artifacts": io_spec.get("required_artifacts", []),
                "optional_artifacts": io_spec.get("optional_artifacts", []),
                "context_requirements": io_spec.get("context_requirements", []),
            }
        return {}

    def _format_phases(self, phases: dict[str, Any]) -> list[dict[str, Any]]:
        """Format phases with steps and agents."""
        formatted = []

        for phase_id, phase in phases.items():
            if not isinstance(phase, dict):
                continue

            phase_info: dict[str, Any] = {
                "id": phase_id,
                "name": phase.get("name", phase_id),
                "purpose": phase.get("purpose", ""),
            }

            # Dependencies
            if phase.get("depends_on"):
                phase_info["depends_on"] = phase["depends_on"]

            # Steps
            steps = phase.get("steps", {})
            if steps:
                phase_info["steps"] = []
                for step_id, step in steps.items():
                    if not isinstance(step, dict):
                        continue

                    step_info: dict[str, Any] = {
                        "id": step_id,
                        "action": step.get("action", ""),
                    }

                    # Agent assignment
                    if step.get("specific_agent"):
                        step_info["agent"] = step["specific_agent"]
                    elif step.get("agent_archetype"):
                        step_info["agent_archetype"] = step["agent_archetype"]

                    # Guidance
                    if step.get("guidance"):
                        step_info["guidance"] = step["guidance"]

                    # Dependencies
                    if step.get("depends_on"):
                        step_info["depends_on"] = step["depends_on"]

                    # Inputs/Outputs
                    if step.get("inputs"):
                        step_info["inputs"] = step["inputs"]
                    if step.get("outputs"):
                        step_info["outputs"] = step["outputs"]

                    phase_info["steps"].append(step_info)

            # Completion criteria
            if phase.get("completion_criteria"):
                phase_info["completion_criteria"] = phase["completion_criteria"]

            # Quality checkpoint
            if phase.get("quality_checkpoint"):
                phase_info["quality_checkpoint"] = phase["quality_checkpoint"]

            # Transitions
            if phase.get("on_success"):
                phase_info["on_success"] = phase["on_success"]
            if phase.get("on_failure"):
                phase_info["on_failure"] = phase["on_failure"]

            formatted.append(phase_info)

        return formatted

    def _build_summary(self, playbook: Playbook) -> str:
        """Build human-readable summary of the playbook."""
        lines = [f"# {playbook.name or playbook.id}"]

        if playbook.purpose:
            lines.append(f"\n{playbook.purpose}")

        if playbook.phases:
            lines.append("\n## Workflow")
            for phase_id, phase in playbook.phases.items():
                if not isinstance(phase, dict):
                    continue

                phase_name = phase.get("name", phase_id)
                lines.append(f"\n### {phase_name}")

                if phase.get("purpose"):
                    lines.append(phase["purpose"])

                steps = phase.get("steps", {})
                if steps:
                    for step_id, step in steps.items():
                        if not isinstance(step, dict):
                            continue

                        agent = (
                            step.get("specific_agent") or f"[{step.get('agent_archetype', 'any')}]"
                        )
                        action = step.get("action", step_id)
                        lines.append(f"  - {agent}: {action}")

                if phase.get("completion_criteria"):
                    lines.append("\n  **Done when:**")
                    for criterion in phase["completion_criteria"]:
                        lines.append(f"  - {criterion}")

        return "\n".join(lines)
