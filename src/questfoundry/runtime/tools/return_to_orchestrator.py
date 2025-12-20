"""
Return to Orchestrator tool implementation.

Allows specialist agents to signal completion and return control
to the orchestrator (Showrunner) after completing delegated work.

This enforces the hub-and-spoke delegation pattern where:
- Orchestrator delegates to specialists
- Specialists complete work and return to orchestrator
- Orchestrator decides next steps
"""

from __future__ import annotations

import logging
from typing import Any

from questfoundry.runtime.messaging import create_delegation_response
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool("return_to_orchestrator")
class ReturnToOrchestratorTool(BaseTool):
    """
    Signal task completion and return control to the orchestrator.

    This is the standard way for specialist agents to end their turn
    after completing delegated work. The orchestrator will receive
    the summary and decide next steps.

    NOTE: This tool is for NON-ORCHESTRATOR agents only.
    Orchestrators use 'delegate' or 'communicate' to end their turns.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Create and route a delegation response to the orchestrator.

        Args:
            args: Tool arguments including:
                - summary: Brief summary of what was accomplished
                - result_assessment: pass, partial, fail, skip, info
                - recommendation: proceed, rework, escalate, hold
                - artifacts_produced: IDs of artifacts created/updated
                - artifacts_ready_for_review: IDs ready for quality gating
                - blockers: What's preventing progress (if any)
        """
        summary = args.get("summary")
        result_assessment = args.get("result_assessment")
        recommendation = args.get("recommendation")
        artifacts_produced = args.get("artifacts_produced", [])
        artifacts_ready_for_review = args.get("artifacts_ready_for_review", [])
        blockers = args.get("blockers", [])

        # Validate required fields
        if not summary:
            raise ToolValidationError("summary is required")
        if not result_assessment:
            raise ToolValidationError("result_assessment is required")
        if not recommendation:
            raise ToolValidationError("recommendation is required")

        # Validate enum values
        valid_assessments = ("pass", "partial", "fail", "skip", "info")
        if result_assessment not in valid_assessments:
            raise ToolValidationError(
                f"result_assessment must be one of {valid_assessments}, got '{result_assessment}'"
            )

        valid_recommendations = ("proceed", "rework", "escalate", "hold")
        if recommendation not in valid_recommendations:
            raise ToolValidationError(
                f"recommendation must be one of {valid_recommendations}, got '{recommendation}'"
            )

        # Find the orchestrator agent
        orchestrator_id = self._find_orchestrator()
        if not orchestrator_id:
            return ToolResult(
                success=False,
                data={},
                error="No orchestrator agent found in studio",
            )

        # Derive task_completion from blockers and recommendation
        # If there are blockers and recommendation is hold/escalate, task is blocked
        # Otherwise, the task completed (even if results are partial/fail)
        if blockers and recommendation in ("hold", "escalate"):
            task_completion = "blocked"
        else:
            task_completion = "completed"

        # Derive success from task_completion (for message payload)
        success = task_completion == "completed"

        # Build result data with new structure
        result_data = {
            "task_completion": task_completion,
            "result": {
                "assessment": result_assessment,
                "summary": summary,
            },
            "recommendation": recommendation,
            "artifacts_ready_for_review": artifacts_ready_for_review,
        }
        if blockers:
            result_data["result"]["details"] = blockers

        # Build error string for blocked status
        error_msg = None
        if task_completion == "blocked" and blockers:
            error_msg = "; ".join(
                (b.get("description") or str(b)) if isinstance(b, dict) else str(b)
                for b in blockers
            )

        # Get delegation context from tool context (if available)
        delegation_id = self._context.delegation_id or "implicit"
        delegator_id = self._context.delegator_agent_id or orchestrator_id

        # Create delegation response message
        message = create_delegation_response(
            from_agent=self._context.agent_id or "unknown",
            to_agent=delegator_id,
            delegation_id=delegation_id,
            success=success,
            result=result_data,
            error=error_msg,
            artifacts_produced=artifacts_produced,
        )

        # Route via broker if available
        if self._context.broker:
            await self._context.broker.send(message)
            logger.info(
                "Return to orchestrator: %s -> %s (assessment: %s, recommendation: %s, delegation: %s)",
                self._context.agent_id,
                delegator_id,
                result_assessment,
                recommendation,
                delegation_id,
            )
        else:
            logger.warning("No broker available - return message created but not routed")

        return ToolResult(
            success=True,
            data={
                "acknowledged": True,
                "message_id": message.id,
                "returned_to": delegator_id,
                "task_completion": task_completion,
                "result_assessment": result_assessment,
                "recommendation": recommendation,
            },
        )

    def _find_orchestrator(self) -> str | None:
        """
        Find the orchestrator agent in the studio.

        Returns the ID of the first agent with the 'orchestrator' archetype.
        """
        for agent in self._context.studio.agents:
            if "orchestrator" in agent.archetypes:
                return agent.id
        return None
