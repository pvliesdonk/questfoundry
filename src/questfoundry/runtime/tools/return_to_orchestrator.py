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
                - status: complete, partial, blocked, needs_decision
                - summary: Brief summary of what was accomplished
                - artifacts_produced: IDs of artifacts created/updated
                - artifacts_ready_for_review: IDs ready for quality gating
                - blockers: What's preventing progress (if blocked)
                - recommendations: Suggestions for next steps
        """
        status = args.get("status")
        summary = args.get("summary")
        artifacts_produced = args.get("artifacts_produced", [])
        artifacts_ready_for_review = args.get("artifacts_ready_for_review", [])
        blockers = args.get("blockers", [])
        recommendations = args.get("recommendations")

        # Validate required fields
        if not status:
            raise ToolValidationError("Status is required")
        if not summary:
            raise ToolValidationError("Summary is required")

        # Find the orchestrator agent
        orchestrator_id = self._find_orchestrator()
        if not orchestrator_id:
            return ToolResult(
                success=False,
                data={},
                error="No orchestrator agent found in studio",
            )

        # Determine success based on status
        success = status in ("complete", "partial")

        # Build result data (artifacts_produced passed separately to create_delegation_response)
        result_data = {
            "status": status,
            "summary": summary,
            "artifacts_ready_for_review": artifacts_ready_for_review,
        }
        if blockers:
            result_data["blockers"] = blockers
        if recommendations:
            result_data["recommendations"] = recommendations

        # Build error string for blocked/needs_decision
        error_msg = None
        if status == "blocked" and blockers:
            error_msg = "; ".join(
                (b.get("description") or str(b)) if isinstance(b, dict) else str(b)
                for b in blockers
            )
        elif status == "needs_decision":
            error_msg = f"Needs orchestrator decision: {summary}"

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
                "Return to orchestrator: %s -> %s (status: %s, delegation: %s)",
                self._context.agent_id,
                delegator_id,
                status,
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
                "status": status,
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
