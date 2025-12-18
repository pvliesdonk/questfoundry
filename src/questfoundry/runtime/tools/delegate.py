"""
Delegate Work tool implementation.

Routes delegation requests to target agents via the message broker.
The broker handles:
- Message routing to target agent's inbox
- Persistence to SQLite and JSONL logging
- Priority-based delivery

The actual execution of delegated work is handled by the
AsyncDelegationExecutor which:
- Performs bouncer checks (concurrent limits, playbook budgets)
- Activates the delegatee agent
- Tracks playbook phase entries
- Creates and routes response messages
"""

from __future__ import annotations

import logging
from typing import Any

from questfoundry.runtime.delegation import PlaybookLoader
from questfoundry.runtime.messaging import create_delegation_request
from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool("delegate")
class DelegateTool(BaseTool):
    """
    Formally assign work to another agent.

    Creates a delegation request message and routes it to the target
    agent's inbox via the message broker.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Create and route a delegation request.

        Creates a delegation_request message and sends it via the broker.
        The AsyncDelegationExecutor handles the actual execution.
        """
        to_agent = args.get("to_agent")
        to_archetype = args.get("to_archetype")
        task = args.get("task")
        context = args.get("context", {})
        playbook_ref = args.get("playbook_ref")
        playbook_instance_id = args.get("playbook_instance_id")
        phase_ref = args.get("phase_ref")

        # Auto-resolve is_rework_target from playbook definition if not explicitly set
        is_rework_target = args.get("is_rework_target")
        if is_rework_target is None and playbook_ref and phase_ref:
            is_rework_target = self._lookup_is_rework_target(playbook_ref, phase_ref)
        is_rework_target = is_rework_target or False

        # Validate: must have task
        if not task:
            raise ToolValidationError("Task description is required")

        # Validate: must specify target
        if not to_agent and not to_archetype:
            raise ToolValidationError("Must specify either 'to_agent' or 'to_archetype'")

        # Resolve target agent
        target_agent = self._resolve_target_agent(to_agent, to_archetype)

        if not target_agent:
            return ToolResult(
                success=False,
                data={},
                error=f"Target agent not found: {to_agent or to_archetype}",
            )

        # Prevent self-delegation
        if self._context.agent_id and target_agent == self._context.agent_id:
            return ToolResult(
                success=False,
                data={},
                error="Cannot delegate to self",
            )

        # Create delegation request message
        message = create_delegation_request(
            from_agent=self._context.agent_id or "unknown",
            to_agent=target_agent,
            task=task,
            context=context,
            playbook_id=playbook_ref,
            playbook_instance_id=playbook_instance_id,
            phase_id=phase_ref,
            is_rework_target=is_rework_target,
        )

        # Route via broker if available
        if self._context.broker:
            await self._context.broker.send(message)
            logger.info(
                "Delegation %s sent: %s -> %s (task: %s)",
                message.delegation_id,
                self._context.agent_id,
                target_agent,
                task[:50] + "..." if len(task) > 50 else task,
            )
        else:
            # No broker available - log warning but don't fail
            logger.warning(
                "No broker available - delegation %s created but not routed",
                message.delegation_id,
            )

        return ToolResult(
            success=True,
            data={
                "delegation_id": message.delegation_id,
                "message_id": message.id,
                "status": "sent" if self._context.broker else "created",
                "assigned_to": target_agent,
            },
        )

    def _resolve_target_agent(self, to_agent: str | None, to_archetype: str | None) -> str | None:
        """
        Resolve target agent ID.

        If to_agent specified, validates it exists.
        If to_archetype specified, finds first matching agent.
        """
        if to_agent:
            # Direct agent reference - validate it exists
            for agent in self._context.studio.agents:
                if agent.id == to_agent:
                    return to_agent
            return None

        if to_archetype:
            # Find agent by archetype
            for agent in self._context.studio.agents:
                if to_archetype in agent.archetypes:
                    return agent.id
            return None

        return None

    def _lookup_is_rework_target(self, playbook_id: str, phase_id: str) -> bool:
        """
        Look up whether a phase is a rework target from playbook definitions.

        Phases marked as rework targets have their re-entries counted against
        the playbook's rework budget. When the budget is exhausted, re-entering
        these phases triggers escalation.

        Args:
            playbook_id: ID of the playbook
            phase_id: ID of the phase

        Returns:
            True if the phase is a rework target, False otherwise
        """
        loader = PlaybookLoader.from_playbooks(self._context.studio.playbooks)
        return loader.is_rework_target(playbook_id, phase_id)
