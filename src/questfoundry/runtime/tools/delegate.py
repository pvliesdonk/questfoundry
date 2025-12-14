"""
Delegate Work tool implementation.

PARTIAL IMPLEMENTATION - Phase 2

This tool validates delegation requests and returns a delegation marker.
The actual routing and execution of delegated work is handled by the
orchestration layer in Phase 3 (#147).

What this implementation does:
- Validates the target agent exists
- Validates the task description
- Prevents self-delegation
- Creates a delegation request record
- Returns a delegation ID for tracking

What Phase 3 (#147) will add:
- Message broker integration for routing
- Delegation lifecycle management (accept, complete, reject)
- Result collection from delegate
- Timeout handling for delegations
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult, ToolValidationError
from questfoundry.runtime.tools.registry import register_tool

logger = logging.getLogger(__name__)


@register_tool("delegate")
class DelegateTool(BaseTool):
    """
    Formally assign work to another agent.

    PARTIAL IMPLEMENTATION: Creates delegation request but does not
    execute routing. See Phase 3 (#147) for full delegation orchestration.
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """
        Create a delegation request.

        NOTE: This returns a delegation ID but does NOT actually route
        the work to the target agent. That requires the orchestration
        layer from Phase 3 (#147).
        """
        to_agent = args.get("to_agent")
        to_archetype = args.get("to_archetype")
        task = args.get("task")
        context = args.get("context", {})
        expected_outputs = args.get("expected_outputs", [])
        quality_criteria = args.get("quality_criteria", [])
        priority = args.get("priority", "normal")
        playbook_ref = args.get("playbook_ref")
        phase_ref = args.get("phase_ref")

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

        # Create delegation request
        delegation_id = str(uuid.uuid4())

        # Build delegation record
        delegation = {
            "delegation_id": delegation_id,
            "from_agent": self._context.agent_id,
            "to_agent": target_agent,
            "task": task,
            "context": context,
            "expected_outputs": expected_outputs,
            "quality_criteria": quality_criteria,
            "priority": priority,
            "playbook_ref": playbook_ref,
            "phase_ref": phase_ref,
            "created_at": datetime.now().isoformat(),
            "status": "pending",  # TODO Phase 3: This will be tracked by message broker
        }

        # TODO Phase 3 (#147): Send delegation message to message broker
        # The message broker will:
        # 1. Queue the message for the target agent
        # 2. Track delegation lifecycle (pending -> accepted -> in_progress -> completed)
        # 3. Handle timeouts and retries
        # 4. Collect results from delegate
        #
        # For now, we just return the delegation ID as a marker.
        # The caller (orchestrator) can use this to track the delegation manually.

        # Persist delegation to project storage if available
        if self._context.project:
            self._persist_delegation(delegation)

        return ToolResult(
            success=True,
            data={
                "delegation_id": delegation_id,
                "status": "pending",
                "assigned_to": target_agent,
                # TODO Phase 3: Remove this warning once routing is implemented
                "_phase2_warning": (
                    "Delegation created but NOT routed. "
                    "Phase 3 (#147) orchestration required for actual execution."
                ),
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

    def _persist_delegation(self, delegation: dict[str, Any]) -> None:
        """
        Persist delegation to project storage.

        TODO Phase 3: This will be replaced by message broker persistence.
        For now, store in messages table.
        """
        if not self._context.project:
            return

        try:
            import json

            conn = self._context.project._get_connection()
            conn.execute(
                """
                INSERT INTO messages (message_id, message_type, from_agent, to_agent, payload, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delegation["delegation_id"],
                    "delegation",
                    delegation["from_agent"],
                    delegation["to_agent"],
                    json.dumps(delegation),
                    delegation["created_at"],
                    "pending",
                ),
            )
            conn.commit()
        except Exception as e:
            # Non-fatal - delegation still works, just not persisted
            logger.warning(
                "Failed to persist delegation %s: %s",
                delegation.get("delegation_id"),
                e,
            )


# =============================================================================
# TODO Phase 3 (#147): Delegation Orchestration
# =============================================================================
#
# The full delegation system requires:
#
# 1. Message Broker Integration
#    - Send delegation as message to target agent's inbox
#    - Track message delivery and acknowledgment
#    - Handle agent unavailability
#
# 2. Delegation Lifecycle
#    class DelegationStatus(Enum):
#        PENDING = "pending"      # Created, not yet delivered
#        SENT = "sent"            # Delivered to agent inbox
#        ACCEPTED = "accepted"    # Agent acknowledged receipt
#        IN_PROGRESS = "in_progress"  # Agent working on it
#        COMPLETED = "completed"  # Work done, results available
#        REJECTED = "rejected"    # Agent declined
#        FAILED = "failed"        # Error during execution
#        TIMED_OUT = "timed_out"  # Exceeded deadline
#
# 3. Result Collection
#    - Delegate posts result message when complete
#    - Orchestrator retrieves result
#    - Quality criteria validation on result
#
# 4. Timeout Handling
#    - Configurable timeout per delegation
#    - Automatic escalation on timeout
#    - Retry with different agent option
#
# 5. Bouncer Pattern
#    - Rate limiting delegations per agent
#    - Priority queue management
#    - Backpressure handling
#
# See: domain-v4/schemas/delegation.schema.json (if exists)
# See: meta/docs/delegation-protocol.md (if exists)
