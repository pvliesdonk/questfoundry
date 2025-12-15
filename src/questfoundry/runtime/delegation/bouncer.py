"""
Delegation bouncer.

Pre-flight checks for delegations including:
- Concurrent delegation limits (per agent)
- Playbook rework budgets
- Agent existence validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.delegation.tracker import PlaybookTracker
    from questfoundry.runtime.messaging.broker import AsyncMessageBroker


# Default limits when not specified in agent definition
DEFAULT_MAX_ACTIVE_DELEGATIONS = 5
DEFAULT_MAX_INBOX_SIZE = 20


@dataclass
class AgentFlowControl:
    """Flow control settings for an agent."""

    max_inbox_size: int = DEFAULT_MAX_INBOX_SIZE
    max_active_delegations: int = DEFAULT_MAX_ACTIVE_DELEGATIONS
    priority_boost: int = 0


@dataclass
class BouncerResult:
    """Result of a delegation pre-flight check."""

    allowed: bool
    reason: str | None = None
    should_escalate: bool = False
    escalation_payload: dict[str, Any] | None = None

    @classmethod
    def allow(cls) -> BouncerResult:
        """Create an allowed result."""
        return cls(allowed=True)

    @classmethod
    def reject(
        cls,
        reason: str,
        *,
        should_escalate: bool = False,
        escalation_payload: dict[str, Any] | None = None,
    ) -> BouncerResult:
        """Create a rejection result."""
        return cls(
            allowed=False,
            reason=reason,
            should_escalate=should_escalate,
            escalation_payload=escalation_payload,
        )


class DelegationBouncer:
    """
    Pre-flight checks for delegations.

    The bouncer enforces:
    1. Concurrent delegation limits - An agent can only have N active delegations
    2. Playbook rework budgets - Track rework cycles per playbook instance
    3. Target agent existence - Validate the target agent exists

    This is called BEFORE a delegation is sent to ensure the system
    doesn't get overloaded or stuck in infinite loops.
    """

    def __init__(
        self,
        agent_configs: dict[str, AgentFlowControl] | None = None,
    ) -> None:
        """
        Initialize the bouncer.

        Args:
            agent_configs: Flow control settings per agent ID.
                           If not provided, uses defaults.
        """
        self._agent_configs = agent_configs or {}

    def get_flow_control(self, agent_id: str) -> AgentFlowControl:
        """Get flow control settings for an agent."""
        return self._agent_configs.get(agent_id, AgentFlowControl())

    def set_flow_control(
        self,
        agent_id: str,
        config: AgentFlowControl,
    ) -> None:
        """Set flow control settings for an agent."""
        self._agent_configs[agent_id] = config

    async def check(
        self,
        from_agent: str,
        to_agent: str,
        broker: AsyncMessageBroker,
        playbook_tracker: PlaybookTracker | None = None,
        playbook_instance_id: str | None = None,
    ) -> BouncerResult:
        """
        Check if a delegation is allowed.

        Performs the following checks in order:
        1. Target agent concurrent delegation limit
        2. Playbook rework budget (if in playbook context)

        Args:
            from_agent: Agent initiating the delegation
            to_agent: Agent receiving the delegation
            broker: Message broker for checking current delegations
            playbook_tracker: Tracker for playbook budgets (optional)
            playbook_instance_id: Current playbook instance (optional)

        Returns:
            BouncerResult indicating whether delegation is allowed
        """
        # Check concurrent delegation limit
        concurrent_result = await self._check_concurrent_limit(
            to_agent,
            broker,
        )
        if not concurrent_result.allowed:
            return concurrent_result

        # Check playbook rework budget
        if playbook_tracker is not None and playbook_instance_id is not None:
            budget_result = await self._check_playbook_budget(
                from_agent,
                to_agent,
                playbook_tracker,
                playbook_instance_id,
            )
            if not budget_result.allowed:
                return budget_result

        return BouncerResult.allow()

    async def _check_concurrent_limit(
        self,
        agent_id: str,
        broker: AsyncMessageBroker,
    ) -> BouncerResult:
        """Check if agent has capacity for more delegations."""
        config = self.get_flow_control(agent_id)

        # Get current active delegations for this agent
        active_count = await broker.get_active_delegations(agent_id)

        if active_count >= config.max_active_delegations:
            return BouncerResult.reject(
                f"Agent '{agent_id}' has reached concurrent delegation limit "
                f"({active_count}/{config.max_active_delegations})",
                should_escalate=False,  # Not an error, just temporary overload
            )

        return BouncerResult.allow()

    async def _check_playbook_budget(
        self,
        from_agent: str,  # noqa: ARG002
        to_agent: str,  # noqa: ARG002
        tracker: PlaybookTracker,
        instance_id: str,
    ) -> BouncerResult:
        """Check if playbook has rework budget remaining."""
        budget_check = await tracker.check_rework_budget(instance_id)

        if not budget_check.allowed:
            return BouncerResult.reject(
                budget_check.reason or "Playbook budget exhausted",
                should_escalate=budget_check.should_escalate,
                escalation_payload=budget_check.escalation_details,
            )

        return BouncerResult.allow()

    async def check_inbox_capacity(
        self,
        agent_id: str,
        broker: AsyncMessageBroker,
    ) -> BouncerResult:
        """
        Check if agent's inbox has capacity for more messages.

        This is a softer check than delegation limits - it helps
        prevent message queue buildup.

        Args:
            agent_id: Agent to check
            broker: Message broker

        Returns:
            BouncerResult indicating whether inbox has capacity
        """
        config = self.get_flow_control(agent_id)
        pending_count = await broker.get_pending_count(agent_id)

        if pending_count >= config.max_inbox_size:
            return BouncerResult.reject(
                f"Agent '{agent_id}' inbox is full ({pending_count}/{config.max_inbox_size})",
            )

        return BouncerResult.allow()


def create_bouncer_from_agent_defs(
    agent_definitions: list[dict[str, Any]],
) -> DelegationBouncer:
    """
    Create a DelegationBouncer from agent JSON definitions.

    Extracts flow_control_override settings from each agent.

    Args:
        agent_definitions: List of agent JSON definitions

    Returns:
        Configured DelegationBouncer
    """
    configs: dict[str, AgentFlowControl] = {}

    for agent_def in agent_definitions:
        agent_id = agent_def.get("id")
        if not agent_id:
            continue

        flow_control = agent_def.get("flow_control_override", {})
        configs[agent_id] = AgentFlowControl(
            max_inbox_size=flow_control.get(
                "max_inbox_size",
                DEFAULT_MAX_INBOX_SIZE,
            ),
            max_active_delegations=flow_control.get(
                "max_active_delegations",
                DEFAULT_MAX_ACTIVE_DELEGATIONS,
            ),
            priority_boost=flow_control.get("priority_boost", 0),
        )

    return DelegationBouncer(agent_configs=configs)
