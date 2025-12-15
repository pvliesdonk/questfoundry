"""
Async delegation executor.

Executes delegations by activating delegatee agents and handling
the full delegation lifecycle including response handling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from questfoundry.runtime.messaging.message import (
    create_delegation_response,
    create_escalation,
)
from questfoundry.runtime.messaging.types import MessageType

if TYPE_CHECKING:
    from questfoundry.runtime.delegation.bouncer import DelegationBouncer
    from questfoundry.runtime.delegation.tracker import PlaybookTracker
    from questfoundry.runtime.messaging.broker import AsyncMessageBroker
    from questfoundry.runtime.messaging.message import Message


class AgentActivator(Protocol):
    """
    Protocol for activating agents.

    The actual implementation is in agent/runtime.py, but we define
    the interface here to avoid circular imports.
    """

    async def activate(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Activate an agent with the given context.

        Args:
            agent_id: Agent to activate
            context: Execution context including:
                - task: The delegation task
                - inbox: Pending messages
                - artifacts: Relevant artifacts
                - playbook_context: Current playbook state

        Returns:
            Agent execution result including:
                - success: Whether execution succeeded
                - result: Output data
                - artifacts_produced: IDs of created artifacts
                - error: Error message if failed
        """
        ...


@dataclass
class DelegationContext:
    """Context for a delegation execution."""

    delegation_id: str
    from_agent: str
    to_agent: str
    task: str
    context: dict[str, Any] = field(default_factory=dict)

    # Playbook context (optional)
    playbook_id: str | None = None
    playbook_instance_id: str | None = None
    phase_id: str | None = None
    is_rework_target: bool = False

    # Original message reference
    request_message_id: str | None = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=None))


@dataclass
class DelegationResult:
    """Result of a delegation execution."""

    delegation_id: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    artifacts_produced: list[str] = field(default_factory=list)
    error: str | None = None

    # Response message
    response_message: Message | None = None

    # Timing
    completed_at: datetime = field(default_factory=lambda: datetime.now(tz=None))
    duration_seconds: float = 0.0


class AsyncDelegationExecutor:
    """
    Executes delegations asynchronously.

    The executor handles:
    1. Pre-flight checks via bouncer
    2. Context building for delegatee
    3. Playbook phase tracking
    4. Agent activation
    5. Response message creation
    6. Error handling and escalation

    This is designed to work with async primitives throughout,
    no synchronous blocking.
    """

    def __init__(
        self,
        broker: AsyncMessageBroker,
        bouncer: DelegationBouncer,
        tracker: PlaybookTracker,
        activator: AgentActivator | None = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            broker: Message broker for sending responses
            bouncer: Pre-flight check bouncer
            tracker: Playbook state tracker
            activator: Agent activator (can be set later via set_activator)
        """
        self._broker = broker
        self._bouncer = bouncer
        self._tracker = tracker
        self._activator = activator
        self._pending: dict[str, asyncio.Event] = {}

    def set_activator(self, activator: AgentActivator) -> None:
        """Set the agent activator (for late binding)."""
        self._activator = activator

    async def execute(
        self,
        delegation: Message,
        *,
        timeout: float | None = None,
    ) -> DelegationResult:
        """
        Execute a delegation request.

        This is the main entry point for processing delegation_request
        messages. It:
        1. Extracts context from the message
        2. Performs bouncer checks
        3. Builds delegatee context
        4. Records playbook phase entry if applicable
        5. Activates the delegatee agent
        6. Creates and sends response message

        Args:
            delegation: The delegation_request message
            timeout: Optional timeout in seconds

        Returns:
            DelegationResult with execution outcome
        """
        start_time = datetime.now(tz=None)

        # Extract delegation context
        ctx = self._extract_context(delegation)

        # Create completion event for this delegation
        completion_event = asyncio.Event()
        self._pending[ctx.delegation_id] = completion_event

        try:
            # Perform bouncer checks
            bouncer_result = await self._bouncer.check(
                from_agent=ctx.from_agent,
                to_agent=ctx.to_agent,
                broker=self._broker,
                playbook_tracker=self._tracker if ctx.playbook_instance_id else None,
                playbook_instance_id=ctx.playbook_instance_id,
            )

            if not bouncer_result.allowed:
                if bouncer_result.should_escalate:
                    return await self._handle_escalation(
                        ctx,
                        bouncer_result.reason or "Delegation rejected",
                        bouncer_result.escalation_payload,
                        start_time,
                    )
                else:
                    return self._create_failure_result(
                        ctx,
                        bouncer_result.reason or "Delegation rejected",
                        start_time,
                    )

            # Track playbook phase entry if in playbook context
            if ctx.playbook_instance_id and ctx.phase_id:
                phase_result = await self._tracker.record_phase_entry(
                    ctx.playbook_instance_id,
                    ctx.phase_id,
                    ctx.is_rework_target,
                )
                if not phase_result.allowed and phase_result.should_escalate:
                    return await self._handle_escalation(
                        ctx,
                        phase_result.reason or "Playbook budget exhausted",
                        phase_result.escalation_details,
                        start_time,
                    )

            # Build delegatee context
            delegatee_context = await self._build_delegatee_context(ctx)

            # Activate delegatee agent
            if self._activator is None:
                return self._create_failure_result(
                    ctx,
                    "No agent activator configured",
                    start_time,
                )

            activation_result = await self._run_with_timeout(
                self._activator.activate(ctx.to_agent, delegatee_context),
                timeout,
            )

            # Create result
            result = DelegationResult(
                delegation_id=ctx.delegation_id,
                success=activation_result.get("success", False),
                result=activation_result.get("result", {}),
                artifacts_produced=activation_result.get("artifacts_produced", []),
                error=activation_result.get("error"),
                completed_at=datetime.now(tz=None),
                duration_seconds=(datetime.now(tz=None) - start_time).total_seconds(),
            )

            # Create and send response message
            response_msg = create_delegation_response(
                from_agent=ctx.to_agent,
                to_agent=ctx.from_agent,
                delegation_id=ctx.delegation_id,
                success=result.success,
                result=result.result,
                error=result.error,
                artifacts_produced=result.artifacts_produced,
                in_reply_to=ctx.request_message_id,
                playbook_id=ctx.playbook_id,
                playbook_instance_id=ctx.playbook_instance_id,
                phase_id=ctx.phase_id,
            )
            await self._broker.send(response_msg)
            result.response_message = response_msg

            return result

        except TimeoutError:
            return self._create_failure_result(
                ctx,
                f"Delegation timed out after {timeout}s",
                start_time,
            )
        except Exception as e:
            return self._create_failure_result(
                ctx,
                f"Delegation failed: {e!s}",
                start_time,
            )
        finally:
            # Clean up pending tracking
            self._pending.pop(ctx.delegation_id, None)
            completion_event.set()

    async def execute_from_message(
        self,
        message: Message,
        *,
        timeout: float | None = None,
    ) -> DelegationResult:
        """
        Execute a delegation from a Message object.

        Convenience wrapper around execute() that validates
        the message type first.

        Args:
            message: Delegation request message
            timeout: Optional timeout

        Returns:
            DelegationResult
        """
        if message.type != MessageType.DELEGATION_REQUEST:
            return DelegationResult(
                delegation_id=message.delegation_id or "unknown",
                success=False,
                error=f"Expected DELEGATION_REQUEST, got {message.type.value}",
            )
        return await self.execute(message, timeout=timeout)

    def _extract_context(self, delegation: Message) -> DelegationContext:
        """Extract DelegationContext from a message."""
        if not delegation.to_agent:
            raise ValueError("Delegation message must have a to_agent")
        payload = delegation.payload
        return DelegationContext(
            delegation_id=delegation.delegation_id or delegation.id,
            from_agent=delegation.from_agent,
            to_agent=delegation.to_agent,
            task=payload.get("task", ""),
            context=payload.get("context", {}),
            playbook_id=delegation.playbook_id,
            playbook_instance_id=delegation.playbook_instance_id,
            phase_id=delegation.phase_id,
            is_rework_target=payload.get("is_rework_target", False),
            request_message_id=delegation.id,
        )

    async def _build_delegatee_context(
        self,
        ctx: DelegationContext,
    ) -> dict[str, Any]:
        """
        Build the context for the delegatee agent.

        This includes:
        - The delegation task
        - Pending inbox messages
        - Relevant artifacts from context
        - Playbook state
        """
        # Get pending messages for the delegatee
        inbox = await self._broker.get_inbox(ctx.to_agent)

        return {
            "task": ctx.task,
            "delegation_id": ctx.delegation_id,
            "from_agent": ctx.from_agent,
            "context": ctx.context,
            "inbox": [msg.to_dict() for msg in inbox],
            "playbook_context": {
                "playbook_id": ctx.playbook_id,
                "instance_id": ctx.playbook_instance_id,
                "phase_id": ctx.phase_id,
            }
            if ctx.playbook_id
            else None,
        }

    async def _run_with_timeout(
        self,
        coro: Any,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Run a coroutine with optional timeout."""
        if timeout is not None:
            result: dict[str, Any] = await asyncio.wait_for(coro, timeout=timeout)
            return result
        return await coro  # type: ignore[no-any-return]

    async def _handle_escalation(
        self,
        ctx: DelegationContext,
        reason: str,
        details: dict[str, Any] | None,
        start_time: datetime,
    ) -> DelegationResult:
        """Handle escalation by creating and sending escalation message."""
        # Mark playbook as escalated if applicable
        if ctx.playbook_instance_id:
            await self._tracker.escalate_playbook(ctx.playbook_instance_id)

        # Create escalation message
        escalation_msg = create_escalation(
            from_agent=ctx.to_agent,
            to_agent=ctx.from_agent,
            reason=reason,
            details=f"Delegation {ctx.delegation_id} requires escalation",
            playbook_id=ctx.playbook_id,
            playbook_instance_id=ctx.playbook_instance_id,
            phase_id=ctx.phase_id,
            rework_count=details.get("rework_count") if details else None,
            attempted_resolutions=list(details.get("rework_target_visits", {}).keys())
            if details
            else None,
            suggested_action="Orchestrator review required",
        )
        await self._broker.send(escalation_msg)

        # Also send a failure response
        response_msg = create_delegation_response(
            from_agent=ctx.to_agent,
            to_agent=ctx.from_agent,
            delegation_id=ctx.delegation_id,
            success=False,
            error=f"Escalated: {reason}",
            in_reply_to=ctx.request_message_id,
            playbook_id=ctx.playbook_id,
            playbook_instance_id=ctx.playbook_instance_id,
            phase_id=ctx.phase_id,
        )
        await self._broker.send(response_msg)

        return DelegationResult(
            delegation_id=ctx.delegation_id,
            success=False,
            error=f"Escalated: {reason}",
            response_message=response_msg,
            completed_at=datetime.now(tz=None),
            duration_seconds=(datetime.now(tz=None) - start_time).total_seconds(),
        )

    def _create_failure_result(
        self,
        ctx: DelegationContext,
        error: str,
        start_time: datetime,
    ) -> DelegationResult:
        """Create a failure result without sending messages."""
        return DelegationResult(
            delegation_id=ctx.delegation_id,
            success=False,
            error=error,
            completed_at=datetime.now(tz=None),
            duration_seconds=(datetime.now(tz=None) - start_time).total_seconds(),
        )

    async def wait_for_completion(
        self,
        delegation_id: str,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait for a delegation to complete.

        Args:
            delegation_id: ID of the delegation to wait for
            timeout: Optional timeout in seconds

        Returns:
            True if completed, False if timed out or not found
        """
        event = self._pending.get(delegation_id)
        if event is None:
            return False

        try:
            if timeout is not None:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
            return True
        except TimeoutError:
            return False

    def get_pending_count(self) -> int:
        """Get count of pending delegations."""
        return len(self._pending)

    def get_pending_ids(self) -> list[str]:
        """Get IDs of pending delegations."""
        return list(self._pending.keys())
