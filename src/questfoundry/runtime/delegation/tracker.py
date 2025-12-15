"""
Playbook execution tracker.

Tracks playbook instances and their rework budget consumption.
Uses domain-aligned termination via playbook-scoped rework cycles,
not simplistic depth limits.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from questfoundry.runtime.messaging.types import PlaybookStatus


@dataclass
class PlaybookInstance:
    """
    Tracks a single playbook execution instance.

    Each playbook has a max_rework_cycles budget. The tracker counts
    how many times rework_target phases are entered. When budget is
    exhausted, escalation is triggered.
    """

    playbook_id: str
    instance_id: str
    max_rework_cycles: int

    # Rework tracking
    rework_count: int = 0
    current_phase: str | None = None
    rework_target_visits: dict[str, int] = field(default_factory=dict)

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=None))
    completed_at: datetime | None = None

    # Status
    status: PlaybookStatus = PlaybookStatus.ACTIVE

    # Context
    parent_delegation_id: str | None = None
    initiating_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "playbook_id": self.playbook_id,
            "instance_id": self.instance_id,
            "max_rework_cycles": self.max_rework_cycles,
            "rework_count": self.rework_count,
            "current_phase": self.current_phase,
            "rework_target_visits": self.rework_target_visits,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "parent_delegation_id": self.parent_delegation_id,
            "initiating_agent": self.initiating_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaybookInstance:
        """Deserialize from dictionary."""
        return cls(
            playbook_id=data["playbook_id"],
            instance_id=data["instance_id"],
            max_rework_cycles=data["max_rework_cycles"],
            rework_count=data.get("rework_count", 0),
            current_phase=data.get("current_phase"),
            rework_target_visits=data.get("rework_target_visits", {}),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            status=PlaybookStatus(data.get("status", "active")),
            parent_delegation_id=data.get("parent_delegation_id"),
            initiating_agent=data.get("initiating_agent"),
        )


@dataclass
class BudgetCheckResult:
    """Result of a rework budget check."""

    allowed: bool
    remaining: int  # Remaining rework cycles
    reason: str | None = None  # Reason if not allowed
    should_escalate: bool = False
    escalation_details: dict[str, Any] | None = None


class PlaybookTracker:
    """
    Tracks playbook execution state and rework budgets.

    This is the core mechanism for domain-aligned loop termination.
    Instead of simplistic depth limits, we track rework cycles per
    playbook instance. Each playbook defines max_rework_cycles (typically 1-3),
    and we count visits to is_rework_target phases.

    When a section legitimately needs 5-8 draft↔review cycles, that's fine
    as long as the playbook's rework budget isn't exhausted. When it is,
    we escalate rather than failing silently.
    """

    def __init__(self) -> None:
        self._instances: dict[str, PlaybookInstance] = {}
        self._lock = asyncio.Lock()
        # Track active instances by playbook_id for querying
        self._by_playbook: dict[str, set[str]] = {}

    async def start_playbook(
        self,
        playbook_id: str,
        max_rework_cycles: int,
        *,
        parent_delegation_id: str | None = None,
        initiating_agent: str | None = None,
    ) -> PlaybookInstance:
        """
        Start a new playbook execution instance.

        Args:
            playbook_id: The playbook being executed
            max_rework_cycles: Maximum rework cycles allowed (from playbook def)
            parent_delegation_id: Delegation that triggered this playbook
            initiating_agent: Agent that started the playbook

        Returns:
            New PlaybookInstance with unique ID
        """
        instance_id = str(uuid.uuid4())
        instance = PlaybookInstance(
            playbook_id=playbook_id,
            instance_id=instance_id,
            max_rework_cycles=max_rework_cycles,
            parent_delegation_id=parent_delegation_id,
            initiating_agent=initiating_agent,
        )

        async with self._lock:
            self._instances[instance_id] = instance
            if playbook_id not in self._by_playbook:
                self._by_playbook[playbook_id] = set()
            self._by_playbook[playbook_id].add(instance_id)

        return instance

    async def get_instance(self, instance_id: str) -> PlaybookInstance | None:
        """Get a playbook instance by ID."""
        async with self._lock:
            return self._instances.get(instance_id)

    async def record_phase_entry(
        self,
        instance_id: str,
        phase_id: str,
        is_rework_target: bool,
    ) -> BudgetCheckResult:
        """
        Record entry into a playbook phase.

        If the phase is a rework target (is_rework_target: true in playbook),
        this counts against the rework budget. Returns whether the operation
        is allowed and if budget is exhausted.

        Args:
            instance_id: Playbook instance ID
            phase_id: Phase being entered
            is_rework_target: Whether this phase is a rework target

        Returns:
            BudgetCheckResult with allowed status and remaining budget
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return BudgetCheckResult(
                    allowed=False,
                    remaining=0,
                    reason=f"Unknown playbook instance: {instance_id}",
                )

            if instance.status != PlaybookStatus.ACTIVE:
                return BudgetCheckResult(
                    allowed=False,
                    remaining=0,
                    reason=f"Playbook instance is not active: {instance.status.value}",
                )

            # Track the phase entry
            previous_phase = instance.current_phase
            instance.current_phase = phase_id

            # Only count rework if re-entering a rework target phase
            if is_rework_target:
                visits = instance.rework_target_visits.get(phase_id, 0)
                if visits > 0:
                    # This is a rework (re-entry), increment count
                    instance.rework_count += 1
                instance.rework_target_visits[phase_id] = visits + 1

            # Check budget
            remaining = instance.max_rework_cycles - instance.rework_count
            if remaining < 0:
                # Budget exhausted - should escalate
                return BudgetCheckResult(
                    allowed=False,
                    remaining=0,
                    reason="max_rework_exceeded",
                    should_escalate=True,
                    escalation_details={
                        "playbook_id": instance.playbook_id,
                        "instance_id": instance_id,
                        "phase_id": phase_id,
                        "rework_count": instance.rework_count,
                        "max_rework_cycles": instance.max_rework_cycles,
                        "rework_target_visits": instance.rework_target_visits.copy(),
                        "previous_phase": previous_phase,
                    },
                )

            return BudgetCheckResult(
                allowed=True,
                remaining=remaining,
            )

    async def check_rework_budget(
        self,
        instance_id: str,
    ) -> BudgetCheckResult:
        """
        Check rework budget without recording a phase entry.

        Args:
            instance_id: Playbook instance ID

        Returns:
            BudgetCheckResult with current budget status
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return BudgetCheckResult(
                    allowed=False,
                    remaining=0,
                    reason=f"Unknown playbook instance: {instance_id}",
                )

            remaining = instance.max_rework_cycles - instance.rework_count
            if remaining < 0:
                return BudgetCheckResult(
                    allowed=False,
                    remaining=0,
                    reason="max_rework_exceeded",
                    should_escalate=True,
                    escalation_details={
                        "playbook_id": instance.playbook_id,
                        "instance_id": instance_id,
                        "phase_id": instance.current_phase,
                        "rework_count": instance.rework_count,
                        "max_rework_cycles": instance.max_rework_cycles,
                        "rework_target_visits": instance.rework_target_visits.copy(),
                    },
                )

            return BudgetCheckResult(
                allowed=True,
                remaining=remaining,
            )

    async def complete_playbook(
        self,
        instance_id: str,
        status: PlaybookStatus = PlaybookStatus.COMPLETED,
    ) -> bool:
        """
        Mark a playbook instance as complete.

        Args:
            instance_id: Playbook instance ID
            status: Final status (COMPLETED, ESCALATED, FAILED, CANCELLED)

        Returns:
            True if successfully completed, False if not found
        """
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                return False

            instance.status = status
            instance.completed_at = datetime.now(tz=None)
            return True

    async def escalate_playbook(
        self,
        instance_id: str,
    ) -> bool:
        """
        Mark a playbook instance as escalated (budget exhausted).

        Args:
            instance_id: Playbook instance ID

        Returns:
            True if successfully escalated, False if not found
        """
        return await self.complete_playbook(instance_id, PlaybookStatus.ESCALATED)

    async def get_active_instances(
        self,
        playbook_id: str | None = None,
    ) -> list[PlaybookInstance]:
        """
        Get all active playbook instances.

        Args:
            playbook_id: Optional filter by playbook ID

        Returns:
            List of active playbook instances
        """
        async with self._lock:
            if playbook_id is not None:
                instance_ids = self._by_playbook.get(playbook_id, set())
                return [
                    inst
                    for inst_id in instance_ids
                    if (inst := self._instances.get(inst_id))
                    and inst.status == PlaybookStatus.ACTIVE
                ]
            else:
                return [
                    inst
                    for inst in self._instances.values()
                    if inst.status == PlaybookStatus.ACTIVE
                ]

    async def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        async with self._lock:
            active = sum(
                1 for inst in self._instances.values() if inst.status == PlaybookStatus.ACTIVE
            )
            completed = sum(
                1 for inst in self._instances.values() if inst.status == PlaybookStatus.COMPLETED
            )
            escalated = sum(
                1 for inst in self._instances.values() if inst.status == PlaybookStatus.ESCALATED
            )
            failed = sum(
                1 for inst in self._instances.values() if inst.status == PlaybookStatus.FAILED
            )

            return {
                "total_instances": len(self._instances),
                "active": active,
                "completed": completed,
                "escalated": escalated,
                "failed": failed,
                "by_playbook": {
                    playbook_id: len(instance_ids)
                    for playbook_id, instance_ids in self._by_playbook.items()
                },
            }

    async def clear_completed(self) -> int:
        """
        Remove completed/escalated/failed instances.

        Returns:
            Number of instances removed
        """
        async with self._lock:
            to_remove = [
                inst_id
                for inst_id, inst in self._instances.items()
                if inst.status != PlaybookStatus.ACTIVE
            ]

            for inst_id in to_remove:
                inst = self._instances.pop(inst_id)
                if inst.playbook_id in self._by_playbook:
                    self._by_playbook[inst.playbook_id].discard(inst_id)

            return len(to_remove)
