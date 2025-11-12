"""Hook lifecycle state machine"""

from .base import Lifecycle


class HookLifecycle(Lifecycle):
    """
    Lifecycle state machine for Hook Card artifacts.

    States:
    - proposed: Hook is proposed and awaiting review
    - accepted: Hook has been accepted for work
    - in-progress: Work is actively being done on the hook
    - resolved: Hook has been resolved (work complete)
    - canonized: Hook resolution has been merged into cold storage
    - deferred: Hook has been deferred to a later time
    - rejected: Hook has been rejected

    Transitions:
    - proposed → accepted: Hook is accepted for work
    - proposed → deferred: Hook is deferred
    - proposed → rejected: Hook is rejected
    - accepted → in-progress: Work begins on hook
    - accepted → deferred: Accepted hook is deferred
    - in-progress → resolved: Work is complete
    - in-progress → deferred: In-progress work is deferred
    - resolved → canonized: Resolution is merged to cold
    - deferred → proposed: Deferred hook is reactivated
    """

    STATES = {
        "proposed",
        "accepted",
        "in-progress",
        "resolved",
        "canonized",
        "deferred",
        "rejected",
    }

    INITIAL_STATE = "proposed"

    # Define valid transitions
    # Format: (from_state, to_state): validation_function
    TRANSITIONS = {
        # From proposed
        ("proposed", "accepted"): None,
        ("proposed", "deferred"): None,
        ("proposed", "rejected"): None,
        # From accepted
        ("accepted", "in-progress"): None,
        ("accepted", "deferred"): None,
        # From in-progress
        ("in-progress", "resolved"): None,
        ("in-progress", "deferred"): None,
        # From resolved
        ("resolved", "canonized"): None,
        # From deferred - can return to proposed
        ("deferred", "proposed"): None,
    }

    def accept(self, reason: str | None = None) -> None:
        """
        Accept a proposed hook.

        Args:
            reason: Optional reason for acceptance

        Raises:
            StateTransitionError: If not in proposed state
        """
        self.transition_to("accepted", reason=reason)

    def start_work(self, reason: str | None = None) -> None:
        """
        Begin work on accepted hook.

        Args:
            reason: Optional reason for starting work

        Raises:
            StateTransitionError: If not in accepted state
        """
        self.transition_to("in-progress", reason=reason)

    def resolve(self, reason: str | None = None) -> None:
        """
        Mark hook as resolved (work complete).

        Args:
            reason: Optional resolution summary

        Raises:
            StateTransitionError: If not in in-progress state
        """
        self.transition_to("resolved", reason=reason)

    def canonize(self, reason: str | None = None) -> None:
        """
        Merge resolved hook into cold storage.

        Args:
            reason: Optional canonization notes

        Raises:
            StateTransitionError: If not in resolved state
        """
        self.transition_to("canonized", reason=reason)

    def defer(self, reason: str | None = None) -> None:
        """
        Defer hook for later work.

        Args:
            reason: Reason for deferral

        Raises:
            StateTransitionError: If not in a deferrable state
        """
        self.transition_to("deferred", reason=reason)

    def reject(self, reason: str | None = None) -> None:
        """
        Reject proposed hook.

        Args:
            reason: Reason for rejection

        Raises:
            StateTransitionError: If not in proposed state
        """
        self.transition_to("rejected", reason=reason)

    def reactivate(self, reason: str | None = None) -> None:
        """
        Reactivate deferred hook.

        Args:
            reason: Optional reason for reactivation

        Raises:
            StateTransitionError: If not in deferred state
        """
        self.transition_to("proposed", reason=reason)
