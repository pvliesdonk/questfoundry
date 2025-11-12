"""TU (Thematic Unit) lifecycle state machine"""

from typing import Any

from .base import Lifecycle


class TULifecycle(Lifecycle):
    """
    Lifecycle state machine for Thematic Unit (TU) artifacts.

    States:
    - hot-proposed: TU is proposed in hot workspace
    - stabilizing: TU is being refined and stabilized
    - gatecheck: TU is undergoing gatekeeper quality validation
    - cold-merged: TU has been merged into cold storage

    Transitions:
    - hot-proposed → stabilizing: TU begins stabilization
    - stabilizing → gatecheck: TU is ready for gatecheck
    - stabilizing → hot-proposed: TU needs more work
    - gatecheck → cold-merged: TU passes gatecheck and is merged
    - gatecheck → stabilizing: TU fails gatecheck, needs rework
    """

    STATES = {
        "hot-proposed",
        "stabilizing",
        "gatecheck",
        "cold-merged",
    }

    INITIAL_STATE = "hot-proposed"

    # Define valid transitions
    # Format: (from_state, to_state): validation_function
    TRANSITIONS = {
        # From hot-proposed
        ("hot-proposed", "stabilizing"): None,
        # From stabilizing
        ("stabilizing", "gatecheck"): None,
        ("stabilizing", "hot-proposed"): None,  # Back to proposed if needed
        # From gatecheck
        ("gatecheck", "cold-merged"): lambda data: data.get("gatecheck_passed", False),
        ("gatecheck", "stabilizing"): None,  # Failed gatecheck
    }

    def begin_stabilization(self, reason: str | None = None) -> None:
        """
        Begin TU stabilization process.

        Args:
            reason: Optional reason for starting stabilization

        Raises:
            StateTransitionError: If not in hot-proposed state
        """
        self.transition_to("stabilizing", reason=reason)

    def request_gatecheck(self, reason: str | None = None) -> None:
        """
        Request gatekeeper validation.

        Args:
            reason: Optional notes for gatekeeper

        Raises:
            StateTransitionError: If not in stabilizing state
        """
        self.transition_to("gatecheck", reason=reason)

    def merge_to_cold(
        self, data: dict[str, Any] | None = None, reason: str | None = None
    ) -> None:
        """
        Merge TU to cold storage after passing gatecheck.

        Args:
            data: Data containing gatecheck_passed flag
            reason: Optional merge notes

        Raises:
            StateTransitionError: If not in gatecheck state or gatecheck not passed
        """
        self.transition_to("cold-merged", data=data, reason=reason)

    def return_to_stabilization(self, reason: str | None = None) -> None:
        """
        Return to stabilization from gatecheck.

        Args:
            reason: Reason for returning (e.g., failed quality bars)

        Raises:
            StateTransitionError: If not in gatecheck state
        """
        self.transition_to("stabilizing", reason=reason)

    def return_to_proposed(self, reason: str | None = None) -> None:
        """
        Return to proposed state from stabilizing.

        Args:
            reason: Reason for returning to proposed

        Raises:
            StateTransitionError: If not in stabilizing state
        """
        self.transition_to("hot-proposed", reason=reason)
