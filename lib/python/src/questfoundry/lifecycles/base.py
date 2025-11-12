"""Base lifecycle state machine"""

from datetime import datetime
from typing import Any


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""

    pass


class Lifecycle:
    """
    Base class for artifact lifecycle state machines.

    Manages state transitions, validation, and history tracking.

    Subclasses should define:
    - STATES: Set of valid states
    - INITIAL_STATE: Starting state
    - TRANSITIONS: Dict mapping (from_state, to_state) -> validation_fn

    Example:
        >>> class MyLifecycle(Lifecycle):
        ...     STATES = {"draft", "review", "published"}
        ...     INITIAL_STATE = "draft"
        ...     TRANSITIONS = {
        ...         ("draft", "review"): lambda data: True,
        ...         ("review", "published"): lambda data: True,
        ...     }
    """

    # Subclasses must define these
    STATES: set[str] = set()
    INITIAL_STATE: str = ""
    TRANSITIONS: dict[tuple[str, str], Any] = {}

    def __init__(self, current_state: str | None = None):
        """
        Initialize lifecycle.

        Args:
            current_state: Current state (defaults to INITIAL_STATE)

        Raises:
            ValueError: If current_state is not valid
        """
        self.current_state = current_state or self.INITIAL_STATE
        self.history: list[dict[str, Any]] = []

        if self.current_state not in self.STATES:
            raise ValueError(
                f"Invalid state '{self.current_state}'. Must be one of: {self.STATES}"
            )

    def can_transition_to(
        self, new_state: str, data: dict[str, Any] | None = None
    ) -> bool:
        """
        Check if transition to new state is valid.

        Args:
            new_state: Target state
            data: Optional artifact data for validation

        Returns:
            True if transition is valid, False otherwise
        """
        if new_state not in self.STATES:
            return False

        # Same state transition is always allowed
        if new_state == self.current_state:
            return True

        transition = (self.current_state, new_state)
        if transition not in self.TRANSITIONS:
            return False

        # Run validation function if present
        validator = self.TRANSITIONS[transition]
        if validator and callable(validator):
            try:
                result: bool = bool(validator(data or {}))
                return result
            except (ValueError, KeyError, AttributeError):
                # Validator raised expected exception - invalid transition
                return False

        return True

    def transition_to(
        self,
        new_state: str,
        data: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        """
        Transition to new state.

        Args:
            new_state: Target state
            data: Optional artifact data for validation
            reason: Optional reason for transition

        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(new_state, data):
            raise StateTransitionError(
                f"Cannot transition from '{self.current_state}' to '{new_state}'"
            )

        # Record history
        self.history.append(
            {
                "from_state": self.current_state,
                "to_state": new_state,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
            }
        )

        self.current_state = new_state

    def get_valid_transitions(self) -> list[str]:
        """
        Get list of valid next states from current state.

        Returns:
            List of state names that can be transitioned to
        """
        valid = {
            to_state
            for from_state, to_state in self.TRANSITIONS.keys()
            if from_state == self.current_state
        }
        return sorted(valid)

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get state transition history.

        Returns:
            List of transition records
        """
        return self.history.copy()
