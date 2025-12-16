"""
Artifact lifecycle management.

Loads lifecycle state machines from artifact type definitions and validates
state transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LifecycleState:
    """A state in the lifecycle state machine."""

    id: str
    name: str
    description: str | None = None
    terminal: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleState:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description"),
            terminal=data.get("terminal", False),
        )


@dataclass
class LifecycleTransition:
    """A transition between lifecycle states."""

    from_state: str
    to_state: str
    allowed_agents: list[str] = field(default_factory=list)
    requires_validation: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleTransition:
        """Create from dictionary."""
        return cls(
            from_state=data["from"],
            to_state=data["to"],
            allowed_agents=data.get("allowed_agents", []),
            requires_validation=data.get("requires_validation", []),
        )

    def is_agent_allowed(self, agent_id: str | None) -> bool:
        """Check if an agent is allowed to make this transition."""
        # Empty list means any agent can make this transition
        if not self.allowed_agents:
            return True
        return agent_id in self.allowed_agents


@dataclass
class ArtifactLifecycle:
    """
    Lifecycle state machine for an artifact type.

    Tracks states and valid transitions between them.
    """

    artifact_type_id: str
    states: dict[str, LifecycleState] = field(default_factory=dict)
    transitions: list[LifecycleTransition] = field(default_factory=list)
    initial_state: str = "draft"

    @classmethod
    def from_dict(
        cls, artifact_type_id: str, data: dict[str, Any] | None
    ) -> ArtifactLifecycle | None:
        """
        Create lifecycle from artifact type definition.

        Args:
            artifact_type_id: ID of the artifact type
            data: Lifecycle section from artifact type definition

        Returns:
            ArtifactLifecycle or None if no lifecycle defined
        """
        if data is None:
            return None

        # Parse states
        states = {}
        for state_data in data.get("states", []):
            state = LifecycleState.from_dict(state_data)
            states[state.id] = state

        # Parse transitions
        transitions = []
        for trans_data in data.get("transitions", []):
            transitions.append(LifecycleTransition.from_dict(trans_data))

        return cls(
            artifact_type_id=artifact_type_id,
            states=states,
            transitions=transitions,
            initial_state=data.get("initial_state", "draft"),
        )

    def get_state(self, state_id: str) -> LifecycleState | None:
        """Get a state by ID."""
        return self.states.get(state_id)

    def is_terminal_state(self, state_id: str) -> bool:
        """Check if a state is terminal (no further transitions allowed)."""
        state = self.states.get(state_id)
        return state.terminal if state else False

    def get_valid_transitions(
        self, from_state: str, agent_id: str | None = None
    ) -> list[LifecycleTransition]:
        """
        Get valid transitions from a state.

        Args:
            from_state: Current state
            agent_id: Optional agent ID to filter by allowed_agents

        Returns:
            List of valid transitions
        """
        valid = []
        for trans in self.transitions:
            if trans.from_state == from_state and (
                agent_id is None or trans.is_agent_allowed(agent_id)
            ):
                valid.append(trans)
        return valid

    def get_transition(self, from_state: str, to_state: str) -> LifecycleTransition | None:
        """Get a specific transition if it exists."""
        for trans in self.transitions:
            if trans.from_state == from_state and trans.to_state == to_state:
                return trans
        return None

    def validate_transition(
        self, from_state: str, to_state: str, agent_id: str | None = None
    ) -> tuple[bool, str | None]:
        """
        Validate if a transition is allowed.

        Args:
            from_state: Current state
            to_state: Target state
            agent_id: Agent requesting the transition

        Returns:
            Tuple of (allowed, reason if not allowed)
        """
        # Check if from_state is terminal
        if self.is_terminal_state(from_state):
            return False, f"Cannot transition from terminal state '{from_state}'"

        # Find the transition
        transition = self.get_transition(from_state, to_state)
        if transition is None:
            return False, f"No transition defined from '{from_state}' to '{to_state}'"

        # Check agent permission
        if not transition.is_agent_allowed(agent_id):
            allowed_str = ", ".join(transition.allowed_agents)
            return (
                False,
                f"Agent '{agent_id}' not allowed for this transition. "
                f"Allowed agents: {allowed_str}",
            )

        return True, None

    def list_states(self) -> list[LifecycleState]:
        """List all states in order of definition."""
        return list(self.states.values())


class LifecycleManager:
    """
    Manages lifecycles for multiple artifact types.

    Loads lifecycle definitions from artifact types and provides
    lookup and validation services.
    """

    def __init__(self) -> None:
        """Initialize empty lifecycle manager."""
        self._lifecycles: dict[str, ArtifactLifecycle] = {}

    def register_lifecycle(self, lifecycle: ArtifactLifecycle) -> None:
        """Register a lifecycle for an artifact type."""
        self._lifecycles[lifecycle.artifact_type_id] = lifecycle
        logger.debug(f"Registered lifecycle for artifact type: {lifecycle.artifact_type_id}")

    def get_lifecycle(self, artifact_type_id: str) -> ArtifactLifecycle | None:
        """Get lifecycle for an artifact type."""
        return self._lifecycles.get(artifact_type_id)

    def has_lifecycle(self, artifact_type_id: str) -> bool:
        """Check if an artifact type has a lifecycle."""
        return artifact_type_id in self._lifecycles

    def get_initial_state(self, artifact_type_id: str) -> str:
        """Get initial state for an artifact type (defaults to 'draft')."""
        lifecycle = self._lifecycles.get(artifact_type_id)
        if lifecycle:
            return lifecycle.initial_state
        return "draft"

    def validate_transition(
        self,
        artifact_type_id: str,
        from_state: str,
        to_state: str,
        agent_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate a lifecycle transition.

        Args:
            artifact_type_id: Artifact type
            from_state: Current state
            to_state: Target state
            agent_id: Agent requesting the transition

        Returns:
            Tuple of (allowed, reason if not allowed)
        """
        lifecycle = self._lifecycles.get(artifact_type_id)
        if lifecycle is None:
            # No lifecycle defined - allow any state change
            return True, None

        return lifecycle.validate_transition(from_state, to_state, agent_id)

    def get_required_validations(
        self, artifact_type_id: str, from_state: str, to_state: str
    ) -> list[str]:
        """
        Get required validations for a transition.

        Args:
            artifact_type_id: Artifact type
            from_state: Current state
            to_state: Target state

        Returns:
            List of validation IDs required for this transition
        """
        lifecycle = self._lifecycles.get(artifact_type_id)
        if lifecycle is None:
            return []

        transition = lifecycle.get_transition(from_state, to_state)
        if transition is None:
            return []

        return transition.requires_validation

    @classmethod
    def from_artifact_types(cls, artifact_types: list[Any]) -> LifecycleManager:
        """
        Create manager from artifact type definitions.

        Args:
            artifact_types: List of artifact type objects with lifecycle attribute

        Returns:
            LifecycleManager with all lifecycles registered
        """
        manager = cls()

        for artifact_type in artifact_types:
            # Handle both dict and object types
            if isinstance(artifact_type, dict):
                type_id = artifact_type.get("id")
                lifecycle_data = artifact_type.get("lifecycle")
            else:
                type_id = getattr(artifact_type, "id", None)
                lifecycle_data = getattr(artifact_type, "lifecycle", None)
                # If lifecycle is an object, convert to dict
                if lifecycle_data and hasattr(lifecycle_data, "to_dict"):
                    lifecycle_data = lifecycle_data.to_dict()
                elif lifecycle_data and not isinstance(lifecycle_data, dict):
                    # Try to access as object attributes
                    lifecycle_data = {
                        "states": getattr(lifecycle_data, "states", []),
                        "initial_state": getattr(lifecycle_data, "initial_state", "draft"),
                        "transitions": getattr(lifecycle_data, "transitions", []),
                    }

            if type_id and lifecycle_data:
                lifecycle = ArtifactLifecycle.from_dict(type_id, lifecycle_data)
                if lifecycle:
                    manager.register_lifecycle(lifecycle)

        return manager
