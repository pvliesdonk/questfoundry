"""
Artifact lifecycle management.

Loads lifecycle state machines from artifact type definitions and validates
state transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> dict[str, Any] | None:
    """
    Convert an object to a dictionary.

    Handles Pydantic v2 (.model_dump), Pydantic v1 (.dict),
    and custom objects with .to_dict() method.

    Args:
        obj: Object to convert

    Returns:
        Dictionary representation or None if conversion not possible
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)

    # Prefer Pydantic v2-style export
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump(by_alias=True))
        except TypeError:  # pragma: no cover - defensive
            return cast(dict[str, Any], obj.model_dump())

    # Fallback to Pydantic v1-style export
    if hasattr(obj, "dict"):
        try:
            return cast(dict[str, Any], obj.dict(by_alias=True))
        except TypeError:  # pragma: no cover - defensive
            return cast(dict[str, Any], obj.dict())

    # Generic custom object export
    if hasattr(obj, "to_dict"):
        return cast(dict[str, Any], obj.to_dict())

    return None


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
    target_store: str | None = None  # Store migration on transition

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleTransition:
        """Create from dictionary."""
        return cls(
            from_state=data["from"],
            to_state=data["to"],
            allowed_agents=data.get("allowed_agents", []),
            requires_validation=data.get("requires_validation", []),
            target_store=data.get("target_store"),
        )

    def is_agent_allowed(self, agent_id: str | None) -> bool:
        """Check if an agent is allowed to make this transition."""
        # Empty list means any agent can make this transition
        if not self.allowed_agents:
            return True
        return agent_id in self.allowed_agents


@dataclass
class LifecyclePolicy:
    """
    Runtime enforcement behavior for an artifact type.

    Defines what happens automatically when artifacts are edited,
    separate from the state machine definition.
    """

    edit_policy: str = "allow"  # "allow", "demote", "disallow"
    demote_trigger_states: list[str] | None = None  # None = all except initial
    demote_target_state: str | None = None  # Defaults to initial_state
    demote_target_store: str | None = None  # Defaults to artifact's default_store

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LifecyclePolicy:
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            edit_policy=data.get("edit_policy", "allow"),
            demote_trigger_states=data.get("demote_trigger_states"),
            demote_target_state=data.get("demote_target_state"),
            demote_target_store=data.get("demote_target_store"),
        )


@dataclass
class ArtifactLifecycle:
    """
    Lifecycle state machine for an artifact type.

    Tracks states and valid transitions between them, plus
    runtime enforcement policy for edit behavior.
    """

    artifact_type_id: str
    states: dict[str, LifecycleState] = field(default_factory=dict)
    transitions: list[LifecycleTransition] = field(default_factory=list)
    initial_state: str = "draft"
    policy: LifecyclePolicy | None = None  # Runtime enforcement behavior
    default_store: str | None = None  # Fallback for demotion store

    @classmethod
    def from_dict(
        cls,
        artifact_type_id: str,
        data: dict[str, Any] | None,
        policy_data: dict[str, Any] | None = None,
        default_store: str | None = None,
    ) -> ArtifactLifecycle | None:
        """
        Create lifecycle from artifact type definition.

        Args:
            artifact_type_id: ID of the artifact type
            data: Lifecycle section from artifact type definition
            policy_data: lifecycle_policy section from artifact type definition
            default_store: default_store from artifact type definition

        Returns:
            ArtifactLifecycle or None if no lifecycle defined
        """
        if data is None:
            return None

        # Parse states (may already be LifecycleState objects)
        states = {}
        for state_data in data.get("states", []):
            if isinstance(state_data, dict):
                state = LifecycleState.from_dict(state_data)
            else:
                # Already a LifecycleState object
                state = state_data
            states[state.id] = state

        # Parse transitions (may already be LifecycleTransition objects)
        transitions = []
        for trans_data in data.get("transitions", []):
            if isinstance(trans_data, dict):
                transitions.append(LifecycleTransition.from_dict(trans_data))
            else:
                # Already a LifecycleTransition object
                transitions.append(trans_data)

        # Parse lifecycle policy
        policy = LifecyclePolicy.from_dict(policy_data) if policy_data else None

        return cls(
            artifact_type_id=artifact_type_id,
            states=states,
            transitions=transitions,
            initial_state=data.get("initial_state", "draft"),
            policy=policy,
            default_store=default_store,
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

    def get_edit_policy(self, current_state: str) -> str:
        """
        Get edit policy for the given state.

        Args:
            current_state: Current lifecycle state of the artifact

        Returns:
            "allow", "demote", or "disallow"
        """
        if self.policy is None:
            return "allow"

        # Determine which states trigger the policy
        trigger_states = self.policy.demote_trigger_states
        if trigger_states is None:
            # Default: all states except initial trigger demotion
            trigger_states = [s for s in self.states if s != self.initial_state]

        if current_state not in trigger_states:
            return "allow"

        return self.policy.edit_policy

    def get_demote_target(self) -> tuple[str, str | None]:
        """
        Get target state and store for demotion.

        Returns:
            Tuple of (target_state, target_store). target_store may be None.
        """
        if self.policy is None:
            return (self.initial_state, self.default_store)

        target_state = self.policy.demote_target_state or self.initial_state
        target_store = self.policy.demote_target_store or self.default_store

        return (target_state, target_store)

    def get_transition_store(self, from_state: str, to_state: str) -> str | None:
        """
        Get target_store for a transition, if specified.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            Store ID if transition specifies target_store, None otherwise
        """
        transition = self.get_transition(from_state, to_state)
        return transition.target_store if transition else None


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
                policy_data = artifact_type.get("lifecycle_policy")
                default_store = artifact_type.get("default_store")
            else:
                type_id = getattr(artifact_type, "id", None)
                lifecycle_data = getattr(artifact_type, "lifecycle", None)
                policy_data = getattr(artifact_type, "lifecycle_policy", None)
                default_store = getattr(artifact_type, "default_store", None)

                # Convert lifecycle_data to dict if needed
                if lifecycle_data and not isinstance(lifecycle_data, dict):
                    lifecycle_dict = _to_dict(lifecycle_data)
                    if lifecycle_dict is not None:
                        lifecycle_data = lifecycle_dict
                    else:
                        # Last-resort attribute access – keeps behavior for
                        # simple container objects used in tests and docs.
                        lifecycle_data = {
                            "states": getattr(lifecycle_data, "states", []),
                            "initial_state": getattr(lifecycle_data, "initial_state", "draft"),
                            "transitions": getattr(lifecycle_data, "transitions", []),
                        }

                # Convert policy_data to dict if needed
                if policy_data and not isinstance(policy_data, dict):
                    policy_dict = _to_dict(policy_data)
                    if policy_dict is not None:
                        policy_data = policy_dict
                    else:
                        # Last-resort attribute access for policy objects
                        policy_data = {
                            "edit_policy": getattr(policy_data, "edit_policy", "allow"),
                            "demote_trigger_states": getattr(
                                policy_data, "demote_trigger_states", None
                            ),
                            "demote_target_state": getattr(
                                policy_data, "demote_target_state", None
                            ),
                            "demote_target_store": getattr(
                                policy_data, "demote_target_store", None
                            ),
                        }

            if type_id and lifecycle_data:
                lifecycle = ArtifactLifecycle.from_dict(
                    type_id, lifecycle_data, policy_data, default_store
                )
                if lifecycle:
                    manager.register_lifecycle(lifecycle)

        return manager
