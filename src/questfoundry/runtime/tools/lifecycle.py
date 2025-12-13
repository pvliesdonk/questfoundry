"""Lifecycle transition tools for v4 runtime.

Implements the lifecycle_transition_request/response protocol from meta/.
Agents use request_lifecycle_transition to formally request state changes
for artifacts. The runtime validates the request and responds with
committed, rejected, or deferred.

Lifecycle states typically follow: draft -> review -> approved -> canon
Some transitions require validation (e.g., review -> approved needs GK approval).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


# Default lifecycle states and transitions
DEFAULT_LIFECYCLE = {
    "states": ["draft", "review", "approved", "canon", "archived"],
    "transitions": {
        "draft": ["review"],
        "review": ["draft", "approved"],
        "approved": ["canon", "review"],
        "canon": ["archived"],
        "archived": [],
    },
    "requires_validation": {
        "review": ["approved"],  # review -> approved requires validation
    },
}


class RequestLifecycleTransitionTool(BaseTool):
    """Request a lifecycle state transition for an artifact.

    Use this tool to formally request moving an artifact from one
    lifecycle state to another. The runtime will validate the request
    and respond with:
    - committed: transition applied
    - rejected: criteria not met
    - deferred: needs orchestrator/customer decision

    Transitions that require validation (e.g., review -> approved) will
    check quality criteria before committing.
    """

    name: str = "request_lifecycle_transition"
    description: str = (
        "Request a lifecycle state transition for an artifact. "
        "Input: artifact_id, from_state (current), to_state (target), "
        "justification (why transition criteria are met), "
        "evidence (optional list of criterion claims)"
    )

    # Injected by registry
    studio: Any = Field(default=None, exclude=True)
    state: Any = Field(default=None, exclude=True)

    def _run(
        self,
        artifact_id: str,
        from_state: str,
        to_state: str,
        justification: str = "",
        evidence: list[dict[str, str]] | None = None,
    ) -> str:
        """Request a lifecycle transition.

        Parameters
        ----------
        artifact_id : str
            ID of the artifact to transition
        from_state : str
            Current state (for consistency check)
        to_state : str
            Requested target state
        justification : str
            Explanation of why transition criteria are met
        evidence : list[dict] | None
            Evidence for validation criteria (optional)
            Each dict has: criterion_id, claim

        Returns
        -------
        str
            JSON lifecycle_transition_response with result
        """
        # Validate inputs
        if not artifact_id:
            return self._response(
                result="rejected",
                rejection_reason="artifact_id is required",
                guidance="Provide the ID of the artifact to transition.",
            )

        if not from_state:
            return self._response(
                result="rejected",
                rejection_reason="from_state is required",
                guidance="Specify the artifact's current lifecycle state.",
            )

        if not to_state:
            return self._response(
                result="rejected",
                rejection_reason="to_state is required",
                guidance="Specify the target lifecycle state.",
            )

        # Get the artifact
        artifact = self._get_artifact(artifact_id)
        if not artifact:
            return self._response(
                result="rejected",
                rejection_reason=f"Artifact '{artifact_id}' not found",
                guidance="Check the artifact ID. Use read_artifact to verify it exists.",
            )

        # Verify current state matches
        current_state = self._get_artifact_state(artifact)
        if current_state != from_state:
            return self._response(
                result="rejected",
                rejection_reason=f"State mismatch: artifact is in '{current_state}', not '{from_state}'",
                new_state=current_state,
                guidance=f"The artifact is already in '{current_state}' state.",
            )

        # Check if transition is valid
        lifecycle = self._get_lifecycle_config(artifact)
        if not self._is_valid_transition(from_state, to_state, lifecycle):
            valid_targets = lifecycle["transitions"].get(from_state, [])
            return self._response(
                result="rejected",
                rejection_reason=f"Invalid transition: {from_state} -> {to_state}",
                new_state=current_state,
                guidance=f"Valid transitions from '{from_state}': {valid_targets or 'none'}",
            )

        # Check if transition requires validation
        requires_validation = self._requires_validation(from_state, to_state, lifecycle)
        if requires_validation:
            validation_result = self._validate_transition(
                artifact, from_state, to_state, justification, evidence
            )
            if not validation_result["passed"]:
                return self._response(
                    result="rejected",
                    rejection_reason="Validation criteria not met",
                    new_state=current_state,
                    validation_results=validation_result.get("results", []),
                    guidance=validation_result.get("guidance", "Address the failed criteria and try again."),
                )

        # Apply the transition
        self._apply_transition(artifact, artifact_id, to_state)

        return self._response(
            result="committed",
            new_state=to_state,
            guidance=f"Artifact '{artifact_id}' transitioned to '{to_state}'.",
            validation_results=validation_result.get("results") if requires_validation else None,
        )

    def _response(
        self,
        result: str,
        new_state: str | None = None,
        rejection_reason: str | None = None,
        guidance: str | None = None,
        validation_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format a lifecycle_transition_response."""
        response: dict[str, Any] = {"result": result}

        if new_state:
            response["new_state"] = new_state
        if rejection_reason:
            response["rejection_reason"] = rejection_reason
        if guidance:
            response["guidance"] = guidance
        if validation_results:
            response["validation_results"] = validation_results

        return json.dumps(response)

    def _get_artifact(self, artifact_id: str) -> Any | None:
        """Get artifact from hot_store or cold_store."""
        if self.state is None:
            return None

        # Check hot_store
        hot_store = getattr(self.state, "hot_store", None)
        if hot_store is None and isinstance(self.state, dict):
            hot_store = self.state.get("hot_store", {})

        if hot_store and artifact_id in hot_store:
            return hot_store[artifact_id]

        # Check cold_store (dict fallback)
        cold_store = getattr(self.state, "cold_store", None)
        if cold_store is None and isinstance(self.state, dict):
            cold_store = self.state.get("cold_store", {})

        if cold_store and artifact_id in cold_store:
            return cold_store[artifact_id]

        return None

    def _get_artifact_state(self, artifact: Any) -> str:
        """Get the lifecycle state of an artifact."""
        if hasattr(artifact, "status"):
            return artifact.status

        if isinstance(artifact, dict):
            # Try _lifecycle_state first (meta/ spec)
            if "_lifecycle_state" in artifact:
                return artifact["_lifecycle_state"]
            return artifact.get("status", "draft")

        return "draft"

    def _get_lifecycle_config(self, artifact: Any) -> dict[str, Any]:
        """Get lifecycle configuration for an artifact type."""
        # Try to get from studio artifact type definition
        if self.studio:
            artifact_type = None
            if hasattr(artifact, "type"):
                artifact_type = artifact.type
            elif isinstance(artifact, dict):
                artifact_type = artifact.get("type")

            if artifact_type and hasattr(self.studio, "artifact_types"):
                type_def = self.studio.artifact_types.get(artifact_type)
                if type_def and hasattr(type_def, "lifecycle"):
                    return self._convert_lifecycle_def(type_def.lifecycle)

        # Return default lifecycle
        return DEFAULT_LIFECYCLE

    def _convert_lifecycle_def(self, lifecycle_def: Any) -> dict[str, Any]:
        """Convert studio lifecycle definition to dict format."""
        if isinstance(lifecycle_def, dict):
            return lifecycle_def

        # Extract from model
        result: dict[str, Any] = {
            "states": [],
            "transitions": {},
            "requires_validation": {},
        }

        if hasattr(lifecycle_def, "states"):
            for state in lifecycle_def.states:
                state_id = state.id if hasattr(state, "id") else str(state)
                result["states"].append(state_id)

        if hasattr(lifecycle_def, "transitions"):
            for trans in lifecycle_def.transitions:
                from_state = trans.from_state if hasattr(trans, "from_state") else None
                to_state = trans.to_state if hasattr(trans, "to_state") else None
                if from_state and to_state:
                    result["transitions"].setdefault(from_state, []).append(to_state)
                    if hasattr(trans, "requires_validation") and trans.requires_validation:
                        result["requires_validation"].setdefault(from_state, []).append(to_state)

        return result if result["states"] else DEFAULT_LIFECYCLE

    def _is_valid_transition(
        self, from_state: str, to_state: str, lifecycle: dict[str, Any]
    ) -> bool:
        """Check if a transition is valid."""
        valid_targets = lifecycle.get("transitions", {}).get(from_state, [])
        return to_state in valid_targets

    def _requires_validation(
        self, from_state: str, to_state: str, lifecycle: dict[str, Any]
    ) -> bool:
        """Check if a transition requires validation."""
        requires = lifecycle.get("requires_validation", {}).get(from_state, [])
        return to_state in requires

    def _validate_transition(
        self,
        _artifact: Any,
        _from_state: str,
        _to_state: str,
        justification: str,
        evidence: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        """Validate a transition against quality criteria.

        For now, requires justification for validated transitions.
        Future: integrate with quality gate evaluation.
        """
        results: list[dict[str, Any]] = []

        # Require justification for validated transitions
        if not justification or not justification.strip():
            results.append({
                "criterion_id": "justification_required",
                "passed": False,
                "message": "Justification is required for this transition",
            })
            return {
                "passed": False,
                "results": results,
                "guidance": "Provide a justification explaining why the artifact meets transition criteria.",
            }

        results.append({
            "criterion_id": "justification_required",
            "passed": True,
            "message": "Justification provided",
        })

        # If evidence provided, record it
        if evidence:
            for ev in evidence:
                results.append({
                    "criterion_id": ev.get("criterion_id", "unknown"),
                    "passed": True,  # Accept claims for now
                    "message": ev.get("claim", "Evidence provided"),
                })

        return {
            "passed": True,
            "results": results,
        }

    def _apply_transition(self, artifact: Any, artifact_id: str, new_state: str) -> None:
        """Apply the state transition to the artifact."""
        # Update artifact status
        if hasattr(artifact, "status"):
            artifact.status = new_state
        elif isinstance(artifact, dict):
            artifact["status"] = new_state
            artifact["_lifecycle_state"] = new_state

        logger.info(f"Artifact '{artifact_id}' transitioned to '{new_state}'")


def create_lifecycle_transition_tool(
    studio: Any = None,
    state: Any = None,
) -> RequestLifecycleTransitionTool:
    """Factory function to create a lifecycle transition tool.

    Args:
        studio: The loaded studio (for lifecycle configs)
        state: The current studio state (for artifact access)

    Returns:
        Configured RequestLifecycleTransitionTool
    """
    tool = RequestLifecycleTransitionTool()
    tool.studio = studio
    tool.state = state
    return tool
