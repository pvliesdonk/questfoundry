"""SR (Showrunner) tools - orchestration tools for the hub agent.

These tools are used exclusively by the Showrunner to:
- Delegate work to specialist roles
- Terminate the workflow

The SR orchestrates via hub-and-spoke pattern:
1. SR decides to delegate work
2. SR calls delegate_to(role, task)
3. Orchestrator executes the role
4. Role returns DelegationResult
5. SR evaluates result and decides next action
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from questfoundry.runtime.resources import get_resource_loader

logger = logging.getLogger(__name__)


class DelegateTo(BaseTool):
    """Delegate work to a specialist role.

    Use this to assign a task to another role. The role will execute
    autonomously and return a DelegationResult with:
    - status: completed, blocked, needs_review, error
    - artifacts: IDs of artifacts created/modified
    - message: Summary of work done
    - recommendation: Suggested next action

    After delegation, evaluate the result and decide:
    - Delegate to another role
    - Terminate if work is complete
    """

    name: str = "delegate_to"
    description: str = (
        "Delegate a task to a specialist role. "
        "The role executes autonomously and returns a result summary. "
        "Inputs: role (role_id), task (description of work to do)"
    )

    def _run(self, role: str, task: str) -> str:
        """Validate delegation request.

        Note: Actual delegation is handled by the orchestrator.
        This tool validates the request and returns a marker for the orchestrator.
        """
        loader = get_resource_loader()

        # Normalize role_id
        role_id = role.lower().replace(" ", "_").replace("-", "_")

        # Validate role exists
        role_def = loader.load_role(role_id)
        if role_def is None:
            # List available roles
            try:
                from questfoundry.generated.roles import ALL_ROLES

                available = list(ALL_ROLES.keys())
            except ImportError:
                available = loader.list_roles()

            return json.dumps(
                {
                    "success": False,
                    "error": f"Role '{role}' not found",
                    "available_roles": available,
                    "hint": "Use one of the available roles listed above.",
                }
            )

        # Cannot delegate to self
        if role_id == "showrunner":
            return json.dumps(
                {
                    "success": False,
                    "error": "Cannot delegate to yourself (showrunner)",
                    "hint": "Delegate to a specialist role like plotwright, scene_smith, etc.",
                }
            )

        # Validate task is not empty
        if not task or not task.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Task description is required",
                    "hint": "Provide a clear description of what work the role should do.",
                }
            )

        # Return delegation request for orchestrator to handle
        # The orchestrator will intercept this and execute the role
        return json.dumps(
            {
                "success": True,
                "delegation_request": {
                    "role": role_id,
                    "task": task.strip(),
                },
                "note": "Delegation will be executed by orchestrator.",
            }
        )


class Terminate(BaseTool):
    """Terminate the workflow.

    Use this when:
    - All requested work is complete
    - The workflow cannot proceed (unrecoverable error)
    - Human intervention is required

    Provide a reason and summary of what was accomplished.
    """

    name: str = "terminate"
    description: str = (
        "Terminate the workflow. Use when work is complete or cannot proceed. "
        "Inputs: reason (why terminating), summary (what was accomplished)"
    )

    def _run(self, reason: str, summary: str = "") -> str:
        """Signal workflow termination."""
        if not reason or not reason.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Termination reason is required",
                    "hint": "Explain why the workflow is ending (completed, blocked, error, etc.)",
                }
            )

        return json.dumps(
            {
                "success": True,
                "termination": {
                    "reason": reason.strip(),
                    "summary": summary.strip() if summary else "",
                },
            }
        )


class ReadArtifact(BaseTool):
    """Read an artifact from hot_store or cold_store.

    Use this to inspect artifacts created by roles or to check
    existing content before making decisions.
    """

    name: str = "read_artifact"
    description: str = (
        "Read an artifact by ID from hot_store or cold_store. "
        "Input: artifact_id (the artifact's unique ID)"
    )

    # State is injected by orchestrator
    state: dict[str, Any] = Field(default_factory=dict)

    def _run(self, artifact_id: str) -> str:
        """Read artifact from state."""
        if not artifact_id:
            return json.dumps(
                {
                    "success": False,
                    "error": "artifact_id is required",
                }
            )

        # Check hot_store first
        hot_store = self.state.get("hot_store", {})
        if artifact_id in hot_store:
            artifact = hot_store[artifact_id]
            return json.dumps(
                {
                    "success": True,
                    "store": "hot_store",
                    "artifact": artifact.model_dump()
                    if hasattr(artifact, "model_dump")
                    else artifact,
                }
            )

        # Check cold_store
        cold_store = self.state.get("cold_store", {})
        if artifact_id in cold_store:
            artifact = cold_store[artifact_id]
            return json.dumps(
                {
                    "success": True,
                    "store": "cold_store",
                    "artifact": artifact.model_dump()
                    if hasattr(artifact, "model_dump")
                    else artifact,
                }
            )

        # List available artifacts
        hot_ids = list(hot_store.keys())
        cold_ids = list(cold_store.keys())

        return json.dumps(
            {
                "success": False,
                "error": f"Artifact '{artifact_id}' not found",
                "hot_store_artifacts": hot_ids[:10],
                "cold_store_artifacts": cold_ids[:10],
            }
        )


class WriteArtifact(BaseTool):
    """Write an artifact to hot_store.

    Use this to create or update artifacts. Artifacts in hot_store
    are mutable drafts; use Gatekeeper to move to cold_store.
    """

    name: str = "write_artifact"
    description: str = (
        "Write an artifact to hot_store (mutable draft storage). "
        "Inputs: artifact_id, artifact_type, data (dict of fields)"
    )

    # State is injected by orchestrator
    state: dict[str, Any] = Field(default_factory=dict)

    def _run(
        self,
        artifact_id: str,
        artifact_type: str,
        data: dict[str, Any],
        status: str = "draft",
    ) -> str:
        """Write artifact to hot_store."""

        from questfoundry.runtime.state import Artifact

        if not artifact_id:
            return json.dumps(
                {
                    "success": False,
                    "error": "artifact_id is required",
                }
            )

        if not artifact_type:
            return json.dumps(
                {
                    "success": False,
                    "error": "artifact_type is required",
                    "hint": "Use consult_schema to see available artifact types.",
                }
            )

        # Create artifact
        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            status=status,
            created_by="showrunner",
            data=data or {},
        )

        # Write to hot_store
        hot_store = self.state.setdefault("hot_store", {})
        is_update = artifact_id in hot_store
        hot_store[artifact_id] = artifact

        return json.dumps(
            {
                "success": True,
                "action": "updated" if is_update else "created",
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "status": status,
            }
        )
