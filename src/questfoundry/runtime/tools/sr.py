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

import contextlib
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

    IMPORTANT: When delegating validation or follow-up work, include the
    artifact IDs from previous delegation results in the `artifacts` parameter.
    This ensures the role knows exactly which artifacts to work with.
    """

    name: str = "delegate_to"
    description: str = (
        "Delegate a task to a specialist role. "
        "The role executes autonomously and returns a result summary. "
        "Inputs: role (role_id), task (description of work), "
        "artifacts (optional list of artifact IDs from previous delegations to pass to the role)"
    )

    def _run(self, role: str, task: str, artifacts: list[str] | None = None) -> str:
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
        delegation_request: dict[str, Any] = {
            "role": role_id,
            "task": task.strip(),
        }

        # Include artifact IDs if provided
        if artifacts:
            delegation_request["artifacts"] = artifacts

        return json.dumps(
            {
                "success": True,
                "delegation_request": delegation_request,
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

    NOTE: This tool will warn (but not block) if there are unpromoted
    artifacts in hot_store that should be in cold_store.
    """

    name: str = "terminate"
    description: str = (
        "Terminate the workflow. Use when work is complete or cannot proceed. "
        "Inputs: reason (why terminating), summary (what was accomplished)"
    )

    # State is injected by orchestrator for validation
    state: dict[str, Any] = Field(default_factory=dict)
    # ColdStore is injected for promotion check
    cold_store: Any = Field(default=None)

    def _run(self, reason: str, summary: str = "") -> str:
        """Signal workflow termination with cold_store validation nudge."""
        if not reason or not reason.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": "Termination reason is required",
                    "hint": "Explain why the workflow is ending (completed, blocked, error, etc.)",
                }
            )

        # Check for unpromoted artifacts using domain ontology
        hot_store = self.state.get("hot_store", {})

        # Get promotable type names from domain ontology
        try:
            from questfoundry.generated.models.artifacts import (
                ARTIFACT_REGISTRY,
                PROMOTABLE_ARTIFACTS,
            )
            promotable_types = {
                type_name for type_name, cls in ARTIFACT_REGISTRY.items()
                if cls.__name__ in PROMOTABLE_ARTIFACTS
                and not type_name.startswith("cold_")  # cold_* are output types, not input
            }
        except ImportError:
            # Fallback if generated models not available
            promotable_types = {"act", "chapter", "scene", "canon_entry"}

        # Find artifacts with promotable types
        promotable_keys = []
        for key, artifact in hot_store.items():
            artifact_type = None
            if hasattr(artifact, "type"):
                artifact_type = artifact.type
            elif isinstance(artifact, dict):
                artifact_type = artifact.get("type")
            if artifact_type and artifact_type.lower() in promotable_types:
                promotable_keys.append(key)

        unpromoted: list[str] = []
        if self.cold_store is not None and promotable_keys:
            try:
                # Get all anchors from cold_store (all tables)
                cold_anchors: set[str] = set()

                # Sections (scenes/narrative prose)
                with contextlib.suppress(Exception):
                    cold_anchors.update(self.cold_store.list_sections())

                # Acts and Chapters (structural)
                with contextlib.suppress(Exception):
                    cold_anchors.update(a.anchor for a in self.cold_store.list_acts())
                    cold_anchors.update(c.anchor for c in self.cold_store.list_chapters())

                # Codex (player-safe encyclopedia: character, location, item, relationship)
                with contextlib.suppress(Exception):
                    cold_anchors.update(self.cold_store.list_codex())

                # Canon (internal world facts: canon_entry, event, fact, timeline)
                with contextlib.suppress(Exception):
                    cold_anchors.update(self.cold_store.list_canon())

                unpromoted = [k for k in promotable_keys if k not in cold_anchors]
            except Exception:
                pass  # If cold_store check fails, don't block termination

        # BLOCK termination if there are unpromoted artifacts
        if unpromoted:
            logger.warning(
                "Termination blocked: %d unpromoted artifacts: %s",
                len(unpromoted),
                unpromoted,
            )
            return json.dumps(
                {
                    "success": False,
                    "error": "Cannot terminate with unpromoted content",
                    "unpromoted_artifacts": unpromoted,
                    "hint": (
                        f"{len(unpromoted)} artifact(s) in hot_store have NOT been promoted to cold_store: "
                        f"{', '.join(unpromoted)}. "
                        "Delegate to Lorekeeper to promote these artifacts before terminating. "
                        "Content in hot_store will be LOST if you terminate now."
                    ),
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
                    "hint": "Provide an artifact ID. Use list_hot_store_keys or list_cold_store_keys to see available artifacts.",
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
                "hint": "Check the artifact ID spelling. Use list_hot_store_keys "
                "or list_cold_store_keys to see all available artifacts.",
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
                    "hint": "Provide a unique artifact ID like 'scene_1' or 'act_1'.",
                }
            )

        if not artifact_type:
            return json.dumps(
                {
                    "success": False,
                    "error": "artifact_type is required",
                    "hint": "Use consult_schema to see available artifact types "
                    "(scene, act, chapter, etc.).",
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
