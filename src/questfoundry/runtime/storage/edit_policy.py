"""
Edit policy enforcement for artifact modifications.

Enforces lifecycle policies and relationship cascade rules when
artifacts are edited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.storage.lifecycle import LifecycleManager
    from questfoundry.runtime.storage.project import Project
    from questfoundry.runtime.storage.relationship import RelationshipManager

logger = logging.getLogger(__name__)


@dataclass
class EditPolicyResult:
    """Result of checking edit policy for an artifact."""

    allowed: bool
    reason: str | None = None
    demote_to_state: str | None = None
    demote_to_store: str | None = None
    cascade_demotions: list[dict[str, Any]] | None = None


class EditPolicyGuard:
    """
    Enforces edit policies based on lifecycle and relationship rules.

    Checks whether an artifact can be edited and determines any
    automatic consequences (demotion, cascade to children).
    """

    def __init__(
        self,
        lifecycle_manager: LifecycleManager,
        relationship_manager: RelationshipManager,
    ) -> None:
        """
        Initialize the edit policy guard.

        Args:
            lifecycle_manager: Manager for artifact lifecycle state machines
            relationship_manager: Manager for artifact relationships
        """
        self._lifecycle_manager = lifecycle_manager
        self._relationship_manager = relationship_manager

    def check_edit(
        self,
        artifact: dict[str, Any],
        project: Project,
    ) -> EditPolicyResult:
        """
        Check if an artifact can be edited and determine consequences.

        Evaluates:
        1. Lifecycle policy for the artifact type (allow/demote/disallow)
        2. Relationship cascade policies for any children

        Args:
            artifact: The artifact being edited
            project: The project containing the artifact

        Returns:
            EditPolicyResult with allowed status and any required actions
        """
        artifact_type = artifact.get("_type")
        if not artifact_type:
            # No type - allow edit
            return EditPolicyResult(allowed=True)

        current_state = artifact.get("_lifecycle_state", "draft")

        # Get lifecycle for this artifact type
        lifecycle = self._lifecycle_manager.get_lifecycle(artifact_type)
        if lifecycle is None:
            # No lifecycle defined - edits always allowed
            return EditPolicyResult(allowed=True)

        # Check edit policy based on current state
        edit_policy = lifecycle.get_edit_policy(current_state)

        if edit_policy == "disallow":
            return EditPolicyResult(
                allowed=False,
                reason=f"Edits not allowed for {artifact_type} in state '{current_state}'",
            )

        if edit_policy == "allow":
            # Still need to check for cascade effects
            cascades = self._get_cascade_demotions(artifact, project)
            return EditPolicyResult(
                allowed=True,
                cascade_demotions=cascades if cascades else None,
            )

        # edit_policy == "demote"
        demote_state, demote_store = lifecycle.get_demote_target()
        cascades = self._get_cascade_demotions(artifact, project)

        return EditPolicyResult(
            allowed=True,
            demote_to_state=demote_state,
            demote_to_store=demote_store,
            cascade_demotions=cascades if cascades else None,
        )

    def _get_cascade_demotions(
        self,
        parent_artifact: dict[str, Any],
        project: Project,
    ) -> list[dict[str, Any]]:
        """
        Find children that need demotion due to parent edit.

        Looks up relationships where this artifact type is the parent
        and impact_policy.on_parent_edit is "demote".

        Args:
            parent_artifact: The parent artifact being edited
            project: The project containing artifacts

        Returns:
            List of dicts with child_id, demote_to_state, demote_to_store
        """
        demotions: list[dict[str, Any]] = []
        parent_type = parent_artifact.get("_type")
        if not parent_type:
            return demotions

        # Get relationships where this type is the parent
        relationships = self._relationship_manager.get_relationships_from_type(parent_type)

        for rel in relationships:
            if rel.impact_policy.on_parent_edit != "demote":
                continue

            # Find children linked via this relationship
            children = project.get_children_by_relationship(parent_artifact, rel)

            for child in children:
                child_id = child.get("_id")
                child_type = child.get("_type")
                if not child_type:
                    continue
                child_state = child.get("_lifecycle_state", "draft")

                # Get child's lifecycle to determine demotion target
                child_lifecycle = self._lifecycle_manager.get_lifecycle(child_type)
                if child_lifecycle is None:
                    continue

                # Skip if already in initial state
                if child_state == child_lifecycle.initial_state:
                    continue

                # Determine demotion target
                demote_state = child_lifecycle.initial_state
                demote_store = (
                    rel.impact_policy.demote_target_store or child_lifecycle.default_store
                )

                demotions.append(
                    {
                        "child_id": child_id,
                        "child_type": child_type,
                        "relationship_id": rel.id,
                        "demote_to_state": demote_state,
                        "demote_to_store": demote_store,
                    }
                )

        return demotions

    def apply_demotions(
        self,
        policy_result: EditPolicyResult,
        artifact: dict[str, Any],
        project: Project,
        agent_id: str | None = None,
    ) -> None:
        """
        Apply demotion effects from an edit policy result.

        Updates the edited artifact and any cascade children to their
        demoted states.

        Args:
            policy_result: Result from check_edit()
            artifact: The artifact being edited
            project: The project containing artifacts
            agent_id: Agent making the edit
        """
        artifact_id = artifact.get("_id")

        # Apply self-demotion if required
        if policy_result.demote_to_state:
            update_fields: dict[str, Any] = {"_lifecycle_state": policy_result.demote_to_state}
            if policy_result.demote_to_store:
                update_fields["_store"] = policy_result.demote_to_store

            if artifact_id:
                logger.info(
                    f"Demoting artifact {artifact_id} to state '{policy_result.demote_to_state}'"
                )
                project.update_artifact(artifact_id, update_fields, _updated_by=agent_id)

        # Apply cascade demotions
        if policy_result.cascade_demotions:
            for demotion in policy_result.cascade_demotions:
                child_id: str = demotion["child_id"]
                child_update: dict[str, Any] = {"_lifecycle_state": demotion["demote_to_state"]}
                if demotion.get("demote_to_store"):
                    child_update["_store"] = demotion["demote_to_store"]

                logger.info(
                    f"Cascade demoting child {child_id} to state "
                    f"'{demotion['demote_to_state']}' (via {demotion['relationship_id']})"
                )
                project.update_artifact(child_id, child_update, _updated_by=agent_id)
