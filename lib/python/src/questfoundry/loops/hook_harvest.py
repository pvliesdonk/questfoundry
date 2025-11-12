"""Hook Harvest loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class HookHarvestLoop(Loop):
    """
    Hook Harvest: Triage and prioritize proposed hooks.

    This loop sweeps up newly proposed hooks (narrative, scene, factual, taxonomy),
    removes duplicates, clusters related ideas, and triages what should advance now,
    later, or never. Outcome: A prioritized, tagged hook set ready for Lore Deepening
    and follow-on loops, with risks and dependencies made explicit.

    Steps:
    1. Collect hooks (Showrunner)
    2. Cluster hooks by theme (Showrunner)
    3. Annotate hooks with tags and dependencies (Showrunner with consultants)
    4. Decide on triage (accept/defer/reject) (Showrunner)
    5. Package harvest sheet (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="hook_harvest",
        display_name="Hook Harvest",
        description="Triage and prioritize proposed hooks",
        typical_duration="1-2 hours",
        primary_roles=["showrunner"],
        consulted_roles=[
            "lore_weaver",
            "plotwright",
            "scene_smith",
            "codex_curator",
        ],
        entry_conditions=[
            "After Story Spark or burst of drafting",
            "Before stabilization window",
            "When backlog looks fuzzy or drifted",
        ],
        exit_conditions=[
            "Hooks deduped and clustered",
            "Each hook has triage decision",
            "Clear next loop and owner for accepted hooks",
        ],
        output_artifacts=["hook_card"],
        inputs=["Hook cards with status proposed", "Recent topology notes"],
        tags=["discovery", "triage", "planning"],
    )

    steps = [
        LoopStep(
            step_id="collect_hooks",
            description="Collect all proposed hooks and remove duplicates",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["hook_card"],
            artifacts_output=["hook_card"],
            validation_required=True,
        ),
        LoopStep(
            step_id="cluster_hooks",
            description="Group hooks by theme and type",
            assigned_roles=["showrunner"],
            consulted_roles=["lore_weaver", "plotwright"],
            artifacts_input=["hook_card"],
            artifacts_output=["hook_card"],
            validation_required=True,
        ),
        LoopStep(
            step_id="annotate_hooks",
            description="Add triage tags, uncertainty levels, and dependencies",
            assigned_roles=["showrunner"],
            consulted_roles=[
                "lore_weaver",
                "plotwright",
                "scene_smith",
                "codex_curator",
            ],
            artifacts_input=["hook_card"],
            artifacts_output=["hook_card"],
            validation_required=True,
        ),
        LoopStep(
            step_id="decide_triage",
            description="Mark each hook as accepted, deferred, or rejected",
            assigned_roles=["showrunner"],
            consulted_roles=["lore_weaver", "plotwright"],
            artifacts_input=["hook_card"],
            artifacts_output=["hook_card"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_harvest",
            description="Produce harvest sheet with decisions and handoffs",
            assigned_roles=["showrunner"],
            consulted_roles=[],
            artifacts_input=["hook_card"],
            artifacts_output=["harvest_sheet"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Hook Harvest loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.hooks_collected: list[dict[str, Any]] = []
        self.clusters: list[dict[str, Any]] = []
        self.triage_decisions: dict[str, str] = {}

    def execute(self) -> LoopResult:
        """
        Execute the Hook Harvest loop.

        Returns:
            Result of loop execution
        """
        artifacts_created: list[Artifact] = []
        artifacts_modified: list[Artifact] = []
        steps_completed = 0
        steps_failed = 0

        # Execute each step in sequence
        for step in self.steps:
            try:
                self.execute_step(step)

                if step.status == StepStatus.COMPLETED:
                    steps_completed += 1

                    # Collect artifacts created in this step
                    if step.result and isinstance(step.result, dict):
                        if "artifacts" in step.result:
                            artifacts_created.extend(step.result["artifacts"])

                elif step.status == StepStatus.FAILED:
                    steps_failed += 1

                    # Abort on failure
                    return LoopResult(
                        success=False,
                        loop_id=self.metadata.loop_id,
                        artifacts_created=artifacts_created,
                        artifacts_modified=artifacts_modified,
                        steps_completed=steps_completed,
                        steps_failed=steps_failed,
                        error=f"Step '{step.step_id}' failed: {step.error}",
                    )

            except Exception as e:
                return LoopResult(
                    success=False,
                    loop_id=self.metadata.loop_id,
                    artifacts_created=artifacts_created,
                    artifacts_modified=artifacts_modified,
                    steps_completed=steps_completed,
                    steps_failed=steps_failed + 1,
                    error=f"Exception in step '{step.step_id}': {str(e)}",
                )

        # All steps completed successfully
        return LoopResult(
            success=True,
            loop_id=self.metadata.loop_id,
            artifacts_created=artifacts_created,
            artifacts_modified=artifacts_modified,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            metadata={
                "hooks_processed": len(self.hooks_collected),
                "clusters_created": len(self.clusters),
                "accepted": len(
                    [d for d in self.triage_decisions.values() if d == "accepted"]
                ),
                "deferred": len(
                    [d for d in self.triage_decisions.values() if d == "deferred"]
                ),
                "rejected": len(
                    [d for d in self.triage_decisions.values() if d == "rejected"]
                ),
            },
        )

    def _execute_step_logic(self, step: LoopStep, roles: dict[str, Role]) -> Any:
        """
        Execute specific logic for each step.

        Args:
            step: Step being executed
            roles: Available roles

        Returns:
            Step result
        """
        if step.step_id == "collect_hooks":
            return self._collect_hooks(roles)
        elif step.step_id == "cluster_hooks":
            return self._cluster_hooks(roles)
        elif step.step_id == "annotate_hooks":
            return self._annotate_hooks(roles)
        elif step.step_id == "decide_triage":
            return self._decide_triage(roles)
        elif step.step_id == "package_harvest":
            return self._package_harvest(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _collect_hooks(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Collect all proposed hooks and remove duplicates."""
        showrunner = roles["showrunner"]

        # Find all hook_card artifacts with status "proposed"
        proposed_hooks = [
            a
            for a in self.context.artifacts
            if a.type == "hook_card"
            and isinstance(a.data, dict)
            and a.data.get("status") == "proposed"
        ]

        context = RoleContext(
            task="review_progress",
            artifacts=proposed_hooks,
            project_metadata=self.context.project_metadata,
            additional_context={
                "tu_id": "hook-harvest",
                "steps_completed": ["Finding proposed hooks"],
                "steps_remaining": ["Dedupe", "Cluster", "Triage"],
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Store hooks for later steps
            self.hooks_collected = [
                {
                    "id": a.artifact_id,
                    "title": a.data.get("title", "Untitled")
                    if isinstance(a.data, dict)
                    else "Untitled",
                    "data": a.data,
                }
                for a in proposed_hooks
            ]

            # Create collection artifact
            artifact = Artifact(
                type="hook_card",
                data={
                    "status": "collected",
                    "hooks": self.hooks_collected,
                    "count": len(self.hooks_collected),
                },
                metadata={"created_by": "showrunner", "loop": "hook_harvest"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "hooks_count": len(self.hooks_collected),
            }
        else:
            raise RuntimeError(f"Hook collection failed: {result.error}")

    def _cluster_hooks(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Group hooks by theme and type."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "cluster_hooks",
                "from_role": "showrunner",
                "to_role": "showrunner",
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create simple clusters based on hook types
            # In real implementation, would use LLM to create thematic clusters
            clusters: dict[str, list[dict[str, Any]]] = {
                "narrative": [],
                "scene": [],
                "factual": [],
                "taxonomy": [],
            }

            for hook in self.hooks_collected:
                hook_type = (
                    hook.get("data", {}).get("type", "narrative")
                    if isinstance(hook.get("data"), dict)
                    else "narrative"
                )
                if hook_type in clusters:
                    clusters[hook_type].append(hook)
                else:
                    clusters["narrative"].append(hook)

            self.clusters = [
                {"theme": theme, "hooks": hooks}
                for theme, hooks in clusters.items()
                if hooks
            ]

            # Create cluster artifact
            artifact = Artifact(
                type="hook_card",
                data={
                    "status": "clustered",
                    "clusters": self.clusters,
                    "cluster_count": len(self.clusters),
                },
                metadata={"created_by": "showrunner", "loop": "hook_harvest"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "clusters": self.clusters,
            }
        else:
            raise RuntimeError(f"Hook clustering failed: {result.error}")

    def _annotate_hooks(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Add triage tags, uncertainty levels, and dependencies."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "annotate_hooks",
                "from_role": "showrunner",
                "to_role": "consultants",
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Annotate hooks with triage tags
            # In real implementation, would consult other roles
            annotated_hooks = []
            for hook in self.hooks_collected:
                hook_data = hook.copy()
                hook_data["tags"] = ["needs-review"]
                hook_data["uncertainty"] = "low"
                hook_data["dependencies"] = []
                annotated_hooks.append(hook_data)

            # Create annotation artifact
            artifact = Artifact(
                type="hook_card",
                data={
                    "status": "annotated",
                    "hooks": annotated_hooks,
                },
                metadata={"created_by": "showrunner", "loop": "hook_harvest"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "annotated_count": len(annotated_hooks),
            }
        else:
            raise RuntimeError(f"Hook annotation failed: {result.error}")

    def _decide_triage(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Mark each hook as accepted, deferred, or rejected."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "decide_triage",
                "from_role": "consultants",
                "to_role": "showrunner",
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Make triage decisions
            # In real implementation, would use LLM and consultation
            for hook in self.hooks_collected:
                hook_id = hook.get("id", "unknown")
                # Default: accept all hooks for now
                self.triage_decisions[hook_id] = "accepted"

            # Create triage artifact
            artifact = Artifact(
                type="hook_card",
                data={
                    "status": "triaged",
                    "decisions": self.triage_decisions,
                    "accepted": [
                        h
                        for h in self.hooks_collected
                        if self.triage_decisions.get(h.get("id", ""), "") == "accepted"
                    ],
                    "deferred": [],
                    "rejected": [],
                },
                metadata={"created_by": "showrunner", "loop": "hook_harvest"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "decisions": self.triage_decisions,
            }
        else:
            raise RuntimeError(f"Triage decision failed: {result.error}")

    def _package_harvest(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Produce harvest sheet with decisions and handoffs."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="review_progress",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "tu_id": "hook-harvest",
                "steps_completed": [
                    "Collect",
                    "Cluster",
                    "Annotate",
                    "Decide",
                ],
                "steps_remaining": [],
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create harvest sheet
            harvest_sheet = {
                "date": "2025-11-07",
                "tu_id": "hook-harvest",
                "clusters": self.clusters,
                "decisions": self.triage_decisions,
                "accepted": [
                    h
                    for h in self.hooks_collected
                    if self.triage_decisions.get(h.get("id", ""), "") == "accepted"
                ],
                "deferred": [
                    h
                    for h in self.hooks_collected
                    if self.triage_decisions.get(h.get("id", ""), "") == "deferred"
                ],
                "rejected": [
                    h
                    for h in self.hooks_collected
                    if self.triage_decisions.get(h.get("id", ""), "") == "rejected"
                ],
                "handoffs": {
                    "lore_deepening": "Accepted narrative hooks",
                    "story_spark": "Topology-affecting hooks",
                    "codex_expansion": "Taxonomy hooks",
                },
            }

            artifact = Artifact(
                type="harvest_sheet",
                data=harvest_sheet,
                metadata={"created_by": "showrunner", "loop": "hook_harvest"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "harvest_sheet": harvest_sheet,
            }
        else:
            raise RuntimeError(f"Harvest packaging failed: {result.error}")

    def validate_step(self, step: LoopStep, result: Any) -> bool:
        """
        Validate step completion.

        Args:
            step: Step that was executed
            result: Result from step execution

        Returns:
            True if step is valid, False otherwise
        """
        if not isinstance(result, dict):
            return False

        if not result.get("success", False):
            return False

        # Check if artifacts were created
        if "artifacts" in result:
            return len(result["artifacts"]) > 0

        return True
