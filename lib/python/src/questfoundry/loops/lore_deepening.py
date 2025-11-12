"""Lore Deepening loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class LoreDeepeningLoop(Loop):
    """
    Lore Deepening: Transform accepted hooks into coherent canon.

    This loop takes accepted hooks and expands them into canonical entries
    (backstories, timelines, metaphysics, causal links). It keeps spoilers
    in canon notes and publishes player-safe surfaces via Codex Expansion.

    Steps:
    1. Frame canon questions (Lore Weaver)
    2. Draft canon answers (Lore Weaver)
    3. Check contradictions and coupling (Lore Weaver)
    4. Create topology and prose notes (Lore Weaver)
    5. Package canon entries (Lore Weaver)
    """

    metadata = LoopMetadata(
        loop_id="lore_deepening",
        display_name="Lore Deepening",
        description="Transform accepted hooks into coherent canon",
        typical_duration="2-3 hours",
        primary_roles=["lore_weaver"],
        consulted_roles=["showrunner", "researcher", "plotwright", "scene_smith"],
        entry_conditions=[
            "After Hook Harvest marks hooks as accepted",
            "When Plot/Scene need causal backfill",
            "When contradictions must be adjudicated",
        ],
        exit_conditions=[
            "Hooks canonized or deferred with reason",
            "Contradictions resolved or marked as mystery",
            "Player-safe summaries ready for Codex",
            "Topology/prose impacts enumerated",
        ],
        output_artifacts=["canon_pack"],
        inputs=["Accepted hooks", "Prior Canon Packs", "Topology notes"],
        tags=["discovery", "canon", "world-building"],
    )

    steps = [
        LoopStep(
            step_id="frame_questions",
            description="Frame canon questions from hooks",
            assigned_roles=["lore_weaver"],
            consulted_roles=["showrunner"],
            artifacts_input=["hook_card"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="draft_canon",
            description="Draft spoiler-level canon answers",
            assigned_roles=["lore_weaver"],
            consulted_roles=["researcher"],
            artifacts_input=["hook_card", "canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="check_contradictions",
            description="Compare against Cold and resolve conflicts",
            assigned_roles=["lore_weaver"],
            consulted_roles=["showrunner"],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="create_impact_notes",
            description="Document topology and prose implications",
            assigned_roles=["lore_weaver"],
            consulted_roles=["plotwright", "scene_smith"],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_canon",
            description="Package canon entries with player-safe summaries",
            assigned_roles=["lore_weaver"],
            consulted_roles=[],
            artifacts_input=["canon_pack"],
            artifacts_output=["canon_pack"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Lore Deepening loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.canon_questions: list[dict[str, Any]] = []
        self.canon_entries: list[dict[str, Any]] = []
        self.contradictions_found: list[dict[str, Any]] = []

    def execute(self) -> LoopResult:
        """
        Execute the Lore Deepening loop.

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
                "canon_entries_created": len(self.canon_entries),
                "contradictions_resolved": len(self.contradictions_found),
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
        if step.step_id == "frame_questions":
            return self._frame_questions(roles)
        elif step.step_id == "draft_canon":
            return self._draft_canon(roles)
        elif step.step_id == "check_contradictions":
            return self._check_contradictions(roles)
        elif step.step_id == "create_impact_notes":
            return self._create_impact_notes(roles)
        elif step.step_id == "package_canon":
            return self._package_canon(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _frame_questions(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Frame canon questions from hooks."""
        lore_weaver = roles["lore_weaver"]

        # Get accepted hooks
        accepted_hooks = [
            a
            for a in self.context.artifacts
            if a.type == "hook_card"
            and isinstance(a.data, dict)
            and a.data.get("status") == "accepted"
        ]

        # Extract hooks data
        hooks = []
        for artifact in accepted_hooks:
            if isinstance(artifact.data, dict):
                hooks.append(
                    {
                        "title": artifact.data.get("title", "Untitled"),
                        "summary": artifact.data.get("summary", ""),
                    }
                )

        context = RoleContext(
            task="expand_canon",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "hooks": hooks,
                "cluster_name": "accepted_hooks",
            },
        )

        result = lore_weaver.execute_task(context)

        if result.success:
            # Store questions
            self.canon_questions = result.metadata.get("entries", [])

            # Create canon pack with questions
            artifact = Artifact(
                type="canon_pack",
                data={
                    "status": "questions_framed",
                    "questions": self.canon_questions,
                },
                metadata={"created_by": "lore_weaver", "loop": "lore_deepening"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "questions": self.canon_questions,
            }
        else:
            raise RuntimeError(f"Question framing failed: {result.error}")

    def _draft_canon(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Draft spoiler-level canon answers."""
        lore_weaver = roles["lore_weaver"]

        # Use the questions we framed
        context = RoleContext(
            task="expand_canon",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "hooks": self.canon_questions,
                "cluster_name": "canon_answers",
            },
        )

        result = lore_weaver.execute_task(context)

        if result.success:
            # Store canon entries
            self.canon_entries = result.metadata.get("entries", [])

            # Create canon pack with answers
            artifact = Artifact(
                type="canon_pack",
                data={
                    "status": "canon_drafted",
                    "entries": self.canon_entries,
                },
                metadata={"created_by": "lore_weaver", "loop": "lore_deepening"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "entries": self.canon_entries,
            }
        else:
            raise RuntimeError(f"Canon drafting failed: {result.error}")

    def _check_contradictions(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Compare against Cold and resolve conflicts."""
        lore_weaver = roles["lore_weaver"]

        # Check each canon entry for consistency
        for entry in self.canon_entries:
            context = RoleContext(
                task="check_canon_consistency",
                artifacts=self.context.artifacts,
                project_metadata=self.context.project_metadata,
                additional_context={"new_canon": entry},
            )

            result = lore_weaver.execute_task(context)

            if result.success:
                is_consistent = result.metadata.get("is_consistent", True)
                if not is_consistent:
                    issues = result.metadata.get("issues", [])
                    self.contradictions_found.extend(issues)

        # Create contradiction report
        artifact = Artifact(
            type="canon_pack",
            data={
                "status": "contradictions_checked",
                "entries": self.canon_entries,
                "contradictions": self.contradictions_found,
                "is_consistent": len(self.contradictions_found) == 0,
            },
            metadata={"created_by": "lore_weaver", "loop": "lore_deepening"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "contradictions": self.contradictions_found,
        }

    def _create_impact_notes(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Document topology and prose implications."""
        # Generate impact notes for each entry
        impact_notes: dict[str, list[dict[str, str]]] = {
            "plotwright": [],
            "scene_smith": [],
            "style_lead": [],
        }

        for entry in self.canon_entries:
            # Extract downstream impacts from entry
            if "downstream_impacts" in entry:
                impacts = entry["downstream_impacts"]
                for role, note in impacts.items():
                    if role in impact_notes:
                        impact_notes[role].append(
                            {"entry": entry.get("title", ""), "note": note}
                        )

        # Create impact notes artifact
        artifact = Artifact(
            type="canon_pack",
            data={
                "status": "impacts_noted",
                "entries": self.canon_entries,
                "impact_notes": impact_notes,
            },
            metadata={"created_by": "lore_weaver", "loop": "lore_deepening"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "impact_notes": impact_notes,
        }

    def _package_canon(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package canon entries with player-safe summaries."""
        lore_weaver = roles["lore_weaver"]

        # Generate player-safe summaries for each entry
        for entry in self.canon_entries:
            if "player_safe_summary" not in entry:
                context = RoleContext(
                    task="generate_player_summary",
                    artifacts=self.context.artifacts,
                    project_metadata=self.context.project_metadata,
                    additional_context={"canon_entry": entry},
                )

                result = lore_weaver.execute_task(context)

                if result.success:
                    entry["player_safe_summary"] = result.output

        # Create final canon pack
        canon_pack = {
            "date": "2025-11-07",
            "tu_id": "lore-deepening",
            "entries": self.canon_entries,
            "contradictions_resolved": len(self.contradictions_found),
            "handoffs": {
                "codex_expansion": "Player-safe summaries ready",
                "scene_smith": "Scene callback notes included",
                "plotwright": "Topology implications noted",
            },
        }

        artifact = Artifact(
            type="canon_pack",
            data=canon_pack,
            metadata={"created_by": "lore_weaver", "loop": "lore_deepening"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "canon_pack": canon_pack,
        }

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
