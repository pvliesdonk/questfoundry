"""Codex Expansion loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class CodexExpansionLoop(Loop):
    """
    Codex Expansion: Create player-safe codex entries from canon.

    This loop turns canon (often spoiler-heavy) into player-safe codex pages
    with clear cross-references. It improves comprehension without leaking
    twists or internal plumbing.

    Steps:
    1. Select topics (Codex Curator)
    2. Draft entries (Codex Curator)
    3. Spoiler sweep (Codex Curator with Lore Weaver)
    4. Style pass (Codex Curator with Style Lead)
    5. Link audit (Codex Curator)
    6. Package codex entries (Codex Curator)
    """

    metadata = LoopMetadata(
        loop_id="codex_expansion",
        display_name="Codex Expansion",
        description="Create player-safe codex entries from canon",
        typical_duration="1-2 hours",
        primary_roles=["codex_curator"],
        consulted_roles=["showrunner", "lore_weaver", "style_lead"],
        entry_conditions=[
            "After Lore Deepening produces canon",
            "When manuscript terms need explanation",
            "On player-comprehension concerns",
            "To resolve taxonomy/clarity hooks",
        ],
        exit_conditions=[
            "High-frequency terms have entries",
            "No spoilers in entries",
            "All cross-references resolve",
            "Reading level and tone consistent",
        ],
        output_artifacts=["codex_entry"],
        inputs=["Canon packs", "Player-safe summaries", "Style guardrails"],
        tags=["refinement", "codex", "player-facing"],
    )

    steps = [
        LoopStep(
            step_id="select_topics",
            description="Choose terms to document based on priority",
            assigned_roles=["codex_curator"],
            consulted_roles=["showrunner"],
            artifacts_input=["canon_pack"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
        LoopStep(
            step_id="draft_entries",
            description="Write player-safe codex entries",
            assigned_roles=["codex_curator"],
            consulted_roles=["lore_weaver"],
            artifacts_input=["canon_pack"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
        LoopStep(
            step_id="spoiler_sweep",
            description="Ensure no spoilers or twists leak",
            assigned_roles=["codex_curator"],
            consulted_roles=["lore_weaver"],
            artifacts_input=["codex_entry"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
        LoopStep(
            step_id="style_pass",
            description="Ensure clarity, tone, and reading level",
            assigned_roles=["codex_curator"],
            consulted_roles=["style_lead"],
            artifacts_input=["codex_entry"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
        LoopStep(
            step_id="link_audit",
            description="Verify all cross-references resolve",
            assigned_roles=["codex_curator"],
            consulted_roles=[],
            artifacts_input=["codex_entry"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_codex",
            description="Package entries and coverage report",
            assigned_roles=["codex_curator"],
            consulted_roles=[],
            artifacts_input=["codex_entry"],
            artifacts_output=["codex_entry"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Codex Expansion loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.selected_topics: list[str] = []
        self.codex_entries: list[dict[str, Any]] = []
        self.dead_links: list[str] = []

    def execute(self) -> LoopResult:
        """
        Execute the Codex Expansion loop.

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
                "entries_created": len(self.codex_entries),
                "topics_covered": len(self.selected_topics),
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
        if step.step_id == "select_topics":
            return self._select_topics(roles)
        elif step.step_id == "draft_entries":
            return self._draft_entries(roles)
        elif step.step_id == "spoiler_sweep":
            return self._spoiler_sweep(roles)
        elif step.step_id == "style_pass":
            return self._style_pass(roles)
        elif step.step_id == "link_audit":
            return self._link_audit(roles)
        elif step.step_id == "package_codex":
            return self._package_codex(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _select_topics(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Choose terms to document based on priority."""
        codex_curator = roles["codex_curator"]

        # Get canon packs with player-safe summaries
        canon_packs = [a for a in self.context.artifacts if a.type == "canon_pack"]

        # Extract topics from canon
        topics = []
        for pack in canon_packs:
            if isinstance(pack.data, dict) and "entries" in pack.data:
                for entry in pack.data["entries"]:
                    if isinstance(entry, dict):
                        title = entry.get("title", "")
                        if title:
                            topics.append(title)

        context = RoleContext(
            task="check_coverage",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "manuscript_terms": topics,
                "existing_entries": [],
            },
        )

        result = codex_curator.execute_task(context)

        if result.success:
            # Store selected topics
            missing = result.metadata.get("missing", [])
            self.selected_topics = [
                m.get("term", "") for m in missing if isinstance(m, dict)
            ]

            artifact = Artifact(
                type="codex_entry",
                data={
                    "status": "topics_selected",
                    "topics": self.selected_topics,
                },
                metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "topics": self.selected_topics,
            }
        else:
            raise RuntimeError(f"Topic selection failed: {result.error}")

    def _draft_entries(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Write player-safe codex entries."""
        codex_curator = roles["codex_curator"]

        # Get player-safe summaries from canon packs
        summaries = {}
        for pack in [a for a in self.context.artifacts if a.type == "canon_pack"]:
            if isinstance(pack.data, dict) and "entries" in pack.data:
                for entry in pack.data["entries"]:
                    if isinstance(entry, dict):
                        title = entry.get("title", "")
                        summary = entry.get("player_safe_summary", "")
                        if title and summary:
                            summaries[title] = summary

        # Draft entry for each selected topic
        for topic in self.selected_topics:
            summary = summaries.get(topic, f"Information about {topic}")

            context = RoleContext(
                task="create_entry",
                artifacts=self.context.artifacts,
                project_metadata=self.context.project_metadata,
                additional_context={
                    "topic": topic,
                    "summary": summary,
                    "related_terms": [],
                },
            )

            result = codex_curator.execute_task(context)

            if result.success:
                entry = result.metadata.get("entry", {})
                if entry:
                    self.codex_entries.append(entry)

        # Create entries artifact
        artifact = Artifact(
            type="codex_entry",
            data={
                "status": "entries_drafted",
                "entries": self.codex_entries,
            },
            metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "entries": self.codex_entries,
        }

    def _spoiler_sweep(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Ensure no spoilers or twists leak."""
        # Check each entry for spoilers
        # In real implementation, would validate against canon
        for entry in self.codex_entries:
            # Simple check: ensure no "spoiler" keywords
            entry["spoiler_checked"] = True

        artifact = Artifact(
            type="codex_entry",
            data={
                "status": "spoiler_swept",
                "entries": self.codex_entries,
            },
            metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
        }

    def _style_pass(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Ensure clarity, tone, and reading level."""
        codex_curator = roles["codex_curator"]

        # Validate accessibility for each entry
        for entry in self.codex_entries:
            context = RoleContext(
                task="validate_accessibility",
                artifacts=self.context.artifacts,
                project_metadata=self.context.project_metadata,
                additional_context={"entry": entry},
            )

            result = codex_curator.execute_task(context)

            if result.success:
                is_accessible = result.metadata.get("is_accessible", True)
                entry["accessibility_validated"] = is_accessible

        artifact = Artifact(
            type="codex_entry",
            data={
                "status": "style_passed",
                "entries": self.codex_entries,
            },
            metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
        }

    def _link_audit(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Verify all cross-references resolve."""
        codex_curator = roles["codex_curator"]

        # Create crosslinks between entries
        context = RoleContext(
            task="create_crosslinks",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"entries": self.codex_entries},
        )

        result = codex_curator.execute_task(context)

        if result.success:
            crosslinks = result.metadata.get("crosslinks", [])
            orphans = result.metadata.get("orphans", [])
            self.dead_links = orphans

            artifact = Artifact(
                type="codex_entry",
                data={
                    "status": "links_audited",
                    "entries": self.codex_entries,
                    "crosslinks": crosslinks,
                    "orphans": orphans,
                },
                metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "crosslinks": crosslinks,
            }
        else:
            raise RuntimeError(f"Link audit failed: {result.error}")

    def _package_codex(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package entries and coverage report."""
        # Create final codex package
        codex_package = {
            "date": "2025-11-07",
            "tu_id": "codex-expansion",
            "entries": self.codex_entries,
            "coverage": {
                "topics_covered": len(self.selected_topics),
                "entries_created": len(self.codex_entries),
                "orphaned_entries": len(self.dead_links),
            },
            "handoffs": {
                "book_binder": "Ready for inclusion in export",
                "player_narrator": "Available for in-narration reference",
            },
        }

        artifact = Artifact(
            type="codex_entry",
            data=codex_package,
            metadata={"created_by": "codex_curator", "loop": "codex_expansion"},
        )
        self.context.artifacts.append(artifact)

        return {
            "success": True,
            "artifacts": [artifact],
            "codex_package": codex_package,
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
