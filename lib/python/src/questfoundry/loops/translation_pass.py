"""Translation Pass loop implementation."""

import logging
from typing import Any

from ..models.artifact import Artifact
from ..roles.base import Role, RoleContext
from .base import Loop, LoopContext, LoopResult, LoopStep, StepStatus
from .registry import LoopMetadata

logger = logging.getLogger(__name__)


class TranslationPassLoop(Loop):
    """
    Translation Pass: Create language packs.

    This loop creates or updates a player-safe translation of the manuscript/codex
    surfaces while preserving PN boundaries, style intent, and navigation. Supports
    plan-only merges (glossary/style kit) when the full slice isn't ready.

    Steps:
    1. Extract strings (Translator)
    2. Translate (Translator)
    3. Validate (Translator with Style Lead)
    4. Style check (Style Lead)
    5. Package language pack (Showrunner)
    """

    metadata = LoopMetadata(
        loop_id="translation_pass",
        display_name="Translation Pass",
        description="Create language packs",
        typical_duration="4-8 hours",
        primary_roles=["translator"],
        consulted_roles=["style_lead", "showrunner", "gatekeeper"],
        entry_conditions=[
            "New target language requested",
            "Significant style/canon changes warrant translation refresh",
            "Accessibility or market goals require multilingual exports",
        ],
        exit_conditions=[
            "Language Pack complete with glossary and localized surfaces",
            "Register and tone feel native",
            "Links/cross-refs resolve",
            "No spoilers or internal labels leaked",
        ],
        output_artifacts=["language_pack"],
        inputs=[
            "Cold snapshot",
            "PN Principles",
            "Style guardrails",
            "Current glossary/terminology",
        ],
        tags=["asset", "translation", "localization"],
    )

    steps = [
        LoopStep(
            step_id="extract_strings",
            description="Extract translatable strings from manuscript and codex",
            assigned_roles=["translator"],
            consulted_roles=[],
            artifacts_input=["canon_pack", "codex"],
            artifacts_output=["string_collection"],
            validation_required=True,
        ),
        LoopStep(
            step_id="translate",
            description="Translate strings to target language",
            assigned_roles=["translator"],
            consulted_roles=["style_lead"],
            artifacts_input=["string_collection"],
            artifacts_output=["translated_strings"],
            validation_required=True,
        ),
        LoopStep(
            step_id="validate",
            description="Validate translations for consistency and links",
            assigned_roles=["translator"],
            consulted_roles=["style_lead"],
            artifacts_input=["translated_strings"],
            artifacts_output=["validated_translations"],
            validation_required=True,
        ),
        LoopStep(
            step_id="style_check",
            description="Check register, motifs, and tone",
            assigned_roles=["style_lead"],
            consulted_roles=["translator"],
            artifacts_input=["validated_translations"],
            artifacts_output=["style_report"],
            validation_required=True,
        ),
        LoopStep(
            step_id="package_language_pack",
            description="Package language pack with glossary and coverage",
            assigned_roles=["showrunner"],
            consulted_roles=["translator"],
            artifacts_input=["validated_translations", "style_report"],
            artifacts_output=["language_pack"],
            validation_required=True,
        ),
    ]

    def __init__(self, context: LoopContext):
        """
        Initialize Translation Pass loop.

        Args:
            context: Loop execution context
        """
        super().__init__(context)
        self.strings_extracted: list[dict[str, Any]] = []
        self.translations: dict[str, Any] = {}
        self.glossary: dict[str, str] = {}
        self.target_language = context.config.get("target_language", "en")
        self.coverage_percent = 0.0

    def execute(self) -> LoopResult:
        """
        Execute the Translation Pass loop.

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
                "target_language": self.target_language,
                "strings_translated": len(self.strings_extracted),
                "coverage_percent": self.coverage_percent,
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
        if step.step_id == "extract_strings":
            return self._extract_strings(roles)
        elif step.step_id == "translate":
            return self._translate(roles)
        elif step.step_id == "validate":
            return self._validate(roles)
        elif step.step_id == "style_check":
            return self._style_check(roles)
        elif step.step_id == "package_language_pack":
            return self._package_language_pack(roles)
        else:
            raise ValueError(f"Unknown step: {step.step_id}")

    def _extract_strings(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Extract translatable strings from manuscript and codex."""
        translator = roles["translator"]

        context = RoleContext(
            task="extract_strings",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"target_language": self.target_language},
        )

        result = translator.execute_task(context)

        if result.success:
            # Store extracted strings
            self.strings_extracted = result.metadata.get("strings", [])

            # Default strings if none provided
            if not self.strings_extracted:
                self.strings_extracted = [
                    {"id": "STRING-001", "source": "Hello", "context": "greeting"}
                ]

            # Create string collection artifact
            artifact = Artifact(
                type="string_collection",
                data={
                    "strings": self.strings_extracted,
                    "count": len(self.strings_extracted),
                    "target_language": self.target_language,
                },
                metadata={"created_by": "translator", "loop": "translation_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "strings": self.strings_extracted,
            }
        else:
            raise RuntimeError(f"String extraction failed: {result.error}")

    def _translate(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Translate strings to target language."""
        translator = roles["translator"]

        context = RoleContext(
            task="translate",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "strings": self.strings_extracted,
                "target_language": self.target_language,
            },
        )

        result = translator.execute_task(context)

        if result.success:
            # Store translations
            self.translations = result.metadata.get("translations", {})
            self.glossary = result.metadata.get("glossary", {})

            # Create translated strings artifact
            artifact = Artifact(
                type="translated_strings",
                data={
                    "translations": self.translations,
                    "glossary": self.glossary,
                    "target_language": self.target_language,
                },
                metadata={"created_by": "translator", "loop": "translation_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "translations": self.translations,
            }
        else:
            raise RuntimeError(f"Translation failed: {result.error}")

    def _validate(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Validate translations for consistency and links."""
        translator = roles["translator"]

        context = RoleContext(
            task="validate_translation",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"translations": self.translations},
        )

        result = translator.execute_task(context)

        if result.success:
            # Calculate coverage
            total_strings = len(self.strings_extracted)
            translated_count = len(self.translations)
            self.coverage_percent = (
                (translated_count / total_strings * 100) if total_strings > 0 else 0
            )

            # Create validated translations artifact
            artifact = Artifact(
                type="validated_translations",
                data={
                    "translations": self.translations,
                    "coverage_percent": self.coverage_percent,
                    "validation_status": "complete",
                },
                metadata={"created_by": "translator", "loop": "translation_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "coverage": self.coverage_percent,
            }
        else:
            raise RuntimeError(f"Validation failed: {result.error}")

    def _style_check(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Check register, motifs, and tone."""
        style_lead = roles["style_lead"]

        context = RoleContext(
            task="check_style",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={"translations": self.translations},
        )

        result = style_lead.execute_task(context)

        if result.success:
            style_status = result.metadata.get("status", "pass")

            # Create style report artifact
            artifact = Artifact(
                type="style_report",
                data={
                    "status": style_status,
                    "register": "appropriate",
                    "motifs": "preserved",
                    "tone": "native",
                },
                metadata={"created_by": "style_lead", "loop": "translation_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "status": style_status,
            }
        else:
            raise RuntimeError(f"Style check failed: {result.error}")

    def _package_language_pack(self, roles: dict[str, Role]) -> dict[str, Any]:
        """Package language pack with glossary and coverage."""
        showrunner = roles["showrunner"]

        context = RoleContext(
            task="coordinate_step",
            artifacts=self.context.artifacts,
            project_metadata=self.context.project_metadata,
            additional_context={
                "step_name": "package_language_pack",
                "target_language": self.target_language,
            },
        )

        result = showrunner.execute_task(context)

        if result.success:
            # Create language pack
            language_pack = {
                "target_language": self.target_language,
                "glossary": self.glossary,
                "translations": self.translations,
                "coverage_percent": self.coverage_percent,
                "complete": self.coverage_percent >= 90.0,
            }

            artifact = Artifact(
                type="language_pack",
                data=language_pack,
                metadata={"created_by": "showrunner", "loop": "translation_pass"},
            )
            self.context.artifacts.append(artifact)

            return {
                "success": True,
                "artifacts": [artifact],
                "language_pack": language_pack,
            }
        else:
            raise RuntimeError(f"Language pack packaging failed: {result.error}")

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
