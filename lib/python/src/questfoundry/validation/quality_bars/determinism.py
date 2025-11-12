"""
Determinism Quality Bar.

Validates that:
- Visual/audio assets can be reproduced from parameters (when promised)
- Parameters logged (seed, prompt, model, version, etc.)
- Plan-only items properly marked and deferred
- Convergence reflects entering state
"""

import logging

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class DeterminismBar(QualityBar):
    """
    Determinism Bar: Promised assets reproducible from parameters.

    Checks:
    - Asset parameters logged (seed, model, version)
    - Plan-only items marked as deferred
    - Params sufficient for reproduction
    - N/A if determinism not promised
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "determinism"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Promised assets are reproducible from recorded parameters"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for determinism.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating determinism in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Check visual assets
        visual_artifacts = [
            a for a in artifacts if a.type in ["visual_asset", "image_plan", "art_plan"]
        ]

        logger.trace("Found %d visual artifacts to validate", len(visual_artifacts))

        for artifact in visual_artifacts:
            artifact_id = artifact.data.get("id", "unknown")

            # Check if this is a plan or executed asset
            is_plan = (
                artifact.type.endswith("_plan")
                or artifact.data.get("status") == "planned"
            )

            if is_plan:
                # Plans should be marked deferred
                if not artifact.data.get("deferred"):
                    logger.debug("Plan asset %s not marked as deferred", artifact_id)
                    issues.append(
                        QualityIssue(
                            severity="info",
                            message="Plan-only asset not marked as deferred",
                            location=f"artifact:{artifact_id}",
                            fix="Mark with 'deferred: true' or 'status: planned'",
                        )
                    )
            else:
                # Executed assets should have generation params
                issues.extend(self._check_asset_params(artifact))

        # Check audio assets
        audio_artifacts = [
            a for a in artifacts if a.type in ["audio_asset", "audio_plan"]
        ]

        logger.trace("Found %d audio artifacts to validate", len(audio_artifacts))

        for artifact in audio_artifacts:
            is_plan = (
                artifact.type == "audio_plan"
                or artifact.data.get("status") == "planned"
            )

            if not is_plan:
                issues.extend(self._check_asset_params(artifact))

        logger.debug("Determinism validation complete: %d issues found", len(issues))
        return self._create_result(
            issues,
            visual_assets=len(visual_artifacts),
            audio_assets=len(audio_artifacts),
        )

    def _check_asset_params(self, artifact: Artifact) -> list[QualityIssue]:
        """Check that asset has required generation parameters."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        # Required params for reproducibility
        params = artifact.data.get("params", artifact.data.get("parameters", {}))

        if not params:
            issues.append(
                QualityIssue(
                    severity="warning",
                    message="Asset missing generation parameters",
                    location=f"artifact:{artifact_id}",
                    fix="Add 'params' field with seed, model, prompt_version",
                )
            )
            return issues

        # Check for key params
        if not params.get("seed"):
            issues.append(
                QualityIssue(
                    severity="warning",
                    message="Asset params missing 'seed' for reproducibility",
                    location=f"artifact:{artifact_id}.params",
                    fix="Record seed used for generation",
                )
            )

        if not params.get("model") and not params.get("model_version"):
            issues.append(
                QualityIssue(
                    severity="warning",
                    message="Asset params missing 'model' or 'model_version'",
                    location=f"artifact:{artifact_id}.params",
                    fix="Record model name/version used",
                )
            )

        return issues
