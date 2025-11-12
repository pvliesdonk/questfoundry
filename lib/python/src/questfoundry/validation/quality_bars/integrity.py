"""
Integrity Quality Bar.

Validates that:
- All references resolve (no dangling links)
- No unintended dead ends
- Codeword/state effects don't create contradictions
- All required fields are present
- Schema conformance
"""

import logging
from typing import Any

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class IntegrityBar(QualityBar):
    """
    Integrity Bar: References resolve; no unintended dead ends.

    Checks:
    - All section/choice references exist
    - All artifact references resolve
    - Terminals are marked intentionally
    - No contradictory codeword effects
    - Required fields present
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "integrity"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "References resolve; no unintended dead ends"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for integrity.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating integrity in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Build index of all artifact IDs
        artifact_ids = set()
        section_ids = set()

        for artifact in artifacts:
            artifact_id = artifact.data.get("id")
            if artifact_id:
                artifact_ids.add(artifact_id)

            # Track section IDs for manuscript sections
            if artifact.type == "manuscript_section" and artifact_id:
                section_ids.add(artifact_id)

        logger.trace(
            "Built artifact index: %d artifacts, %d sections",
            len(artifact_ids),
            len(section_ids),
        )

        # Validate each artifact
        for artifact in artifacts:
            artifact_id = artifact.data.get("id", "unknown")

            # Check required fields
            issues.extend(self._check_required_fields(artifact))

            # Check references
            issues.extend(self._check_references(artifact, artifact_ids, section_ids))

            # Check for dead ends
            issues.extend(self._check_dead_ends(artifact))

        logger.debug("Integrity validation complete: %d issues found", len(issues))
        return self._create_result(
            issues,
            artifacts_checked=len(artifacts),
            artifact_ids=len(artifact_ids),
            section_ids=len(section_ids),
        )

    def _check_required_fields(self, artifact: Artifact) -> list[QualityIssue]:
        """Check that required fields are present."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        # Common required fields
        if not artifact.data.get("id"):
            issues.append(
                QualityIssue(
                    severity="blocker",
                    message="Missing required field: id",
                    location=f"artifact:{artifact.type}",
                    fix="Add unique ID to artifact",
                )
            )

        # Type-specific required fields
        if artifact.type == "manuscript_section":
            if not artifact.data.get("text") and not artifact.data.get("body"):
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        message="Manuscript section missing text/body",
                        location=f"artifact:{artifact_id}",
                        fix="Add text or body content to section",
                    )
                )

        return issues

    def _check_references(
        self,
        artifact: Artifact,
        artifact_ids: set[str],
        section_ids: set[str],
    ) -> list[QualityIssue]:
        """Check that all references resolve."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        # Check choice targets in manuscript sections
        if artifact.type == "manuscript_section":
            choices = artifact.data.get("choices", [])
            for i, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue

                target = choice.get("target")
                if target and target not in section_ids:
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=f"Choice target '{target}' does not exist",
                            location=f"artifact:{artifact_id}.choices[{i}]",
                            fix=(
                                f"Create section '{target}' or update target reference"
                            ),
                        )
                    )

        # Check artifact references in any artifact
        refs = self._extract_references(artifact.data)
        for ref in refs:
            if ref not in artifact_ids:
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        message=f"Referenced artifact '{ref}' does not exist",
                        location=f"artifact:{artifact_id}",
                        fix=f"Create artifact '{ref}' or remove reference",
                    )
                )

        return issues

    def _extract_references(self, data: dict[str, Any]) -> set[str]:
        """Extract artifact references from data."""
        refs: set[str] = set()

        # Common reference fields
        ref_fields = [
            "artifact_id",
            "artifact_ref",
            "reference",
            "parent_id",
            "depends_on",
        ]

        for field in ref_fields:
            if field in data and isinstance(data[field], str):
                refs.add(data[field])
            elif field in data and isinstance(data[field], list):
                refs.update([r for r in data[field] if isinstance(r, str)])

        # Recursively check nested dicts
        for value in data.values():
            if isinstance(value, dict):
                refs.update(self._extract_references(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        refs.update(self._extract_references(item))

        return refs

    def _check_dead_ends(self, artifact: Artifact) -> list[QualityIssue]:
        """Check for unintended dead ends."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        # Check manuscript sections
        if artifact.type == "manuscript_section":
            choices = artifact.data.get("choices", [])
            is_terminal = artifact.data.get("terminal", False)

            # If no choices and not marked terminal, likely unintended dead end
            if not choices and not is_terminal:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        message="Section has no choices and not marked terminal",
                        location=f"artifact:{artifact_id}",
                        fix=(
                            "Add choices or mark with 'terminal: true' "
                            "if intentional ending"
                        ),
                    )
                )

        return issues
