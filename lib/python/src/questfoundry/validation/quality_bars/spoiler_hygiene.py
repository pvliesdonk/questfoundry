"""
Spoiler Hygiene Quality Bar.

Validates that:
- Spoilers properly masked in player-safe views
- PN boundaries maintained
- Cold content player-safe
- Content masking applied correctly
"""

import logging

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class SpoilerHygieneBar(QualityBar):
    """
    Spoiler Hygiene Bar: Proper content masking and PN boundaries.

    Checks:
    - Spoilers stripped from player-safe views
    - PN only receives cold + player_safe
    - Hidden content properly marked
    - No leaks of future content
    """

    # Fields that contain spoilers
    SPOILER_FIELDS = [
        "spoilers",
        "hidden",
        "secret",
        "gm_notes",
        "author_notes",
        "internal_notes",
        "canon_notes",
    ]

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "spoiler_hygiene"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Spoilers properly masked; PN boundaries maintained"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for spoiler hygiene.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating spoiler hygiene in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Check each artifact for spoiler hygiene
        for artifact in artifacts:
            artifact_id = artifact.data.get("id", "unknown")

            # Check if artifact is marked player_safe
            metadata = artifact.metadata or {}
            player_safe = metadata.get("player_safe", False)
            temperature = metadata.get("temperature", "hot")

            logger.trace(
                "Checking artifact %s: player_safe=%s, temperature=%s",
                artifact_id,
                player_safe,
                temperature,
            )

            # If cold and player_safe, check for spoiler leaks
            if temperature == "cold" and player_safe:
                issues.extend(self._check_spoiler_leaks(artifact))

            # If has spoiler fields, should not be player_safe
            if player_safe:
                for field in self.SPOILER_FIELDS:
                    if artifact.data.get(field):
                        issues.append(
                            QualityIssue(
                                severity="blocker",
                                message=(
                                    f"Player-safe artifact contains spoiler "
                                    f"field '{field}'"
                                ),
                                location=f"artifact:{artifact_id}.{field}",
                                fix=(
                                    f"Remove '{field}' from player-safe artifact "
                                    f"or move to separate canon artifact"
                                ),
                            )
                        )

            # Check for future content leaks in choices
            if artifact.type == "manuscript_section":
                issues.extend(self._check_choice_spoilers(artifact))

        logger.debug(
            "Spoiler hygiene validation complete: %d issues found", len(issues)
        )
        return self._create_result(issues, artifacts_checked=len(artifacts))

    def _check_spoiler_leaks(self, artifact: Artifact) -> list[QualityIssue]:
        """Check for spoiler leaks in text fields."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        # Check main text for spoiler markers
        text = artifact.data.get("text", artifact.data.get("body", ""))

        if text:
            # Look for spoiler markup that wasn't stripped
            spoiler_markers = [
                "||",  # Discord-style spoiler
                "[spoiler]",
                "{{spoiler}}",
                "<spoiler>",
            ]

            for marker in spoiler_markers:
                if marker in text:
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=(
                                f"Spoiler marker '{marker}' found in player-safe text"
                            ),
                            location=f"artifact:{artifact_id}",
                            fix="Remove or properly mask spoiler content",
                        )
                    )

        return issues

    def _check_choice_spoilers(self, artifact: Artifact) -> list[QualityIssue]:
        """Check for future content spoilers in choice text."""
        issues: list[QualityIssue] = []
        artifact_id = artifact.data.get("id", "unknown")

        choices = artifact.data.get("choices", [])

        for i, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue

            choice_text = choice.get("text", "")

            # Check for overly revealing choice text
            # Simple heuristic: choices shouldn't reveal outcomes
            revealing_phrases = [
                "and then",
                "which leads to",
                "and die",
                "and live",
                "successfully",
                "and fail",
            ]

            for phrase in revealing_phrases:
                if phrase in choice_text.lower():
                    issues.append(
                        QualityIssue(
                            severity="info",
                            message=(
                                f"Choice text may reveal outcome: '{choice_text[:50]}'"
                            ),
                            location=f"artifact:{artifact_id}.choices[{i}]",
                            fix=(
                                "Avoid revealing outcomes in choice text; "
                                "keep possibilities open"
                            ),
                        )
                    )
                    break

        return issues
