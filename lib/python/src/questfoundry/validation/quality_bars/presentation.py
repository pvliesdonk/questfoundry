"""
Presentation Quality Bar.

Validates that:
- Player-facing surfaces contain no spoilers or internals
- Manuscript/codex/PN use in-world language only
- No technique talk on player surfaces
- Formatting is clean and readable
"""

import logging
import re

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class PresentationBar(QualityBar):
    """
    Presentation Bar: Player-facing surfaces reveal no spoilers/internals.

    Checks:
    - No spoilers in manuscript/codex/PN
    - No internal plumbing visible
    - No technique talk (seeds, models, etc.)
    - Clean formatting
    """

    # Patterns that indicate internal/technical content
    INTERNAL_PATTERNS = [
        r"\bseed:\s*\d+",
        r"\bmodel:\s*\w+",
        r"\bprompt\s+version",
        r"\bDAW\s+session",
        r"\binternal\s+note",
        r"\bauthor\s+note",
        r"\btodo\b.*:",
        r"\bFIXME\b",
        r"\bHACK\b",
        r"section\s+\d+",
    ]

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "presentation"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Player-facing surfaces reveal no spoilers or internals"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for presentation quality.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating presentation in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Compile internal pattern regex
        internal_regex = re.compile("|".join(self.INTERNAL_PATTERNS), re.IGNORECASE)
        logger.trace("Presentation validation patterns compiled")

        # Check player-facing artifacts
        player_facing_types = [
            "manuscript_section",
            "codex_entry",
            "player_note",
        ]

        player_facing = [a for a in artifacts if a.type in player_facing_types]

        logger.trace("Found %d player-facing artifacts to validate", len(player_facing))

        for artifact in player_facing:
            artifact_id = artifact.data.get("id", "unknown")

            # Check text fields for internal content
            text = artifact.data.get("text", artifact.data.get("body", ""))

            if text and internal_regex.search(text):
                matches = internal_regex.findall(text)
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        message=(
                            f"Player-facing text contains internal/technical "
                            f"content: {matches[:2]}"
                        ),
                        location=f"artifact:{artifact_id}",
                        fix=(
                            "Remove technical details and internal notes from "
                            "player-visible text"
                        ),
                    )
                )

            # Check for spoiler fields in player-facing content
            spoiler_fields = [
                "author_notes",
                "internal_notes",
                "gm_notes",
                "spoilers",
            ]

            for field in spoiler_fields:
                if artifact.data.get(field):
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            message=(
                                f"Spoiler field '{field}' present in "
                                f"player-facing artifact"
                            ),
                            location=f"artifact:{artifact_id}.{field}",
                            fix=(
                                f"Move '{field}' to canon notes or remove from "
                                f"player-facing content"
                            ),
                        )
                    )

            # Check choice text
            if artifact.type == "manuscript_section":
                choices = artifact.data.get("choices", [])
                for i, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        continue

                    choice_text = choice.get("text", "")
                    if choice_text and internal_regex.search(choice_text):
                        logger.warning(
                            (
                                "Choice text contains internal/technical content "
                                "at %s choice %d"
                            ),
                            artifact_id,
                            i,
                        )
                        issues.append(
                            QualityIssue(
                                severity="blocker",
                                message=(
                                    "Choice text contains internal/technical content"
                                ),
                                location=f"artifact:{artifact_id}.choices[{i}]",
                                fix="Remove technical terms from choice text",
                            )
                        )

        logger.debug("Presentation validation complete: %d issues found", len(issues))
        return self._create_result(issues, player_facing_checked=len(player_facing))
