"""
Style Quality Bar.

Validates that:
- Voice and register are consistent
- Motifs are used consistently
- Tone matches style guidelines
- Visual/aesthetic cues align with style
"""

import logging
import re
from collections import Counter

from ...models.artifact import Artifact
from .base import QualityBar, QualityBarResult, QualityIssue

logger = logging.getLogger(__name__)


class StyleBar(QualityBar):
    """
    Style Bar: Voice, register, motifs hold across content.

    Checks:
    - Consistent voice/register
    - Motif usage patterns
    - Tone consistency
    - Style guide compliance (when present)
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this quality bar."""
        return "style"

    @property
    def description(self) -> str:
        """Return a human-readable description of what this bar validates."""
        return "Voice, register, motifs, and visual guardrails hold"

    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts for style consistency.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult
        """
        logger.debug("Validating style in %d artifacts", len(artifacts))
        issues: list[QualityIssue] = []

        # Extract style guide if present
        style_guide = self._find_style_guide(artifacts)
        if style_guide:
            logger.debug("Style guide found: %s", style_guide.data.get("id", "unknown"))

        # Check manuscript sections for style
        sections = [a for a in artifacts if a.type == "manuscript_section"]

        logger.trace("Found %d manuscript sections for style validation", len(sections))

        if sections:
            # Check voice consistency
            issues.extend(self._check_voice_consistency(sections))

            # Check motif usage if style guide present
            if style_guide:
                logger.debug("Checking motif usage against style guide")
                issues.extend(self._check_motifs(sections, style_guide))

        # Check style artifacts
        style_artifacts = [a for a in artifacts if a.type == "style_guide"]
        for artifact in style_artifacts:
            logger.debug(
                "Validating style guide artifact: %s",
                artifact.data.get("id", "unknown"),
            )
            issues.extend(self._validate_style_guide(artifact))

        logger.debug("Style validation complete: %d issues found", len(issues))
        return self._create_result(
            issues,
            sections_checked=len(sections),
            has_style_guide=style_guide is not None,
        )

    def _find_style_guide(self, artifacts: list[Artifact]) -> Artifact | None:
        """Find style guide artifact if present."""
        for artifact in artifacts:
            if artifact.type == "style_guide":
                return artifact
        return None

    def _check_voice_consistency(self, sections: list[Artifact]) -> list[QualityIssue]:
        """Check for voice/register consistency across sections."""
        issues: list[QualityIssue] = []

        # Simple heuristics for voice shifts
        for section in sections:
            text = section.data.get("text", section.data.get("body", ""))
            if not text:
                continue

            section_id = section.data.get("id", "unknown")

            # Check for common voice shift indicators
            # Present tense vs past tense mixing
            has_present = bool(re.search(r"\b(am|is|are|walks|runs|says|see)\b", text))
            has_past = bool(re.search(r"\b(was|were|walked|ran|said)\b", text))

            if has_present and has_past:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        message="Mixed tense usage detected",
                        location=f"section:{section_id}",
                        fix="Use consistent tense (past or present) throughout",
                    )
                )

            # Check for first vs second vs third person mixing
            has_second_person = bool(re.search(r"\byou\b", text, re.I))
            has_first_person = bool(re.search(r"\b(I|me|my|mine)\b", text))

            if has_first_person and has_second_person:
                issues.append(
                    QualityIssue(
                        severity="info",
                        message="Mixed person (I/you) detected - verify intentional",
                        location=f"section:{section_id}",
                        fix=(
                            "Ensure consistent point of view unless "
                            "intentionally switching"
                        ),
                    )
                )

        return issues

    def _check_motifs(
        self, sections: list[Artifact], style_guide: Artifact
    ) -> list[QualityIssue]:
        """Check motif usage against style guide."""
        issues: list[QualityIssue] = []

        # Extract motifs from style guide
        motifs = style_guide.data.get("motifs", [])
        if not motifs:
            return issues

        # Check motif presence across sections
        motif_usage: Counter[str] = Counter()

        for section in sections:
            text = section.data.get("text", section.data.get("body", ""))
            if not text:
                continue

            for motif in motifs:
                if isinstance(motif, str):
                    pattern = motif
                elif isinstance(motif, dict):
                    pattern = str(motif.get("pattern") or motif.get("name") or "")
                else:
                    continue

                if pattern.lower() in text.lower():
                    motif_usage[pattern] += 1

        # Warn if key motifs not used
        for motif in motifs:
            if isinstance(motif, dict):
                name = str(motif.get("name") or motif.get("pattern") or "")
                if motif.get("required") and motif_usage[name] == 0:
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            message=f"Required motif '{name}' not found in any section",
                            location="manuscript",
                            fix=f"Incorporate motif '{name}' or update style guide",
                        )
                    )

        return issues

    def _validate_style_guide(self, artifact: Artifact) -> list[QualityIssue]:
        """Validate style guide artifact structure."""
        issues: list[QualityIssue] = []

        # Check for required style guide fields
        if not artifact.data.get("voice"):
            issues.append(
                QualityIssue(
                    severity="warning",
                    message="Style guide missing 'voice' field",
                    location=f"artifact:{artifact.data.get('id', 'unknown')}",
                    fix="Add voice/tone guidance (e.g., 'sardonic', 'earnest')",
                )
            )

        if not artifact.data.get("register"):
            issues.append(
                QualityIssue(
                    severity="warning",
                    message="Style guide missing 'register' field",
                    location=f"artifact:{artifact.data.get('id', 'unknown')}",
                    fix="Add register guidance (e.g., 'formal', 'casual', 'literary')",
                )
            )

        return issues
