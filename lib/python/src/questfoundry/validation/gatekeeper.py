"""
Gatekeeper Integration.

Runs all quality bar validators and generates gatecheck reports.
Blocks hot->cold promotion on failures.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..models.artifact import Artifact
from .quality_bars import (
    QUALITY_BARS,
    QualityBar,
    QualityBarResult,
    QualityIssue,
    get_quality_bar,
)

logger = logging.getLogger(__name__)


@dataclass
class GatecheckReport:
    """
    Complete gatecheck report.

    Attributes:
        passed: Whether all bars passed (no blockers)
        merge_safe: Whether content can be merged to Cold
        bar_results: Results from each quality bar
        summary: Human-readable summary
        metadata: Additional context
    """

    passed: bool
    merge_safe: bool
    bar_results: dict[str, QualityBarResult] = field(default_factory=dict)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_issues(self) -> list[tuple[str, QualityIssue]]:
        """Get all issues across all bars."""
        issues: list[tuple[str, QualityIssue]] = []
        for bar_name, result in self.bar_results.items():
            for issue in result.issues:
                issues.append((bar_name, issue))
        return issues

    @property
    def blockers(self) -> list[tuple[str, QualityIssue]]:
        """Get all blocker issues."""
        return [
            (bar, issue)
            for bar, issue in self.all_issues
            if issue.severity == "blocker"
        ]

    @property
    def warnings(self) -> list[tuple[str, QualityIssue]]:
        """Get all warning issues."""
        return [
            (bar, issue)
            for bar, issue in self.all_issues
            if issue.severity == "warning"
        ]

    def to_artifact(self) -> Artifact:
        """Convert report to gatecheck_report artifact."""
        return Artifact(
            type="gatecheck_report",
            data={
                "id": f"gatecheck_{self.metadata.get('timestamp', 'unknown')}",
                "passed": self.passed,
                "merge_safe": self.merge_safe,
                "summary": self.summary,
                "bars": {
                    name: {
                        "passed": result.passed,
                        "issues": [
                            {
                                "severity": issue.severity,
                                "message": issue.message,
                                "location": issue.location,
                                "fix": issue.fix,
                            }
                            for issue in result.issues
                        ],
                        "metadata": result.metadata,
                    }
                    for name, result in self.bar_results.items()
                },
                "blockers_count": len(self.blockers),
                "warnings_count": len(self.warnings),
            },
            metadata=self.metadata,
        )


class Gatekeeper:
    """
    Gatekeeper: Runs quality bar validation and generates reports.

    Integrates all quality bars:
    - Integrity: References resolve, no dead ends
    - Reachability: Keystones reachable
    - Style: Voice/register consistent
    - Gateways: Conditions diegetic
    - Nonlinearity: Hubs/loops meaningful
    - Determinism: Assets reproducible
    - Presentation: No spoilers/internals on player surfaces
    - Spoiler Hygiene: PN boundaries maintained
    - Canon Conflict: No contradictions in imported canon (Layer 6/7)
    - Timeline Chronology: Anchors properly ordered (Layer 6/7)
    - Entity Reference: Entity registry consistency (Layer 6/7)
    """

    def __init__(
        self,
        bars: list[str] | None = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize Gatekeeper.

        Args:
            bars: List of bar names to run (default: all 8)
            strict: If True, warnings also block merge (default: True)
        """
        logger.debug("Initializing Gatekeeper with strict=%s", strict)
        self.strict = strict

        # Default to all bars
        if bars is None:
            bars = list(QUALITY_BARS.keys())

        logger.trace("Configured quality bars: %s", bars)

        # Initialize quality bar instances
        self.bars: dict[str, QualityBar] = {}
        for bar_name in bars:
            bar_class = get_quality_bar(bar_name)
            self.bars[bar_name] = bar_class()

        logger.debug("Gatekeeper initialized with %d quality bars", len(self.bars))

    def run_gatecheck(
        self, artifacts: list[Artifact], **metadata: Any
    ) -> GatecheckReport:
        """
        Run full gatecheck on artifacts.

        Args:
            artifacts: List of artifacts to validate
            **metadata: Additional metadata to include in report

        Returns:
            GatecheckReport with results from all bars
        """
        logger.info("Starting gatecheck on %d artifacts", len(artifacts))
        logger.trace("Gatecheck metadata: %s", metadata)

        bar_results: dict[str, QualityBarResult] = {}

        # Run each quality bar
        for bar_name, bar in self.bars.items():
            logger.debug("Running quality bar: %s", bar_name)
            result = bar.validate(artifacts)
            bar_results[bar_name] = result

            if result.passed:
                logger.debug("Quality bar %s PASSED", bar_name)
            else:
                logger.warning(
                    "Quality bar %s FAILED with %d issues", bar_name, len(result.issues)
                )
                for issue in result.issues:
                    logger.trace(
                        "  Issue [%s] %s at %s",
                        issue.severity,
                        issue.message,
                        issue.location,
                    )

        # Determine overall pass/fail
        has_blockers = any(not result.passed for result in bar_results.values())

        # In strict mode, warnings also block
        has_warnings = any(len(result.warnings) > 0 for result in bar_results.values())

        passed = not has_blockers and (not self.strict or not has_warnings)
        merge_safe = not has_blockers  # Blockers always block merge

        # Generate summary
        summary = self._generate_summary(bar_results, passed)

        if passed:
            logger.info("Gatecheck PASSED - All quality bars passed")
        else:
            logger.warning(
                "Gatecheck FAILED - %d blocker(s), %d warning(s)",
                sum(len(r.blockers) for r in bar_results.values()),
                sum(len(r.warnings) for r in bar_results.values()),
            )

        return GatecheckReport(
            passed=passed,
            merge_safe=merge_safe,
            bar_results=bar_results,
            summary=summary,
            metadata=metadata,
        )

    def run_bar(self, bar_name: str, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Run a single quality bar.

        Args:
            bar_name: Name of the bar to run
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult

        Raises:
            ValueError: If bar_name not in configured bars
        """
        logger.debug(
            "Running single quality bar: %s on %d artifacts", bar_name, len(artifacts)
        )

        if bar_name not in self.bars:
            logger.error(
                "Quality bar '%s' not configured. Available: %s",
                bar_name,
                list(self.bars.keys()),
            )
            raise ValueError(
                f"Bar '{bar_name}' not configured. Available: {list(self.bars.keys())}"
            )

        result = self.bars[bar_name].validate(artifacts)
        logger.debug(
            "Quality bar %s result: passed=%s, issues=%d",
            bar_name,
            result.passed,
            len(result.issues),
        )

        return result

    def _generate_summary(
        self, bar_results: dict[str, QualityBarResult], passed: bool
    ) -> str:
        """Generate human-readable summary."""
        lines = []

        if passed:
            lines.append("✓ Gatecheck PASSED - Content safe to merge to Cold")
        else:
            lines.append("✗ Gatecheck FAILED - Blockers must be resolved")

        lines.append("")
        lines.append("Quality Bar Results:")

        for bar_name, result in bar_results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            blocker_count = len(result.blockers)
            warning_count = len(result.warnings)

            line = f"  {status} {bar_name}"
            if blocker_count or warning_count:
                counts = []
                if blocker_count:
                    counts.append(f"{blocker_count} blocker(s)")
                if warning_count:
                    counts.append(f"{warning_count} warning(s)")
                line += f" - {', '.join(counts)}"

            lines.append(line)

        # Add blocker details if any
        all_blockers = [
            (bar, issue)
            for bar, result in bar_results.items()
            for issue in result.blockers
        ]

        if all_blockers:
            lines.append("")
            lines.append("Blockers:")
            for bar_name, issue in all_blockers:
                lines.append(f"  [{bar_name}] {issue.message}")
                lines.append(f"    Location: {issue.location}")
                if issue.fix:
                    lines.append(f"    Fix: {issue.fix}")

        return "\n".join(lines)
