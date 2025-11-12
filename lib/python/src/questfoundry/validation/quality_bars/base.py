"""
Base classes for quality bar validators.

Quality bars are validation checks that must pass before content can be
merged from Hot to Cold (Cold = ship-ready content). Each bar focuses on
a specific quality aspect (integrity, reachability, style, etc.).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...models.artifact import Artifact

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """
    An issue found during quality bar validation.

    Attributes:
        severity: 'blocker' (must fix), 'warning' (should review), or 'info' (FYI)
        message: Human-readable description
        location: Where the issue was found (artifact_id, field, section, etc.)
        fix: Suggested remediation
    """

    severity: str  # 'blocker', 'warning', or 'info'
    message: str
    location: str
    fix: str = ""


@dataclass
class QualityBarResult:
    """
    Result of a quality bar check.

    Attributes:
        bar_name: Name of the quality bar
        passed: Whether the bar passed (no blockers)
        issues: List of issues found
        metadata: Additional context about the check
    """

    bar_name: str
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blockers(self) -> list[QualityIssue]:
        """Get only blocker issues."""
        return [i for i in self.issues if i.severity == "blocker"]

    @property
    def warnings(self) -> list[QualityIssue]:
        """Get only warning issues."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def info(self) -> list[QualityIssue]:
        """Get only info issues."""
        return [i for i in self.issues if i.severity == "info"]


class QualityBar(ABC):
    """
    Base class for quality bar validators.

    Each quality bar implements a specific validation check that content
    must pass before merging to Cold (ship-ready state).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this quality bar."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this bar validates."""
        pass

    @abstractmethod
    def validate(self, artifacts: list[Artifact]) -> QualityBarResult:
        """
        Validate artifacts against this quality bar.

        Args:
            artifacts: List of artifacts to validate

        Returns:
            QualityBarResult with pass/fail and issues found
        """
        pass

    def _create_result(
        self, issues: list[QualityIssue], **metadata: Any
    ) -> QualityBarResult:
        """
        Helper to create a QualityBarResult.

        Args:
            issues: List of issues found
            **metadata: Additional metadata to include

        Returns:
            QualityBarResult
        """
        blockers = [i for i in issues if i.severity == "blocker"]
        warnings = [i for i in issues if i.severity == "warning"]
        passed = len(blockers) == 0

        logger.trace(
            (
                "Quality bar %s result: passed=%s, blockers=%d, "
                "warnings=%d, total_issues=%d"
            ),
            self.name,
            passed,
            len(blockers),
            len(warnings),
            len(issues),
        )

        return QualityBarResult(
            bar_name=self.name,
            passed=passed,
            issues=issues,
            metadata=metadata,
        )
