"""Shared validation types used by GROW and POLISH validation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ValidationCheck:
    """Result of a single validation check.

    Attributes:
        name: Identifier for the check.
        severity: "pass", "warn", or "fail".
        message: Human-readable description of the result.
    """

    name: str
    severity: Literal["pass", "warn", "fail"]
    message: str = ""


@dataclass
class ValidationReport:
    """Aggregated results of validation checks.

    Attributes:
        checks: List of individual validation check results.
    """

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """True if any check has severity 'fail'."""
        return any(c.severity == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """True if any check has severity 'warn'."""
        return any(c.severity == "warn" for c in self.checks)

    @property
    def summary(self) -> str:
        """Human-readable summary of all checks."""
        fails = [c for c in self.checks if c.severity == "fail"]
        warns = [c for c in self.checks if c.severity == "warn"]
        passes = [c for c in self.checks if c.severity == "pass"]

        parts: list[str] = []
        if fails:
            parts.append(f"{len(fails)} failed")
        if warns:
            parts.append(f"{len(warns)} warnings")
        if passes:
            parts.append(f"{len(passes)} passed")
        return ", ".join(parts)
