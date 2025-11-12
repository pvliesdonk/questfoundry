"""
Conflict detection for canon import.

Detects conflicts between invariant canon from transfer packages and new
project seed ideas. Provides resolution strategies and escalation paths.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    REJECT = "reject"  # Abandon conflicting seed idea, keep invariant
    REVISE = "revise"  # Modify seed to align with invariant canon
    DOWNGRADE = "downgrade"  # Change invariant to mutable, allow seed


class ConflictSeverity(Enum):
    """Severity of detected conflicts."""

    CRITICAL = "critical"  # Direct contradiction requiring resolution
    MAJOR = "major"  # Significant inconsistency
    MINOR = "minor"  # Minor inconsistency, may be resolvable
    INFO = "info"  # Informational, no action required


@dataclass
class CanonConflict:
    """
    A detected conflict between invariant canon and seed ideas.

    Attributes:
        invariant_canon: The invariant canon rule being violated
        seed_idea: The conflicting seed idea
        severity: Conflict severity level
        explanation: Human-readable explanation of the conflict
        suggested_resolution: Recommended resolution strategy
        canon_source: Source of the invariant canon
        metadata: Additional conflict data
    """

    invariant_canon: str
    seed_idea: str
    severity: ConflictSeverity
    explanation: str
    suggested_resolution: ConflictResolution
    canon_source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert conflict to dictionary."""
        return {
            "invariant_canon": self.invariant_canon,
            "seed_idea": self.seed_idea,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "suggested_resolution": self.suggested_resolution.value,
            "canon_source": self.canon_source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonConflict":
        """Create conflict from dictionary."""
        return cls(
            invariant_canon=data["invariant_canon"],
            seed_idea=data["seed_idea"],
            severity=ConflictSeverity(data["severity"]),
            explanation=data["explanation"],
            suggested_resolution=ConflictResolution(data["suggested_resolution"]),
            canon_source=data["canon_source"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConflictReport:
    """
    Report of all detected conflicts.

    Attributes:
        conflicts: List of detected conflicts
        total_invariants: Total number of invariant canon rules checked
        total_seeds: Total number of seed ideas checked
        critical_count: Number of critical conflicts
        major_count: Number of major conflicts
        minor_count: Number of minor conflicts
        requires_escalation: Whether conflicts require Showrunner decision
    """

    conflicts: list[CanonConflict] = field(default_factory=list)
    total_invariants: int = 0
    total_seeds: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    requires_escalation: bool = False

    def add_conflict(self, conflict: CanonConflict) -> None:
        """Add conflict to report and update counts."""
        self.conflicts.append(conflict)

        if conflict.severity == ConflictSeverity.CRITICAL:
            self.critical_count += 1
            self.requires_escalation = True
        elif conflict.severity == ConflictSeverity.MAJOR:
            self.major_count += 1
            self.requires_escalation = True
        elif conflict.severity == ConflictSeverity.MINOR:
            self.minor_count += 1

    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    def get_critical_conflicts(self) -> list[CanonConflict]:
        """Get critical conflicts only."""
        return [c for c in self.conflicts if c.severity == ConflictSeverity.CRITICAL]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "conflicts": [c.to_dict() for c in self.conflicts],
            "total_invariants": self.total_invariants,
            "total_seeds": self.total_seeds,
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "requires_escalation": self.requires_escalation,
        }


class ConflictDetector:
    """
    Detector for canon import conflicts.

    Analyzes invariant canon rules against new project seed ideas to identify
    contradictions. Uses keyword matching and semantic analysis to detect conflicts.

    Example:
        >>> detector = ConflictDetector()
        >>> # Invariant canon from transfer package
        >>> invariants = [
        ...     "Wormhole to Alpha Centauri collapsed in 2203 AD",
        ...     "FTL travel requires antimatter fuel",
        ...     "Mars colony established 2147 AD"
        ... ]
        >>> # Seed ideas for new project
        >>> seeds = [
        ...     "Story about repairing the wormhole to Alpha Centauri",
        ...     "Hero discovers faster-than-light travel without fuel",
        ...     "Set in Mars colony in 2100"
        ... ]
        >>> report = detector.detect_conflicts(invariants, seeds, "star-colony-1")
        >>> if report.has_conflicts():
        ...     for conflict in report.get_critical_conflicts():
        ...         print(f"Conflict: {conflict.explanation}")
    """

    # Keywords indicating contradictory actions
    CONTRADICTION_KEYWORDS = {
        "repair": ["collapsed", "destroyed", "broken", "damaged"],
        "restore": ["lost", "gone", "destroyed", "extinct", "collapsed"],
        "revive": ["dead", "extinct", "destroyed"],
        "resurrect": ["dead", "extinct"],
        "destroy": ["indestructible", "eternal", "permanent", "unbreakable"],
        "destroys": ["indestructible", "eternal", "permanent", "unbreakable"],
        "discover": ["known", "established", "exists"],
        "invent": ["exists", "discovered", "established"],
        "create": ["exists", "already", "established"],
        "founding": ["established", "founded", "created"],
        "travel": ["does not exist", "impossible", "cannot"],
        "travels": ["does not exist", "impossible", "cannot"],
    }

    # Keywords indicating temporal conflicts
    TEMPORAL_KEYWORDS = ["before", "after", "during", "established", "founded", "in"]

    def detect_conflicts(
        self,
        invariant_canon: list[str],
        seed_ideas: list[str],
        canon_source: str = "unknown",
    ) -> ConflictReport:
        """
        Detect conflicts between invariant canon and seed ideas.

        Args:
            invariant_canon: List of invariant canon rules
            seed_ideas: List of new project seed ideas
            canon_source: Source of the invariant canon

        Returns:
            ConflictReport with all detected conflicts
        """
        report = ConflictReport(
            total_invariants=len(invariant_canon),
            total_seeds=len(seed_ideas),
        )

        for seed in seed_ideas:
            for canon_rule in invariant_canon:
                conflict = self._check_pair(seed, canon_rule, canon_source)
                if conflict:
                    report.add_conflict(conflict)

        return report

    def _check_pair(
        self, seed: str, canon_rule: str, canon_source: str
    ) -> CanonConflict | None:
        """
        Check a single seed/canon pair for conflicts.

        Args:
            seed: Seed idea to check
            canon_rule: Canon rule to check against
            canon_source: Source of canon rule

        Returns:
            CanonConflict if detected, None otherwise
        """
        seed_lower = seed.lower()
        canon_lower = canon_rule.lower()

        # Check for direct contradictions
        for action, contradictory_states in self.CONTRADICTION_KEYWORDS.items():
            if action in seed_lower:
                for state in contradictory_states:
                    if state in canon_lower:
                        return CanonConflict(
                            invariant_canon=canon_rule,
                            seed_idea=seed,
                            severity=ConflictSeverity.CRITICAL,
                            explanation=(
                                f"Seed idea wants to {action} something that "
                                f"canon establishes as {state}"
                            ),
                            suggested_resolution=ConflictResolution.REJECT,
                            canon_source=canon_source,
                        )

        # Check for temporal conflicts
        if self._has_temporal_conflict(seed_lower, canon_lower):
            return CanonConflict(
                invariant_canon=canon_rule,
                seed_idea=seed,
                severity=ConflictSeverity.MAJOR,
                explanation="Seed idea has temporal inconsistency with canon timeline",
                suggested_resolution=ConflictResolution.REVISE,
                canon_source=canon_source,
            )

        # Check for entity conflicts (same entity, different attributes)
        entity_conflict = self._check_entity_conflict(seed_lower, canon_lower)
        if entity_conflict:
            return CanonConflict(
                invariant_canon=canon_rule,
                seed_idea=seed,
                severity=ConflictSeverity.MAJOR,
                explanation=entity_conflict,
                suggested_resolution=ConflictResolution.REVISE,
                canon_source=canon_source,
            )

        return None

    def _has_temporal_conflict(self, seed: str, canon: str) -> bool:
        """
        Check for temporal conflicts between seed and canon.

        Args:
            seed: Seed idea (lowercase)
            canon: Canon rule (lowercase)

        Returns:
            True if temporal conflict detected
        """
        # Extract years from both strings
        import re

        seed_years = re.findall(r"\b(1\d{3}|2\d{3})\b", seed)
        canon_years = re.findall(r"\b(1\d{3}|2\d{3})\b", canon)

        if seed_years and canon_years:
            # Check if seed has earlier year than canon for same event
            for keyword in self.TEMPORAL_KEYWORDS:
                if keyword in seed and keyword in canon:
                    seed_year = int(seed_years[0])
                    canon_year = int(canon_years[0])
                    if abs(seed_year - canon_year) > 5:  # More than 5 years difference
                        return True

        return False

    def _check_entity_conflict(self, seed: str, canon: str) -> str | None:
        """
        Check for entity-level conflicts.

        Args:
            seed: Seed idea (lowercase)
            canon: Canon rule (lowercase)

        Returns:
            Conflict explanation if detected, None otherwise
        """
        # Extract potential entity names (capitalized words)
        import re

        # Look for shared proper nouns
        seed_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", seed))
        canon_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", canon))

        shared_entities = seed_entities & canon_entities

        if shared_entities:
            # Check for contradictory attributes
            contradictory_pairs = [
                ("alive", "dead"),
                ("active", "destroyed"),
                ("functioning", "broken"),
                ("inhabited", "abandoned"),
            ]

            for entity in shared_entities:
                for pos, neg in contradictory_pairs:
                    if pos in seed and neg in canon:
                        return (
                            f"Entity '{entity}' has contradictory state "
                            "in seed vs canon"
                        )
                    if neg in seed and pos in canon:
                        return (
                            f"Entity '{entity}' has contradictory state "
                            "in seed vs canon"
                        )

        return None

    def suggest_revision(self, conflict: CanonConflict) -> dict[str, str]:
        """
        Suggest a revision to resolve conflict.

        Args:
            conflict: Conflict to resolve

        Returns:
            Dictionary with original and revised seed idea
        """
        seed = conflict.seed_idea
        canon = conflict.invariant_canon

        # Generate revision suggestion based on conflict type
        if conflict.suggested_resolution == ConflictResolution.REJECT:
            revised = f"[REJECTED: Conflicts with canon: {canon}]"
        elif conflict.suggested_resolution == ConflictResolution.REVISE:
            # Try to suggest a compatible version
            revised = f"{seed} [Note: Must align with canon: {canon}]"
        else:  # DOWNGRADE
            revised = seed  # Keep as-is but downgrade canon

        return {
            "original": seed,
            "revised": revised,
            "strategy": conflict.suggested_resolution.value,
        }
