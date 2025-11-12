"""Type stubs for conflict detection (Layer 6/7 canon workflows)"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    REJECT: str
    REVISE: str
    DOWNGRADE: str

class ConflictSeverity(Enum):
    """Conflict severity levels."""

    CRITICAL: str
    MAJOR: str
    MINOR: str
    INFO: str

@dataclass
class CanonConflict:
    """A detected conflict between canon and seed ideas."""

    canon_statement: str
    seed_idea: str
    severity: ConflictSeverity
    recommended_resolution: ConflictResolution
    fix_suggestion: str
    canon_source: str

@dataclass
class ConflictReport:
    """Report of all detected conflicts."""

    conflicts: list[CanonConflict]
    canon_source: str
    metadata: dict[str, Any]

    def has_critical_conflicts(self) -> bool: ...
    def to_summary(self) -> dict[str, Any]: ...

class ConflictDetector:
    """Detector for canon import conflicts."""

    CONTRADICTION_KEYWORDS: dict[str, list[str]]

    def __init__(self) -> None: ...
    def detect_conflicts(
        self,
        invariant_canon: list[str],
        seed_ideas: list[str],
        canon_source: str = "unknown",
    ) -> ConflictReport: ...
