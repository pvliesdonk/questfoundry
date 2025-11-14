"""Core types for the spec compiler."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CompilationError(Exception):
    """Raised when compilation fails."""

    pass


@dataclass
class BehaviorPrimitive:
    """Base class for atomic behavior components."""

    id: str
    type: str  # 'expertise', 'procedure', 'snippet', 'playbook', 'adapter'
    content: str
    metadata: dict[str, Any]
    references: dict[str, list[str]] = field(default_factory=dict)
    source_path: Path | None = None
