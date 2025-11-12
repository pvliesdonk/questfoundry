"""Quality bar validators for gatekeeper checks."""

from .base import QualityBar, QualityBarResult, QualityIssue
from .canon import CanonConflictBar, EntityReferenceBar, TimelineChronologyBar
from .determinism import DeterminismBar
from .gateways import GatewaysBar
from .integrity import IntegrityBar
from .nonlinearity import NonlinearityBar
from .presentation import PresentationBar
from .reachability import ReachabilityBar
from .spoiler_hygiene import SpoilerHygieneBar
from .style import StyleBar

__all__ = [
    "QualityBar",
    "QualityBarResult",
    "QualityIssue",
    "IntegrityBar",
    "ReachabilityBar",
    "StyleBar",
    "GatewaysBar",
    "NonlinearityBar",
    "DeterminismBar",
    "PresentationBar",
    "SpoilerHygieneBar",
    "CanonConflictBar",
    "TimelineChronologyBar",
    "EntityReferenceBar",
]

# Registry of all quality bars
QUALITY_BARS = {
    "integrity": IntegrityBar,
    "reachability": ReachabilityBar,
    "style": StyleBar,
    "gateways": GatewaysBar,
    "nonlinearity": NonlinearityBar,
    "determinism": DeterminismBar,
    "presentation": PresentationBar,
    "spoiler_hygiene": SpoilerHygieneBar,
    # Canon workflow quality bars (Layer 6/7)
    "canon_conflict": CanonConflictBar,
    "timeline_chronology": TimelineChronologyBar,
    "entity_reference": EntityReferenceBar,
}


def get_quality_bar(name: str) -> type[QualityBar]:
    """Get a quality bar class by name."""
    if name not in QUALITY_BARS:
        raise ValueError(f"Unknown quality bar: {name}")
    return QUALITY_BARS[name]
