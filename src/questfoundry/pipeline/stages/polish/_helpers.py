"""Module-level helpers shared across the POLISH stage package."""

from __future__ import annotations

from questfoundry.observability.logging import get_logger

log = get_logger(__name__)

# Graph node used to persist Phase 1 rejection warnings before PolishPlan exists.
# Written by llm_phases.py (Phase 1), read by deterministic.py (Phase 4).
_PRE_PLAN_WARNINGS_NODE = "polish_meta::pre_plan_warnings"


class PolishStageError(ValueError):
    """Error raised when POLISH stage cannot proceed."""

    pass
