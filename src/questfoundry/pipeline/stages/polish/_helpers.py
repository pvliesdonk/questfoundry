"""Module-level helpers shared across the POLISH stage package."""

from __future__ import annotations

from questfoundry.observability.logging import get_logger

log = get_logger(__name__)


class PolishStageError(ValueError):
    """Error raised when POLISH stage cannot proceed."""

    pass
