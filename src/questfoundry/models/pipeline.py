"""Shared pipeline models used across multiple stages.

These models provide common types for stage phase execution,
allowing infrastructure like PhaseGateHook to be reused by
any stage with a multi-phase algorithm (GROW, FILL, etc.).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhaseResult(BaseModel):
    """Result of a single phase execution within a stage.

    Used by any stage with a multi-phase algorithm (GROW, FILL).
    Captures phase name, completion status, and resource usage.
    """

    phase: str = Field(min_length=1)
    status: Literal["completed", "skipped", "failed"]
    detail: str = ""
    llm_calls: int = 0
    tokens_used: int = 0
