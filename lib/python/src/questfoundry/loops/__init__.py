"""
Loop system for QuestFoundry.

Loops are workflows that coordinate multiple roles to produce and refine artifacts.
The system uses a two-tier context architecture:
- Registry: Lightweight metadata for loop selection (~90 lines)
- Active Loop: Detailed context for execution (~500 lines)
"""

from .base import Loop, LoopContext, LoopResult, LoopStep
from .registry import LoopMetadata, LoopRegistry

__all__ = [
    "Loop",
    "LoopContext",
    "LoopResult",
    "LoopStep",
    "LoopMetadata",
    "LoopRegistry",
]
