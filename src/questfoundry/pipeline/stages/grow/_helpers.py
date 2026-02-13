"""Module-level helpers shared across the GROW stage package."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from questfoundry.graph.mutations import GrowValidationError  # noqa: TC001 - used at runtime
from questfoundry.observability.logging import get_logger

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)


class GrowStageError(ValueError):
    """Error raised when GROW stage cannot proceed."""

    pass


def _get_prompts_path() -> Path:
    """Get the prompts directory path.

    Returns prompts from package first, then falls back to project root.
    """
    pkg_path = Path(__file__).parents[5] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


def _format_structural_feedback(errors: list[GrowValidationError]) -> str:
    """Format structural compatibility errors as targeted LLM feedback.

    Categorizes errors by type and produces a correction message that
    tells the LLM exactly what went wrong, enabling targeted repair.
    """
    dilemma_substrings = ("same dilemma", "span only", "at most 1 beat per dilemma")
    same_dilemma_count = sum(
        1 for e in errors if any(sub in e.issue.lower() for sub in dilemma_substrings)
    )

    lines = ["\n\n## CORRECTION REQUIRED"]
    lines.append(f"Your previous {len(errors)} error(s) caused ALL intersections to be REJECTED.")

    if same_dilemma_count > 0:
        lines.append(
            "PROBLEM: You grouped beats from the SAME dilemma together. "
            "Each intersection MUST combine beats from DIFFERENT dilemmas. "
            "Each candidate group already contains beats from different dilemmas -- "
            "select beats from WITHIN a group, not across groups."
        )

    # Show up to 3 specific errors for clarity
    max_shown = 3
    for err in errors[:max_shown]:
        lines.append(f"  - {err.issue}")
    if len(errors) > max_shown:
        lines.append(f"  ... and {len(errors) - max_shown} more error(s)")

    lines.append(
        "Try again. Use the beat IDs from WITHIN each candidate group to form intersections."
    )
    return "\n".join(lines)
