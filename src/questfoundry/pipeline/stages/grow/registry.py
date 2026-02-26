"""Phase registry with decorator-based dependency validation for GROW.

Phases register via the ``@grow_phase`` decorator at import time. The registry
validates the dependency DAG and produces a stable topological execution order.

Usage::

    @grow_phase(name="validate_dag", is_deterministic=True)
    async def phase_validate_dag(graph, model):
        ...

    @grow_phase(name="passages", depends_on=["collapse_linear_beats"], is_deterministic=True)
    async def phase_passages(graph, model):
        ...

The execution order is produced by ``get_registry().execution_order()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.pipeline.registry import PHASE_META_ATTR, PhaseMeta, PhaseRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


# Re-export for backward compatibility
__all__ = ["PHASE_META_ATTR", "PhaseMeta", "PhaseRegistry", "get_registry", "grow_phase"]


# -- Module-level singleton ---------------------------------------------------

_registry = PhaseRegistry()


def get_registry() -> PhaseRegistry:
    """Return the module-level phase registry singleton."""
    return _registry


def grow_phase(
    name: str,
    *,
    depends_on: list[str] | None = None,
    is_deterministic: bool = False,
    priority: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a GROW phase function.

    Args:
        name: Unique phase name (used in execution order and logging).
        depends_on: Phase names that must run before this one.
        is_deterministic: True for phases that don't call an LLM.
        priority: Tiebreaker for topological sort. Defaults to registration
            order (auto-incremented).
    """
    resolved_priority = priority if priority is not None else len(_registry)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        meta = PhaseMeta(
            name=name,
            depends_on=tuple(depends_on or []),
            is_deterministic=is_deterministic,
            priority=resolved_priority,
        )
        _registry.register(fn, meta)
        setattr(fn, PHASE_META_ATTR, meta)
        return fn

    return decorator
