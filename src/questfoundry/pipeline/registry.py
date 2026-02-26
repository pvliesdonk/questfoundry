"""Shared phase registry for multi-phase pipeline stages.

Provides ``PhaseRegistry`` and ``PhaseMeta`` used by both GROW and POLISH
stages. Each stage creates its own registry instance and decorator.

Usage (in a stage's registry module)::

    from questfoundry.pipeline.registry import PhaseRegistry, PhaseMeta, PHASE_META_ATTR

    _registry = PhaseRegistry()

    def my_stage_phase(name, *, depends_on=None, is_deterministic=False, priority=None):
        resolved_priority = priority if priority is not None else len(_registry)
        def decorator(fn):
            meta = PhaseMeta(name=name, depends_on=tuple(depends_on or []),
                             is_deterministic=is_deterministic, priority=resolved_priority)
            _registry.register(fn, meta)
            setattr(fn, PHASE_META_ATTR, meta)
            return fn
        return decorator
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class PhaseMeta:
    """Metadata attached to a registered phase function."""

    name: str
    depends_on: tuple[str, ...]
    is_deterministic: bool
    priority: int


PHASE_META_ATTR = "_phase_meta"


class PhaseRegistry:
    """Collects phase-decorated functions and validates their DAG.

    The registry is populated at module import time. Call ``validate()`` to
    check for missing dependencies, cycles, or duplicates. Call
    ``execution_order()`` to get a stable topological ordering.
    """

    def __init__(self) -> None:
        self._phases: dict[str, PhaseMeta] = {}
        self._functions: dict[str, Callable[..., Any]] = {}

    # -- Registration ----------------------------------------------------------

    def register(self, fn: Callable[..., Any], meta: PhaseMeta) -> None:
        """Register a phase function with its metadata.

        Raises:
            ValueError: If a phase with the same name is already registered.
        """
        if meta.name in self._phases:
            msg = (
                f"Duplicate phase name {meta.name!r}: "
                f"already registered by {self._functions[meta.name].__qualname__}"
            )
            raise ValueError(msg)
        self._phases[meta.name] = meta
        self._functions[meta.name] = fn

    # -- Validation ------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the dependency DAG.

        Returns:
            List of error strings. Empty means valid.
        """
        errors: list[str] = []

        # Check for missing dependencies
        for meta in self._phases.values():
            for dep in meta.depends_on:
                if dep not in self._phases:
                    errors.append(
                        f"Phase {meta.name!r} depends on {dep!r}, which is not registered"
                    )

        # Check for cycles using Kahn's algorithm
        if not errors:
            in_degree: dict[str, int] = dict.fromkeys(self._phases, 0)
            for meta in self._phases.values():
                for _dep in meta.depends_on:
                    in_degree[meta.name] += 1

            queue = [name for name, deg in in_degree.items() if deg == 0]
            visited = 0
            adj: dict[str, list[str]] = defaultdict(list)
            for meta in self._phases.values():
                for dep in meta.depends_on:
                    adj[dep].append(meta.name)

            temp_queue = list(queue)
            while temp_queue:
                node = temp_queue.pop()
                visited += 1
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        temp_queue.append(neighbor)

            if visited != len(self._phases):
                cycle_members = [name for name, deg in in_degree.items() if deg > 0]
                errors.append(
                    f"Dependency cycle detected among: {', '.join(sorted(cycle_members))}"
                )

        return errors

    # -- Execution order -------------------------------------------------------

    def execution_order(self) -> list[str]:
        """Return phase names in stable topological order.

        Uses Kahn's algorithm with a min-heap on ``priority`` for stable
        tiebreaking. Phases with no dependency relationship preserve their
        registration order.

        Raises:
            ValueError: If the DAG is invalid (call ``validate()`` first for
                detailed error messages).
        """
        in_degree: dict[str, int] = dict.fromkeys(self._phases, 0)
        adj: dict[str, list[str]] = defaultdict(list)
        for meta in self._phases.values():
            for dep in meta.depends_on:
                adj[dep].append(meta.name)
                in_degree[meta.name] += 1

        # Min-heap keyed on priority for stable ordering
        heap: list[tuple[int, str]] = []
        for name, deg in in_degree.items():
            if deg == 0:
                heapq.heappush(heap, (self._phases[name].priority, name))

        result: list[str] = []
        while heap:
            _priority, name = heapq.heappop(heap)
            result.append(name)
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, (self._phases[neighbor].priority, neighbor))

        if len(result) != len(self._phases):
            msg = "Dependency cycle detected — call validate() for details"
            raise ValueError(msg)

        return result

    # -- Lookup ----------------------------------------------------------------

    def get_meta(self, name: str) -> PhaseMeta | None:
        """Get metadata for a registered phase, or None."""
        return self._phases.get(name)

    def get_function(self, name: str) -> Callable[..., Any] | None:
        """Get the registered function for a phase, or None."""
        return self._functions.get(name)

    @property
    def phase_names(self) -> list[str]:
        """All registered phase names (insertion order)."""
        return list(self._phases.keys())

    def __len__(self) -> int:
        return len(self._phases)

    def __contains__(self, name: str) -> bool:
        return name in self._phases

    def phase_table(self) -> str:
        """Human-readable table of registered phases.

        Returns a markdown-formatted table with columns:
        Priority | Name | Type | Depends On
        """
        lines = ["| Priority | Name | Type | Depends On |"]
        lines.append("|----------|------|------|------------|")
        for name in self.execution_order():
            meta = self._phases[name]
            phase_type = "deterministic" if meta.is_deterministic else "llm"
            deps = ", ".join(meta.depends_on) if meta.depends_on else "—"
            lines.append(f"| {meta.priority} | {name} | {phase_type} | {deps} |")
        return "\n".join(lines)
