"""Graph invariant assertions for pipeline stages.

Each function raises PipelineInvariantError if an invariant is violated.
These are detective checks called after phases that mutate the graph,
complementing the preventive checks inside insertion helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


class PipelineInvariantError(RuntimeError):
    """Raised when a post-phase graph invariant is violated.

    Unlike validation errors (which are recoverable), invariant violations
    indicate a bug in phase logic and must stop pipeline execution immediately.
    """


def assert_predecessor_dag_acyclic(graph: Graph, phase_name: str) -> None:
    """Raise PipelineInvariantError if the predecessor DAG contains a cycle.

    Uses topological_sort_beats() from grow_algorithms.  If the sort produces
    fewer beats than exist in the graph, at least one cycle is present.

    Called after any phase that writes predecessor edges.  Failures here
    indicate a bug in the phase — the error is hard (not a warning) so the
    pipeline stops rather than producing a corrupt graph.

    Args:
        graph: Graph to inspect.
        phase_name: Name of the phase just completed (included in the error
            message so the caller can identify which phase introduced the cycle).

    Raises:
        PipelineInvariantError: If the predecessor DAG contains one or more cycles.
    """
    from questfoundry.graph.grow_algorithms import topological_sort_beats

    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return

    beat_ids = list(beat_nodes.keys())

    try:
        topological_sort_beats(graph, beat_ids)
    except ValueError as exc:
        # topological_sort_beats raises ValueError on cycle detection
        raise PipelineInvariantError(
            f"Cycle detected in predecessor DAG after phase '{phase_name}': {exc}"
        ) from exc

    log.debug("predecessor_dag_acyclic", phase=phase_name, beats=len(beat_ids))
