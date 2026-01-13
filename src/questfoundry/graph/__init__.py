"""Graph package - unified story graph storage.

This package provides the runtime implementation of the unified story graph.
The graph stores all story state as nodes and edges, with stages applying
mutations through the runtime.

See docs/architecture/graph-storage.md for architecture details.
"""

from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import (
    MutationError,
    apply_brainstorm_mutations,
    apply_dream_mutations,
    apply_mutations,
    apply_seed_mutations,
    has_mutation_handler,
)
from questfoundry.graph.snapshots import (
    list_snapshots,
    rollback_to_snapshot,
    save_snapshot,
)

__all__ = [
    "Graph",
    "MutationError",
    "apply_brainstorm_mutations",
    "apply_dream_mutations",
    "apply_mutations",
    "apply_seed_mutations",
    "has_mutation_handler",
    "list_snapshots",
    "rollback_to_snapshot",
    "save_snapshot",
]
