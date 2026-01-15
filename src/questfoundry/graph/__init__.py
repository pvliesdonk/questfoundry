"""Graph package - unified story graph storage.

This package provides the runtime implementation of the unified story graph.
The graph stores all story state as nodes and edges, with stages applying
mutations through the runtime.

See docs/architecture/graph-storage.md for architecture details.
"""

from questfoundry.graph.context import format_valid_ids_context
from questfoundry.graph.errors import (
    EdgeEndpointError,
    GraphCorruptionError,
    GraphIntegrityError,
    NodeExistsError,
    NodeNotFoundError,
    NodeReferencedError,
)
from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    BrainstormValidationError,
    MutationError,
    SeedMutationError,
    SeedValidationError,
    apply_brainstorm_mutations,
    apply_dream_mutations,
    apply_mutations,
    apply_seed_mutations,
    has_mutation_handler,
    validate_brainstorm_mutations,
    validate_seed_mutations,
)
from questfoundry.graph.snapshots import (
    list_snapshots,
    rollback_to_snapshot,
    save_snapshot,
)

__all__ = [
    "BrainstormMutationError",
    "BrainstormValidationError",
    "EdgeEndpointError",
    "Graph",
    "GraphCorruptionError",
    "GraphIntegrityError",
    "MutationError",
    "NodeExistsError",
    "NodeNotFoundError",
    "NodeReferencedError",
    "SeedMutationError",
    "SeedValidationError",
    "apply_brainstorm_mutations",
    "apply_dream_mutations",
    "apply_mutations",
    "apply_seed_mutations",
    "format_valid_ids_context",
    "has_mutation_handler",
    "list_snapshots",
    "rollback_to_snapshot",
    "save_snapshot",
    "validate_brainstorm_mutations",
    "validate_seed_mutations",
]
