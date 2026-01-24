"""Graph package - unified story graph storage.

This package provides the runtime implementation of the unified story graph.
The graph stores all story state as nodes and edges, with stages applying
mutations through the runtime.

See docs/architecture/graph-storage.md for architecture details.
"""

from questfoundry.graph.context import (
    SCOPE_ENTITY,
    SCOPE_TENSION,
    SCOPE_THREAD,
    check_structural_completeness,
    format_scoped_id,
    format_summarize_manifest,
    format_valid_ids_context,
    normalize_scoped_id,
    parse_scoped_id,
)
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
    GrowErrorCategory,
    GrowMutationError,
    GrowValidationError,
    MutationError,
    SeedErrorCategory,
    SeedMutationError,
    SeedValidationError,
    apply_brainstorm_mutations,
    apply_dream_mutations,
    apply_mutations,
    apply_seed_mutations,
    categorize_error,
    categorize_errors,
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
    "SCOPE_ENTITY",
    "SCOPE_TENSION",
    "SCOPE_THREAD",
    "BrainstormMutationError",
    "BrainstormValidationError",
    "EdgeEndpointError",
    "Graph",
    "GraphCorruptionError",
    "GraphIntegrityError",
    "GrowErrorCategory",
    "GrowMutationError",
    "GrowValidationError",
    "MutationError",
    "NodeExistsError",
    "NodeNotFoundError",
    "NodeReferencedError",
    "SeedErrorCategory",
    "SeedMutationError",
    "SeedValidationError",
    "apply_brainstorm_mutations",
    "apply_dream_mutations",
    "apply_mutations",
    "apply_seed_mutations",
    "categorize_error",
    "categorize_errors",
    "check_structural_completeness",
    "format_scoped_id",
    "format_summarize_manifest",
    "format_valid_ids_context",
    "has_mutation_handler",
    "list_snapshots",
    "normalize_scoped_id",
    "parse_scoped_id",
    "rollback_to_snapshot",
    "save_snapshot",
    "validate_brainstorm_mutations",
    "validate_seed_mutations",
]
