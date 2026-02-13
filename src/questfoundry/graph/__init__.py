"""Graph package - unified story graph storage.

This package provides the runtime implementation of the unified story graph.
The graph stores all story state as nodes and edges, with stages applying
mutations through the runtime.

See docs/architecture/graph-storage.md for architecture details.
"""

from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    SCOPE_DILEMMA,
    SCOPE_PATH,
    check_structural_completeness,
    format_entity_id,
    format_hierarchical_path_id,
    format_scoped_id,
    format_summarize_manifest,
    format_valid_ids_context,
    is_entity_id,
    normalize_scoped_id,
    parse_entity_id,
    parse_hierarchical_path_id,
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
from questfoundry.graph.migration import migrate_json_to_sqlite
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
from questfoundry.graph.store import DictGraphStore, GraphStore

__all__ = [
    "ENTITY_CATEGORIES",
    "SCOPE_DILEMMA",
    "SCOPE_PATH",
    "BrainstormMutationError",
    "BrainstormValidationError",
    "DictGraphStore",
    "EdgeEndpointError",
    "Graph",
    "GraphCorruptionError",
    "GraphIntegrityError",
    "GraphStore",
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
    "SqliteGraphStore",
    "apply_brainstorm_mutations",
    "apply_dream_mutations",
    "apply_mutations",
    "apply_seed_mutations",
    "categorize_error",
    "categorize_errors",
    "check_structural_completeness",
    "format_entity_id",
    "format_hierarchical_path_id",
    "format_scoped_id",
    "format_summarize_manifest",
    "format_valid_ids_context",
    "has_mutation_handler",
    "is_entity_id",
    "list_snapshots",
    "migrate_json_to_sqlite",
    "normalize_scoped_id",
    "parse_entity_id",
    "parse_hierarchical_path_id",
    "parse_scoped_id",
    "rollback_to_snapshot",
    "save_snapshot",
    "validate_brainstorm_mutations",
    "validate_seed_mutations",
]
