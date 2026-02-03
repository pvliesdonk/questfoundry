"""Graph integrity error types with LLM-actionable feedback.

These errors are raised when graph operations violate referential integrity,
similar to foreign key constraint violations in databases.

Each error type provides semantic context and can format itself as actionable
feedback for LLM retry loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from questfoundry.graph.context import parse_scoped_id, strip_scope_prefix


class GraphIntegrityError(Exception):
    """Base class for graph integrity violations.

    Subclasses must implement to_llm_feedback() to provide actionable
    error messages for LLM retry loops.
    """

    def to_llm_feedback(self) -> str:
        """Format error as actionable feedback for LLM retry.

        Returns:
            Human-readable error message explaining what's wrong,
            why it's wrong, and how to fix it.
        """
        raise NotImplementedError


@dataclass
class NodeNotFoundError(GraphIntegrityError):
    """Raised when referencing a non-existent node.

    This is the graph equivalent of a foreign key violation - you're
    trying to reference something that doesn't exist.

    Attributes:
        node_id: The ID that was referenced but doesn't exist.
        available: List of valid IDs that could be used instead.
        context: Description of where the reference occurred.
    """

    node_id: str
    available: list[str] = field(default_factory=list)
    context: str = ""

    def __post_init__(self) -> None:
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"Node '{self.node_id}' not found"
        if self.context:
            msg += f" ({self.context})"
        return msg

    def _get_suggestions(self) -> list[str]:
        """Find similar IDs that might be typos."""
        # Extract raw_id from prefixed ID (e.g., "entity::kay" -> "kay")
        raw_id = strip_scope_prefix(self.node_id)
        raw_available = [strip_scope_prefix(a) for a in self.available]
        matches = get_close_matches(raw_id, raw_available, n=3, cutoff=0.6)

        # Reconstruct full IDs with prefix
        prefix, _ = parse_scoped_id(self.node_id)
        if prefix:
            return [f"{prefix}::{m}" for m in matches]
        return matches

    def to_llm_feedback(self) -> str:
        """Format as actionable LLM feedback."""
        suggestions = self._get_suggestions()

        lines = [
            "## Reference Error: Node Not Found",
            "",
            f"**You referenced**: `{self.node_id}`",
        ]

        if self.context:
            lines.append(f"**Context**: {self.context}")

        lines.extend(
            [
                "",
                "**Problem**: This node does not exist in the graph.",
                "",
            ]
        )

        if suggestions:
            lines.append("**Did you mean one of these?**")
            for s in suggestions:
                lines.append(f"  - `{s}`")
            lines.append("")

        if self.available:
            lines.append("**Valid IDs** (use exactly one of these):")
            # Show up to 20 IDs, sorted for consistency
            for a in sorted(self.available)[:20]:
                lines.append(f"  - `{a}`")
            if len(self.available) > 20:
                lines.append(f"  - ... and {len(self.available) - 20} more")

        return "\n".join(lines)


@dataclass
class NodeExistsError(GraphIntegrityError):
    """Raised when creating a node that already exists.

    Use update_node() to modify existing nodes, or use a different ID
    for new nodes.

    Attributes:
        node_id: The ID that already exists.
    """

    node_id: str

    def __post_init__(self) -> None:
        super().__init__(f"Node '{self.node_id}' already exists")

    def to_llm_feedback(self) -> str:
        """Format as actionable LLM feedback."""
        return f"""## Error: Node Already Exists

**You tried to create**: `{self.node_id}`

**Problem**: A node with this ID already exists in the graph.

**Solutions**:
1. Use a different ID if this is meant to be a new node
2. If you want to modify the existing node, use update instead of create
"""


@dataclass
class NodeReferencedError(GraphIntegrityError):
    """Raised when deleting a node that is still referenced by edges.

    Similar to a foreign key constraint preventing deletion of a
    referenced row.

    Attributes:
        node_id: The node that cannot be deleted.
        referenced_by: List of edges that reference this node.
    """

    node_id: str
    referenced_by: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(
            f"Node '{self.node_id}' is referenced by {len(self.referenced_by)} edge(s)"
        )

    def to_llm_feedback(self) -> str:
        """Format as actionable LLM feedback."""
        lines = [
            "## Error: Cannot Delete Referenced Node",
            "",
            f"**Node**: `{self.node_id}`",
            f"**Problem**: This node is referenced by {len(self.referenced_by)} edge(s) and cannot be deleted.",
            "",
            "**Referenced by**:",
        ]

        for ref in self.referenced_by[:5]:
            edge_type = ref.get("type", "unknown")
            from_id = ref.get("from", "?")
            to_id = ref.get("to", "?")
            if from_id == self.node_id:
                lines.append(f"  - `{edge_type}` edge to `{to_id}`")
            else:
                lines.append(f"  - `{edge_type}` edge from `{from_id}`")

        if len(self.referenced_by) > 5:
            lines.append(f"  - ... and {len(self.referenced_by) - 5} more")

        lines.extend(
            [
                "",
                "**Solutions**:",
                "1. Delete or update the referencing edges first",
                "2. Use cascade=True to delete this node and all its edges",
            ]
        )

        return "\n".join(lines)


@dataclass
class EdgeEndpointError(GraphIntegrityError):
    """Raised when an edge references non-existent endpoints.

    Both the source (from) and target (to) nodes must exist before
    an edge can be created between them.

    Attributes:
        edge_type: Type of edge being created.
        from_id: Source node ID.
        to_id: Target node ID.
        missing: Which endpoint is missing ("from", "to", or "both").
        available_from: Valid IDs for the source endpoint.
        available_to: Valid IDs for the target endpoint.
    """

    edge_type: str
    from_id: str
    to_id: str
    missing: str  # "from", "to", or "both"
    available_from: list[str] = field(default_factory=list)
    available_to: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.missing == "both":
            msg = (
                f"Edge '{self.edge_type}' endpoints not found: '{self.from_id}' and '{self.to_id}'"
            )
        elif self.missing == "from":
            msg = f"Edge '{self.edge_type}' source not found: '{self.from_id}'"
        else:
            msg = f"Edge '{self.edge_type}' target not found: '{self.to_id}'"
        super().__init__(msg)

    def to_llm_feedback(self) -> str:
        """Format as actionable LLM feedback."""
        lines = [
            "## Error: Edge Endpoint Not Found",
            "",
            f"**Edge type**: `{self.edge_type}`",
            f"**From**: `{self.from_id}`",
            f"**To**: `{self.to_id}`",
            "",
        ]

        if self.missing in ("from", "both"):
            lines.append(f"**Problem**: Source node `{self.from_id}` does not exist.")
            if self.available_from:
                lines.append("**Valid source IDs**:")
                for a in sorted(self.available_from)[:10]:
                    lines.append(f"  - `{a}`")
                if len(self.available_from) > 10:
                    lines.append(f"  - ... and {len(self.available_from) - 10} more")
            lines.append("")

        if self.missing in ("to", "both"):
            lines.append(f"**Problem**: Target node `{self.to_id}` does not exist.")
            if self.available_to:
                lines.append("**Valid target IDs**:")
                for a in sorted(self.available_to)[:10]:
                    lines.append(f"  - `{a}`")
                if len(self.available_to) > 10:
                    lines.append(f"  - ... and {len(self.available_to) - 10} more")

        lines.extend(
            [
                "",
                "**Solution**: Use IDs that exist in the graph. Create the nodes first if needed.",
            ]
        )

        return "\n".join(lines)


@dataclass
class GraphCorruptionError(Exception):
    """Raised when post-mutation invariant checks detect graph corruption.

    Unlike GraphIntegrityError, this indicates a code bug rather than
    invalid LLM output. The graph should be rolled back to the last
    known good state.

    Attributes:
        violations: List of invariant violations found.
        stage: Stage during which corruption was detected.
    """

    violations: list[str]
    stage: str = ""

    def __post_init__(self) -> None:
        msg = f"Graph corruption detected after {self.stage or 'unknown'} stage"
        if self.violations:
            msg += f": {len(self.violations)} violation(s)"
        super().__init__(msg)

    def __str__(self) -> str:
        lines = [f"Graph corruption detected after {self.stage or 'unknown'} stage:"]
        for v in self.violations[:5]:
            lines.append(f"  - {v}")
        if len(self.violations) > 5:
            lines.append(f"  - ... and {len(self.violations) - 5} more")
        return "\n".join(lines)
