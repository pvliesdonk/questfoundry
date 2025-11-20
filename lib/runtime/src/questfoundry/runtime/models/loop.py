"""
Loop pattern models - represents a loop_pattern.schema.json definition.

Based on spec: components/graph_factory.md
"""

from typing import Any, Dict, List, Literal, Optional


class Edge:
    """Represents an edge in loop topology."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize from edge definition."""
        self.raw = data
        self.source = data.get("source", "")
        self.target = data.get("target", "")
        self.type = data.get("type", "direct")  # direct | conditional
        self.condition = data.get("condition", {})


class ExitCondition:
    """Represents an exit condition for the loop."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize from exit condition definition."""
        self.raw = data
        self.name = data.get("name", "")
        self.condition = data.get("condition", {})


class LoopPattern:
    """Represents a loop pattern YAML definition."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize from parsed YAML data."""
        self.raw = data

        # Extract key fields
        self.id = data.get("id", "")
        metadata = data.get("metadata", {})
        self.name = metadata.get("name", "")
        self.description = metadata.get("description", "")

        # Topology
        topology = data.get("topology", {})
        self.entry_node = topology.get("entry_node", "")
        self.nodes = topology.get("nodes", [])

        # Parse edges
        self.edges: List[Edge] = []
        for edge_data in topology.get("edges", []):
            self.edges.append(Edge(edge_data))

        # Parse exit conditions
        self.exit_conditions: List[ExitCondition] = []
        for exit_cond_data in topology.get("exit_conditions", []):
            self.exit_conditions.append(ExitCondition(exit_cond_data))

        # Roles
        roles = data.get("roles", {})
        self.required_roles = roles.get("required", [])
        self.optional_roles = roles.get("optional", [])

    def get_node_ids(self) -> List[str]:
        """Get all node IDs in the topology."""
        return [node if isinstance(node, str) else node.get("id", "") for node in self.nodes]

    def get_entry_node_id(self) -> str:
        """Get the entry node ID."""
        return self.entry_node

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in topology."""
        return node_id in self.get_node_ids()
