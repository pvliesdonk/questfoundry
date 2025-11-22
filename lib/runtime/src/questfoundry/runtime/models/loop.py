"""
Loop pattern models - represents a loop_pattern.schema.json definition.

Based on spec: components/graph_factory.md
"""

from typing import Any


class Edge:
    """Represents an edge in loop topology."""

    def __init__(self, data: dict[str, Any]):
        """Initialize from edge definition."""
        self.raw = data
        self.source = data.get("source", "")
        self.target = data.get("target", "")
        self.type = data.get("type", "direct")  # direct | conditional
        self.condition = data.get("condition", {})


class ExitCondition:
    """Represents an exit condition for the loop."""

    def __init__(self, data: dict[str, Any]):
        """Initialize from exit condition definition."""
        self.raw = data
        self.name = data.get("name", "")
        self.condition = data.get("condition", {})


class LoopPattern:
    """Represents a loop pattern YAML definition."""

    def __init__(self, data: dict[str, Any]):
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
        self.edges: list[Edge] = []
        for edge_data in topology.get("edges", []):
            self.edges.append(Edge(edge_data))

        # Parse exit conditions
        self.exit_conditions: list[ExitCondition] = []
        for exit_cond_data in topology.get("exit_conditions", []):
            self.exit_conditions.append(ExitCondition(exit_cond_data))

        # Roles
        roles = data.get("roles", {})
        self.required_roles = roles.get("required", [])
        self.optional_roles = roles.get("optional", [])

    def get_node_ids(self) -> list[str]:
        """Get all node IDs in the topology."""
        return [node if isinstance(node, str) else node.get("id", "") for node in self.nodes]

    def get_entry_node_id(self) -> str:
        """Get the entry node ID."""
        return self.entry_node

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in topology."""
        return node_id in self.get_node_ids()

    def get_node_role_id(self, node_id: str) -> str:
        """
        Get the role_id for a given node_id.

        Args:
            node_id: The node identifier (e.g., "showrunner_initiate")

        Returns:
            The role_id field from the node definition (e.g., "Showrunner")

        Raises:
            ValueError: If node_id doesn't exist in topology
        """
        for node in self.nodes:
            if isinstance(node, dict) and node.get("id") == node_id:
                role_id = node.get("role_id")
                if not role_id:
                    raise ValueError(f"Node {node_id} missing role_id field")
                return role_id
        raise ValueError(f"Node {node_id} not found in loop topology")

    def get_node_sub_nodes(self, node_id: str) -> list[dict[str, Any | None]]:
        """
        Get sub_nodes for a Multi-role node.

        Args:
            node_id: The node identifier

        Returns:
            List of sub_node definitions, or None if not a Multi-role node

        Example sub_node structure:
            [
                {"role": "Lore Weaver", "task": "Review narrative hooks..."},
                {"role": "Plotwright", "task": "Review hooks for structural impact..."}
            ]
        """
        for node in self.nodes:
            if isinstance(node, dict) and node.get("id") == node_id:
                return node.get("sub_nodes")
        return None

    def is_parallel_node(self, node_id: str) -> bool:
        """
        Check if a node should execute roles in parallel.

        Args:
            node_id: The node identifier

        Returns:
            True if node has parallel_execution flag set to True
        """
        for node in self.nodes:
            if isinstance(node, dict) and node.get("id") == node_id:
                return node.get("parallel_execution", False)
        return False
