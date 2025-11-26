"""
Core runtime components for QuestFoundry.

Primary Components (Mesh Architecture):
- ControlPlane: Protocol-driven mesh routing (recommended)
- NodeFactory: Transforms role profiles into LangGraph nodes
- SchemaRegistry: Loads and validates YAML definitions
- StateManager: Manages StudioState during execution

Legacy Components (Static Topology):
- GraphFactory: Transforms loop patterns into static graphs (deprecated)
- EdgeEvaluator: Evaluates conditional edges (deprecated)
"""

from questfoundry.runtime.core.control_plane import ControlPlane, DormancyRegistry
from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager

# Legacy exports (deprecated - use ControlPlane instead)
from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.edge_evaluator import EdgeEvaluator

__all__ = [
    # Primary (Mesh Architecture)
    "ControlPlane",
    "DormancyRegistry",
    "NodeFactory",
    "SchemaRegistry",
    "StateManager",
    # Legacy (Static Topology - deprecated)
    "GraphFactory",
    "EdgeEvaluator",
]
