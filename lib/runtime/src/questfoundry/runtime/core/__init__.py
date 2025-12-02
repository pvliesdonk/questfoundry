"""
Core runtime components for QuestFoundry.

Primary Components (Mesh Architecture):
- ControlPlane: Protocol-driven mesh routing (recommended)
- NodeFactory: Transforms role profiles into LangGraph nodes
- SchemaRegistry: Loads and validates YAML definitions
- StateManager: Manages StudioState during execution
- RuntimeContextAssembler: Dynamically assembles agent prompts from YAML
- CapabilityMapper: Maps abstract capabilities to concrete tools

Legacy Components (Static Topology):
- GraphFactory: Transforms loop patterns into static graphs (deprecated)
- EdgeEvaluator: Evaluates conditional edges (deprecated)
"""


# Use lazy imports to avoid dependency issues
def __getattr__(name):
    """Lazy import to avoid loading all modules at once."""
    if name == "ControlPlane":
        from questfoundry.runtime.core.control_plane import ControlPlane

        return ControlPlane
    elif name == "DormancyRegistry":
        from questfoundry.runtime.core.control_plane import DormancyRegistry

        return DormancyRegistry
    elif name == "NodeFactory":
        from questfoundry.runtime.core.node_factory import NodeFactory

        return NodeFactory
    elif name == "SchemaRegistry":
        from questfoundry.runtime.core.schema_registry import SchemaRegistry

        return SchemaRegistry
    elif name == "StateManager":
        from questfoundry.runtime.core.state_manager import StateManager

        return StateManager
    elif name == "GraphFactory":
        from questfoundry.runtime.core.graph_factory import GraphFactory

        return GraphFactory
    elif name == "EdgeEvaluator":
        from questfoundry.runtime.core.edge_evaluator import EdgeEvaluator

        return EdgeEvaluator
    elif name == "RuntimeContextAssembler":
        from questfoundry.runtime.core.runtime_context_assembler import RuntimeContextAssembler

        return RuntimeContextAssembler
    elif name == "CapabilityMapper":
        from questfoundry.runtime.core.capability_mapper import CapabilityMapper

        return CapabilityMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Primary (Mesh Architecture)
    "ControlPlane",
    "DormancyRegistry",
    "NodeFactory",
    "SchemaRegistry",
    "StateManager",
    "RuntimeContextAssembler",
    "CapabilityMapper",
    # Legacy (Static Topology - deprecated)
    "GraphFactory",
    "EdgeEvaluator",
]
