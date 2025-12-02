"""
QuestFoundry Runtime - Phase 5B Implementation

Transform YAML role and loop definitions into executable LangGraph StateGraphs.
"""

# Import only models (no external dependencies)
from questfoundry.runtime.models.loop import Edge, ExitCondition, LoopPattern
from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.models.state import Artifact, BarStatus, Message, StudioState


# Use lazy imports for everything else to avoid dependency issues
def __getattr__(name):
    """Lazy import to avoid loading modules with heavy dependencies."""
    if name == "EdgeEvaluator":
        from questfoundry.runtime.core.edge_evaluator import EdgeEvaluator

        return EdgeEvaluator
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
        try:
            from questfoundry.runtime.core.graph_factory import GraphFactory

            return GraphFactory
        except ImportError:
            return None
    elif name == "RuntimeContextAssembler":
        from questfoundry.runtime.core.runtime_context_assembler import RuntimeContextAssembler

        return RuntimeContextAssembler
    elif name == "CapabilityMapper":
        from questfoundry.runtime.core.capability_mapper import CapabilityMapper

        return CapabilityMapper
    elif name == "AnthropicAdapter":
        from questfoundry.runtime.plugins.llm.anthropic import AnthropicAdapter

        return AnthropicAdapter
    elif name == "get_llm":
        from questfoundry.runtime.plugins.llm.anthropic import get_llm

        return get_llm
    elif name == "Tool":
        from questfoundry.runtime.plugins.tools.registry import Tool

        return Tool
    elif name == "ToolRegistry":
        from questfoundry.runtime.plugins.tools.registry import ToolRegistry

        return ToolRegistry
    elif name == "get_tool_registry":
        from questfoundry.runtime.plugins.tools.registry import get_tool_registry

        return get_tool_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    # Core
    "SchemaRegistry",
    "StateManager",
    "NodeFactory",
    "EdgeEvaluator",
    "GraphFactory",
    # Models
    "StudioState",
    "BarStatus",
    "Message",
    "Artifact",
    "RoleProfile",
    "LoopPattern",
    "Edge",
    "ExitCondition",
    # Plugins
    "get_llm",
    "AnthropicAdapter",
    "get_tool_registry",
    "Tool",
    "ToolRegistry",
]
