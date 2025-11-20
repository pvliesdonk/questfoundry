"""
QuestFoundry Runtime - Phase 5B Implementation

Transform YAML role and loop definitions into executable LangGraph StateGraphs.
"""

# Import models first (no dependencies)
from questfoundry.runtime.models.state import StudioState, BarStatus, Message, Artifact
from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.models.loop import LoopPattern, Edge, ExitCondition

# Import core (models already loaded)
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.edge_evaluator import EdgeEvaluator

# Import graph factory (requires LangGraph)
try:
    from questfoundry.runtime.core.graph_factory import GraphFactory
except ImportError:
    # GraphFactory requires langgraph, which is optional
    GraphFactory = None

# Import plugins
from questfoundry.runtime.plugins.llm.anthropic import get_llm, AnthropicAdapter
from questfoundry.runtime.plugins.tools.registry import get_tool_registry, Tool, ToolRegistry

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
