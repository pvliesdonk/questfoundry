"""
Graph Factory - transforms loop patterns into executable LangGraph StateGraphs.

DEPRECATED: This module uses static topology routing (hardcoded edges from loop YAML).
Use ControlPlane instead for protocol-driven mesh routing.

Based on spec: components/graph_factory.md
"""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from questfoundry.runtime.core.edge_evaluator import EdgeEvaluator
from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.schema_registry import SchemaRegistry
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.loop import LoopPattern
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


class GraphFactory:
    """
    Transform loop patterns into executable LangGraph StateGraphs.

    DEPRECATED: Use ControlPlane instead for protocol-driven mesh routing.

    This class uses static topology routing where edges are defined in loop YAML files.
    The new ControlPlane class implements dynamic routing based on message envelopes,
    which is the intended architecture per the Studio Graph Architecture spec.
    """

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        node_factory: NodeFactory | None = None,
        edge_evaluator: EdgeEvaluator | None = None,
        state_manager: StateManager | None = None,
        preferred_provider: str | None = None,
    ):
        """Initialize graph factory with dependent components."""
        self.schema_registry = schema_registry or SchemaRegistry()
        self.state_manager = state_manager or StateManager()
        # Pass state_manager and preferred_provider to node_factory
        self.node_factory = node_factory or NodeFactory(
            self.schema_registry,
            state_manager=self.state_manager,
            preferred_provider=preferred_provider,
        )
        self.edge_evaluator = edge_evaluator or EdgeEvaluator()

        self._graph_cache: dict[str, Any] = {}

    def load_loop(self, loop_id: str) -> LoopPattern:
        """
        Load and validate loop YAML file.

        Args:
            loop_id: Loop identifier (e.g., "story_spark")

        Returns:
            LoopPattern object

        Raises:
            FileNotFoundError: If loop YAML doesn't exist
            ValidationError: If YAML doesn't match schema
        """
        return self.schema_registry.load_loop(loop_id)

    def validate_topology(self, loop: LoopPattern) -> None:
        """
        Validate loop topology before building graph.

        Checks:
        1. entry_node is in topology.nodes
        2. All edge sources are in topology.nodes
        3. All edge targets are in topology.nodes or "__end__"
        4. All nodes are reachable from entry_node
        5. At least one exit condition exists

        Args:
            loop: LoopPattern to validate

        Raises:
            ValueError: If topology is invalid
        """
        node_ids = set(loop.get_node_ids())
        entry_node = loop.get_entry_node_id()

        # Check 1: entry_node in nodes
        if entry_node not in node_ids:
            raise ValueError(f"Entry node '{entry_node}' not in topology nodes: {node_ids}")

        # Check 2: All edge sources in nodes
        for edge in loop.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' not in topology nodes: {node_ids}")

            # Check 3: All edge targets in nodes or END
            # Note: YAML files use 'END' while LangGraph uses '__end__'
            if edge.target not in node_ids and edge.target != "END" and edge.target != END:
                raise ValueError(f"Edge target '{edge.target}' not in topology nodes: {node_ids}")

        # Check 5: At least one way to exit (either exit conditions or edges to END)
        has_end_edge = any(edge.target == "END" or edge.target == END for edge in loop.edges)
        if not loop.exit_conditions and not has_end_edge:
            raise ValueError("Loop must have at least one exit condition or an edge to END")

        logger.info(f"Topology validation passed for loop: {loop.id}")

    def create_state_graph(self) -> StateGraph:
        """
        Create LangGraph StateGraph with StudioState annotation.

        Returns:
            Empty StateGraph ready for nodes and edges
        """
        graph = StateGraph(StudioState)
        return graph

    def add_nodes(self, graph: StateGraph, loop: LoopPattern) -> None:
        """
        Add all nodes from loop.topology.nodes.

        For each node_id in loop.topology.nodes:
        1. Load role definition
        2. Create Runnable using NodeFactory.create_role_node()
        3. Add to graph

        Args:
            graph: StateGraph to add nodes to
            loop: LoopPattern with node definitions
        """
        for node_id in loop.get_node_ids():
            try:
                # Get the role_id for this node
                role_id = loop.get_node_role_id(node_id)

                # Handle special "Multi" role (parallel execution)
                if role_id == "Multi":
                    # Check if this is a parallel execution node
                    if loop.is_parallel_node(node_id):
                        # Get sub-nodes for parallel execution
                        sub_nodes = loop.get_node_sub_nodes(node_id)
                        if sub_nodes:
                            logger.info(
                                f"Creating parallel multi-role node: [cyan]{node_id}[/cyan] with {len(sub_nodes)} sub-roles"
                            )
                            node_runnable = self.node_factory.create_multi_role_node(
                                sub_nodes, node_id
                            )
                        else:
                            logger.warning(
                                f"Multi-role node {node_id} has no sub_nodes, creating placeholder"
                            )

                            def multi_placeholder(state):
                                logger.warning(
                                    f"Multi-role node {node_id} executed but has no sub_nodes"
                                )
                                return {}

                            node_runnable = multi_placeholder
                    else:
                        logger.warning(
                            f"Multi-role node {node_id} missing parallel_execution flag, creating placeholder"
                        )

                        def multi_placeholder(state):
                            logger.warning(
                                f"Multi-role node {node_id} missing parallel_execution=true"
                            )
                            return {}

                        node_runnable = multi_placeholder
                else:
                    # Create node runnable using role_id
                    node_runnable = self.node_factory.create_role_node(role_id)

                # Add to graph with node_id as the graph node name
                graph.add_node(node_id, node_runnable)
                logger.debug(f"Added node: [cyan]{node_id}[/cyan] → role: [bold]{role_id}[/bold]")

            except Exception as e:
                logger.error(f"Failed to add node {node_id}: {e}")
                raise

    def add_direct_edge(self, graph: StateGraph, edge: Any) -> None:
        """
        Add unconditional edge from source to target.

        Args:
            graph: StateGraph
            edge: Edge definition with source and target
        """
        # Convert 'END' from YAML to LangGraph's END constant
        target = END if edge.target == "END" else edge.target
        graph.add_edge(edge.source, target)
        logger.debug(f"Added direct edge: {edge.source} → {edge.target}")

    def add_conditional_edge(self, graph: StateGraph, edge: Any) -> None:
        """
        Add conditional edge with routing function.

        Args:
            graph: StateGraph
            edge: Edge definition with condition
        """
        condition = edge.raw.get("condition", {}) if hasattr(edge, "raw") else {}
        routes = condition.get("routes", {}) if isinstance(condition, dict) else {}
        default_route = condition.get("default_route") if isinstance(condition, dict) else None

        # Create routing function
        routing_fn = self.edge_evaluator.create_routing_function(edge.raw)

        # Convert 'END' from YAML to LangGraph's END constant
        target = END if edge.target == "END" else edge.target

        path_map: dict[Any, Any] = {
            edge.target: target,
            edge.source: edge.source,
            END: END,
            "__end__": END,
        }

        for mapped_target in routes.values():
            if mapped_target is None:
                continue
            norm = END if mapped_target == "END" else mapped_target
            path_map[mapped_target] = norm

        if default_route:
            norm = END if default_route == "END" else default_route
            path_map[default_route] = norm

        # Drop any routes that resolve to None to avoid LangGraph type errors
        cleaned_path_map = {k: v for k, v in path_map.items() if v is not None}

        # Add conditional edge with correct API
        graph.add_conditional_edges(
            source=edge.source,
            path=routing_fn,
            path_map=cleaned_path_map,
        )

        logger.debug(f"Added conditional edge: {edge.source} → {edge.target}")

    def add_edges(self, graph: StateGraph, loop: LoopPattern) -> None:
        """
        Add all edges from loop.topology.edges.

        Handles both direct and conditional edges.

        Args:
            graph: StateGraph
            loop: LoopPattern with edge definitions
        """
        for edge in loop.edges:
            try:
                if edge.type == "direct":
                    self.add_direct_edge(graph, edge)
                elif edge.type == "conditional":
                    self.add_conditional_edge(graph, edge)
                else:
                    logger.warning(f"Unknown edge type: {edge.type}")

            except Exception as e:
                logger.error(f"Failed to add edge {edge.source} → {edge.target}: {e}")
                raise

    def set_entry_point(self, graph: StateGraph, loop: LoopPattern) -> None:
        """
        Set the entry node for the graph.

        Args:
            graph: StateGraph
            loop: LoopPattern with entry_node defined
        """
        entry_node = loop.get_entry_node_id()
        graph.set_entry_point(entry_node)
        logger.debug(f"Set entry point: {entry_node}")

    def add_exit_conditions(self, graph: StateGraph, loop: LoopPattern) -> None:
        """
        Add conditional edges to END for each exit condition from YAML.

        Evaluates exit conditions defined in loop YAML and creates proper
        routing logic to terminate the loop when conditions are met.

        Args:
            graph: StateGraph
            loop: LoopPattern with exit_conditions defined
        """
        if not loop.exit_conditions:
            logger.warning(f"Loop {loop.id} has no exit conditions defined")
            return

        if not loop.edges:
            logger.warning(f"Loop {loop.id} has no edges")
            return

        # Add exit condition routing from every node; EdgeEvaluator decides
        for node_id in loop.get_node_ids():
            try:

                def make_routing(nid: str):
                    def routing(state: StudioState) -> str:
                        for exit_cond in loop.exit_conditions:
                            condition = exit_cond.condition
                            if self.edge_evaluator.evaluate_condition(condition, state):
                                logger.info(f"Exit condition '{exit_cond.name}' met at node {nid}")
                                return END
                        return nid

                    return routing

                routing_fn = make_routing(node_id)
                graph.add_conditional_edges(
                    source=node_id,
                    path=routing_fn,
                    path_map={node_id: node_id, END: END},
                )
            except Exception as e:
                logger.error(f"Failed to add exit condition routing from {node_id}: {e}")
                raise

    def compile_graph(self, graph: StateGraph) -> Any:
        """
        Compile the graph into executable form.

        Args:
            graph: StateGraph to compile

        Returns:
            Compiled graph ready for invocation

        Note:
            recursion_limit is set at invocation time (in showrunner.py),
            not during compilation.
        """
        try:
            compiled = graph.compile()
            logger.info("Graph compiled successfully")
            return compiled

        except Exception as e:
            logger.error(f"Graph compilation failed: {e}")
            raise

    def create_loop_graph(self, loop_id: str, context: dict[str, Any | None] = None) -> Any:
        """
        Complete graph creation flow.

        Args:
            loop_id: Loop pattern identifier (e.g., "story_spark")
            context: Initial context for StudioState (optional)

        Returns:
            Compiled LangGraph ready to invoke()

        Raises:
            FileNotFoundError: If loop doesn't exist
            ValueError: If loop definition is invalid
        """
        if context is None:
            context = {}

        # Check cache
        if loop_id in self._graph_cache:
            logger.debug(f"Returning cached graph for loop: {loop_id}")
            return self._graph_cache[loop_id]

        logger.info(f"Creating graph for loop: {loop_id}")

        # 1. Load loop definition
        loop = self.load_loop(loop_id)

        # 2. Validate topology
        self.validate_topology(loop)

        # 3. Create StateGraph
        graph = self.create_state_graph()

        # 4. Add nodes
        self.add_nodes(graph, loop)

        # 5. Add edges
        self.add_edges(graph, loop)

        # 6. Set entry point
        self.set_entry_point(graph, loop)

        # 7. Add exit conditions
        self.add_exit_conditions(graph, loop)

        # 8. Compile
        compiled = self.compile_graph(graph)

        # Cache
        self._graph_cache[loop_id] = compiled

        logger.info(f"Graph created successfully for loop: {loop_id}")
        return compiled

    def clear_cache(self) -> None:
        """Clear graph cache."""
        self._graph_cache.clear()
        logger.debug("Cleared graph cache")
