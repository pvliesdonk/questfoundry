"""
Analyze Story Graph tool implementation.

Programmatically analyzes the story topology to check:
1. Reachability - Can all nodes be reached from entry?
2. Dead ends - Nodes with no outgoing connections
3. Orphans - Nodes with no incoming connections (except entry)
4. Missing targets - References to non-existent nodes
5. Lifecycle status - What state are the paths in?

This provides agents with concrete graph data instead of
requiring LLM reasoning about graph structure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@dataclass
class NodeInfo:
    """Information about a node in the story graph."""

    anchor_id: str
    artifact_id: str
    title: str
    lifecycle_state: str
    artifact_type: str  # section_brief or section
    outgoing: list[str] = field(default_factory=list)  # target anchors
    incoming: list[str] = field(default_factory=list)  # source anchors


@dataclass
class GraphAnalysis:
    """Result of story graph analysis."""

    # All nodes in the graph
    nodes: dict[str, NodeInfo] = field(default_factory=dict)

    # Entry point(s)
    entry_points: list[str] = field(default_factory=list)

    # Reachability from entry
    reachable: set[str] = field(default_factory=set)
    unreachable: set[str] = field(default_factory=set)

    # Structural issues
    dead_ends: list[str] = field(default_factory=list)  # no outgoing
    orphans: list[str] = field(default_factory=list)  # no incoming (except entry)
    missing_targets: list[dict[str, str]] = field(default_factory=list)  # {from, to}

    # Lifecycle breakdown
    by_lifecycle: dict[str, list[str]] = field(default_factory=dict)

    # Whether any reachable node is in a stable state (approved/cold)
    has_reachable_stable_node: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for tool result."""
        return {
            "summary": {
                "total_nodes": len(self.nodes),
                "entry_points": self.entry_points,
                "reachable_count": len(self.reachable),
                "unreachable_count": len(self.unreachable),
                "dead_end_count": len(self.dead_ends),
                "orphan_count": len(self.orphans),
                "missing_target_count": len(self.missing_targets),
                "has_reachable_stable_node": self.has_reachable_stable_node,
            },
            "reachable": sorted(self.reachable),
            "unreachable": sorted(self.unreachable),
            "dead_ends": self.dead_ends,
            "orphans": self.orphans,
            "missing_targets": self.missing_targets,
            "by_lifecycle": self.by_lifecycle,
            "nodes": {
                k: {
                    "anchor_id": v.anchor_id,
                    "artifact_id": v.artifact_id,
                    "title": v.title,
                    "lifecycle_state": v.lifecycle_state,
                    "type": v.artifact_type,
                    "outgoing": v.outgoing,
                    "incoming": v.incoming,
                }
                for k, v in self.nodes.items()
            },
        }


@register_tool("analyze_story_graph")
class AnalyzeStoryGraphTool(BaseTool):
    """
    Analyze story topology for reachability and structural issues.

    Builds a directed graph from section_briefs and sections,
    then performs reachability analysis from entry points.

    Returns structured data about:
    - Which nodes are reachable from entry
    - Dead ends (no way forward)
    - Orphans (no way to reach them)
    - Missing targets (broken links)
    - Lifecycle status of paths
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute story graph analysis."""
        include_drafts = args.get("include_drafts", True)
        entry_anchor = args.get("entry_anchor")  # Optional explicit entry

        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot analyze story graph",
            )

        # Build the graph
        analysis = self._build_and_analyze_graph(include_drafts, entry_anchor)

        # Categorize issues by severity (following semantic clarity from issue #228)
        blocking_issues, warnings = self._categorize_issues(analysis)

        # Determine analysis outcome
        if blocking_issues:
            analysis_outcome = "failed"
        elif warnings:
            analysis_outcome = "warnings"
        else:
            analysis_outcome = "passed"

        # Generate directive recovery actions (not passive hints)
        recovery_actions = self._generate_recovery_actions(blocking_issues)

        return ToolResult(
            success=True,
            data={
                "analysis_outcome": analysis_outcome,
                "blocking_issues": blocking_issues,
                "warnings": warnings,
                "recovery_actions": recovery_actions,
                "analysis": analysis.to_dict(),
            },
        )

    def _build_and_analyze_graph(
        self,
        include_drafts: bool,
        entry_anchor: str | None,
    ) -> GraphAnalysis:
        """Build graph from artifacts and analyze."""
        analysis = GraphAnalysis()
        project = self._context.project
        assert project is not None  # Checked in execute()

        # Query section_briefs (no limit - get all)
        briefs = project.query_artifacts(artifact_type="section_brief")

        # Query sections (no limit - get all)
        sections = project.query_artifacts(artifact_type="section")

        # Build nodes from briefs
        for brief in briefs:
            lifecycle = brief.get("_lifecycle_state", "draft")
            if not include_drafts and lifecycle == "draft":
                continue

            anchor: str = brief.get("target_anchor") or brief.get("_id") or ""
            if not anchor:
                continue  # Skip briefs without valid anchor

            node = NodeInfo(
                anchor_id=anchor,
                artifact_id=brief.get("_id", ""),
                title=brief.get("section_title", brief.get("title", "Untitled")),
                lifecycle_state=lifecycle,
                artifact_type="section_brief",
            )

            # Extract outgoing connections from choice_intents (deduplicate)
            # choice_intents are objects with target_anchor field
            choice_intents = brief.get("choice_intents", [])
            for choice in choice_intents:
                if isinstance(choice, dict):
                    target = choice.get("target_anchor")
                    if target and target not in node.outgoing:
                        node.outgoing.append(target)

            analysis.nodes[anchor] = node

        # Build nodes from sections (may override briefs)
        for section in sections:
            lifecycle = section.get("_lifecycle_state", "draft")
            if not include_drafts and lifecycle == "draft":
                continue

            section_anchor: str = (
                section.get("anchor") or section.get("anchor_id") or section.get("_id") or ""
            )
            if not section_anchor:
                continue  # Skip sections without valid anchor

            # Check if we already have this from briefs
            if section_anchor in analysis.nodes:
                existing = analysis.nodes[section_anchor]
                # Update with section info if section is more advanced
                if self._lifecycle_priority(lifecycle) > self._lifecycle_priority(
                    existing.lifecycle_state
                ):
                    existing.lifecycle_state = lifecycle
                    existing.artifact_type = "section"
                    existing.artifact_id = section.get("_id", "")
                    # Also update title from section when it becomes authoritative
                    existing.title = section.get("title", existing.title)
                # Merge outgoing connections (choices can be objects with target_anchor)
                section_choices = section.get("choices", [])
                for choice in section_choices:
                    if isinstance(choice, dict):
                        target = choice.get("target_anchor")
                        if target and target not in existing.outgoing:
                            existing.outgoing.append(target)
            else:
                node = NodeInfo(
                    anchor_id=section_anchor,
                    artifact_id=section.get("_id", ""),
                    title=section.get("title", "Untitled"),
                    lifecycle_state=lifecycle,
                    artifact_type="section",
                )
                # choices are objects with target_anchor field
                section_choices = section.get("choices", [])
                for choice in section_choices:
                    if isinstance(choice, dict):
                        target = choice.get("target_anchor")
                        if target and target not in node.outgoing:
                            node.outgoing.append(target)
                analysis.nodes[section_anchor] = node

        # Build incoming connections and detect missing targets in one pass
        for anchor, node in analysis.nodes.items():
            for target in node.outgoing:
                if target in analysis.nodes:
                    analysis.nodes[target].incoming.append(anchor)
                else:
                    # Target doesn't exist - record as missing
                    analysis.missing_targets.append({"from": anchor, "to": target})

        # Identify entry points
        if entry_anchor and entry_anchor in analysis.nodes:
            analysis.entry_points = [entry_anchor]
        else:
            # Entry points are nodes with no incoming connections
            for anchor, node in analysis.nodes.items():
                if not node.incoming:
                    analysis.entry_points.append(anchor)

        # Compute reachability via BFS from entry points (using deque for O(1) popleft)
        analysis.reachable = set()
        queue: deque[str] = deque(analysis.entry_points)
        while queue:
            current = queue.popleft()
            if current in analysis.reachable:
                continue
            if current not in analysis.nodes:
                continue
            analysis.reachable.add(current)
            for target in analysis.nodes[current].outgoing:
                if target not in analysis.reachable and target in analysis.nodes:
                    queue.append(target)

        # Find unreachable nodes
        all_anchors = set(analysis.nodes.keys())
        analysis.unreachable = all_anchors - analysis.reachable

        # Find dead ends (no outgoing connections)
        for anchor, node in analysis.nodes.items():
            if not node.outgoing:
                analysis.dead_ends.append(anchor)

        # Find orphans (no incoming except entry points)
        for anchor, node in analysis.nodes.items():
            if not node.incoming and anchor not in analysis.entry_points:
                analysis.orphans.append(anchor)

        # Group by lifecycle
        by_lifecycle: dict[str, list[str]] = defaultdict(list)
        for anchor, node in analysis.nodes.items():
            by_lifecycle[node.lifecycle_state].append(anchor)
        analysis.by_lifecycle = dict(by_lifecycle)

        # Check if any reachable node is in a stable state (approved/cold)
        stable_states = {"approved", "cold"}
        stable_nodes = {a for a, n in analysis.nodes.items() if n.lifecycle_state in stable_states}
        analysis.has_reachable_stable_node = bool(analysis.reachable & stable_nodes)

        return analysis

    def _lifecycle_priority(self, state: str) -> int:
        """Higher number = more advanced lifecycle state."""
        priorities = {
            "draft": 0,
            "ready": 1,
            "in_use": 1,
            "review": 2,
            "approved": 3,
            "cold": 4,
            "archived": 5,
        }
        return priorities.get(state, 0)

    def _categorize_issues(
        self, analysis: GraphAnalysis
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Categorize issues into blocking vs warnings.

        Blocking issues prevent graph approval:
        - Missing targets (broken links)
        - No entry points

        Warnings are concerns but don't block:
        - Unreachable nodes
        - Dead ends
        - Orphans
        - Multiple entry points
        - All drafts (no stable paths)

        Returns:
            (blocking_issues, warnings) - each is a list of structured issue dicts
        """
        blocking: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        # BLOCKING: Missing targets (broken links)
        if analysis.missing_targets:
            missing = {m["to"] for m in analysis.missing_targets}
            blocking.append(
                {
                    "issue_type": "missing_targets",
                    "severity": "blocking",
                    "description": f"{len(missing)} target node(s) referenced but don't exist",
                    "affected_nodes": sorted(missing)[:5],
                    "count": len(missing),
                }
            )

        # BLOCKING: No entry points
        if not analysis.entry_points:
            blocking.append(
                {
                    "issue_type": "no_entry_point",
                    "severity": "blocking",
                    "description": "No entry point found - graph has no starting node",
                    "affected_nodes": [],
                    "count": 0,
                }
            )

        # WARNING: Unreachable nodes
        if analysis.unreachable:
            warnings.append(
                {
                    "issue_type": "unreachable_nodes",
                    "severity": "warning",
                    "description": f"{len(analysis.unreachable)} node(s) cannot be reached from entry",
                    "affected_nodes": sorted(analysis.unreachable)[:5],
                    "count": len(analysis.unreachable),
                }
            )

        # WARNING: Multiple entry points (may be intentional)
        if len(analysis.entry_points) > 1:
            warnings.append(
                {
                    "issue_type": "multiple_entry_points",
                    "severity": "warning",
                    "description": f"{len(analysis.entry_points)} entry points detected",
                    "affected_nodes": analysis.entry_points[:5],
                    "count": len(analysis.entry_points),
                }
            )

        # WARNING: Dead ends (reachable only)
        reachable_dead_ends = [d for d in analysis.dead_ends if d in analysis.reachable]
        if reachable_dead_ends:
            warnings.append(
                {
                    "issue_type": "dead_ends",
                    "severity": "warning",
                    "description": f"{len(reachable_dead_ends)} reachable node(s) have no outgoing paths",
                    "affected_nodes": sorted(reachable_dead_ends)[:5],
                    "count": len(reachable_dead_ends),
                }
            )

        # WARNING: Orphans
        if analysis.orphans:
            warnings.append(
                {
                    "issue_type": "orphan_nodes",
                    "severity": "warning",
                    "description": f"{len(analysis.orphans)} node(s) have no incoming paths",
                    "affected_nodes": sorted(analysis.orphans)[:5],
                    "count": len(analysis.orphans),
                }
            )

        # WARNING: No stable paths
        all_drafts = set(analysis.by_lifecycle.get("draft", []))
        reachable_drafts = all_drafts & analysis.reachable
        if reachable_drafts and not analysis.has_reachable_stable_node:
            warnings.append(
                {
                    "issue_type": "no_stable_paths",
                    "severity": "warning",
                    "description": f"All {len(reachable_drafts)} reachable node(s) are drafts",
                    "affected_nodes": sorted(reachable_drafts)[:5],
                    "count": len(reachable_drafts),
                }
            )

        return blocking, warnings

    def _generate_recovery_actions(
        self,
        blocking_issues: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """
        Generate directive recovery actions (not passive hints).

        Each action tells the agent exactly what to do next.
        """
        actions: list[dict[str, str]] = []

        # Actions for blocking issues first (highest priority)
        for issue in blocking_issues:
            if issue["issue_type"] == "missing_targets":
                nodes = issue["affected_nodes"]
                actions.append(
                    {
                        "priority": "high",
                        "action": "Create missing target nodes",
                        "details": f"Create section_briefs for: {', '.join(nodes[:3])}"
                        + ("..." if len(nodes) > 3 else ""),
                    }
                )
            elif issue["issue_type"] == "no_entry_point":
                actions.append(
                    {
                        "priority": "high",
                        "action": "Designate entry point",
                        "details": "Create a section_brief with no incoming connections to serve as entry",
                    }
                )

        # If graph is valid, say so explicitly
        if not blocking_issues:
            actions.append(
                {
                    "priority": "info",
                    "action": "Graph structure is valid",
                    "details": "No blocking issues found. Review warnings if any.",
                }
            )

        return actions
