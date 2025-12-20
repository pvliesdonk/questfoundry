"""
Validate Story Graph tool implementation.

Programmatically analyzes the story topology to check:
1. Reachability - Can all nodes be reached from entry?
2. Dead ends - Nodes with no outgoing connections
3. Orphans - Nodes with no incoming connections (except entry)
4. Missing targets - References to non-existent nodes
5. Lifecycle status - What state are the paths in?

This provides gatekeeper with concrete data instead of
requiring LLM reasoning about graph structure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    pass


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

    # Paths requiring only approved/cold content
    stable_paths: bool = False

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
                "has_stable_paths": self.stable_paths,
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


@register_tool("validate_story_graph")
class ValidateStoryGraphTool(BaseTool):
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
        """Execute story graph validation."""
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

        # Determine overall status
        has_issues = len(analysis.unreachable) > 0 or len(analysis.missing_targets) > 0

        return ToolResult(
            success=True,
            data={
                "status": "issues_found" if has_issues else "ok",
                "analysis": analysis.to_dict(),
                "recommendations": self._generate_recommendations(analysis),
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

        # Query section_briefs
        briefs = project.query_artifacts(
            artifact_type="section_brief",
            limit=1000,
        )

        # Query sections
        sections = project.query_artifacts(
            artifact_type="section",
            limit=1000,
        )

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

            # Extract outgoing connections from choice_intents
            choice_intents = brief.get("choice_intents", [])
            for choice in choice_intents:
                target = choice.get("target_anchor")
                if target:
                    node.outgoing.append(target)

            analysis.nodes[anchor] = node

        # Build nodes from sections (may override briefs)
        for section in sections:
            lifecycle = section.get("_lifecycle_state", "draft")
            if not include_drafts and lifecycle == "draft":
                continue

            section_anchor: str = section.get("anchor_id") or section.get("_id") or ""
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
                # Merge outgoing connections
                section_choices = section.get("choices", [])
                for choice in section_choices:
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
                section_choices = section.get("choices", [])
                for choice in section_choices:
                    target = choice.get("target_anchor")
                    if target:
                        node.outgoing.append(target)
                analysis.nodes[section_anchor] = node

        # Build incoming connections
        for anchor, node in analysis.nodes.items():
            for target in node.outgoing:
                if target in analysis.nodes:
                    analysis.nodes[target].incoming.append(anchor)

        # Identify entry points
        if entry_anchor and entry_anchor in analysis.nodes:
            analysis.entry_points = [entry_anchor]
        else:
            # Entry points are nodes with no incoming connections
            for anchor, node in analysis.nodes.items():
                if not node.incoming:
                    analysis.entry_points.append(anchor)

        # Compute reachability via BFS from entry points
        analysis.reachable = set()
        queue = list(analysis.entry_points)
        while queue:
            current = queue.pop(0)
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

        # Find dead ends (no outgoing, not counting terminal sections)
        for anchor, node in analysis.nodes.items():
            if not node.outgoing:
                analysis.dead_ends.append(anchor)

        # Find orphans (no incoming except entry points)
        for anchor, node in analysis.nodes.items():
            if not node.incoming and anchor not in analysis.entry_points:
                analysis.orphans.append(anchor)

        # Find missing targets
        all_targets = set()
        for node in analysis.nodes.values():
            all_targets.update(node.outgoing)
        for target in all_targets:
            if target not in analysis.nodes:
                # Find which nodes reference this missing target
                for anchor, node in analysis.nodes.items():
                    if target in node.outgoing:
                        analysis.missing_targets.append(
                            {
                                "from": anchor,
                                "to": target,
                            }
                        )

        # Group by lifecycle
        by_lifecycle: dict[str, list[str]] = defaultdict(list)
        for anchor, node in analysis.nodes.items():
            by_lifecycle[node.lifecycle_state].append(anchor)
        analysis.by_lifecycle = dict(by_lifecycle)

        # Check if stable paths exist (all approved/cold)
        stable_states = {"approved", "cold"}
        stable_nodes = {a for a, n in analysis.nodes.items() if n.lifecycle_state in stable_states}
        # Check if entry can reach at least one stable path
        analysis.stable_paths = bool(analysis.reachable & stable_nodes)

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

    def _generate_recommendations(self, analysis: GraphAnalysis) -> list[str]:
        """Generate actionable recommendations from analysis."""
        recommendations = []

        if analysis.unreachable:
            recommendations.append(
                f"Add paths to {len(analysis.unreachable)} unreachable node(s): "
                f"{', '.join(sorted(analysis.unreachable)[:3])}"
                + ("..." if len(analysis.unreachable) > 3 else "")
            )

        if analysis.missing_targets:
            missing = {m["to"] for m in analysis.missing_targets}
            recommendations.append(
                f"Create missing target(s): {', '.join(sorted(missing)[:3])}"
                + ("..." if len(missing) > 3 else "")
            )

        if not analysis.entry_points:
            recommendations.append("No entry point found - designate a starting section")

        if len(analysis.entry_points) > 1:
            recommendations.append(
                f"Multiple entry points detected: {', '.join(analysis.entry_points)}. "
                "Verify this is intentional."
            )

        draft_count = len(analysis.by_lifecycle.get("draft", []))
        if draft_count > 0 and not analysis.stable_paths:
            recommendations.append(
                f"All {draft_count} reachable nodes are drafts. "
                "Promote some to approved for stable paths."
            )

        if not recommendations:
            recommendations.append("Story graph structure is valid.")

        return recommendations
