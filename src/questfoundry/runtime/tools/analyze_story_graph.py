"""
Analyze Story Graph tool implementation.

Programmatically analyzes the story topology to check:
1. Reachability - Can all passages be reached from the start passage?
2. Dead ends - Passages with no outgoing links (except endings)
3. Orphans - Passages with no incoming links (except start)
4. Missing targets - Links to non-existent passages
5. Lifecycle status - What state are the passage artifacts in?

This tool reads from the authoritative `topology` artifact and validates
alignment with `passage_brief` and `passage` artifacts.

Architecture (post #284):
- `story` artifact defines `start` (entry point pid)
- `topology` artifact defines graph structure (passages + links)
- `passage_brief` and `passage` artifacts reference topology via `topology_passage_ref`
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@dataclass
class NodeInfo:
    """Information about a node (passage) in the story graph."""

    pid: str  # Passage identifier
    name: str  # Human-readable title
    topology_role: str | None  # hub, loop, gateway, linear
    is_ending: bool
    tags: list[str]
    # Artifact alignment
    brief_artifact_id: str | None = None
    brief_lifecycle_state: str | None = None
    passage_artifact_id: str | None = None
    passage_lifecycle_state: str | None = None
    # Graph connections (from topology.links)
    outgoing: list[str] = field(default_factory=list)  # target pids
    incoming: list[str] = field(default_factory=list)  # source pids


@dataclass
class GraphAnalysis:
    """Result of story graph analysis."""

    # Story identity
    ifid: str | None = None
    story_name: str | None = None
    start_pid: str | None = None

    # All nodes in the graph (from topology)
    nodes: dict[str, NodeInfo] = field(default_factory=dict)

    # Reachability from start
    reachable: set[str] = field(default_factory=set)
    unreachable: set[str] = field(default_factory=set)

    # Structural issues
    dead_ends: list[str] = field(default_factory=list)  # no outgoing (non-endings)
    endings: list[str] = field(default_factory=list)  # intentional endings
    orphans: list[str] = field(default_factory=list)  # no incoming (except start)
    missing_targets: list[dict[str, str]] = field(default_factory=list)  # {from, to}

    # Artifact alignment
    briefs_missing: list[str] = field(default_factory=list)  # pids without briefs
    passages_missing: list[str] = field(default_factory=list)  # pids without passages
    briefs_extra: list[str] = field(default_factory=list)  # briefs not in topology
    passages_extra: list[str] = field(default_factory=list)  # passages not in topology

    # Lifecycle breakdown
    by_brief_lifecycle: dict[str, list[str]] = field(default_factory=dict)
    by_passage_lifecycle: dict[str, list[str]] = field(default_factory=dict)

    # Whether any reachable passage is in a stable state
    has_reachable_stable_passage: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for tool result."""
        return {
            "story": {
                "ifid": self.ifid,
                "name": self.story_name,
                "start": self.start_pid,
            },
            "summary": {
                "total_passages": len(self.nodes),
                "reachable_count": len(self.reachable),
                "unreachable_count": len(self.unreachable),
                "dead_end_count": len(self.dead_ends),
                "ending_count": len(self.endings),
                "orphan_count": len(self.orphans),
                "missing_target_count": len(self.missing_targets),
                "has_reachable_stable_passage": self.has_reachable_stable_passage,
            },
            "reachability": {
                "reachable": sorted(self.reachable),
                "unreachable": sorted(self.unreachable),
            },
            "structural": {
                "dead_ends": self.dead_ends,
                "endings": self.endings,
                "orphans": self.orphans,
                "missing_targets": self.missing_targets,
            },
            "alignment": {
                "briefs_missing": self.briefs_missing,
                "passages_missing": self.passages_missing,
                "briefs_extra": self.briefs_extra,
                "passages_extra": self.passages_extra,
            },
            "lifecycle": {
                "by_brief": self.by_brief_lifecycle,
                "by_passage": self.by_passage_lifecycle,
            },
            "nodes": {
                pid: {
                    "pid": node.pid,
                    "name": node.name,
                    "topology_role": node.topology_role,
                    "is_ending": node.is_ending,
                    "tags": node.tags,
                    "brief_artifact_id": node.brief_artifact_id,
                    "brief_lifecycle": node.brief_lifecycle_state,
                    "passage_artifact_id": node.passage_artifact_id,
                    "passage_lifecycle": node.passage_lifecycle_state,
                    "outgoing": node.outgoing,
                    "incoming": node.incoming,
                }
                for pid, node in self.nodes.items()
            },
        }


@register_tool("analyze_story_graph")
class AnalyzeStoryGraphTool(BaseTool):
    """
    Analyze story topology for reachability and structural issues.

    Reads the authoritative `topology` artifact for graph structure,
    uses `story.start` as the entry point, and validates alignment
    with `passage_brief` and `passage` artifacts.

    Returns structured data about:
    - Which passages are reachable from start
    - Dead ends (no way forward, excluding intentional endings)
    - Orphans (no way to reach them)
    - Missing targets (broken links)
    - Artifact alignment (briefs/passages vs topology)
    - Lifecycle status of artifacts
    """

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute story graph analysis."""
        include_drafts = args.get("include_drafts", True)
        entry_pid = args.get("entry_pid")  # Optional override for entry

        if not self._context.project:
            return ToolResult(
                success=False,
                data={},
                error="No project available - cannot analyze story graph",
            )

        # Build the graph from topology
        analysis = self._build_and_analyze_graph(include_drafts, entry_pid)

        # Categorize issues by severity
        blocking_issues, warnings = self._categorize_issues(analysis)

        # Determine analysis outcome
        if blocking_issues:
            analysis_outcome = "failed"
        elif warnings:
            analysis_outcome = "warnings"
        else:
            analysis_outcome = "passed"

        # Generate directive recovery actions
        recovery_actions = self._generate_recovery_actions(blocking_issues, analysis)

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
        entry_pid: str | None,
    ) -> GraphAnalysis:
        """Build graph from topology artifact and analyze."""
        analysis = GraphAnalysis()
        project = self._context.project
        assert project is not None  # Checked in execute()

        # 1. Load story artifact for IFID and start passage
        stories = project.query_artifacts(artifact_type="story")
        if stories:
            story = stories[0]
            analysis.ifid = story.get("ifid")
            analysis.story_name = story.get("name")
            analysis.start_pid = story.get("start")

        # 2. Load topology artifact (authoritative graph structure)
        topologies = project.query_artifacts(artifact_type="topology")
        if not topologies:
            # No topology yet - return empty analysis
            return analysis

        topology = topologies[0]
        topology_ifid = topology.get("ifid")

        # Verify topology matches story
        if analysis.ifid and topology_ifid and analysis.ifid != topology_ifid:
            # Mismatch - this is a data integrity issue
            pass  # Will be caught as a warning

        # 3. Build nodes from topology.passages
        topology_passages = topology.get("passages", [])
        topology_pids: set[str] = set()

        for passage in topology_passages:
            pid = passage.get("pid")
            if not pid:
                continue

            topology_pids.add(pid)
            node = NodeInfo(
                pid=pid,
                name=passage.get("name", "Untitled"),
                topology_role=passage.get("topology_role"),
                is_ending=passage.get("is_ending", False),
                tags=passage.get("tags", []),
            )
            analysis.nodes[pid] = node

            # Track endings
            if node.is_ending:
                analysis.endings.append(pid)

        # 4. Build edges from topology.links
        topology_links = topology.get("links", [])
        for link in topology_links:
            from_pid = link.get("from")
            to_pid = link.get("to")

            if not from_pid or not to_pid:
                continue

            # Add outgoing connection
            if from_pid in analysis.nodes and to_pid not in analysis.nodes[from_pid].outgoing:
                analysis.nodes[from_pid].outgoing.append(to_pid)

            # Check for missing targets
            if to_pid not in analysis.nodes:
                analysis.missing_targets.append({"from": from_pid, "to": to_pid})
            else:
                # Add incoming connection
                if from_pid not in analysis.nodes[to_pid].incoming:
                    analysis.nodes[to_pid].incoming.append(from_pid)

        # 5. Determine entry point
        if entry_pid and entry_pid in analysis.nodes:
            # Explicit override
            effective_start = entry_pid
        elif analysis.start_pid and analysis.start_pid in analysis.nodes:
            # From story.start
            effective_start = analysis.start_pid
        else:
            # Fallback: find nodes with no incoming connections
            effective_start = None
            for pid, node in analysis.nodes.items():
                if not node.incoming:
                    effective_start = pid
                    break

        # 6. Compute reachability via BFS from start
        analysis.reachable = set()
        if effective_start:
            queue: deque[str] = deque([effective_start])
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
        all_pids = set(analysis.nodes.keys())
        analysis.unreachable = all_pids - analysis.reachable

        # 7. Find dead ends (no outgoing, excluding endings)
        for pid, node in analysis.nodes.items():
            if not node.outgoing and not node.is_ending:
                analysis.dead_ends.append(pid)

        # 8. Find orphans (no incoming except start)
        for pid, node in analysis.nodes.items():
            if not node.incoming and pid != effective_start:
                analysis.orphans.append(pid)

        # 9. Load passage_briefs and check alignment
        briefs = project.query_artifacts(artifact_type="passage_brief")
        brief_pids: set[str] = set()
        by_brief_lifecycle: dict[str, list[str]] = defaultdict(list)

        for brief in briefs:
            lifecycle = brief.get("_lifecycle_state", "draft")
            if not include_drafts and lifecycle == "draft":
                continue

            # Get the pid this brief references
            brief_pid = brief.get("target") or brief.get("topology_passage_ref")
            if brief_pid:
                brief_pids.add(brief_pid)
                by_brief_lifecycle[lifecycle].append(brief_pid)

                # Link to node if exists
                if brief_pid in analysis.nodes:
                    analysis.nodes[brief_pid].brief_artifact_id = brief.get("_id")
                    analysis.nodes[brief_pid].brief_lifecycle_state = lifecycle

        analysis.by_brief_lifecycle = dict(by_brief_lifecycle)

        # Check for missing/extra briefs
        analysis.briefs_missing = sorted(topology_pids - brief_pids)
        analysis.briefs_extra = sorted(brief_pids - topology_pids)

        # 10. Load passages and check alignment
        passages = project.query_artifacts(artifact_type="passage")
        passage_pids: set[str] = set()
        by_passage_lifecycle: dict[str, list[str]] = defaultdict(list)

        for passage in passages:
            lifecycle = passage.get("_lifecycle_state", "draft")
            if not include_drafts and lifecycle == "draft":
                continue

            # Get the pid this passage implements
            passage_pid = passage.get("pid") or passage.get("topology_passage_ref")
            if passage_pid:
                passage_pids.add(passage_pid)
                by_passage_lifecycle[lifecycle].append(passage_pid)

                # Link to node if exists
                if passage_pid in analysis.nodes:
                    analysis.nodes[passage_pid].passage_artifact_id = passage.get("_id")
                    analysis.nodes[passage_pid].passage_lifecycle_state = lifecycle

        analysis.by_passage_lifecycle = dict(by_passage_lifecycle)

        # Check for missing/extra passages
        analysis.passages_missing = sorted(topology_pids - passage_pids)
        analysis.passages_extra = sorted(passage_pids - topology_pids)

        # 11. Check if any reachable passage is in stable state
        stable_states = {"review", "approved", "cold"}
        for pid in analysis.reachable:
            node_info = analysis.nodes.get(pid)
            if node_info is not None and node_info.passage_lifecycle_state in stable_states:
                analysis.has_reachable_stable_passage = True
                break

        return analysis

    def _categorize_issues(
        self, analysis: GraphAnalysis
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Categorize issues into blocking vs warnings.

        Blocking issues prevent graph approval:
        - Missing targets (broken links)
        - No topology (empty graph)
        - No start passage defined

        Warnings are concerns but don't block:
        - Unreachable passages
        - Dead ends (excluding endings)
        - Orphans
        - Missing briefs/passages
        - Extra briefs/passages not in topology
        - All drafts (no stable paths)
        """
        blocking: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        # BLOCKING: No topology
        if not analysis.nodes:
            blocking.append(
                {
                    "issue_type": "no_topology",
                    "severity": "blocking",
                    "description": "No topology artifact found - create topology first",
                    "affected_nodes": [],
                    "count": 0,
                }
            )
            return blocking, warnings  # No point checking further

        # BLOCKING: No start passage
        if not analysis.start_pid:
            blocking.append(
                {
                    "issue_type": "no_start_passage",
                    "severity": "blocking",
                    "description": "No start passage defined in story artifact",
                    "affected_nodes": [],
                    "count": 0,
                }
            )
        elif analysis.start_pid not in analysis.nodes:
            blocking.append(
                {
                    "issue_type": "invalid_start_passage",
                    "severity": "blocking",
                    "description": f"Start passage '{analysis.start_pid}' not found in topology",
                    "affected_nodes": [analysis.start_pid],
                    "count": 1,
                }
            )

        # BLOCKING: Missing targets (broken links)
        if analysis.missing_targets:
            missing = {m["to"] for m in analysis.missing_targets}
            blocking.append(
                {
                    "issue_type": "missing_targets",
                    "severity": "blocking",
                    "description": f"{len(missing)} link target(s) reference non-existent passages",
                    "affected_nodes": sorted(missing)[:5],
                    "count": len(missing),
                }
            )

        # WARNING: Unreachable passages
        if analysis.unreachable:
            warnings.append(
                {
                    "issue_type": "unreachable_passages",
                    "severity": "warning",
                    "description": f"{len(analysis.unreachable)} passage(s) cannot be reached from start",
                    "affected_nodes": sorted(analysis.unreachable)[:5],
                    "count": len(analysis.unreachable),
                }
            )

        # WARNING: Dead ends (excluding intentional endings)
        if analysis.dead_ends:
            warnings.append(
                {
                    "issue_type": "dead_ends",
                    "severity": "warning",
                    "description": f"{len(analysis.dead_ends)} passage(s) have no outgoing links and are not marked as endings",
                    "affected_nodes": sorted(analysis.dead_ends)[:5],
                    "count": len(analysis.dead_ends),
                }
            )

        # WARNING: Orphans
        if analysis.orphans:
            warnings.append(
                {
                    "issue_type": "orphan_passages",
                    "severity": "warning",
                    "description": f"{len(analysis.orphans)} passage(s) have no incoming links",
                    "affected_nodes": sorted(analysis.orphans)[:5],
                    "count": len(analysis.orphans),
                }
            )

        # WARNING: Missing passage_briefs
        if analysis.briefs_missing:
            warnings.append(
                {
                    "issue_type": "briefs_missing",
                    "severity": "warning",
                    "description": f"{len(analysis.briefs_missing)} topology passage(s) have no passage_brief",
                    "affected_nodes": analysis.briefs_missing[:5],
                    "count": len(analysis.briefs_missing),
                }
            )

        # WARNING: Extra passage_briefs not in topology
        if analysis.briefs_extra:
            warnings.append(
                {
                    "issue_type": "briefs_extra",
                    "severity": "warning",
                    "description": f"{len(analysis.briefs_extra)} passage_brief(s) reference pids not in topology",
                    "affected_nodes": analysis.briefs_extra[:5],
                    "count": len(analysis.briefs_extra),
                }
            )

        # WARNING: No stable passages
        if analysis.reachable and not analysis.has_reachable_stable_passage:
            draft_count = len(analysis.by_passage_lifecycle.get("draft", []))
            if draft_count > 0:
                warnings.append(
                    {
                        "issue_type": "no_stable_passages",
                        "severity": "warning",
                        "description": f"All {draft_count} passage(s) are drafts - none promoted to review/cold",
                        "affected_nodes": analysis.by_passage_lifecycle.get("draft", [])[:5],
                        "count": draft_count,
                    }
                )

        # WARNING: No endings defined
        if not analysis.endings and analysis.nodes:
            warnings.append(
                {
                    "issue_type": "no_endings",
                    "severity": "warning",
                    "description": "No passages marked as endings (is_ending: true)",
                    "affected_nodes": [],
                    "count": 0,
                }
            )

        return blocking, warnings

    def _generate_recovery_actions(
        self,
        blocking_issues: list[dict[str, Any]],
        analysis: GraphAnalysis,
    ) -> list[dict[str, str]]:
        """
        Generate directive recovery actions (not passive hints).

        Each action tells the agent exactly what to do next.
        """
        actions: list[dict[str, str]] = []

        # Actions for blocking issues first (highest priority)
        for issue in blocking_issues:
            if issue["issue_type"] == "no_topology":
                actions.append(
                    {
                        "priority": "high",
                        "action": "Create topology artifact",
                        "details": "Run story_spark playbook to create story and topology artifacts",
                    }
                )
            elif issue["issue_type"] == "no_start_passage":
                actions.append(
                    {
                        "priority": "high",
                        "action": "Set start passage in story artifact",
                        "details": "Update story artifact with 'start' field pointing to entry passage pid",
                    }
                )
            elif issue["issue_type"] == "invalid_start_passage":
                pid = issue["affected_nodes"][0] if issue["affected_nodes"] else "unknown"
                actions.append(
                    {
                        "priority": "high",
                        "action": "Fix start passage reference",
                        "details": f"Story.start references '{pid}' which doesn't exist in topology. "
                        f"Valid pids: {', '.join(sorted(analysis.nodes.keys())[:5])}",
                    }
                )
            elif issue["issue_type"] == "missing_targets":
                nodes = issue["affected_nodes"]
                actions.append(
                    {
                        "priority": "high",
                        "action": "Add missing passages to topology",
                        "details": f"Add passages to topology for: {', '.join(nodes[:3])}"
                        + ("..." if len(nodes) > 3 else ""),
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
