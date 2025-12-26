"""
Analyze Story Graph tool implementation.

Programmatically analyzes the story topology to check:
1. Reachability - Can all passages be reached from the start passage?
2. Dead ends - Passages with no outgoing links (except endings)
3. Orphans - Passages with no incoming links (except start)
4. Missing targets - Links to non-existent passages
5. Lifecycle status - What state are the passage artifacts in?
6. Alignment - Do passage_brief and passage artifacts match topology?

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
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.storage.project import Project


@dataclass
class PassageInfo:
    """Information about a passage in the story graph."""

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

    # Topology identity (for IFID mismatch detection)
    topology_ifid: str | None = None

    # All nodes in the graph (from topology)
    nodes: dict[str, PassageInfo] = field(default_factory=dict)

    # Reachability from start
    reachable: set[str] = field(default_factory=set)
    unreachable: set[str] = field(default_factory=set)

    # Entry point detection
    potential_entry_points: list[str] = field(default_factory=list)  # orphan pids

    # Structural issues
    dead_ends: list[str] = field(default_factory=list)  # no outgoing (non-endings)
    endings: list[str] = field(default_factory=list)  # intentional endings
    orphans: list[str] = field(default_factory=list)  # no incoming (except start)
    missing_targets: list[dict[str, str]] = field(default_factory=list)  # {from, to}
    duplicate_pids: list[str] = field(default_factory=list)  # duplicate pids in topology

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

        # Build the graph from topology using helper methods
        analysis = GraphAnalysis()
        project = self._context.project

        self._load_story_info(analysis, project)
        topology = self._load_topology(analysis, project)

        if topology:
            self._build_nodes_from_topology(analysis, topology)
            self._build_edges_from_topology(analysis, topology)
            effective_start = self._determine_effective_start(analysis, entry_pid)
            self._compute_reachability(analysis, effective_start)
            self._find_structural_issues(analysis, effective_start)
            self._align_artifacts(analysis, project, include_drafts)
            self._check_stable_passages(analysis)

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

    def _load_story_info(self, analysis: GraphAnalysis, project: Project) -> None:
        """Load story artifact for IFID and start passage."""
        stories = project.query_artifacts(artifact_type="story")
        if stories:
            story = stories[0]
            analysis.ifid = story.get("ifid")
            analysis.story_name = story.get("name")
            analysis.start_pid = story.get("start")

    def _load_topology(self, analysis: GraphAnalysis, project: Project) -> dict[str, Any] | None:
        """Load topology artifact (authoritative graph structure)."""
        topologies = project.query_artifacts(artifact_type="topology")
        if not topologies:
            return None

        topology = topologies[0]
        analysis.topology_ifid = topology.get("ifid")
        return topology

    def _build_nodes_from_topology(self, analysis: GraphAnalysis, topology: dict[str, Any]) -> None:
        """Build nodes from topology.passages, detecting duplicates."""
        topology_passages = topology.get("passages", [])
        seen_pids: set[str] = set()

        for passage in topology_passages:
            pid = passage.get("pid")
            if not pid:
                continue

            # Detect duplicate pids
            if pid in seen_pids and pid not in analysis.duplicate_pids:
                analysis.duplicate_pids.append(pid)
            seen_pids.add(pid)

            node = PassageInfo(
                pid=pid,
                name=passage.get("name", "Untitled"),
                topology_role=passage.get("topology_role"),
                is_ending=passage.get("is_ending", False),
                tags=passage.get("tags", []),
            )
            analysis.nodes[pid] = node

        # Build endings from final node state (handles duplicates correctly)
        analysis.endings = [pid for pid, node in analysis.nodes.items() if node.is_ending]

    def _build_edges_from_topology(self, analysis: GraphAnalysis, topology: dict[str, Any]) -> None:
        """Build edges from topology.links."""
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

    def _determine_effective_start(
        self, analysis: GraphAnalysis, entry_pid_override: str | None
    ) -> str | None:
        """Determine entry point, tracking potential alternatives."""
        # Explicit override
        if entry_pid_override and entry_pid_override in analysis.nodes:
            return entry_pid_override

        # From story.start
        if analysis.start_pid and analysis.start_pid in analysis.nodes:
            return analysis.start_pid

        # Fallback: find all nodes with no incoming connections
        orphan_pids = [pid for pid, node in analysis.nodes.items() if not node.incoming]
        analysis.potential_entry_points = orphan_pids

        # Return first orphan if any exist
        return orphan_pids[0] if orphan_pids else None

    def _compute_reachability(self, analysis: GraphAnalysis, effective_start: str | None) -> None:
        """Compute reachability via BFS from start."""
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

    def _find_structural_issues(self, analysis: GraphAnalysis, effective_start: str | None) -> None:
        """Find dead ends and orphans."""
        # Find dead ends (no outgoing, excluding endings)
        for pid, node in analysis.nodes.items():
            if not node.outgoing and not node.is_ending:
                analysis.dead_ends.append(pid)

        # Find orphans (no incoming except start)
        for pid, node in analysis.nodes.items():
            if not node.incoming and pid != effective_start:
                analysis.orphans.append(pid)

    def _align_artifacts(
        self,
        analysis: GraphAnalysis,
        project: Project,
        include_drafts: bool,
    ) -> None:
        """Align passage_brief and passage artifacts with topology."""
        topology_pids = set(analysis.nodes.keys())

        # Track which pids are included based on include_drafts setting
        included_brief_pids, included_passage_pids = self._align_artifact_type(
            analysis, project, "passage_brief", include_drafts
        )
        passage_brief_pids, passage_passage_pids = self._align_artifact_type(
            analysis, project, "passage", include_drafts
        )

        # Merge passage alignment results
        included_passage_pids = passage_passage_pids

        # Check for missing/extra artifacts
        # When include_drafts=False, only compare against included artifacts
        if include_drafts:
            # Compare against all topology pids
            analysis.briefs_missing = sorted(topology_pids - included_brief_pids)
            analysis.passages_missing = sorted(topology_pids - included_passage_pids)
        else:
            # When excluding drafts, missing artifacts are only those not covered
            # by any non-draft artifact
            analysis.briefs_missing = sorted(topology_pids - included_brief_pids)
            analysis.passages_missing = sorted(topology_pids - included_passage_pids)

        analysis.briefs_extra = sorted(included_brief_pids - topology_pids)
        analysis.passages_extra = sorted(included_passage_pids - topology_pids)

    def _align_artifact_type(
        self,
        analysis: GraphAnalysis,
        project: Project,
        artifact_type: str,
        include_drafts: bool,
    ) -> tuple[set[str], set[str]]:
        """
        Align a specific artifact type with topology.

        Returns (included_pids, all_pids) where included_pids respects
        the include_drafts setting.
        """
        artifacts = project.query_artifacts(artifact_type=artifact_type)
        included_pids: set[str] = set()
        all_pids: set[str] = set()
        by_lifecycle: dict[str, list[str]] = defaultdict(list)

        for artifact in artifacts:
            lifecycle = artifact.get("_lifecycle_state", "draft")

            # Get the pid this artifact references (prefer topology_passage_ref)
            if artifact_type == "passage_brief":
                artifact_pid = artifact.get("topology_passage_ref") or artifact.get("target")
            else:  # passage
                artifact_pid = artifact.get("topology_passage_ref") or artifact.get("pid")

            if not artifact_pid:
                continue

            all_pids.add(artifact_pid)

            # Only include in analysis if respecting include_drafts
            if include_drafts or lifecycle != "draft":
                included_pids.add(artifact_pid)
                by_lifecycle[lifecycle].append(artifact_pid)

                # Link to node if exists
                if artifact_pid in analysis.nodes:
                    if artifact_type == "passage_brief":
                        analysis.nodes[artifact_pid].brief_artifact_id = artifact.get("_id")
                        analysis.nodes[artifact_pid].brief_lifecycle_state = lifecycle
                    else:
                        analysis.nodes[artifact_pid].passage_artifact_id = artifact.get("_id")
                        analysis.nodes[artifact_pid].passage_lifecycle_state = lifecycle

        # Store lifecycle breakdown
        if artifact_type == "passage_brief":
            analysis.by_brief_lifecycle = dict(by_lifecycle)
        else:
            analysis.by_passage_lifecycle = dict(by_lifecycle)

        return included_pids, all_pids

    def _check_stable_passages(self, analysis: GraphAnalysis) -> None:
        """Check if any reachable passage is in stable state."""
        stable_states = {"review", "approved", "cold"}
        for pid in analysis.reachable:
            node_info = analysis.nodes.get(pid)
            if node_info is not None and node_info.passage_lifecycle_state in stable_states:
                analysis.has_reachable_stable_passage = True
                break

    def _categorize_issues(
        self, analysis: GraphAnalysis
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Categorize issues into blocking vs warnings.

        Blocking issues prevent graph approval:
        - Missing targets (broken links)
        - No topology (empty graph)
        - No start passage defined
        - IFID mismatch between story and topology

        Warnings are concerns but don't block:
        - Unreachable passages
        - Dead ends (excluding endings)
        - Orphans
        - Missing briefs/passages
        - Extra briefs/passages not in topology
        - All drafts (no stable paths)
        - Multiple potential entry points
        - Duplicate pids in topology
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

        # WARNING: IFID mismatch between story and topology
        if analysis.ifid and analysis.topology_ifid and analysis.ifid != analysis.topology_ifid:
            warnings.append(
                {
                    "issue_type": "ifid_mismatch",
                    "severity": "warning",
                    "description": (
                        f"Story IFID '{analysis.ifid}' does not match "
                        f"topology IFID '{analysis.topology_ifid}'"
                    ),
                    "affected_nodes": [],
                    "count": 1,
                }
            )

        # WARNING: Multiple potential entry points (ambiguous start)
        if len(analysis.potential_entry_points) > 1 and not analysis.start_pid:
            warnings.append(
                {
                    "issue_type": "multiple_entry_points",
                    "severity": "warning",
                    "description": (
                        f"{len(analysis.potential_entry_points)} passages have no incoming links. "
                        "Define story.start to specify the entry point explicitly."
                    ),
                    "affected_nodes": analysis.potential_entry_points[:5],
                    "count": len(analysis.potential_entry_points),
                }
            )

        # WARNING: Duplicate pids in topology
        if analysis.duplicate_pids:
            warnings.append(
                {
                    "issue_type": "duplicate_pids",
                    "severity": "warning",
                    "description": (
                        f"{len(analysis.duplicate_pids)} duplicate pid(s) in topology - "
                        "later entries overwrite earlier ones"
                    ),
                    "affected_nodes": analysis.duplicate_pids[:5],
                    "count": len(analysis.duplicate_pids),
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

        # WARNING: Missing passages
        if analysis.passages_missing:
            warnings.append(
                {
                    "issue_type": "passages_missing",
                    "severity": "warning",
                    "description": f"{len(analysis.passages_missing)} topology passage(s) have no passage artifact",
                    "affected_nodes": analysis.passages_missing[:5],
                    "count": len(analysis.passages_missing),
                }
            )

        # WARNING: Extra passages not in topology
        if analysis.passages_extra:
            warnings.append(
                {
                    "issue_type": "passages_extra",
                    "severity": "warning",
                    "description": f"{len(analysis.passages_extra)} passage(s) reference pids not in topology",
                    "affected_nodes": analysis.passages_extra[:5],
                    "count": len(analysis.passages_extra),
                }
            )

        # WARNING: No stable passages (check stable states properly)
        if analysis.reachable and not analysis.has_reachable_stable_passage:
            stable_states = ("review", "approved", "cold")
            stable_count = sum(
                len(analysis.by_passage_lifecycle.get(state, [])) for state in stable_states
            )
            if stable_count == 0:
                # Count total passages
                total_passages = sum(
                    len(pids) for pids in analysis.by_passage_lifecycle.values()
                ) or len(analysis.nodes)

                # Collect example passages
                affected_nodes: list[str] = []
                for pids in analysis.by_passage_lifecycle.values():
                    for pid in pids:
                        if len(affected_nodes) >= 5:
                            break
                        affected_nodes.append(pid)
                    if len(affected_nodes) >= 5:
                        break

                warnings.append(
                    {
                        "issue_type": "no_stable_passages",
                        "severity": "warning",
                        "description": (
                            f"No passages are in a stable lifecycle state "
                            f"(review, approved, or cold) - {total_passages} passage(s) in other states"
                        ),
                        "affected_nodes": affected_nodes,
                        "count": total_passages,
                    }
                )

        # WARNING: No endings defined
        if not analysis.endings and analysis.nodes:
            warnings.append(
                {
                    "issue_type": "no_endings",
                    "severity": "warning",
                    "description": (
                        "No passages explicitly marked as endings "
                        "(set is_ending: true; passages default to is_ending: false)"
                    ),
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
