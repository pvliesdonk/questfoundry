"""Project inspection and quality analysis.

Automates the manual quality review checks performed on completed projects.
Pure graph analysis — no LLM calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from questfoundry.graph.fill_context import compute_lexical_diversity
from questfoundry.graph.fill_validation import path_has_prose
from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import find_max_consecutive_linear, run_all_checks
from questfoundry.observability.logging import get_logger
from questfoundry.pipeline.config import ProjectConfigError, load_project_config

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


@dataclass
class GraphSummary:
    """High-level graph statistics."""

    project_name: str
    last_stage: str | None
    total_nodes: int
    total_edges: int
    node_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ProseStats:
    """Prose quality metrics across all passages."""

    total_passages: int
    passages_with_prose: int
    flagged_passages: list[dict[str, str]] = field(default_factory=list)
    total_words: int = 0
    avg_words: float = 0.0
    min_words: int = 0
    max_words: int = 0
    lexical_diversity: float | None = None


@dataclass
class BranchingStats:
    """Branching structure metrics."""

    total_choices: int = 0
    meaningful_choices: int = 0
    contextual_choices: int = 0
    continue_choices: int = 0
    max_consecutive_linear: int = 0
    total_dilemmas: int = 0
    fully_explored: int = 0
    partially_explored: int = 0
    start_passages: int = 0
    ending_passages: int = 0


@dataclass
class CoverageStats:
    """Entity, codex, and illustration coverage."""

    entity_count: int = 0
    entity_types: dict[str, int] = field(default_factory=dict)
    codex_entries: int = 0
    entities_with_codex: int = 0
    illustration_briefs: int = 0
    illustration_nodes: int = 0
    asset_files: int = 0


@dataclass
class InspectionReport:
    """Complete project inspection report."""

    summary: GraphSummary
    prose: ProseStats | None = None
    branching: BranchingStats | None = None
    coverage: CoverageStats = field(default_factory=CoverageStats)
    validation_checks: list[dict[str, str]] = field(default_factory=list)


def inspect_project(project_path: Path) -> InspectionReport:
    """Run all inspection checks on a project.

    Args:
        project_path: Path to the project root directory.

    Returns:
        InspectionReport with all analysis results.
    """
    graph = Graph.load(project_path)

    summary = _graph_summary(graph, project_path)
    prose = _prose_stats(graph)
    branching = _branching_stats(graph)
    coverage = _coverage_stats(graph, project_path)
    validation = _run_validation(graph)

    log.info(
        "inspection_complete",
        project=summary.project_name,
        nodes=summary.total_nodes,
        edges=summary.total_edges,
    )

    return InspectionReport(
        summary=summary,
        prose=prose,
        branching=branching,
        coverage=coverage,
        validation_checks=validation,
    )


def _graph_summary(graph: Graph, project_path: Path) -> GraphSummary:
    """Extract high-level graph statistics."""
    data = graph.to_dict()
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    node_counts: dict[str, int] = Counter(n.get("type", "unknown") for n in nodes.values())

    project_name = graph.get_project_name() or ""
    if not project_name:
        try:
            config = load_project_config(project_path)
            project_name = config.name
        except (ProjectConfigError, FileNotFoundError):
            project_name = project_path.name

    return GraphSummary(
        project_name=project_name,
        last_stage=graph.get_last_stage(),
        total_nodes=len(nodes),
        total_edges=len(edges),
        node_counts=dict(sorted(node_counts.items(), key=lambda x: -x[1])),
    )


def _prose_stats(graph: Graph) -> ProseStats | None:
    """Analyze prose quality across all passages."""
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return None

    word_counts: list[int] = []
    prose_texts: list[str] = []
    flagged: list[dict[str, str]] = []

    for pid, data in passages.items():
        prose = data.get("prose")
        if not prose or not str(prose).strip():
            entry: dict[str, str] = {"id": pid}
            if data.get("flag"):
                entry["flag"] = data["flag"]
            if data.get("flag_reason"):
                entry["flag_reason"] = data["flag_reason"]
            flagged.append(entry)
        else:
            text = str(prose)
            wc = len(text.split())
            word_counts.append(wc)
            prose_texts.append(text)

    diversity = None
    if len(prose_texts) >= 2:
        diversity = round(compute_lexical_diversity(prose_texts), 3)

    return ProseStats(
        total_passages=len(passages),
        passages_with_prose=len(word_counts),
        flagged_passages=flagged,
        total_words=sum(word_counts),
        avg_words=round(sum(word_counts) / len(word_counts), 1) if word_counts else 0.0,
        min_words=min(word_counts) if word_counts else 0,
        max_words=max(word_counts) if word_counts else 0,
        lexical_diversity=diversity,
    )


def _branching_stats(graph: Graph) -> BranchingStats | None:
    """Analyze branching structure."""
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return None

    choices = graph.get_nodes_by_type("choice")

    # Classify by graph structure: count outgoing choices per source passage.
    # choice_from edges point choice→source_passage, so e["to"] = source passage.
    choice_from_edges_all = graph.get_edges(edge_type="choice_from")
    outgoing_per_passage: dict[str, int] = Counter(e["to"] for e in choice_from_edges_all)

    meaningful = 0
    contextual = 0
    continue_count = 0
    for _cid, cdata in choices.items():
        from_passage = cdata.get("from_passage", "")
        outgoing = outgoing_per_passage.get(from_passage, 1)
        if outgoing >= 2:
            meaningful += 1
        elif cdata.get("label", "continue") == "continue":
            continue_count += 1
        else:
            contextual += 1

    # Dilemma exploration: check if both answers have prose passages
    dilemmas = graph.get_nodes_by_type("dilemma")
    has_answer_edges = graph.get_edges(edge_type="has_answer")
    answers_by_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_by_dilemma.setdefault(edge["from"], []).append(edge["to"])

    # Build answer→path mapping via explores edges (path explores answer)
    explores_edges = graph.get_edges(edge_type="explores")
    answer_to_path: dict[str, str] = {}
    for edge in explores_edges:
        answer_to_path[edge["to"]] = edge["from"]

    # Build path→has_prose lookup
    paths = graph.get_nodes_by_type("path")
    prose_by_path: dict[str, bool] = {}
    for path_id in paths:
        prose_by_path[path_id] = path_has_prose(graph, path_id)

    fully_explored = 0
    partially_explored = 0
    for did in dilemmas:
        answer_ids = answers_by_dilemma.get(did, [])
        answer_results: list[bool] = []
        for aid in answer_ids:
            answer_path = answer_to_path.get(aid)
            if answer_path:
                answer_results.append(prose_by_path.get(answer_path, False))

        if len(answer_results) >= 2 and all(answer_results):
            fully_explored += 1
        elif any(answer_results):
            partially_explored += 1

    # Compute max consecutive linear using shared BFS (with confront/resolve exemptions)
    max_linear = find_max_consecutive_linear(graph)

    # Start and ending passages
    # choice_to edges: choice → destination passage (incoming)
    # choice_from edges: choice → source passage (outgoing)
    choice_to_edges = graph.get_edges(edge_type="choice_to")
    has_incoming = {e["to"] for e in choice_to_edges}
    choice_from_edges = graph.get_edges(edge_type="choice_from")
    has_outgoing = {e["to"] for e in choice_from_edges}

    start_count = sum(1 for pid in passages if pid not in has_incoming)
    ending_count = sum(1 for pid in passages if pid not in has_outgoing)

    return BranchingStats(
        total_choices=len(choices),
        meaningful_choices=meaningful,
        contextual_choices=contextual,
        continue_choices=continue_count,
        max_consecutive_linear=max_linear,
        total_dilemmas=len(dilemmas),
        fully_explored=fully_explored,
        partially_explored=partially_explored,
        start_passages=start_count,
        ending_passages=ending_count,
    )


def _coverage_stats(graph: Graph, project_path: Path) -> CoverageStats:
    """Analyze entity, codex, and illustration coverage."""
    entities = graph.get_nodes_by_type("entity")
    entity_types: dict[str, int] = Counter(
        d.get("entity_type", "unknown") for d in entities.values()
    )

    codex_entries = graph.get_nodes_by_type("codex_entry")
    has_entry_edges = graph.get_edges(edge_type="HasEntry")
    entities_with_codex = len({e["to"] for e in has_entry_edges})

    briefs = graph.get_nodes_by_type("illustration_brief")
    illustrations = graph.get_nodes_by_type("illustration")

    assets_dir = project_path / "assets"
    asset_count = len(list(assets_dir.glob("*.png"))) if assets_dir.exists() else 0

    return CoverageStats(
        entity_count=len(entities),
        entity_types=dict(sorted(entity_types.items(), key=lambda x: -x[1])),
        codex_entries=len(codex_entries),
        entities_with_codex=entities_with_codex,
        illustration_briefs=len(briefs),
        illustration_nodes=len(illustrations),
        asset_files=asset_count,
    )


def _run_validation(graph: Graph) -> list[dict[str, str]]:
    """Run structural validation checks."""
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        return []

    try:
        report = run_all_checks(graph)
    except Exception as exc:
        log.warning("validation_skipped", reason=str(exc))
        return [{"name": "validation", "severity": "warn", "message": f"Skipped: {exc}"}]

    return [{"name": c.name, "severity": c.severity, "message": c.message} for c in report.checks]
