"""Project inspection and quality analysis.

Automates the manual quality review checks performed on completed projects.
Pure graph analysis — no LLM calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from questfoundry.graph.context import get_primary_beat, normalize_scoped_id, strip_scope_prefix
from questfoundry.graph.fill_context import compute_lexical_diversity
from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import (
    _get_spine_sequence,
    find_max_consecutive_linear,
    run_all_checks,
)
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
class BranchingQualityScore:
    """Informational branching quality metrics (not pass/fail).

    Provides a quantitative view of story structure diversity:
    convergence policy mix, beat exclusivity, and ending variety.
    """

    terminal_count: int = 0
    ending_variants: int = 0
    policy_distribution: dict[str, int] = field(default_factory=dict)
    avg_exclusive_beats: float = 0.0
    meaningful_choice_ratio: float = 0.0


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
class ProseNeutralityStats:
    """Prose-layer neutrality analysis for shared passages."""

    shared_passages: int = 0
    routed_passages: int = 0
    unrouted_heavy: list[str] = field(default_factory=list)
    unrouted_light: list[str] = field(default_factory=list)


@dataclass
class InspectionReport:
    """Complete project inspection report."""

    summary: GraphSummary
    prose: ProseStats | None = None
    branching: BranchingStats | None = None
    branching_quality: BranchingQualityScore | None = None
    coverage: CoverageStats = field(default_factory=CoverageStats)
    prose_neutrality: ProseNeutralityStats | None = None
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
    branching_quality = _branching_quality_score(graph, branching)
    coverage = _coverage_stats(graph, project_path)
    prose_neutrality = _prose_neutrality_stats(graph)
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
        branching_quality=branching_quality,
        coverage=coverage,
        prose_neutrality=prose_neutrality,
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
            flagged.append({"id": pid})
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

    # Structural exploration: a dilemma answer is explored when it has a path,
    # regardless of whether prose has been generated (prose is a FILL concern).
    fully_explored = 0
    partially_explored = 0
    for did in dilemmas:
        answer_ids = answers_by_dilemma.get(did, [])
        answer_results: list[bool] = []
        for aid in answer_ids:
            answer_results.append(bool(answer_to_path.get(aid)))

        if len(answer_results) >= 2 and all(answer_results):
            fully_explored += 1
        elif any(answer_results):
            partially_explored += 1

    # Compute max consecutive linear using shared BFS (with confront/resolve exemptions)
    max_linear = find_max_consecutive_linear(graph)

    # Start and ending passages
    # A passage is a "start" when it has no *forward* incoming choices.
    # Exclude hub-spoke return links (spoke→hub with is_return=True), otherwise
    # hubs can incorrectly appear to have incoming edges and start_count becomes 0.
    choice_nodes = graph.get_nodes_by_type("choice")
    has_incoming: set[str] = {
        data["to_passage"]
        for data in choice_nodes.values()
        if data.get("to_passage") and not data.get("is_return")
    }
    # For outgoing, it's more direct to use the `choice_from` edges.
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


def _branching_quality_score(
    graph: Graph, branching: BranchingStats | None
) -> BranchingQualityScore | None:
    """Compute informational branching quality metrics.

    Returns None if the graph has no arc nodes (pre-GROW or degenerate).
    """
    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return None

    spine_seq_set = _get_spine_sequence(arc_nodes)

    # Policy distribution + exclusive beat counts across branch arcs
    policy_counts: dict[str, int] = Counter()
    exclusive_counts: list[int] = []
    for data in arc_nodes.values():
        if data.get("arc_type") == "spine":
            continue
        policy = data.get("dilemma_role", "unknown")
        policy_counts[policy] += 1
        seq: list[str] = data.get("sequence", [])
        diverges_at = data.get("diverges_at")
        if seq and diverges_at:
            try:
                div_idx = seq.index(diverges_at)
            except ValueError:
                continue
            exclusive = [b for b in seq[div_idx + 1 :] if b not in spine_seq_set]
            exclusive_counts.append(len(exclusive))

    # 0.0 when no branch arcs have divergence metadata (pre-policy graphs)
    avg_exclusive = (
        round(sum(exclusive_counts) / len(exclusive_counts), 1) if exclusive_counts else 0.0
    )

    # Terminal count + ending variants
    choice_from_edges = graph.get_edges(edge_type="choice_from")
    has_outgoing = {e["to"] for e in choice_from_edges}
    passages = graph.get_nodes_by_type("passage")
    ending_ids = [pid for pid in passages if pid not in has_outgoing]

    # Ending variants: distinct state flag signatures per ending
    # For each ending, find which arcs cover its from_beat, then collect state flags
    beat_to_arcs: dict[str, list[str]] = {}
    for arc_id, data in arc_nodes.items():
        for beat_id in data.get("sequence", []):
            beat_to_arcs.setdefault(beat_id, []).append(arc_id)

    # Build arc → state flags: arc paths → consequences → state flags via tracks edges
    arc_state_flags: dict[str, frozenset[str]] = {}
    tracks_edges = graph.get_edges(edge_type="tracks")
    consequence_to_state_flag: dict[str, str] = {}
    for edge in tracks_edges:
        consequence_to_state_flag[edge["to"]] = edge["from"]

    has_consequence_edges = graph.get_edges(edge_type="has_consequence")
    path_consequences: dict[str, list[str]] = {}
    for edge in has_consequence_edges:
        path_consequences.setdefault(edge["from"], []).append(edge["to"])

    for arc_id, data in arc_nodes.items():
        sfs: set[str] = set()
        for path_raw in data.get("paths", []):
            path_id = normalize_scoped_id(path_raw, "path")
            for cons_id in path_consequences.get(path_id, []):
                if sf := consequence_to_state_flag.get(cons_id):
                    sfs.add(sf)
        arc_state_flags[arc_id] = frozenset(sfs)

    variant_signatures: set[frozenset[str]] = set()
    for pid in ending_ids:
        pdata = passages[pid]
        # Synthetic endings (from split_ending_families) carry family_state_flags
        # directly; non-synthetic endings need beat→arc→state flag lookup.
        family_sfs = pdata.get("family_state_flags")
        if family_sfs is not None:
            variant_signatures.add(frozenset(family_sfs))
        else:
            from_beat = get_primary_beat(graph, pid) or ""
            covering_arcs = beat_to_arcs.get(from_beat, [])
            for arc_id in covering_arcs:
                variant_signatures.add(arc_state_flags.get(arc_id, frozenset()))

    # Meaningful choice ratio
    ratio = 0.0
    if branching and branching.total_choices > 0:
        ratio = round(branching.meaningful_choices / branching.total_choices, 2)

    return BranchingQualityScore(
        terminal_count=len(ending_ids),
        ending_variants=len(variant_signatures),
        policy_distribution=dict(sorted(policy_counts.items())),
        avg_exclusive_beats=avg_exclusive,
        meaningful_choice_ratio=ratio,
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


def _prose_neutrality_stats(graph: Graph) -> ProseNeutralityStats | None:
    """Analyze prose-layer neutrality for shared passages."""
    arc_nodes = graph.get_nodes_by_type("arc")
    passage_nodes = graph.get_nodes_by_type("passage")
    choice_nodes = graph.get_nodes_by_type("choice")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    if not arc_nodes or not passage_nodes:
        return None

    # Build beat → covering arc count
    beat_arcs: dict[str, set[str]] = {}
    for arc_id, adata in arc_nodes.items():
        for beat_id in adata.get("sequence", []):
            beat_arcs.setdefault(str(beat_id), set()).add(arc_id)

    # Find shared passages (beat covered by 2+ arcs)
    shared: list[str] = []
    for pid in passage_nodes:
        from_beat = get_primary_beat(graph, pid) or ""
        if from_beat and len(beat_arcs.get(from_beat, set())) >= 2:
            shared.append(pid)

    # Find routed passages
    routed: set[str] = set()
    for _cid, cdata in choice_nodes.items():
        if cdata.get("is_routing"):
            source = str(cdata.get("from_passage", ""))
            if source:
                routed.add(source)

    routed_shared = [p for p in shared if p in routed]

    # Build dilemma raw_id → set of path raw_ids for convergence check
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_path_map: dict[str, set[str]] = {}
    for _pid_p, pdata_p in path_nodes.items():
        did = pdata_p.get("dilemma_id", "")
        raw = pdata_p.get("raw_id", "")
        if did and raw:
            dilemma_path_map.setdefault(strip_scope_prefix(did), set()).add(raw)

    # Check unrouted shared passages for heavy/light dilemmas that converge here
    unrouted_heavy: list[str] = []
    unrouted_light: list[str] = []
    for pid in shared:
        if pid in routed:
            continue
        # Find beat paths for this passage
        beat_id = get_primary_beat(graph, pid) or ""
        beat = graph.get_node(beat_id) if beat_id else None
        beat_paths: set[str] = set(beat.get("paths", [])) if beat else set()

        # Only check dilemmas whose paths converge at this passage's beat
        has_heavy = False
        has_light = False
        for did, ddata in dilemma_nodes.items():
            raw_did = ddata.get("raw_id", strip_scope_prefix(did))
            d_paths = dilemma_path_map.get(raw_did, set())
            if len(d_paths & beat_paths) < 2:
                continue  # Dilemma doesn't converge here
            weight = ddata.get("residue_weight", "light")
            salience = ddata.get("ending_salience", "low")
            if weight == "heavy" or salience == "high":
                has_heavy = True
            elif weight == "light" and salience == "low":
                has_light = True
        if has_heavy:
            unrouted_heavy.append(pid)
        elif has_light:
            unrouted_light.append(pid)

    return ProseNeutralityStats(
        shared_passages=len(shared),
        routed_passages=len(routed_shared),
        unrouted_heavy=sorted(unrouted_heavy),
        unrouted_light=sorted(unrouted_light),
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
