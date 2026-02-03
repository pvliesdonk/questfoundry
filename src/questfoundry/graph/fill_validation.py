"""Arc-level validation checks for post-FILL quality assurance.

Validates narrative arc integrity after prose generation. These are pure,
deterministic functions operating on the graph -- no LLM calls.

Validation checks:
- Intensity progression: final third should have more high-intensity beats
- Dramatic questions closed: every opened question should be committed before arc end
- Narrative function variety: no 5+ consecutive identical functions, must include
  at least one confront or resolve
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.fill_context import compute_open_questions, derive_pacing
from questfoundry.graph.grow_validation import ValidationCheck, ValidationReport

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Maximum allowed run of identical consecutive narrative_function values.
MAX_CONSECUTIVE_SAME_FUNCTION = 4


def check_intensity_progression(graph: Graph, arc_id: str) -> ValidationCheck:
    """Check that the final third of an arc has more high-intensity beats than the first third.

    Uses derive_pacing() to classify each beat's intensity from its
    narrative_function and scene_type.

    Args:
        graph: Graph containing arc and beat nodes.
        arc_id: The arc to validate.

    Returns:
        ValidationCheck with pass/warn/fail severity.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ValidationCheck(
            name="intensity_progression",
            severity="pass",
            message=f"Arc {arc_id} not found, skipping",
        )

    sequence = arc_node.get("sequence", [])
    if len(sequence) < 3:
        return ValidationCheck(
            name="intensity_progression",
            severity="pass",
            message=f"Arc has {len(sequence)} beats (< 3), too short to check",
        )

    # Split into approximate thirds (first N, middle, last N where N = len//3)
    third = len(sequence) // 3
    first_third = sequence[:third]
    final_third = sequence[-third:]

    def count_high(beat_ids: list[str]) -> int:
        count = 0
        for bid in beat_ids:
            beat = graph.get_node(bid)
            if not beat:
                continue
            nf = beat.get("narrative_function", "")
            st = beat.get("scene_type", "")
            if nf and st:
                intensity, _ = derive_pacing(nf, st)
                if intensity == "high":
                    count += 1
        return count

    first_high = count_high(first_third)
    final_high = count_high(final_third)

    if final_high > first_high:
        return ValidationCheck(
            name="intensity_progression",
            severity="pass",
            message=f"Intensity rises: first third {first_high} high, final third {final_high} high",
        )

    if final_high == first_high:
        return ValidationCheck(
            name="intensity_progression",
            severity="warn",
            message=(
                f"Flat intensity: first third {first_high} high, final third {final_high} high. "
                "Consider escalating tension toward the end."
            ),
        )

    return ValidationCheck(
        name="intensity_progression",
        severity="warn",
        message=(
            f"Intensity drops: first third {first_high} high, final third {final_high} high. "
            "The arc may feel anticlimactic."
        ),
    )


def check_dramatic_questions_closed(graph: Graph, arc_id: str) -> ValidationCheck:
    """Check that every opened dramatic question has a commits before arc end.

    Walks the full beat sequence and uses compute_open_questions() with
    a sentinel beat ID to get questions still open at the end.

    Args:
        graph: Graph containing arc, beat, and dilemma nodes.
        arc_id: The arc to validate.

    Returns:
        ValidationCheck with pass/warn severity.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ValidationCheck(
            name="dramatic_questions_closed",
            severity="pass",
            message=f"Arc {arc_id} not found, skipping",
        )

    sequence = arc_node.get("sequence", [])
    if not sequence:
        return ValidationCheck(
            name="dramatic_questions_closed",
            severity="pass",
            message="Arc has no beats",
        )

    # Use a sentinel that won't match any beat to walk the full sequence
    sentinel = "__sentinel_end__"
    open_qs = compute_open_questions(graph, arc_id, sentinel)

    if not open_qs:
        return ValidationCheck(
            name="dramatic_questions_closed",
            severity="pass",
            message="All dramatic questions are resolved by arc end",
        )

    # str() needed: compute_open_questions returns dict[str, str | int]
    unclosed = [str(q.get("dilemma_id", "?")) for q in open_qs]
    return ValidationCheck(
        name="dramatic_questions_closed",
        severity="warn",
        message=f"{len(unclosed)} unclosed dramatic question(s): {', '.join(unclosed)}",
    )


def check_narrative_function_variety(graph: Graph, arc_id: str) -> ValidationCheck:
    """Check narrative function variety within an arc.

    Two checks:
    1. No run of 5+ consecutive identical narrative_function values
    2. Arc must contain at least one 'confront' or 'resolve'

    Args:
        graph: Graph containing arc and beat nodes.
        arc_id: The arc to validate.

    Returns:
        ValidationCheck with pass/warn severity.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ValidationCheck(
            name="narrative_function_variety",
            severity="pass",
            message=f"Arc {arc_id} not found, skipping",
        )

    sequence = arc_node.get("sequence", [])
    if not sequence:
        return ValidationCheck(
            name="narrative_function_variety",
            severity="pass",
            message="Arc has no beats",
        )

    functions: list[str] = []
    for bid in sequence:
        beat = graph.get_node(bid)
        if not beat:
            continue
        nf = beat.get("narrative_function", "")
        if nf:
            functions.append(nf)

    if not functions:
        return ValidationCheck(
            name="narrative_function_variety",
            severity="warn",
            message="No beats have narrative_function set",
        )

    # Check for long runs
    max_run = 1
    run_fn = ""
    current_run = 1
    for i in range(1, len(functions)):
        if functions[i] == functions[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                run_fn = functions[i]
        else:
            current_run = 1

    issues: list[str] = []
    if max_run > MAX_CONSECUTIVE_SAME_FUNCTION:
        issues.append(
            f"{max_run} consecutive '{run_fn}' beats (max {MAX_CONSECUTIVE_SAME_FUNCTION})"
        )

    # Check for climactic functions
    has_climactic = any(f in ("confront", "resolve") for f in functions)
    if not has_climactic:
        issues.append("No 'confront' or 'resolve' beats in arc")

    if issues:
        return ValidationCheck(
            name="narrative_function_variety",
            severity="warn",
            message="; ".join(issues),
        )

    return ValidationCheck(
        name="narrative_function_variety",
        severity="pass",
        message=f"Good variety: {len(set(functions))} distinct functions, max run {max_run}",
    )


def path_has_prose(graph: Graph, path_id: str) -> bool:
    """Check if a path has any passages with prose content.

    Walks the beats belonging to the path and checks if any of their
    corresponding passages have non-empty prose.

    Args:
        graph: Graph containing path, beat, and passage nodes.
        path_id: The path node ID to check.

    Returns:
        True if at least one passage in the path has prose.
    """
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_ids = {e["from"] for e in belongs_to_edges if e["to"] == path_id}

    if not beat_ids:
        return False

    passages = graph.get_nodes_by_type("passage")
    for _pid, pdata in passages.items():
        from_beat = pdata.get("from_beat", "")
        if from_beat in beat_ids:
            prose = pdata.get("prose")
            if prose and str(prose).strip():
                return True
    return False


def check_dilemma_prose_coverage(graph: Graph) -> list[ValidationCheck]:
    """Check that all dilemma paths have prose content.

    For each dilemma, verifies that both answer paths have at least one
    passage with prose. A dilemma where only one path has prose represents
    wasted branching — the player sees a choice but one option leads nowhere.

    Args:
        graph: Graph containing dilemma, answer, path, and passage nodes.

    Returns:
        List of ValidationCheck results (one per dilemma with issues).
    """
    dilemmas = graph.get_nodes_by_type("dilemma")
    if not dilemmas:
        return []

    has_answer_edges = graph.get_edges(edge_type="has_answer")
    answers_by_dilemma: dict[str, list[str]] = {}
    for edge in has_answer_edges:
        answers_by_dilemma.setdefault(edge["from"], []).append(edge["to"])

    explores_edges = graph.get_edges(edge_type="explores")
    answer_to_path: dict[str, str] = {}
    for edge in explores_edges:
        answer_to_path[edge["to"]] = edge["from"]

    checks: list[ValidationCheck] = []
    for did in sorted(dilemmas):
        answer_ids = answers_by_dilemma.get(did, [])
        if len(answer_ids) < 2:
            continue

        paths_with_prose = 0
        paths_without_prose: list[str] = []
        for aid in answer_ids:
            answer_path = answer_to_path.get(aid)
            if answer_path and path_has_prose(graph, answer_path):
                paths_with_prose += 1
            elif answer_path:
                paths_without_prose.append(answer_path)

        if paths_without_prose:
            checks.append(
                ValidationCheck(
                    name="dilemma_prose_coverage",
                    severity="warn",
                    message=(
                        f"Dilemma {did}: {paths_with_prose}/{len(answer_ids)} paths have prose. "
                        f"Missing: {', '.join(paths_without_prose)}"
                    ),
                )
            )

    return checks


def run_arc_validation(graph: Graph) -> ValidationReport:
    """Run all arc-level validation checks across all arcs.

    Args:
        graph: Graph containing arc and beat nodes.

    Returns:
        ValidationReport with all check results.
    """
    report = ValidationReport()
    arcs = graph.get_nodes_by_type("arc")

    if not arcs:
        report.checks.append(
            ValidationCheck(
                name="arc_validation",
                severity="pass",
                message="No arcs found, skipping validation",
            )
        )
        return report

    for arc_id in sorted(arcs.keys()):
        report.checks.append(check_intensity_progression(graph, arc_id))
        report.checks.append(check_dramatic_questions_closed(graph, arc_id))
        report.checks.append(check_narrative_function_variety(graph, arc_id))

    report.checks.extend(check_dilemma_prose_coverage(graph))

    return report
