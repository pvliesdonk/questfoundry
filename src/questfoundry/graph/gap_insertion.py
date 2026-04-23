"""Shared gap-beat insertion validation.

Used by:

- POLISH Phase 1a (Narrative Gap Insertion) — moved from GROW per the
  spec migration in PR #1366 (structural-vs-narrative principle: gap
  insertion is narrative-prep work).
- GROW Phase 4c (Pacing Gaps) — temporary co-tenant; moves to POLISH
  Phase 2 in the next migration PR (issue #1368).

Both callers produce the same kind of `GapProposal` and need the same
validation + insertion logic. This module is the single source of truth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph
    from questfoundry.models.grow import GapProposal

log = get_logger(__name__)


@dataclass
class GapInsertionReport:
    """Summary of gap insertion validation and results."""

    inserted: int = 0
    invalid_path_id: int = 0
    invalid_after_beat: int = 0
    invalid_before_beat: int = 0
    invalid_beat_order: int = 0
    beat_not_in_sequence: int = 0
    anchor_wrong_path: int = 0

    @property
    def total_invalid(self) -> int:
        return (
            self.invalid_path_id
            + self.invalid_after_beat
            + self.invalid_before_beat
            + self.invalid_beat_order
            + self.beat_not_in_sequence
            + self.anchor_wrong_path
        )


def validate_and_insert_gaps(
    graph: Graph,
    gaps: list[GapProposal],
    valid_path_ids: set[str] | dict[str, Any],
    valid_beat_ids: set[str] | dict[str, Any],
    phase_name: str,
) -> GapInsertionReport:
    """Validate gap proposals and insert valid ones into the graph.

    Checks path_id prefixing, beat ID existence, ordering, and cycle
    safety before inserting each gap beat. A cycle check runs before
    every insertion: if inserting the gap beat would close a cycle in
    the predecessor DAG, the gap is skipped with a log message.

    Args:
        graph: Graph to insert beats into.
        gaps: List of GapProposal instances from LLM output.
        valid_path_ids: Set or dict of valid path IDs.
        valid_beat_ids: Set or dict of valid beat IDs.
        phase_name: Phase name for log event prefixing.

    Returns:
        Report with counts of inserted and invalid gaps.
    """
    from questfoundry.graph.grow_algorithms import (
        _would_create_cycle,
        get_path_beat_sequence,
        insert_gap_beat,
    )

    report = GapInsertionReport()
    valid_path_set = (
        set(valid_path_ids.keys()) if isinstance(valid_path_ids, dict) else set(valid_path_ids)
    )
    valid_beat_set = (
        set(valid_beat_ids.keys()) if isinstance(valid_beat_ids, dict) else set(valid_beat_ids)
    )

    # Build the successors dict (prerequisite → dependents) for cycle detection.
    # predecessor(A, B) means B comes before A; successors[B] contains A.
    # We keep this in sync after each successful insertion.
    beat_set: set[str] = set(graph.get_nodes_by_type("beat").keys())
    successors: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="predecessor"):
        # edge["from"] requires edge["to"] → edge["to"] is a prereq of edge["from"]
        successors[edge["to"]].add(edge["from"])

    def _normalize_beat_id(beat_id: str | None) -> str | None:
        if not beat_id:
            return None
        if beat_id in valid_beat_set:
            return beat_id
        if not beat_id.startswith("beat::"):
            prefixed = f"beat::{beat_id}"
            if prefixed in valid_beat_set:
                log.info(
                    f"{phase_name}_unprefixed_beat_id",
                    beat_id=beat_id,
                    prefixed=prefixed,
                )
                return prefixed
        return beat_id

    for gap in gaps:
        prefixed_pid = gap.path_id if gap.path_id.startswith("path::") else f"path::{gap.path_id}"
        if prefixed_pid != gap.path_id:
            log.info(
                f"{phase_name}_unprefixed_path_id",
                path_id=gap.path_id,
                prefixed=prefixed_pid,
            )
        if prefixed_pid not in valid_path_set:
            log.info(f"{phase_name}_invalid_path_id", path_id=gap.path_id)
            report.invalid_path_id += 1
            continue
        after_beat = _normalize_beat_id(gap.after_beat)
        before_beat = _normalize_beat_id(gap.before_beat)
        if after_beat and after_beat not in valid_beat_set:
            log.info(f"{phase_name}_invalid_after_beat", beat_id=after_beat)
            report.invalid_after_beat += 1
            continue
        if before_beat and before_beat not in valid_beat_set:
            log.info(f"{phase_name}_invalid_before_beat", beat_id=before_beat)
            report.invalid_before_beat += 1
            continue
        # Validate path membership: anchors must belong to the gap's path.
        # A beat belongs to a path if it has a belongs_to edge pointing to it.
        if after_beat:
            after_paths = {
                e["to"] for e in graph.get_edges(edge_type="belongs_to", from_id=after_beat)
            }
            if prefixed_pid not in after_paths:
                log.info(
                    f"{phase_name}_anchor_wrong_path",
                    after_beat=after_beat,
                    path_id=prefixed_pid,
                )
                report.anchor_wrong_path += 1
                continue
        if before_beat:
            before_paths = {
                e["to"] for e in graph.get_edges(edge_type="belongs_to", from_id=before_beat)
            }
            if prefixed_pid not in before_paths:
                log.info(
                    f"{phase_name}_anchor_wrong_path",
                    before_beat=before_beat,
                    path_id=prefixed_pid,
                )
                report.anchor_wrong_path += 1
                continue
        # Validate ordering: after_beat must come before before_beat
        if after_beat and before_beat:
            sequence = get_path_beat_sequence(graph, prefixed_pid)
            try:
                after_idx = sequence.index(after_beat)
                before_idx = sequence.index(before_beat)
                if after_idx >= before_idx:
                    log.info(
                        f"{phase_name}_invalid_beat_order",
                        after_beat=after_beat,
                        before_beat=before_beat,
                    )
                    report.invalid_beat_order += 1
                    continue
            except ValueError:
                log.info(f"{phase_name}_beat_not_in_sequence", path_id=gap.path_id)
                report.beat_not_in_sequence += 1
                continue

        # Cycle prevention: if after_beat and before_beat are both given,
        # check whether inserting the gap would create a cycle.
        # The gap insertion adds predecessor(gap, after_beat) and
        # predecessor(before_beat, gap). A cycle forms if after_beat is
        # already a transitive successor of before_beat — i.e. inserting
        # gap closes a circle: before_beat → gap → after_beat → ... → before_beat.
        # This is equivalent to asking: is after_beat reachable from
        # before_beat via the current successors graph?
        # _would_create_cycle(before_beat, after_beat, ...) returns True iff
        # after_beat is reachable from before_beat, which is exactly that case.
        if (
            after_beat
            and before_beat
            and _would_create_cycle(before_beat, after_beat, successors, beat_set)
        ):
            log.info(
                "gap_skipped_would_create_cycle",
                phase=phase_name,
                after_beat=after_beat,
                before_beat=before_beat,
                path_id=prefixed_pid,
            )
            continue

        new_beat_id = insert_gap_beat(
            graph,
            path_id=prefixed_pid,
            after_beat=after_beat,
            before_beat=before_beat,
            summary=gap.summary,
            scene_type=gap.scene_type,
            dilemma_impacts=[i.model_dump() for i in gap.dilemma_impacts],
        )
        report.inserted += 1

        # Update successors and beat_set to reflect the newly inserted beat.
        # predecessor(new_beat, after_beat) → after_beat is prereq of new_beat
        # predecessor(before_beat, new_beat) → new_beat is prereq of before_beat
        beat_set.add(new_beat_id)
        if after_beat:
            successors[after_beat].add(new_beat_id)
        if before_beat:
            successors[new_beat_id].add(before_beat)

    return report
