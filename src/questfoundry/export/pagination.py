"""Shared passage-numbering utility for gamebook-style exports.

The PDF exporter used to own this directly. R-3.5 (gamebook PDF
shuffles passages with a seeded RNG and renumbers) requires the
mapping to be reproducible AND inspectable; R-3.6 / #1336 requires
the ``passage_id → page_number`` map to be exported as metadata so
authors can trace which passage became which page after a bug report.

Living in a shared module lets the PDF exporter render the numbering
and a sidecar writer surface the same map without recomputing.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.export.base import ExportPassage


def compute_passage_numbering(passages: list[ExportPassage]) -> dict[str, int]:
    """Map passage IDs to randomized section numbers (start passage = 1).

    Uses a SHA256 of the sorted-and-joined passage IDs as the seed so
    the shuffle is reproducible across runs and Python versions
    (Python's built-in ``hash()`` is randomized per process).

    Args:
        passages: All passages destined for the export.

    Returns:
        Mapping from passage ID to its assigned 1-based section number.
        Returns an empty dict for an empty list.
    """
    if not passages:
        return {}

    start_id = next((p.id for p in passages if p.is_start), passages[0].id)

    other_ids = sorted(p.id for p in passages if p.id != start_id)

    all_ids = sorted(p.id for p in passages)
    id_string = "|".join(all_ids)
    seed = int(hashlib.md5(id_string.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    numbers = list(range(2, len(passages) + 1))
    rng.shuffle(numbers)

    numbering: dict[str, int] = {start_id: 1}
    for pid, num in zip(other_ids, numbers, strict=True):
        numbering[pid] = num

    return numbering
