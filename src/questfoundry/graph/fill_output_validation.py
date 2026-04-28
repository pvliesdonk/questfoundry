"""FILL Stage Output Contract validator.

Implements the FILL §Stage Output Contract checks (R from
``docs/design/procedures/fill.md``):

1. Voice Document singleton exists with all required fields populated.
2. Every Passage has non-empty ``prose``.
3. Entity base-state enriched with zero or more universal micro-details
   (additive only).
4. No new node types created or destroyed by FILL (structural validation
   is upstream — POLISH owns DAG/passage shape).

Called at FILL exit (from ``ShipStage._validate_fill_output`` cited
indirectly by DRESS entry) and at DRESS entry. Read-only — never
mutates the graph.

Note that ``fill_validation.FillContractError`` already exists as the
exception type for FILL escalations (R-2.14 / R-5.2). Output-contract
violations reuse the same error type so callers only need to handle
one exception family from FILL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


# Voice Document fields required to be present and non-empty after FILL
# Phase 0. Mirrors the Pydantic VoiceDocument model in models/fill.py;
# updating either the model or this list requires updating the other.
# `story_title` lives on the same node but is added by FILL alongside the
# VoiceDocument — not a VoiceDocument field — so it isn't validated here.
_REQUIRED_VOICE_FIELDS = (
    "pov",
    "tense",
    "voice_register",
    "sentence_rhythm",
    "tone_words",
)


def validate_fill_output(graph: Graph) -> list[str]:
    """Return contract violations after FILL completion. Empty = compliant.

    Pure read-only — never mutates the graph.
    """
    errors: list[str] = []

    # Output-1: Voice Document singleton with required fields populated.
    voice_node = graph.get_node("voice::voice")
    if voice_node is None:
        errors.append("Output-1: FILL must produce a Voice Document at `voice::voice` (none found)")
    else:
        missing = [
            field
            for field in _REQUIRED_VOICE_FIELDS
            if not voice_node.get(field)
            or (isinstance(voice_node.get(field), str) and not voice_node[field].strip())
        ]
        if missing:
            errors.append(
                f"Output-1: Voice Document missing required field(s): {', '.join(missing)}"
            )

    # Output-2: every Passage has non-empty prose.
    passages = graph.get_nodes_by_type("passage")
    if not passages:
        errors.append("Output-2: FILL must produce at least one Passage (none found)")
    else:
        missing_prose = sorted(
            pid
            for pid, data in passages.items()
            if not data.get("prose") or not str(data["prose"]).strip()
        )
        if missing_prose:
            errors.append(
                f"Output-2: {len(missing_prose)} Passage(s) without prose: "
                f"{', '.join(repr(p) for p in missing_prose[:5])}"
                f"{' …' if len(missing_prose) > 5 else ''}"
            )

    return errors
