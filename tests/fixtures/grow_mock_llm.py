from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import find_passage_successors
from questfoundry.models.grow import ChoiceLabel, Phase9Output

if TYPE_CHECKING:
    from pathlib import Path


def build_phase9_output(
    messages: list[Any],
    *,
    project_path: Path | None = None,
) -> Phase9Output:
    """Build a deterministic Phase9Output for mock LLMs.

    Prefer graph-derived transitions when a project_path is provided; fall
    back to parsing the prompt text when the graph is unavailable.
    """
    text = "\n".join(
        getattr(message, "content", str(message)) for message in messages if message is not None
    )
    text = text.split("Semantic validation errors", 1)[0]
    is_divergence_prompt = "Divergence Points to Label" in text or "Divergence at" in text

    pairs = _pairs_from_graph(project_path, is_divergence_prompt)
    if not pairs:
        pairs = _pairs_from_text(text, is_divergence_prompt)

    labels = [ChoiceLabel(from_passage=src, to_passage=dst, label="continue") for src, dst in pairs]
    return Phase9Output(labels=labels)


def _pairs_from_graph(
    project_path: Path | None,
    is_divergence_prompt: bool,
) -> list[tuple[str, str]]:
    if project_path is None:
        return []

    snapshot = Graph.load(project_path / "graph.db")
    successors = find_passage_successors(snapshot)
    if not successors:
        return []

    if is_divergence_prompt:
        return [
            (p_id, succ.to_passage)
            for p_id, succ_list in successors.items()
            if len(succ_list) > 1
            for succ in succ_list
        ]
    return [
        (p_id, succ_list[0].to_passage)
        for p_id, succ_list in successors.items()
        if len(succ_list) == 1
    ]


def _pairs_from_text(text: str, is_divergence_prompt: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    if is_divergence_prompt:
        current_from: str | None = None
        for line in text.splitlines():
            from_match = re.search(r"Divergence at (passage::[^\s,:]+)", line)
            if from_match:
                current_from = from_match.group(1)
                continue
            to_match = re.search(r"-\s*(passage::[^\s,:]+)", line)
            if current_from and to_match:
                pair = (current_from, to_match.group(1))
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    for line in text.splitlines():
        if "→" not in line and "->" not in line:
            continue
        ids = re.findall(r"passage::[^\s,:]+", line)
        if len(ids) >= 2:
            pair = (ids[0], ids[1])
            if pair not in pairs:
                pairs.append(pair)

    if pairs:
        return pairs

    from_ids: list[str] = []
    to_ids: list[str] = []
    if "valid_from_ids:" in text:
        segment = text.split("valid_from_ids:", 1)[1].split("valid_to_ids:", 1)[0]
        from_ids = _extract_ids(segment)
    if "valid_to_ids:" in text:
        segment = text.split("valid_to_ids:", 1)[1]
        segment = segment.split("output_language_instruction", 1)[0]
        to_ids = _extract_ids(segment)

    if from_ids and to_ids and len(from_ids) == len(to_ids):
        for src, dst in zip(from_ids, to_ids, strict=True):
            pair = (src, dst)
            if pair not in pairs:
                pairs.append(pair)
        return pairs

    for line in text.splitlines():
        ids = re.findall(r"passage::[^\s,:]+", line)
        if len(ids) >= 2:
            pair = (ids[0], ids[1])
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def _extract_ids(segment: str) -> list[str]:
    tokens = re.split(r"[\s,]+", segment)
    ids: list[str] = []
    for token in tokens:
        cleaned = token.strip().strip('"').strip("'").strip(")").strip("]").rstrip(":")
        if cleaned.startswith("passage::"):
            ids.append(cleaned)
    return ids
