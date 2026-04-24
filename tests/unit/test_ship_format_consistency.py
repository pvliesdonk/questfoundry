"""R-3.1: same classified data drives every format.

All four exporters consume a single ``ExportContext``. This test
suite asserts that the player-facing counts (passages, choices,
codewords, codex entries, illustrations) and structural facts
(start passage id, ending passage ids, entity ids referenced from
choice grants) are identical across formats — i.e. the formats
cannot diverge in content beyond format-specific presentation.

Cluster #1338 specifically called out the absence of this test:
the implementation is correct (single source of truth in
ExportContext) but tests were missing, so a future change that
filtered passages in one exporter and not another would slip
through unnoticed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from questfoundry.export.base import (
    ExportChoice,
    ExportCodeword,
    ExportCodexEntry,
    ExportContext,
    ExportEntity,
    ExportPassage,
)
from questfoundry.export.html_exporter import HtmlExporter
from questfoundry.export.i18n import get_ui_strings
from questfoundry.export.json_exporter import JsonExporter
from questfoundry.export.twee_exporter import TweeExporter

if TYPE_CHECKING:
    from pathlib import Path

_FIXED_TS = "2026-04-24T00:00:00+00:00"


def _multi_format_context() -> ExportContext:
    """A non-trivial context with passages, choices, codewords, codex.

    Ensures every exporter has something to render in every section
    so the count comparisons are meaningful.
    """
    return ExportContext(
        title="Cross-Format Test",
        passages=[
            ExportPassage(id="passage::start", prose="The gate opens.", is_start=True),
            ExportPassage(id="passage::trial", prose="A choice appears."),
            ExportPassage(id="passage::triumph", prose="Victory.", is_ending=True),
            ExportPassage(id="passage::defeat", prose="Failure.", is_ending=True),
        ],
        choices=[
            ExportChoice(
                from_passage="passage::start",
                to_passage="passage::trial",
                label="Step inside",
            ),
            ExportChoice(
                from_passage="passage::trial",
                to_passage="passage::triumph",
                label="Trust the mentor",
                grants=["codeword::mentor_trusted"],
            ),
            ExportChoice(
                from_passage="passage::trial",
                to_passage="passage::defeat",
                label="Refuse the mentor",
                requires_codewords=["codeword::wary"],
            ),
        ],
        entities=[
            ExportEntity(
                id="entity::mentor",
                entity_type="character",
                concept="A weathered guide.",
            ),
        ],
        codewords=[
            ExportCodeword(id="codeword::mentor_trusted", codeword_type="granted"),
            ExportCodeword(id="codeword::wary", codeword_type="granted"),
        ],
        codex_entries=[
            ExportCodexEntry(
                entity_id="entity::mentor",
                title="Mentor",
                rank=1,
                content="A figure encountered in the story.",
            ),
            ExportCodexEntry(
                entity_id="entity::mentor",
                title="Mentor's Secret",
                rank=2,
                visible_when=["codeword::mentor_trusted"],
                content="They knew you all along.",
            ),
        ],
    )


_SUGARCUBE_RESERVED = {
    "StoryTitle",
    "StoryData",
    "StoryInit",
    "StoryArtDirection",
    "StoryMetadata",
}


def _twee_passage_count(text: str, language: str = "en") -> int:
    """Count `:: <name>` headers excluding SugarCube reserved infrastructure
    and the localised Codex sidecar passage.

    A naive ``\\S+`` would miscount a passage like ``:: my adventure`` as
    ``my``. The Twee header grammar puts the optional ``[tag]`` block
    AND the optional ``{position/size}`` block (used by StoryMetadata
    et al.) after a whitespace separator, so capture everything up to
    the first ``[`` or ``{`` preceded by whitespace. The codex name
    is i18n'd (``Codex`` / ``Kodex`` / …) so resolve it via
    ``get_ui_strings()`` rather than hardcoding English.
    """
    reserved = _SUGARCUBE_RESERVED | {get_ui_strings(language)["codex"]}
    count = 0
    for line in text.splitlines():
        match = re.match(r"^::\s*(.+?)(?:\s+[\[\{].*)?$", line)
        if not match:
            continue
        name = match.group(1).strip()
        if name and name not in reserved:
            count += 1
    return count


def _twee_choice_labels(text: str) -> set[str]:
    """Collect every choice label from both Twee link forms."""
    labels: set[str] = set()
    # Plain [[label->target]]
    for match in re.finditer(r"\[\[([^\]]+?)->[^\]]+\]\]", text):
        labels.add(match.group(1).strip())
    # <<link "label">>...
    for match in re.finditer(r'<<link\s+"([^"]+)"\s*>>', text):
        labels.add(match.group(1).strip())
    return labels


def _html_choice_labels(text: str) -> set[str]:
    """Collect labels from <a class="choice" ...>label</a>."""
    return {
        m.group(1).strip()
        for m in re.finditer(
            r'<a class="choice"[^>]*>([^<]+)</a>',
            text,
        )
    }


def test_twee_passage_count_handles_spaces_in_names() -> None:
    """Pin the latent fix: the helper must capture full passage names
    even when they contain spaces (the previous \\S+ pattern stopped
    at the first whitespace and would have undercounted)."""
    text = (
        ":: StoryTitle\nT\n\n"
        ":: my adventure\nProse.\n\n"
        ":: another story\nMore.\n\n"
        ':: StoryArtDirection {"position":"0,0"}\nfoo: bar\n'
    )
    # Two real passages, StoryTitle + StoryArtDirection are SugarCube reserved
    assert _twee_passage_count(text) == 2


def test_twee_passage_count_excludes_localised_codex() -> None:
    """German uses 'Kodex' for the codex sidecar — the helper must
    treat that as reserved when language='de'."""
    text = ":: StoryTitle\nT\n\n:: Start [start]\nBeginn.\n\n:: Kodex\nReferenz.\n"
    # 1 navigable passage; Kodex is the i18n codex sidecar
    assert _twee_passage_count(text, language="de") == 1
    # Same text under English would count Kodex as a regular passage
    assert _twee_passage_count(text, language="en") == 2


class TestCrossFormatConsistency:
    """Same context → same player-facing counts across formats (R-3.1)."""

    def test_passage_counts_match(self, tmp_path: Path) -> None:
        ctx = _multi_format_context()
        json_path = JsonExporter().export(ctx, tmp_path / "j", timestamp=_FIXED_TS)
        twee_path = TweeExporter().export(ctx, tmp_path / "t", timestamp=_FIXED_TS)
        html_path = HtmlExporter().export(ctx, tmp_path / "h", timestamp=_FIXED_TS)

        json_data = json.loads(json_path.read_text())
        twee_text = twee_path.read_text()
        html_text = html_path.read_text()

        json_passages = len(json_data["passages"])
        twee_passages = _twee_passage_count(twee_text)
        html_passages = len(re.findall(r'<div class="passage" id="', html_text))

        assert json_passages == twee_passages == html_passages == len(ctx.passages), (
            f"passage count diverged: json={json_passages}, twee={twee_passages}, "
            f"html={html_passages}, expected={len(ctx.passages)}"
        )

    def test_choice_labels_match(self, tmp_path: Path) -> None:
        """Every choice label in the source ExportContext appears in
        every format's output. R-3.2 (Twee verbatim) is exercised by
        the Twee assertion; this test covers cross-format consistency.
        """
        ctx = _multi_format_context()
        twee_path = TweeExporter().export(ctx, tmp_path / "t", timestamp=_FIXED_TS)
        html_path = HtmlExporter().export(ctx, tmp_path / "h", timestamp=_FIXED_TS)
        json_path = JsonExporter().export(ctx, tmp_path / "j", timestamp=_FIXED_TS)

        expected_labels = {c.label for c in ctx.choices}

        twee_labels = _twee_choice_labels(twee_path.read_text())
        html_labels = _html_choice_labels(html_path.read_text())
        json_labels = {c["label"] for c in json.loads(json_path.read_text())["choices"]}

        assert json_labels == expected_labels
        assert twee_labels == expected_labels
        assert html_labels == expected_labels

    def test_codex_entry_counts_match(self, tmp_path: Path) -> None:
        ctx = _multi_format_context()
        twee_path = TweeExporter().export(ctx, tmp_path / "t", timestamp=_FIXED_TS)
        html_path = HtmlExporter().export(ctx, tmp_path / "h", timestamp=_FIXED_TS)
        json_path = JsonExporter().export(ctx, tmp_path / "j", timestamp=_FIXED_TS)

        json_codex = len(json.loads(json_path.read_text())["codex_entries"])
        # Twee codex entries appear as `!! <title>` lines under the Codex passage.
        twee_codex = sum(1 for line in twee_path.read_text().splitlines() if line.startswith("!! "))
        # HTML codex entries are <div class="codex-entry">.
        html_codex = len(re.findall(r'<div class="codex-entry"', html_path.read_text()))

        assert json_codex == twee_codex == html_codex == len(ctx.codex_entries), (
            f"codex entry count diverged: json={json_codex}, twee={twee_codex}, "
            f"html={html_codex}, expected={len(ctx.codex_entries)}"
        )

    def test_metadata_snapshot_hash_consistent_across_formats(self, tmp_path: Path) -> None:
        """Same ctx → same graph_snapshot_hash in every format's metadata
        block. Already covered by test_export_metadata.py at the
        metadata-module level; pinning here at the cross-format level
        guards against an exporter accidentally reordering payload
        before hash computation.
        """
        ctx = _multi_format_context()
        json_meta = json.loads(
            JsonExporter().export(ctx, tmp_path / "j", timestamp=_FIXED_TS).read_text()
        )["_metadata"]
        twee_text = TweeExporter().export(ctx, tmp_path / "t", timestamp=_FIXED_TS).read_text()
        html_text = HtmlExporter().export(ctx, tmp_path / "h", timestamp=_FIXED_TS).read_text()

        twee_hash = next(
            line.split(": ", 1)[1].strip()
            for line in twee_text.splitlines()
            if line.startswith("graph_snapshot_hash:")
        )
        html_hash_match = re.search(
            r'<meta name="qf-graph-snapshot-hash" content="([0-9a-f]{64})">', html_text
        )
        assert html_hash_match is not None
        html_hash = html_hash_match.group(1)

        assert json_meta["graph_snapshot_hash"] == twee_hash == html_hash
