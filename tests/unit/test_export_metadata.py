"""Tests for the deterministic export metadata header (R-3.6).

Covers:

- ``compute_graph_snapshot_hash`` is deterministic and changes when any
  observable field of the ExportContext changes.
- ``build_export_metadata`` populates all four R-3.6 fields and accepts
  an injected timestamp for test stability.
- Each exporter (JSON, Twee, HTML) embeds the metadata in the appropriate
  format-specific shape, with the timestamp seam producing identical
  output across two runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from questfoundry.export.base import ExportContext, ExportPassage
from questfoundry.export.html_exporter import HtmlExporter
from questfoundry.export.json_exporter import JsonExporter
from questfoundry.export.metadata import (
    ExportMetadata,
    build_export_metadata,
    compute_graph_snapshot_hash,
    get_pipeline_version,
)
from questfoundry.export.twee_exporter import TweeExporter

if TYPE_CHECKING:
    from pathlib import Path


_FIXED_TS = "2026-04-24T00:00:00+00:00"


def _ctx(prose: str = "Hello.") -> ExportContext:
    """Minimal-but-real context: one start passage, no choices."""
    return ExportContext(
        title="test",
        passages=[ExportPassage(id="passage::start", prose=prose, is_start=True)],
        choices=[],
    )


# ---------------------------------------------------------------------------
# metadata module
# ---------------------------------------------------------------------------


class TestGraphSnapshotHash:
    def test_same_context_same_hash(self) -> None:
        h1 = compute_graph_snapshot_hash(_ctx("A"))
        h2 = compute_graph_snapshot_hash(_ctx("A"))
        assert h1 == h2
        # SHA256 hex
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_changed_prose_changes_hash(self) -> None:
        h1 = compute_graph_snapshot_hash(_ctx("A"))
        h2 = compute_graph_snapshot_hash(_ctx("B"))
        assert h1 != h2

    def test_changed_title_changes_hash(self) -> None:
        ctx_a = _ctx()
        ctx_b = _ctx()
        ctx_b.title = "different"
        assert compute_graph_snapshot_hash(ctx_a) != compute_graph_snapshot_hash(ctx_b)


class TestBuildExportMetadata:
    def test_all_four_fields_populated(self) -> None:
        meta = build_export_metadata(_ctx(), "1.2.3", timestamp=_FIXED_TS)
        assert isinstance(meta, ExportMetadata)
        assert meta.pipeline_version  # non-empty (sentinel or real)
        assert len(meta.graph_snapshot_hash) == 64
        assert meta.format_version == "1.2.3"
        assert meta.generation_timestamp == _FIXED_TS

    def test_to_dict_roundtrips(self) -> None:
        meta = build_export_metadata(_ctx(), "1.0.0", timestamp=_FIXED_TS)
        d = meta.to_dict()
        assert set(d.keys()) == {
            "pipeline_version",
            "graph_snapshot_hash",
            "format_version",
            "generation_timestamp",
        }
        assert d["generation_timestamp"] == _FIXED_TS

    def test_default_timestamp_is_iso8601_with_tz(self) -> None:
        meta = build_export_metadata(_ctx(), "1.0.0")
        # Just confirm the default produces an ISO-formatted string with timezone
        assert "T" in meta.generation_timestamp
        assert "+" in meta.generation_timestamp or meta.generation_timestamp.endswith("Z")

    def test_pipeline_version_returns_string(self) -> None:
        v = get_pipeline_version()
        assert isinstance(v, str)
        assert v


# ---------------------------------------------------------------------------
# Per-exporter metadata embedding
# ---------------------------------------------------------------------------


class TestJsonExporterMetadata:
    def test_metadata_block_top_level(self, tmp_path: Path) -> None:
        out = JsonExporter().export(_ctx(), tmp_path, timestamp=_FIXED_TS)
        payload = json.loads(out.read_text())
        assert "_metadata" in payload
        meta = payload["_metadata"]
        assert meta["format_version"] == JsonExporter.format_version
        assert meta["generation_timestamp"] == _FIXED_TS
        assert len(meta["graph_snapshot_hash"]) == 64
        # Story payload still present
        assert "passages" in payload

    def test_byte_identical_with_fixed_timestamp(self, tmp_path: Path) -> None:
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        out1 = JsonExporter().export(_ctx(), d1, timestamp=_FIXED_TS)
        out2 = JsonExporter().export(_ctx(), d2, timestamp=_FIXED_TS)
        assert out1.read_bytes() == out2.read_bytes()


class TestTweeExporterMetadata:
    def test_metadata_passage_present(self, tmp_path: Path) -> None:
        out = TweeExporter().export(_ctx(), tmp_path, timestamp=_FIXED_TS)
        text = out.read_text()
        assert ":: StoryMetadata" in text
        assert f"generation_timestamp: {_FIXED_TS}" in text
        assert f"format_version: {TweeExporter.format_version}" in text
        assert "graph_snapshot_hash:" in text
        assert "pipeline_version:" in text

    def test_byte_identical_with_fixed_timestamp(self, tmp_path: Path) -> None:
        # Twee header includes a random IFID, so we cannot expect full byte
        # equality. But the metadata block itself MUST be deterministic.
        out1 = TweeExporter().export(_ctx(), tmp_path / "a", timestamp=_FIXED_TS)
        out2 = TweeExporter().export(_ctx(), tmp_path / "b", timestamp=_FIXED_TS)

        def _metadata_block(text: str) -> str:
            lines = text.splitlines()
            start = next(i for i, line in enumerate(lines) if ":: StoryMetadata" in line)
            return "\n".join(lines[start : start + 5])

        assert _metadata_block(out1.read_text()) == _metadata_block(out2.read_text())


class TestHtmlExporterMetadata:
    def test_meta_tags_emitted(self, tmp_path: Path) -> None:
        out = HtmlExporter().export(_ctx(), tmp_path, timestamp=_FIXED_TS)
        text = out.read_text()
        assert '<meta name="qf-format-version"' in text
        assert f'content="{HtmlExporter.format_version}"' in text
        assert '<meta name="qf-generation-timestamp"' in text
        assert f'content="{_FIXED_TS}"' in text
        assert '<meta name="qf-graph-snapshot-hash"' in text
        assert '<meta name="qf-pipeline-version"' in text

    def test_byte_identical_with_fixed_timestamp(self, tmp_path: Path) -> None:
        out1 = HtmlExporter().export(_ctx(), tmp_path / "a", timestamp=_FIXED_TS)
        out2 = HtmlExporter().export(_ctx(), tmp_path / "b", timestamp=_FIXED_TS)
        assert out1.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------
# All exporters share a snapshot hash for the same context
# ---------------------------------------------------------------------------


def test_snapshot_hash_consistent_across_formats(tmp_path: Path) -> None:
    """JSON, Twee, and HTML compute the SAME graph_snapshot_hash for one ctx."""
    ctx = _ctx("Shared content.")

    json_out = json.loads(
        JsonExporter().export(ctx, tmp_path / "j", timestamp=_FIXED_TS).read_text()
    )
    twee_text = TweeExporter().export(ctx, tmp_path / "t", timestamp=_FIXED_TS).read_text()
    html_text = HtmlExporter().export(ctx, tmp_path / "h", timestamp=_FIXED_TS).read_text()

    json_hash = json_out["_metadata"]["graph_snapshot_hash"]
    twee_hash_line = next(
        line for line in twee_text.splitlines() if line.startswith("graph_snapshot_hash:")
    )
    twee_hash = twee_hash_line.split(": ", 1)[1].strip()

    # Pull HTML hash from the meta tag
    import re

    html_hash_match = re.search(
        r'<meta name="qf-graph-snapshot-hash" content="([0-9a-f]{64})">', html_text
    )
    assert html_hash_match is not None
    html_hash = html_hash_match.group(1)

    assert json_hash == twee_hash == html_hash


# ---------------------------------------------------------------------------
# fixture: confirm integration without weasyprint by exercising compute path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format_version", ["0.1.0", "1.0.0", "2.5.3"])
def test_format_version_passed_through(format_version: str) -> None:
    meta = build_export_metadata(_ctx(), format_version, timestamp=_FIXED_TS)
    assert meta.format_version == format_version
