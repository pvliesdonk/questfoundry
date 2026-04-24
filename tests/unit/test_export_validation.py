"""Tests for SHIP Phase 4 per-format validators (R-4.1 through R-4.4).

Each validator receives a generated file, parses it, and raises
ExportValidationError on internal-consistency failures (broken links,
missing metadata, page numbers out of range, etc.). Tests cover both
the happy path against real exporter output and forged broken files
to confirm the failure mode is loud, specific, and structurally
matches what R-4.4 mandates.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from questfoundry.export.base import (
    ExportChoice,
    ExportContext,
    ExportPassage,
)
from questfoundry.export.html_exporter import HtmlExporter
from questfoundry.export.json_exporter import JsonExporter
from questfoundry.export.twee_exporter import TweeExporter
from questfoundry.export.validation import (
    VALIDATORS,
    ExportValidationError,
    validate_export,
    validate_html,
    validate_json,
    validate_pdf,
    validate_twee,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ctx() -> ExportContext:
    return ExportContext(
        title="Test",
        passages=[
            ExportPassage(id="passage::start", prose="Begin.", is_start=True),
            ExportPassage(id="passage::end", prose="End.", is_ending=True),
        ],
        choices=[
            ExportChoice(
                from_passage="passage::start",
                to_passage="passage::end",
                label="Continue",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Twee
# ---------------------------------------------------------------------------


class TestValidateTwee:
    def test_valid_export_passes(self, tmp_path: Path) -> None:
        out = TweeExporter().export(_ctx(), tmp_path)
        validate_twee(out)  # no exception

    def test_broken_link_raises_with_target_name(self, tmp_path: Path) -> None:
        twee = tmp_path / "story.twee"
        twee.write_text(
            ":: StoryTitle\nT\n\n:: Start [start]\nHello.\n[[Continue->ghost_passage]]\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportValidationError, match=r"ghost_passage"):
            validate_twee(twee)

    def test_no_passage_headers_raises(self, tmp_path: Path) -> None:
        twee = tmp_path / "story.twee"
        twee.write_text("This file has no headers at all.\n", encoding="utf-8")
        with pytest.raises(ExportValidationError, match="no passage headers"):
            validate_twee(twee)

    def test_missing_start_raises(self, tmp_path: Path) -> None:
        twee = tmp_path / "story.twee"
        twee.write_text(":: somewhere\nHello.\n", encoding="utf-8")
        with pytest.raises(ExportValidationError, match="missing a `:: Start`"):
            validate_twee(twee)

    def test_goto_macro_form_validated(self, tmp_path: Path) -> None:
        """<<goto "target">> inside a <<link>> is a valid Twee link form."""
        twee = tmp_path / "story.twee"
        twee.write_text(
            ':: Start [start]\nHello.\n<<link "Go">><<goto "missing_target">><</link>>\n',
            encoding="utf-8",
        )
        with pytest.raises(ExportValidationError, match="missing_target"):
            validate_twee(twee)

    def test_reserved_headers_dont_count_as_broken(self, tmp_path: Path) -> None:
        """A link to e.g. `StoryTitle` shouldn't be flagged — but no
        real export does that. We test that referencing a reserved
        header by name (unusual but possible) doesn't trip the check.
        """
        twee = tmp_path / "story.twee"
        twee.write_text(
            ":: StoryTitle\nT\n\n:: Start [start]\nHello.\n[[ToTitle->StoryTitle]]\n",
            encoding="utf-8",
        )
        validate_twee(twee)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestValidateJson:
    def test_valid_export_passes(self, tmp_path: Path) -> None:
        out = JsonExporter().export(_ctx(), tmp_path)
        validate_json(out)

    def test_unparseable_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.json"
        bad.write_text("{not json,", encoding="utf-8")
        with pytest.raises(ExportValidationError, match="could not be parsed"):
            validate_json(bad)

    def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.json"
        bad.write_text(json.dumps({"title": "x", "passages": [], "choices": []}))
        with pytest.raises(ExportValidationError, match=r"_metadata"):
            validate_json(bad)

    def test_missing_metadata_field_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.json"
        bad.write_text(
            json.dumps(
                {
                    "_metadata": {"pipeline_version": "x"},  # missing 3 of 4
                    "title": "x",
                    "passages": [],
                    "choices": [],
                }
            )
        )
        with pytest.raises(ExportValidationError, match=r"_metadata.*missing"):
            validate_json(bad)

    def test_choice_to_undefined_passage_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.json"
        bad.write_text(
            json.dumps(
                {
                    "_metadata": {
                        "pipeline_version": "0.0.0",
                        "graph_snapshot_hash": "deadbeef",
                        "format_version": "1.0.0",
                        "generation_timestamp": "2026-04-24T00:00:00+00:00",
                    },
                    "title": "x",
                    "passages": [{"id": "passage::start"}],
                    "choices": [{"from_passage": "passage::start", "to_passage": "passage::ghost"}],
                }
            )
        )
        with pytest.raises(ExportValidationError, match=r"passage::ghost"):
            validate_json(bad)

    def test_top_level_not_object_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.json"
        bad.write_text("[1, 2, 3]")
        with pytest.raises(ExportValidationError, match="top-level is not an object"):
            validate_json(bad)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


class TestValidateHtml:
    def test_valid_export_passes(self, tmp_path: Path) -> None:
        out = HtmlExporter().export(_ctx(), tmp_path)
        validate_html(out)

    def test_no_body_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.html"
        bad.write_text("<html><head></head></html>", encoding="utf-8")
        with pytest.raises(ExportValidationError, match="no <body>"):
            validate_html(bad)

    def test_no_passage_divs_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.html"
        bad.write_text("<html><body><p>nothing here</p></body></html>", encoding="utf-8")
        with pytest.raises(ExportValidationError, match="no passage divs"):
            validate_html(bad)

    def test_broken_choice_target_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "story.html"
        bad.write_text(
            "<html><body>"
            '<div class="passage" id="p1">hello</div>'
            '<a class="choice" href="#" data-target="ghost">go</a>'
            "</body></html>",
            encoding="utf-8",
        )
        with pytest.raises(ExportValidationError, match="ghost"):
            validate_html(bad)


# ---------------------------------------------------------------------------
# PDF (sidecar-only — no WeasyPrint required)
# ---------------------------------------------------------------------------


class TestValidatePdf:
    def _write_sidecar(self, pdf_path: Path, page_map: dict[str, int]) -> None:
        sidecar = pdf_path.with_suffix(".pdf.map.json")
        sidecar.write_text(
            json.dumps(
                {
                    "_metadata": {
                        "pipeline_version": "0.0.0",
                        "graph_snapshot_hash": "deadbeef",
                        "format_version": "1.0.0",
                        "generation_timestamp": "2026-04-24T00:00:00+00:00",
                    },
                    "page_map": page_map,
                }
            )
        )

    def test_valid_sidecar_passes(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        self._write_sidecar(pdf, {"passage::a": 1, "passage::b": 2})
        validate_pdf(pdf)

    def test_missing_sidecar_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        with pytest.raises(ExportValidationError, match="missing its sidecar"):
            validate_pdf(pdf)

    def test_unparseable_sidecar_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        pdf.with_suffix(".pdf.map.json").write_text("{nope")
        with pytest.raises(ExportValidationError, match="could not be parsed"):
            validate_pdf(pdf)

    def test_empty_page_map_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        self._write_sidecar(pdf, {})
        with pytest.raises(ExportValidationError, match=r"page_map.*missing or empty"):
            validate_pdf(pdf)

    def test_out_of_range_page_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        # 2 distinct pages but one entry points to page 99
        self._write_sidecar(pdf, {"passage::a": 1, "passage::b": 99})
        with pytest.raises(ExportValidationError, match=r"outside the valid range"):
            validate_pdf(pdf)

    def test_duplicate_page_numbers_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "story.pdf"
        pdf.touch()
        self._write_sidecar(pdf, {"passage::a": 1, "passage::b": 1})
        with pytest.raises(ExportValidationError, match="duplicate page numbers"):
            validate_pdf(pdf)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestValidateExportDispatch:
    def test_dispatches_to_correct_validator(self, tmp_path: Path) -> None:
        out = JsonExporter().export(_ctx(), tmp_path)
        validate_export("json", out)

    def test_unknown_format_raises_keyerror(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            validate_export("xml", tmp_path / "story.xml")

    def test_all_known_formats_have_validators(self) -> None:
        """Every format the project exports must have a validator (R-4.1)."""
        from questfoundry.export import _EXPORTERS

        for name in _EXPORTERS:
            assert name in VALIDATORS, f"format {name!r} is exportable but has no validator"
