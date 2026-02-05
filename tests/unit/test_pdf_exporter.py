"""Tests for gamebook-style PDF exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from questfoundry.export.base import (
    ExportChoice,
    ExportCodeword,
    ExportCodexEntry,
    ExportContext,
    ExportIllustration,
    ExportPassage,
)
from questfoundry.export.pdf_exporter import (
    PdfExporter,
    _build_passage_numbering,
    _format_codeword_name,
    _render_codeword_checklist,
    _render_codex,
    _render_html,
)

if TYPE_CHECKING:
    from pathlib import Path


def _simple_context() -> ExportContext:
    """Build a minimal ExportContext for testing."""
    return ExportContext(
        title="Test Story",
        passages=[
            ExportPassage(id="passage::intro", prose="You stand at the gates.", is_start=True),
            ExportPassage(id="passage::castle", prose="You enter the castle.", is_ending=True),
        ],
        choices=[
            ExportChoice(
                from_passage="passage::intro",
                to_passage="passage::castle",
                label="Enter the castle",
            ),
        ],
    )


def _multi_passage_context() -> ExportContext:
    """Build a context with many passages for numbering tests."""
    passages = [
        ExportPassage(id="passage::start", prose="Start here.", is_start=True),
        ExportPassage(id="passage::alpha", prose="Alpha passage."),
        ExportPassage(id="passage::beta", prose="Beta passage."),
        ExportPassage(id="passage::gamma", prose="Gamma passage."),
        ExportPassage(id="passage::end", prose="The end.", is_ending=True),
    ]
    choices = [
        ExportChoice(from_passage="passage::start", to_passage="passage::alpha", label="Go alpha"),
        ExportChoice(from_passage="passage::alpha", to_passage="passage::beta", label="Go beta"),
        ExportChoice(from_passage="passage::beta", to_passage="passage::gamma", label="Go gamma"),
        ExportChoice(from_passage="passage::gamma", to_passage="passage::end", label="Finish"),
    ]
    return ExportContext(title="Multi Test", passages=passages, choices=choices)


class TestPassageNumbering:
    """Tests for _build_passage_numbering function."""

    def test_empty_passages_returns_empty_dict(self) -> None:
        result = _build_passage_numbering([])
        assert result == {}

    def test_start_passage_is_always_one(self) -> None:
        passages = [
            ExportPassage(id="z_last", prose="Last."),
            ExportPassage(id="a_first", prose="First.", is_start=True),
            ExportPassage(id="m_middle", prose="Middle."),
        ]
        numbering = _build_passage_numbering(passages)

        assert numbering["a_first"] == 1

    def test_all_passages_get_unique_numbers(self) -> None:
        ctx = _multi_passage_context()
        numbering = _build_passage_numbering(ctx.passages)

        # All passages should have a number
        assert len(numbering) == len(ctx.passages)

        # All numbers should be unique
        numbers = list(numbering.values())
        assert len(set(numbers)) == len(numbers)

        # Numbers should be 1..N
        assert sorted(numbers) == list(range(1, len(ctx.passages) + 1))

    def test_numbering_is_randomized(self) -> None:
        """Non-start passages should not be numbered in sorted ID order."""
        passages = [
            ExportPassage(id="passage::aaa", prose="A passage.", is_start=True),
            ExportPassage(id="passage::bbb", prose="B passage."),
            ExportPassage(id="passage::ccc", prose="C passage."),
            ExportPassage(id="passage::ddd", prose="D passage."),
            ExportPassage(id="passage::eee", prose="E passage."),
            ExportPassage(id="passage::fff", prose="F passage."),
            ExportPassage(id="passage::ggg", prose="G passage."),
        ]
        numbering = _build_passage_numbering(passages)

        # Get numbers for non-start passages in ID order
        non_start_ids = sorted(p.id for p in passages if not p.is_start)
        non_start_numbers = [numbering[pid] for pid in non_start_ids]

        # With enough passages, it's extremely unlikely they'd be in order 2,3,4,5,6,7
        assert non_start_numbers != list(range(2, 8)), "Numbering appears to not be randomized"

    def test_numbering_is_deterministic(self) -> None:
        """Same passages should produce same numbering every time."""
        ctx = _multi_passage_context()

        numbering1 = _build_passage_numbering(ctx.passages)
        numbering2 = _build_passage_numbering(ctx.passages)
        numbering3 = _build_passage_numbering(ctx.passages)

        assert numbering1 == numbering2 == numbering3

    def test_single_passage(self) -> None:
        passages = [ExportPassage(id="only", prose="Only passage.", is_start=True)]
        numbering = _build_passage_numbering(passages)

        assert numbering == {"only": 1}

    def test_no_start_passage_uses_first(self) -> None:
        """When no passage has is_start=True, first passage gets number 1."""
        passages = [
            ExportPassage(id="passage::alpha", prose="Alpha."),
            ExportPassage(id="passage::beta", prose="Beta."),
            ExportPassage(id="passage::gamma", prose="Gamma."),
        ]
        numbering = _build_passage_numbering(passages)

        # First passage (alphabetically sorted) should be 1
        assert numbering["passage::alpha"] == 1
        # All passages should have unique numbers
        assert len(set(numbering.values())) == 3


class TestCodewordNameFormatting:
    """Tests for _format_codeword_name helper."""

    def test_strips_codeword_prefix(self) -> None:
        assert _format_codeword_name("codeword::golden_key") == "Golden Key"

    def test_handles_no_prefix(self) -> None:
        assert _format_codeword_name("secret_door") == "Secret Door"

    def test_converts_snake_case(self) -> None:
        assert _format_codeword_name("ancient_blade_found") == "Ancient Blade Found"

    def test_single_word(self) -> None:
        assert _format_codeword_name("visited") == "Visited"


class TestRenderHtml:
    """Tests for HTML rendering (no WeasyPrint required)."""

    def test_html_structure(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Story</title>" in html
        assert "</html>" in html
        assert 'lang="en"' in html

    def test_title_page_rendered(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="title-page"' in html
        assert "<h1>Test Story</h1>" in html

    def test_instructions_page_rendered(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="instructions"' in html
        assert "How to Play" in html
        assert "section 1" in html

    def test_passages_have_section_numbers(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="passage-number"' in html
        assert "— 1 —" in html  # Start passage

    def test_prose_content_rendered(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert "You stand at the gates." in html
        assert "You enter the castle." in html

    def test_choices_with_go_to_format(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        choices_by_passage = {"passage::intro": ctx.choices}
        html = _render_html(ctx, numbering, choices_by_passage, {})

        assert "Enter the castle" in html
        assert "go to" in html
        assert 'class="choices"' in html

    def test_ending_shows_the_end(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="ending"' in html
        assert "The End" in html

    def test_conditional_choice_shows_codeword(self) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
                ExportPassage(id="p2", prose="Secret.", is_ending=True),
            ],
            choices=[
                ExportChoice(
                    from_passage="p1",
                    to_passage="p2",
                    label="Open the door",
                    requires=["codeword::golden_key"],
                ),
            ],
        )
        numbering = _build_passage_numbering(ctx.passages)
        choices_by_passage = {"p1": ctx.choices}
        html = _render_html(ctx, numbering, choices_by_passage, {})

        assert "If you have" in html
        assert "Golden Key" in html

    def test_illustration_rendered(self) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[ExportPassage(id="p1", prose="Start.", is_start=True)],
            choices=[],
            illustrations=[
                ExportIllustration(
                    passage_id="p1",
                    asset_path="assets/gate.png",
                    caption="The ancient gate",
                    category="scene",
                ),
            ],
        )
        numbering = _build_passage_numbering(ctx.passages)
        illustrations_by_passage = {ill.passage_id: ill for ill in ctx.illustrations}
        html = _render_html(ctx, numbering, {}, illustrations_by_passage)

        assert 'class="passage-illustration"' in html
        assert "assets/gate.png" in html
        assert "The ancient gate" in html

    def test_cover_on_title_page(self) -> None:
        ctx = _simple_context()
        ctx.cover = ExportIllustration(
            passage_id="",
            asset_path="assets/cover.png",
            caption="Cover art",
            category="cover",
        )
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="cover-image"' in html
        assert "assets/cover.png" in html
        assert "Cover art" in html

    def test_genre_subtitle_from_art_direction(self) -> None:
        ctx = _simple_context()
        ctx.art_direction = {"genre": "Dark Fantasy"}
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="subtitle"' in html
        assert "Dark Fantasy" in html

    def test_dutch_language_ui_strings(self) -> None:
        ctx = ExportContext(
            title="Test",
            language="nl",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True, is_ending=True),
            ],
            choices=[],
        )
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'lang="nl"' in html
        assert "Einde" in html  # Dutch "The End"


class TestCodewordChecklist:
    """Tests for codeword checklist appendix rendering."""

    def test_checklist_rendered(self) -> None:
        codewords = [
            ExportCodeword(id="codeword::golden_key", codeword_type="item"),
            ExportCodeword(id="codeword::secret_passage", codeword_type="discovery"),
        ]
        ui = {"codeword_checklist": "Codeword Checklist"}
        html = _render_codeword_checklist(codewords, ui)

        assert 'class="appendix codeword-checklist"' in html
        assert "Codeword Checklist" in html
        assert "Golden Key" in html
        assert "Secret Passage" in html
        assert 'class="codeword-checkbox"' in html

    def test_codewords_sorted_alphabetically(self) -> None:
        codewords = [
            ExportCodeword(id="codeword::zebra", codeword_type="item"),
            ExportCodeword(id="codeword::alpha", codeword_type="item"),
            ExportCodeword(id="codeword::middle", codeword_type="item"),
        ]
        ui = {"codeword_checklist": "Checklist"}
        html = _render_codeword_checklist(codewords, ui)

        alpha_pos = html.find("Alpha")
        middle_pos = html.find("Middle")
        zebra_pos = html.find("Zebra")

        assert alpha_pos < middle_pos < zebra_pos

    def test_checklist_in_full_html_when_codewords_present(self) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[ExportPassage(id="p1", prose="Start.", is_start=True)],
            choices=[],
            codewords=[
                ExportCodeword(id="codeword::key", codeword_type="item"),
            ],
        )
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert "codeword-checklist" in html

    def test_no_checklist_when_no_codewords(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        # CSS class appears in stylesheet, but the actual section element should not exist
        assert '<section class="appendix codeword-checklist">' not in html


class TestCodexAppendix:
    """Tests for codex appendix rendering."""

    def test_codex_rendered(self) -> None:
        entries = [
            ExportCodexEntry(
                entity_id="sword",
                title="Ancient Sword",
                rank=1,
                content="A legendary blade forged by the ancients.",
            ),
        ]
        ui = {"codex": "Codex", "requires_codeword": "Requires"}
        html = _render_codex(entries, ui)

        assert 'class="appendix codex"' in html
        assert "Ancient Sword" in html
        assert "A legendary blade forged by the ancients." in html

    def test_codex_entries_sorted_by_rank_then_title(self) -> None:
        entries = [
            ExportCodexEntry(entity_id="z", title="Zebra", rank=2, content="Z."),
            ExportCodexEntry(entity_id="a", title="Alpha", rank=1, content="A."),
            ExportCodexEntry(entity_id="b", title="Beta", rank=1, content="B."),
        ]
        ui = {"codex": "Codex", "requires_codeword": "Requires"}
        html = _render_codex(entries, ui)

        alpha_pos = html.find("Alpha")
        beta_pos = html.find("Beta")
        zebra_pos = html.find("Zebra")

        # Rank 1 entries first (alpha, beta), then rank 2 (zebra)
        assert alpha_pos < zebra_pos
        assert beta_pos < zebra_pos

    def test_spoiler_entries_filtered_out(self) -> None:
        """Entries with visible_when restrictions are excluded from PDF."""
        entries = [
            ExportCodexEntry(
                entity_id="public",
                title="Public Entry",
                rank=1,
                content="Visible to all.",
            ),
            ExportCodexEntry(
                entity_id="secret",
                title="Secret Entry",
                rank=1,
                visible_when=["codeword::discovery"],
                content="Hidden knowledge.",
            ),
        ]
        ui = {"codex": "Codex"}
        html = _render_codex(entries, ui)

        assert "Public Entry" in html
        assert "Secret Entry" not in html

    def test_returns_empty_when_all_entries_are_spoilers(self) -> None:
        """Returns empty string when only spoiler entries exist."""
        entries = [
            ExportCodexEntry(
                entity_id="secret",
                title="Secret",
                rank=1,
                visible_when=["codeword::x"],
                content="Hidden.",
            ),
        ]
        ui = {"codex": "Codex"}
        html = _render_codex(entries, ui)

        assert html == ""

    def test_codex_in_full_html_when_entries_present(self) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[ExportPassage(id="p1", prose="Start.", is_start=True)],
            choices=[],
            codex_entries=[
                ExportCodexEntry(entity_id="lore", title="Lore", rank=1, content="Some lore."),
            ],
        )
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="appendix codex"' in html

    def test_no_codex_when_no_entries(self) -> None:
        ctx = _simple_context()
        numbering = _build_passage_numbering(ctx.passages)
        html = _render_html(ctx, numbering, {}, {})

        assert 'class="appendix codex"' not in html


def _weasyprint_available() -> bool:
    """Check if WeasyPrint is installed."""
    try:
        import weasyprint  # noqa: F401

        return True
    except ImportError:
        return False


class TestPdfExporter:
    """Tests for PdfExporter class."""

    def test_format_name(self) -> None:
        assert PdfExporter.format_name == "pdf"

    @pytest.mark.skipif(
        not _weasyprint_available(),
        reason="WeasyPrint not installed (optional dependency)",
    )
    def test_creates_output_file(self, tmp_path: Path) -> None:
        exporter = PdfExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        assert result.exists()
        assert result.name == "story.pdf"

    @pytest.mark.skipif(
        not _weasyprint_available(),
        reason="WeasyPrint not installed (optional dependency)",
    )
    def test_creates_output_directory(self, tmp_path: Path) -> None:
        exporter = PdfExporter()
        nested = tmp_path / "deep" / "nested"
        result = exporter.export(_simple_context(), nested)

        assert result.exists()
        assert nested.exists()

    def test_raises_import_error_without_weasyprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify helpful error when WeasyPrint is not installed."""
        # Mock the import to fail
        import builtins
        from typing import Any

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "weasyprint":
                raise ImportError("No module named 'weasyprint'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        exporter = PdfExporter()
        with pytest.raises(ImportError, match="WeasyPrint is required"):
            exporter.export(_simple_context(), tmp_path / "out")
