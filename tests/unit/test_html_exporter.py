"""Tests for standalone HTML exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export.base import (
    ExportChoice,
    ExportCodexEntry,
    ExportContext,
    ExportIllustration,
    ExportPassage,
)
from questfoundry.export.html_exporter import HtmlExporter

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


class TestHtmlExporter:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        assert result.exists()
        assert result.name == "story.html"

    def test_html_structure(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<title>Test Story</title>" in content
        assert "</html>" in content

    def test_passages_present(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert 'class="passage"' in content
        assert "You stand at the gates." in content
        assert "You enter the castle." in content

    def test_choices_present(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert 'class="choice"' in content
        assert "Enter the castle" in content
        assert 'data-target="passage--castle"' in content

    def test_start_passage_id(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert 'const startId = "passage--intro"' in content

    def test_codeword_requires(self, tmp_path: Path) -> None:
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
                    label="Enter",
                    requires=["codeword::has_key"],
                ),
            ],
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert "data-requires=" in content
        assert "codeword::has_key" in content

    def test_codeword_grants(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
                ExportPassage(id="p2", prose="End.", is_ending=True),
            ],
            choices=[
                ExportChoice(
                    from_passage="p1",
                    to_passage="p2",
                    label="Go",
                    grants=["codeword::visited"],
                ),
            ],
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert "data-grants=" in content
        assert "codeword::visited" in content

    def test_ending_marker(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert 'class="ending"' in content
        assert "The End" in content

    def test_illustration(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            illustrations=[
                ExportIllustration(
                    passage_id="p1",
                    asset_path="assets/abc.png",
                    caption="A gate.",
                    category="scene",
                ),
            ],
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert '<img src="assets/abc.png"' in content
        assert "A gate." in content

    def test_javascript_engine(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert "const codewords = new Set()" in content
        assert "function showPassage" in content
        assert "showPassage(startId)" in content

    def test_format_name(self) -> None:
        assert HtmlExporter.format_name == "html"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        nested = tmp_path / "deep" / "nested"
        result = exporter.export(_simple_context(), nested)

        assert result.exists()
        assert nested.exists()

    def test_codex_panel_rendered(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(
                    entity_id="Ancient Sword",
                    rank=1,
                    visible_when=["codeword::sword_found"],
                    content="A legendary blade.",
                ),
            ],
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert 'id="codex"' in content
        assert 'class="codex-entry"' in content
        assert "Ancient Sword" in content
        assert "A legendary blade." in content
        assert "data-visible-when=" in content
        assert "codeword::sword_found" in content

    def test_codex_toggle_button(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(entity_id="Lore", rank=1, content="Some lore."),
            ],
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert 'id="codex-toggle"' in content
        assert "updateCodexVisibility" in content

    def test_codex_not_rendered_when_empty(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert 'id="codex"' not in content
        assert 'id="codex-toggle"' not in content

    def test_art_direction_meta(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            art_direction={"palette": "dark fantasy", "mood": "brooding"},
        )
        exporter = HtmlExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert 'name="art-direction"' in content
        assert "dark fantasy" in content

    def test_art_direction_not_rendered_when_none(self, tmp_path: Path) -> None:
        exporter = HtmlExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert "art-direction" not in content
