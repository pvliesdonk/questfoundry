"""Tests for Twee/SugarCube exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export.base import (
    ExportChoice,
    ExportContext,
    ExportIllustration,
    ExportPassage,
)
from questfoundry.export.twee_exporter import TweeExporter

if TYPE_CHECKING:
    from pathlib import Path


def _simple_context() -> ExportContext:
    """Build a minimal ExportContext for testing."""
    return ExportContext(
        title="Test Story",
        passages=[
            ExportPassage(id="passage::intro", prose="You stand at the gates.", is_start=True),
            ExportPassage(id="passage::castle", prose="You enter the castle.", is_ending=True),
            ExportPassage(id="passage::forest", prose="You flee into the forest.", is_ending=True),
        ],
        choices=[
            ExportChoice(
                from_passage="passage::intro",
                to_passage="passage::castle",
                label="Enter the castle",
            ),
            ExportChoice(
                from_passage="passage::intro",
                to_passage="passage::forest",
                label="Flee to the forest",
            ),
        ],
    )


class TestTweeExporter:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        assert result.exists()
        assert result.name == "story.twee"

    def test_story_title(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert ":: StoryTitle\nTest Story" in content

    def test_story_data_header(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert ":: StoryData" in content
        assert '"format": "SugarCube"' in content

    def test_start_passage(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert ":: Start [start]" in content
        assert "You stand at the gates." in content

    def test_passage_names_stripped(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        # passage:: prefix should be stripped
        assert ":: castle" in content
        assert ":: forest" in content
        assert "passage::" not in content.split(":: StoryData")[1]  # Not in passage names

    def test_simple_choices(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert "[[Enter the castle->castle]]" in content
        assert "[[Flee to the forest->forest]]" in content

    def test_conditional_choice(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
                ExportPassage(id="p2", prose="Secret room.", is_ending=True),
            ],
            choices=[
                ExportChoice(
                    from_passage="p1",
                    to_passage="p2",
                    label="Enter secret room",
                    requires=["codeword::has_key"],
                ),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert "<<if $has_key>>[[Enter secret room->p2]]<</if>>" in content

    def test_choice_with_grants(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
                ExportPassage(id="p2", prose="Castle.", is_ending=True),
            ],
            choices=[
                ExportChoice(
                    from_passage="p1",
                    to_passage="p2",
                    label="Take the sword",
                    grants=["codeword::has_sword"],
                ),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        # Should use <<link>> macro with <<set>> and <<goto>>
        assert (
            '<<link "Take the sword">><<set $has_sword to true>><<goto "p2">><</link>>' in content
        )

    def test_choice_with_requires_and_grants(self, tmp_path: Path) -> None:
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
                    label="Use key",
                    requires=["codeword::has_key"],
                    grants=["codeword::door_opened"],
                ),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert "<<if $has_key>>" in content
        assert '<<link "Use key">><<set $door_opened to true>><<goto "p2">><</link>>' in content
        assert "<</if>>" in content

    def test_illustration(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="passage::intro", prose="Start.", is_start=True),
            ],
            choices=[],
            illustrations=[
                ExportIllustration(
                    passage_id="passage::intro",
                    asset_path="assets/abc123.png",
                    caption="A dark gate.",
                    category="scene",
                ),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert "[img[assets/abc123.png]]" in content
        assert "//A dark gate.//" in content

    def test_format_name(self) -> None:
        assert TweeExporter.format_name == "twee"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        nested = tmp_path / "deep" / "nested"
        result = exporter.export(_simple_context(), nested)

        assert result.exists()
        assert nested.exists()
