"""Tests for Twee/SugarCube exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export.base import (
    ExportChoice,
    ExportCodexEntry,
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

    def test_codex_passage_rendered(self, tmp_path: Path) -> None:
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
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert ":: Codex" in content
        assert "!! Ancient Sword" in content
        assert "A legendary blade." in content
        assert "<<if $sword_found>>" in content
        assert "<</if>>" in content

    def test_codex_not_rendered_when_empty(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert ":: Codex" not in content

    def test_codex_entry_without_visible_when(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(
                    entity_id="World Lore",
                    rank=1,
                    visible_when=[],
                    content="Always visible lore.",
                ),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert ":: Codex" in content
        assert "!! World Lore" in content
        assert "Always visible lore." in content
        assert "<<if" not in content.split(":: Codex")[1]

    def test_codex_sorted_by_rank(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(entity_id="Zeta", rank=3, content="Third."),
                ExportCodexEntry(entity_id="Alpha", rank=1, content="First."),
                ExportCodexEntry(entity_id="Beta", rank=2, content="Second."),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        codex_section = content.split(":: Codex")[1]
        alpha_pos = codex_section.index("Alpha")
        beta_pos = codex_section.index("Beta")
        zeta_pos = codex_section.index("Zeta")
        assert alpha_pos < beta_pos < zeta_pos

    def test_art_direction_passage(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            art_direction={"palette": "dark fantasy", "mood": "brooding"},
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert ":: StoryArtDirection" in content
        assert "mood: brooding" in content
        assert "palette: dark fantasy" in content

    def test_art_direction_not_rendered_when_none(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert "StoryArtDirection" not in content

    def test_cover_rendered_in_story_init(self, tmp_path: Path) -> None:
        ctx = _simple_context()
        ctx.cover = ExportIllustration(
            passage_id="",
            asset_path="assets/cover.png",
            caption="",
            category="cover",
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        assert ":: StoryInit" in content
        assert "[img[Cover illustration|assets/cover.png]]" in content

    def test_no_story_init_without_cover(self, tmp_path: Path) -> None:
        exporter = TweeExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")
        content = result.read_text()

        assert ":: StoryInit" not in content

    def test_dutch_codex_passage_name(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            language="nl",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(entity_id="Held", rank=1, content="De held."),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        # Dutch uses "Codex" (same as English for nl)
        assert ":: Codex" in content

    def test_german_codex_passage_name(self, tmp_path: Path) -> None:
        ctx = ExportContext(
            title="Test",
            language="de",
            passages=[
                ExportPassage(id="p1", prose="Start.", is_start=True),
            ],
            choices=[],
            codex_entries=[
                ExportCodexEntry(entity_id="Held", rank=1, content="Der Held."),
            ],
        )
        exporter = TweeExporter()
        result = exporter.export(ctx, tmp_path / "out")
        content = result.read_text()

        # German uses "Kodex"
        assert ":: Kodex" in content
