"""Tests for embedding export assets as data URLs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export.assets import embed_assets
from questfoundry.export.base import ExportIllustration

if TYPE_CHECKING:
    from pathlib import Path


def _write_test_png(path: Path) -> None:
    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc\xf8"
        b"\xff\xff?\x00\x05\xfe\x02\xfeA\xe2!\xbc\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_illustration(asset_path: str) -> ExportIllustration:
    return ExportIllustration(
        passage_id="passage::intro",
        asset_path=asset_path,
        caption="Caption",
        category="scene",
    )


def test_embed_assets_embeds_cover_and_scene(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scene_path = project / "assets" / "scene.png"
    cover_path = project / "assets" / "cover.png"
    _write_test_png(scene_path)
    _write_test_png(cover_path)

    illustrations = [_make_illustration("assets/scene.png")]
    cover = ExportIllustration(
        passage_id="",
        asset_path="assets/cover.png",
        caption="Cover",
        category="cover",
    )

    embedded, embedded_cover = embed_assets(illustrations, cover, project)

    assert embedded[0].asset_path.startswith("data:image/png;base64,")
    assert embedded_cover is not None
    assert embedded_cover.asset_path.startswith("data:image/png;base64,")


def test_embed_assets_skips_unsupported_and_outside(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "assets").mkdir(parents=True, exist_ok=True)
    (project / "assets" / "note.txt").write_text("hello", encoding="utf-8")

    illustrations = [
        _make_illustration("assets/note.txt"),
        _make_illustration("../secret/steal.png"),
        _make_illustration(""),
    ]

    embedded, embedded_cover = embed_assets(illustrations, None, project)

    assert embedded_cover is None
    assert embedded[0].asset_path == "assets/note.txt"
    assert embedded[1].asset_path == "../secret/steal.png"
    assert embedded[2].asset_path == ""
