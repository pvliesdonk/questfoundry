"""Tests for asset bundling during SHIP export."""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export.assets import bundle_assets
from questfoundry.export.base import ExportIllustration

if TYPE_CHECKING:
    from pathlib import Path


def _make_illustration(
    passage_id: str = "p1", asset_path: str = "assets/img.png"
) -> ExportIllustration:
    return ExportIllustration(
        passage_id=passage_id,
        asset_path=asset_path,
        caption="Test image",
        category="scene",
    )


class TestBundleAssets:
    def test_copies_existing_asset(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        assets_dir = project / "assets"
        assets_dir.mkdir()
        (assets_dir / "img.png").write_bytes(b"fake-png-data")

        output = tmp_path / "output"
        ill = _make_illustration()
        copied = bundle_assets([ill], project, output)

        assert copied == 1
        assert (output / "assets" / "img.png").exists()
        assert (output / "assets" / "img.png").read_bytes() == b"fake-png-data"

    def test_skips_missing_asset(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        output = tmp_path / "output"
        ill = _make_illustration()
        copied = bundle_assets([ill], project, output)

        assert copied == 0
        assert not (output / "assets" / "img.png").exists()

    def test_empty_illustrations_list(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        output = tmp_path / "output"

        copied = bundle_assets([], project, output)

        assert copied == 0

    def test_nested_asset_path(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        (project / "assets" / "scenes").mkdir(parents=True)
        (project / "assets" / "scenes" / "castle.png").write_bytes(b"castle")

        output = tmp_path / "output"
        ill = _make_illustration(asset_path="assets/scenes/castle.png")
        copied = bundle_assets([ill], project, output)

        assert copied == 1
        assert (output / "assets" / "scenes" / "castle.png").exists()

    def test_multiple_assets(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        (project / "assets").mkdir(parents=True)
        (project / "assets" / "a.png").write_bytes(b"aaa")
        (project / "assets" / "b.png").write_bytes(b"bbb")

        output = tmp_path / "output"
        ills = [
            _make_illustration(passage_id="p1", asset_path="assets/a.png"),
            _make_illustration(passage_id="p2", asset_path="assets/b.png"),
        ]
        copied = bundle_assets(ills, project, output)

        assert copied == 2
        assert (output / "assets" / "a.png").read_bytes() == b"aaa"
        assert (output / "assets" / "b.png").read_bytes() == b"bbb"

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        """Asset paths that resolve outside the project are skipped."""
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "secret"
        outside.mkdir()
        (outside / "passwd.txt").write_bytes(b"secret")

        output = tmp_path / "output"
        ill = _make_illustration(asset_path="../secret/passwd.txt")
        copied = bundle_assets([ill], project, output)

        assert copied == 0
        assert not (output / ".." / "secret" / "passwd.txt").exists()

    def test_partial_missing(self, tmp_path: Path) -> None:
        """When some assets exist and some don't, copies what it can."""
        project = tmp_path / "project"
        (project / "assets").mkdir(parents=True)
        (project / "assets" / "exists.png").write_bytes(b"data")

        output = tmp_path / "output"
        ills = [
            _make_illustration(passage_id="p1", asset_path="assets/exists.png"),
            _make_illustration(passage_id="p2", asset_path="assets/missing.png"),
        ]
        copied = bundle_assets(ills, project, output)

        assert copied == 1
        assert (output / "assets" / "exists.png").exists()
        assert not (output / "assets" / "missing.png").exists()
