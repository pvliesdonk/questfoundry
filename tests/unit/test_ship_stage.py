"""Tests for SHIP stage (deterministic export)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.ship import ShipStage, ShipStageError

if TYPE_CHECKING:
    from pathlib import Path


def _create_project_with_graph(project_path: Path, *, with_prose: bool = True) -> None:
    """Create a minimal project with graph.db for SHIP testing."""
    from ruamel.yaml import YAML

    project_path.mkdir(parents=True, exist_ok=True)

    # project.yaml
    yaml_writer = YAML()
    config = {"name": "test-story", "version": "1.0", "providers": {"default": "ollama/test"}}
    with (project_path / "project.yaml").open("w") as f:
        yaml_writer.dump(config, f)

    # Build a graph with passages and choices
    g = Graph()
    g.create_node(
        "passage::intro",
        {
            "type": "passage",
            "raw_id": "intro",
            "prose": "You stand at the gates." if with_prose else None,
        },
    )
    g.create_node(
        "passage::castle",
        {
            "type": "passage",
            "raw_id": "castle",
            "prose": "You enter the castle." if with_prose else None,
            "is_ending": True,
        },
    )
    g.create_node(
        "choice::enter",
        {
            "type": "choice",
            "from_passage": "passage::intro",
            "to_passage": "passage::castle",
            "label": "Enter the castle",
        },
    )
    g.add_edge("choice_from", "choice::enter", "passage::intro")
    g.add_edge("choice_to", "choice::enter", "passage::castle")

    g.save(project_path / "graph.db")


def _write_test_png(path: Path) -> None:
    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc\xf8"
        b"\xff\xff?\x00\x05\xfe\x02\xfeA\xe2!\xbc\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class TestShipStage:
    def test_export_twee(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        result = stage.execute(export_format="twee")

        assert result.exists()
        assert result.suffix == ".twee"
        content = result.read_text()
        assert ":: StoryTitle" in content

    def test_export_json(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        assert result.exists()
        assert result.suffix == ".json"

    def test_export_html(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        result = stage.execute(export_format="html")

        assert result.exists()
        assert result.name == "story.html"
        content = result.read_text()
        assert "<!DOCTYPE html>" in content

    def test_export_html_with_embedded_assets(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        asset_path = project / "assets" / "scene.png"
        _write_test_png(asset_path)

        g = Graph.load(project)
        g.create_node(
            "illustration::intro",
            {
                "type": "illustration",
                "raw_id": "intro",
                "asset": "assets/scene.png",
                "caption": "A scene",
                "category": "scene",
            },
        )
        g.add_edge("Depicts", "illustration::intro", "passage::intro")
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="html", embed_assets=True)

        content = result.read_text()
        assert "data:image/png;base64," in content

        exported_asset = result.parent / "assets" / "scene.png"
        assert not exported_asset.exists()

    def test_export_json_with_embed_assets_still_bundles(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        asset_path = project / "assets" / "scene.png"
        _write_test_png(asset_path)

        g = Graph.load(project)
        g.create_node(
            "illustration::intro",
            {
                "type": "illustration",
                "raw_id": "intro",
                "asset": "assets/scene.png",
                "caption": "A scene",
                "category": "scene",
            },
        )
        g.add_edge("Depicts", "illustration::intro", "passage::intro")
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="json", embed_assets=True)

        exported_asset = result.parent / "assets" / "scene.png"
        assert exported_asset.exists()

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)
        custom_dir = tmp_path / "custom-output"

        stage = ShipStage(project)
        result = stage.execute(export_format="json", output_dir=custom_dir)

        assert result.parent == custom_dir

    def test_default_output_dir(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        result = stage.execute(export_format="twee")

        assert "exports" in str(result)
        assert "twee" in str(result)

    def test_missing_graph_raises(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        project.mkdir(parents=True)

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="no passages"):
            stage.execute()

    def test_missing_prose_raises(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project, with_prose=False)

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="missing prose"):
            stage.execute()

    def test_empty_graph_raises(self, tmp_path: Path) -> None:
        from ruamel.yaml import YAML

        project = tmp_path / "my-story"
        project.mkdir(parents=True)

        yaml_writer = YAML()
        config = {"name": "test", "version": "1.0", "providers": {"default": "ollama/test"}}
        with (project / "project.yaml").open("w") as f:
            yaml_writer.dump(config, f)

        # Graph with no passages
        g = Graph()
        g.save(project / "graph.db")

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="no passages"):
            stage.execute()

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="Unknown export format"):
            stage.execute(export_format="docx")

    def test_whitespace_prose_raises(self, tmp_path: Path) -> None:
        """Whitespace-only prose should be rejected."""
        from ruamel.yaml import YAML

        project = tmp_path / "my-story"
        project.mkdir(parents=True, exist_ok=True)

        yaml_writer = YAML()
        config = {"name": "test", "version": "1.0", "providers": {"default": "ollama/test"}}
        with (project / "project.yaml").open("w") as f:
            yaml_writer.dump(config, f)

        g = Graph()
        g.create_node("passage::intro", {"type": "passage", "raw_id": "intro", "prose": "   "})
        g.save(project / "graph.db")

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="missing prose"):
            stage.execute()

    def test_project_name_from_config(self, tmp_path: Path) -> None:
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        import json

        data = json.loads(result.read_text())
        assert data["title"] == "test-story"

    def test_project_name_fallback(self, tmp_path: Path) -> None:
        """Project name falls back to directory name if config is missing."""
        project = tmp_path / "my-fallback-story"
        project.mkdir(parents=True, exist_ok=True)

        g = Graph()
        g.create_node("passage::intro", {"type": "passage", "raw_id": "intro", "prose": "Hello."})
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        import json

        data = json.loads(result.read_text())
        assert data["title"] == "my-fallback-story"

    def test_graph_title_takes_priority(self, tmp_path: Path) -> None:
        """Story title from voice node (FILL) takes priority over project config name."""
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        # Add a voice node with a generated story title
        g = Graph.load(project)
        g.create_node(
            "voice::voice",
            {"type": "voice", "raw_id": "voice", "story_title": "The Hollow Crown"},
        )
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        import json

        data = json.loads(result.read_text())
        assert data["title"] == "The Hollow Crown"

    def test_graph_title_fallback_to_config(self, tmp_path: Path) -> None:
        """Falls back to config name when voice node has no story_title."""
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        # Voice node exists but without story_title
        g = Graph.load(project)
        g.create_node("voice::voice", {"type": "voice", "raw_id": "voice"})
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        import json

        data = json.loads(result.read_text())
        assert data["title"] == "test-story"

    def test_graph_title_none_fallback_to_config(self, tmp_path: Path) -> None:
        """Falls back to config name when story_title is explicitly None."""
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        g = Graph.load(project)
        g.create_node("voice::voice", {"type": "voice", "raw_id": "voice", "story_title": None})
        g.save(project / "graph.db")

        stage = ShipStage(project)
        result = stage.execute(export_format="json")

        import json

        data = json.loads(result.read_text())
        assert data["title"] == "test-story"


class TestShipPhase4Validation:
    """R-4.1 through R-4.4: SHIP halts with ERROR before delivering a
    bundle that fails per-format validation. The validator failure
    message must be embedded in the raised ShipStageError so the user
    can see exactly what's broken without spelunking through logs.
    """

    def test_validation_failure_halts_ship(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A validator that raises must surface as ShipStageError, not a
        silently-delivered broken file (R-4.2).
        """
        from questfoundry.export.validation import ExportValidationError

        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        def _always_fail(_path: Path) -> None:
            msg = "synthetic broken target: passage::ghost"
            raise ExportValidationError(msg)

        # Patch the dispatcher's table for the json validator
        from questfoundry.export import validation as validation_module

        monkeypatch.setitem(validation_module.VALIDATORS, "json", _always_fail)

        stage = ShipStage(project)
        with pytest.raises(ShipStageError, match="passage::ghost"):
            stage.execute(export_format="json")

    def test_valid_export_passes_phase_4(self, tmp_path: Path) -> None:
        """End-to-end: a clean run from the test fixture must satisfy
        Phase 4 validation without monkey-patching anything.
        """
        project = tmp_path / "my-story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        # No exception = Phase 4 passed
        stage.execute(export_format="twee")
        stage.execute(export_format="json")
        stage.execute(export_format="html")


class TestShipDeterminism:
    """R-2.4: SHIP must not mutate the graph; running it twice on the
    same graph must produce byte-identical output (modulo the metadata
    generation_timestamp, which the test seam pins).
    """

    _FIXED_TS = "2026-04-24T00:00:00+00:00"

    @pytest.mark.parametrize("export_format", ["twee", "json", "html"])
    def test_two_runs_produce_byte_identical_output(
        self, tmp_path: Path, export_format: str
    ) -> None:
        """Same project + same fixed timestamp → identical bytes.

        Confirms SHIP itself is deterministic (no in-memory iteration
        order leaking into output, no hidden randomness, no graph
        mutation between runs).
        """
        project = tmp_path / "story"
        _create_project_with_graph(project)

        stage = ShipStage(project)
        out1 = stage.execute(
            export_format=export_format,
            output_dir=tmp_path / f"{export_format}-1",
            timestamp=self._FIXED_TS,
        )
        out2 = stage.execute(
            export_format=export_format,
            output_dir=tmp_path / f"{export_format}-2",
            timestamp=self._FIXED_TS,
        )
        assert out1.read_bytes() == out2.read_bytes()

    def test_graph_unchanged_after_ship(self, tmp_path: Path) -> None:
        """SHIP is read-only: the graph file must be byte-identical
        before and after a run.
        """
        project = tmp_path / "story"
        _create_project_with_graph(project)
        graph_path = project / "graph.db"
        before = graph_path.read_bytes()

        ShipStage(project).execute(export_format="json", timestamp=self._FIXED_TS)

        after = graph_path.read_bytes()
        assert before == after, "SHIP mutated graph.db (R-2.4 violation)"
