"""Tests for JSON exporter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from questfoundry.export.base import ExportChoice, ExportContext, ExportPassage
from questfoundry.export.json_exporter import JsonExporter

if TYPE_CHECKING:
    from pathlib import Path


def _simple_context() -> ExportContext:
    """Build a minimal ExportContext for testing."""
    return ExportContext(
        title="Test Story",
        passages=[
            ExportPassage(id="p1", prose="Opening scene.", is_start=True),
            ExportPassage(id="p2", prose="The end.", is_ending=True),
        ],
        choices=[
            ExportChoice(
                from_passage="p1",
                to_passage="p2",
                label="Continue",
                requires_codewords=[],
                grants=["codeword::done"],
            ),
        ],
    )


class TestJsonExporter:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        exporter = JsonExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        assert result.exists()
        assert result.name == "story.json"

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        exporter = JsonExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        data = json.loads(result.read_text())
        assert isinstance(data, dict)

    def test_structure(self, tmp_path: Path) -> None:
        exporter = JsonExporter()
        result = exporter.export(_simple_context(), tmp_path / "out")

        data = json.loads(result.read_text())
        assert data["title"] == "Test Story"
        assert len(data["passages"]) == 2
        assert len(data["choices"]) == 1
        assert data["passages"][0]["prose"] == "Opening scene."
        assert data["passages"][0]["is_start"] is True
        assert data["choices"][0]["grants"] == ["codeword::done"]

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        exporter = JsonExporter()
        nested = tmp_path / "deep" / "nested" / "dir"
        result = exporter.export(_simple_context(), nested)

        assert result.exists()
        assert nested.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        exporter = JsonExporter()
        ctx = _simple_context()
        result = exporter.export(ctx, tmp_path / "out")

        data = json.loads(result.read_text())
        # Verify key fields survive serialization
        assert data["passages"][1]["is_ending"] is True
        assert data["art_direction"] is None
        assert data["illustrations"] == []

    def test_format_name(self) -> None:
        assert JsonExporter.format_name == "json"
