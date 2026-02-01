"""SHIP stage — export story graph to playable formats.

SHIP is deterministic (no LLM). It reads the completed story graph,
validates that all passages have prose, builds an ExportContext, and
delegates to a format-specific exporter (JSON, Twee, HTML).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.export import build_export_context, get_exporter
from questfoundry.graph.graph import Graph
from questfoundry.observability.logging import get_logger
from questfoundry.pipeline.config import load_project_config

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


class ShipStageError(ValueError):
    """Error during SHIP stage execution."""


class ShipStage:
    """Export story graph to a playable format.

    Unlike other stages, SHIP does not use an LLM. It reads the graph
    directly and transforms it to the requested export format.

    Args:
        project_path: Path to the project directory.
    """

    def __init__(self, project_path: Path) -> None:
        self._project_path = project_path

    def execute(
        self,
        export_format: str = "twee",
        output_dir: Path | None = None,
    ) -> Path:
        """Export the story graph to a playable format.

        Args:
            export_format: Export format name (json, twee, html).
            output_dir: Custom output directory. Defaults to
                ``{project}/exports/{format}/``.

        Returns:
            Path to the main output file.

        Raises:
            ShipStageError: If the graph is missing, has no passages,
                or passages are missing prose.
        """
        log.info(
            "ship_start",
            project=str(self._project_path),
            format=export_format,
        )

        # Load graph
        graph_file = self._project_path / "graph.json"
        if not graph_file.exists():
            raise ShipStageError(
                f"No graph.json found in {self._project_path}. "
                "Run pipeline stages (dream → fill) first."
            )
        graph = Graph.load(self._project_path)

        # Validate passages have prose
        passages = graph.get_nodes_by_type("passage")
        if not passages:
            raise ShipStageError("Graph contains no passages — nothing to export.")

        missing_prose = [pid for pid, data in passages.items() if not data.get("prose")]
        if missing_prose:
            raise ShipStageError(
                f"{len(missing_prose)} passage(s) missing prose. "
                f"Run FILL stage first. Examples: {missing_prose[:3]}"
            )

        # Get project name for title
        try:
            config = load_project_config(self._project_path)
            project_name = config.name
        except Exception:
            project_name = self._project_path.name

        # Build export context
        context = build_export_context(graph, project_name)

        log.info(
            "ship_context_built",
            passages=len(context.passages),
            choices=len(context.choices),
            entities=len(context.entities),
            codewords=len(context.codewords),
            illustrations=len(context.illustrations),
            codex_entries=len(context.codex_entries),
            has_art_direction=context.art_direction is not None,
        )

        # Get exporter
        try:
            exporter = get_exporter(export_format)
        except ValueError as e:
            raise ShipStageError(str(e)) from e

        # Export
        target_dir = output_dir or (self._project_path / "exports" / export_format)
        output_file = exporter.export(context, target_dir)

        log.info(
            "ship_complete",
            format=export_format,
            output=str(output_file),
        )

        return output_file
