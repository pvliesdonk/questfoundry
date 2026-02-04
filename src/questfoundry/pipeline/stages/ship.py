"""SHIP stage — export story graph to playable formats.

SHIP is deterministic (no LLM). It reads the completed story graph,
validates that all passages have prose, builds an ExportContext, and
delegates to a format-specific exporter (JSON, Twee, HTML).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from questfoundry.export import build_export_context, get_exporter
from questfoundry.export.assets import bundle_assets
from questfoundry.export.assets import embed_assets as embed_assets_data_urls
from questfoundry.graph.graph import Graph
from questfoundry.observability.logging import get_logger
from questfoundry.pipeline.config import ProjectConfigError, load_project_config

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
        *,
        embed_assets: bool = False,
    ) -> Path:
        """Export the story graph to a playable format.

        Args:
            export_format: Export format name (json, twee, html).
            output_dir: Custom output directory. Defaults to
                ``{project}/exports/{format}/``.
            embed_assets: Embed illustration assets as base64 data URLs
                in HTML output. Ignored for non-HTML formats.

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

        missing_prose = [
            pid
            for pid, data in passages.items()
            if not data.get("prose") or not str(data["prose"]).strip()
        ]
        if missing_prose:
            raise ShipStageError(
                f"{len(missing_prose)} passage(s) missing prose. "
                f"Run FILL stage first. Examples: {missing_prose[:3]}"
            )

        # Get story title and language from config
        voice = graph.get_node("voice::voice")
        story_title = (voice or {}).get("story_title") or ""
        language = "en"
        config_name: str | None = None

        try:
            config = load_project_config(self._project_path)
            language = config.language
            config_name = config.name
        except (ProjectConfigError, FileNotFoundError, KeyError):
            log.warning(
                "project_config_load_failed",
                path=str(self._project_path),
                detail="Falling back to defaults for title and language.",
            )

        if not story_title:
            story_title = config_name or self._project_path.name

        # Build export context
        context = build_export_context(graph, story_title, language=language)

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

        # Optionally embed assets for HTML exports
        if embed_assets and export_format != "html":
            log.warning("ship_embed_ignored", format=export_format)

        if embed_assets and export_format == "html" and (context.illustrations or context.cover):
            embedded_illustrations, embedded_cover = embed_assets_data_urls(
                context.illustrations,
                context.cover,
                self._project_path,
            )
            context = replace(
                context,
                illustrations=embedded_illustrations,
                cover=embedded_cover,
            )

        # Export
        target_dir = output_dir or (self._project_path / "exports" / export_format)
        output_file = exporter.export(context, target_dir)

        # Bundle illustration assets (non-fatal — export is valid without them)
        if context.illustrations and not (embed_assets and export_format == "html"):
            try:
                bundle_assets(context.illustrations, self._project_path, target_dir)
            except OSError as e:
                log.warning("asset_bundling_failed", error=str(e))

        log.info(
            "ship_complete",
            format=export_format,
            output=str(output_file),
        )

        return output_file
