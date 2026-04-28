"""JSON export format.

Serializes the ExportContext to a structured JSON file suitable
for external tools, custom game engines, or programmatic analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from questfoundry.export.metadata import build_export_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import ExportContext

# Backward-compatible additive changes only (R-3.4): never rename or
# remove fields; bump only when a NEW field appears or shape changes.
JSON_FORMAT_VERSION = "1.0.0"


class JsonExporter:
    """Export story as structured JSON."""

    format_name = "json"
    format_version = JSON_FORMAT_VERSION

    def export(
        self,
        context: ExportContext,
        output_dir: Path,
        *,
        timestamp: str | None = None,
    ) -> Path:
        """Write story data as formatted JSON.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.
            timestamp: Optional override for the metadata generation
                timestamp (test seam for deterministic assertions).

        Returns:
            Path to the generated story.json file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "story.json"

        # R-3.6: emit a top-level "_metadata" key BEFORE the story payload
        # so consumers see provenance first. Underscore-prefix keeps it
        # visually separate from gameplay fields and (by convention)
        # signals "tooling, not story content."
        metadata = build_export_metadata(context, JSON_FORMAT_VERSION, timestamp=timestamp)
        payload = {"_metadata": metadata.to_dict(), **asdict(context)}
        output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        return output_file
