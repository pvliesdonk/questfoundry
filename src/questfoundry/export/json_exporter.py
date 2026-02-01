"""JSON export format.

Serializes the ExportContext to a structured JSON file suitable
for external tools, custom game engines, or programmatic analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import ExportContext


class JsonExporter:
    """Export story as structured JSON."""

    format_name = "json"

    def export(self, context: ExportContext, output_dir: Path) -> Path:
        """Write story data as formatted JSON.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.

        Returns:
            Path to the generated story.json file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "story.json"

        data = asdict(context)
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        return output_file
