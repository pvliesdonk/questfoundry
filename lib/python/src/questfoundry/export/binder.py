"""Book Binder - Render views to various export formats

Transforms view artifacts into player-facing formats (HTML, Markdown, etc.)
"""

import html
import logging
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .view import ViewArtifact

logger = logging.getLogger(__name__)


class BookBinder:
    """
    Render views to various export formats.

    Takes a view artifact and renders it into player-facing formats
    like HTML and Markdown. Handles artifact ordering, formatting,
    and presentation.

    Example:
        >>> binder = BookBinder()
        >>> html = binder.render_html(view)
        >>> binder.save_html(html, "output.html")
    """

    # Default HTML template
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Georgia, serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .artifact {{
            margin-bottom: 40px;
            padding: 20px;
            background: #f9f9f9;
            border-left: 4px solid #3498db;
        }}
        .artifact-type {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .artifact-header {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metadata {{
            font-size: 0.85em;
            color: #95a5a6;
            margin-top: 10px;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

    def __init__(
        self,
        html_template: str | None = None,
        sort_artifacts: bool = True,
    ):
        """
        Initialize book binder.

        Args:
            html_template: Custom HTML template (uses default if None)
            sort_artifacts: Whether to sort artifacts by type and ID
        """
        logger.debug("Initializing BookBinder with sort_artifacts=%s", sort_artifacts)
        self.html_template = html_template or self.HTML_TEMPLATE
        self.sort_artifacts = sort_artifacts
        logger.trace(
            "BookBinder initialized with custom template=%s", html_template is not None
        )

    def render_html(self, view: ViewArtifact, title: str | None = None) -> str:
        """
        Render view to HTML format.

        Args:
            view: ViewArtifact to render
            title: Page title (uses view_id if None)

        Returns:
            HTML string
        """
        logger.info(
            "Rendering HTML view: %s with %d artifacts",
            view.view_id,
            len(view.artifacts),
        )
        page_title = title or f"View: {view.view_id}"

        # Prepare artifacts
        artifacts = self._prepare_artifacts(view.artifacts)
        logger.trace("Prepared %d artifacts for HTML rendering", len(artifacts))

        # Render content
        content_parts = [f"<h1>{page_title}</h1>"]

        # Add view metadata
        content_parts.append('<div class="metadata">')
        content_parts.append(f"View ID: {view.view_id}<br>")
        content_parts.append(f"Snapshot: {view.snapshot_id}<br>")
        created_str = view.created.strftime("%Y-%m-%d %H:%M:%S")
        content_parts.append(f"Created: {created_str}<br>")
        content_parts.append(f"Artifacts: {len(artifacts)}")
        content_parts.append("</div>")

        # Render each artifact
        for artifact in artifacts:
            logger.trace(
                "Rendering artifact to HTML: %s (%s)",
                artifact.artifact_id,
                artifact.type,
            )
            content_parts.append(self._render_artifact_html(artifact))

        content = "\n".join(content_parts)

        # Apply template
        html_output = self.html_template.format(title=page_title, content=content)
        logger.debug("HTML rendering complete: %d bytes", len(html_output))
        return html_output

    def render_markdown(
        self,
        view: ViewArtifact,
        title: str | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Render view to Markdown format.

        Args:
            view: ViewArtifact to render
            title: Document title (uses view_id if None)
            include_metadata: Whether to include view metadata

        Returns:
            Markdown string
        """
        logger.info(
            "Rendering Markdown view: %s with %d artifacts",
            view.view_id,
            len(view.artifacts),
        )
        page_title = title or f"View: {view.view_id}"

        # Prepare artifacts
        artifacts = self._prepare_artifacts(view.artifacts)
        logger.trace(
            "Prepared %d artifacts for Markdown rendering, include_metadata=%s",
            len(artifacts),
            include_metadata,
        )

        # Render content
        content_parts = [f"# {page_title}\n"]

        # Add view metadata
        if include_metadata:
            content_parts.append("---")
            content_parts.append(f"View ID: {view.view_id}  ")
            content_parts.append(f"Snapshot: {view.snapshot_id}  ")
            content_parts.append(
                f"Created: {view.created.strftime('%Y-%m-%d %H:%M:%S')}  "
            )
            content_parts.append(f"Artifacts: {len(artifacts)}")
            content_parts.append("---\n")

        # Render each artifact
        for artifact in artifacts:
            content_parts.append(self._render_artifact_markdown(artifact))

        return "\n".join(content_parts)

    def save_html(self, html: str, output_path: str | Path) -> Path:
        """
        Save HTML to file.

        Args:
            html: HTML content
            output_path: Output file path

        Returns:
            Path to saved file
        """
        logger.info("Saving HTML to %s", output_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            f.write(html)

        logger.debug("HTML saved successfully: %d bytes to %s", len(html), output)
        return output

    def save_markdown(self, markdown: str, output_path: str | Path) -> Path:
        """
        Save Markdown to file.

        Args:
            markdown: Markdown content
            output_path: Output file path

        Returns:
            Path to saved file
        """
        logger.info("Saving Markdown to %s", output_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.debug(
            "Markdown saved successfully: %d bytes to %s", len(markdown), output
        )
        return output

    def _prepare_artifacts(self, artifacts: list[Artifact]) -> list[Artifact]:
        """
        Prepare artifacts for rendering (sort, filter, etc.).

        Args:
            artifacts: List of artifacts

        Returns:
            Prepared list of artifacts
        """
        if not self.sort_artifacts:
            return artifacts

        # Sort by type then by ID
        return sorted(
            artifacts,
            key=lambda a: (a.type, a.artifact_id or ""),
        )

    def _render_artifact_html(self, artifact: Artifact) -> str:
        """
        Render single artifact to HTML.

        Args:
            artifact: Artifact to render

        Returns:
            HTML fragment
        """
        parts = ['<div class="artifact">']

        # Type badge
        parts.append(f'<div class="artifact-type">{artifact.type}</div>')

        # Artifact header (try common name fields)
        header = self._extract_artifact_header(artifact)
        if header:
            escaped_header = self._escape_html(header)
            parts.append(f'<div class="artifact-header">{escaped_header}</div>')

        # Render data fields
        parts.append(self._render_data_html(artifact.data))

        # Artifact ID
        if artifact.artifact_id:
            parts.append(f'<div class="metadata">ID: {artifact.artifact_id}</div>')

        parts.append("</div>")
        return "\n".join(parts)

    def _render_artifact_markdown(self, artifact: Artifact) -> str:
        """
        Render single artifact to Markdown.

        Args:
            artifact: Artifact to render

        Returns:
            Markdown fragment
        """
        parts = []

        # Header
        header = self._extract_artifact_header(artifact)
        if header:
            parts.append(f"## {header}")
        else:
            parts.append(f"## {artifact.type}")

        # Type and ID
        parts.append(f"*Type: {artifact.type}*")
        if artifact.artifact_id:
            parts.append(f"*ID: {artifact.artifact_id}*\n")

        # Render data fields
        parts.append(self._render_data_markdown(artifact.data))

        parts.append("\n---\n")
        return "\n".join(parts)

    def _render_data_html(self, data: dict[str, Any], level: int = 0) -> str:
        """
        Render artifact data as HTML.

        Args:
            data: Data dictionary
            level: Nesting level

        Returns:
            HTML fragment
        """
        if not data:
            return ""

        parts = []
        indent = "  " * level

        for key, value in data.items():
            if isinstance(value, dict):
                parts.append(f"{indent}<strong>{self._escape_html(key)}:</strong>")
                parts.append(self._render_data_html(value, level + 1))
            elif isinstance(value, list):
                parts.append(f"{indent}<strong>{self._escape_html(key)}:</strong>")
                parts.append(f"{indent}<ul>")
                for item in value:
                    if isinstance(item, dict):
                        parts.append(f"{indent}  <li>")
                        parts.append(self._render_data_html(item, level + 2))
                        parts.append(f"{indent}  </li>")
                    else:
                        escaped_item = self._escape_html(str(item))
                        parts.append(f"{indent}  <li>{escaped_item}</li>")
                parts.append(f"{indent}</ul>")
            else:
                parts.append(
                    f"{indent}<strong>{self._escape_html(key)}:</strong> "
                    f"{self._escape_html(str(value))}<br>"
                )

        return "\n".join(parts)

    def _render_data_markdown(self, data: dict[str, Any], level: int = 0) -> str:
        """
        Render artifact data as Markdown.

        Args:
            data: Data dictionary
            level: Nesting level

        Returns:
            Markdown fragment
        """
        if not data:
            return ""

        parts = []
        indent = "  " * level

        for key, value in data.items():
            if isinstance(value, dict):
                parts.append(f"{indent}**{key}:**")
                parts.append(self._render_data_markdown(value, level + 1))
            elif isinstance(value, list):
                parts.append(f"{indent}**{key}:**")
                for item in value:
                    if isinstance(item, dict):
                        parts.append(self._render_data_markdown(item, level + 1))
                    else:
                        # Wrap in backticks to prevent markdown formatting
                        parts.append(f"{indent}- `{item}`")
            else:
                # Wrap in backticks to prevent markdown formatting
                parts.append(f"{indent}**{key}:** `{value}`")

        return "\n".join(parts)

    def _extract_artifact_header(self, artifact: Artifact) -> str | None:
        """
        Extract a suitable header from artifact data.

        Tries common field names like 'name', 'title', 'short_name', etc.

        Args:
            artifact: Artifact

        Returns:
            Header string or None
        """
        data = artifact.data

        # Try common header fields
        for field in ["name", "title", "short_name", "header", "label"]:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "short_name" in value:
                    short_name = value["short_name"]
                    if isinstance(short_name, str):
                        return short_name

        # For nested header structures
        if "header" in data and isinstance(data["header"], dict):
            header_dict = data["header"]
            for field in ["short_name", "title", "name"]:
                if field in header_dict:
                    value = header_dict[field]
                    if isinstance(value, str):
                        return value

        return None

    def _escape_html(self, text: str) -> str:
        """
        Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        return html.escape(text, quote=True)
