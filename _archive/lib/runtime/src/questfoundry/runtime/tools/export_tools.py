from __future__ import annotations

"""Document conversion/export tools (pandoc-first with graceful fallback)."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _find_binary(name: str) -> str | None:
    return shutil.which(os.getenv(f"{name.upper()}_PATH", name))


class PandocConvert(BaseTool):
    """Convert documents between formats using pandoc if available."""

    name: str = "pandoc"
    description: str = "Convert documents using pandoc CLI"

    def _run(
        self,
        input_path: str,
        output_path: str | None = None,
        output_format: str | None = None,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        pandoc_bin = _find_binary("pandoc")
        if not pandoc_bin:
            return {
                "status": "mock",
                "message": "pandoc not available; skipping conversion",
            }

        in_path = Path(input_path)
        if not in_path.exists():
            return {"status": "error", "message": f"input not found: {input_path}"}

        fmt = output_format or in_path.suffix.lstrip(".") or "pdf"
        out_path = Path(output_path) if output_path else Path(tempfile.mkstemp(suffix=f".{fmt}")[1])
        args = [pandoc_bin, str(in_path), "-o", str(out_path)]
        if extra_args:
            args.extend(extra_args)

        try:
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - validated via monkeypatch
            logger.error("pandoc failed: %s", exc)
            return {"status": "error", "message": f"pandoc failed: {exc}"}

        return {"status": "success", "output": str(out_path)}


class PdfExport(BaseTool):
    """Export helper that prefers pandoc, with graceful mock fallback."""

    name: str = "pdf_export"
    description: str = "Export a document to PDF/EPUB using available backends"

    def __init__(self, output_format: str = "pdf", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._format = output_format

    def _run(self, input_path: str, output_path: str | None = None) -> dict[str, Any]:  # type: ignore[override]
        # First try pandoc
        pandoc_bin = _find_binary("pandoc")
        if pandoc_bin:
            converter = PandocConvert()
            return converter._run(
                input_path=input_path, output_path=output_path, output_format=self._format
            )

        # Fallback: weasyprint for HTML -> PDF/EPUB when installed
        backend = None
        try:
            import weasyprint  # type: ignore  # noqa: F401

            backend = "weasyprint"
        except Exception:
            backend = None

        if backend:
            # Keep it simple: weasyprint only for HTML to PDF
            if self._format != "pdf":
                return {"status": "mock", "message": "weasyprint only supports PDF here"}
            try:
                import weasyprint  # type: ignore

                in_path = Path(input_path)
                out_path = Path(output_path or tempfile.mkstemp(suffix=".pdf")[1])
                weasyprint.HTML(filename=str(in_path)).write_pdf(str(out_path))
                return {"status": "success", "output": str(out_path), "backend": backend}
            except Exception as exc:  # pragma: no cover
                return {"status": "error", "message": f"weasyprint failed: {exc}"}

        # Final fallback: mock
        return {
            "status": "mock",
            "message": "No PDF/EPUB backend available (pandoc/weasyprint)",
        }


def _get_exports_dir(project_id: str = "default") -> Path:
    """Get the exports directory for a project."""
    from questfoundry.runtime.core.cold_store import _default_project_root

    base = _default_project_root()
    exports_dir = base / project_id / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


class ReadExports(BaseTool):
    """
    Read exported artifacts from the export directory.

    The Book Binder and Player-Narrator use this to read previously
    exported files for review or further processing.
    """

    name: str = "read_exports"
    description: str = (
        "Read exported artifacts from the export directory. "
        "Input: export_id (optional: specific export to read), "
        "format (optional: filter by format like 'pdf', 'epub', 'html')"
    )

    def _run(
        self,
        export_id: str | None = None,
        format: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Read exports from the export directory."""
        import os

        pid = project_id or os.getenv("QF_PROJECT_ID", "default")
        exports_dir = _get_exports_dir(pid)

        if export_id:
            # Read specific export file
            matching = list(exports_dir.glob(f"{export_id}*"))
            if not matching:
                return {
                    "success": False,
                    "error": f"Export not found: {export_id}",
                    "exports_dir": str(exports_dir),
                }

            export_file = matching[0]
            try:
                # For text-based formats, read content
                if export_file.suffix in {".txt", ".md", ".html"}:
                    content = export_file.read_text(encoding="utf-8")
                    return {
                        "success": True,
                        "export_id": export_id,
                        "filename": export_file.name,
                        "format": export_file.suffix.lstrip("."),
                        "content": content,
                        "size_bytes": export_file.stat().st_size,
                    }
                else:
                    # Binary formats - return metadata only
                    return {
                        "success": True,
                        "export_id": export_id,
                        "filename": export_file.name,
                        "format": export_file.suffix.lstrip("."),
                        "path": str(export_file),
                        "size_bytes": export_file.stat().st_size,
                        "note": "Binary file - content not included",
                    }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        # List all exports (optionally filtered by format)
        pattern = f"*.{format}" if format else "*"
        exports = []
        for f in exports_dir.glob(pattern):
            if f.is_file():
                exports.append(
                    {
                        "filename": f.name,
                        "format": f.suffix.lstrip("."),
                        "size_bytes": f.stat().st_size,
                        "modified": f.stat().st_mtime,
                    }
                )

        logger.info(f"Listed {len(exports)} exports from {exports_dir}")
        return {
            "success": True,
            "project_id": pid,
            "exports_dir": str(exports_dir),
            "count": len(exports),
            "exports": exports,
        }


class WriteExports(BaseTool):
    """
    Write artifacts to the export directory.

    The Book Binder uses this to save finalized exports
    (PDF, EPUB, HTML) to the project's export location.
    """

    name: str = "write_exports"
    description: str = (
        "Write an artifact to the export directory. "
        "Input: content (the content to export), "
        "filename (target filename), "
        "format (export format: 'pdf', 'epub', 'html', 'txt')"
    )

    def _run(
        self,
        content: str,
        filename: str,
        format: str = "txt",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Write content to export directory."""
        import os

        valid_formats = ["pdf", "epub", "html", "txt", "md"]
        if format not in valid_formats:
            return {
                "success": False,
                "error": f"Invalid format: {format}. Valid formats: {valid_formats}",
            }

        pid = project_id or os.getenv("QF_PROJECT_ID", "default")
        exports_dir = _get_exports_dir(pid)

        # Ensure filename has correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        export_path = exports_dir / filename

        try:
            # For PDF/EPUB, content might be base64-encoded or we need conversion
            if format in {"pdf", "epub"}:
                # Try to decode as base64 if it looks like binary
                import base64

                try:
                    binary_content = base64.b64decode(content)
                    export_path.write_bytes(binary_content)
                except Exception:
                    # If not base64, write as text and let pandoc convert
                    temp_md = exports_dir / f"{filename}.md"
                    temp_md.write_text(content, encoding="utf-8")

                    # Try pandoc conversion
                    converter = PandocConvert()
                    result = converter._run(
                        input_path=str(temp_md),
                        output_path=str(export_path),
                        output_format=format,
                    )
                    temp_md.unlink(missing_ok=True)

                    if result.get("status") != "success":
                        # Fall back to just saving the content as-is
                        export_path.write_text(content, encoding="utf-8")
                        return {
                            "success": True,
                            "filename": filename,
                            "path": str(export_path),
                            "format": format,
                            "note": f"Saved as text (conversion unavailable: {result.get('message')})",
                        }
            else:
                # Text-based formats
                export_path.write_text(content, encoding="utf-8")

            logger.info(f"Wrote export: {export_path}")
            return {
                "success": True,
                "filename": filename,
                "path": str(export_path),
                "format": format,
                "size_bytes": export_path.stat().st_size,
                "project_id": pid,
            }

        except Exception as exc:
            logger.error(f"Failed to write export {filename}: {exc}")
            return {
                "success": False,
                "filename": filename,
                "error": str(exc),
            }
