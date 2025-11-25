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
            return converter._run(input_path=input_path, output_path=output_path, output_format=self._format)

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
